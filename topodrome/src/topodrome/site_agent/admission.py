"""Atomic physical-GPU admission and persistent lease fencing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .inventory import HostInventoryProvider, HostSnapshot, InventoryProvider

GIB = 1024**3


class AdmissionError(RuntimeError):
    """Base class for a rejected local assignment."""


class StaleFenceError(AdmissionError):
    """The scheduler offer is older than a fence already observed locally."""


class GPUUnavailableError(AdmissionError):
    """One or more exact physical GPUs cannot be acquired."""


@dataclass(frozen=True, slots=True)
class LeaseRequest:
    lease_id: str
    attempt_id: str
    gpu_uuids: tuple[str, ...]
    fence_epochs: Mapping[str, int]
    min_free_vram_bytes: int
    expires_at: float
    cpu_cores: float = 1.0
    memory_bytes: int = GIB
    reuse_actor_id: str | None = None
    model_key: str | None = None
    renewal_ttl_seconds: float = 90.0
    allow_actor_reuse: bool = True
    replace_idle_actors: bool = False
    # Site-private proof, never accepted from the scheduler/user wire payload.
    runtime_switch_actor_id: str | None = None
    runtime_switch_model_key: str | None = None
    runtime_switch_process_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.lease_id or not self.attempt_id:
            raise ValueError("lease_id and attempt_id are required")
        if not self.gpu_uuids:
            raise ValueError("at least one exact GPU UUID is required")
        if len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("GPU UUIDs must be unique")
        if set(self.fence_epochs) != set(self.gpu_uuids):
            raise ValueError("one fence epoch is required for every GPU UUID")
        if any(epoch < 1 for epoch in self.fence_epochs.values()):
            raise ValueError("fence epochs must be positive")
        if self.min_free_vram_bytes < 0:
            raise ValueError("minimum free VRAM cannot be negative")
        if self.cpu_cores <= 0 or self.memory_bytes <= 0:
            raise ValueError("CPU and memory reservations must be positive")
        if self.renewal_ttl_seconds <= 0:
            raise ValueError("lease renewal TTL must be positive")
        if self.replace_idle_actors and self.reuse_actor_id is not None:
            raise ValueError("an assignment cannot replace and reuse a resident actor")
        if self.runtime_switch_actor_id is not None and (
            not self.replace_idle_actors
            or not self.allow_actor_reuse
            or not self.runtime_switch_model_key
            or not self.runtime_switch_process_ids
            or any(pid <= 0 for pid in self.runtime_switch_process_ids)
        ):
            raise ValueError("runtime replacement requires exact site-private ownership proof")


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    accepted: bool
    lease_id: str
    gpu_uuids: tuple[str, ...]
    reason: str | None = None
    reused_actor_id: str | None = None
    cooperative_pids: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    gpu_indices: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActorAllocation:
    gpu_uuid: str
    actor_id: str
    model_key: str
    state: str
    replica_rank: int = 0


@dataclass(frozen=True, slots=True)
class ActorEvictionIntent:
    """Durable identity and state of one scheduler-directed actor disposal."""

    eviction_id: str
    actor_id: str
    model_key: str
    gpu_uuids: tuple[str, ...]
    state: Literal["stopping", "retryable", "completed"]


class LeaseFenceStore:
    """SQLite-backed local authority for fences, claims, and actor residency.

    Workload claims and warm actor allocations are intentionally different
    tables.  Releasing an attempt therefore does not pretend that a GPU holding
    a warm model is physically free.  SQLite's ``BEGIN IMMEDIATE`` makes a
    multi-GPU claim all-or-none across concurrent admission requests.
    """

    _ACTOR_STATES = frozenset({"loading", "idle", "busy", "dirty", "stopping"})

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self._create_schema()

    def _configure(self) -> None:
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")

    def _create_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS gpu_fences (
                    gpu_uuid TEXT PRIMARY KEY,
                    highest_epoch INTEGER NOT NULL,
                    last_lease_id TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS active_claims (
                    gpu_uuid TEXT PRIMARY KEY,
                    lease_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    fence_epoch INTEGER NOT NULL,
                    expires_at REAL NOT NULL,
                    accepted_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_active_claims_lease
                    ON active_claims(lease_id);

                CREATE TABLE IF NOT EXISTS active_lease_resources (
                    lease_id TEXT PRIMARY KEY,
                    attempt_id TEXT NOT NULL,
                    cpu_cores REAL NOT NULL,
                    memory_bytes INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS actor_allocations (
                    gpu_uuid TEXT PRIMARY KEY,
                    actor_id TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    state TEXT NOT NULL,
                    replica_rank INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS ix_actor_allocations_actor_gpu
                    ON actor_allocations(actor_id, gpu_uuid);
                CREATE INDEX IF NOT EXISTS ix_actor_allocations_state_updated
                    ON actor_allocations(state, updated_at);

                CREATE TABLE IF NOT EXISTS idle_actor_rebalance_cooldowns (
                    gpu_uuid TEXT PRIMARY KEY,
                    actor_id TEXT NOT NULL,
                    cooldown_until REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS actor_eviction_intents (
                    eviction_id TEXT PRIMARY KEY,
                    actor_id TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    gpu_uuids_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('stopping', 'retryable', 'completed')
                    ),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_actor_eviction_intents_actor
                    ON actor_eviction_intents(actor_id, state);
                """
            )
            columns = {
                str(row[1])
                for row in self._connection.execute(
                    "PRAGMA table_info(actor_allocations)"
                ).fetchall()
            }
            if "replica_rank" not in columns:
                self._connection.execute(
                    "ALTER TABLE actor_allocations "
                    "ADD COLUMN replica_rank INTEGER NOT NULL DEFAULT 0"
                )
                # Older agents could record one private worker-group identity
                # on several GPUs without an ordering column. Give those rows
                # deterministic ranks so opening the durable store is safe;
                # private groups are still evicted rather than reconnected.
                self._connection.execute(
                    """
                    UPDATE actor_allocations AS current
                       SET replica_rank = (
                           SELECT COUNT(*) FROM actor_allocations AS peer
                            WHERE peer.actor_id = current.actor_id
                              AND peer.gpu_uuid < current.gpu_uuid
                       )
                    """
                )
            self._connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_actor_allocations_actor_rank "
                "ON actor_allocations(actor_id, replica_rank)"
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def claim(
        self,
        request: LeaseRequest,
        *,
        fresh_eligible: frozenset[str],
        pressure_safe: frozenset[str],
        implicit_reuse_actor_id: str | None = None,
        host_snapshot: HostSnapshot | None = None,
        now: float | None = None,
    ) -> None:
        """Atomically claim every requested GPU or claim none of them."""

        accepted_at = time.time() if now is None else now
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                idempotent = True
                existing_claims = 0
                allocations: dict[str, sqlite3.Row | None] = {}
                for gpu_uuid in request.gpu_uuids:
                    claim = connection.execute(
                        "SELECT * FROM active_claims WHERE gpu_uuid = ?", (gpu_uuid,)
                    ).fetchone()
                    epoch = int(request.fence_epochs[gpu_uuid])
                    if claim is not None:
                        existing_claims += 1
                        same_claim = (
                            claim["lease_id"] == request.lease_id
                            and claim["attempt_id"] == request.attempt_id
                            and int(claim["fence_epoch"]) == epoch
                        )
                        if not same_claim:
                            raise GPUUnavailableError(
                                f"GPU {gpu_uuid} already has an active workload claim"
                            )
                    else:
                        same_claim = False
                        idempotent = False

                    fence = connection.execute(
                        "SELECT highest_epoch FROM gpu_fences WHERE gpu_uuid = ?",
                        (gpu_uuid,),
                    ).fetchone()
                    if claim is None and fence is not None and epoch <= int(fence[0]):
                        raise StaleFenceError(
                            f"fence epoch {epoch} for {gpu_uuid} is not newer than {int(fence[0])}"
                        )

                    allocation = connection.execute(
                        "SELECT * FROM actor_allocations WHERE gpu_uuid = ?",
                        (gpu_uuid,),
                    ).fetchone()
                    allocations[gpu_uuid] = allocation

                    # An exact replay of an already accepted multi-GPU lease is
                    # successful even though its actor allocation is now busy.
                    if same_claim:
                        continue

                    if gpu_uuid not in pressure_safe:
                        raise GPUUnavailableError(
                            f"GPU {gpu_uuid} has unacceptable foreign pressure"
                        )
                    if allocation is None:
                        if request.reuse_actor_id is not None:
                            raise GPUUnavailableError(
                                f"warm actor {request.reuse_actor_id} is not on GPU {gpu_uuid}"
                            )
                        if gpu_uuid not in fresh_eligible:
                            raise GPUUnavailableError(
                                f"GPU {gpu_uuid} lacks safe physical capacity"
                            )
                    else:
                        reuse_actor_id = request.reuse_actor_id or implicit_reuse_actor_id
                        switching = (
                            request.runtime_switch_actor_id == allocation["actor_id"]
                            and request.runtime_switch_model_key == allocation["model_key"]
                            and allocation["state"] == "idle"
                        )
                        if not switching and (
                            reuse_actor_id is None
                            or allocation["actor_id"] != reuse_actor_id
                            or allocation["state"] != "idle"
                            or (
                                request.model_key is not None
                                and allocation["model_key"] != request.model_key
                            )
                        ):
                            raise GPUUnavailableError(
                                f"GPU {gpu_uuid} is allocated to actor "
                                f"{allocation['actor_id']} ({allocation['state']})"
                            )

                if request.runtime_switch_actor_id is not None and not idempotent:
                    members = connection.execute(
                        "SELECT gpu_uuid, model_key, state, replica_rank FROM actor_allocations "
                        "WHERE actor_id = ? ORDER BY replica_rank",
                        (request.runtime_switch_actor_id,),
                    ).fetchall()
                    if tuple(row["gpu_uuid"] for row in members) != request.gpu_uuids or any(
                        row["state"] != "idle"
                        or row["model_key"] != request.runtime_switch_model_key
                        for row in members
                    ):
                        raise GPUUnavailableError(
                            "runtime replacement requires the complete exact idle replica"
                        )

                if 0 < existing_claims < len(request.gpu_uuids):
                    raise GPUUnavailableError("partial replay of a multi-GPU lease was rejected")

                resource_claim = connection.execute(
                    "SELECT * FROM active_lease_resources WHERE lease_id = ?",
                    (request.lease_id,),
                ).fetchone()
                if idempotent:
                    if resource_claim is None:
                        raise GPUUnavailableError(
                            "replayed lease has no durable host resource reservation"
                        )
                    if (
                        resource_claim["attempt_id"] != request.attempt_id
                        or float(resource_claim["cpu_cores"]) != request.cpu_cores
                        or int(resource_claim["memory_bytes"]) != request.memory_bytes
                    ):
                        raise GPUUnavailableError(
                            "replayed lease changed its host resource reservation"
                        )
                elif resource_claim is not None:
                    raise GPUUnavailableError("lease identity already owns host resources")

                if host_snapshot is not None and not idempotent:
                    reserved = connection.execute(
                        """
                        SELECT COALESCE(SUM(cpu_cores), 0),
                               COALESCE(SUM(memory_bytes), 0)
                          FROM active_lease_resources
                         WHERE lease_id != ?
                        """,
                        (request.lease_id,),
                    ).fetchone()
                    remaining_cpu = min(
                        host_snapshot.available_cpu,
                        max(0.0, host_snapshot.total_cpu - float(reserved[0])),
                    )
                    remaining_memory = min(
                        host_snapshot.available_memory_bytes,
                        max(0, host_snapshot.total_memory_bytes - int(reserved[1])),
                    )
                    if request.cpu_cores > remaining_cpu:
                        raise GPUUnavailableError(
                            f"host has {remaining_cpu:g} CPU cores available; "
                            f"{request.cpu_cores:g} required"
                        )
                    if request.memory_bytes > remaining_memory:
                        raise GPUUnavailableError(
                            f"host has {remaining_memory} memory bytes available; "
                            f"{request.memory_bytes} required"
                        )

                if not idempotent:
                    connection.execute(
                        """
                        INSERT INTO active_lease_resources(
                            lease_id, attempt_id, cpu_cores, memory_bytes
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            request.lease_id,
                            request.attempt_id,
                            request.cpu_cores,
                            request.memory_bytes,
                        ),
                    )
                    for gpu_uuid in request.gpu_uuids:
                        epoch = int(request.fence_epochs[gpu_uuid])
                        connection.execute(
                            """
                            INSERT INTO gpu_fences(
                                gpu_uuid, highest_epoch, last_lease_id, updated_at
                            ) VALUES (?, ?, ?, ?)
                            ON CONFLICT(gpu_uuid) DO UPDATE SET
                                highest_epoch = excluded.highest_epoch,
                                last_lease_id = excluded.last_lease_id,
                                updated_at = excluded.updated_at
                            """,
                            (gpu_uuid, epoch, request.lease_id, accepted_at),
                        )
                        connection.execute(
                            """
                            INSERT INTO active_claims(
                                gpu_uuid, lease_id, attempt_id, fence_epoch,
                                expires_at, accepted_at
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                gpu_uuid,
                                request.lease_id,
                                request.attempt_id,
                                epoch,
                                request.expires_at,
                                accepted_at,
                            ),
                        )
                        allocation = allocations[gpu_uuid]
                        if allocation is not None:
                            connection.execute(
                                """
                                UPDATE actor_allocations
                                   SET state = 'busy', updated_at = ?
                                 WHERE gpu_uuid = ?
                                """,
                                (accepted_at, gpu_uuid),
                            )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def release_lease(self, lease_id: str, *, now: float | None = None) -> int:
        released_at = time.time() if now is None else now
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    "SELECT gpu_uuid FROM active_claims WHERE lease_id = ?", (lease_id,)
                ).fetchall()
                for row in rows:
                    connection.execute(
                        """
                        UPDATE actor_allocations
                           SET state = CASE WHEN state = 'busy' THEN 'idle' ELSE state END,
                               updated_at = ?
                         WHERE gpu_uuid = ?
                        """,
                        (released_at, row["gpu_uuid"]),
                    )
                cursor = connection.execute(
                    "DELETE FROM active_claims WHERE lease_id = ?", (lease_id,)
                )
                connection.execute(
                    "DELETE FROM active_lease_resources WHERE lease_id = ?",
                    (lease_id,),
                )
                connection.execute("COMMIT")
                return int(cursor.rowcount)
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def renew_lease(
        self,
        request: LeaseRequest,
        *,
        expires_at: float,
        now: float | None = None,
    ) -> int:
        """Extend exactly the locally fenced claim acknowledged by the scheduler."""

        observed_at = time.time() if now is None else now
        if expires_at <= observed_at:
            raise ValueError("renewed lease expiry must be in the future")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    """
                    SELECT gpu_uuid, attempt_id, fence_epoch
                      FROM active_claims
                     WHERE lease_id = ?
                     ORDER BY gpu_uuid
                    """,
                    (request.lease_id,),
                ).fetchall()
                actual = {
                    str(row["gpu_uuid"]): (str(row["attempt_id"]), int(row["fence_epoch"]))
                    for row in rows
                }
                expected = {
                    gpu_uuid: (request.attempt_id, int(request.fence_epochs[gpu_uuid]))
                    for gpu_uuid in request.gpu_uuids
                }
                if actual != expected:
                    raise StaleFenceError("local lease claim no longer matches the renewal")
                cursor = connection.execute(
                    """
                    UPDATE active_claims SET expires_at = ?
                     WHERE lease_id = ?
                    """,
                    (expires_at, request.lease_id),
                )
                connection.execute("COMMIT")
                return int(cursor.rowcount)
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def lease_expiry(self, lease_id: str) -> float | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT MIN(expires_at) FROM active_claims WHERE lease_id = ?",
                (lease_id,),
            ).fetchone()
        return None if row is None or row[0] is None else float(row[0])

    def expired_lease_ids(self, *, now: float | None = None) -> tuple[str, ...]:
        observed_at = time.time() if now is None else now
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT DISTINCT lease_id FROM active_claims
                 WHERE expires_at <= ? ORDER BY lease_id
                """,
                (observed_at,),
            ).fetchall()
        return tuple(str(row["lease_id"]) for row in rows)

    def reconcile_after_runtime_stop(self) -> tuple[str, ...]:
        """Clear crash-left claims after Ray has been stopped.

        Fence counters deliberately survive. Clearing actor allocations is safe
        only because the caller first stopped the entire site-local Ray runtime.
        """

        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    "SELECT DISTINCT lease_id FROM active_claims ORDER BY lease_id"
                ).fetchall()
                connection.execute("DELETE FROM active_claims")
                connection.execute("DELETE FROM active_lease_resources")
                connection.execute("DELETE FROM actor_allocations")
                connection.execute(
                    """
                    UPDATE actor_eviction_intents
                       SET state = 'completed', updated_at = ?
                     WHERE state != 'completed'
                    """,
                    (time.time(),),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return tuple(str(row["lease_id"]) for row in rows)

    def active_claims(self) -> dict[str, str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT gpu_uuid, lease_id FROM active_claims ORDER BY gpu_uuid"
            ).fetchall()
        return {str(row["gpu_uuid"]): str(row["lease_id"]) for row in rows}

    def active_resource_totals(self) -> tuple[float, int]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COALESCE(SUM(cpu_cores), 0),
                       COALESCE(SUM(memory_bytes), 0)
                  FROM active_lease_resources
                """
            ).fetchone()
        return float(row[0]), int(row[1])

    def owns_claims(self, lease_id: str, gpu_uuids: tuple[str, ...]) -> bool:
        with self._lock:
            rows = self._connection.execute(
                "SELECT gpu_uuid FROM active_claims WHERE lease_id = ?",
                (lease_id,),
            ).fetchall()
        return {str(row["gpu_uuid"]) for row in rows} == set(gpu_uuids)

    def highest_fence(self, gpu_uuid: str) -> int | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT highest_epoch FROM gpu_fences WHERE gpu_uuid = ?", (gpu_uuid,)
            ).fetchone()
        return None if row is None else int(row[0])

    def reserve_actor(
        self,
        *,
        gpu_uuid: str,
        actor_id: str,
        model_key: str,
        state: str = "loading",
        now: float | None = None,
    ) -> None:
        self.reserve_actor_group(
            gpu_uuids=(gpu_uuid,),
            actor_id=actor_id,
            model_key=model_key,
            state=state,
            now=now,
        )

    def reserve_actor_group(
        self,
        *,
        gpu_uuids: tuple[str, ...],
        actor_id: str,
        model_key: str,
        state: str = "loading",
        now: float | None = None,
    ) -> None:
        """Publish every allocation member atomically or publish none."""

        if state not in self._ACTOR_STATES:
            raise ValueError(f"invalid actor allocation state: {state}")
        if not gpu_uuids or len(set(gpu_uuids)) != len(gpu_uuids):
            raise ValueError("actor allocation GPU UUIDs must be non-empty and unique")
        timestamp = time.time() if now is None else now
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                for gpu_uuid in gpu_uuids:
                    existing = connection.execute(
                        "SELECT actor_id FROM actor_allocations WHERE gpu_uuid = ?",
                        (gpu_uuid,),
                    ).fetchone()
                    if existing is not None and existing["actor_id"] != actor_id:
                        raise GPUUnavailableError(
                            f"GPU {gpu_uuid} is allocated to actor {existing['actor_id']}"
                        )
                existing_members = connection.execute(
                    "SELECT gpu_uuid FROM actor_allocations WHERE actor_id = ?",
                    (actor_id,),
                ).fetchall()
                unexpected = {str(row["gpu_uuid"]) for row in existing_members} - set(gpu_uuids)
                if unexpected:
                    raise GPUUnavailableError(
                        "actor identity is already bound to a different GPU group"
                    )
                for rank, gpu_uuid in enumerate(gpu_uuids):
                    connection.execute(
                        """
                        INSERT INTO actor_allocations(
                            gpu_uuid, actor_id, model_key, state, replica_rank, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(gpu_uuid) DO UPDATE SET
                            actor_id = excluded.actor_id,
                            model_key = excluded.model_key,
                            state = excluded.state,
                            replica_rank = excluded.replica_rank,
                            updated_at = excluded.updated_at
                        """,
                        (gpu_uuid, actor_id, model_key, state, rank, timestamp),
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def set_actor_state(self, actor_id: str, state: str, *, now: float | None = None) -> int:
        if state not in self._ACTOR_STATES:
            raise ValueError(f"invalid actor allocation state: {state}")
        timestamp = time.time() if now is None else now
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE actor_allocations SET state = ?, updated_at = ?
                 WHERE actor_id = ?
                """,
                (state, timestamp, actor_id),
            )
        return int(cursor.rowcount)

    def expired_idle_actor_groups(
        self,
        *,
        idle_before: float,
    ) -> tuple[tuple[ActorAllocation, ...], ...]:
        """Return complete replicas whose every member has remained idle.

        This is only a candidate query.  Callers must use
        :meth:`begin_expired_idle_actor_eviction` immediately before disposal;
        that transaction is the authority when a lease races the timeout.
        """

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT gpu_uuid, actor_id, model_key, state, replica_rank
                  FROM actor_allocations
                 WHERE actor_id IN (
                    SELECT actor_id
                      FROM actor_allocations
                     GROUP BY actor_id
                    HAVING SUM(CASE WHEN state = 'idle' THEN 0 ELSE 1 END) = 0
                       AND MAX(updated_at) <= ?
                 )
                 ORDER BY actor_id, replica_rank, gpu_uuid
                """,
                (idle_before,),
            ).fetchall()

        grouped: dict[str, list[ActorAllocation]] = {}
        for row in rows:
            allocation = ActorAllocation(
                gpu_uuid=str(row["gpu_uuid"]),
                actor_id=str(row["actor_id"]),
                model_key=str(row["model_key"]),
                state=str(row["state"]),
                replica_rank=int(row["replica_rank"]),
            )
            grouped.setdefault(allocation.actor_id, []).append(allocation)

        complete: list[tuple[ActorAllocation, ...]] = []
        for members in grouped.values():
            ordered = tuple(members)
            if (
                ordered
                and tuple(member.replica_rank for member in ordered) == tuple(range(len(ordered)))
                and len({member.model_key for member in ordered}) == 1
            ):
                complete.append(ordered)
        return tuple(complete)

    def begin_idle_actor_eviction_batch(
        self,
        actor_groups: tuple[tuple[ActorAllocation, ...], ...],
        *,
        now: float | None = None,
    ) -> bool:
        """Atomically fence every member of several complete idle replicas.

        This is the local half of a queued assignment replacing one or more
        resident models. No runtime may be disposed until this all-or-none
        transaction succeeds.
        """

        if not actor_groups:
            return True
        actor_ids = tuple(group[0].actor_id if group else "" for group in actor_groups)
        all_gpu_uuids = tuple(allocation.gpu_uuid for group in actor_groups for allocation in group)
        valid_shape = bool(
            all(actor_ids)
            and len(set(actor_ids)) == len(actor_ids)
            and len(set(all_gpu_uuids)) == len(all_gpu_uuids)
            and all(
                group
                and len({item.actor_id for item in group}) == 1
                and len({item.model_key for item in group}) == 1
                and all(item.state == "idle" for item in group)
                and tuple(item.replica_rank for item in group) == tuple(range(len(group)))
                and len({item.gpu_uuid for item in group}) == len(group)
                for group in actor_groups
            )
        )
        if not valid_shape:
            raise ValueError("actor eviction batch must contain complete disjoint idle replicas")
        observed_at = time.time() if now is None else now

        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                for expected in actor_groups:
                    actor_id = expected[0].actor_id
                    rows = connection.execute(
                        """
                        SELECT gpu_uuid, actor_id, model_key, state, replica_rank
                          FROM actor_allocations
                         WHERE actor_id = ? ORDER BY replica_rank, gpu_uuid
                        """,
                        (actor_id,),
                    ).fetchall()
                    actual = tuple(
                        ActorAllocation(
                            gpu_uuid=str(row["gpu_uuid"]),
                            actor_id=str(row["actor_id"]),
                            model_key=str(row["model_key"]),
                            state=str(row["state"]),
                            replica_rank=int(row["replica_rank"]),
                        )
                        for row in rows
                    )
                    if actual != expected:
                        connection.execute("ROLLBACK")
                        return False
                    active = connection.execute(
                        """
                        SELECT 1 FROM actor_allocations a
                        JOIN active_claims c ON c.gpu_uuid = a.gpu_uuid
                        WHERE a.actor_id = ? LIMIT 1
                        """,
                        (actor_id,),
                    ).fetchone()
                    if active is not None:
                        connection.execute("ROLLBACK")
                        return False

                changed = 0
                for actor_id in actor_ids:
                    cursor = connection.execute(
                        """
                        UPDATE actor_allocations SET state = 'stopping', updated_at = ?
                         WHERE actor_id = ? AND state = 'idle'
                        """,
                        (observed_at, actor_id),
                    )
                    changed += int(cursor.rowcount)
                if changed != len(all_gpu_uuids):
                    connection.execute("ROLLBACK")
                    return False
                connection.execute("COMMIT")
                return True
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def begin_expired_idle_actor_eviction(
        self,
        actor_id: str,
        *,
        expected_gpu_uuids: tuple[str, ...],
        idle_before: float,
        now: float | None = None,
    ) -> bool:
        """Atomically fence one complete, expired idle replica for disposal."""

        if not actor_id:
            raise ValueError("actor identity is required")
        if not expected_gpu_uuids or len(set(expected_gpu_uuids)) != len(expected_gpu_uuids):
            raise ValueError("expected GPU gang must be non-empty and unique")
        observed_at = time.time() if now is None else now

        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    """
                    SELECT gpu_uuid, model_key, state, replica_rank, updated_at
                      FROM actor_allocations
                     WHERE actor_id = ?
                     ORDER BY replica_rank, gpu_uuid
                    """,
                    (actor_id,),
                ).fetchall()
                exact_expired_replica = bool(
                    tuple(str(row["gpu_uuid"]) for row in rows) == expected_gpu_uuids
                    and all(str(row["state"]) == "idle" for row in rows)
                    and tuple(int(row["replica_rank"]) for row in rows) == tuple(range(len(rows)))
                    and len({str(row["model_key"]) for row in rows}) == 1
                    and all(float(row["updated_at"]) <= idle_before for row in rows)
                )
                if not exact_expired_replica:
                    connection.execute("ROLLBACK")
                    return False

                active = connection.execute(
                    """
                    SELECT 1 FROM actor_allocations a
                    JOIN active_claims c ON c.gpu_uuid = a.gpu_uuid
                    WHERE a.actor_id = ? LIMIT 1
                    """,
                    (actor_id,),
                ).fetchone()
                if active is not None:
                    connection.execute("ROLLBACK")
                    return False

                cursor = connection.execute(
                    """
                    UPDATE actor_allocations SET state = 'stopping', updated_at = ?
                     WHERE actor_id = ? AND state = 'idle' AND updated_at <= ?
                    """,
                    (observed_at, actor_id, idle_before),
                )
                if int(cursor.rowcount) != len(expected_gpu_uuids):
                    connection.execute("ROLLBACK")
                    return False
                connection.execute("COMMIT")
                return True
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def begin_scheduler_actor_eviction(
        self,
        eviction_id: str,
        *,
        actor_id: str,
        expected_replica_key_hash: str,
        expected_gpu_uuids: tuple[str, ...],
        minimum_idle_seconds: float | None = None,
        now: float | None = None,
    ) -> ActorEvictionIntent:
        """Atomically persist and fence one exact idle replica for disposal.

        ``eviction_id`` is the scheduler's stable idempotency key. Replaying it
        with different coordinates is rejected, while replaying a completed
        intent is a no-op success. The transaction is also the final local
        authority against a workload lease racing the control command.
        """

        if not eviction_id or not actor_id:
            raise ValueError("eviction and actor identities are required")
        if not _valid_sha256(expected_replica_key_hash):
            raise ValueError("replica key hash must be a lowercase SHA-256 digest")
        if not expected_gpu_uuids or len(set(expected_gpu_uuids)) != len(expected_gpu_uuids):
            raise ValueError("expected GPU gang must be non-empty and unique")
        if minimum_idle_seconds is not None and minimum_idle_seconds < 0:
            raise ValueError("minimum idle interval cannot be negative")
        observed_at = time.time() if now is None else now
        encoded_gpu_uuids = _encode_gpu_uuids(expected_gpu_uuids)

        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM actor_eviction_intents WHERE eviction_id = ?",
                    (eviction_id,),
                ).fetchone()
                if existing is not None:
                    intent = _actor_eviction_intent(existing)
                    if (
                        intent.actor_id != actor_id
                        or intent.gpu_uuids != expected_gpu_uuids
                        or _sha256_text(intent.model_key) != expected_replica_key_hash
                    ):
                        raise GPUUnavailableError(
                            "eviction identity was already used for different actor coordinates"
                        )
                    if intent.state == "completed":
                        connection.execute("COMMIT")
                        return intent
                    required_state = "stopping" if intent.state == "stopping" else "idle"
                else:
                    required_state = "idle"

                rows = connection.execute(
                    """
                    SELECT gpu_uuid, model_key, state, replica_rank, updated_at
                      FROM actor_allocations
                     WHERE actor_id = ?
                     ORDER BY replica_rank, gpu_uuid
                    """,
                    (actor_id,),
                ).fetchall()
                model_keys = {str(row["model_key"]) for row in rows}
                if existing is None:
                    if len(model_keys) != 1:
                        raise GPUUnavailableError("actor is not the expected complete idle replica")
                    model_key = next(iter(model_keys))
                    if _sha256_text(model_key) != expected_replica_key_hash:
                        raise GPUUnavailableError("actor replica key hash does not match")
                else:
                    model_key = intent.model_key
                exact_replica = bool(
                    tuple(str(row["gpu_uuid"]) for row in rows) == expected_gpu_uuids
                    and tuple(int(row["replica_rank"]) for row in rows) == tuple(range(len(rows)))
                    and all(str(row["model_key"]) == model_key for row in rows)
                    and all(str(row["state"]) == required_state for row in rows)
                )
                if not exact_replica:
                    raise GPUUnavailableError("actor is not the expected complete idle replica")
                if (
                    required_state == "idle"
                    and minimum_idle_seconds is not None
                    and any(
                        float(row["updated_at"]) > observed_at - minimum_idle_seconds
                        for row in rows
                    )
                ):
                    raise GPUUnavailableError(
                        "actor has not remained idle for the required interval"
                    )

                active = connection.execute(
                    """
                    SELECT 1 FROM actor_allocations a
                    JOIN active_claims c ON c.gpu_uuid = a.gpu_uuid
                    WHERE a.actor_id = ? LIMIT 1
                    """,
                    (actor_id,),
                ).fetchone()
                if active is not None:
                    raise GPUUnavailableError(
                        "actor cannot be evicted while its workload lease is active"
                    )

                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO actor_eviction_intents(
                            eviction_id, actor_id, model_key, gpu_uuids_json,
                            state, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 'stopping', ?, ?)
                        """,
                        (
                            eviction_id,
                            actor_id,
                            model_key,
                            encoded_gpu_uuids,
                            observed_at,
                            observed_at,
                        ),
                    )
                elif required_state == "idle":
                    connection.execute(
                        """
                        UPDATE actor_eviction_intents
                           SET state = 'stopping', updated_at = ?
                         WHERE eviction_id = ? AND state = 'retryable'
                        """,
                        (observed_at, eviction_id),
                    )

                if required_state == "idle":
                    cursor = connection.execute(
                        """
                        UPDATE actor_allocations SET state = 'stopping', updated_at = ?
                         WHERE actor_id = ? AND state = 'idle'
                        """,
                        (observed_at, actor_id),
                    )
                    if int(cursor.rowcount) != len(expected_gpu_uuids):
                        raise GPUUnavailableError(
                            "actor allocation changed while eviction was being fenced"
                        )
                connection.execute("COMMIT")
                return ActorEvictionIntent(
                    eviction_id=eviction_id,
                    actor_id=actor_id,
                    model_key=model_key,
                    gpu_uuids=expected_gpu_uuids,
                    state="stopping",
                )
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def complete_scheduler_actor_eviction(self, eviction_id: str) -> int:
        """Commit removal of an exact, confirmed-stopped scheduler intent."""

        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM actor_eviction_intents WHERE eviction_id = ?",
                    (eviction_id,),
                ).fetchone()
                if row is None:
                    raise GPUUnavailableError("actor eviction intent is unknown")
                intent = _actor_eviction_intent(row)
                if intent.state == "completed":
                    connection.execute("COMMIT")
                    return 0
                if intent.state != "stopping":
                    raise GPUUnavailableError("actor eviction is not fenced for disposal")

                rows = connection.execute(
                    """
                    SELECT gpu_uuid, model_key, state, replica_rank
                      FROM actor_allocations
                     WHERE actor_id = ? ORDER BY replica_rank, gpu_uuid
                    """,
                    (intent.actor_id,),
                ).fetchall()
                exact = bool(
                    tuple(str(item["gpu_uuid"]) for item in rows) == intent.gpu_uuids
                    and tuple(int(item["replica_rank"]) for item in rows) == tuple(range(len(rows)))
                    and all(str(item["model_key"]) == intent.model_key for item in rows)
                    and all(str(item["state"]) == "stopping" for item in rows)
                )
                if not exact:
                    raise GPUUnavailableError("actor allocation changed before eviction completion")
                active = connection.execute(
                    """
                    SELECT 1 FROM actor_allocations a
                    JOIN active_claims c ON c.gpu_uuid = a.gpu_uuid
                    WHERE a.actor_id = ? LIMIT 1
                    """,
                    (intent.actor_id,),
                ).fetchone()
                if active is not None:
                    raise GPUUnavailableError(
                        "actor cannot complete eviction while a workload lease is active"
                    )
                cursor = connection.execute(
                    "DELETE FROM actor_allocations WHERE actor_id = ? AND state = 'stopping'",
                    (intent.actor_id,),
                )
                connection.execute(
                    """
                    UPDATE actor_eviction_intents
                       SET state = 'completed', updated_at = ?
                     WHERE eviction_id = ? AND state = 'stopping'
                    """,
                    (time.time(), eviction_id),
                )
                connection.execute("COMMIT")
                return int(cursor.rowcount)
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def restore_scheduler_actor_eviction(
        self,
        eviction_id: str,
        *,
        now: float | None = None,
    ) -> bool:
        """Make a definitely-live failed teardown retryable without losing identity."""

        observed_at = time.time() if now is None else now
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM actor_eviction_intents WHERE eviction_id = ?",
                    (eviction_id,),
                ).fetchone()
                if row is None:
                    connection.execute("ROLLBACK")
                    return False
                intent = _actor_eviction_intent(row)
                if intent.state != "stopping":
                    connection.execute("ROLLBACK")
                    return intent.state == "retryable"
                rows = connection.execute(
                    """
                    SELECT gpu_uuid, model_key, state, replica_rank
                      FROM actor_allocations
                     WHERE actor_id = ? ORDER BY replica_rank, gpu_uuid
                    """,
                    (intent.actor_id,),
                ).fetchall()
                exact = bool(
                    tuple(str(item["gpu_uuid"]) for item in rows) == intent.gpu_uuids
                    and tuple(int(item["replica_rank"]) for item in rows) == tuple(range(len(rows)))
                    and all(str(item["model_key"]) == intent.model_key for item in rows)
                    and all(str(item["state"]) == "stopping" for item in rows)
                )
                if not exact:
                    connection.execute("ROLLBACK")
                    return False
                active = connection.execute(
                    """
                    SELECT 1 FROM actor_allocations a
                    JOIN active_claims c ON c.gpu_uuid = a.gpu_uuid
                    WHERE a.actor_id = ? LIMIT 1
                    """,
                    (intent.actor_id,),
                ).fetchone()
                if active is not None:
                    connection.execute("ROLLBACK")
                    return False
                changed = connection.execute(
                    """
                    UPDATE actor_allocations SET state = 'idle', updated_at = ?
                     WHERE actor_id = ? AND state = 'stopping'
                    """,
                    (observed_at, intent.actor_id),
                )
                connection.execute(
                    """
                    UPDATE actor_eviction_intents
                       SET state = 'retryable', updated_at = ?
                     WHERE eviction_id = ? AND state = 'stopping'
                    """,
                    (observed_at, eviction_id),
                )
                connection.execute("COMMIT")
                return int(changed.rowcount) == len(intent.gpu_uuids)
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def actor_eviction_intent(self, eviction_id: str) -> ActorEvictionIntent | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM actor_eviction_intents WHERE eviction_id = ?",
                (eviction_id,),
            ).fetchone()
        return None if row is None else _actor_eviction_intent(row)

    def begin_idle_actor_eviction(
        self,
        actor_id: str,
        *,
        expected_gpu_uuids: tuple[str, ...],
        alternative_gpu_uuids: tuple[str, ...],
        cooldown_seconds: float,
        now: float | None = None,
    ) -> bool:
        """Atomically fence one complete idle replica before exact disposal.

        Alternative GPUs are not reserved: they are only the material reason to
        shed a noisy warm copy.  They are checked in the same local transaction
        so a concurrent Topodrome admission cannot make the decision stale before
        the source actor changes from ``idle`` to ``stopping``.
        """

        if not actor_id:
            raise ValueError("actor identity is required")
        if (
            not expected_gpu_uuids
            or len(set(expected_gpu_uuids)) != len(expected_gpu_uuids)
            or len(alternative_gpu_uuids) != len(expected_gpu_uuids)
            or len(set(alternative_gpu_uuids)) != len(alternative_gpu_uuids)
            or set(expected_gpu_uuids) & set(alternative_gpu_uuids)
        ):
            raise ValueError("source and alternative GPU gangs must be disjoint and equal-sized")
        if cooldown_seconds < 0:
            raise ValueError("idle actor rebalance cooldown cannot be negative")
        observed_at = time.time() if now is None else now
        cooldown_until = observed_at + cooldown_seconds
        involved_gpu_uuids = (*expected_gpu_uuids, *alternative_gpu_uuids)

        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    """
                    SELECT gpu_uuid, actor_id, model_key, state, replica_rank
                      FROM actor_allocations
                     WHERE actor_id = ?
                     ORDER BY replica_rank, gpu_uuid
                    """,
                    (actor_id,),
                ).fetchall()
                actual_gpu_uuids = tuple(str(row["gpu_uuid"]) for row in rows)
                complete_idle_replica = bool(
                    actual_gpu_uuids == expected_gpu_uuids
                    and all(str(row["state"]) == "idle" for row in rows)
                    and tuple(int(row["replica_rank"]) for row in rows) == tuple(range(len(rows)))
                    and len({str(row["model_key"]) for row in rows}) == 1
                )
                if not complete_idle_replica:
                    connection.execute("ROLLBACK")
                    return False

                claim_exists = any(
                    connection.execute(
                        "SELECT 1 FROM active_claims WHERE gpu_uuid = ?",
                        (gpu_uuid,),
                    ).fetchone()
                    is not None
                    for gpu_uuid in involved_gpu_uuids
                )
                alternative_is_allocated = any(
                    connection.execute(
                        "SELECT 1 FROM actor_allocations WHERE gpu_uuid = ?",
                        (gpu_uuid,),
                    ).fetchone()
                    is not None
                    for gpu_uuid in alternative_gpu_uuids
                )
                cooling = any(
                    connection.execute(
                        """
                        SELECT 1 FROM idle_actor_rebalance_cooldowns
                         WHERE gpu_uuid = ? AND cooldown_until > ?
                        """,
                        (gpu_uuid, observed_at),
                    ).fetchone()
                    is not None
                    for gpu_uuid in involved_gpu_uuids
                )
                if claim_exists or alternative_is_allocated or cooling:
                    connection.execute("ROLLBACK")
                    return False

                cursor = connection.execute(
                    """
                    UPDATE actor_allocations SET state = 'stopping', updated_at = ?
                     WHERE actor_id = ? AND state = 'idle'
                    """,
                    (observed_at, actor_id),
                )
                if int(cursor.rowcount) != len(expected_gpu_uuids):
                    connection.execute("ROLLBACK")
                    return False
                for gpu_uuid in involved_gpu_uuids:
                    connection.execute(
                        """
                        INSERT INTO idle_actor_rebalance_cooldowns(
                            gpu_uuid, actor_id, cooldown_until, updated_at
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(gpu_uuid) DO UPDATE SET
                            actor_id = excluded.actor_id,
                            cooldown_until = excluded.cooldown_until,
                            updated_at = excluded.updated_at
                        """,
                        (gpu_uuid, actor_id, cooldown_until, observed_at),
                    )
                connection.execute("COMMIT")
                return True
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def complete_idle_actor_eviction(
        self,
        actor_id: str,
        *,
        expected_gpu_uuids: tuple[str, ...],
    ) -> int:
        """Remove exactly one confirmed-stopped replica, or remove nothing."""

        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    """
                    SELECT gpu_uuid, state, replica_rank FROM actor_allocations
                     WHERE actor_id = ? ORDER BY replica_rank, gpu_uuid
                    """,
                    (actor_id,),
                ).fetchall()
                exact = bool(
                    tuple(str(row["gpu_uuid"]) for row in rows) == expected_gpu_uuids
                    and all(str(row["state"]) == "stopping" for row in rows)
                    and tuple(int(row["replica_rank"]) for row in rows) == tuple(range(len(rows)))
                )
                if not exact:
                    connection.execute("ROLLBACK")
                    return 0
                busy = connection.execute(
                    """
                    SELECT 1 FROM actor_allocations a
                    JOIN active_claims c ON c.gpu_uuid = a.gpu_uuid
                    WHERE a.actor_id = ? LIMIT 1
                    """,
                    (actor_id,),
                ).fetchone()
                if busy is not None:
                    connection.execute("ROLLBACK")
                    return 0
                cursor = connection.execute(
                    "DELETE FROM actor_allocations WHERE actor_id = ? AND state = 'stopping'",
                    (actor_id,),
                )
                _mark_actor_eviction_intents_completed(
                    connection,
                    actor_id=actor_id,
                    gpu_uuids=expected_gpu_uuids,
                )
                connection.execute("COMMIT")
                return int(cursor.rowcount)
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def restore_idle_actor_eviction(
        self,
        actor_id: str,
        *,
        expected_gpu_uuids: tuple[str, ...],
        now: float | None = None,
    ) -> bool:
        """Roll back a failed stop only after the caller reproves actor liveness."""

        observed_at = time.time() if now is None else now
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    """
                    SELECT gpu_uuid, state, replica_rank FROM actor_allocations
                     WHERE actor_id = ? ORDER BY replica_rank, gpu_uuid
                    """,
                    (actor_id,),
                ).fetchall()
                exact = bool(
                    tuple(str(row["gpu_uuid"]) for row in rows) == expected_gpu_uuids
                    and all(str(row["state"]) == "stopping" for row in rows)
                    and tuple(int(row["replica_rank"]) for row in rows) == tuple(range(len(rows)))
                )
                if not exact:
                    connection.execute("ROLLBACK")
                    return False
                busy = connection.execute(
                    """
                    SELECT 1 FROM actor_allocations a
                    JOIN active_claims c ON c.gpu_uuid = a.gpu_uuid
                    WHERE a.actor_id = ? LIMIT 1
                    """,
                    (actor_id,),
                ).fetchone()
                if busy is not None:
                    connection.execute("ROLLBACK")
                    return False
                cursor = connection.execute(
                    """
                    UPDATE actor_allocations SET state = 'idle', updated_at = ?
                     WHERE actor_id = ? AND state = 'stopping'
                    """,
                    (observed_at, actor_id),
                )
                connection.execute(
                    """
                    UPDATE actor_eviction_intents
                       SET state = 'retryable', updated_at = ?
                     WHERE actor_id = ?
                       AND gpu_uuids_json = ?
                       AND state = 'stopping'
                    """,
                    (
                        observed_at,
                        actor_id,
                        _encode_gpu_uuids(expected_gpu_uuids),
                    ),
                )
                connection.execute("COMMIT")
                return int(cursor.rowcount) == len(expected_gpu_uuids)
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def idle_actor_rebalance_cooldown_until(self, gpu_uuid: str) -> float | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT cooldown_until FROM idle_actor_rebalance_cooldowns
                 WHERE gpu_uuid = ?
                """,
                (gpu_uuid,),
            ).fetchone()
        return None if row is None else float(row[0])

    def evict_actor(self, actor_id: str) -> int:
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                allocation_rows = connection.execute(
                    """
                    SELECT gpu_uuid FROM actor_allocations
                     WHERE actor_id = ? ORDER BY replica_rank, gpu_uuid
                    """,
                    (actor_id,),
                ).fetchall()
                busy = connection.execute(
                    """
                    SELECT 1 FROM actor_allocations a
                    JOIN active_claims c ON c.gpu_uuid = a.gpu_uuid
                    WHERE a.actor_id = ? LIMIT 1
                    """,
                    (actor_id,),
                ).fetchone()
                if busy is not None:
                    raise GPUUnavailableError(
                        f"cannot evict actor {actor_id} while its workload lease is active"
                    )
                cursor = connection.execute(
                    "DELETE FROM actor_allocations WHERE actor_id = ?", (actor_id,)
                )
                if allocation_rows:
                    _mark_actor_eviction_intents_completed(
                        connection,
                        actor_id=actor_id,
                        gpu_uuids=tuple(str(row["gpu_uuid"]) for row in allocation_rows),
                    )
                connection.execute("COMMIT")
                return int(cursor.rowcount)
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def actor_allocation(self, gpu_uuid: str) -> ActorAllocation | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM actor_allocations WHERE gpu_uuid = ?", (gpu_uuid,)
            ).fetchone()
        if row is None:
            return None
        return ActorAllocation(
            gpu_uuid=str(row["gpu_uuid"]),
            actor_id=str(row["actor_id"]),
            model_key=str(row["model_key"]),
            state=str(row["state"]),
            replica_rank=int(row["replica_rank"]),
        )

    def actor_allocations(self) -> tuple[ActorAllocation, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM actor_allocations ORDER BY actor_id, replica_rank, gpu_uuid"
            ).fetchall()
        return tuple(
            ActorAllocation(
                gpu_uuid=str(row["gpu_uuid"]),
                actor_id=str(row["actor_id"]),
                model_key=str(row["model_key"]),
                state=str(row["state"]),
                replica_rank=int(row["replica_rank"]),
            )
            for row in rows
        )

    def actor_idle_since(
        self,
        actor_id: str,
        *,
        expected_gpu_uuids: tuple[str, ...],
    ) -> float | None:
        """Return the durable start of one exact replica's idle interval.

        The timestamp is returned only when every ordered allocation member is
        still part of the same complete idle replica. Claim/release updates the
        same rows transactionally, so successful reuse always starts a fresh
        interval while an exact Ray reconnect preserves the old one.
        """

        if not actor_id or not expected_gpu_uuids:
            return None
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT gpu_uuid, model_key, state, replica_rank, updated_at
                  FROM actor_allocations
                 WHERE actor_id = ?
                 ORDER BY replica_rank, gpu_uuid
                """,
                (actor_id,),
            ).fetchall()
        exact_idle_replica = bool(
            tuple(str(row["gpu_uuid"]) for row in rows) == expected_gpu_uuids
            and tuple(int(row["replica_rank"]) for row in rows) == tuple(range(len(rows)))
            and len({str(row["model_key"]) for row in rows}) == 1
            and all(str(row["state"]) == "idle" for row in rows)
        )
        if not exact_idle_replica:
            return None
        # Group transitions use one timestamp. max() stays conservative if a
        # legacy/local database ever contains slightly different member times.
        return max(float(row["updated_at"]) for row in rows)


def _encode_gpu_uuids(gpu_uuids: tuple[str, ...]) -> str:
    return json.dumps(gpu_uuids, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _actor_eviction_intent(row: sqlite3.Row) -> ActorEvictionIntent:
    raw_gpu_uuids = json.loads(str(row["gpu_uuids_json"]))
    if not isinstance(raw_gpu_uuids, list) or not all(
        isinstance(value, str) and value for value in raw_gpu_uuids
    ):
        raise RuntimeError("durable actor eviction GPU coordinates are invalid")
    state = str(row["state"])
    if state not in {"stopping", "retryable", "completed"}:
        raise RuntimeError("durable actor eviction state is invalid")
    return ActorEvictionIntent(
        eviction_id=str(row["eviction_id"]),
        actor_id=str(row["actor_id"]),
        model_key=str(row["model_key"]),
        gpu_uuids=tuple(raw_gpu_uuids),
        state=state,  # type: ignore[arg-type]
    )


def _mark_actor_eviction_intents_completed(
    connection: sqlite3.Connection,
    *,
    actor_id: str,
    gpu_uuids: tuple[str, ...],
) -> None:
    """Close only durable commands matching the exact disposed replica."""

    connection.execute(
        """
        UPDATE actor_eviction_intents
           SET state = 'completed', updated_at = ?
         WHERE actor_id = ?
           AND gpu_uuids_json = ?
           AND state IN ('stopping', 'retryable')
        """,
        (time.time(), actor_id, _encode_gpu_uuids(gpu_uuids)),
    )


class AdmissionController:
    """Recheck physical state immediately before accepting a scheduler offer."""

    def __init__(
        self,
        inventory: InventoryProvider,
        fences: LeaseFenceStore,
        *,
        safety_margin_bytes: int = GIB,
        max_foreign_memory_bytes: int = 0,
        max_unattributed_utilization_percent: int = 10,
        allow_foreign_processes_when_capacity_sufficient: bool = False,
        warm_actor_validator: Callable[[ActorAllocation], bool] | None = None,
        host_inventory: HostInventoryProvider | None = None,
    ) -> None:
        if safety_margin_bytes < 0 or max_foreign_memory_bytes < 0:
            raise ValueError("admission thresholds cannot be negative")
        self._inventory = inventory
        self._fences = fences
        self.safety_margin_bytes = safety_margin_bytes
        self.max_foreign_memory_bytes = max_foreign_memory_bytes
        self.max_unattributed_utilization_percent = max_unattributed_utilization_percent
        self.allow_foreign_processes_when_capacity_sufficient = (
            allow_foreign_processes_when_capacity_sufficient
        )
        self.warm_actor_validator = warm_actor_validator
        self.host_inventory = host_inventory

    def admit(self, request: LeaseRequest, *, now: float | None = None) -> AdmissionDecision:
        observed_at = time.time() if now is None else now
        if request.expires_at <= observed_at:
            return AdmissionDecision(
                False,
                request.lease_id,
                request.gpu_uuids,
                reason="assignment lease offer has expired",
            )
        snapshots = {snapshot.uuid: snapshot for snapshot in self._inventory.sample()}
        missing = [uuid for uuid in request.gpu_uuids if uuid not in snapshots]
        if missing:
            return AdmissionDecision(
                False,
                request.lease_id,
                request.gpu_uuids,
                reason=f"requested physical GPU UUIDs are absent: {', '.join(missing)}",
            )

        required = request.min_free_vram_bytes + self.safety_margin_bytes
        pressure_safe: set[str] = set()
        fresh_eligible: set[str] = set()
        cooperative_pids: dict[str, tuple[int, ...]] = {}
        for gpu_uuid in request.gpu_uuids:
            snapshot = snapshots[gpu_uuid]
            if snapshot.externally_reserved:
                # A validated cooperative lock represents an explicit foreign
                # allocation even before its worker opens a CUDA context.
                continue
            if snapshot.cooperative_processes:
                cooperative_pids[gpu_uuid] = tuple(
                    process.pid for process in snapshot.cooperative_processes
                )
            foreign_unknown = any(
                process.used_memory_bytes is None for process in snapshot.foreign_processes
            )
            safe = self.allow_foreign_processes_when_capacity_sufficient or (
                not foreign_unknown
                and snapshot.foreign_memory_bytes <= self.max_foreign_memory_bytes
            )
            if safe:
                pressure_safe.add(gpu_uuid)
            if request.runtime_switch_actor_id is not None:
                # Reclaim only the exact old worker's VRAM, never arbitrary
                # managed/foreign processes or another member of a larger gang.
                processes = snapshot.topodrome_processes
                valid = (
                    bool(processes)
                    and all(
                        process.pid in request.runtime_switch_process_ids
                        and process.used_memory_bytes is not None
                        for process in processes
                    )
                    and snapshot.free_memory_bytes
                    + sum(process.used_memory_bytes or 0 for process in processes)
                    >= required
                )
                if not valid:
                    pressure_safe.discard(gpu_uuid)
            no_managed_process = not snapshot.topodrome_processes
            cooperative_only = bool(snapshot.cooperative_processes) and safe
            utilization_safe = (
                snapshot.utilization_percent <= self.max_unattributed_utilization_percent
                or bool(snapshot.cooperative_processes)
                or self.allow_foreign_processes_when_capacity_sufficient
            )
            if (
                safe
                and no_managed_process
                and utilization_safe
                and (cooperative_only or snapshot.free_memory_bytes >= required)
            ):
                fresh_eligible.add(gpu_uuid)

        implicit_reuse_actor_id: str | None = None
        if request.runtime_switch_actor_id is not None and not self._fences.owns_claims(
            request.lease_id, request.gpu_uuids
        ):
            allocations = tuple(self._fences.actor_allocation(uuid) for uuid in request.gpu_uuids)
            if self.warm_actor_validator is None or any(
                item is None
                or item.actor_id != request.runtime_switch_actor_id
                or not self.warm_actor_validator(item)
                for item in allocations
            ):
                return AdmissionDecision(
                    False,
                    request.lease_id,
                    request.gpu_uuids,
                    reason="runtime replacement ownership changed",
                )
        if request.allow_actor_reuse and request.reuse_actor_id is None:
            allocations = tuple(
                self._fences.actor_allocation(gpu_uuid) for gpu_uuid in request.gpu_uuids
            )
            if all(allocation is not None for allocation in allocations):
                exact = tuple(allocation for allocation in allocations if allocation is not None)
                actor_ids = {allocation.actor_id for allocation in exact}
                if (
                    len(actor_ids) == 1
                    and tuple(allocation.replica_rank for allocation in exact)
                    == tuple(range(len(exact)))
                    and all(allocation.state == "idle" for allocation in exact)
                    and request.model_key is not None
                    and all(allocation.model_key == request.model_key for allocation in exact)
                    and self.warm_actor_validator is not None
                    and all(self.warm_actor_validator(allocation) for allocation in exact)
                ):
                    implicit_reuse_actor_id = exact[0].actor_id
        try:
            host_snapshot = None if self.host_inventory is None else self.host_inventory.sample()
            self._fences.claim(
                request,
                fresh_eligible=frozenset(fresh_eligible),
                pressure_safe=frozenset(pressure_safe),
                implicit_reuse_actor_id=implicit_reuse_actor_id,
                host_snapshot=host_snapshot,
                now=observed_at,
            )
        except (AdmissionError, RuntimeError, OSError) as exc:
            return AdmissionDecision(
                False,
                request.lease_id,
                request.gpu_uuids,
                reason=str(exc),
            )
        return AdmissionDecision(
            True,
            request.lease_id,
            request.gpu_uuids,
            reused_actor_id=request.reuse_actor_id or implicit_reuse_actor_id,
            cooperative_pids=cooperative_pids,
            gpu_indices={uuid: snapshots[uuid].index for uuid in request.gpu_uuids},
        )

    def renew(self, request: LeaseRequest, *, now: float | None = None) -> int:
        """Persist a fresh local expiry only after scheduler renewal ACK."""

        observed_at = time.time() if now is None else now
        return self._fences.renew_lease(
            request,
            expires_at=observed_at + request.renewal_ttl_seconds,
            now=observed_at,
        )

    def verify_reserved_capacity(
        self,
        request: LeaseRequest,
        *,
        allowed_owned_pids: frozenset[int] = frozenset(),
    ) -> AdmissionDecision:
        """Re-probe after a cooperative sentinel handoff and before model load."""

        if not self._fences.owns_claims(request.lease_id, request.gpu_uuids):
            return AdmissionDecision(
                False,
                request.lease_id,
                request.gpu_uuids,
                reason="local GPU reservation was lost during cooperative handoff",
            )
        snapshots = {snapshot.uuid: snapshot for snapshot in self._inventory.sample()}
        required = request.min_free_vram_bytes + self.safety_margin_bytes
        for gpu_uuid in request.gpu_uuids:
            snapshot = snapshots.get(gpu_uuid)
            if snapshot is None:
                return AdmissionDecision(
                    False,
                    request.lease_id,
                    request.gpu_uuids,
                    reason=f"GPU {gpu_uuid} disappeared during cooperative handoff",
                )
            if (
                snapshot.foreign_processes
                and not self.allow_foreign_processes_when_capacity_sufficient
            ):
                return AdmissionDecision(
                    False,
                    request.lease_id,
                    request.gpu_uuids,
                    reason=f"GPU {gpu_uuid} gained an unknown foreign process",
                )
            if snapshot.cooperative_processes:
                return AdmissionDecision(
                    False,
                    request.lease_id,
                    request.gpu_uuids,
                    reason=f"GPU {gpu_uuid} cooperative scavenger did not exit",
                )
            unexpected_owned = [
                process.pid
                for process in snapshot.topodrome_processes
                if process.pid not in allowed_owned_pids
            ]
            if unexpected_owned:
                return AdmissionDecision(
                    False,
                    request.lease_id,
                    request.gpu_uuids,
                    reason=f"GPU {gpu_uuid} has an unexpected managed process",
                )
            if snapshot.free_memory_bytes < required:
                return AdmissionDecision(
                    False,
                    request.lease_id,
                    request.gpu_uuids,
                    reason=(
                        f"GPU {gpu_uuid} has {snapshot.free_memory_bytes} bytes free; "
                        f"{required} required after safety margin"
                    ),
                )
        return AdmissionDecision(
            True,
            request.lease_id,
            request.gpu_uuids,
            reused_actor_id=request.reuse_actor_id,
            gpu_indices={uuid: snapshots[uuid].index for uuid in request.gpu_uuids},
        )

    def release(self, lease_id: str) -> int:
        return self._fences.release_lease(lease_id)


__all__ = [
    "GIB",
    "ActorAllocation",
    "ActorEvictionIntent",
    "AdmissionController",
    "AdmissionDecision",
    "AdmissionError",
    "GPUUnavailableError",
    "LeaseFenceStore",
    "LeaseRequest",
    "StaleFenceError",
]
