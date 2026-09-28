"""Central policy for safely shedding surplus institutional warm actors.

This module only selects complete idle replicas.  The site agent remains the
physical authority: it atomically rechecks the exact actor, every member, and
all local workload/kernel claims before disposing anything.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from topodrome.domain.enums import ActorState, SiteKind
from topodrome.domain.specs import JobSpec
from topodrome.scheduler.placement import GPUCandidate, SiteCandidate

InstitutionalActorEvictionReason = Literal[
    "foreign_contention_surplus",
    "excess_idle_warm_actor",
]

_MODEL_IDENTITY_KEYS = (
    "alias",
    "revision",
    "backend",
    "backend_version",
    "dtype",
    "quantization",
    "adapter",
    "runtime_fingerprint",
)


@dataclass(frozen=True, slots=True)
class WarmActorReplica:
    site_id: str
    actor_id: str
    gpu_uuids: tuple[str, ...]
    replica_key_hash: str
    model_identity: dict[str, object]
    foreign_process_count: int
    foreign_memory_bytes: int
    max_utilization_percent: float
    connection_generation: str
    idle_since: float | None

    @property
    def key(self) -> tuple[str, str, tuple[str, ...], tuple[tuple[str, object], ...]]:
        return (
            self.site_id,
            self.actor_id,
            self.gpu_uuids,
            (
                ("connection_generation", self.connection_generation),
                ("replica_key_hash", self.replica_key_hash),
                ("idle_since", self.idle_since),
                *((name, self.model_identity.get(name)) for name in _MODEL_IDENTITY_KEYS),
            ),
        )

    @property
    def shares_with_foreign_processes(self) -> bool:
        return self.foreign_process_count > 0


@dataclass(frozen=True, slots=True)
class ActorEvictionPlan:
    site_id: str
    actor_id: str
    gpu_uuids: tuple[str, ...]
    replica_key_hash: str
    model_identity: dict[str, object]
    connection_generation: str
    reason_code: InstitutionalActorEvictionReason
    occupied_institutional_gpu_count: int
    residency_epoch: int
    minimum_idle_seconds: float | None

    @property
    def key(self) -> tuple[str, str, tuple[str, ...]]:
        return (self.site_id, self.actor_id, self.gpu_uuids)

    def eviction_id(self, *, policy_generation: int) -> str:
        """Return a stable site idempotency key for this observed residency.

        The connection generation and observed-residency epoch make a later
        actor incarnation distinct even when Ray reuses the same actor name,
        model key, and GPU topology.  Repeated delivery attempts for the same
        observation deliberately produce the same ID.
        """

        digest = hashlib.sha256(
            json.dumps(
                {
                    "actor_id": self.actor_id,
                    "connection_generation": self.connection_generation,
                    "gpu_uuids": self.gpu_uuids,
                    "minimum_idle_seconds": self.minimum_idle_seconds,
                    "policy_generation": policy_generation,
                    "reason_code": self.reason_code,
                    "replica_key_hash": self.replica_key_hash,
                    "residency_epoch": self.residency_epoch,
                    "site_id": self.site_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:32]
        return f"actor-policy:{policy_generation}:{digest}"


class InstitutionalActorPolicy:
    """Track continuous observations and return at most one safe eviction.

    Contention stability and delivery cooldowns live in scheduler memory.
    Idle age comes from the site's durable allocation fence and is repeated in
    authenticated heartbeats, so the configured expiry survives reconnects
    and scheduler restarts while every successful reuse resets it.
    """

    def __init__(self) -> None:
        self._contended_since: dict[
            tuple[str, str, tuple[str, ...], tuple[tuple[str, object], ...]], float
        ] = {}
        self._cooldown_until: dict[tuple[str, str, tuple[str, ...]], float] = {}
        self._residency_epochs: dict[
            tuple[str, str, tuple[str, ...], tuple[tuple[str, object], ...]], int
        ] = {}
        self._next_residency_epoch = 0

    def choose_eviction(
        self,
        sites: list[SiteCandidate],
        queued_specs: tuple[JobSpec, ...],
        *,
        actor_floor_gpu_count: int,
        contention_threshold_gpu_count: int,
        stability_seconds: float,
        idle_timeout_seconds: float | None,
        idle_timeout_seconds_by_site: Mapping[str, float | None] | None = None,
        now: float | None = None,
    ) -> ActorEvictionPlan | None:
        observed_at = time.monotonic() if now is None else now
        wall_observed_at = time.time() if now is None else now
        replicas = _complete_idle_replicas(sites)
        current_keys = {replica.key for replica in replicas}
        self._contended_since = {
            key: since for key, since in self._contended_since.items() if key in current_keys
        }
        self._residency_epochs = {
            key: epoch for key, epoch in self._residency_epochs.items() if key in current_keys
        }
        self._cooldown_until = {
            key: until for key, until in self._cooldown_until.items() if until > observed_at
        }
        for replica in replicas:
            if replica.key not in self._residency_epochs:
                self._next_residency_epoch += 1
                self._residency_epochs[replica.key] = self._next_residency_epoch
            if replica.shares_with_foreign_processes:
                self._contended_since.setdefault(replica.key, observed_at)
            else:
                self._contended_since.pop(replica.key, None)

        occupied = occupied_institutional_gpu_count(sites)
        resident_members = {
            (replica.site_id, gpu_uuid) for replica in replicas for gpu_uuid in replica.gpu_uuids
        }

        # The retained warm floor protects throughput only while real work is
        # waiting. With an empty queue, the 20-minute expiry may return all
        # institutional GPUs to other users. Queue-driven replacement is not
        # targetless release: the real offered lease preserves occupancy.
        effective_floor = actor_floor_gpu_count if queued_specs else 0

        def floor_is_preserved(replica: WarmActorReplica) -> bool:
            after = resident_members - {
                (replica.site_id, gpu_uuid) for gpu_uuid in replica.gpu_uuids
            }
            return len(after) >= effective_floor

        eligible = [
            replica
            for replica in replicas
            if floor_is_preserved(replica)
            and self._cooldown_until.get(
                (replica.site_id, replica.actor_id, replica.gpu_uuids),
                0,
            )
            <= observed_at
        ]
        demanded_aliases = {spec.model.alias for spec in queued_specs}

        if occupied >= contention_threshold_gpu_count:
            contended = [
                replica
                for replica in eligible
                if replica.shares_with_foreign_processes
                and observed_at - self._contended_since[replica.key] >= stability_seconds
            ]
            if contended:
                selected = min(
                    contended,
                    key=lambda replica: (
                        replica.model_identity.get("alias") in demanded_aliases,
                        len(replica.gpu_uuids),
                        -replica.foreign_memory_bytes,
                        -replica.foreign_process_count,
                        -replica.max_utilization_percent,
                        replica.site_id,
                        replica.actor_id,
                    ),
                )
                return _plan(
                    selected,
                    reason_code="foreign_contention_surplus",
                    occupied=occupied,
                    residency_epoch=self._residency_epochs[selected.key],
                    minimum_idle_seconds=None,
                )

        overrides = idle_timeout_seconds_by_site or {}
        if idle_timeout_seconds is not None or overrides:
            expired = [
                replica
                for replica in eligible
                if replica.idle_since is not None
                and (timeout := overrides.get(replica.site_id, idle_timeout_seconds)) is not None
                and wall_observed_at - replica.idle_since >= timeout
            ]
            if expired:
                selected = min(
                    expired,
                    key=lambda replica: (
                        replica.model_identity.get("alias") in demanded_aliases,
                        replica.idle_since,
                        len(replica.gpu_uuids),
                        replica.site_id,
                        replica.actor_id,
                    ),
                )
                return _plan(
                    selected,
                    reason_code="excess_idle_warm_actor",
                    occupied=occupied,
                    residency_epoch=self._residency_epochs[selected.key],
                    minimum_idle_seconds=overrides.get(selected.site_id, idle_timeout_seconds),
                )
        return None

    def mark_dispatched(
        self,
        plan: ActorEvictionPlan,
        *,
        cooldown_seconds: float,
        now: float | None = None,
    ) -> None:
        observed_at = time.monotonic() if now is None else now
        self._cooldown_until[plan.key] = observed_at + cooldown_seconds

    def clear(self) -> None:
        self._contended_since.clear()
        self._cooldown_until.clear()
        self._residency_epochs.clear()


def occupied_institutional_gpu_count(sites: list[SiteCandidate]) -> int:
    """Count unique institutional GPUs with a lease or resident warm actor."""

    return sum(
        1
        for site in sites
        if site.kind in {SiteKind.LAB, SiteKind.JUPYTER}
        for gpu in site.gpus
        if gpu.active_lease or gpu.actor_key_hash is not None
    )


def _complete_idle_replicas(sites: list[SiteCandidate]) -> tuple[WarmActorReplica, ...]:
    replicas: list[WarmActorReplica] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for site in sites:
        if site.kind not in {SiteKind.LAB, SiteKind.JUPYTER}:
            continue
        if not site.agent_connection_generation:
            continue
        by_uuid = {gpu.uuid: gpu for gpu in site.gpus}
        for gpu in site.gpus:
            key_data = gpu.actor_key_data
            if key_data is None or gpu.actor_key_hash is None:
                continue
            actor_id = key_data.get("replica_id")
            raw_members = key_data.get("replica_gpu_uuids")
            if not isinstance(actor_id, str) or not actor_id:
                continue
            if not isinstance(raw_members, (tuple, list)) or not raw_members:
                continue
            members = tuple(str(member) for member in raw_members)
            identity = (site.site_id, actor_id, members)
            if identity in seen or len(set(members)) != len(members):
                continue
            selected = tuple(by_uuid.get(member) for member in members)
            if any(member is None for member in selected):
                continue
            exact_members = tuple(member for member in selected if member is not None)
            model_identity = _model_identity(key_data)
            replica_key_hash = key_data.get("replica_key_hash")
            if (
                not isinstance(replica_key_hash, str)
                or len(replica_key_hash) != 64
                or any(character not in "0123456789abcdef" for character in replica_key_hash)
            ):
                continue
            if not _complete_idle_replica(
                actor_id,
                members,
                replica_key_hash,
                model_identity,
                exact_members,
            ):
                continue
            seen.add(identity)
            replicas.append(
                WarmActorReplica(
                    site_id=site.site_id,
                    actor_id=actor_id,
                    gpu_uuids=members,
                    replica_key_hash=replica_key_hash,
                    model_identity=model_identity,
                    foreign_process_count=sum(
                        member.foreign_process_count for member in exact_members
                    ),
                    foreign_memory_bytes=sum(
                        member.foreign_memory_bytes for member in exact_members
                    ),
                    max_utilization_percent=max(
                        member.utilization_percent for member in exact_members
                    ),
                    connection_generation=site.agent_connection_generation,
                    idle_since=_replica_idle_since(exact_members),
                )
            )
    return tuple(replicas)


def _complete_idle_replica(
    actor_id: str,
    members: tuple[str, ...],
    replica_key_hash: str,
    model_identity: dict[str, object],
    gpus: tuple[GPUCandidate, ...],
) -> bool:
    for member_uuid, gpu in zip(members, gpus, strict=True):
        key_data = gpu.actor_key_data
        raw_members = None if key_data is None else key_data.get("replica_gpu_uuids")
        if (
            key_data is None
            or not isinstance(raw_members, (tuple, list))
            or gpu.actor_key_hash is None
            or gpu.actor_state is not ActorState.IDLE
            or gpu.active_lease
            or gpu.externally_reserved
            or key_data.get("replica_id") != actor_id
            or key_data.get("replica_key_hash") != replica_key_hash
            or tuple(str(value) for value in raw_members) != members
            or key_data.get("gpu_uuid") != member_uuid
            or _model_identity(key_data) != model_identity
        ):
            return False
    return True


def _model_identity(key_data: dict[str, object]) -> dict[str, object]:
    return {key: key_data.get(key) for key in _MODEL_IDENTITY_KEYS}


def _replica_idle_since(gpus: tuple[GPUCandidate, ...]) -> float | None:
    values: list[float] = []
    for gpu in gpus:
        key_data = gpu.actor_key_data
        value = None if key_data is None else key_data.get("idle_since")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
            continue
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return None
            values.append(parsed.timestamp())
            continue
        return None
    return max(values) if len(values) == len(gpus) and values else None


def _plan(
    replica: WarmActorReplica,
    *,
    reason_code: InstitutionalActorEvictionReason,
    occupied: int,
    residency_epoch: int,
    minimum_idle_seconds: float | None,
) -> ActorEvictionPlan:
    return ActorEvictionPlan(
        site_id=replica.site_id,
        actor_id=replica.actor_id,
        gpu_uuids=replica.gpu_uuids,
        replica_key_hash=replica.replica_key_hash,
        model_identity=replica.model_identity,
        connection_generation=replica.connection_generation,
        reason_code=reason_code,
        occupied_institutional_gpu_count=occupied,
        residency_epoch=residency_epoch,
        minimum_idle_seconds=minimum_idle_seconds,
    )
