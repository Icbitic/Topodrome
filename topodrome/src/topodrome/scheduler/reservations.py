"""Durable reconciliation for timed, exact physical GPU reservations."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select

from topodrome.db.models import (
    AttemptRow,
    GPUReservationRow,
    GPURow,
    KernelSessionRow,
    LeaseGPUClaimRow,
    LeaseRow,
    ModelActorRow,
    SiteRow,
)
from topodrome.db.repositories import SchedulerRepository
from topodrome.db.session import Database
from topodrome.domain.enums import (
    ActorState,
    AttemptState,
    GPUReservationState,
    KernelState,
    LeaseState,
    SiteState,
    WorkloadKind,
)
from topodrome.domain.events import SchedulerEnvelope
from topodrome.domain.specs import JobSpec, utc_now
from topodrome.scheduler.control import AgentConnectionRegistry

log = structlog.get_logger(__name__)

_EVICTION_RETRY_SECONDS = 5
_EVICTION_SEND_TIMEOUT_SECONDS = 5


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


@dataclass(frozen=True, slots=True)
class _EvictionCommand:
    reservation_id: str
    site_id: str
    target_gpu_uuid: str
    expected_generation: str
    actor_id: str
    gpu_uuids: tuple[str, ...]
    replica_key_hash: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ForcedCommand:
    reservation_id: str
    site_id: str
    gpu_uuid: str
    attempt_id: str
    expected_generation: str
    cancel: bool


_ACTIVE_ATTEMPT_STATES = (
    AttemptState.OFFERED,
    AttemptState.ACCEPTED,
    AttemptState.STARTING,
    AttemptState.RUNNING,
    AttemptState.PREEMPT_REQUESTED,
    AttemptState.CHECKPOINTING,
    AttemptState.COMMITTING,
)


class GPUReservationCoordinator:
    """Converge ordinary idle-only and explicitly forced reservations.

    Reservation admission is enforced directly by repository inventory reads.
    Forced reservations additionally preempt exact active attempts. The site
    agent performs the final transactional ownership and activity checks.
    """

    def __init__(
        self,
        *,
        database: Database,
        repository: SchedulerRepository,
        connections: AgentConnectionRegistry,
        heartbeat_ttl_seconds: float,
    ) -> None:
        self.database = database
        self.repository = repository
        self.connections = connections
        self.heartbeat_ttl_seconds = heartbeat_ttl_seconds

    async def reconcile_once(self, *, now: datetime | None = None) -> int:
        current = _aware(now or utc_now())
        identities = await self.connections.identities()
        commands: list[_EvictionCommand] = []
        forced_commands: list[_ForcedCommand] = []
        async with self.database.transaction() as session:
            reservations = await self.repository.lock_active_gpu_reservations(
                session,
                now=current,
            )
            if not reservations:
                return 0
            targets = {(row.site_id, row.gpu_uuid) for row in reservations}
            inventory_keys = set(
                (
                    await session.execute(
                        select(GPURow.site_id, GPURow.uuid).where(
                            GPURow.site_id.in_({site_id for site_id, _ in targets})
                        )
                    )
                ).all()
            )
            active_claims: set[tuple[str, str]] = {
                (site_id, gpu_uuid)
                for site_id, gpu_uuid in (
                    await session.execute(
                        select(
                            LeaseGPUClaimRow.site_id,
                            LeaseGPUClaimRow.gpu_uuid,
                        ).where(LeaseGPUClaimRow.active.is_(True))
                    )
                ).all()
            }
            actors = list(
                await session.scalars(
                    select(ModelActorRow).where(ModelActorRow.active_allocation.is_(True))
                )
            )
            actors_by_gpu = {(row.site_id, row.gpu_uuid): row for row in actors}
            actors_by_replica: dict[tuple[str, str], list[ModelActorRow]] = {}
            for actor in actors:
                replica_id = actor.key_data.get("replica_id")
                if isinstance(replica_id, str) and replica_id:
                    actors_by_replica.setdefault((actor.site_id, replica_id), []).append(actor)
            sites = {
                row.id: row
                for row in await session.scalars(
                    select(SiteRow).where(SiteRow.id.in_({site_id for site_id, _ in targets}))
                )
            }
            heartbeat_cutoff = current - timedelta(seconds=self.heartbeat_ttl_seconds)

            for reservation in reservations:
                key = (reservation.site_id, reservation.gpu_uuid)
                site = sites.get(reservation.site_id)
                current_identity = identities.get(reservation.site_id)
                inventory_current = bool(
                    key in inventory_keys
                    and site is not None
                    and site.state is SiteState.READY
                    and site.heartbeat_at is not None
                    and _aware(site.heartbeat_at) >= heartbeat_cutoff
                    and current_identity
                    == (site.agent_session_id, site.agent_connection_generation)
                )
                target_actor = actors_by_gpu.get(key)
                if key in active_claims or target_actor is not None or not inventory_current:
                    reservation.state = GPUReservationState.PENDING
                else:
                    reservation.state = GPUReservationState.READY
                reservation.updated_at = current

                if (
                    reservation.force
                    and key in active_claims
                    and inventory_current
                    and (
                        reservation.eviction_requested_at is None
                        or _aware(reservation.eviction_requested_at)
                        <= current - timedelta(seconds=_EVICTION_RETRY_SECONDS)
                    )
                ):
                    assert current_identity is not None
                    attempts = list(
                        await session.scalars(
                            select(AttemptRow)
                            .join(LeaseRow, LeaseRow.attempt_id == AttemptRow.id)
                            .join(LeaseGPUClaimRow, LeaseGPUClaimRow.lease_id == LeaseRow.id)
                            .where(
                                LeaseGPUClaimRow.site_id == reservation.site_id,
                                LeaseGPUClaimRow.gpu_uuid == reservation.gpu_uuid,
                                LeaseGPUClaimRow.active.is_(True),
                                LeaseRow.site_id == reservation.site_id,
                                LeaseRow.state.in_(
                                    (LeaseState.OFFERED, LeaseState.ACCEPTED, LeaseState.ACTIVE)
                                ),
                                LeaseRow.expires_at > current,
                                AttemptRow.site_id == reservation.site_id,
                                AttemptRow.state.in_(_ACTIVE_ATTEMPT_STATES),
                            )
                            .with_for_update(of=AttemptRow)
                        )
                    )
                    for attempt in attempts:
                        kernels = list(
                            await session.scalars(
                                select(KernelSessionRow)
                                .where(
                                    KernelSessionRow.attempt_id == attempt.id,
                                    KernelSessionRow.state.in_(
                                        (
                                            KernelState.REQUESTED,
                                            KernelState.STARTING,
                                            KernelState.RUNNING,
                                            KernelState.IDLE,
                                            KernelState.STOPPING,
                                        )
                                    ),
                                )
                                .with_for_update()
                            )
                        )
                        job = await self.repository.get_job(session, attempt.job_id)
                        interactive = bool(kernels) or (
                            JobSpec.model_validate(job.spec).workload_kind
                            is WorkloadKind.INTERACTIVE
                        )
                        if not interactive:
                            # Persist retry attribution before sending any stop.
                            # Ordinary cancellation remains authoritative.
                            attempt.forced_reservation_id = (
                                attempt.forced_reservation_id or reservation.id
                            )
                        forced_commands.append(
                            _ForcedCommand(
                                reservation.id,
                                reservation.site_id,
                                reservation.gpu_uuid,
                                attempt.id,
                                current_identity[1],
                                interactive,
                            )
                        )

                if key in active_claims or target_actor is None or not inventory_current:
                    continue
                assert current_identity is not None
                command = _command_for_idle_actor(
                    reservation,
                    target_actor,
                    actors_by_replica=actors_by_replica,
                    active_claims=active_claims,
                    now=current,
                    expected_generation=current_identity[1],
                )
                if command is not None:
                    commands.append(command)

        delivered = 0
        for command in forced_commands:
            if await self._deliver_forced_if_still_live(command):
                delivered += 1
        for command in commands:
            if await self._deliver_if_still_live(command):
                delivered += 1
        return delivered

    async def _deliver_forced_if_still_live(self, command: _ForcedCommand) -> bool:
        # Cancel/expiry and final delivery share the durable reservation lock.
        # Never look up a replacement attempt by job ID: stop only the exact
        # attempt whose active physical claim still covers the reserved UUID.
        async with self.database.transaction() as session:
            reservation = await self.repository.get_gpu_reservation(
                session, command.reservation_id, lock=True
            )
            remaining = (_aware(reservation.expires_at) - _aware(utc_now())).total_seconds()
            if (
                not reservation.force
                or remaining <= 0
                or reservation.state not in {GPUReservationState.PENDING, GPUReservationState.READY}
            ):
                return False
            if (reservation.site_id, reservation.gpu_uuid) != (command.site_id, command.gpu_uuid):
                return False
            attempt = await session.scalar(
                select(AttemptRow)
                .join(LeaseRow, LeaseRow.attempt_id == AttemptRow.id)
                .join(LeaseGPUClaimRow, LeaseGPUClaimRow.lease_id == LeaseRow.id)
                .where(
                    AttemptRow.id == command.attempt_id,
                    AttemptRow.site_id == command.site_id,
                    AttemptRow.state.in_(_ACTIVE_ATTEMPT_STATES),
                    LeaseGPUClaimRow.site_id == command.site_id,
                    LeaseGPUClaimRow.gpu_uuid == command.gpu_uuid,
                    LeaseGPUClaimRow.active.is_(True),
                    LeaseRow.site_id == command.site_id,
                    LeaseRow.state.in_(
                        (LeaseState.OFFERED, LeaseState.ACCEPTED, LeaseState.ACTIVE)
                    ),
                    LeaseRow.expires_at > utc_now(),
                )
                .with_for_update(of=AttemptRow)
            )
            if attempt is None:
                return False
            if command.cancel:
                await self.repository.cancel_job(session, attempt.job_id)
                kernels = await session.scalars(
                    select(KernelSessionRow)
                    .where(
                        KernelSessionRow.attempt_id == attempt.id,
                        KernelSessionRow.state.in_(
                            (
                                KernelState.REQUESTED,
                                KernelState.STARTING,
                                KernelState.RUNNING,
                                KernelState.IDLE,
                                KernelState.STOPPING,
                            )
                        ),
                    )
                    .with_for_update()
                )
                for kernel in kernels:
                    kernel.state = KernelState.STOPPING
            try:
                async with asyncio.timeout(min(remaining, _EVICTION_SEND_TIMEOUT_SECONDS)):
                    sent = await self.connections.send(
                        command.site_id,
                        SchedulerEnvelope(
                            site_id=command.site_id,
                            kind="request_checkpoint",
                            payload={
                                "attempt_id": command.attempt_id,
                                "cancel_requested": command.cancel,
                                "reason_code": "operator_forced_gpu_reservation",
                            },
                        ),
                        expected_generation=command.expected_generation,
                    )
            except Exception as exc:
                log.warning(
                    "forced_reservation_delivery_failed",
                    reservation_id=command.reservation_id,
                    error_type=type(exc).__name__,
                )
                return False
            if sent:
                reservation.eviction_requested_at = _aware(utc_now())
            return sent

    async def _deliver_if_still_live(self, command: _EvictionCommand) -> bool:
        """Serialize final delivery with cancellation and expiry.

        Selection deliberately happens outside the network send, so a cancel
        can commit after selection. Re-locking the durable intent and retaining
        that lock until the outbound writer has accepted the command gives the
        two operations one database order: a committed cancel wins and sends
        nothing; otherwise cancellation waits until the already-authorized
        delivery completes. The transport is bounded by the remaining intent
        lifetime so waiting on a slow socket cannot authorize a late eviction.
        """

        async with self.database.transaction() as session:
            reservation = await self.repository.get_gpu_reservation(
                session,
                command.reservation_id,
                lock=True,
            )
            delivery_now = _aware(utc_now())
            if reservation.state not in {
                GPUReservationState.PENDING,
                GPUReservationState.READY,
            }:
                return False
            if _aware(reservation.expires_at) <= delivery_now:
                reservation.state = GPUReservationState.EXPIRED
                reservation.updated_at = delivery_now
                await session.flush()
                return False
            if (
                reservation.site_id != command.site_id
                or reservation.gpu_uuid != command.target_gpu_uuid
                or command.target_gpu_uuid not in command.gpu_uuids
            ):
                return False

            # Repeat the active-work and exact-replica proof under the final
            # intent lock. The site repeats the same proof against its durable
            # local claims before disposing anything.
            active_claim = await session.scalar(
                select(LeaseGPUClaimRow.lease_id)
                .where(
                    LeaseGPUClaimRow.site_id == command.site_id,
                    LeaseGPUClaimRow.gpu_uuid.in_(command.gpu_uuids),
                    LeaseGPUClaimRow.active.is_(True),
                )
                .limit(1)
            )
            if active_claim is not None:
                reservation.state = GPUReservationState.PENDING
                reservation.updated_at = delivery_now
                return False

            actor_rows = list(
                await session.scalars(
                    select(ModelActorRow)
                    .where(
                        ModelActorRow.site_id == command.site_id,
                        ModelActorRow.gpu_uuid.in_(command.gpu_uuids),
                        ModelActorRow.active_allocation.is_(True),
                    )
                    .with_for_update()
                )
            )
            actors_by_gpu = {row.gpu_uuid: row for row in actor_rows}
            if set(actors_by_gpu) != set(command.gpu_uuids):
                if command.target_gpu_uuid not in actors_by_gpu:
                    reservation.state = GPUReservationState.READY
                    reservation.updated_at = delivery_now
                return False
            if any(
                row.state is not ActorState.IDLE
                or row.owner_attempt_id is not None
                or row.key_data.get("replica_id") != command.actor_id
                or not isinstance(row.key_data.get("replica_gpu_uuids"), list)
                or tuple(row.key_data["replica_gpu_uuids"]) != command.gpu_uuids
                or row.key_data.get("replica_key_hash") != command.replica_key_hash
                for row in actors_by_gpu.values()
            ):
                reservation.state = GPUReservationState.PENDING
                reservation.updated_at = delivery_now
                return False

            remaining = (_aware(reservation.expires_at) - _aware(utc_now())).total_seconds()
            if remaining <= 0:
                reservation.state = GPUReservationState.EXPIRED
                reservation.updated_at = _aware(utc_now())
                await session.flush()
                return False
            try:
                async with asyncio.timeout(min(remaining, _EVICTION_SEND_TIMEOUT_SECONDS)):
                    sent = await self.connections.send(
                        command.site_id,
                        SchedulerEnvelope(
                            site_id=command.site_id,
                            kind="evict_actor",
                            payload=command.payload,
                        ),
                        expected_generation=command.expected_generation,
                    )
            except TimeoutError:
                timed_out_at = _aware(utc_now())
                if _aware(reservation.expires_at) <= timed_out_at:
                    reservation.state = GPUReservationState.EXPIRED
                    reservation.updated_at = timed_out_at
                    await session.flush()
                else:
                    log.warning(
                        "gpu_reservation_eviction_delivery_timed_out",
                        reservation_id=command.reservation_id,
                        site_id=command.site_id,
                    )
                return False
            except Exception as exc:
                log.warning(
                    "gpu_reservation_eviction_delivery_failed",
                    reservation_id=command.reservation_id,
                    site_id=command.site_id,
                    error_type=type(exc).__name__,
                )
                return False
            if not sent:
                return False

            # A crash after command delivery but before commit is harmless: the
            # stable site eviction ID makes the next reconciliation idempotent.
            delivered_at = _aware(utc_now())
            reservation.eviction_requested_at = delivered_at
            reservation.updated_at = delivered_at
            return True


def _command_for_idle_actor(
    reservation: GPUReservationRow,
    target_actor: ModelActorRow,
    *,
    actors_by_replica: dict[tuple[str, str], list[ModelActorRow]],
    active_claims: set[tuple[str, str]],
    now: datetime,
    expected_generation: str,
) -> _EvictionCommand | None:
    if target_actor.state is not ActorState.IDLE or target_actor.owner_attempt_id is not None:
        return None
    if reservation.eviction_requested_at is not None and _aware(
        reservation.eviction_requested_at
    ) > now - timedelta(seconds=_EVICTION_RETRY_SECONDS):
        return None
    replica_id = target_actor.key_data.get("replica_id")
    raw_members = target_actor.key_data.get("replica_gpu_uuids")
    replica_key_hash = target_actor.key_data.get("replica_key_hash")
    if (
        not isinstance(replica_id, str)
        or not replica_id
        or not isinstance(raw_members, list)
        or not raw_members
        or not all(isinstance(value, str) and value for value in raw_members)
        or len(set(raw_members)) != len(raw_members)
        or not isinstance(replica_key_hash, str)
        or len(replica_key_hash) != 64
    ):
        return None
    members = tuple(raw_members)
    group = actors_by_replica.get((reservation.site_id, replica_id), [])
    by_gpu = {row.gpu_uuid: row for row in group}
    if set(by_gpu) != set(members):
        return None
    if any(
        row.state is not ActorState.IDLE
        or row.owner_attempt_id is not None
        or row.key_data.get("replica_key_hash") != replica_key_hash
        for row in by_gpu.values()
    ):
        return None
    if any((reservation.site_id, gpu_uuid) in active_claims for gpu_uuid in members):
        return None
    digest = hashlib.sha256(
        json.dumps(
            {
                "actor_id": replica_id,
                "gpu_uuids": members,
                "replica_key_hash": replica_key_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:32]
    return _EvictionCommand(
        reservation_id=reservation.id,
        site_id=reservation.site_id,
        target_gpu_uuid=reservation.gpu_uuid,
        expected_generation=expected_generation,
        actor_id=replica_id,
        gpu_uuids=members,
        replica_key_hash=replica_key_hash,
        payload={
            "eviction_id": f"reservation:{reservation.id}:{digest}",
            "actor_id": replica_id,
            "gpu_uuids": list(members),
            "replica_key_hash": replica_key_hash,
            "reason_code": "operator_gpu_reservation",
        },
    )
