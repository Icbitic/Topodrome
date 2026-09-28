"""Manual AutoDL drain coordination and journal-backed status; no rental API."""

from __future__ import annotations

import asyncio
import hmac
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from topodrome.config import SchedulerSettings
from topodrome.db.models import (
    ArtifactRow,
    AttemptRow,
    KernelSessionRow,
    ProviderInstanceRow,
    ProviderOperationRow,
    SiteRow,
)
from topodrome.db.session import Database
from topodrome.domain.enums import (
    ArtifactState,
    AttemptState,
    KernelState,
    ProviderInstanceState,
    SiteKind,
    SiteState,
)
from topodrome.domain.events import AgentEnvelope, SchedulerEnvelope
from topodrome.domain.specs import SiteHeartbeat, utc_now
from topodrome.providers.autodl.lifecycle import IdleActivity
from topodrome.scheduler.control import AgentConnectionRegistry

_RUNNING_ATTEMPTS = (
    AttemptState.STARTING,
    AttemptState.RUNNING,
    AttemptState.PREEMPT_REQUESTED,
    AttemptState.CHECKPOINTING,
    AttemptState.COMMITTING,
)
_LOCALLY_QUEUED_ATTEMPTS = (AttemptState.OFFERED, AttemptState.ACCEPTED)
_ACTIVE_KERNELS = (
    KernelState.STARTING,
    KernelState.RUNNING,
    KernelState.IDLE,
    KernelState.STOPPING,
)
_PENDING_ARTIFACTS = (ArtifactState.STAGING, ArtifactState.FLUSHING)


class ProviderOrchestrator(Protocol):
    """Provider hook consumed by the otherwise provider-neutral scheduler."""

    async def reconcile_once(self) -> AutoDLReconcileResult: ...

    async def observe_agent_event(
        self,
        envelope: AgentEnvelope,
        *,
        connection_generation: str | None = None,
    ) -> None: ...

    async def instance_summaries(self) -> list[dict[str, Any]]: ...

    async def operation_summaries(self) -> list[dict[str, Any]]: ...

    async def close(self) -> None: ...

    def control_status(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AutoDLReconcileResult:
    observed_instances: int
    reconciled_operations: int
    powered_off_instances: int


@dataclass(frozen=True, slots=True)
class _ShutdownContext:
    instance_uuid: str
    site_id: str
    generation: int
    agent_session_id: str
    connection_generation: str
    drain_nonce: str
    durability_nonce: str | None = None


class SchedulerAutoDLBridge:
    """Map site-agent state to lifecycle drain and durability protocols.

    Missing or stale telemetry is deliberately considered active.  A provider
    power-off can only follow a delivered drain command, a durable drain-ready
    event, and a second durability-ready event emitted after the idle site has
    synchronized its filesystems.
    """

    def __init__(
        self,
        database: Database,
        connections: AgentConnectionRegistry,
        *,
        heartbeat_ttl_seconds: int,
        acknowledgement_timeout_seconds: float,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        self.database = database
        self.connections = connections
        self.heartbeat_ttl = timedelta(seconds=heartbeat_ttl_seconds)
        self.acknowledgement_timeout_seconds = acknowledgement_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    async def mark_draining(self, instance_uuid: str) -> None:
        context = await self._set_draining(instance_uuid)
        delivered = await self.connections.send(
            context.site_id,
            SchedulerEnvelope(
                site_id=context.site_id,
                kind="begin_drain",
                payload={
                    "shutdown_generation": context.generation,
                    "command_nonce": context.drain_nonce,
                },
            ),
            expected_generation=context.connection_generation,
        )
        if not delivered:
            raise RuntimeError("AutoDL drain command could not be delivered")
        if not await self._wait_for_shutdown_ack(context, stage="drain"):
            raise TimeoutError("AutoDL site did not acknowledge a complete drain")

    async def current_activity(self, instance_uuid: str) -> IdleActivity:
        async with self.database.session() as session:
            instance = await session.scalar(
                select(ProviderInstanceRow).where(
                    ProviderInstanceRow.provider == "autodl",
                    ProviderInstanceRow.external_id == instance_uuid,
                )
            )
            if instance is None or not instance.site_id:
                return _unknown_activity()
            site = await session.get(SiteRow, instance.site_id)
            if (
                site is None
                or site.heartbeat_at is None
                or _as_utc(site.heartbeat_at) < utc_now() - self.heartbeat_ttl
            ):
                return _unknown_activity()
            identity = await self.connections.identity(site.id)
            if identity != (
                site.agent_session_id,
                site.agent_connection_generation,
            ):
                return _unknown_activity()

            running = await _has_rows(
                session,
                AttemptRow,
                AttemptRow.site_id == site.id,
                AttemptRow.state.in_(_RUNNING_ATTEMPTS),
            )
            queued = await _has_rows(
                session,
                AttemptRow,
                AttemptRow.site_id == site.id,
                AttemptRow.state.in_(_LOCALLY_QUEUED_ATTEMPTS),
            )
            interactive = await _has_rows(
                session,
                KernelSessionRow,
                KernelSessionRow.site_id == site.id,
                KernelSessionRow.state.in_(_ACTIVE_KERNELS),
            )
            pending_artifacts = await _has_rows(
                session,
                ArtifactRow,
                ArtifactRow.state.in_(_PENDING_ARTIFACTS),
                ArtifactRow.attempt_id.in_(
                    select(AttemptRow.id).where(AttemptRow.site_id == site.id)
                ),
            )
            return IdleActivity(
                job_running=site.job_running or running,
                assignment_queued_locally=site.assignment_queued_locally or queued,
                interactive_kernel_active=site.interactive_kernel_active or interactive,
                artifact_flush_pending=site.artifact_flush_pending or pending_artifacts,
                staging_operation_active=site.staging_operation_active,
            )

    async def flush_and_verify(self, instance_uuid: str) -> bool:
        context = await self._begin_durability(instance_uuid)
        if context is None:
            return False
        delivered = await self.connections.send(
            context.site_id,
            SchedulerEnvelope(
                site_id=context.site_id,
                kind="verify_durability",
                payload={
                    "shutdown_generation": context.generation,
                    "command_nonce": context.durability_nonce,
                },
            ),
            expected_generation=context.connection_generation,
        )
        return delivered and await self._wait_for_shutdown_ack(
            context,
            stage="durability",
        )

    async def persist_regional_state(self, instance_uuid: str) -> bool:
        return await self._shutdown_acknowledged(instance_uuid, stage="durability")

    async def is_durable(self, instance_uuid: str) -> bool:
        async with self.database.session() as session:
            tracked = await self._instance_row(session, instance_uuid)
            if (
                tracked.state
                in {
                    ProviderInstanceState.STOPPED,
                    ProviderInstanceState.STARTING_FAILED,
                    ProviderInstanceState.DURABILITY_CONFIRMED,
                }
                and tracked.durable_state_confirmed
            ):
                return True
        if not await self._shutdown_acknowledged(instance_uuid, stage="durability"):
            return False
        activity = await self.current_activity(instance_uuid)
        if activity.artifact_flush_pending or activity.staging_operation_active:
            return False
        async with self.database.transaction() as session:
            row = await session.scalar(
                select(ProviderInstanceRow)
                .where(
                    ProviderInstanceRow.provider == "autodl",
                    ProviderInstanceRow.external_id == instance_uuid,
                )
                .with_for_update()
            )
            if row is None:
                return False
            context = _shutdown_context(row)
            if context is None or not _context_acknowledged(row, context, "durability"):
                return False
            row.durable_state_confirmed = True
        return True

    async def stop_runtime(self, instance_uuid: str) -> int:
        context = await self._current_shutdown_context(
            instance_uuid,
            require_durability=True,
        )
        if context is None or context.durability_nonce is None:
            raise RuntimeError("AutoDL shutdown does not have a current durability barrier")
        delivered = await self.connections.send(
            context.site_id,
            SchedulerEnvelope(
                site_id=context.site_id,
                kind="stop_agent",
                payload={
                    "stop_ray": True,
                    "shutdown_generation": context.generation,
                    "command_nonce": context.durability_nonce,
                },
            ),
            expected_generation=context.connection_generation,
        )
        if not delivered:
            raise RuntimeError("AutoDL stop-agent command could not be delivered")
        if not await self._wait_for_disconnect(context):
            raise TimeoutError("AutoDL site agent did not stop cleanly")
        async with self.database.transaction() as session:
            row = await self._instance_row(session, instance_uuid, lock=True)
            if not _context_matches(row, context):
                raise RuntimeError("AutoDL shutdown generation changed before runtime stopped")
            row.state = ProviderInstanceState.STOPPING
        return context.generation

    async def mark_powered_off(
        self,
        instance_uuid: str,
        *,
        expected_generation: int,
    ) -> bool:
        async with self.database.transaction() as session:
            row = await self._instance_row(session, instance_uuid, lock=True)
            context = _shutdown_context(row)
            if context is None or context.generation != expected_generation:
                return False
            site = await session.get(SiteRow, context.site_id, with_for_update=True)
            if (
                row.state is not ProviderInstanceState.STOPPING
                or not row.durable_state_confirmed
                or not _context_acknowledged(row, context, "durability")
                or site is None
                or site.state is not SiteState.DRAINING
            ):
                return False
            row.state = ProviderInstanceState.STOPPED
            row.idle_since = None
            row.metadata_json = {
                **row.metadata_json,
                "power_cycle": int(row.metadata_json.get("power_cycle", 0)) + 1,
            }
            _clear_shutdown_coordinates(row)
            site.state = SiteState.STOPPED
            site.drain_ready = False
            site.durability_ready = False
            return True

    async def cancel_draining(self, instance_uuid: str) -> None:
        context = await self._current_shutdown_context(instance_uuid)
        if context is None:
            raise RuntimeError("AutoDL shutdown has no current drain generation")
        delivered = await self.connections.send(
            context.site_id,
            SchedulerEnvelope(
                site_id=context.site_id,
                kind="cancel_drain",
                payload={
                    "shutdown_generation": context.generation,
                    "command_nonce": context.drain_nonce,
                },
            ),
            expected_generation=context.connection_generation,
        )
        if not delivered:
            raise RuntimeError("AutoDL cancel-drain command could not be delivered")
        async with self.database.transaction() as session:
            row = await self._instance_row(session, instance_uuid, lock=True)
            if not _context_matches(row, context):
                raise RuntimeError("AutoDL shutdown generation changed during cancellation")
            row.state = ProviderInstanceState.RUNNING
            row.idle_since = None
            row.durable_state_confirmed = False
            _clear_shutdown_coordinates(row)
            site = await session.get(SiteRow, context.site_id, with_for_update=True)
            if site is not None:
                site.state = SiteState.READY
                site.drain_ready = False
                site.durability_ready = False

    async def power_off_retry_authorized(self, instance_uuid: str) -> bool:
        """Authorize replay only for the exact stopped, durable drain generation."""

        async with self.database.session() as session:
            row = await self._instance_row(session, instance_uuid)
            context = _shutdown_context(row)
            if context is None or not await self._power_off_context_authorized(
                session,
                row,
                context,
            ):
                return False
        # stop_runtime required the exact connection to disappear. A replacement
        # connection means new activity may exist and must invalidate the retry.
        return await self.connections.identity(context.site_id) is None

    async def run_power_off_if_authorized(
        self,
        instance_uuid: str,
        *,
        expected_generation: int,
        action: Callable[[], Awaitable[None]],
    ) -> bool:
        """Run a drained power-off while excluding replacement registration."""

        async with self.database.session() as session:
            row = await self._instance_row(session, instance_uuid)
            expected_context = _shutdown_context(row)
            if (
                expected_context is None
                or expected_context.generation != expected_generation
                or not await self._power_off_context_authorized(
                    session,
                    row,
                    expected_context,
                )
            ):
                return False

        async def guarded_action() -> bool:
            # Revalidate after acquiring the per-site connection lock.  This
            # closes the check/use window in which a replacement agent could
            # otherwise register and begin new work before the provider call.
            async with self.database.session() as session:
                current = await self._instance_row(session, instance_uuid)
                if not _context_matches(
                    current, expected_context
                ) or not await self._power_off_context_authorized(
                    session,
                    current,
                    expected_context,
                ):
                    return False
            await action()
            return True

        disconnected, executed = await self.connections.run_if_disconnected(
            expected_context.site_id,
            guarded_action,
        )
        return disconnected and executed is True

    async def _power_off_context_authorized(
        self,
        session: AsyncSession,
        row: ProviderInstanceRow,
        context: _ShutdownContext,
    ) -> bool:
        if (
            row.state is not ProviderInstanceState.STOPPING
            or not row.durable_state_confirmed
            or not _context_acknowledged(row, context, "durability")
        ):
            return False
        site = await session.get(SiteRow, context.site_id)
        return site is not None and site.state is SiteState.DRAINING

    async def record_idle_observation(
        self,
        instance_uuid: str,
        activity: IdleActivity,
        *,
        now: datetime,
    ) -> datetime | None:
        async with self.database.transaction() as session:
            row = await self._instance_row(session, instance_uuid, lock=True)
            if not activity.is_idle:
                row.idle_since = None
                return None
            if row.idle_since is None:
                row.idle_since = now
            return _as_utc(row.idle_since)

    async def associate_heartbeat(
        self,
        heartbeat: SiteHeartbeat,
        *,
        connection_generation: str | None = None,
    ) -> None:
        """Bind an AutoDL site heartbeat to its durable provider record."""

        if heartbeat.site_kind is not SiteKind.AUTODL:
            return
        async with self.database.transaction() as session:
            statement = select(ProviderInstanceRow).where(
                ProviderInstanceRow.provider == "autodl",
                ProviderInstanceRow.state.not_in(
                    [ProviderInstanceState.RELEASED, ProviderInstanceState.STOPPED]
                ),
            )
            if heartbeat.provider_instance_id:
                statement = statement.where(
                    ProviderInstanceRow.external_id == heartbeat.provider_instance_id
                )
            else:
                statement = statement.where(ProviderInstanceRow.site_id == heartbeat.site_id)
            rows = list(await session.scalars(statement.with_for_update()))
            if len(rows) != 1:
                site = await session.get(SiteRow, heartbeat.site_id, with_for_update=True)
                if site is not None:
                    site.state = SiteState.ERROR
                return
            row = rows[0]
            site = await session.get(SiteRow, heartbeat.site_id, with_for_update=True)
            expected_gpu_count = _requested_gpu_count(row)
            if (
                expected_gpu_count is None
                or len(heartbeat.gpus) != expected_gpu_count
                or (row.site_id is not None and row.site_id != heartbeat.site_id)
            ):
                # The provider UUID and physical inventory together form the
                # site identity.  A partial/wrong-shape heartbeat must never
                # make an atomically requested multi-GPU instance schedulable.
                if site is not None:
                    site.state = SiteState.ERROR
                return
            row.site_id = heartbeat.site_id
            heartbeat_region = heartbeat.region
            row.region = heartbeat_region or row.region
            row.metadata_json = {
                **row.metadata_json,
                "startup_heartbeat_seen": True,
                "startup_deadline": None,
            }
            if site is not None:
                # The managed provider record is authoritative for identity
                # and region when a bootstrap heartbeat omitted its optional
                # hints.  This also repairs the nulls written by the generic
                # heartbeat upsert immediately before provider association.
                site.provider_instance_id = row.external_id
                site.region = row.region
            bound_identity = (
                row.shutdown_agent_session_id,
                row.shutdown_connection_generation,
            )
            observed_identity = (
                heartbeat.agent_session_id,
                connection_generation,
            )
            if (
                row.state
                in {
                    ProviderInstanceState.DRAINING,
                    ProviderInstanceState.STOPPING,
                }
                and bound_identity != observed_identity
            ):
                # A replacement agent cannot inherit acknowledgements from the
                # displaced process.  Abort the old shutdown; the next idle
                # pass will start a fresh generation against this connection.
                row.state = ProviderInstanceState.RUNNING
                row.idle_since = None
                row.durable_state_confirmed = False
                _clear_shutdown_coordinates(row)
                if site is not None:
                    site.state = SiteState.READY
                    site.drain_ready = False
                    site.durability_ready = False
            elif row.state in {
                ProviderInstanceState.DRAINING,
                ProviderInstanceState.STOPPING,
            }:
                if site is not None:
                    site.state = SiteState.DRAINING
                    site.drain_ready = row.drain_ack_nonce == row.drain_command_nonce
                    site.durability_ready = (
                        row.durability_command_nonce is not None
                        and row.durability_ack_nonce == row.durability_command_nonce
                    )
            else:
                row.state = ProviderInstanceState.RUNNING
                row.durable_state_confirmed = False
                _clear_shutdown_coordinates(row)

    async def note_agent_event(
        self,
        envelope: AgentEnvelope,
        *,
        connection_generation: str | None = None,
    ) -> None:
        if envelope.kind not in {"drain_ready", "durability_ready"}:
            return
        coordinates = _event_shutdown_coordinates(envelope.payload)
        if coordinates is None or connection_generation is None:
            return
        generation, nonce = coordinates
        current_identity = await self.connections.identity(envelope.site_id)
        if current_identity != (envelope.agent_session_id, connection_generation):
            return
        async with self.database.transaction() as session:
            site = await session.get(SiteRow, envelope.site_id, with_for_update=True)
            if site is None or site.kind is not SiteKind.AUTODL:
                return
            if (
                site.agent_session_id != envelope.agent_session_id
                or site.agent_connection_generation != connection_generation
            ):
                return
            rows = list(
                await session.scalars(
                    select(ProviderInstanceRow)
                    .where(
                        ProviderInstanceRow.provider == "autodl",
                        ProviderInstanceRow.site_id == envelope.site_id,
                        ProviderInstanceRow.state.in_(
                            [
                                ProviderInstanceState.DRAINING,
                                ProviderInstanceState.STOPPING,
                            ]
                        ),
                    )
                    .with_for_update()
                )
            )
            if len(rows) != 1:
                return
            row = rows[0]
            context = _shutdown_context(row)
            if (
                context is None
                or context.generation != generation
                or context.agent_session_id != envelope.agent_session_id
                or context.connection_generation != connection_generation
            ):
                return
            if envelope.kind == "drain_ready":
                if not hmac.compare_digest(context.drain_nonce, nonce):
                    return
                row.drain_ack_nonce = nonce
                site.drain_ready = True
                site.state = SiteState.DRAINING
            else:
                if (
                    context.durability_nonce is None
                    or not _context_acknowledged(row, context, "drain")
                    or not hmac.compare_digest(context.durability_nonce, nonce)
                ):
                    return
                row.durability_ack_nonce = nonce
                site.durability_ready = True

    async def _set_draining(self, instance_uuid: str) -> _ShutdownContext:
        async with self.database.transaction() as session:
            row = await self._instance_row(session, instance_uuid, lock=True)
            if not row.site_id:
                raise RuntimeError("AutoDL provider instance is not associated with a site")
            site = await session.get(SiteRow, row.site_id, with_for_update=True)
            if site is None:
                raise RuntimeError("AutoDL site record does not exist")
            identity = await self.connections.identity(site.id)
            if identity is None or identity != (
                site.agent_session_id,
                site.agent_connection_generation,
            ):
                raise RuntimeError("AutoDL site does not have current authenticated telemetry")
            agent_session_id, connection_generation = identity
            row.state = ProviderInstanceState.DRAINING
            row.durable_state_confirmed = False
            row.shutdown_generation += 1
            row.shutdown_agent_session_id = agent_session_id
            row.shutdown_connection_generation = connection_generation
            row.drain_command_nonce = secrets.token_urlsafe(24)
            row.drain_ack_nonce = None
            row.durability_command_nonce = None
            row.durability_ack_nonce = None
            site.state = SiteState.DRAINING
            site.drain_ready = False
            site.durability_ready = False
            context = _shutdown_context(row)
            assert context is not None
            return context

    async def _begin_durability(
        self,
        instance_uuid: str,
    ) -> _ShutdownContext | None:
        async with self.database.transaction() as session:
            row = await self._instance_row(session, instance_uuid, lock=True)
            context = _shutdown_context(row)
            if context is None or not _context_acknowledged(row, context, "drain"):
                return None
            if await self.connections.identity(context.site_id) != (
                context.agent_session_id,
                context.connection_generation,
            ):
                return None
            row.durability_command_nonce = secrets.token_urlsafe(24)
            row.durability_ack_nonce = None
            row.durable_state_confirmed = False
            site = await session.get(SiteRow, context.site_id, with_for_update=True)
            if site is None:
                return None
            site.durability_ready = False
            updated = _shutdown_context(row)
            assert updated is not None
            return updated

    async def _current_shutdown_context(
        self,
        instance_uuid: str,
        *,
        require_durability: bool = False,
    ) -> _ShutdownContext | None:
        async with self.database.session() as session:
            row = await self._instance_row(session, instance_uuid)
            context = _shutdown_context(row)
            if context is None:
                return None
            if require_durability and not _context_acknowledged(
                row,
                context,
                "durability",
            ):
                return None
            return context

    async def _site_id(self, instance_uuid: str) -> str:
        async with self.database.session() as session:
            row = await self._instance_row(session, instance_uuid)
            if not row.site_id:
                raise RuntimeError("AutoDL provider instance is not associated with a site")
            return row.site_id

    async def _shutdown_acknowledged(self, instance_uuid: str, *, stage: str) -> bool:
        async with self.database.session() as session:
            row = await self._instance_row(session, instance_uuid)
            context = _shutdown_context(row)
            return bool(context is not None and _context_acknowledged(row, context, stage))

    async def _wait_for_shutdown_ack(
        self,
        context: _ShutdownContext,
        *,
        stage: str,
    ) -> bool:
        deadline = time.monotonic() + self.acknowledgement_timeout_seconds
        while time.monotonic() < deadline:
            async with self.database.session() as session:
                row = await session.scalar(
                    select(ProviderInstanceRow).where(
                        ProviderInstanceRow.provider == "autodl",
                        ProviderInstanceRow.external_id == context.instance_uuid,
                    )
                )
                if (
                    row is not None
                    and _context_matches(row, context)
                    and _context_acknowledged(row, context, stage)
                ):
                    return True
            await asyncio.sleep(self.poll_interval_seconds)
        return False

    async def _wait_for_disconnect(self, context: _ShutdownContext) -> bool:
        deadline = time.monotonic() + self.acknowledgement_timeout_seconds
        while time.monotonic() < deadline:
            identity = await self.connections.identity(context.site_id)
            if identity is None:
                return True
            if identity != (
                context.agent_session_id,
                context.connection_generation,
            ):
                return False
            await asyncio.sleep(self.poll_interval_seconds)
        return False

    @staticmethod
    async def _instance_row(
        session: AsyncSession,
        instance_uuid: str,
        *,
        lock: bool = False,
    ) -> ProviderInstanceRow:
        statement = select(ProviderInstanceRow).where(
            ProviderInstanceRow.provider == "autodl",
            ProviderInstanceRow.external_id == instance_uuid,
        )
        if lock:
            statement = statement.with_for_update()
        row = await session.scalar(statement)
        if row is None:
            raise RuntimeError("AutoDL provider instance is not tracked")
        return row


def _shutdown_context(row: ProviderInstanceRow) -> _ShutdownContext | None:
    if not (
        row.external_id
        and row.site_id
        and row.shutdown_generation > 0
        and row.shutdown_agent_session_id
        and row.shutdown_connection_generation
        and row.drain_command_nonce
    ):
        return None
    return _ShutdownContext(
        instance_uuid=row.external_id,
        site_id=row.site_id,
        generation=row.shutdown_generation,
        agent_session_id=row.shutdown_agent_session_id,
        connection_generation=row.shutdown_connection_generation,
        drain_nonce=row.drain_command_nonce,
        durability_nonce=row.durability_command_nonce,
    )


def _context_matches(row: ProviderInstanceRow, context: _ShutdownContext) -> bool:
    current = _shutdown_context(row)
    return current == context


def _context_acknowledged(
    row: ProviderInstanceRow,
    context: _ShutdownContext,
    stage: str,
) -> bool:
    if not _context_matches(row, context):
        return False
    if stage == "drain":
        return bool(
            row.drain_ack_nonce and hmac.compare_digest(row.drain_ack_nonce, context.drain_nonce)
        )
    if stage == "durability":
        return bool(
            context.durability_nonce
            and row.durability_ack_nonce
            and hmac.compare_digest(
                row.durability_ack_nonce,
                context.durability_nonce,
            )
        )
    raise ValueError(f"unsupported shutdown stage: {stage}")


def _clear_shutdown_coordinates(row: ProviderInstanceRow) -> None:
    # Keep the monotonic generation across power cycles and restarts, while
    # deleting every credential-like correlation value from a completed or
    # canceled shutdown.
    row.shutdown_agent_session_id = None
    row.shutdown_connection_generation = None
    row.drain_command_nonce = None
    row.drain_ack_nonce = None
    row.durability_command_nonce = None
    row.durability_ack_nonce = None


def _event_shutdown_coordinates(payload: dict[str, Any]) -> tuple[int, str] | None:
    generation = payload.get("shutdown_generation")
    nonce = payload.get("command_nonce")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
        or not isinstance(nonce, str)
        or not 16 <= len(nonce) <= 120
    ):
        return None
    return generation, nonce


async def _has_rows(session: Any, model: Any, *criteria: Any) -> bool:
    count = await session.scalar(select(func.count()).select_from(model).where(*criteria))
    return bool(count)


def _unknown_activity() -> IdleActivity:
    # One asserted signal is enough to prevent an unsafe shutdown when the
    # scheduler cannot establish all five negatives.
    return IdleActivity(staging_operation_active=True)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _safe_instance_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "release_requested_at",
        "release_force",
        "management",
        "manual_error",
        "needs_attention",
        "instance_name",
        "provider_status",
        "provider_sub_status",
        "provider_region_sign",
        "provider_observed",
        "provision_action",
        "demand_job_id",
        "power_cycle",
        "requested_gpu_count",
        "observed_gpu_count",
        "shape_mismatch",
        "startup_generation",
        "startup_started_at",
        "startup_deadline",
        "startup_heartbeat_seen",
        "startup_powered_off_at",
    }
    return {key: value for key, value in metadata.items() if key in allowed}


def _requested_gpu_count(row: ProviderInstanceRow) -> int | None:
    # Rows created before multi-GPU support were unconditionally one-GPU.
    if "requested_gpu_count" not in row.metadata_json:
        return 1
    value = row.metadata_json.get("requested_gpu_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


class AutoDLStatusMixin:
    """Shared observation and audit views for manual registrations."""

    async def observe_agent_event(
        self,
        envelope: AgentEnvelope,
        *,
        connection_generation: str | None = None,
    ) -> None:
        if envelope.kind in {"heartbeat", "inventory", "capability_report"}:
            heartbeat = SiteHeartbeat.model_validate(envelope.payload)
            await self.bridge.associate_heartbeat(
                heartbeat,
                connection_generation=connection_generation,
            )
        else:
            await self.bridge.note_agent_event(
                envelope,
                connection_generation=connection_generation,
            )

    async def instance_summaries(self) -> list[dict[str, Any]]:
        async with self.database.session() as session:
            rows = list(
                await session.scalars(
                    select(ProviderInstanceRow)
                    .where(ProviderInstanceRow.provider == "autodl")
                    .where(
                        ProviderInstanceRow.state.not_in(
                            [ProviderInstanceState.RELEASED, ProviderInstanceState.STOPPED]
                        )
                    )
                    .order_by(ProviderInstanceRow.created_at.desc())
                )
            )
            return [
                {
                    "id": row.id,
                    "external_id": row.external_id,
                    "site_id": row.site_id,
                    "state": row.state.value,
                    "region": row.region,
                    "gpu_spec": row.gpu_spec,
                    "gpu_count": _requested_gpu_count(row),
                    "image_uuid": row.image_uuid,
                    "durable_state_confirmed": row.durable_state_confirmed,
                    "shutdown_generation": row.shutdown_generation,
                    "idle_since": row.idle_since,
                    "metadata": _safe_instance_metadata(row.metadata_json),
                    "updated_at": row.updated_at,
                }
                for row in rows
            ]

    async def operation_summaries(self) -> list[dict[str, Any]]:
        async with self.database.session() as session:
            rows = list(
                await session.scalars(
                    select(ProviderOperationRow)
                    .where(ProviderOperationRow.provider == "autodl")
                    .order_by(ProviderOperationRow.created_at.desc())
                    .limit(100)
                )
            )
            return [
                {
                    "id": row.id,
                    "action": row.action.value,
                    "state": row.state.value,
                    "external_instance_id": row.external_instance_id,
                    "result_external_instance_id": row.result_external_instance_id,
                    "external_request_id": row.external_request_id,
                    "error_code": row.error_code,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
                for row in rows
            ]


def build_autodl_orchestrator(
    settings: SchedulerSettings, database: Database, connections: AgentConnectionRegistry
) -> ProviderOrchestrator | None:
    if not settings.autodl.manual_ssh:
        return None
    from topodrome.scheduler.manual_autodl import ManualAutoDLController

    return ManualAutoDLController(settings, database, connections)
