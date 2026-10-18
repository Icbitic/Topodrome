"""Admission fencing and fresh, generation-bound idle checks for deployments."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import func, select

from topodrome.db.models import (
    ArtifactRow,
    AttemptRow,
    InputStageRow,
    JobRow,
    KernelSessionRow,
    LeaseGPUClaimRow,
    LeaseRow,
    ProviderInstanceRow,
    ProviderOperationRow,
    SiteRow,
)
from topodrome.db.session import CURRENT_SCHEMA_REVISION
from topodrome.deployment.upgrade_store import UpgradeStore
from topodrome.domain.errors import ConflictError
from topodrome.domain.specs import utc_now
from topodrome.domain.upgrades import (
    UpgradeClaim,
    UpgradePauseRequest,
    UpgradeRecord,
    UpgradeRequest,
)

if TYPE_CHECKING:
    from topodrome.scheduler.service import SchedulerService

_ATTEMPTS = (
    "offered",
    "accepted",
    "starting",
    "running",
    "preempt_requested",
    "checkpointing",
    "committing",
)
_LEASES = ("offered", "accepted", "active", "releasing")
_KERNELS = ("starting", "running", "idle", "stopping")
_ACTIVITY = (
    "job_running",
    "assignment_queued_locally",
    "interactive_kernel_active",
    "artifact_flush_pending",
    "staging_operation_active",
)


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class UpgradeController:
    def __init__(self, service: SchedulerService):
        self.service = service
        self.store = UpgradeStore(service.settings.global_root)
        self.instance_id = uuid4().hex

    async def schedule(self, request: UpgradeRequest) -> dict[str, object]:
        # Freezing runs off-loop; lease renewal and terminal ACKs stay responsive.
        # Only publication of the journal/fence is ordered with admission.
        async with self.service._dispatch_reload_lock:
            record = await asyncio.to_thread(self.store.schedule, request)
            return record.summary()

    async def blockers(
        self, record: UpgradeRecord, *, after: bool = False
    ) -> tuple[dict[str, int], dict[str, str]]:
        service = self.service
        identities = await service.connections.identities()
        global_scope = record.plan.scheduler
        requested = set(record.plan.sites)
        watched = requested | (set(identities) if global_scope else set())
        if after and global_scope:
            watched |= set(record.sessions_before)
        counts: dict[str, int] = {}
        counts["database_not_ready"] = int(
            not await service.database.ready(
                expected_schema_revision=(
                    None if service.settings.insecure_development_mode else CURRENT_SCHEMA_REVISION
                )
            )
        )
        sessions: dict[str, str] = {}
        cutoff = utc_now() - timedelta(seconds=service.settings.heartbeat_ttl_seconds)
        async with service.database.session() as session:
            counts["jobs"] = (
                int(
                    await session.scalar(
                        select(func.count())
                        .select_from(JobRow)
                        .where(JobRow.state.in_(("assigning", "running", "canceling")))
                    )
                    or 0
                )
                if global_scope
                else 0
            )
            for model, label, states in (
                (AttemptRow, "attempts", _ATTEMPTS),
                (LeaseRow, "leases", _LEASES),
                (KernelSessionRow, "kernels", _KERNELS),
            ):
                query = select(func.count()).select_from(model).where(model.state.in_(states))
                if not global_scope:
                    query = query.where(model.site_id.in_(requested))
                counts[label] = int(await session.scalar(query) or 0)
            query = (
                select(func.count())
                .select_from(LeaseGPUClaimRow)
                .where(LeaseGPUClaimRow.active.is_(True))
            )
            if not global_scope:
                query = query.where(LeaseGPUClaimRow.site_id.in_(requested))
            counts["physical_claims"] = int(await session.scalar(query) or 0)
            query = (
                select(func.count())
                .select_from(InputStageRow)
                .where(
                    InputStageRow.state == "downloading",
                    InputStageRow.expires_at > utc_now(),
                )
            )
            if not global_scope:
                query = query.where(InputStageRow.site_id.in_(requested))
            counts["input_staging"] = int(await session.scalar(query) or 0)
            attempts = select(AttemptRow.id).where(AttemptRow.state.in_(_ATTEMPTS))
            if not global_scope:
                attempts = attempts.where(AttemptRow.site_id.in_(requested))
            counts["pending_artifacts"] = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ArtifactRow)
                    .where(
                        ArtifactRow.attempt_id.in_(attempts),
                        ArtifactRow.state.in_(("staging", "flushing")),
                    )
                )
                or 0
            )
            counts["unhealthy_sites"] = 0
            counts["site_activity"] = 0
            counts["old_agent_sessions"] = 0
            counts["heartbeat_before_last_work"] = 0
            for site_id in watched:
                site = await session.get(SiteRow, site_id)
                identity = identities.get(site_id)
                if (
                    site is None
                    or identity is None
                    or site.heartbeat_at is None
                    or _aware(site.heartbeat_at) < cutoff
                    or identity != (site.agent_session_id, site.agent_connection_generation)
                    or site.state.value != "ready"
                    or not site.storage_available
                ):
                    counts["unhealthy_sites"] += 1
                    continue
                sessions[site_id] = identity[0]
                latest_changes = []
                for latest_model in (AttemptRow, LeaseRow, InputStageRow):
                    latest = await session.scalar(
                        select(func.max(latest_model.updated_at)).where(
                            latest_model.site_id == site_id
                        )
                    )
                    if latest is not None:
                        latest_changes.append(_aware(latest))
                # A sample taken before a fast job finished cannot prove its
                # failure-log outbox or detached finalizer has subsequently drained.
                if latest_changes and _aware(site.heartbeat_at) <= max(latest_changes):
                    counts["heartbeat_before_last_work"] += 1
                counts["site_activity"] += int(any(getattr(site, key) for key in _ACTIVITY))
                if after and record.claimed_at is not None:
                    if _aware(site.heartbeat_at) <= record.claimed_at:
                        counts["unhealthy_sites"] += 1
                    if (
                        site_id in requested
                        and record.plan.require_new_agent_session
                        and identity[0] == record.sessions_before.get(site_id)
                    ):
                        counts["old_agent_sessions"] += 1
            # Provider reconciliation must not race a site update with shutdown.
            provider_ids = select(ProviderInstanceRow.id)
            if not global_scope:
                provider_ids = provider_ids.where(ProviderInstanceRow.site_id.in_(requested))
            counts["provider_operations"] = int(
                await session.scalar(
                    select(func.count())
                    .select_from(ProviderOperationRow)
                    .where(
                        ProviderOperationRow.instance_id.in_(provider_ids),
                        ProviderOperationRow.state.in_(("intent", "calling", "reconciling")),
                    )
                )
                or 0
            )
            query = (
                select(func.count())
                .select_from(ProviderInstanceRow)
                .where(
                    ProviderInstanceRow.state.in_(
                        (
                            "provisioning",
                            "starting",
                            "draining",
                            "stopping",
                            "durability_confirmed",
                            "releasing",
                        )
                    ),
                )
            )
            if not global_scope:
                query = query.where(ProviderInstanceRow.site_id.in_(requested))
            counts["provider_transitions"] = int(await session.scalar(query) or 0)
        provider = service.provider_controller
        tasks = getattr(provider, "tasks", {})
        counts["provider_tasks"] = sum(
            int(not task.done())
            for site_id, task in tasks.items()
            if global_scope or site_id in requested
        )
        bootstrap = service.jupyter_bootstrap_controller
        counts["bootstrap_tasks"] = int(
            bool(
                bootstrap is not None
                and bootstrap.attempt_in_flight
                and (global_scope or bootstrap.settings.site_id in requested)
            )
        )
        if after and record.plan.scheduler:
            counts["old_scheduler"] = int(self.instance_id == record.scheduler_before)
        return counts, sessions

    async def claim(self, upgrade_id: str, request: UpgradeClaim) -> dict[str, object]:
        async with self.service._dispatch_reload_lock:
            record = self.store.get(upgrade_id)
            if record.state == "applying" and record.runner_id == request.runner_id:
                return {**record.summary(), "claimed": True}
            if record.state != "waiting":
                raise ConflictError("upgrade is not waiting for idle")
            counts, sessions = await self.blockers(record)
            if any(counts.values()):
                record = self.store.change(upgrade_id, expected={"waiting"}, blockers=counts)
                return {**record.summary(), "claimed": False}
            # For --pause none, this is the atomic short maintenance fence. No
            # late offer can appear between this check and the external restart.
            record = self.store.change(
                upgrade_id,
                expected={"waiting"},
                state="applying",
                mutation_started=True,
                runner_id=request.runner_id,
                claimed_at=utc_now(),
                sessions_before=sessions,
                scheduler_before=self.instance_id,
                blockers={},
                error_type=None,
            )
            return {**record.summary(), "claimed": True}

    async def complete(self, upgrade_id: str, request: UpgradeClaim) -> dict[str, object]:
        async with self.service._dispatch_reload_lock:
            record = self.store.get(upgrade_id)
            if (
                record.state in {"succeeded", "rolled_back"}
                and record.runner_id == request.runner_id
            ):
                return {**record.summary(), "completed": True}
            if record.runner_id != request.runner_id or record.state not in {
                "verifying",
                "verifying_rollback",
            }:
                raise ConflictError("upgrade does not have this verification owner")
            counts, _ = await self.blockers(record, after=True)
            if any(counts.values()):
                record = self.store.change(upgrade_id, expected={record.state}, blockers=counts)
                return {**record.summary(), "completed": False}
            state = "rolled_back" if record.state == "verifying_rollback" else "succeeded"
            record = self.store.change(
                upgrade_id, expected={record.state}, state=state, blockers={}, error_type=None
            )
            return {**record.summary(), "completed": True}

    async def cancel(self, upgrade_id: str) -> dict[str, object]:
        async with self.service._dispatch_reload_lock:
            record = self.store.get(upgrade_id)
            if record.state == "canceled":
                return record.summary()
            if record.mutation_started:
                raise ConflictError("deployment already began; verify or roll back instead")
            return self.store.change(
                upgrade_id,
                expected={"queued", "preparing", "waiting"},
                state="canceled",
                blockers={},
            ).summary()

    async def recover(self, upgrade_id: str, action: str) -> dict[str, object]:
        async with self.service._dispatch_reload_lock:
            record = self.store.get(upgrade_id)
            if action == "rollback" and not record.plan.rollback:
                raise ConflictError("this release has no declared rollback")
            return self.store.change(
                upgrade_id,
                expected={"needs_attention"},
                state="rolling_back" if action == "rollback" else "verifying",
                runner_id=None,
                error_type=None,
            ).summary()

    async def pause(self, upgrade_id: str, request: UpgradePauseRequest) -> dict[str, object]:
        async with self.service._dispatch_reload_lock:
            record = self.store.get(upgrade_id)
            if record.mutation_started or record.state not in {"queued", "preparing", "waiting"}:
                raise ConflictError("dispatch scope cannot change after deployment begins")
            if record.pause == request.pause:
                return record.summary()
            if record.pause != request.expected_pause:
                raise ConflictError("upgrade pause scope changed concurrently")
            return self.store.change(
                upgrade_id, expected={record.state}, pause=request.pause
            ).summary()
