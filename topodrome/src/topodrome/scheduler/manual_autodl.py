"""Manually supplied AutoDL capacity; no provider API or automatic rentals."""

from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select

from topodrome.autodl_identity import manual_token, valid_manual_site
from topodrome.db.models import (
    GPUReservationRow,
    GPURow,
    KernelSessionRow,
    LeaseRow,
    ProviderInstanceRow,
    ProviderOperationRow,
    SiteRow,
)
from topodrome.db.repositories import SchedulerRepository
from topodrome.deployment.manual_autodl import (
    ManualRegistration,
    ShutdownPreflightBusy,
    inspect,
    install,
    private_read,
    private_write,
    shutdown,
)
from topodrome.domain.enums import (
    GPUReservationState,
    KernelState,
    LeaseState,
    ProviderOperationState,
    SiteState,
)
from topodrome.domain.enums import ProviderInstanceState as State
from topodrome.domain.specs import GPUReservationRequest, new_id, utc_now
from topodrome.providers.autodl.lifecycle import OperationKind, ProviderOperationIntent
from topodrome.scheduler.autodl import (
    AutoDLReconcileResult,
    AutoDLStatusMixin,
    SchedulerAutoDLBridge,
)
from topodrome.scheduler.provider_controller import SQLProviderOperationStore

log = structlog.get_logger(__name__)


class ManualAutoDLController(AutoDLStatusMixin):
    def __init__(self, settings, database, connections):
        self.settings, self.database, self.connections = settings, database, connections
        self.root = settings.autodl.manual_state_dir
        self.bundle = settings.autodl.manual_bootstrap_dir
        if self.root is None or self.bundle is None:
            raise ValueError("manual AutoDL requires private state and bootstrap directories")
        if self.root == settings.global_root or settings.global_root in self.root.parents:
            raise ValueError("manual AutoDL credentials must be outside global storage")
        self.bridge = SchedulerAutoDLBridge(
            database,
            connections,
            heartbeat_ttl_seconds=settings.heartbeat_ttl_seconds,
            acknowledgement_timeout_seconds=settings.autodl.drain_timeout_seconds,
        )
        self.operations = SQLProviderOperationStore(database)
        self.tasks: dict[str, asyncio.Task] = {}
        self.maintenance_excluded = lambda _site_id: False
        self.lock = asyncio.Lock()

    def enrollment_token(self, site_id):
        if not valid_manual_site(site_id):
            return None
        try:
            record = private_read(self.root / (site_id + ".json"))
        except FileNotFoundError:
            return None
        if record.get("disabled"):
            return None
        base = self.settings.agent_tokens.get(self.settings.autodl.site_id)
        return manual_token(base.get_secret_value(), site_id) if base else None

    def control_status(self):
        return {
            "configured": True,
            "enabled": False,
            "mode": "manual_ssh",
            "idle_timeout_seconds": self.settings.autodl.idle_timeout_seconds,
        }

    async def register(self, request: ManualRegistration):
        async with self.lock:
            record = await inspect(request)
            site_id = record["site_id"]
            if self.maintenance_excluded(site_id):
                raise ValueError("site installation is fenced for an upgrade")
            path = self.root / (site_id + ".json")
            if path.exists():
                previous = private_read(path)
                if previous["host_key"] != record["host_key"]:
                    raise ValueError("SSH host key changed; registration refused")
            async with self.database.transaction() as session:
                row = await session.scalar(
                    select(ProviderInstanceRow)
                    .where(ProviderInstanceRow.external_id == site_id)
                    .with_for_update()
                )
                if row is not None and row.state not in {
                    State.STOPPED,
                    State.STARTING_FAILED,
                    State.RELEASED,
                }:
                    return {"site_id": site_id, "state": row.state.value, "existing": True}
                record["identity"] = secrets.token_hex(32)
                private_write(path, record)
                if row is None:
                    row = ProviderInstanceRow(
                        id=new_id("instance"),
                        provider="autodl",
                        external_id=site_id,
                        site_id=site_id,
                        state=State.STARTING,
                        gpu_spec="manual",
                        image_uuid="manual",
                    )
                    session.add(row)
                else:
                    await self._clear_release_reservations(session, row)
                row.state = State.STARTING
                row.metadata_json = {
                    "management": "manual_ssh",
                    "requested_gpu_count": len(record["gpus"]),
                    "environment": {k: record[k] for k in ("architecture", "libc", "gpus")},
                    "registration_at": utc_now().isoformat(),
                }
            self.tasks[site_id] = asyncio.create_task(self._install(record))
            return {
                "site_id": site_id,
                "state": "starting",
                "environment": {k: record[k] for k in ("architecture", "libc", "gpus")},
            }

    async def unregister(self, site_id, *, automatic=False):
        if self.maintenance_excluded(site_id):
            raise ValueError("site removal is fenced for an upgrade")
        completed, result = await self.connections.run_if_disconnected(
            site_id, lambda: self._unregister_disconnected(site_id, automatic=automatic)
        )
        if not completed:
            raise ValueError("instance is still connected; stop it externally first")
        return result

    async def request_release(self, site_id: str, *, force: bool = False):
        """Persist an operator release, ordered with dispatch by the caller.

        Exact GPU reservations reuse the normal admission/preemption protocol.
        They survive scheduler restarts and are canceled when shutdown finishes.
        """
        if self.maintenance_excluded(site_id):
            raise ValueError("site is fenced for an upgrade; cancel or resolve that upgrade first")
        if not valid_manual_site(site_id):
            raise ValueError("expected a manual AutoDL site ID")
        repository = SchedulerRepository()
        async with self.lock:
            async with self.database.transaction() as session:
                row = await session.scalar(
                    select(ProviderInstanceRow)
                    .where(
                        ProviderInstanceRow.provider == "autodl",
                        ProviderInstanceRow.external_id == site_id,
                    )
                    .with_for_update()
                )
                if row is None:
                    raise ValueError("unknown manual AutoDL site")
                if row.state in {State.STOPPED, State.RELEASED}:
                    return {"site_id": site_id, "state": row.state.value}
                if row.state in {State.DRAINING, State.STOPPING}:
                    return {"site_id": site_id, "state": row.state.value}
                if row.state is not State.RUNNING:
                    raise ValueError("release requires a running registered instance")
                metadata = dict(row.metadata_json)
                reservation_ids = metadata.get("release_reservation_ids", [])
                if not reservation_ids:
                    gpus = list(
                        await session.scalars(select(GPURow).where(GPURow.site_id == site_id))
                    )
                    if not gpus:
                        raise ValueError("no registered GPU inventory; wait for the agent")
                    reservation_ids = []
                    borrowed = {}
                    for gpu in gpus:
                        existing = await session.scalar(
                            select(GPUReservationRow)
                            .where(
                                GPUReservationRow.site_id == site_id,
                                GPUReservationRow.gpu_uuid == gpu.uuid,
                                GPUReservationRow.state.in_(
                                    [GPUReservationState.PENDING, GPUReservationState.READY]
                                ),
                                GPUReservationRow.expires_at > utc_now(),
                            )
                            .with_for_update()
                        )
                        if existing is not None:
                            borrowed[existing.id] = existing.force
                            existing.force = existing.force or force
                            reservation_ids.append(existing.id)
                            continue
                        reservation = await repository.create_gpu_reservation(
                            session,
                            GPUReservationRequest(
                                site_id=site_id,
                                gpu_uuid=gpu.uuid,
                                duration_seconds=31536000,
                                reason="operator AutoDL instance release",
                                force=force,
                            ),
                            requested_by="@operator",
                        )
                        reservation_ids.append(reservation.id)
                    metadata["release_borrowed_reservations"] = borrowed
                elif force:
                    for reservation_id in reservation_ids:
                        reservation = await session.get(GPUReservationRow, reservation_id)
                        if reservation is not None:
                            reservation.force = True
                metadata.update(
                    release_requested_at=metadata.get("release_requested_at")
                    or utc_now().isoformat(),
                    release_force=force or bool(metadata.get("release_force")),
                    release_reservation_ids=reservation_ids,
                )
                row.metadata_json = metadata
                return {
                    "site_id": site_id,
                    "state": "release_pending",
                    "force": metadata["release_force"],
                    "reservation_ids": reservation_ids,
                }

    async def _clear_release_reservations(self, session, row):
        repository = SchedulerRepository()
        borrowed = row.metadata_json.get("release_borrowed_reservations", {})
        for reservation_id in row.metadata_json.get("release_reservation_ids", []):
            if reservation_id in borrowed:
                reservation = await session.get(GPUReservationRow, reservation_id)
                if reservation is not None:
                    reservation.force = borrowed[reservation_id]
                continue
            await repository.cancel_gpu_reservation(session, reservation_id)

    async def _unregister_disconnected(self, site_id, *, automatic=False):
        """Forget disconnected capacity; never claim that provider billing stopped."""
        if not valid_manual_site(site_id):
            raise ValueError("expected a manual AutoDL site ID")
        async with self.lock:
            if self.maintenance_excluded(site_id):
                raise ValueError("site removal is fenced for an upgrade")
            task = self.tasks.get(site_id)
            if task and not task.done():
                raise ValueError("instance operation is still in progress")
            if await self.connections.identity(site_id) is not None:
                raise ValueError("instance is still connected; stop it externally first")
            async with self.database.transaction() as session:
                row = await session.scalar(
                    select(ProviderInstanceRow)
                    .where(
                        ProviderInstanceRow.external_id == site_id,
                        ProviderInstanceRow.provider == "autodl",
                    )
                    .with_for_update()
                )
                if row is None:
                    raise ValueError("unknown manual AutoDL site")
                if row.state in {State.STARTING, State.DRAINING, State.STOPPING}:
                    raise ValueError("instance lifecycle operation is unresolved")
                site = await session.get(SiteRow, site_id, with_for_update=True)
                # Recheck the full automatic grace under the registration and
                # row locks: registration/telemetry may have changed since scan.
                if automatic and not self._offline_cleanup_due(row, site, utc_now()):
                    raise ValueError("offline cleanup grace has not elapsed")
                if site and site.heartbeat_at:
                    heartbeat = (
                        site.heartbeat_at.replace(tzinfo=UTC)
                        if site.heartbeat_at.tzinfo is None
                        else site.heartbeat_at
                    )
                    if utc_now() - heartbeat < timedelta(
                        seconds=self.settings.heartbeat_ttl_seconds
                    ):
                        raise ValueError("heartbeat is still fresh; wait before unregistering")
                lease = await session.scalar(
                    select(LeaseRow.id)
                    .where(
                        LeaseRow.site_id == site_id,
                        LeaseRow.state.in_(
                            [
                                LeaseState.OFFERED,
                                LeaseState.ACCEPTED,
                                LeaseState.ACTIVE,
                                LeaseState.RELEASING,
                            ]
                        ),
                    )
                    .limit(1)
                )
                kernel = await session.scalar(
                    select(KernelSessionRow.id)
                    .where(
                        KernelSessionRow.site_id == site_id,
                        KernelSessionRow.state.not_in([KernelState.STOPPED, KernelState.LOST]),
                    )
                    .limit(1)
                )
                if lease or kernel:
                    raise ValueError("active leases or kernels remain; wait for recovery")
                path = self.root / (site_id + ".json")
                record = private_read(path)
                record["disabled"] = True
                private_write(path, record)
                await self._clear_release_reservations(session, row)
                row.state = State.RELEASED
                row.metadata_json = {
                    **row.metadata_json,
                    "unregistered_at": utc_now().isoformat(),
                    "provider_power_state": "unknown",
                }
                if site:
                    site.state = SiteState.STOPPED
            return {"site_id": site_id, "state": "unregistered", "provider_power_state": "unknown"}

    async def _install(self, record):
        try:
            await install(record, self.bundle)
        except Exception as exc:
            # Never persist or log SSH exception bodies, which may contain input.
            async with self.database.transaction() as session:
                row = await session.scalar(
                    select(ProviderInstanceRow)
                    .where(ProviderInstanceRow.external_id == record["site_id"])
                    .with_for_update()
                )
                if row and row.state is State.STARTING:
                    row.state = State.STARTING_FAILED
                    row.metadata_json = {**row.metadata_json, "manual_error": type(exc).__name__}
            log.warning(
                "manual_autodl_install_failed",
                site_id=record["site_id"],
                error_type=type(exc).__name__,
            )

    async def reconcile_once(self):
        async with self.database.session() as session:
            rows = list(
                await session.scalars(
                    select(ProviderInstanceRow).where(
                        ProviderInstanceRow.provider == "autodl",
                        ProviderInstanceRow.state.in_(
                            [
                                State.STARTING,
                                State.RUNNING,
                                State.DRAINING,
                                State.STOPPING,
                                State.STARTING_FAILED,
                                State.ERROR,
                            ]
                        ),
                    )
                )
            )
        for row in rows:
            site_id = row.site_id
            if not site_id or not valid_manual_site(site_id):
                continue
            if self.maintenance_excluded(site_id) and row.state not in {
                State.DRAINING, State.STOPPING,
            }:
                continue
            if site_id in self.tasks and not self.tasks[site_id].done():
                continue
            if row.state is State.STARTING:
                registered = datetime.fromisoformat(row.metadata_json["registration_at"])
                if utc_now() - registered > timedelta(
                    seconds=self.settings.autodl.startup_heartbeat_timeout_seconds
                ):
                    async with self.database.transaction() as session:
                        current = await session.get(
                            ProviderInstanceRow, row.id, with_for_update=True
                        )
                        if current and current.state is State.STARTING:
                            current.state = State.STARTING_FAILED
                            current.metadata_json = {
                                **current.metadata_json,
                                "manual_error": "StartupHeartbeatTimeout",
                                "needs_attention": True,
                            }
                continue
            if row.state is State.STOPPING:
                self.tasks[site_id] = asyncio.create_task(
                    self._resume_stop(site_id, row.shutdown_generation)
                )
                continue
            if row.state is State.DRAINING:
                # A scheduler restart may interrupt the barrier. Re-establish
                # a fresh generation only while the exact agent is connected.
                self.tasks[site_id] = asyncio.create_task(self._drain(site_id))
                continue
            now = utc_now()
            async with self.database.session() as session:
                site = await session.get(SiteRow, site_id)
            if self._offline_cleanup_due(row, site, now):
                try:
                    await self.unregister(site_id, automatic=True)
                except ValueError:
                    # Active leases/kernels, new registrations and reconnects
                    # still protect the instance. Retry on the next scan.
                    pass
                else:
                    continue
            if row.state in {State.STARTING_FAILED, State.ERROR}:
                # Forgetting stale registration is not a shutdown retry and
                # must never replay an ambiguous external operation.
                continue
            activity = await self.bridge.current_activity(site_id)
            since = await self.bridge.record_idle_observation(site_id, activity, now=now)
            if activity.is_idle and (
                row.metadata_json.get("release_requested_at")
                or (
                    since
                    and now - since >= timedelta(seconds=self.settings.autodl.idle_timeout_seconds)
                )
            ):
                if not self.maintenance_excluded(site_id):
                    self.tasks[site_id] = asyncio.create_task(self._drain(site_id))
        return AutoDLReconcileResult(len(rows), 0, 0)

    def _offline_cleanup_due(self, row, site, now):
        def aware(value):
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value

        observations = [
            aware(site.heartbeat_at)
            if site is not None and site.heartbeat_at is not None
            else aware(row.created_at)
        ]
        registered = row.metadata_json.get("registration_at")
        if isinstance(registered, str):
            try:
                observations.append(aware(datetime.fromisoformat(registered)))
            except ValueError:
                pass  # Legacy malformed metadata cannot disable cleanup forever.
        if row.state in {State.STARTING_FAILED, State.ERROR}:
            # Allow recovery after a newly reported failure even when no agent
            # ever connected. Keep that timestamp durable across restarts.
            observations.append(aware(row.updated_at))
        return now - max(observations) >= timedelta(
            seconds=self.settings.autodl.manual_offline_unregister_seconds
        )

    async def _drain(self, site_id):
        try:
            await self.bridge.mark_draining(site_id)
            if not (await self.bridge.current_activity(site_id)).is_idle:
                await self.bridge.cancel_draining(site_id)
                return
            if not (
                await self.bridge.flush_and_verify(site_id)
                and await self.bridge.persist_regional_state(site_id)
                and await self.bridge.is_durable(site_id)
            ):
                await self.bridge.cancel_draining(site_id)
                return
            generation = await self.bridge.stop_runtime(site_id)
            await self._resume_stop(site_id, generation)
        except Exception as exc:
            await self._stop_error(site_id, exc)

    async def _resume_stop(self, site_id, generation):
        try:
            intent = ProviderOperationIntent(
                operation_id=new_id("provider_op"),
                idempotency_key=f"manual-ssh-stop:{site_id}:{generation}",
                kind=OperationKind.POWER_OFF,
                instance_uuid=site_id,
                request={"transport": "ssh", "generation": generation},
            )
            claim = await self.operations.claim(intent)

            async def action():
                operation_id = claim.record.intent.operation_id
                async with self.database.session() as session:
                    op = await session.get(ProviderOperationRow, operation_id)
                    state = op.state
                if state is not ProviderOperationState.APPLIED:
                    if state is not ProviderOperationState.INTENT:
                        raise RuntimeError("ShutdownOutcomeUnknown")

                    async def before_submit():
                        if not await self.operations.claim_intent_for_call(operation_id):
                            raise RuntimeError("shutdown intent already claimed")

                    record = private_read(self.root / (site_id + ".json"))
                    await shutdown(record, before_submit=before_submit)
                    await self.operations.mark_succeeded(
                        operation_id, provider_request_id=None, result_instance_uuid=site_id
                    )
                # Commit under the same registration lock as the SSH command.
                if not await self.bridge.mark_powered_off(site_id, expected_generation=generation):
                    raise RuntimeError("shutdown generation changed")
                async with self.database.transaction() as session:
                    row = await session.scalar(
                        select(ProviderInstanceRow)
                        .where(ProviderInstanceRow.external_id == site_id)
                        .with_for_update()
                    )
                    await self._clear_release_reservations(session, row)
                record = private_read(self.root / (site_id + ".json"))
                record["disabled"] = True
                private_write(self.root / (site_id + ".json"), record)

            called = await self.bridge.run_power_off_if_authorized(
                site_id, expected_generation=generation, action=action
            )
            if not called:
                raise RuntimeError("shutdown authorization changed")
        except Exception as exc:
            await self._stop_error(site_id, exc, generation=generation)

    async def _stop_error(self, site_id, exc, *, generation=None):
        async with self.database.transaction() as session:
            row = await session.scalar(
                select(ProviderInstanceRow)
                .where(ProviderInstanceRow.external_id == site_id)
                .with_for_update()
            )
            if (
                row
                and row.state in {State.DRAINING, State.STOPPING}
                and (generation is None or row.shutdown_generation == generation)
            ):
                retries = int(row.metadata_json.get("shutdown_preflight_retries", 0))
                if isinstance(exc, ShutdownPreflightBusy) and retries < 3:
                    row.metadata_json = {
                        **row.metadata_json,
                        "shutdown_preflight_retries": retries + 1,
                    }
                    return
                row.metadata_json = {
                    **row.metadata_json,
                    "manual_error": type(exc).__name__,
                    "needs_attention": True,
                }
                # Never leave a silently stuck spinner or blindly replay
                # an ambiguous command. Keep the site fenced for attention.
                row.state = State.ERROR
                site = await session.get(SiteRow, site_id, with_for_update=True)
                if site is not None:
                    site.state = SiteState.ERROR
        log.warning(
            "manual_autodl_shutdown_needs_attention",
            site_id=site_id,
            error_type=type(exc).__name__,
        )

    async def close(self):
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
