from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from topodrome.config import ModelCatalog, SchedulerSettings
from topodrome.db.models import (
    ArtifactRow,
    AttemptRow,
    GPUReservationRow,
    JobArrayRow,
    JobRow,
    KernelSessionRow,
    LeaseRow,
    ProviderInstanceRow,
    ProviderOperationRow,
    SiteRow,
)
from topodrome.db.repositories import SchedulerRepository
from topodrome.db.session import CURRENT_SCHEMA_REVISION, Database
from topodrome.domain.enums import (
    AttemptState,
    GPUReservationState,
    JobState,
    KernelState,
    LeaseState,
    ProviderInstanceState,
    ProviderOperationState,
    SiteKind,
    SiteState,
    WorkloadKind,
)
from topodrome.domain.errors import (
    AdmissionRejectedError,
    ConflictError,
    DatabaseUnavailableError,
    DurabilityError,
    NotFoundError,
    StaleFenceError,
)
from topodrome.domain.events import AgentEnvelope, SchedulerEnvelope
from topodrome.domain.logs import LiveLogChunkReport
from topodrome.domain.scheduling import SitePreferenceRequest
from topodrome.domain.specs import (
    AssetMember,
    AssetRef,
    AttemptLogDescriptor,
    AttemptReport,
    GPUReservationExtensionRequest,
    GPUReservationRequest,
    GPUReservationSummary,
    JobSpec,
    JobSummary,
    KernelLifecycleReport,
    KernelOperationReport,
    KernelOperationRequest,
    KernelOperationResult,
    KernelRequest,
    KernelSummary,
    LeaseRenewal,
    LeaseRenewalGrant,
    SiteHeartbeat,
    new_id,
    utc_now,
)
from topodrome.jupyter.auth import KernelConnectClaims, KernelConnectSigner
from topodrome.protocol.fencing import FencingAuthority
from topodrome.scheduler.actor_policy import InstitutionalActorPolicy
from topodrome.scheduler.assets import (
    AssetCatalog,
    AssetDeleteResult,
    AssetPublishRequest,
    AssetPublishResult,
    AssetSummary,
    AssetTagUpdate,
    AssetVisibility,
    ResolvedAssetInput,
    job_id_from_ref,
    resolve_job_asset,
)
from topodrome.scheduler.control import AgentConnectionRegistry
from topodrome.scheduler.live_logs import LiveLogAppendResult, LiveLogStore
from topodrome.scheduler.placement import choose_placement
from topodrome.scheduler.policy import (
    PolicyStateCommitUncertainError,
    SchedulingPolicySnapshot,
    SchedulingPolicyState,
    SchedulingPolicyStore,
    initial_policy_snapshot,
    is_live_priority_change,
    load_policy_snapshot,
    same_policy_content,
)
from topodrome.scheduler.reservations import GPUReservationCoordinator
from topodrome.scheduler.source_store import SourceStore
from topodrome.scheduler.storage_retention import (
    collect as collect_retained_storage,
)
from topodrome.scheduler.storage_retention import (
    finish_io,
    require_members,
    serialized_storage,
)
from topodrome.scheduler.storage_retention import (
    plan as plan_storage_retention,
)
from topodrome.scheduler.storage_retention import (
    snapshot as storage_snapshot,
)
from topodrome.source_limits import SourceLimits
from topodrome.storage.artifacts import (
    ArtifactCommitError,
    CommitManifest,
    load_published_manifest,
)
from topodrome.storage.local import LocalStorage

if TYPE_CHECKING:
    from topodrome.scheduler.autodl import ProviderOrchestrator
    from topodrome.scheduler.jupyter_bootstrap import JupyterBootstrapController
    from topodrome.scheduler.managed_jupyter import ManagedJupyterController

log = structlog.get_logger(__name__)

_DISPATCH_CANDIDATE_LIMIT = 32
_POLICY_DRAIN_ATTEMPT_STATES = (
    AttemptState.OFFERED,
    AttemptState.ACCEPTED,
    AttemptState.STARTING,
    AttemptState.RUNNING,
    AttemptState.PREEMPT_REQUESTED,
    AttemptState.CHECKPOINTING,
    AttemptState.COMMITTING,
)
_POLICY_DRAIN_LEASE_STATES = (
    LeaseState.OFFERED,
    LeaseState.ACCEPTED,
    LeaseState.ACTIVE,
    LeaseState.RELEASING,
)
_POLICY_DRAIN_PROVIDER_INSTANCE_STATES = (
    ProviderInstanceState.PROVISIONING,
    ProviderInstanceState.STARTING,
    ProviderInstanceState.DRAINING,
    ProviderInstanceState.STOPPING,
    ProviderInstanceState.RELEASING,
)
_POLICY_DRAIN_PROVIDER_OPERATION_STATES = (
    ProviderOperationState.INTENT,
    ProviderOperationState.CALLING,
    ProviderOperationState.RECONCILING,
)


async def _finish_policy_save(
    store: SchedulingPolicyStore,
    state: SchedulingPolicyState,
) -> tuple[Exception | None, bool]:
    """Finish a threaded atomic save even if the awaiting request is canceled."""

    save_task = asyncio.create_task(asyncio.to_thread(store.save, state))
    was_cancelled = False
    while True:
        try:
            await asyncio.shield(save_task)
        except asyncio.CancelledError:
            was_cancelled = True
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            continue
        except Exception as exc:
            return exc, was_cancelled
        return None, was_cancelled


def _asset_ref_from_resolved(resolved: ResolvedAssetInput) -> AssetRef:
    """Convert catalog coordinates into the strict persisted wire model."""

    return AssetRef(
        source_ref=resolved.source_ref,
        kind=resolved.kind,
        digest=resolved.digest,
        total_bytes=resolved.total_bytes,
        members=tuple(
            AssetMember(
                logical_path=member.logical_path,
                relative_path=member.relative_path,
                size_bytes=member.size_bytes,
                sha256=member.sha256,
            )
            for member in resolved.members
        ),
    )


def _require_manifest_matches_rows(
    manifest: CommitManifest,
    rows: Sequence[ArtifactRow],
) -> None:
    """Reject a filesystem manifest that disagrees with scheduler metadata."""

    durable = {(row.attempt_id, row.logical_path, row.size_bytes, row.sha256) for row in rows}
    published = {
        (
            manifest.attempt_id,
            artifact.logical_path,
            artifact.size_bytes,
            artifact.sha256,
        )
        for artifact in manifest.artifacts
    }
    if durable != published:
        raise DurabilityError(
            "committed artifact manifest does not match durable scheduler metadata"
        )


class SchedulerService:
    def __init__(
        self,
        *,
        database: Database,
        settings: SchedulerSettings,
        connections: AgentConnectionRegistry | None = None,
        repository: SchedulerRepository | None = None,
        model_catalog: ModelCatalog | None = None,
        provider_controller: ProviderOrchestrator | None = None,
        jupyter_bootstrap_controller: JupyterBootstrapController | None = None,
        managed_jupyter_controller: ManagedJupyterController | None = None,
        owner_access_authorizer: Callable[[str], bool] | None = None,
        owner_access_revision_provider: Callable[[str], str | None] | None = None,
        settings_path: Path | None = None,
    ) -> None:
        self.database = database
        self.settings = settings
        self.connections = connections or AgentConnectionRegistry()
        from .process_telemetry import ProcessTelemetry

        self.process_telemetry = ProcessTelemetry()
        from .system_metrics import SystemMetrics

        self.system_metrics = SystemMetrics(self)
        self.repository = repository or SchedulerRepository()
        self.model_catalog = model_catalog
        self.provider_controller = provider_controller
        self.jupyter_bootstrap_controller = jupyter_bootstrap_controller
        self.managed_jupyter_controller = managed_jupyter_controller
        self._owner_access_authorizer = owner_access_authorizer
        self._owner_access_revision_provider = owner_access_revision_provider
        self._settings_path = (
            settings_path.expanduser().resolve() if settings_path is not None else None
        )
        initial_policy = initial_policy_snapshot(settings, model_catalog)
        self._policy_store: SchedulingPolicyStore | None = None
        self._pending_policy_snapshot: SchedulingPolicySnapshot | None
        if self._settings_path is not None:
            self._policy_store = SchedulingPolicyStore(settings.global_root)
            persisted_policy = self._policy_store.load_or_seed(initial_policy)
            self._policy_snapshot = persisted_policy.active
            self._pending_policy_snapshot = persisted_policy.pending
        else:
            self._policy_snapshot = initial_policy
            self._pending_policy_snapshot = None
        # Total-order policy staging/promotion with offer creation. Provider
        # mutations are counted separately so reload staging remains prompt
        # while promotion still waits for an old-generation mutation to end.
        self._dispatch_reload_lock = asyncio.Lock()
        from .upgrades import UpgradeController

        self.upgrades = UpgradeController(self)
        if self.provider_controller is not None and hasattr(
            self.provider_controller, "maintenance_excluded"
        ):
            self.provider_controller.maintenance_excluded = self.upgrades.store.blocked
        self._storage_reference_lock = asyncio.Lock()
        self._policy_drain_blocker_count: int | None = (
            None if self._pending_policy_snapshot is not None else 0
        )
        self.fencing = FencingAuthority(settings.fence_signing_key.get_secret_value())
        self.kernel_signer = KernelConnectSigner(
            settings.fence_signing_key.get_secret_value(),
            public_url=settings.public_url,
            ttl_seconds=settings.kernel_connect_ttl_seconds,
        )
        self.sources = SourceStore(
            settings.global_root,
            limits_provider=lambda: self._policy_snapshot.settings.source_limits,
        )
        self._global_storage = LocalStorage(settings.global_root)
        from topodrome.storage.blobs import validate_layout

        validate_layout(settings.global_root / "storage-layout.json", settings.storage_layout)
        self.assets = AssetCatalog(
            settings.global_root,
            content_addressed=settings.storage_layout == "blobs-v1",
        )
        self.live_logs = LiveLogStore(settings.global_root)
        self._live_tokens: dict[str, str] = {}
        self._kernel_operations: dict[str, asyncio.Future[KernelOperationResult]] = {}
        self._kernel_recent_results: OrderedDict[str, tuple[bool, Any, str | None, int]] = (
            OrderedDict()
        )
        self._kernel_session_operation_locks: dict[str, asyncio.Lock] = {}
        self._institutional_actor_policy = InstitutionalActorPolicy()
        self._gpu_reservations = GPUReservationCoordinator(
            database=database,
            repository=self.repository,
            connections=self.connections,
            heartbeat_ttl_seconds=settings.heartbeat_ttl_seconds,
        )
        self._stop_event = asyncio.Event()

    @serialized_storage
    async def submit(
        self,
        spec: JobSpec,
        *,
        idempotency_key: str | None = None,
        owner_username: str = "@operator",
        is_admin: bool = False,
    ) -> JobSummary:
        spec = await self.prepare_submission(spec, owner_username=owner_username, is_admin=is_admin)
        try:
            async with self.database.transaction() as session:
                job = await self.repository.submit_job(
                    session,
                    spec,
                    idempotency_key=idempotency_key,
                    owner_username=owner_username,
                )
                return self.repository.summary(job)
        except OSError as exc:
            raise DatabaseUnavailableError("scheduler database is unavailable") from exc

    async def prepare_submission(self, spec, *, owner_username, is_admin):
        if spec.source_ref is None:
            raise AdmissionRejectedError("batch submission requires an uploaded source_ref")
        try:
            source_root = await asyncio.to_thread(self.sources.resolve, spec.source_ref)
            entrypoint = (source_root / spec.entrypoint).resolve()
            if not entrypoint.is_relative_to(source_root.resolve()) or not entrypoint.is_file():
                raise FileNotFoundError(spec.entrypoint)
        except (FileNotFoundError, RuntimeError, ValueError):
            raise AdmissionRejectedError(
                "source_ref does not name a verified uploaded source bundle"
            ) from None
        spec = await self._pin_assets(
            spec,
            owner_username=owner_username,
            is_admin=is_admin,
        )
        return self._resolve_model(spec)

    @serialized_storage
    async def submit_array(
        self, request, *, owner_username="@operator", is_admin=False, idempotency_key=None
    ):
        from .arrays import submit_array

        return await submit_array(
            self,
            request,
            owner_username=owner_username,
            is_admin=is_admin,
            idempotency_key=idempotency_key,
        )

    @serialized_storage
    async def publish_asset(
        self,
        request: AssetPublishRequest,
        *,
        owner_username: str = "@operator",
        is_admin: bool = False,
    ) -> AssetPublishResult:
        """Publish a friendly immutable name for already committed job output.

        Publication writes only a small catalog manifest.  The authoritative
        bytes remain at their attempt-specific committed paths, which makes a
        multi-GiB publication constant-space and immediately available.
        """

        manifest = await self._validated_commit_manifest(
            request.job_id,
            owner_username=None if is_admin else owner_username,
        )
        from topodrome.scheduler.assets import _select_artifacts

        selected = _select_artifacts(manifest, request.path)
        await finish_io(require_members, self.settings.global_root, selected.members)
        return await finish_io(
            self.assets.publish,
            request,
            owner_username=owner_username,
            source_manifest=manifest,
        )

    async def list_assets(
        self,
        *,
        owner_username: str = "@operator",
        visibility: AssetVisibility | None = None,
        limit: int = 100,
        tags: tuple[str, ...] = (),
        all_versions: bool = False,
    ) -> list[AssetSummary]:
        return await asyncio.to_thread(
            self.assets.list_visible,
            requester_username=owner_username,
            visibility=visibility,
            limit=limit,
            tags=tags,
            all_versions=all_versions,
        )

    @serialized_storage
    async def delete_asset(
        self,
        name: str,
        version: str,
        *,
        owner_username: str = "@operator",
        visibility: AssetVisibility = "private",
        is_admin: bool = False,
    ) -> AssetDeleteResult:
        return await finish_io(
            self.assets.delete,
            name,
            version,
            requester_username=owner_username,
            visibility=visibility,
            is_admin=is_admin,
        )

    @serialized_storage
    async def update_asset_tags(
        self,
        name: str,
        version: str,
        update: AssetTagUpdate,
        *,
        owner_username: str = "@operator",
        visibility: AssetVisibility = "private",
        is_admin: bool = False,
    ) -> AssetSummary:
        return await finish_io(
            self.assets.update_tags,
            name,
            version,
            update,
            requester_username=owner_username,
            visibility=visibility,
            is_admin=is_admin,
        )

    async def asset(
        self,
        name: str,
        version: str,
        *,
        owner_username: str = "@operator",
        visibility: AssetVisibility = "private",
    ) -> AssetSummary:
        return await asyncio.to_thread(
            self.assets.get,
            name,
            version,
            requester_username=owner_username,
            visibility=visibility,
        )

    async def _pin_assets(
        self,
        spec: JobSpec,
        *,
        owner_username: str,
        is_admin: bool,
    ) -> JobSpec:
        """Resolve every public asset name to immutable dispatch coordinates."""

        if not spec.assets:
            return spec
        pinned: dict[str, AssetRef] = {}
        for alias, reference in spec.assets.items():
            try:
                if reference.source_ref.startswith("job://"):
                    producer_job_id = job_id_from_ref(reference.source_ref)
                    manifest = await self._validated_commit_manifest(
                        producer_job_id,
                        owner_username=None if is_admin else owner_username,
                    )
                    resolved = resolve_job_asset(reference.source_ref, manifest)
                elif reference.source_ref.startswith("asset://"):
                    resolved = await asyncio.to_thread(
                        self.assets.resolve,
                        reference.source_ref,
                        requester_username=owner_username,
                    )
                else:
                    raise ValueError("asset inputs require an asset:// or job:// reference")
                await finish_io(require_members, self.settings.global_root, resolved.members)
                pinned[alias] = _asset_ref_from_resolved(resolved)
            except (ArtifactCommitError, ValueError) as exc:
                raise AdmissionRejectedError(
                    f"asset input {alias!r} is invalid or unavailable"
                ) from exc
        return spec.model_copy(update={"assets": pinned})

    @serialized_storage
    async def create_download(self, reference, *, owner_username, is_admin=False):
        from topodrome.storage.download_pins import DownloadPins

        try:
            if reference.startswith("job://"):
                manifest = await self._validated_commit_manifest(
                    job_id_from_ref(reference),
                    owner_username=None if is_admin else owner_username,
                )
                resolved = resolve_job_asset(reference, manifest)
            elif reference.startswith("asset://"):
                resolved = await finish_io(
                    self.assets.resolve,
                    reference,
                    requester_username=owner_username,
                )
            else:
                raise ValueError("use a job:// or asset:// reference")
        except NotFoundError:
            raise NotFoundError("download reference does not exist") from None
        except ValueError as exc:
            raise AdmissionRejectedError("invalid download reference") from exc
        await finish_io(require_members, self.settings.global_root, resolved.members)
        return await finish_io(
            DownloadPins(self.settings.global_root).create, resolved, owner_username
        )

    @serialized_storage
    async def renew_download(self, identity, *, owner_username):
        from topodrome.storage.download_pins import DownloadPins

        return await finish_io(
            DownloadPins(self.settings.global_root).get,
            identity,
            owner_username,
            renew=True,
        )

    @serialized_storage
    async def close_download(self, identity, *, owner_username):
        from topodrome.storage.download_pins import DownloadPins

        await finish_io(DownloadPins(self.settings.global_root).delete, identity, owner_username)

    async def _validated_commit_manifest(
        self,
        job_id: str,
        *,
        owner_username: str | None,
    ) -> CommitManifest:
        """Cross-check the filesystem publication boundary with durable rows."""

        try:
            async with self.database.session() as session:
                rows = await self.repository.committed_artifacts_for_job(
                    session,
                    job_id,
                    owner_username=owner_username,
                )
        except OSError as exc:
            raise DatabaseUnavailableError("scheduler database is unavailable") from exc
        try:
            manifest = await asyncio.to_thread(
                load_published_manifest,
                self._global_storage,
                job_id,
            )
        except (ArtifactCommitError, OSError) as exc:
            raise DurabilityError("committed artifact manifest is unavailable or invalid") from exc
        _require_manifest_matches_rows(manifest, rows)
        return manifest

    async def status(
        self,
        job_id: str,
        *,
        owner_username: str | None = None,
    ) -> JobSummary:
        async with self.database.session() as session:
            job = await self.repository.get_job(
                session,
                job_id,
                owner_username=owner_username,
            )
            return self.repository.summary(job)

    async def reserve_gpu(
        self,
        request: GPUReservationRequest,
        *,
        requested_by: str = "operator",
        idempotency_key: str | None = None,
    ) -> GPUReservationSummary:
        """Durably reserve an exact GPU, totally ordered with offer creation."""

        async with self._dispatch_reload_lock:
            async with self.database.transaction() as session:
                row = await self.repository.create_gpu_reservation(
                    session,
                    request,
                    requested_by=requested_by,
                    idempotency_key=idempotency_key,
                )
                reservation_id = row.id
        await self.reconcile_gpu_reservations_once()
        async with self.database.session() as session:
            row = await self.repository.get_gpu_reservation(session, reservation_id)
            return self.repository.gpu_reservation_summary(row)

    async def get_gpu_reservation(
        self,
        reservation_id: str,
    ) -> GPUReservationSummary:
        async with self.database.transaction() as session:
            await self.repository.expire_gpu_reservations(session)
            row = await self.repository.get_gpu_reservation(session, reservation_id)
            return self.repository.gpu_reservation_summary(row)

    async def list_gpu_reservations(
        self,
        *,
        limit: int = 100,
        active_only: bool = False,
    ) -> list[GPUReservationSummary]:
        async with self.database.transaction() as session:
            await self.repository.expire_gpu_reservations(session)
            rows = await self.repository.list_gpu_reservations(
                session,
                limit=limit,
                active_only=active_only,
            )
            return [self.repository.gpu_reservation_summary(row) for row in rows]

    async def cancel_gpu_reservation(
        self,
        reservation_id: str,
    ) -> GPUReservationSummary:
        """Cancel a reservation idempotently without touching running work."""

        async with self._dispatch_reload_lock:
            async with self.database.transaction() as session:
                row = await self.repository.cancel_gpu_reservation(
                    session,
                    reservation_id,
                )
                return self.repository.gpu_reservation_summary(row)

    async def extend_gpu_reservation(
        self,
        reservation_id: str,
        request: GPUReservationExtensionRequest,
    ) -> GPUReservationSummary:
        """Update only expiry, totally ordered with dispatch and reservation cancellation."""

        async with self._dispatch_reload_lock:
            async with self.database.transaction() as session:
                row = await self.repository.extend_gpu_reservation(session, reservation_id, request)
                return self.repository.gpu_reservation_summary(row)

    async def reconcile_gpu_reservations_once(self) -> int:
        # The shared lock gives reservation readiness/eviction a total order
        # with central offer creation.  The site agent remains the final
        # authority and refuses eviction if live work raced telemetry.
        async with self._dispatch_reload_lock:
            return await self._gpu_reservations.reconcile_once()

    async def attempts(
        self,
        job_id: str,
        *,
        owner_username: str | None = None,
    ) -> list[dict[str, Any]]:
        from .timing import attempt_timing

        async with self.database.session() as session:
            attempts = await self.repository.list_attempts(
                session,
                job_id,
                owner_username=owner_username,
            )
            return [
                {
                    "attempt_id": attempt.id,
                    "attempt_number": attempt.number,
                    "state": attempt.state.value,
                    "site_id": attempt.site_id,
                    "reason_code": attempt.reason_code,
                    "forced_reservation_id": attempt.forced_reservation_id,
                    "checkpoint_ref": attempt.checkpoint_ref,
                    "restore_checkpoint_ref": attempt.restore_checkpoint_ref,
                    "created_at": attempt.created_at,
                    "started_at": attempt.started_at,
                    "completed_at": attempt.completed_at,
                    "compute_finished_at": attempt.compute_finished_at,
                    "timing": attempt_timing(attempt),
                    "gpu_uuids": [
                        claim.gpu_uuid for claim in (attempt.lease.claims if attempt.lease else [])
                    ],
                }
                for attempt in attempts
            ]

    async def cancel(
        self,
        job_id: str,
        *,
        owner_username: str | None = None,
    ) -> JobSummary:
        command: tuple[str, str] | None = None
        from .input_staging import cancel_stage

        stage_command = None
        async with self.database.transaction() as session:
            job = await self.repository.cancel_job(
                session,
                job_id,
                owner_username=owner_username,
            )
            active = await session.scalar(
                select(AttemptRow)
                .where(
                    AttemptRow.job_id == job_id,
                    AttemptRow.state.not_in(
                        [
                            AttemptState.SUCCEEDED,
                            AttemptState.REJECTED,
                            AttemptState.PREEMPTED,
                            AttemptState.FAILED,
                            AttemptState.CANCELED,
                            AttemptState.LOST,
                        ]
                    ),
                )
                .order_by(AttemptRow.number.desc())
                .limit(1)
            )
            if active and active.site_id:
                command = (active.site_id, active.id)
            summary = self.repository.summary(job)
            stage_command = await cancel_stage(session, job_id)
        # Mutate process-local scheduling hints only after the durable owner
        # check succeeds. A user probing another owner's ID must be a complete
        # not-found no-op.
        if stage_command is not None:
            await self.connections.send(stage_command.site_id, stage_command)
        if command:
            site_id, attempt_id = command
            payload: dict[str, Any] = {"attempt_id": attempt_id}
            if attempt_id in self._live_tokens:
                payload["fencing_token"] = self._live_tokens[attempt_id]
            # Existing agents already give ``request_checkpoint`` a
            # cooperative grace followed by an exact-owned hard stop. The
            # marker lets updated agents retain CANCELED attempt semantics.
            payload["cancel_requested"] = True
            await self.connections.send(
                site_id,
                SchedulerEnvelope(site_id=site_id, kind="request_checkpoint", payload=payload),
            )
        return summary

    async def dispatch_once(self) -> bool:
        expected_revision = (
            None if self.settings.insecure_development_mode else CURRENT_SCHEMA_REVISION
        )
        if not await self.database.ready(expected_schema_revision=expected_revision):
            return False
        from topodrome.db.job_ordering import queue_cursor

        from .arrays import array_gate, reconcile_cancellations
        from .fairness import future_gang, protect_gang
        from .timing import set_phase

        await reconcile_cancellations(self)
        offer = None
        stage_commands = []
        offer_connection_generation: str | None = None
        # Serialize the pending-policy barrier with the complete transaction
        # that may create an OFFERED lease. Therefore promotion cannot observe
        # an empty barrier and then be followed by a late old-policy offer.
        async with self._dispatch_reload_lock:
            upgrade_all, upgrade_sites = self.upgrades.store.exclusions()
            if upgrade_all:
                return False
            if self._pending_policy_snapshot is not None:
                await self._promote_pending_policy_if_drained()
                if self._pending_policy_snapshot is not None:
                    return False
            policy = self._policy_snapshot
            dispatch_now = utc_now()
            protected_gang = None
            blocked_arrays = {}
            async with self.database.transaction() as session:
                sites = await self.repository.placement_inventory(session)
                identities = await self.connections.identities()
                sites = [
                    site
                    for site in sites
                    if identities.get(site.site_id)
                    == (site.agent_session_id, site.agent_connection_generation)
                    and site.site_id not in upgrade_sites
                ]
                heartbeat_cutoff = utc_now() - timedelta(
                    seconds=self.settings.heartbeat_ttl_seconds
                )
                cursor: tuple[int, float | None, datetime, str] | None = None
                while offer is None:
                    jobs = await self.repository.lock_job_candidates(
                        session,
                        limit=_DISPATCH_CANDIDATE_LIMIT,
                        after=cursor,
                        now=dispatch_now,
                        starvation_seconds=policy.settings.starvation_seconds,
                    )
                    if not jobs:
                        break
                    for job in jobs:
                        blocked = blocked_arrays.get(job.array_id)
                        if blocked is None:
                            blocked = await array_gate(session, job)
                        if blocked and job.array_id:
                            blocked_arrays[job.array_id] = blocked
                        if blocked:
                            job.blocked_reason = blocked
                            set_phase(job, "held" if blocked == "array_held" else "array_wait")
                            continue
                        spec = JobSpec.model_validate(job.spec)
                        from topodrome.db.models import InputStageRow

                        from .estimates import (
                            history,
                            remaining_seconds,
                            startup_estimates,
                            worthwhile_switch,
                        )
                        from .input_staging import aware, retarget_inputs

                        staged = await session.get(InputStageRow, job.id, with_for_update=True)
                        active_stage = (
                            staged is not None
                            and staged.state in {"downloading", "ready"}
                            and aware(staged.expires_at) > utc_now()
                            and staged.site_id in identities
                        )
                        adaptive = policy.settings.input_placement
                        observations = (
                            await history(
                                session,
                                spec,
                                [s.site_id for s in sites],
                                adaptive,
                            )
                            if adaptive.enabled
                            else {}
                        )
                        placement_options = dict(
                            preferred_site_ids=policy.settings.preferred_site_ids,
                            heartbeat_cutoff=heartbeat_cutoff,
                            fast_gpu_name_patterns=policy.settings.fast_gpu_name_patterns,
                            foreign_capacity_sharing_site_ids=(
                                policy.settings.foreign_capacity_sharing_site_ids
                            ),
                            startup_seconds_by_site=startup_estimates(
                                spec,
                                sites,
                                policy.startup_seconds(
                                    spec.model.repo_id,
                                    spec.model.revision,
                                    spec.model.subfolder,
                                )
                                if spec.model is not None
                                else {},
                                observations,
                            ),
                            queue_delay_seconds_per_job=(
                                policy.settings.queue_delay_seconds_per_job
                            ),
                            light_contention_utilization_percent=(
                                policy.settings.light_contention_utilization_percent
                            ),
                            allow_idle_actor_replacement=(policy.settings.actor_policy_enabled),
                            prefer_autodl_over_foreign_processes=(
                                policy.settings.prefer_autodl_over_foreign_processes
                            ),
                        )
                        available_sites = protect_gang(sites, protected_gang)
                        placements = {
                            site.site_id: choose_placement(spec, [site], **placement_options)
                            for site in available_sites
                        }
                        costs = (
                            {
                                site.site_id: remaining_seconds(
                                    spec,
                                    site,
                                    placements[site.site_id],
                                    observations,
                                    adaptive,
                                    staged if active_stage else None,
                                )
                                for site in sites
                                if placements[site.site_id] is not None
                            }
                            if adaptive.enabled
                            else {}
                        )
                        placement = choose_placement(
                            spec,
                            available_sites,
                            remaining_seconds_by_site=costs,
                            **placement_options,
                        )
                        if active_stage:
                            current = placements.get(staged.site_id)
                            if (
                                adaptive.enabled
                                and placement is not None
                                and placement.site_id != staged.site_id
                                and worthwhile_switch(
                                    costs.get(staged.site_id),
                                    costs[placement.site_id],
                                    staged,
                                    adaptive,
                                )
                            ):
                                commands = await retarget_inputs(
                                    session,
                                    job,
                                    staged,
                                    placement.site_id,
                                )
                                if commands is not None:
                                    for command in commands:
                                        stage_commands.append(
                                            (command, identities[command.site_id][1])
                                        )
                                    log.info(
                                        "input_stage_retargeted",
                                        job_id=job.id,
                                        from_site=commands[0].site_id,
                                        to_site=placement.site_id,
                                    )
                                    continue  # Wait for inputs before creating any lease.
                            placement = current
                        if placement is None:
                            # Stop repeatedly backfilling the GPUs an aged gang
                            # needs. Stage first; only ready inputs may protect
                            # future capacity. No partial lease or preemption.
                            if (
                                protected_gang is None
                                and queue_cursor(
                                    job, dispatch_now, policy.settings.starvation_seconds
                                )[0]
                                == 1001
                            ):
                                future = future_gang(
                                    spec,
                                    sites,
                                    placement_options,
                                    preferred_site=staged.site_id if active_stage else None,
                                )
                                if future is not None:
                                    from .input_staging import prepare_inputs

                                    ready, command = await prepare_inputs(
                                        session, job, future.site_id
                                    )
                                    if command:
                                        stage_commands.append(
                                            (command, identities[future.site_id][1])
                                        )
                                    if ready:
                                        protected_gang = future
                                        job.blocked_reason = "waiting_for_aged_gang"
                                        set_phase(job, "queued")
                                    else:
                                        job.blocked_reason = "staging_inputs"
                                    continue
                            job.blocked_reason = (
                                "aged_gang_protected" if protected_gang else "no_eligible_capacity"
                            )
                            if not active_stage or staged.state == "ready":
                                set_phase(job, "queued")
                            continue
                        if spec.assets or spec.source_ref:
                            from .input_staging import prepare_inputs

                            ready, stage_command = await prepare_inputs(
                                session,
                                job,
                                placement.site_id,
                            )
                            if stage_command is not None:
                                stage_commands.append(
                                    (stage_command, identities[placement.site_id][1])
                                )
                            if not ready:
                                job.blocked_reason = "staging_inputs"
                                continue
                            staged = await session.get(InputStageRow, job.id)
                            staged.state = "consumed"
                        await self.repository.claim_locked_job(session, job)
                        offer_connection_generation = identities[placement.site_id][1]
                        offer = await self.repository.create_offer(
                            session,
                            job=job,
                            site_id=placement.site_id,
                            gpu_uuids=placement.gpu_uuids,
                            authority=self.fencing,
                            offer_ttl=timedelta(seconds=self.settings.offer_ttl_seconds),
                            token_ttl=timedelta(seconds=self.settings.fencing_token_ttl_seconds),
                            lease_ttl=timedelta(seconds=self.settings.lease_ttl_seconds),
                            replace_idle_actors=(placement.idle_actor_replacement_gpu_count > 0),
                        )
                        if job.array_id is not None:
                            from .arrays import next_array_model

                            offer = offer.model_copy(
                                update={"next_model": await next_array_model(session, job)}
                            )
                        break
                    last = jobs[-1]
                    cursor = queue_cursor(last, dispatch_now, policy.settings.starvation_seconds)
        delivered = False
        for stage_command, generation in stage_commands:
            try:
                await self.connections.send(
                    stage_command.site_id, stage_command, expected_generation=generation
                )
            except Exception:
                log.warning("input_stage_delivery_failed", site_id=stage_command.site_id)
        if offer is not None:
            self._live_tokens[offer.attempt_id] = offer.fencing_token
            try:
                delivered = await self.connections.offer(
                    offer,
                    expected_generation=offer_connection_generation,
                )
            except Exception:
                log.warning(
                    "assignment_delivery_failed",
                    site_id=offer.site_id,
                    attempt_id=offer.attempt_id,
                )
                # A send may have reached the worker before raising. Keep its
                # offer fenced until acceptance or expiry; never reassign an
                # ambiguously delivered GPU gang.
            else:
                if not delivered:
                    # The registry returns False only before calling send:
                    # there is no socket or its generation changed. Nothing
                    # reached the worker, so waiting the full offer TTL is
                    # unnecessary. Requeue under durable state locks instead.
                    async with self.database.transaction() as session:
                        withdrawn = await self.repository.withdraw_undelivered_offer(
                            session, attempt_id=offer.attempt_id, site_id=offer.site_id
                        )
                    if withdrawn:
                        self._live_tokens.pop(offer.attempt_id, None)
                        log.info(
                            "assignment_undelivered_requeued",
                            site_id=offer.site_id,
                            attempt_id=offer.attempt_id,
                        )
        return delivered

    def scheduling_policy_snapshot(self) -> SchedulingPolicySnapshot:
        """Return the immutable snapshot used by a newly started dispatch."""

        return self._policy_snapshot

    def source_limits(self) -> SourceLimits:
        """Return the source-admission bounds from one active policy snapshot."""

        return self._policy_snapshot.settings.source_limits

    async def policy_status(self) -> dict[str, object]:
        async with self._dispatch_reload_lock:
            if self._pending_policy_snapshot is not None:
                self._policy_drain_blocker_count = await self._policy_drain_blockers()
            return self._policy_status_payload()

    async def reload_scheduling_policy(self) -> dict[str, object]:
        """Validate and durably stage a drain-before-swap policy generation."""

        if self._settings_path is None:
            raise ConflictError("scheduler was not started with a reloadable config path")
        async with self._dispatch_reload_lock:
            active = self._policy_snapshot
            try:
                candidate = await asyncio.to_thread(
                    load_policy_snapshot,
                    self._settings_path,
                    current_settings=self.settings,
                    current_model_catalog=self.model_catalog,
                    previous=active,
                )
            except Exception as exc:
                # This boundary is deliberately fail-closed. Never include a
                # parser message or configuration value in logs/API errors.
                log.warning(
                    "scheduling_policy_reload_rejected",
                    generation=active.generation,
                    error_type=type(exc).__name__,
                )
                raise AdmissionRejectedError(
                    "scheduling policy reload failed validation; prior policy retained"
                ) from None

            return await self._stage_policy_candidate(candidate)

    async def set_site_preference(self, request: SitePreferenceRequest) -> dict[str, object]:
        """Persist a future-placement order without changing any existing lease."""
        if self._policy_store is None:
            raise ConflictError("scheduler has no durable scheduling policy store")
        async with self._dispatch_reload_lock:
            active = self._policy_snapshot
            if request.expected_generation != active.generation:
                raise ConflictError("scheduling policy changed; refresh before editing")
            async with self.database.session() as session:
                known = set(
                    await session.scalars(
                        select(SiteRow.id).where(SiteRow.state != SiteState.RETIRED)
                    )
                )
            if set(request.preferred_site_ids) - known:
                raise AdmissionRejectedError(
                    "preferred sites must be registered, non-retired sites"
                )
            candidate = SchedulingPolicySnapshot(
                settings=active.settings.model_copy(
                    update={"preferred_site_ids": request.preferred_site_ids},
                    deep=True,
                ),
                startup_seconds_by_model=active.startup_seconds_by_model,
                generation=active.generation + 1,
                loaded_at=utc_now(),
            )
            await self._stage_policy_candidate(candidate)
            if self._pending_policy_snapshot is not None:
                if not await self._promote_pending_policy_if_drained():
                    raise DurabilityError("site preference activation is pending; refresh status")
            return self._policy_status_payload()

    async def _stage_policy_candidate(
        self,
        candidate: SchedulingPolicySnapshot,
    ) -> dict[str, object]:
        """Caller holds the dispatch/reload lock through the durable stage."""
        active = self._policy_snapshot
        pending = self._pending_policy_snapshot
        if pending is not None:
            if same_policy_content(candidate, pending):
                self._policy_drain_blocker_count = await self._policy_drain_blockers()
                return self._policy_status_payload()
            raise ConflictError("a different scheduling policy generation is already draining")
        if same_policy_content(candidate, active):
            return self._policy_status_payload()

        assert self._policy_store is not None
        staged = SchedulingPolicyState(active=active, pending=candidate)
        save_error, was_cancelled = await _finish_policy_save(self._policy_store, staged)
        if save_error is None or isinstance(save_error, PolicyStateCommitUncertainError):
            # Mirror the visible rename before cancellation releases the lock.
            self._pending_policy_snapshot = candidate
            self._policy_drain_blocker_count = None
        if was_cancelled:
            raise asyncio.CancelledError
        if isinstance(save_error, PolicyStateCommitUncertainError):
            self._policy_drain_blocker_count = await self._policy_drain_blockers()
            log.error("scheduling_policy_stage_commit_uncertain", generation=candidate.generation)
            raise DurabilityError(
                "scheduling policy was staged but final durability is uncertain; dispatch is paused"
            ) from None
        if save_error is not None:
            log.error(
                "scheduling_policy_stage_failed",
                generation=candidate.generation,
                error_type=type(save_error).__name__,
            )
            raise DurabilityError("scheduling policy reload could not be staged durably") from None
        self._policy_drain_blocker_count = await self._policy_drain_blockers()
        log.info(
            "scheduling_policy_reload_staged",
            active_generation=active.generation,
            pending_generation=candidate.generation,
            blocker_count=self._policy_drain_blocker_count,
            loaded_at=candidate.loaded_at.isoformat(),
        )
        return self._policy_status_payload()

    async def _promote_pending_policy_if_drained(self) -> bool:
        """Promote a staged generation; caller holds the dispatch/reload lock."""

        pending = self._pending_policy_snapshot
        if pending is None:
            return False
        live_priority_change = is_live_priority_change(self._policy_snapshot, pending)
        blockers = await self._policy_drain_blockers()
        self._policy_drain_blocker_count = blockers
        if blockers and not live_priority_change:
            return False
        assert self._policy_store is not None
        save_error, was_cancelled = await _finish_policy_save(
            self._policy_store,
            SchedulingPolicyState(active=pending),
        )
        if save_error is not None:
            # Keep dispatch paused and the durable pending state intact. A
            # later tick retries promotion without running either generation.
            log.error(
                "scheduling_policy_promotion_failed",
                pending_generation=pending.generation,
                error_type=type(save_error).__name__,
            )
            if was_cancelled:
                raise asyncio.CancelledError
            return False
        self._policy_snapshot = pending
        self._pending_policy_snapshot = None
        self._policy_drain_blocker_count = 0
        if not live_priority_change:
            self._institutional_actor_policy.clear()
        log.info(
            "scheduling_policy_promoted",
            generation=pending.generation,
            loaded_at=pending.loaded_at.isoformat(),
        )
        if was_cancelled:
            raise asyncio.CancelledError
        return True

    async def _policy_drain_blockers(self) -> int:
        async with self.database.session() as session:
            attempt_count = int(
                await session.scalar(
                    select(func.count(AttemptRow.id)).where(
                        AttemptRow.state.in_(_POLICY_DRAIN_ATTEMPT_STATES)
                    )
                )
                or 0
            )
            lease_count = int(
                await session.scalar(
                    select(func.count(LeaseRow.id)).where(
                        LeaseRow.state.in_(_POLICY_DRAIN_LEASE_STATES)
                    )
                )
                or 0
            )
            provider_instance_count = 0
            provider_operation_count = 0
            if self.provider_controller is not None:
                provider_instances = list(
                    await session.scalars(
                        select(ProviderInstanceRow).where(
                            ProviderInstanceRow.provider == "autodl",
                            ProviderInstanceRow.state.in_(
                                (
                                    *_POLICY_DRAIN_PROVIDER_INSTANCE_STATES,
                                    ProviderInstanceState.RUNNING,
                                )
                            ),
                        )
                    )
                )
                provider_instance_count = sum(
                    1
                    for row in provider_instances
                    if row.state in _POLICY_DRAIN_PROVIDER_INSTANCE_STATES
                    or (
                        row.state is ProviderInstanceState.RUNNING
                        and row.metadata_json.get("startup_heartbeat_seen") is False
                    )
                )
                provider_operation_count = int(
                    await session.scalar(
                        select(func.count(ProviderOperationRow.id)).where(
                            ProviderOperationRow.provider == "autodl",
                            ProviderOperationRow.state.in_(_POLICY_DRAIN_PROVIDER_OPERATION_STATES),
                        )
                    )
                    or 0
                )
        # Attempts and leases normally describe the same unit of work. max()
        # avoids double counting while still failing closed on an orphan row.
        return max(attempt_count, lease_count) + provider_instance_count + provider_operation_count

    def _policy_status_payload(self) -> dict[str, object]:
        active = self._policy_snapshot
        pending = self._pending_policy_snapshot
        return {
            "status": "draining" if pending is not None else "active",
            "generation": active.generation,
            "preferred_site_ids": active.settings.preferred_site_ids,
            "pending_preferred_site_ids": (
                pending.settings.preferred_site_ids if pending is not None else None
            ),
            "loaded_at": active.loaded_at,
            "pending_generation": pending.generation if pending is not None else None,
            "pending_loaded_at": pending.loaded_at if pending is not None else None,
            "blocker_count": (self._policy_drain_blocker_count if pending is not None else 0),
        }

    async def handle_agent_envelope(
        self,
        envelope: AgentEnvelope,
        *,
        connection_generation: str | None = None,
    ) -> SchedulerEnvelope:
        if connection_generation is None:
            identity = await self.connections.identity(envelope.site_id)
            if identity is not None and identity[0] == envelope.agent_session_id:
                connection_generation = identity[1]
        duplicate = False
        stale_attempt_event = False
        stale_kernel_event = False
        terminal_lease_renewal: tuple[str, AttemptState] | None = None
        renewal_grant: LeaseRenewalGrant | None = None
        finalization_recovery = None
        terminal_log_pending = False
        finalization_recovery_required = False
        operation_report: KernelOperationReport | None = None
        operation_execution_count = 0
        live_log_report: LiveLogChunkReport | None = None
        live_log_result: LiveLogAppendResult | None = None
        stale_live_log = False
        terminal_log_registration: tuple[str, str, AttemptLogDescriptor | None] | None = None
        process_sample = None
        idle_compute_permission = None
        async with self.database.transaction() as session:
            # Notebook results use the live control lane.  They are correlated
            # by the durable request digest below, but their code/output must
            # never enter the durable agent-event journal.
            event_receipt = None
            if envelope.kind in {"kernel_result", "attempt_log_chunk"}:
                fresh = True
            else:
                fresh, event_receipt = await self.repository.record_agent_event(session, envelope)
            if not fresh:
                duplicate = True
            if envelope.kind in {"lease_renewal", "attempt_event"}:
                from .finalization import expired_upload_message

                finalization_recovery_required = await expired_upload_message(
                    session, self.repository, self.fencing, envelope.site_id, envelope.payload
                )
            if envelope.kind == "input_stage" and fresh:
                from .input_staging import receive_stage

                await receive_stage(session, envelope.site_id, envelope.payload)
            if finalization_recovery_required:
                pass  # Authenticated stale upload message: ACK without a state mutation.
            elif envelope.kind == "attempt_log_complete":
                attempt = await self.repository._get_attempt(
                    session, str(envelope.payload["attempt_id"]), lock=True
                )
                if attempt.site_id != envelope.site_id or attempt.job_id != envelope.payload.get(
                    "job_id"
                ):
                    raise StaleFenceError("log completion site identity mismatch")
                if attempt.state not in {
                    AttemptState.SUCCEEDED,
                    AttemptState.FAILED,
                    AttemptState.CANCELED,
                    AttemptState.LOST,
                    AttemptState.REJECTED,
                    AttemptState.PREEMPTED,
                }:
                    terminal_log_pending = True
                else:
                    descriptor = envelope.payload.get("log")
                    terminal_log_registration = (
                        attempt.job_id,
                        attempt.id,
                        AttemptLogDescriptor.model_validate(descriptor)
                        if descriptor is not None
                        else None,
                    )
            elif envelope.kind == "recover_finalization":
                from .finalization import recover

                finalization_recovery = await recover(
                    session,
                    self.repository,
                    self.fencing,
                    envelope.site_id,
                    envelope.payload,
                    self.settings,
                    event_receipt,
                    replayed=not fresh,
                )
            elif envelope.kind == "lease_renewal":
                assert event_receipt is not None
                renewal = LeaseRenewal.model_validate(envelope.payload)
                terminal = await self.repository.terminal_attempt_for_fenced_message(
                    session,
                    attempt_id=renewal.attempt_id,
                    site_id=envelope.site_id,
                    token=renewal.fencing_token,
                    authority=self.fencing,
                )
                if terminal is None:
                    renewal_grant = await self.repository.renew_lease(
                        session,
                        attempt_id=renewal.attempt_id,
                        site_id=envelope.site_id,
                        token=renewal.fencing_token,
                        authority=self.fencing,
                        lease_ttl=timedelta(seconds=self.settings.lease_ttl_seconds),
                        token_ttl=timedelta(seconds=self.settings.fencing_token_ttl_seconds),
                        renewal_id=envelope.message_id,
                        renewal_issued_at=event_receipt.acknowledged_at,
                        replayed=not fresh,
                    )
                else:
                    terminal_lease_renewal = (renewal.attempt_id, terminal.state)
            elif not fresh:
                if envelope.kind == "attempt_event":
                    # The state transition and journal receipt committed
                    # together. A prior post-commit filesystem verification
                    # may nevertheless have failed, so a byte-identical replay
                    # must retry authoritative log registration before ACKing.
                    try:
                        replayed_report = AttemptReport.model_validate(envelope.payload)
                    except ValueError:
                        raise StaleFenceError("invalid attempt report") from None
                    if replayed_report.site_id != envelope.site_id:
                        raise StaleFenceError("attempt report site identity mismatch")
                    if replayed_report.log_complete:
                        terminal = await self.repository.terminal_attempt_for_fenced_message(
                            session,
                            attempt_id=replayed_report.attempt_id,
                            job_id=replayed_report.job_id,
                            site_id=envelope.site_id,
                            token=replayed_report.fencing_token,
                            authority=self.fencing,
                        )
                        if terminal is not None:
                            terminal_log_registration = (
                                replayed_report.job_id,
                                replayed_report.attempt_id,
                                replayed_report.log,
                            )
            elif envelope.kind in {"heartbeat", "inventory", "capability_report"}:
                assert event_receipt is not None
                heartbeat = SiteHeartbeat.model_validate(envelope.payload)
                if heartbeat.site_id != envelope.site_id:
                    raise ValueError("heartbeat site identity mismatch")
                if heartbeat.agent_session_id != envelope.agent_session_id:
                    raise ValueError("heartbeat agent session identity mismatch")
                accepted = await self.repository.upsert_heartbeat(
                    session,
                    heartbeat,
                    connection_generation=connection_generation,
                    received_at=event_receipt.acknowledged_at,
                )
                if accepted:
                    process_sample = (heartbeat, event_receipt.acknowledged_at)
                    if heartbeat.idle_compute_requested:
                        blocked = await session.scalars(
                            select(GPUReservationRow.gpu_uuid).where(
                                GPUReservationRow.site_id == heartbeat.site_id,
                                GPUReservationRow.state.in_(
                                    [
                                        GPUReservationState.PENDING,
                                        GPUReservationState.READY,
                                    ]
                                ),
                                GPUReservationRow.expires_at > utc_now(),
                            )
                        )
                        idle_compute_permission = {
                            "allowed": heartbeat.state.value == "ready",
                            "blocked_gpu_uuids": sorted(set(blocked)),
                            "valid_for_seconds": 15,
                        }
            elif envelope.kind == "assignment_accepted":
                attempt = await self.repository.accept_offer(
                    session,
                    attempt_id=str(envelope.payload["attempt_id"]),
                    site_id=envelope.site_id,
                    token=str(envelope.payload["fencing_token"]),
                    authority=self.fencing,
                    lease_ttl=timedelta(seconds=self.settings.lease_ttl_seconds),
                )
                await self.repository.bind_kernel_attempt(
                    session,
                    job_id=attempt.job_id,
                    attempt_id=attempt.id,
                    site_id=envelope.site_id,
                )
            elif envelope.kind == "assignment_rejected":
                attempt_id = str(envelope.payload["attempt_id"])
                rejected = await self.repository.reject_offer(
                    session,
                    attempt_id=attempt_id,
                    site_id=envelope.site_id,
                    token=str(envelope.payload["fencing_token"]),
                    authority=self.fencing,
                    reason_code=str(
                        envelope.payload.get("reason_code", "local_admission_rejected")
                    ),
                    message=_optional_string(envelope.payload.get("message")),
                )
                self._live_tokens.pop(attempt_id, None)
                # A rejected offer never started a runtime, so it has no
                # possible authoritative attempt log.
                terminal_log_registration = (rejected.job_id, rejected.id, None)
            elif envelope.kind == "attempt_log_chunk":
                try:
                    candidate_log = LiveLogChunkReport.model_validate(envelope.payload)
                except ValueError:
                    # Validation errors include input values by default. Never
                    # let captured user output or a fencing token reach control
                    # channel errors or service logs.
                    raise StaleFenceError("invalid live attempt log chunk") from None
                if candidate_log.site_id != envelope.site_id:
                    raise StaleFenceError("live log site identity mismatch")
                # Authenticate the immutable identity first so malformed or
                # forged credentials close the stream. An authentic but stale
                # lease is instead acknowledged as a fenced no-op below.
                self.fencing.authenticate(
                    candidate_log.fencing_token,
                    attempt_id=candidate_log.attempt_id,
                    site_id=envelope.site_id,
                )
                try:
                    await self.repository.authenticate_active_attempt_message(
                        session,
                        attempt_id=candidate_log.attempt_id,
                        job_id=candidate_log.job_id,
                        site_id=envelope.site_id,
                        token=candidate_log.fencing_token,
                        authority=self.fencing,
                    )
                except (StaleFenceError, NotFoundError):
                    stale_live_log = True
                else:
                    live_log_report = candidate_log
            elif envelope.kind == "attempt_event":
                try:
                    report = AttemptReport.model_validate(envelope.payload)
                except ValueError:
                    # Validation diagnostics can include the supplied fencing
                    # credential and must not enter WebSocket close reasons.
                    raise StaleFenceError("invalid attempt report") from None
                if report.site_id != envelope.site_id:
                    raise StaleFenceError("attempt report site identity mismatch")
                terminal = await self.repository.terminal_attempt_for_fenced_message(
                    session,
                    attempt_id=report.attempt_id,
                    job_id=report.job_id,
                    site_id=envelope.site_id,
                    token=report.fencing_token,
                    authority=self.fencing,
                )
                if terminal is None:
                    before = await self.repository._get_attempt(
                        session,
                        report.attempt_id,
                        lock=True,
                    )
                    previous_state = before.state
                    await self.repository.update_attempt(
                        session,
                        attempt_id=report.attempt_id,
                        site_id=envelope.site_id,
                        token=report.fencing_token,
                        authority=self.fencing,
                        target=report.state,
                        reason_code=report.reason_code,
                        message=report.message,
                        checkpoint_ref=report.checkpoint_ref,
                        artifacts=report.artifacts,
                        gpu_released=report.gpu_released,
                    )
                    if report.timings is not None and previous_state != report.state:
                        from .estimates import model_context, observe, workload_context

                        job = await self.repository.get_job(session, report.job_id)
                        measured_spec = JobSpec.model_validate(job.spec)
                        if report.state is AttemptState.RUNNING:
                            metric, value = "startup", report.timings.startup_seconds
                            context_key = model_context(measured_spec)
                        elif report.state is AttemptState.COMMITTING and report.gpu_released:
                            metric, value = "compute", report.timings.compute_seconds
                            context_key = workload_context(measured_spec)
                        elif report.state is AttemptState.SUCCEEDED:
                            metric, value = "finalization", report.timings.finalization_seconds
                            context_key = workload_context(measured_spec)
                        else:
                            value = None
                        if value is not None:
                            await observe(session, envelope.site_id, metric, context_key, value)
                else:
                    stale_attempt_event = True
                if stale_attempt_event or report.state in {
                    AttemptState.SUCCEEDED,
                    AttemptState.PREEMPTED,
                    AttemptState.FAILED,
                    AttemptState.CANCELED,
                    AttemptState.LOST,
                }:
                    self._live_tokens.pop(report.attempt_id, None)
                    await self.repository.reconcile_kernel_attempts(session)
                    if report.log_complete:
                        terminal_log_registration = (
                            report.job_id,
                            report.attempt_id,
                            report.log,
                        )
            elif envelope.kind == "kernel_event":
                lifecycle = KernelLifecycleReport.model_validate(envelope.payload)
                terminal_kernel = await self.repository.terminal_kernel_for_fenced_message(
                    session,
                    session_id=lifecycle.session_id,
                    attempt_id=lifecycle.attempt_id,
                    site_id=envelope.site_id,
                    token=lifecycle.fencing_token,
                    authority=self.fencing,
                )
                if terminal_kernel is None:
                    await self._authenticate_kernel_attempt(
                        session,
                        attempt_id=lifecycle.attempt_id,
                        site_id=envelope.site_id,
                        token=lifecycle.fencing_token,
                    )
                    await self.repository.update_kernel_lifecycle(
                        session,
                        session_id=lifecycle.session_id,
                        attempt_id=lifecycle.attempt_id,
                        target=lifecycle.state,
                        site_id=envelope.site_id,
                        relay_id=lifecycle.relay_id,
                        runtime_id=lifecycle.runtime_id,
                        actor_id=lifecycle.actor_id,
                    )
                else:
                    stale_kernel_event = True
            elif envelope.kind == "kernel_result":
                candidate = KernelOperationReport.model_validate(envelope.payload)
                terminal_kernel = await self.repository.terminal_kernel_for_fenced_message(
                    session,
                    session_id=candidate.session_id,
                    attempt_id=candidate.attempt_id,
                    site_id=envelope.site_id,
                    token=candidate.fencing_token,
                    authority=self.fencing,
                )
                if terminal_kernel is None:
                    await self._authenticate_kernel_attempt(
                        session,
                        attempt_id=candidate.attempt_id,
                        site_id=envelope.site_id,
                        token=candidate.fencing_token,
                    )
                    kernel = await self.repository.get_kernel(session, candidate.session_id)
                    if (
                        kernel.attempt_id != candidate.attempt_id
                        or kernel.site_id != envelope.site_id
                        or kernel.state not in {KernelState.RUNNING, KernelState.IDLE}
                    ):
                        raise StaleFenceError("kernel result is not from the current session lease")
                    if _json_size(candidate.result) > self.settings.kernel_max_output_bytes:
                        candidate = candidate.model_copy(
                            update={
                                "succeeded": False,
                                "result": None,
                                "message": "kernel result exceeded the configured output limit",
                            }
                        )
                    result_digest = hashlib.sha256(
                        _canonical_json(
                            {
                                "succeeded": candidate.succeeded,
                                "result": candidate.result,
                                "message": candidate.message,
                            }
                        )
                    ).digest()
                    operation_row = await self.repository.complete_kernel_channel_request(
                        session,
                        operation_id=candidate.operation_id,
                        session_id=candidate.session_id,
                        succeeded=candidate.succeeded,
                        result_digest=result_digest,
                        error_code=None if candidate.succeeded else "kernel_operation_failed",
                    )
                    operation_report = candidate.model_copy(update={"result": candidate.result})
                    operation_execution_count = operation_row.execution_count
                else:
                    stale_kernel_event = True
        if process_sample is not None:
            heartbeat, received_at = process_sample
            self.process_telemetry.record(
                heartbeat,
                generation=connection_generation,
                received_at=received_at,
            )
        if live_log_report is not None:
            try:
                live_log_result = await asyncio.to_thread(
                    self.live_logs.append,
                    live_log_report,
                )
            except OSError:
                # The attempt and its authoritative final log are independent
                # of this advisory side channel. Leave the offset unacknowledged
                # so the agent retries it without exposing a local path/error.
                live_log_result = LiveLogAppendResult(
                    accepted=False,
                    generation=live_log_report.generation,
                    next_offset=live_log_report.offset,
                )
        if terminal_log_registration is not None:
            cleanup_job_id, cleanup_attempt_id, log_descriptor = terminal_log_registration
            attempts_root = self.settings.global_root / "jobs" / cleanup_job_id / "attempts"
            try:
                registered = await asyncio.to_thread(
                    self.live_logs.register_authoritative,
                    cleanup_job_id,
                    cleanup_attempt_id,
                    attempts_root=attempts_root,
                    descriptor=log_descriptor,
                )
            except OSError:
                registered = False
            if not registered:
                # The state/journal transaction is already durable. Closing
                # without an ACK keeps the agent event in its ordered spool;
                # the byte-identical replay above retries this independent
                # filesystem verification without repeating the transition.
                raise DurabilityError("attempt log has not reached verified authoritative storage")
        if renewal_grant is not None:
            # The transaction has committed before either the process-local
            # hint or the ACK can expose the new fencing authority.
            self._live_tokens[renewal_grant.attempt_id] = renewal_grant.fencing_token
        if stale_attempt_event:
            log.info(
                "stale_terminal_attempt_event_ignored",
                site_id=envelope.site_id,
                attempt_id=str(envelope.payload.get("attempt_id", "")),
                reported_state=str(envelope.payload.get("state", "")),
            )
        if stale_kernel_event:
            log.info(
                "stale_terminal_kernel_event_ignored",
                site_id=envelope.site_id,
                session_id=str(envelope.payload.get("session_id", "")),
                attempt_id=str(envelope.payload.get("attempt_id", "")),
                reported_state=str(envelope.payload.get("state", "result")),
            )
        if terminal_lease_renewal is not None:
            attempt_id, terminal_state = terminal_lease_renewal
            self._live_tokens.pop(attempt_id, None)
            log.info(
                "stale_terminal_lease_renewal_ignored",
                site_id=envelope.site_id,
                attempt_id=attempt_id,
                terminal_state=terminal_state.value,
            )
        if self.provider_controller is not None and envelope.kind != "attempt_log_chunk":
            await self.provider_controller.observe_agent_event(
                envelope,
                connection_generation=connection_generation,
            )
        if operation_report is not None:
            self._resolve_kernel_operation(
                operation_report,
                execution_count=operation_execution_count,
            )
        live_log_ack: dict[str, Any] | None = None
        if stale_live_log:
            live_log_ack = {
                "accepted": False,
                "generation": int(envelope.payload.get("generation", 0)),
                "next_offset": int(envelope.payload.get("offset", 0)),
                "duplicate": False,
                "reset": False,
                "stop": False,
                "stale": True,
            }
        elif live_log_result is not None:
            live_log_ack = live_log_result.as_payload()
        return SchedulerEnvelope(
            site_id=envelope.site_id,
            kind="ack",
            payload={
                "message_id": envelope.message_id,
                **(
                    {"idle_compute_permission": idle_compute_permission}
                    if idle_compute_permission is not None
                    else {}
                ),
                **({"terminal_log_pending": True} if terminal_log_pending else {}),
                **(
                    {"finalization_recovery_required": True}
                    if finalization_recovery_required
                    else {}
                ),
                **(
                    {"finalization_recovery": finalization_recovery}
                    if finalization_recovery is not None
                    else {}
                ),
                **({"duplicate": True} if duplicate else {}),
                **(
                    {"stale": True}
                    if (
                        stale_attempt_event
                        or stale_kernel_event
                        or terminal_lease_renewal is not None
                    )
                    else {}
                ),
                **(
                    {
                        "lease_renewal_terminal": {
                            "attempt_id": terminal_lease_renewal[0],
                            "state": terminal_lease_renewal[1].value,
                        }
                    }
                    if terminal_lease_renewal is not None
                    else {}
                ),
                **(
                    {
                        "lease_renewal": renewal_grant.model_dump(mode="json"),
                    }
                    if renewal_grant is not None
                    else {}
                ),
                **({"attempt_log_chunk": live_log_ack} if live_log_ack is not None else {}),
            },
        )

    async def create_kernel(
        self,
        request: KernelRequest,
        *,
        owner_username: str = "@operator",
    ) -> KernelSummary:
        async with self._dispatch_reload_lock:
            return await self._create_kernel(request, owner_username=owner_username)

    async def _create_kernel(
        self,
        request: KernelRequest,
        *,
        owner_username: str = "@operator",
    ) -> KernelSummary:
        session_id = new_id("kernel")
        relay_id = new_id("relay")
        builtin = await asyncio.to_thread(self.sources.ensure_builtin_kernel)
        spec = JobSpec(
            entrypoint="__topodrome__/kernel.py",
            # A normal content-addressed bundle makes the assignment verifiable
            # at every site even though the interactive agent handles it before
            # the batch invocation path.
            source_ref=builtin.source_ref,
            model=request.model,
            resources=request.resources,
            workload_kind=WorkloadKind.INTERACTIVE,
            kwargs={
                "session_id": session_id,
                "relay_id": relay_id,
                "mode": request.mode.value,
                "idle_timeout_seconds": request.idle_timeout_seconds,
            },
            retry={"max_attempts": 1},
        )
        spec = self._resolve_model(spec)
        resolved_request = request.model_copy(
            update={"model": spec.model, "resources": spec.resources}
        )
        request_payload = resolved_request.model_dump(mode="json")
        now = utc_now()
        attach: tuple[str, SchedulerEnvelope, str] | None = None
        async with self.database.transaction() as session:
            host = None
            identity: tuple[str, str] | None = None
            if request.mode.value == "service":
                host = await self.repository.find_shareable_kernel(
                    session,
                    request=request_payload,
                )
                if host is not None:
                    identity = await self.connections.identity(str(host.site_id))
                    site = await session.get(SiteRow, host.site_id)
                    if (
                        site is None
                        or identity is None
                        or self.upgrades.store.blocked(str(host.site_id))
                        or identity != (site.agent_session_id, site.agent_connection_generation)
                    ):
                        host = None
            if host is None:
                job = await self.repository.submit_job(
                    session,
                    spec,
                    owner_username=owner_username,
                )
                kernel = KernelSessionRow(
                    id=session_id,
                    owner_username=owner_username,
                    state=KernelState.REQUESTED,
                    mode=request.mode,
                    request=request_payload,
                    job_id=job.id,
                    relay_id=relay_id,
                    created_at=now,
                    last_activity_at=now,
                    expires_at=now + timedelta(seconds=request.idle_timeout_seconds),
                )
            else:
                assert host.site_id is not None and host.attempt_id is not None
                assert identity is not None
                kernel = KernelSessionRow(
                    id=session_id,
                    owner_username=owner_username,
                    state=KernelState.STARTING,
                    mode=request.mode,
                    request=request_payload,
                    job_id=host.job_id,
                    attempt_id=host.attempt_id,
                    site_id=host.site_id,
                    relay_id=relay_id,
                    runtime_id=host.runtime_id,
                    actor_id=host.actor_id,
                    created_at=now,
                    last_activity_at=now,
                    expires_at=now + timedelta(seconds=request.idle_timeout_seconds),
                )
                attach = (
                    host.site_id,
                    SchedulerEnvelope(
                        site_id=host.site_id,
                        kind="start_kernel",
                        payload={
                            "session_id": session_id,
                            "host_attempt_id": host.attempt_id,
                            "relay_id": relay_id,
                        },
                    ),
                    identity[1],
                )
            session.add(kernel)
            await session.flush()
            summary = _kernel_summary(kernel, resolved_request)
        if attach is not None:
            attach_site_id, attach_envelope, attach_generation = attach
            delivered = await self.connections.send(
                attach_site_id,
                attach_envelope,
                expected_generation=attach_generation,
            )
            if not delivered:
                async with self.database.transaction() as session:
                    row = await self.repository.get_kernel(session, session_id, lock=True)
                    row.state = KernelState.LOST
                    row.runtime_id = None
                    row.actor_id = None
                    row.expires_at = None
                return await self.get_kernel(session_id)
        return summary

    async def get_kernel(
        self,
        session_id: str,
        *,
        owner_username: str | None = None,
    ) -> KernelSummary:
        async with self.database.session() as session:
            kernel = await self.repository.get_kernel(
                session,
                session_id,
                owner_username=owner_username,
            )
            return self._kernel_summary(kernel)

    def set_owner_access_authorizer(
        self,
        authorizer: Callable[[str], bool] | None,
        revision_provider: Callable[[str], str | None] | None = None,
    ) -> None:
        """Install the current local-user whitelist check used by capabilities."""

        self._owner_access_authorizer = authorizer
        self._owner_access_revision_provider = revision_provider
        controller = self.managed_jupyter_controller
        if controller is not None:
            setter = getattr(controller, "set_owner_authorizer", None)
            if callable(setter):
                setter(self.is_owner_authorized, self.owner_access_revision)

    def is_owner_authorized(self, owner_username: str) -> bool:
        """Fail closed for ordinary owners while always retaining operator access."""

        if owner_username == "@operator":
            return True
        authorizer = self._owner_access_authorizer
        if authorizer is None:
            # Multi-user access is disabled in legacy/single-operator mode.
            return True
        try:
            return bool(authorizer(owner_username))
        except Exception as exc:
            log.warning(
                "owner_access_authorization_failed",
                owner_username=owner_username[:64],
                error_type=type(exc).__name__[:80],
            )
            return False

    def owner_access_revision(self, owner_username: str) -> str | None:
        """Return the current non-secret whitelist revision for IDE fencing."""

        if owner_username == "@operator":
            return "@operator"
        provider = self._owner_access_revision_provider
        if provider is None:
            return "legacy" if self.is_owner_authorized(owner_username) else None
        try:
            return provider(owner_username)
        except Exception as exc:
            log.warning(
                "owner_access_revision_failed",
                owner_username=owner_username[:64],
                error_type=type(exc).__name__[:80],
            )
            return None

    async def invalidate_owner_access(self, owner_username: str) -> int:
        """Invalidate relay/IDE capabilities without stopping GPU execution."""

        if owner_username == "@operator":
            raise ConflictError("operator access cannot be invalidated as a user")
        async with self.database.transaction() as session:
            rows = list(
                await session.scalars(
                    select(KernelSessionRow)
                    .where(
                        KernelSessionRow.owner_username == owner_username,
                        KernelSessionRow.state.in_([KernelState.RUNNING, KernelState.IDLE]),
                    )
                    .with_for_update()
                )
            )
            for row in rows:
                row.access_generation += 1
        controller = self.managed_jupyter_controller
        if controller is not None:
            await controller.revoke_owner(owner_username)
        return len(rows)

    async def ensure_managed_jupyter(
        self,
        session_id: str,
        *,
        owner_username: str | None = None,
    ) -> dict[str, Any]:
        await self.get_kernel(session_id, owner_username=owner_username)
        controller = self.managed_jupyter_controller
        if controller is None:
            raise ConflictError("managed Jupyter Server support is disabled")
        return await controller.ensure_server(session_id)

    async def managed_jupyter_info(
        self,
        session_id: str,
        *,
        owner_username: str | None = None,
    ) -> dict[str, Any]:
        await self.get_kernel(session_id, owner_username=owner_username)
        controller = self.managed_jupyter_controller
        if controller is None:
            raise ConflictError("managed Jupyter Server support is disabled")
        return await controller.server_info(session_id)

    async def refresh_managed_kernel_connection(self, session_id: str) -> KernelSummary:
        """Renew one kernel after its exact managed IDE child reports a connection."""

        now = utc_now()
        async with self.database.transaction() as session:
            kernel = await self.repository.refresh_connected_kernel(
                session,
                session_id,
                now=now,
            )
            return self._kernel_summary(kernel)

    async def stop_managed_jupyter(
        self,
        session_id: str,
        *,
        stop_kernel: bool = True,
        owner_username: str | None = None,
    ) -> dict[str, Any]:
        await self.get_kernel(session_id, owner_username=owner_username)
        controller = self.managed_jupyter_controller
        if controller is None:
            raise ConflictError("managed Jupyter Server support is disabled")
        return await controller.stop_server(session_id, stop_kernel=stop_kernel)

    async def connect_kernel_url(
        self,
        session_id: str,
        *,
        owner_username: str | None = None,
    ) -> str:
        now = utc_now()
        async with self.database.transaction() as session:
            kernel = await self.repository.rotate_kernel_access(
                session,
                session_id,
                now=now,
                owner_username=owner_username,
            )
            return self._issue_kernel_url(kernel, now=now)

    async def relay_info(
        self,
        session_id: str,
        access: str,
        *,
        touch: bool = True,
    ) -> dict[str, Any]:
        now = utc_now()
        async with self.database.transaction() as session:
            kernel = await self.repository.get_kernel(session, session_id, lock=True)
            claims = self._verify_kernel_access(kernel, access=access, now=now)
            if touch:
                kernel = await self.repository.touch_kernel(session, session_id, now=now)
            request = KernelRequest.model_validate(kernel.request)
            return {
                "session_id": kernel.id,
                "mode": request.mode.value,
                "model": request.model.model_dump(mode="json"),
                "access_generation": kernel.access_generation,
                "access_expires_at": claims.expires_at,
                "expires_at": kernel.expires_at,
            }

    async def execute_kernel(
        self,
        session_id: str,
        access: str,
        operation: KernelOperationRequest,
        *,
        request_id: str | None = None,
    ) -> KernelOperationResult:
        client_request_id = request_id or new_id("kernel_request")
        request_digest = self._kernel_request_digest(client_request_id, operation)
        operation_lock = self._kernel_session_operation_locks.setdefault(session_id, asyncio.Lock())
        async with operation_lock:
            now = utc_now()
            async with self.database.transaction() as session:
                kernel = await self.repository.get_kernel(session, session_id, lock=True)
                self._verify_kernel_access(kernel, access=access, now=now)
                kernel = await self.repository.touch_kernel(session, session_id, now=now)
                if not kernel.site_id or not kernel.attempt_id or not kernel.relay_id:
                    raise ConflictError("kernel runtime binding is incomplete")
                operation_row, _ = await self.repository.reserve_kernel_channel_request(
                    session,
                    session_id=session_id,
                    request_id=client_request_id,
                    request_digest=request_digest,
                )
                if operation_row.state in {"completed", "failed"}:
                    cached = self._kernel_recent_results.get(operation_row.id)
                    if cached is None:
                        raise ConflictError(
                            "kernel request already completed; its ephemeral output is no "
                            "longer available and the request will not be replayed"
                        )
                    succeeded, cached_result, cached_error, execution_count = cached
                    if not succeeded:
                        raise ConflictError(cached_error or "kernel operation failed")
                    return KernelOperationResult(
                        operation_id=operation_row.id,
                        session_id=session_id,
                        result=cached_result,
                        execution_count=execution_count,
                    )
                site_id = kernel.site_id
                attempt_id = kernel.attempt_id
                relay_id = kernel.relay_id
                operation_id = operation_row.id
                agent_identity = await self.connections.identity(site_id)
                if agent_identity is None:
                    raise ConflictError("kernel site is disconnected")
                agent_session_id, agent_connection_generation = agent_identity
                await self.repository.mark_kernel_channel_request_dispatched(
                    session,
                    operation_id=operation_id,
                    agent_session_id=agent_session_id,
                    agent_connection_generation=agent_connection_generation,
                )

            future: asyncio.Future[KernelOperationResult] = (
                asyncio.get_running_loop().create_future()
            )
            self._kernel_operations[operation_id] = future
            try:
                delivered = await self.connections.send(
                    site_id,
                    SchedulerEnvelope(
                        site_id=site_id,
                        kind="kernel_execute",
                        payload={
                            "operation_id": operation_id,
                            "session_id": session_id,
                            "host_attempt_id": attempt_id,
                            "relay_id": relay_id,
                            "operation": operation.operation,
                            "arguments": operation.arguments,
                            "maximum_output_bytes": self.settings.kernel_max_output_bytes,
                        },
                    ),
                    expected_generation=agent_connection_generation,
                )
                if not delivered:
                    raise ConflictError("kernel site is disconnected")
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=self.settings.kernel_operation_timeout_seconds,
                    )
                except TimeoutError as exc:
                    raise ConflictError("kernel operation timed out") from exc
            finally:
                self._kernel_operations.pop(operation_id, None)

    async def prepare_kernel_operation(
        self,
        session_id: str,
        access: str,
        operation: KernelOperationRequest,
        *,
        request_id: str,
    ) -> int:
        """Durably reserve a Jupyter execution count before execute_input."""

        request_digest = self._kernel_request_digest(request_id, operation)
        operation_lock = self._kernel_session_operation_locks.setdefault(session_id, asyncio.Lock())
        async with operation_lock:
            now = utc_now()
            async with self.database.transaction() as session:
                kernel = await self.repository.get_kernel(session, session_id, lock=True)
                self._verify_kernel_access(kernel, access=access, now=now)
                await self.repository.touch_kernel(session, session_id, now=now)
                row, _ = await self.repository.reserve_kernel_channel_request(
                    session,
                    session_id=session_id,
                    request_id=request_id,
                    request_digest=request_digest,
                )
                return row.execution_count

    async def stop_kernel(
        self,
        session_id: str,
        *,
        owner_username: str | None = None,
    ) -> KernelSummary:
        command: tuple[str, SchedulerEnvelope] | None = None
        async with self.database.transaction() as session:
            kernel = await self.repository.request_kernel_stop(
                session,
                session_id,
                owner_username=owner_username,
            )
            if kernel.site_id and kernel.attempt_id:
                command = (
                    kernel.site_id,
                    SchedulerEnvelope(
                        site_id=kernel.site_id,
                        kind="stop_kernel",
                        payload={
                            "session_id": kernel.id,
                            "host_attempt_id": kernel.attempt_id,
                            "relay_id": kernel.relay_id,
                        },
                    ),
                )
            elif kernel.job_id and kernel.state is KernelState.STOPPING:
                await self.repository.cancel_job(session, kernel.job_id)
                kernel.state = KernelState.STOPPED
                kernel.expires_at = None
            summary = _kernel_summary(kernel)
        # The database transition above already rejects the bridge token. Clear
        # the durable child generation and reap its exact process before waiting
        # on the remote compute stop. Cleanup failure is recoverable from the
        # durable STOPPING state and must not prevent delivery to the site.
        if self.managed_jupyter_controller is not None:
            try:
                await self.managed_jupyter_controller.stop_server(
                    session_id,
                    stop_kernel=False,
                )
            except NotFoundError:
                pass
            except Exception as exc:
                log.warning(
                    "managed_jupyter_stop_failed",
                    session_id=session_id,
                    error_type=type(exc).__name__[:80],
                )
        if command is not None:
            await self.connections.send(*command)
        return summary

    async def interrupt_kernel(self, session_id: str, access: str) -> bool:
        """Request interruption of the newest durable in-flight cell."""

        now = utc_now()
        async with self.database.transaction() as session:
            kernel = await self.repository.get_kernel(session, session_id, lock=True)
            self._verify_kernel_access(kernel, access=access, now=now)
            kernel = await self.repository.touch_kernel(session, session_id, now=now)
            pending = await self.repository.latest_pending_kernel_request(session, session_id)
            if pending is None:
                return False
            if not kernel.site_id or not kernel.attempt_id or not kernel.relay_id:
                raise ConflictError("kernel runtime binding is incomplete")
            command = (
                kernel.site_id,
                SchedulerEnvelope(
                    site_id=kernel.site_id,
                    kind="kernel_interrupt",
                    payload={
                        "operation_id": pending.id,
                        "session_id": session_id,
                        "host_attempt_id": kernel.attempt_id,
                        "relay_id": kernel.relay_id,
                    },
                ),
            )
        return await self.connections.send(*command)

    async def reap_kernels_once(self) -> int:
        now = utc_now()
        commands: list[tuple[str, SchedulerEnvelope]] = []
        async with self.database.transaction() as session:
            expired = await self.repository.claim_expired_kernels(session, now=now)
            await self.repository.reconcile_kernel_attempts(session)
            stopping = await self.repository.stopping_kernels(session)
            for kernel in stopping:
                if kernel.site_id and kernel.attempt_id:
                    commands.append(
                        (
                            kernel.site_id,
                            SchedulerEnvelope(
                                site_id=kernel.site_id,
                                kind="stop_kernel",
                                payload={
                                    "session_id": kernel.id,
                                    "host_attempt_id": kernel.attempt_id,
                                    "relay_id": kernel.relay_id,
                                },
                            ),
                        )
                    )
        for command in commands:
            try:
                await self.connections.send(*command)
            except Exception:
                log.warning("kernel_stop_delivery_failed", site_id=command[0])
        if self.managed_jupyter_controller is not None:
            for kernel in expired:
                try:
                    await self.managed_jupyter_controller.stop_server(
                        kernel.id,
                        stop_kernel=False,
                    )
                except NotFoundError:
                    pass
        return len(expired)

    async def reconcile_institutional_actors_once(self) -> int:
        """Deliver at most one safe, whole-replica warm-actor eviction.

        A queued job changes model residency through its real atomic offer in
        :meth:`dispatch_once`; this maintenance path only sheds surplus idle
        residency.  The policy/reload lock ensures a command selected by an old
        generation is delivered before a new generation can be staged.
        """

        async with self._dispatch_reload_lock:
            if self._pending_policy_snapshot is not None:
                return 0
            policy = self._policy_snapshot
            settings = policy.settings
            if not settings.actor_policy_enabled:
                self._institutional_actor_policy.clear()
                return 0

            identities = await self.connections.identities()
            heartbeat_cutoff = utc_now() - timedelta(seconds=self.settings.heartbeat_ttl_seconds)
            async with self.database.session() as session:
                sites = [
                    site
                    for site in await self.repository.placement_inventory(session)
                    if site.kind in {SiteKind.LAB, SiteKind.JUPYTER}
                    and not self.upgrades.store.blocked(site.site_id)
                    and site.heartbeat_at >= heartbeat_cutoff
                    and identities.get(site.site_id)
                    == (site.agent_session_id, site.agent_connection_generation)
                ]
                queued_rows = list(
                    await session.scalars(
                        select(JobRow)
                        .where(
                            JobRow.cancel_requested.is_(False),
                            JobRow.state.in_([JobState.QUEUED, JobState.RETRY_WAIT]),
                            ~select(JobArrayRow.id)
                            .where(
                                JobArrayRow.id == JobRow.array_id,
                                or_(
                                    JobArrayRow.held.is_(True),
                                    JobArrayRow.cancel_requested.is_(True),
                                ),
                            )
                            .exists(),
                        )
                        .order_by(
                            JobRow.priority.desc(),
                            JobRow.created_at,
                            JobRow.id,
                        )
                    )
                )
            queued_specs = tuple(JobSpec.model_validate(row.spec) for row in queued_rows)
            plan = self._institutional_actor_policy.choose_eviction(
                sites,
                queued_specs,
                actor_floor_gpu_count=settings.institutional_actor_floor_gpu_count,
                contention_threshold_gpu_count=(
                    settings.contended_actor_shedding_threshold_gpu_count
                ),
                stability_seconds=settings.actor_policy_stability_seconds,
                idle_timeout_seconds=(settings.institutional_warm_actor_idle_timeout_seconds),
                idle_timeout_seconds_by_site=(
                    settings.institutional_warm_actor_idle_timeout_seconds_by_site
                ),
            )
            if plan is None:
                return 0
            eviction_id = plan.eviction_id(policy_generation=policy.generation)
            delivered = await self.connections.send(
                plan.site_id,
                SchedulerEnvelope(
                    site_id=plan.site_id,
                    kind="evict_actor",
                    payload={
                        # This is the exact site-published Ray replica identity,
                        # never the scheduler-synthetic ModelActorRow primary key.
                        "eviction_id": eviction_id,
                        "actor_id": plan.actor_id,
                        "gpu_uuids": list(plan.gpu_uuids),
                        "replica_key_hash": plan.replica_key_hash,
                        "model_identity": dict(plan.model_identity),
                        "reason_code": plan.reason_code,
                        "policy_generation": policy.generation,
                        "minimum_idle_seconds": plan.minimum_idle_seconds,
                    },
                ),
                expected_generation=plan.connection_generation,
            )
            if not delivered:
                return 0
            self._institutional_actor_policy.mark_dispatched(
                plan,
                cooldown_seconds=settings.actor_policy_cooldown_seconds,
            )
            log.info(
                "institutional_warm_actor_eviction_requested",
                site_id=plan.site_id,
                actor_id=plan.actor_id,
                gpu_count=len(plan.gpu_uuids),
                reason_code=plan.reason_code,
                occupied_gpu_count=plan.occupied_institutional_gpu_count,
                policy_generation=policy.generation,
            )
            return 1

    @serialized_storage
    async def collect_storage(self, *, apply: bool = False) -> dict[str, Any]:
        """Operator GC; snapshot, pins and publication share a total order."""
        data = await storage_snapshot(self.database)
        policy = self.settings.storage_retention
        if self.settings.storage_layout == "blobs-v1":
            from topodrome.scheduler.storage_retention import collect_blob_storage

            return await finish_io(
                collect_blob_storage,
                self.settings.global_root,
                policy,
                data,
                apply=apply,
            )
        candidates = await finish_io(
            plan_storage_retention,
            self.settings.global_root,
            policy,
            data,
        )
        return await finish_io(
            collect_retained_storage,
            self.settings.global_root,
            policy,
            candidates,
            apply=apply,
        )

    async def run_storage_maintenance(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.settings.storage_retention.interval_seconds,
                )
            except TimeoutError:
                if self.settings.storage_retention.enabled:
                    try:
                        result = await self.collect_storage(apply=True)
                        log.info(
                            "storage_retention_completed",
                            deleted_files=result["deleted_files"],
                            unlinked_bytes=result["unlinked_allocated_bytes"],
                        )
                    except Exception as exc:
                        log.warning("storage_retention_deferred", error_type=type(exc).__name__)

    async def run_background(self) -> None:
        self._stop_event.clear()
        next_provider_reconcile = 0.0
        next_live_log_prune = 0.0
        while not self._stop_event.is_set():
            try:
                await self.dispatch_once()
                async with self.database.transaction() as session:
                    await self.repository.reap_expired_offers(session)
                    await self.repository.reap_expired_leases(
                        session,
                        finalization_grace=timedelta(
                            seconds=self.settings.finalization_recovery_grace_seconds
                        ),
                    )
                # A connected stock-Jupyter child is activity. Reconcile it
                # before claiming elapsed kernel deadlines so its exact verified
                # connection can renew the session after scheduler delay/restart.
                if self.managed_jupyter_controller is not None:
                    await self.managed_jupyter_controller.reconcile_once()
                await self.reap_kernels_once()
                # Explicit reservations override warm-pool floors and are
                # reconciled before discretionary actor maintenance.  Their
                # inventory overlay keeps the same actor out of the automatic
                # policy candidate set while the reservation is live.
                await self.reconcile_gpu_reservations_once()
                await self.reconcile_institutional_actors_once()
                loop_time = asyncio.get_running_loop().time()
                if self.provider_controller is not None and loop_time >= next_provider_reconcile:
                    next_provider_reconcile = (
                        loop_time + self.settings.autodl.reconcile_interval_seconds
                    )
                    await self.provider_controller.reconcile_once()
                if self.jupyter_bootstrap_controller is not None:
                    # Bootstrap task admission must share the idle-claim fence:
                    # a pre-check outside this lock could launch an agent after
                    # the updater had already claimed an idle site.
                    async with self._dispatch_reload_lock:
                        site_id = getattr(
                            getattr(self.jupyter_bootstrap_controller, "settings", None),
                            "site_id",
                            "",
                        )
                        if not self.upgrades.store.blocked(site_id):
                            await self.jupyter_bootstrap_controller.reconcile_once()
                if loop_time >= next_live_log_prune:
                    next_live_log_prune = loop_time + min(
                        3600.0,
                        max(60.0, self.live_logs.retention_seconds / 2),
                    )
                    try:
                        await self.prune_live_logs_once()
                    except Exception as exc:
                        # Advisory sidecar maintenance must not prevent lease,
                        # placement, or provider reconciliation.
                        log.warning(
                            "live_log_prune_failed",
                            error_type=type(exc).__name__,
                        )
            except Exception:
                log.exception("scheduler_iteration_failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.settings.dispatch_interval_seconds,
                )
            except TimeoutError:
                pass

    async def prune_live_logs_once(self) -> int:
        """Retain crash output until a verified final log supersedes it.

        A terminal job can have lost its site before final publication. Its
        bounded, fsynced live bytes may be the only surviving diagnostics.
        Age-based cleanup applies only to sidecars no longer backed by a job.
        """

        async with self.database.session() as session:
            values = await session.execute(
                select(AttemptRow.job_id, AttemptRow.id).join(
                    JobRow, JobRow.id == AttemptRow.job_id
                )
            )
            retained = frozenset((str(job_id), str(attempt_id)) for job_id, attempt_id in values)
        return await asyncio.to_thread(
            self.live_logs.prune_expired,
            retain_attempts=retained,
        )

    def stop_background(self) -> None:
        self._stop_event.set()

    async def provider_instances(self) -> list[dict[str, Any]]:
        if self.provider_controller is None:
            return []
        return await self.provider_controller.instance_summaries()

    async def provider_operations(self) -> list[dict[str, Any]]:
        if self.provider_controller is None:
            return []
        return await self.provider_controller.operation_summaries()

    async def _authenticate_kernel_attempt(
        self,
        session: AsyncSession,
        *,
        attempt_id: str,
        site_id: str,
        token: str,
    ) -> None:
        self.fencing.verify(token, attempt_id=attempt_id, site_id=site_id)
        attempt = await session.get(AttemptRow, attempt_id)
        if (
            attempt is None
            or attempt.site_id != site_id
            or attempt.fencing_token_digest != self.fencing.digest(token)
        ):
            raise StaleFenceError("kernel event fencing token is no longer current")

    def _resolve_kernel_operation(
        self,
        report: KernelOperationReport,
        *,
        execution_count: int,
    ) -> None:
        self._kernel_recent_results[report.operation_id] = (
            report.succeeded,
            report.result,
            report.message,
            execution_count,
        )
        self._kernel_recent_results.move_to_end(report.operation_id)
        while len(self._kernel_recent_results) > self.settings.kernel_request_cache_size:
            self._kernel_recent_results.popitem(last=False)
        future = self._kernel_operations.get(report.operation_id)
        if future is None or future.done():
            return
        if not report.succeeded:
            future.set_exception(ConflictError(report.message or "kernel operation failed"))
            return
        future.set_result(
            KernelOperationResult(
                operation_id=report.operation_id,
                session_id=report.session_id,
                result=report.result,
                execution_count=execution_count,
            )
        )

    def _issue_kernel_url(self, kernel: KernelSessionRow, *, now: datetime) -> str:
        if (
            kernel.state not in {KernelState.RUNNING, KernelState.IDLE}
            or kernel.relay_id is None
            or kernel.expires_at is None
        ):
            raise ConflictError("kernel is not ready")
        url, _ = self.kernel_signer.issue(
            session_id=kernel.id,
            relay_id=kernel.relay_id,
            generation=kernel.access_generation,
            session_expires_at=kernel.expires_at,
            now=now,
        )
        return url

    def _verify_kernel_access(
        self,
        kernel: KernelSessionRow,
        *,
        access: str,
        now: datetime,
    ) -> KernelConnectClaims:
        owner_username = kernel.owner_username
        if not owner_username:
            raise StaleFenceError("kernel access credential is invalid or expired")
        if not self.is_owner_authorized(owner_username):
            # Use the same externally mapped error as a bad signature so a
            # revoked owner cannot use this capability as an existence oracle.
            raise StaleFenceError("kernel access credential is invalid or expired")
        if kernel.state not in {KernelState.RUNNING, KernelState.IDLE} or kernel.relay_id is None:
            raise ConflictError("kernel is not ready")
        try:
            return self.kernel_signer.verify(
                access,
                session_id=kernel.id,
                relay_id=kernel.relay_id,
                generation=kernel.access_generation,
                now=now,
            )
        except ValueError as exc:
            controller = self.managed_jupyter_controller
            if controller is not None and controller.verify_bridge_credential(kernel.id, access):
                return KernelConnectClaims(
                    session_id=kernel.id,
                    relay_id=kernel.relay_id,
                    generation=kernel.access_generation,
                    # Managed bridge capabilities do not expire on the ordinary
                    # user-connect TTL. Session state and process generation are
                    # rechecked on every gateway request and WebSocket frame.
                    expires_at=datetime.max.replace(tzinfo=UTC),
                )
            raise StaleFenceError(str(exc)) from exc

    def _kernel_summary(self, kernel: KernelSessionRow) -> KernelSummary:
        # Credentials are created only by the explicit connect endpoint, which
        # atomically rotates the persisted access generation.  Status reads do
        # not mint invisible bearer URLs.
        return _kernel_summary(kernel)

    def _kernel_request_digest(
        self,
        request_id: str,
        operation: KernelOperationRequest,
    ) -> bytes:
        if not request_id or len(request_id) > 200:
            raise ConflictError("kernel request_id is invalid")
        operation_payload = operation.model_dump(mode="json")
        if _json_size(operation_payload) > self.settings.kernel_max_request_bytes:
            raise ConflictError("kernel operation exceeds the configured request limit")
        return hashlib.sha256(_canonical_json(operation_payload)).digest()

    def _resolve_model(self, spec: JobSpec) -> JobSpec:
        if self.model_catalog is None or spec.model is None:
            return spec
        try:
            definition = self.model_catalog.resolve(
                spec.model.repo_id,
                revision=spec.model.revision,
                subfolder=spec.model.subfolder,
            )
        except ValueError as exc:
            raise AdmissionRejectedError(str(exc)) from None
        model = spec.model.model_copy(
            update={
                "revision": definition.revision,
                "subfolder": definition.subfolder,
                "backend": definition.backend,
                "backend_version": definition.backend_version,
                "dtype": definition.dtype,
                "quantization": definition.quantization,
            }
        )
        resources = spec.resources
        if resources.min_vram_gb == 0:
            resources = resources.model_copy(update={"min_vram_gb": definition.min_vram_gb})
        # ``model_copy(update=...)`` deliberately skips Pydantic validation.
        # Revalidate after catalog defaults so a sharded loader's per-GPU cap
        # cannot exceed the physical free-memory requirement.
        payload = spec.model_dump(mode="python")
        payload.update({"model": model, "resources": resources})
        return JobSpec.model_validate(payload)


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConflictError("kernel operation must be JSON serializable") from exc


def _json_size(value: Any) -> int:
    return len(_canonical_json(value))


def _kernel_summary(
    kernel: KernelSessionRow, request: KernelRequest | None = None
) -> KernelSummary:
    parsed = request or KernelRequest.model_validate(kernel.request)
    return KernelSummary(
        session_id=kernel.id,
        owner_username=kernel.owner_username,
        state=kernel.state,
        mode=kernel.mode,
        model=parsed.model,
        site_id=kernel.site_id,
        created_at=kernel.created_at,
        expires_at=kernel.expires_at,
    )
