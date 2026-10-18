"""Console entrypoint and production wiring for a physical-site agent."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import logging
import os
import shutil
import signal
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit
from uuid import uuid4

from topodrome.config import ModelCatalog, SiteSettings
from topodrome.deployment.ssh_tunnel import (
    LocalForwardLifecycle,
    SSHLocalForwardConfig,
    SSHLocalForwardGroup,
    SSHLocalForwardSupervisor,
)
from topodrome.protocol.fencing import FencingAuthority
from topodrome.runtime import (
    AsyncRuntimeExecutionService,
    DetachedRayActorAdapter,
    HuggingFaceModelAcquirer,
    ModelKey,
    ModelPayloadVerificationError,
    ModelRegistration,
    ModelRegistry,
    RuntimeExecutionService,
    verify_transformers_payload,
)
from topodrome.runtime.environment import configure_ray_driver_environment
from topodrome.storage import (
    AssetResolver,
    LocalStorage,
)
from topodrome.storage.base import StorageAdapter
from topodrome.storage.bulk_connections import AdaptiveBulkConnections
from topodrome.storage.http_mirror import HttpMirrorStorage
from topodrome.storage.http_transport import SchedulerFileTransport
from topodrome.storage.remote_digest import SchedulerDigestReader
from topodrome.storage.small_files import SchedulerSmallFileClient

from .admission import GIB, AdmissionController, LeaseFenceStore
from .agent import SiteAgent
from .cooperative import CooperativeExitPolicy, CooperativeExitVerifier
from .idle_compute import IdleComputeManager
from .inventory import (
    CooperativeProcessMatcher,
    GPUReservationLockReader,
    NVMLInventoryProvider,
    ProcessOwnershipRegistry,
    SystemHostInventory,
)
from .outbound import OutboundClient, WebSocketTransport
from .pressure import PressurePolicy, SustainedPressureMonitor
from .ray_cluster import RayClusterConfig, SubprocessRayLifecycle
from .rebalance import IdleActorRebalancePolicy, IdleActorRebalancer
from .regional import RegionalStateStore
from .runtime_backend import RuntimeExecutionBackend, source_resolver_for_storage
from .sentinel import SubprocessCudaSentinelFactory
from .spool import EventSpool
from .workspace import WorkspaceManager


@dataclass(slots=True)
class AgentApplication:
    agent: SiteAgent
    inventory: NVMLInventoryProvider
    fences: LeaseFenceStore
    spool: EventSpool
    control_tunnel: LocalForwardLifecycle | None = None
    storage_close: Callable[[], None] | None = None
    storage_maintenance: Callable[[], object] | None = None

    async def run(self) -> None:
        # This must precede every lazy Ray import. Topodrome, rather than Ray's
        # crash reporter, owns service-control SIGINT/SIGTERM handling.
        configure_ray_driver_environment()
        loop = asyncio.get_running_loop()
        shutdown_task: asyncio.Task[None] | None = None
        tunnel_stop = asyncio.Event()
        tunnel_task = None
        renewal_task = None

        async def maintain_tunnels():
            while not tunnel_stop.is_set():
                try:
                    await asyncio.wait_for(tunnel_stop.wait(), timeout=5)
                    return
                except TimeoutError:
                    pass
                try:
                    await asyncio.to_thread(self.control_tunnel.ensure_running)
                except Exception as exc:
                    # SSH ownership errors are safe to report by class only;
                    # do not log endpoint commands, credentials or raw errors.
                    logging.getLogger(__name__).warning(
                        "storage tunnel unavailable: %s", type(exc).__name__
                    )

        async def maintain_bulk():
            while not tunnel_stop.is_set():
                try:
                    await asyncio.wait_for(tunnel_stop.wait(), timeout=5)
                    return
                except TimeoutError:
                    pass
                try:
                    await asyncio.to_thread(self.storage_maintenance)
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "bulk maintenance unavailable: %s", type(exc).__name__
                    )

        def request_shutdown() -> None:
            nonlocal shutdown_task
            if shutdown_task is None:
                shutdown_task = asyncio.create_task(
                    self.agent.shutdown(stop_ray=True),
                    name="topodrome-signal-shutdown",
                )

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, request_shutdown)
            except (NotImplementedError, RuntimeError):  # pragma: no cover - platform specific
                pass
        try:
            if self.control_tunnel is not None:
                await asyncio.to_thread(self.control_tunnel.ensure_running)
                tunnel_task = asyncio.create_task(maintain_tunnels(), name="topodrome-tunnels")
            if self.storage_maintenance is not None:
                renewal_task = asyncio.create_task(maintain_bulk(), name="topodrome-bulk-renewal")
            await self.agent.run()
        except BaseException:
            # A background-loop failure otherwise leaves a Ray head and its
            # model actors alive after every local authority has closed.
            with suppress(BaseException):
                await asyncio.wait_for(
                    self.agent.shutdown(stop_ray=True),
                    timeout=30,
                )
            raise
        finally:
            tunnel_stop.set()
            if renewal_task is not None:
                await renewal_task
            if tunnel_task is not None:
                await tunnel_task
            if shutdown_task is not None:
                # Retrieve any failure deterministically. On a normal signal
                # path agent.run cannot finish until shutdown set its stop
                # event; on an unrelated failure this wait remains bounded by
                # the application's outer shutdown deadline.
                await shutdown_task
            self.inventory.close()
            self.fences.close()
            self.spool.close()
            if self.storage_close is not None:
                await asyncio.to_thread(self.storage_close)
            if self.control_tunnel is not None:
                await asyncio.to_thread(self.control_tunnel.stop)


class _AcquisitionRouter:
    def __init__(
        self,
        acquirers: dict[ModelKey, HuggingFaceModelAcquirer],
        local_stage_targets: dict[ModelKey, Path] | None = None,
    ) -> None:
        self._acquirers = acquirers
        self._local_stage_targets = local_stage_targets or {}

    def __call__(self, registration: ModelRegistration) -> ModelRegistration:
        acquirer = self._acquirers.get(registration.key)
        acquired = registration if acquirer is None else acquirer.acquire(registration)
        target = self._local_stage_targets.get(registration.key)
        if target is None or acquired.local_path is None:
            return acquired
        return _stage_model_to_local(acquired, target)


def _stage_model_to_local(
    registration: ModelRegistration,
    destination: Path,
) -> ModelRegistration:
    """Atomically stage one fully verified persistent model onto local SSD."""

    source = Path(registration.local_path or "").expanduser().resolve()
    target = destination.expanduser()
    if not target.is_absolute():
        raise ValueError("model local staging destination must be absolute")
    target = target.resolve()
    if source == target:
        return registration
    if target.exists():
        verified = verify_transformers_payload(
            target,
            revision=registration.key.revision,
            subfolder=registration.key.subfolder,
            verify_lfs_sha256=True,
        )
        return ModelRegistration(
            key=registration.key,
            origin=registration.origin,
            local_path=target,
            payload_identity=verified.identity_digest,
            loader_options=registration.loader_options,
        )

    # Verify before copying and after publication. A partial tree is never made
    # visible at the configured local path.
    verify_transformers_payload(
        source,
        revision=registration.key.revision,
        subfolder=registration.key.subfolder,
        verify_lfs_sha256=True,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.topodrome-stage-{uuid4().hex}")
    try:
        shutil.copytree(source, temporary, symlinks=False)
        verified = verify_transformers_payload(
            temporary,
            revision=registration.key.revision,
            subfolder=registration.key.subfolder,
            verify_lfs_sha256=True,
        )
        try:
            temporary.rename(target)
        except FileExistsError:
            # A concurrent loader won publication. Its exact payload must still
            # verify before this call can reuse it.
            verified = verify_transformers_payload(
                target,
                revision=registration.key.revision,
                subfolder=registration.key.subfolder,
                verify_lfs_sha256=True,
            )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return ModelRegistration(
        key=registration.key,
        origin=registration.origin,
        local_path=target,
        payload_identity=verified.identity_digest,
        loader_options=registration.loader_options,
    )


def build_application(settings: SiteSettings) -> AgentApplication:
    """Build with a rollback stack so partial construction never leaks services."""

    rollback: list[Callable[[], None]] = []
    try:
        application = _build_application(settings, rollback)
    except BaseException:
        for close in reversed(rollback):
            with suppress(BaseException):
                close()
        raise
    rollback.clear()
    return application


def _build_application(
    settings: SiteSettings,
    rollback: list[Callable[[], None]],
) -> AgentApplication:
    """Build the agent without importing NVML, Ray, or Transformers eagerly."""

    if settings.enrollment_token is None:
        raise ValueError("site enrollment_token is required")
    ownership = ProcessOwnershipRegistry()
    # Patterns are command-line ownership markers only. Generic executable or
    # Ray worker names must never be configured as cooperative evidence.
    cooperative = CooperativeProcessMatcher(
        command_line_patterns=settings.cooperative_process_patterns
    )
    reservations = (
        None
        if settings.gpu_reservation_lock_dir is None
        else GPUReservationLockReader(settings.gpu_reservation_lock_dir)
    )
    inventory = NVMLInventoryProvider(
        ownership,
        cooperative_matcher=cooperative,
        reservation_detector=reservations,
        allowed_gpu_indices=settings.allowed_gpu_indices,
        autodl_gpm_fallback=settings.site_kind.value == "autodl",
    )
    rollback.append(inventory.close)
    snapshots = inventory.sample()
    if not snapshots:
        raise RuntimeError("the site exposes no allowed physical GPUs")

    bulk_connections = _bulk_connections(settings)
    if bulk_connections is not None:
        rollback.append(bulk_connections.close)
    control_tunnel = _control_tunnel(settings, bulk_override=bulk_connections)
    if control_tunnel is not None:
        rollback.append(control_tunnel.stop)
        control_tunnel.ensure_running()
    storage = (
        _storage(settings)
        if bulk_connections is None
        else _storage(settings, bulk_connections=bulk_connections)
    )
    storage_close = getattr(storage, "close", None)
    if storage_close is not None:
        rollback.append(storage_close)
    from topodrome.storage.blobs import validate_layout

    storage.content_addressed = settings.storage_layout == "blobs-v1"
    if storage.content_addressed:
        blob_client = SchedulerDigestReader(
            settings.scheduler_url,
            settings.site_id,
            settings.enrollment_token.get_secret_value(),
            health_reader=storage,
        )
        storage.claim_blob = blob_client.claim_blob
        rollback.append(blob_client.close)
        mirror_close = storage_close

        def close_storage() -> None:
            try:
                if mirror_close is not None:
                    mirror_close()
            finally:
                blob_client.close()

        storage_close = close_storage
    validate_layout(storage.resolve("storage-layout.json"), settings.storage_layout)
    if not storage.healthcheck():
        raise RuntimeError("the configured global storage is unavailable")
    workspaces = WorkspaceManager(storage=storage, scratch_root=settings.scratch_root)
    registry, acquirer = _model_registry(settings)
    host_inventory = SystemHostInventory()

    ray_config = _ray_cluster_config(
        tuple(snapshot.uuid for snapshot in snapshots),
        settings.scratch_root / "ray",
        short_temp_root=settings.ray_short_temp_root,
    )
    ray_lifecycle = SubprocessRayLifecycle(
        ray_config,
        process_ownership=ownership,
    )
    runtime = AsyncRuntimeExecutionService(RuntimeExecutionService(ray=DetachedRayActorAdapter()))
    regional_state = (
        RegionalStateStore(settings.persistent_root)
        if settings.persistent_root is not None
        else None
    )
    from .finalization import FinalizationJournal

    backend = RuntimeExecutionBackend(
        site_id=settings.site_id,
        registry=registry,
        runtime=runtime,
        storage=storage,
        source_resolver=source_resolver_for_storage(storage),
        asset_resolver=AssetResolver(
            storage,
            cache_root=settings.scratch_root / "asset-cache",
        ),
        acquirer=acquirer,
        use_ray=True,
        ray_address=ray_config.address,
        regional_state=regional_state,
        finalization_concurrency=settings.finalization_concurrency,
        checkpoint_keep_snapshots=settings.checkpoint_keep_snapshots,
        checkpoint_prefetch_max_bytes=settings.checkpoint_prefetch_max_bytes,
        checkpoint_prefetch_concurrency=settings.checkpoint_prefetch_concurrency,
        checkpoint_prefetch_memory_floor_bytes=settings.checkpoint_prefetch_memory_floor_bytes,
        memory_available=lambda: host_inventory.sample().available_memory_bytes,
        finalization_journal=FinalizationJournal(settings.spool_path.parent / "finalization"),
    )

    spool = EventSpool(settings.spool_path)
    rollback.append(spool.close)
    token = settings.enrollment_token.get_secret_value()
    control_url = _control_url(
        settings.scheduler_url,
        site_id=settings.site_id,
        session_id=spool.stream_id,
    )
    outbound = OutboundClient(
        spool,
        lambda: WebSocketTransport(
            control_url,
            site_id=settings.site_id,
            bearer_token=token,
            agent_session_id=spool.stream_id,
        ),
    )
    fence_path = settings.spool_path.with_name(f"{settings.spool_path.stem}.leases.sqlite3")
    fences = LeaseFenceStore(fence_path)
    rollback.append(fences.close)
    idle_compute = (
        IdleComputeManager(settings.idle_compute, fences, ownership, cooperative)
        if settings.idle_compute.enabled else None
    )
    if idle_compute is not None:
        outbound.heartbeat_ack = idle_compute.acknowledge
    admission = AdmissionController(
        inventory,
        fences,
        safety_margin_bytes=round(settings.gpu_safety_margin_gb * GIB),
        allow_foreign_processes_when_capacity_sufficient=(
            settings.allow_foreign_processes_when_capacity_sufficient
        ),
        warm_actor_validator=backend.is_warm_allocation,
        host_inventory=host_inventory,
    )
    pressure = SustainedPressureMonitor(
        PressurePolicy(
            # Explicit capacity-sharing means those ordinary foreign jobs are
            # authorized peers, not a reason to evict Topodrome immediately.
            enabled=not settings.allow_foreign_processes_when_capacity_sufficient,
            sustained_seconds=settings.foreign_pressure_grace_seconds,
        )
    )
    idle_actor_rebalancer = IdleActorRebalancer(
        IdleActorRebalancePolicy(
            enabled=settings.idle_actor_rebalance_enabled,
            sustained_seconds=settings.foreign_pressure_grace_seconds,
            cooldown_seconds=settings.idle_actor_rebalance_cooldown_seconds,
            light_contention_utilization_percent=(settings.light_contention_utilization_percent),
            allow_foreign_processes_when_capacity_sufficient=(
                settings.allow_foreign_processes_when_capacity_sufficient
            ),
        )
    )
    exit_verifier = CooperativeExitVerifier(
        inventory,
        CooperativeExitPolicy(exit_timeout_seconds=settings.cooperative_exit_timeout_seconds),
    )
    agent = SiteAgent(
        site_id=settings.site_id,
        site_kind=settings.site_kind.value,
        provider_instance_id=settings.provider_instance_id,
        region=settings.region,
        inventory=inventory,
        host_inventory=host_inventory,
        admission=admission,
        fences=fences,
        ray=ray_lifecycle,
        workspaces=workspaces,
        backend=backend,
        outbound=outbound,
        pressure=pressure,
        idle_actor_rebalancer=idle_actor_rebalancer,
        warm_actor_idle_timeout_seconds=settings.warm_actor_idle_timeout_seconds,
        cancellation_grace_seconds=settings.cancellation_grace_seconds,
        finalization_backlog_limit=settings.finalization_backlog_limit,
        asset_cache_policy={
            "max_age": settings.asset_cache_max_age_seconds,
            "max_bytes": settings.asset_cache_max_bytes,
            "interval": settings.asset_cache_gc_interval_seconds,
        },
        fence_verifier=FencingAuthority(settings.fence_verification_key.get_secret_value()),
        process_ownership=ownership,
        cooperative_exit=exit_verifier,
        sentinel_factory=SubprocessCudaSentinelFactory(),
        idle_compute=idle_compute,
        model_capabilities=backend.model_capabilities,
        cached_assets=backend.cached_assets,
        storage_available=storage.healthcheck,
        regional_durability=regional_state,
        heartbeat_interval_seconds=settings.heartbeat_interval_seconds,
    )
    return AgentApplication(
        agent,
        inventory,
        fences,
        spool,
        control_tunnel,
        storage_close,
        None if bulk_connections is None else bulk_connections.maintain,
    )


def _control_tunnel(
    settings: SiteSettings,
    *,
    bulk_override: AdaptiveBulkConnections | None = None,
) -> LocalForwardLifecycle | None:
    tunnels = [
        bulk_override
        if configured is settings.bulk_ssh_local_forward and bulk_override is not None
        else SSHLocalForwardSupervisor(SSHLocalForwardConfig(**configured.model_dump()))
        for configured in (settings.ssh_local_forward, settings.bulk_ssh_local_forward)
        if configured is not None
    ]
    if not tunnels:
        return None
    return tunnels[0] if len(tunnels) == 1 else SSHLocalForwardGroup(tunnels)


def _bulk_connections(settings: SiteSettings) -> AdaptiveBulkConnections | None:
    configured = settings.storage
    if configured is None or not configured.bulk_reconnect.enabled:
        return None
    if settings.enrollment_token is None:
        raise ValueError("adaptive bulk routes require site enrollment")
    bulk_config = settings.bulk_ssh_local_forward
    if bulk_config is None or configured.bulk_url is None:
        raise ValueError("adaptive bulk routes require a configured bulk SSH forward")
    return AdaptiveBulkConnections(
        SSHLocalForwardConfig(**bulk_config.model_dump()),
        configured.bulk_url,
        settings.site_id,
        settings.enrollment_token.get_secret_value(),
        configured.bulk_reconnect,
        timeout=configured.request_timeout_seconds,
    )


def _ray_executable() -> str:
    """Validate the Ray console exposed beside the agent interpreter."""

    candidate = Path(sys.executable).with_name("ray")
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise RuntimeError(f"Ray executable is absent from the agent environment: {candidate}")
    return str(candidate)


def _ray_cluster_config(
    gpu_uuids: tuple[str, ...],
    temp_dir: Path,
    *,
    short_temp_root: Path | None = Path("/tmp"),  # noqa: S108 -- private validated alias
) -> RayClusterConfig:
    return RayClusterConfig(
        gpu_uuids=gpu_uuids,
        ray_executable=_ray_executable(),
        # uv may expose a system-site Ray console script as a symlink. Running
        # that script follows its system-Python shebang and silently drops the
        # agent overlay, so workers cannot import Topodrome's locked packages.
        # The module entry point keeps sys.prefix and every worker on the exact
        # interpreter environment which launched the site agent.
        ray_command_prefix=(sys.executable, "-m", "ray.scripts.scripts"),
        temp_dir=temp_dir,
        short_temp_root=short_temp_root,
    )


def _storage(
    settings: SiteSettings,
    *,
    bulk_connections: AdaptiveBulkConnections | None = None,
) -> StorageAdapter:
    configured = settings.storage
    if configured is None:
        return LocalStorage(settings.global_root, create=False)
    if settings.enrollment_token is None:
        raise ValueError("HTTP storage requires site enrollment")
    token = settings.enrollment_token.get_secret_value()
    small_files = (
        SchedulerSmallFileClient(
            settings.scheduler_url,
            settings.site_id,
            token,
            max_bytes=configured.small_file_max_bytes,
        )
        if configured.small_file_max_bytes
        else None
    )
    bulk = None
    try:
        bulk = SchedulerFileTransport(
            settings.scheduler_url,
            settings.site_id,
            token,
            bulk_url=configured.bulk_url,
            timeout=configured.request_timeout_seconds,
            min_free_space_gb=configured.min_free_space_gb,
            small_files=small_files,
            bulk_connections=bulk_connections,
        )
        storage = HttpMirrorStorage(
            settings.global_root,
            configured.authority_id or settings.scheduler_url,
            transfer_timeout_seconds=configured.transfer_timeout_seconds,
            verification_timeout_seconds=configured.verification_timeout_seconds,
            min_free_space_gb=configured.min_free_space_gb,
            healthcheck_cache_seconds=configured.healthcheck_cache_seconds,
            authoritative_reader=bulk,
            small_file_client=small_files,
            bulk_transport=bulk,
        )
    except BaseException:
        if bulk is not None:
            bulk.close()
        if small_files is not None:
            small_files.close()
        raise
    return storage


def _model_registry(
    settings: SiteSettings,
) -> tuple[ModelRegistry, _AcquisitionRouter]:
    registry = ModelRegistry()
    acquirers: dict[ModelKey, HuggingFaceModelAcquirer] = {}
    local_stage_targets: dict[ModelKey, Path] = {}
    if settings.model_config_path is None:
        return registry, _AcquisitionRouter(acquirers, local_stage_targets)
    catalog = ModelCatalog.from_yaml(settings.model_config_path)
    for definition in catalog.variants():
        location = definition.sites.get(settings.model_catalog_site_id or settings.site_id)
        if location is None:
            continue
        key = ModelKey(
            repo_id=definition.repo_id,
            revision=definition.revision,
            subfolder=definition.subfolder,
            backend=definition.backend,
            backend_version=definition.backend_version,
            dtype=definition.dtype,
            quantization=definition.quantization,
        )
        if definition.backend.casefold() == "transformers":
            installed = importlib.metadata.version("transformers")
            if definition.backend_version not in {"configured", installed}:
                raise RuntimeError(
                    f"model {definition.repo_id!r} requires transformers "
                    f"{definition.backend_version}, but this site has {installed}"
                )
        configured_paths = (
            location.local_path,
            location.path,
            location.persistent_path,
        )
        existing: Path | None = None
        payload_identity: str | None = None
        incomplete: Path | None = None
        for configured in configured_paths:
            if configured is None:
                continue
            candidate = Path(configured)
            if not candidate.exists():
                continue
            try:
                payload = verify_transformers_payload(
                    candidate,
                    revision=definition.revision,
                    subfolder=definition.subfolder,
                    verify_lfs_sha256=settings.verify_preseeded_model_hashes,
                    revision_manifest=location.revision_manifest,
                )
            except ModelPayloadVerificationError:
                incomplete = incomplete or candidate
                continue
            existing = candidate
            payload_identity = payload.identity_digest
            break
        destination = next(
            (
                Path(path)
                for path in (
                    location.persistent_path,
                    location.local_path,
                    location.path,
                )
                if path is not None
            ),
            settings.scratch_root / "models" / key.fingerprint,
        )
        if existing is None and location.origin is None:
            raise RuntimeError(
                f"model {definition.repo_id!r} has no verified local payload and no origin"
            )
        registration = ModelRegistration(
            key=key,
            origin=location.origin,
            local_path=existing or incomplete,
            payload_identity=payload_identity,
        )
        registry.register(registration)
        if location.local_path is not None and location.persistent_path is not None:
            local_stage_targets[key] = Path(location.local_path)
        if location.origin is not None:
            acquirers[key] = HuggingFaceModelAcquirer(
                destination=destination,
                endpoint=location.endpoint or "https://hf-mirror.com",
                token_provider=lambda: (
                    os.environ.get("TOPODROME_HF_TOKEN") or os.environ.get("HF_TOKEN")
                ),
                ignore_patterns=location.exclude,
            )
    return registry, _AcquisitionRouter(acquirers, local_stage_targets)


def _control_url(base_url: str, *, site_id: str, session_id: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https", "ws", "wss"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("scheduler_url must be a credential-free HTTP(S) or WebSocket URL")
    if parsed.scheme in {"http", "ws"} and not _is_loopback_host(parsed.hostname):
        raise ValueError("non-loopback scheduler_url must use HTTPS or WSS")
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    base_path = parsed.path.rstrip("/")
    path = (
        base_path
        if base_path.endswith("/api/v1/agents/control")
        else f"{base_path}/api/v1/agents/control"
    )
    query = urlencode({"site_id": site_id, "session_id": session_id})
    return urlunsplit((scheme, parsed.netloc, path, query, ""))


def _is_loopback_host(value: str) -> bool:
    normalized = value.strip().strip("[]").casefold()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


async def _run_from_path(config_path: Path) -> None:
    application = build_application(SiteSettings.from_yaml(config_path))
    await application.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Topodrome physical-site agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=os.environ.get("TOPODROME_SITE_CONFIG"),
        required=os.environ.get("TOPODROME_SITE_CONFIG") is None,
        help="site YAML configuration (or set TOPODROME_SITE_CONFIG)",
    )
    arguments = parser.parse_args()
    try:
        asyncio.run(_run_from_path(arguments.config))
    except KeyboardInterrupt:
        pass


__all__ = ["AgentApplication", "build_application", "main"]


if __name__ == "__main__":  # pragma: no cover - console entrypoint
    main()
