"""Console entrypoint and production wiring for a physical-site agent."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
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
    SSHLocalForwardConfig,
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
    RcloneCommandReader,
    RcloneMirrorStorage,
    RcloneMountStorage,
    RcloneMountSupervisor,
)
from topodrome.storage.base import AuthoritativeDigestReader, StorageAdapter
from topodrome.storage.remote_digest import SchedulerDigestReader

from .admission import GIB, AdmissionController, LeaseFenceStore
from .agent import SiteAgent
from .cooperative import CooperativeExitPolicy, CooperativeExitVerifier
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
    storage_mount: RcloneMountSupervisor | None = None
    control_tunnel: SSHLocalForwardSupervisor | None = None

    async def run(self) -> None:
        # This must precede every lazy Ray import. Topodrome, rather than Ray's
        # crash reporter, owns service-control SIGINT/SIGTERM handling.
        configure_ray_driver_environment()
        loop = asyncio.get_running_loop()
        shutdown_task: asyncio.Task[None] | None = None

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
            if shutdown_task is not None:
                # Retrieve any failure deterministically. On a normal signal
                # path agent.run cannot finish until shutdown set its stop
                # event; on an unrelated failure this wait remains bounded by
                # the application's outer shutdown deadline.
                await shutdown_task
            self.inventory.close()
            self.fences.close()
            self.spool.close()
            if self.storage_mount is not None:
                self.storage_mount.stop()
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
        verify_lfs_sha256=True,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.topodrome-stage-{uuid4().hex}")
    try:
        shutil.copytree(source, temporary, symlinks=False)
        verified = verify_transformers_payload(
            temporary,
            revision=registration.key.revision,
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

    storage, storage_mount = _storage(settings)
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
    if storage_mount is not None:
        storage_mount.start()
        rollback.append(storage_mount.stop)
    validate_layout(settings.global_root / "storage-layout.json", settings.storage_layout)
    if not storage.healthcheck():
        raise RuntimeError("the configured global storage is unavailable")
    workspaces = WorkspaceManager(storage=storage, scratch_root=settings.scratch_root)
    registry, acquirer = _model_registry(settings)
    host_inventory = SystemHostInventory()
    control_tunnel = _control_tunnel(settings)

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
        storage_mount,
        control_tunnel,
    )


def _control_tunnel(settings: SiteSettings) -> SSHLocalForwardSupervisor | None:
    configured = settings.ssh_local_forward
    if configured is None:
        return None
    return SSHLocalForwardSupervisor(
        SSHLocalForwardConfig(
            identity_file=configured.identity_file,
            known_hosts_file=configured.known_hosts_file,
            remote_host=configured.remote_host,
            remote_user=configured.remote_user,
            remote_port=configured.remote_port,
            local_host=configured.local_host,
            local_port=configured.local_port,
            target_host=configured.target_host,
            target_port=configured.target_port,
            state_directory=configured.state_directory,
            ssh_executable=configured.ssh_executable,
            startup_timeout_seconds=configured.startup_timeout_seconds,
            shutdown_timeout_seconds=configured.shutdown_timeout_seconds,
        )
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
) -> tuple[StorageAdapter, RcloneMountSupervisor | None]:
    if settings.rclone is None:
        return LocalStorage(settings.global_root, create=False), None
    reader: AuthoritativeDigestReader = RcloneCommandReader(
        settings.rclone.remote,
        binary=settings.rclone.binary,
        command_timeout_seconds=settings.rclone.verification_timeout_seconds,
        transfer_timeout_seconds=settings.rclone.transfer_timeout_seconds,
    )
    if settings.rclone.verification_via_scheduler:
        if settings.enrollment_token is None:
            raise ValueError("scheduler storage verification requires site enrollment")
        reader = SchedulerDigestReader(
            settings.scheduler_url,
            settings.site_id,
            settings.enrollment_token.get_secret_value(),
            health_reader=reader,
            timeout=settings.rclone.verification_timeout_seconds,
        )
    if settings.rclone.mode == "mirror":
        return (
            RcloneMirrorStorage(
                settings.global_root,
                settings.rclone.remote,
                binary=settings.rclone.binary,
                command_timeout_seconds=settings.rclone.verification_timeout_seconds,
                transfer_timeout_seconds=settings.rclone.transfer_timeout_seconds,
                verification_timeout_seconds=settings.rclone.verification_timeout_seconds,
                poll_interval_seconds=min(settings.rclone.poll_interval_seconds, 1.0),
                min_free_space_gb=settings.rclone.min_free_space_gb,
                healthcheck_cache_seconds=settings.rclone.poll_interval_seconds,
                authoritative_reader=reader,
            ),
            None,
        )
    storage = RcloneMountStorage(
        settings.rclone.mountpoint,
        reader,
        verification_timeout_seconds=settings.rclone.verification_timeout_seconds,
    )
    supervisor = RcloneMountSupervisor(
        remote=settings.rclone.remote,
        mountpoint=settings.rclone.mountpoint,
        cache_dir=settings.rclone.cache_dir,
        health_reader=storage,
        binary=settings.rclone.binary,
        vfs_cache_max_size_gb=settings.rclone.vfs_cache_max_size_gb,
        min_free_space_gb=settings.rclone.min_free_space_gb,
        write_back_seconds=settings.rclone.write_back_seconds,
        poll_interval_seconds=settings.rclone.poll_interval_seconds,
        startup_timeout_seconds=settings.rclone.startup_timeout_seconds,
    )
    return storage, supervisor


def _model_registry(
    settings: SiteSettings,
) -> tuple[ModelRegistry, _AcquisitionRouter]:
    registry = ModelRegistry()
    acquirers: dict[ModelKey, HuggingFaceModelAcquirer] = {}
    local_stage_targets: dict[ModelKey, Path] = {}
    if settings.model_config_path is None:
        return registry, _AcquisitionRouter(acquirers, local_stage_targets)
    catalog = ModelCatalog.from_yaml(settings.model_config_path)
    for definition in catalog.models.values():
        location = definition.sites.get(settings.model_catalog_site_id or settings.site_id)
        if location is None:
            continue
        key = ModelKey(
            alias=definition.alias,
            revision=definition.revision,
            backend=definition.backend,
            backend_version=definition.backend_version,
            dtype=definition.dtype,
            quantization=definition.quantization,
        )
        if definition.backend.casefold() == "transformers":
            installed = importlib.metadata.version("transformers")
            if definition.backend_version not in {"configured", installed}:
                raise RuntimeError(
                    f"model {definition.alias!r} requires transformers "
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
            settings.scratch_root / "models" / definition.alias,
        )
        if existing is None and location.origin is None:
            raise RuntimeError(
                f"model {definition.alias!r} has no verified local payload and no origin"
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
