"""Asynchronous orchestration for one physical Topodrome site."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import structlog

from topodrome.domain.enums import AttemptState, KernelMode
from topodrome.domain.specs import AttemptLogDescriptor, JobSpec, LeaseRenewalGrant
from topodrome.jupyter.session import SessionLease, SessionLeaseRegistry
from topodrome.storage.progress import cancellable_transfers, observe_transfers

from .admission import (
    GIB,
    ActorAllocation,
    AdmissionController,
    AdmissionDecision,
    LeaseFenceStore,
    LeaseRequest,
)
from .cooperative import CooperativeExitVerifier
from .inventory import (
    GPUSnapshot,
    HostInventoryProvider,
    InventoryProvider,
    ProcessOwnershipRegistry,
)
from .log_stream import LogChunkAcknowledgement, stream_attempt_log
from .outbound import OutboundClient, ReconnectingControlLoop
from .pressure import PressureEvent, SustainedPressureMonitor
from .ray_cluster import RayLifecycle
from .rebalance import IdleActorRebalancer
from .sentinel import CudaSentinel, CudaSentinelFactory
from .transfer_progress import TransferLogReporter
from .workspace import AttemptWorkspace, WorkspaceManager

log = structlog.get_logger(__name__)
_PREEMPT_CONTROL_CANCEL_WAIT_SECONDS = 0.1
_POST_EVICTION_RELEASE_WAIT_SECONDS = 5.0
_POST_EVICTION_RELEASE_POLL_SECONDS = 0.05
_POST_EVICTION_OFFER_RESERVE_SECONDS = 0.25
_LIVE_LOG_FINAL_DRAIN_SECONDS = 2.0
_TERMINAL_LOG_VERIFY_INITIAL_RETRY_SECONDS = 0.1
_TERMINAL_LOG_VERIFY_MAX_RETRY_SECONDS = 2.0
_CHECKPOINT_RECOVERY_WAIT_SECONDS = 5.0


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    """Retrieve a detached control task result without surfacing credentials."""

    with suppress(asyncio.CancelledError, Exception):
        task.result()


@dataclass(frozen=True, slots=True)
class Assignment:
    job_id: str
    attempt_id: str
    lease: LeaseRequest
    model_key: str
    entrypoint: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    raw_spec: Mapping[str, Any] = field(default_factory=dict)
    gpu_indices: Mapping[str, int] = field(default_factory=dict)
    checkpoint_ref: str | None = None
    fencing_token: str | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Assignment:
        payload = _plain_mapping(payload)
        spec = _plain_mapping(payload.get("spec") or payload.get("job_spec") or {})
        job_spec = JobSpec.model_validate(spec)
        claims = payload.get("claims")
        if claims is not None:
            claim_values = tuple(_plain_mapping(value) for value in claims)
            gpu_uuids = tuple(str(value["gpu_uuid"]) for value in claim_values)
            fence_epochs = {str(value["gpu_uuid"]): int(value["epoch"]) for value in claim_values}
        else:
            gpu_uuids = tuple(str(value) for value in payload["gpu_uuids"])
            epochs_payload = payload["fence_epochs"]
            if not isinstance(epochs_payload, Mapping):
                raise ValueError("fence_epochs must be a mapping by GPU UUID")
            fence_epochs = {uuid: int(epochs_payload[uuid]) for uuid in gpu_uuids}

        if len(gpu_uuids) != job_spec.resources.gpu_count:
            raise ValueError("assignment GPU claim count must equal the job's requested GPU count")

        model = job_spec.model.model_dump(mode="json")
        canonical_model_key = json.dumps(
            {
                "model": model,
                "runtime_fingerprint": job_spec.runtime_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        explicit_model_key = payload.get("model_key")
        if explicit_model_key is not None and str(explicit_model_key) != canonical_model_key:
            raise ValueError("assignment model identity does not match its job specification")
        model_key = canonical_model_key
        min_bytes = payload.get("min_free_vram_bytes")
        if min_bytes is None:
            min_bytes = round(
                float(payload.get("min_vram_gb", job_spec.resources.min_vram_gb)) * GIB
            )
        fencing_token = (
            None if payload.get("fencing_token") is None else str(payload["fencing_token"])
        )
        lease_id = payload.get("lease_id")
        if lease_id is None:
            if fencing_token is not None:
                digest = hashlib.sha256(fencing_token.encode()).hexdigest()
                lease_id = f"lease_{digest[:32]}"
            else:
                lease_id = payload.get("message_id")
        if lease_id is None:
            raise ValueError("assignment requires a lease identity")
        lease = LeaseRequest(
            lease_id=str(lease_id),
            attempt_id=str(payload["attempt_id"]),
            gpu_uuids=gpu_uuids,
            fence_epochs=fence_epochs,
            min_free_vram_bytes=int(min_bytes),
            expires_at=_timestamp(payload["expires_at"]),
            cpu_cores=job_spec.resources.cpu,
            memory_bytes=round(job_spec.resources.memory_gb * GIB),
            reuse_actor_id=(
                None if payload.get("reuse_actor_id") is None else str(payload["reuse_actor_id"])
            ),
            model_key=model_key,
            renewal_ttl_seconds=float(payload.get("lease_ttl_seconds", 90)),
            allow_actor_reuse=(
                job_spec.model.parallelism.mode in {"single", "model_sharded"}
                and job_spec.workload_kind.value != "adapter_training"
                and not (
                    job_spec.workload_kind.value == "interactive"
                    and job_spec.kwargs.get("mode") == KernelMode.RAW_MODEL.value
                )
            ),
            replace_idle_actors=bool(payload.get("replace_idle_actors", False)),
        )
        return cls(
            job_id=str(payload["job_id"]),
            attempt_id=str(payload["attempt_id"]),
            lease=lease,
            model_key=model_key,
            entrypoint=str(payload.get("entrypoint") or spec["entrypoint"]),
            kwargs=dict(payload.get("kwargs") or spec.get("kwargs", {})),
            raw_spec=spec,
            checkpoint_ref=(
                None if payload.get("checkpoint_ref") is None else str(payload["checkpoint_ref"])
            ),
            fencing_token=fencing_token,
        )


@dataclass(frozen=True, slots=True)
class PreparedRuntime:
    actor_id: str
    model_key: str
    reused: bool
    opaque: Any = None
    process_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    succeeded: bool
    retryable: bool = False
    keep_warm: bool = True
    dirty: bool = False
    artifacts_verified: bool = False
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ExecutionBackend(Protocol):
    """Runtime boundary; implemented by the local/Ray model actor manager."""

    async def prepare(
        self, assignment: Assignment, workspace: AttemptWorkspace
    ) -> PreparedRuntime: ...

    async def execute(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> ExecutionOutcome: ...

    async def execute_kernel(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
        *,
        operation_id: str,
        session_id: str,
        mode: KernelMode,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Any: ...

    async def request_preemption(self, attempt_id: str) -> None: ...

    async def cancel(self, attempt_id: str) -> None: ...

    async def evict(self, prepared: PreparedRuntime) -> None: ...

    async def evict_warm_replica(
        self,
        allocations: Sequence[ActorAllocation],
    ) -> tuple[int, ...]: ...

    async def force_stop(self, prepared: PreparedRuntime) -> bool: ...

    async def latest_checkpoint(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> str | None: ...

    async def reconnect_warm(
        self,
        allocation: ActorAllocation,
        gpu_index: int,
    ) -> PreparedRuntime | None: ...

    async def shutdown_driver(self) -> bool:
        """Disconnect only this process from its site-local runtime driver."""
        ...


class FenceVerifier(Protocol):
    def verify(
        self,
        token: str,
        *,
        attempt_id: str | None = None,
        site_id: str | None = None,
    ) -> Any: ...


class RegionalDurability(Protocol):
    def durability_barrier(
        self,
        *,
        shutdown_generation: int,
        command_nonce: str,
    ) -> Any: ...


@dataclass(slots=True)
class _ActiveAttempt:
    assignment: Assignment
    task: asyncio.Task[None]
    prepared: PreparedRuntime | None = None
    preempt_requested: bool = False
    cancel_requested: bool = False
    explicit_cancel_requested: bool = False
    lease_expired: bool = False
    sentinel: CudaSentinel | None = None
    fencing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    renewal_pending: bool = False
    scheduler_terminal: bool = False
    gpu_released: bool = False
    transfer_cancel: threading.Event = field(default_factory=threading.Event)
    log_stream_started: bool = False
    workspace: AttemptWorkspace | None = None
    compute_started_at: float | None = None
    finalization_started_at: float | None = None


@dataclass(slots=True)
class _InteractiveHost:
    assignment: Assignment
    prepared: PreparedRuntime
    workspace: AttemptWorkspace
    mode: KernelMode
    runtime_id: str
    sessions: dict[str, str]
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    operation_tasks: dict[str, tuple[str, str, asyncio.Task[None]]] = field(default_factory=dict)
    operation_results: OrderedDict[str, tuple[str, dict[str, Any]]] = field(
        default_factory=OrderedDict
    )
    dirty: bool = False
    failure_message: str | None = None


class _LocalLeaseExpired(RuntimeError):
    pass


class _SchedulerLeaseTerminated(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _PendingTerminalState:
    state: str
    reason_code: str | None = None
    message: str | None = None
    artifacts: tuple[Mapping[str, Any], ...] = ()
    checkpoint_ref: str | None = None
    log: Mapping[str, Any] | None = None
    log_complete: bool = False


class SiteAgent:
    """Local authority coordinating inventory, leases, Ray, and execution."""

    def __init__(
        self,
        *,
        site_id: str,
        inventory: InventoryProvider,
        host_inventory: HostInventoryProvider | None = None,
        admission: AdmissionController,
        fences: LeaseFenceStore,
        ray: RayLifecycle,
        workspaces: WorkspaceManager,
        backend: ExecutionBackend,
        outbound: OutboundClient,
        pressure: SustainedPressureMonitor,
        idle_actor_rebalancer: IdleActorRebalancer | None = None,
        warm_actor_idle_timeout_seconds: float | None = None,
        cancellation_grace_seconds: float = 30.0,
        finalization_backlog_limit: int = 4,
        asset_cache_policy: Mapping[str, int] | None = None,
        fence_verifier: FenceVerifier | None = None,
        process_ownership: ProcessOwnershipRegistry | None = None,
        cooperative_exit: CooperativeExitVerifier | None = None,
        sentinel_factory: CudaSentinelFactory | None = None,
        site_kind: str = "lab",
        provider_instance_id: str | None = None,
        region: str | None = None,
        model_capabilities: Callable[[], Sequence[Mapping[str, Any]]] = tuple,
        cached_assets: Callable[[], Sequence[str]] = tuple,
        supported_workloads: Sequence[str] = (
            "batch",
            "interactive",
            "adapter_training",
        ),
        storage_available: Callable[[], bool] = lambda: True,
        regional_durability: RegionalDurability | None = None,
        heartbeat_interval_seconds: float = 5.0,
        pressure_interval_seconds: float = 2.0,
    ) -> None:
        self.site_id = site_id
        self.inventory = inventory
        self.host_inventory = host_inventory
        self.admission = admission
        self.fences = fences
        self.ray = ray
        self.workspaces = workspaces
        self.backend = backend
        self.outbound = outbound
        self.pressure = pressure
        self.idle_actor_rebalancer = idle_actor_rebalancer
        self.warm_actor_idle_timeout_seconds = warm_actor_idle_timeout_seconds
        self.cancellation_grace_seconds = max(0.0, float(cancellation_grace_seconds))
        if finalization_backlog_limit < 1:
            raise ValueError("finalization backlog limit must be positive")
        self.finalization_backlog_limit = finalization_backlog_limit
        self.asset_cache_policy = asset_cache_policy
        self.fence_verifier = fence_verifier
        self.process_ownership = process_ownership
        self.cooperative_exit = cooperative_exit or CooperativeExitVerifier(inventory)
        self.sentinel_factory = sentinel_factory
        self.site_kind = site_kind
        self.provider_instance_id = provider_instance_id
        self.region = region
        self.model_capabilities = model_capabilities
        self.cached_assets = cached_assets
        self.supported_workloads = tuple(supported_workloads)
        self.storage_available = storage_available
        self.regional_durability = regional_durability
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.pressure_interval_seconds = pressure_interval_seconds
        self._active: dict[str, _ActiveAttempt] = {}
        self._active_lock = asyncio.Lock()
        self._warm_maintenance_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._kernel_hosts: dict[str, _InteractiveHost] = {}
        self._kernel_session_hosts: dict[str, str] = {}
        self._kernel_background_tasks: set[asyncio.Task[Any]] = set()
        self._cancellation_tasks: set[asyncio.Task[Any]] = set()
        self._session_leases = SessionLeaseRegistry()
        self._stop = asyncio.Event()
        self._draining = False
        self._drain_ready_emitted = False
        self._durability_ready_emitted = False
        self._shutdown_generation = 0
        self._drain_command_nonce: str | None = None
        self._durability_command_nonce: str | None = None
        from .input_staging import InputStager
        resolver = getattr(backend, "asset_resolver", None)
        self.input_stager = (
            InputStager(
                resolver, outbound, source_resolver=getattr(backend, "source_resolver", None),
                checkpoint_resolver=getattr(backend, "prefetch_checkpoint", None),
            )
            if resolver is not None else None
        )
        self.finalization_journal = getattr(backend, "finalization_journal", None)
        if self.finalization_journal is not None:
            backend.finalization_authority = self._finalization_authorized
            from .finalization import FinalizationJournal
            self.terminal_log_journal = FinalizationJournal(
                self.finalization_journal.root.parent / "terminal-logs"
            )
        else:
            self.terminal_log_journal = None

    async def run(self) -> None:
        await self._reconcile_startup_claims()
        await self._ensure_ray_and_reconcile_warm_actors()
        await self.outbound.publish("hello", {"site_id": self.site_id, "agent_version": "1"})
        control = ReconnectingControlLoop(self.outbound, self.handle_command)
        tasks = [
            asyncio.create_task(control.run(self._stop), name="topodrome-control"),
            asyncio.create_task(self._heartbeat_loop(), name="topodrome-heartbeat"),
            asyncio.create_task(self._lease_expiry_loop(), name="topodrome-lease-expiry"),
            asyncio.create_task(self._pressure_loop(), name="topodrome-pressure"),
            asyncio.create_task(self._recovery_loop(), name="topodrome-finalization-recovery"),
            asyncio.create_task(self._terminal_log_loop(), name="topodrome-terminal-logs"),
            asyncio.create_task(self._cache_loop(), name="topodrome-cache-maintenance"),
        ]
        stopped = asyncio.create_task(self._stop.wait(), name="topodrome-stop-waiter")
        try:
            done, _ = await asyncio.wait([*tasks, stopped], return_when=asyncio.FIRST_COMPLETED)
            for task in done - {stopped}:
                exception = task.exception()
                if exception is not None:
                    raise exception
                if stopped not in done:
                    raise RuntimeError(f"site-agent background task stopped: {task.get_name()}")
        finally:
            stopped.cancel()
            for task in tasks:
                task.cancel()
            cancellation_tasks = tuple(self._cancellation_tasks)
            for task in cancellation_tasks:
                task.cancel()
            await asyncio.gather(
                stopped,
                *tasks,
                *cancellation_tasks,
                return_exceptions=True,
            )
            await self.outbound.disconnect()

    def _finalization_authorized(self, attempt_id):
        active = self._active.get(attempt_id)
        return bool(active and not active.scheduler_terminal and not active.cancel_requested
                    and active.assignment.lease.expires_at > time.time())

    async def _cache_loop(self):
        policy = self.asset_cache_policy
        if policy is None or self.input_stager is None:
            await self._stop.wait()
            return
        stop = threading.Event()
        task = None
        try:
            while not self._stop.is_set():
                await _wait_or_stop(self._stop, policy["interval"])
                if self._stop.is_set():
                    break
                try:
                    with cancellable_transfers(stop):
                        task = asyncio.create_task(asyncio.to_thread(
                            self.input_stager.resolver.collect,
                            max_age_seconds=policy["max_age"], max_bytes=policy["max_bytes"],
                        ))
                        await asyncio.shield(task)
                except Exception as exc:
                    log.warning("cache_collection_deferred", error_type=type(exc).__name__)
        finally:
            stop.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)

    def _pending_io_receipts(self):
        return any(
            next(journal.pending(), None) is not None
            for journal in (
                getattr(self, "finalization_journal", None),
                getattr(self, "terminal_log_journal", None),
            ) if journal is not None
        )

    def _finalization_backlog(self):
        active = {key for key, item in self._active.items() if getattr(item, "gpu_released", False)}
        journal = getattr(self, "finalization_journal", None)
        if journal is not None:
            active.update(receipt["attempt_id"] for receipt in journal.pending())
        return len(active)

    async def _recovery_loop(self):
        while not self._stop.is_set():
            if self.finalization_journal is not None:
                try:
                    for receipt in tuple(self.finalization_journal.pending()):
                        attempt_id = receipt["attempt_id"]
                        if attempt_id in self._active:
                            continue
                        reply = await self.outbound.publish_request(
                            f"recover-finalization:{attempt_id}", "recover_finalization",
                            {"job_id": receipt["job_id"], "attempt_id": attempt_id},
                        )
                        result = reply["finalization_recovery"]
                        if result.get("retry"):
                            continue
                        if not result["accepted"]:
                            # Retain the payloads, but never revive canceled or
                            # terminal work. They remain available for recovery.
                            self.finalization_journal.complete(attempt_id)
                            continue
                        assignment = Assignment.from_mapping(result["assignment"])
                        self._verify_fencing_token(assignment)
                        async with self._active_lock:
                            task = asyncio.create_task(self._resume_upload(assignment, receipt))
                            self._active[attempt_id] = _ActiveAttempt(
                                assignment, task, gpu_released=True
                            )
                except Exception as exc:
                    log.warning("finalization_recovery_delayed", error_type=type(exc).__name__)
            await _wait_or_stop(self._stop, 5)

    async def _terminal_log_loop(self):
        while not self._stop.is_set():
            if self.terminal_log_journal is not None:
                for receipt in tuple(self.terminal_log_journal.pending()):
                    if receipt["attempt_id"] in self._active:
                        continue
                    try:
                        workspace = await asyncio.to_thread(
                            self.workspaces.prepare, receipt["job_id"], receipt["attempt_id"]
                        )
                        # This interface only needs the immutable coordinates.
                        from types import SimpleNamespace
                        descriptor = await self.backend.verify_attempt_log(
                            SimpleNamespace(**receipt), workspace
                        )
                        reply = await self.outbound.publish_request(
                            "terminal-log:" + receipt["attempt_id"], "attempt_log_complete",
                            {**receipt, "log": descriptor},
                        )
                        if reply.get("terminal_log_pending"):
                            continue
                        self.terminal_log_journal.complete(receipt["attempt_id"])
                    except Exception as exc:
                        log.warning("terminal_log_upload_delayed", error_type=type(exc).__name__)
            await _wait_or_stop(self._stop, 5)

    async def _resume_upload(self, assignment, receipt):
        attempt_id = assignment.attempt_id
        stop = asyncio.Event()
        stream = None
        try:
            workspace = await asyncio.to_thread(
                self.workspaces.prepare, assignment.job_id, attempt_id
            )
            self._active[attempt_id].workspace = workspace
            stream = asyncio.create_task(self._stream_attempt_log_safely(
                attempt_id, workspace.log_dir / "attempt.log", stop
            ))
            outcome = await self.backend.resume_finalization(assignment, workspace, receipt)
            await self._publish_attempt_state(
                assignment, "succeeded", artifacts=tuple(_artifact_payload(outcome.metadata)),
                checkpoint_ref=_optional_metadata_text(outcome.metadata, "checkpoint_ref"),
                attempt_log=outcome.metadata.get("log"), log_complete=True,
            )
            self.finalization_journal.complete(attempt_id)
            await asyncio.to_thread(self.workspaces.cleanup_scratch, workspace)
        except Exception as exc:
            # Receipt and outputs remain durable locally. Reconciliation retries
            # only this uploader, without preparing or acquiring a model actor.
            log.warning("recovered_finalization_delayed", attempt_id=attempt_id,
                        error_type=type(exc).__name__)
            active = self._active[attempt_id]
            if active.explicit_cancel_requested and not active.scheduler_terminal:
                self.terminal_log_journal.save({
                    "job_id": assignment.job_id, "attempt_id": attempt_id,
                })
                await self._publish_attempt_state(
                    assignment, "canceled", reason_code="cancel_requested",
                    message="upload canceled; locally staged outputs retained",
                )
                self.finalization_journal.complete(attempt_id)
        finally:
            stop.set()
            if stream is not None:
                stream.cancel()
                await asyncio.gather(stream, return_exceptions=True)
            async with self._active_lock:
                self._active.pop(attempt_id, None)

    async def _ensure_ray_and_reconcile_warm_actors(self) -> tuple[str, ...]:
        """Make Ray ready and reconcile the allocation ledger to that cluster."""

        ray_started = await self.ray.ensure_running()
        if not ray_started:
            return await self._reconcile_warm_actors()

        # ``True`` is the lifecycle's proof that it started a brand-new Ray
        # cluster. Detached actors recorded for the previous cluster cannot be
        # reconnected, so remove only their durable allocation rows by exact
        # actor identity. No process is signalled here: any unverified orphan
        # remains visible to NVML as foreign pressure and physical admission
        # continues to protect its GPU.
        allocations = await asyncio.to_thread(self.fences.actor_allocations)
        actor_ids = tuple(dict.fromkeys(item.actor_id for item in allocations))
        discarded: list[str] = []
        for actor_id in actor_ids:
            removed = await asyncio.to_thread(self.fences.evict_actor, actor_id)
            if removed > 0:
                discarded.append(actor_id)
        if discarded:
            log.info(
                "stale_warm_allocations_discarded_after_fresh_ray_start",
                actor_ids=discarded,
            )
        return ()

    async def _reconcile_startup_claims(self) -> tuple[str, ...]:
        """Fence off a crashed predecessor before making local GPUs available."""

        # A fresh process cannot safely inherit work owned by the previous site
        # agent. Stop Ray before clearing crash-left claims/actor allocations;
        # durable fence epochs remain and reject stale scheduler commands.
        if not self.fences.active_claims():
            return ()
        stop_confirmed = await self.ray.stop()
        if stop_confirmed is not True:
            raise RuntimeError(
                "cannot reconcile stale GPU claims without proof that the owned Ray runtime stopped"
            )
        return await asyncio.to_thread(self.fences.reconcile_after_runtime_stop)

    async def _reconcile_warm_actors(self) -> tuple[str, ...]:
        """Reconnect exact detached actors before local admission starts."""

        reconnect = getattr(self.backend, "reconnect_warm", None)
        if not callable(reconnect):
            return ()
        allocations = await asyncio.to_thread(self.fences.actor_allocations)
        if not allocations:
            return ()
        snapshots = {item.uuid: item for item in await asyncio.to_thread(self.inventory.sample)}
        recovered: list[str] = []
        grouped: dict[str, list[ActorAllocation]] = defaultdict(list)
        for allocation in allocations:
            grouped[allocation.actor_id].append(allocation)
        for actor_id, members in grouped.items():
            ordered = tuple(sorted(members, key=lambda item: item.replica_rank))
            prepared: PreparedRuntime | None = None
            members_present = all(item.gpu_uuid in snapshots for item in ordered)
            members_idle = all(item.state == "idle" for item in ordered)
            members_stopping = all(item.state == "stopping" for item in ordered)
            complete_replica = bool(
                ordered
                and tuple(item.replica_rank for item in ordered) == tuple(range(len(ordered)))
                and len({item.model_key for item in ordered}) == 1
            )
            if members_present and complete_replica and (members_idle or members_stopping):
                try:
                    if len(ordered) == 1:
                        prepared = await reconnect(ordered[0], snapshots[ordered[0].gpu_uuid].index)
                    else:
                        reconnect_group = getattr(self.backend, "reconnect_warm_group", None)
                        if callable(reconnect_group):
                            prepared = await reconnect_group(
                                ordered,
                                {item.gpu_uuid: snapshots[item.gpu_uuid].index for item in ordered},
                            )
                except Exception as exc:
                    log.warning(
                        "warm_actor_reconnect_failed",
                        actor_id=actor_id,
                        gpu_uuids=[item.gpu_uuid for item in ordered],
                        error=_safe_message(f"{type(exc).__name__}: {exc}"),
                    )
            if (
                prepared is None
                or not prepared.reused
                or prepared.actor_id != actor_id
                or any(prepared.model_key != item.model_key for item in ordered)
            ):
                confirmed = getattr(self.backend, "stale_warm_cleanup_confirmed", None)
                if callable(confirmed) and confirmed(actor_id):
                    # Reconnect proved the exact actor absent or disposed it
                    # first. An ambiguous/unverified process keeps the durable
                    # allocation fenced and is never signalled.
                    if members_stopping:
                        await asyncio.to_thread(
                            self.fences.complete_idle_actor_eviction,
                            actor_id,
                            expected_gpu_uuids=tuple(item.gpu_uuid for item in ordered),
                        )
                    else:
                        await asyncio.to_thread(self.fences.evict_actor, actor_id)
                continue
            if self.process_ownership is not None:
                for pid in prepared.process_ids:
                    self.process_ownership.register(pid, actor_id=prepared.actor_id)
            if members_stopping:
                # A previous generation committed an exact disposal intent but
                # stopped before it could confirm deletion. Resume that intent
                # against the reconnected detached actor before admission.
                evict = getattr(self.backend, "evict_warm_allocation", None)
                gpu_uuids = tuple(item.gpu_uuid for item in ordered)
                evict_replica = getattr(self.backend, "evict_warm_replica", None)
                if not callable(evict) and not callable(evict_replica):
                    continue
                try:
                    worker_pids = await self._evict_exact_warm_replica(ordered)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    restored = False
                    try:
                        if all(
                            self.backend_is_live_allocation(allocation) for allocation in ordered
                        ):
                            restored = await asyncio.to_thread(
                                self.fences.restore_idle_actor_eviction,
                                actor_id,
                                expected_gpu_uuids=gpu_uuids,
                            )
                    except Exception:
                        restored = False
                    if restored:
                        recovered.append(prepared.actor_id)
                    log.warning(
                        "stopping_warm_actor_recovery_failed",
                        actor_id=actor_id,
                        gpu_uuids=list(gpu_uuids),
                        allocation_restored=restored,
                        error=_safe_message(f"{type(exc).__name__}: {exc}"),
                    )
                    continue
                removed = await asyncio.to_thread(
                    self.fences.complete_idle_actor_eviction,
                    actor_id,
                    expected_gpu_uuids=gpu_uuids,
                )
                if self.process_ownership is not None:
                    for pid in {*prepared.process_ids, *worker_pids}:
                        self.process_ownership.unregister(pid)
                if removed != len(ordered):
                    log.error(
                        "stopping_warm_actor_recovery_release_failed",
                        actor_id=actor_id,
                        gpu_uuids=list(gpu_uuids),
                    )
                else:
                    log.info(
                        "stopping_warm_actor_recovery_completed",
                        actor_id=actor_id,
                        gpu_uuids=list(gpu_uuids),
                    )
                continue
            recovered.append(prepared.actor_id)
        return tuple(recovered)

    async def shutdown(self, *, stop_ray: bool = False) -> None:
        # Signals can arrive more than once while a model actor drains. Keep
        # driver disconnection and exact-head termination single-flight so a
        # repeated SIGTERM cannot interleave two teardown sequences.
        async with self._shutdown_lock:
            if self._stop.is_set():
                return
            self._draining = True
            if getattr(self, "input_stager", None) is not None:
                await self.input_stager.close()
            async with self._active_lock:
                active_tasks = tuple(active.task for active in self._active.values())
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            await self._publish_drain_if_ready()
            if stop_ray:
                disconnect = getattr(self.backend, "shutdown_driver", None)
                if callable(disconnect):
                    # The Python driver must detach before the raylet goes
                    # away. Otherwise Ray may terminate this service process
                    # while its asyncio shutdown handler is still draining.
                    await disconnect()
                await self.ray.stop()
            self._stop.set()

    async def wait_for_idle(self) -> None:
        """Wait for the currently accepted attempts without admitting new state."""

        while True:
            async with self._active_lock:
                active_tasks = tuple(active.task for active in self._active.values())
            if not active_tasks:
                return
            await asyncio.gather(*active_tasks, return_exceptions=True)

    async def handle_command(self, message: Mapping[str, Any]) -> None:
        message = _plain_mapping(message)
        command = str(message.get("command") or message.get("kind") or "")
        payload = message.get("payload", message)
        if not isinstance(payload, Mapping):
            payload = {}
        error: str | None = None
        shutdown_ack: dict[str, object] = {}
        actor_ack: dict[str, object] = {}
        try:
            if command == "stage_inputs":
                if self._draining or self.input_stager is None:
                    raise RuntimeError("site cannot accept input staging")
                await self.input_stager.accept(payload)
            elif command == "cancel_input_stage":
                if self.input_stager is not None:
                    self.input_stager.cancel(payload)
            elif command in {"offer_assignment", "assignment"}:
                assignment = Assignment.from_mapping(payload)
                self._verify_fencing_token(assignment)
                await self.offer(assignment)
            elif command == "cancel_attempt":
                await self.cancel(str(payload["attempt_id"]))
            elif command == "request_checkpoint":
                # Rolling compatibility: older agents safely interpret this
                # command as bounded preemption, while updated agents preserve
                # explicit cancellation as the terminal attempt state.
                if payload.get("cancel_requested") is True:
                    await self.cancel(str(payload["attempt_id"]))
                else:
                    await self.request_preemption(str(payload["attempt_id"]), explicit=True)
            elif command == "start_kernel":
                await self._attach_service_kernel(payload)
            elif command == "kernel_execute":
                await self._queue_kernel_operation(payload)
            elif command == "kernel_interrupt":
                await self._interrupt_kernel_operation(payload)
            elif command == "stop_kernel":
                await self._stop_kernel(payload)
            elif command == "evict_actor":
                actor_ack = await self._evict_scheduler_actor(payload)
            elif command == "begin_drain":
                generation, nonce = _shutdown_coordinates(payload)
                shutdown_ack = {
                    "shutdown_generation": generation,
                    "command_nonce": nonce,
                }
                if generation < self._shutdown_generation:
                    raise RuntimeError("stale shutdown generation")
                if generation == self._shutdown_generation:
                    if nonce != self._drain_command_nonce or not self._draining:
                        raise RuntimeError("shutdown generation has conflicting drain command")
                else:
                    self._shutdown_generation = generation
                    self._drain_command_nonce = nonce
                    self._durability_command_nonce = None
                    self._draining = True
                    self._drain_ready_emitted = False
                    self._durability_ready_emitted = False
                await self._publish_drain_if_ready()
            elif command == "cancel_drain":
                generation, nonce = _shutdown_coordinates(payload)
                shutdown_ack = {
                    "shutdown_generation": generation,
                    "command_nonce": nonce,
                }
                self._verify_shutdown_command(
                    generation,
                    nonce,
                    expected_nonce=self._drain_command_nonce,
                )
                self._draining = False
                self._drain_ready_emitted = False
                self._durability_ready_emitted = False
                self._durability_command_nonce = None
            elif command == "verify_durability":
                generation, nonce = _shutdown_coordinates(payload)
                shutdown_ack = {
                    "shutdown_generation": generation,
                    "command_nonce": nonce,
                }
                if not self._draining:
                    raise RuntimeError("durability verification requires a draining site")
                if generation != self._shutdown_generation:
                    raise RuntimeError("stale shutdown generation")
                if not self._drain_ready_emitted:
                    raise RuntimeError("durability verification requires a completed drain")
                if self._durability_command_nonce not in {None, nonce}:
                    raise RuntimeError("shutdown generation has conflicting durability command")
                self._durability_command_nonce = nonce
                await self.wait_for_idle()
                if not self.storage_available():
                    raise RuntimeError("authoritative storage is unavailable")
                # Successful attempts have already passed their per-object
                # authoritative flush_and_verify barriers.  With the site idle,
                # sync the remaining regional filesystem metadata before the
                # scheduler is allowed to power the instance off.
                await asyncio.to_thread(os.sync)
                regional_durability = self.regional_durability
                if regional_durability is None:
                    if self.site_kind == "autodl":
                        raise RuntimeError("regional persistence is required for AutoDL shutdown")
                else:
                    await asyncio.to_thread(
                        regional_durability.durability_barrier,
                        shutdown_generation=generation,
                        command_nonce=nonce,
                    )
                await self.outbound.publish(
                    "durability_ready",
                    {
                        "site_id": self.site_id,
                        "shutdown_generation": generation,
                        "command_nonce": nonce,
                    },
                )
                self._durability_ready_emitted = True
            elif command == "stop_agent":
                generation, nonce = _shutdown_coordinates(payload)
                shutdown_ack = {
                    "shutdown_generation": generation,
                    "command_nonce": nonce,
                }
                self._verify_shutdown_command(
                    generation,
                    nonce,
                    expected_nonce=self._durability_command_nonce,
                )
                if not self._durability_ready_emitted:
                    raise RuntimeError("stop requires a completed durability barrier")
                await self.shutdown(stop_ray=bool(payload.get("stop_ray", True)))
            else:
                raise ValueError(f"unknown scheduler command: {command}")
        except Exception as exc:
            error = _safe_message(f"{type(exc).__name__}: {exc}")
        await self.outbound.publish(
            "command_ack",
            {
                "site_id": self.site_id,
                "command_id": message.get("message_id"),
                "command": command,
                "accepted": error is None,
                "error": error,
                **shutdown_ack,
                **actor_ack,
            },
        )

    def _verify_shutdown_command(
        self,
        generation: int,
        nonce: str,
        *,
        expected_nonce: str | None,
    ) -> None:
        if generation != self._shutdown_generation:
            raise RuntimeError("stale shutdown generation")
        if expected_nonce is None or nonce != expected_nonce:
            raise RuntimeError("shutdown command nonce is not current")

    async def _evict_scheduler_actor(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, object]:
        """Dispose one exact idle replica under a durable scheduler intent.

        This is deliberately teardown-only. A queued job that needs a different
        model receives a normal fenced lease after this command completes and
        constructs a fresh actor; model identity is never mutated in place.
        """

        (
            eviction_id,
            actor_id,
            replica_key_hash,
            gpu_uuids,
            minimum_idle_seconds,
        ) = _actor_eviction_coordinates(payload)
        evict = getattr(self.backend, "evict_warm_allocation", None)
        evict_replica = getattr(self.backend, "evict_warm_replica", None)
        if not callable(evict) and not callable(evict_replica):
            raise RuntimeError("execution backend cannot dispose warm actors")

        async with self._get_warm_maintenance_lock():
            async with self._active_lock:
                active_gpu_uuids = {
                    gpu_uuid
                    for active in self._active.values()
                    if not getattr(active, "gpu_released", False)
                    for gpu_uuid in active.assignment.lease.gpu_uuids
                }
                active_actor_ids = {
                    active.prepared.actor_id
                    for active in self._active.values()
                    if active.prepared is not None
                }
                kernel_actor_ids = {
                    host.prepared.actor_id for host in getattr(self, "_kernel_hosts", {}).values()
                }
                if (
                    set(gpu_uuids) & active_gpu_uuids
                    or actor_id in active_actor_ids
                    or actor_id in kernel_actor_ids
                ):
                    raise RuntimeError(
                        "actor cannot be evicted while an attempt or kernel references it"
                    )

                intent = await asyncio.to_thread(
                    self.fences.begin_scheduler_actor_eviction,
                    eviction_id,
                    actor_id=actor_id,
                    expected_replica_key_hash=replica_key_hash,
                    expected_gpu_uuids=gpu_uuids,
                    minimum_idle_seconds=minimum_idle_seconds,
                )
                if intent.state == "completed":
                    return {
                        "eviction_id": eviction_id,
                        "actor_id": actor_id,
                        "eviction_state": "completed",
                    }
                model_key = intent.model_key
                allocations = tuple(
                    allocation
                    for allocation in await asyncio.to_thread(self.fences.actor_allocations)
                    if allocation.actor_id == actor_id
                )
                exact_stopping_replica = bool(
                    tuple(allocation.gpu_uuid for allocation in allocations) == gpu_uuids
                    and tuple(allocation.replica_rank for allocation in allocations)
                    == tuple(range(len(allocations)))
                    and all(
                        allocation.model_key == model_key and allocation.state == "stopping"
                        for allocation in allocations
                    )
                )
                if not exact_stopping_replica:
                    raise RuntimeError("durable actor eviction fence is inconsistent")

            # The durable stopping allocation now rejects all new claims. Keep
            # only warm-runtime maintenance serialized during the potentially
            # slow Ray control call; unrelated active work may continue.
            if not all(self.backend_is_live_allocation(item) for item in allocations):
                raise RuntimeError("exact live actor ownership cannot be proven")
            worker_pids: tuple[int, ...] = ()
            try:
                worker_pids = await self._evict_exact_warm_replica(allocations)
            except asyncio.CancelledError:
                # Ambiguous disposal retains the durable stopping fence. Startup
                # reconciliation will reconnect and resume only this actor ID.
                raise
            except Exception:
                with suppress(Exception):
                    if all(self.backend_is_live_allocation(item) for item in allocations):
                        await asyncio.to_thread(
                            self.fences.restore_scheduler_actor_eviction,
                            eviction_id,
                        )
                # If exact liveness cannot be reproved, remaining fenced is
                # safer than advertising a possibly occupied physical GPU.
                raise

            removed = await asyncio.to_thread(
                self.fences.complete_scheduler_actor_eviction,
                eviction_id,
            )
            if self.process_ownership is not None:
                # These PIDs come from the exact owned runtime handle. The
                # scheduler payload and NVML foreign-process list are never
                # accepted as signal or unregister targets.
                for pid in worker_pids:
                    self.process_ownership.unregister(pid)
            if removed != len(allocations):
                raise RuntimeError(
                    "actor stopped but durable allocation completion was not confirmed"
                )
            log.info(
                "scheduler_idle_actor_evicted",
                eviction_id=eviction_id,
                actor_id=actor_id,
                gpu_uuids=list(gpu_uuids),
            )
            return {
                "eviction_id": eviction_id,
                "actor_id": actor_id,
                "eviction_state": "completed",
            }

    def _verify_fencing_token(self, assignment: Assignment) -> None:
        if self.fence_verifier is None:
            return
        if assignment.fencing_token is None:
            raise ValueError("assignment is missing its fencing token")
        verified = self.fence_verifier.verify(
            assignment.fencing_token,
            attempt_id=assignment.attempt_id,
            site_id=self.site_id,
        )
        raw_claims = getattr(verified, "claims", ())
        claim_mappings = tuple(_plain_mapping(claim) for claim in raw_claims)
        verified_claims = tuple(
            (str(claim["gpu_uuid"]), int(claim["epoch"])) for claim in claim_mappings
        )
        expected_claims = tuple(
            (gpu_uuid, int(assignment.lease.fence_epochs[gpu_uuid]))
            for gpu_uuid in assignment.lease.gpu_uuids
        )
        if verified_claims != expected_claims:
            raise ValueError("fencing token GPU claims do not match the assignment")

    async def offer(self, assignment: Assignment) -> bool:
        if self._draining:
            await self._publish_rejection(assignment, "site is draining")
            return False
        await self._reconcile_stale_warm_liveness()
        if _is_private_workload(assignment):
            if assignment.lease.reuse_actor_id is not None:
                await self._publish_rejection(
                    assignment,
                    "private workloads cannot reuse a resident actor",
                )
                return False
            assignment = replace(
                assignment,
                lease=replace(assignment.lease, allow_actor_reuse=False),
            )
        async with self._active_lock:
            if assignment.attempt_id in self._active:
                # Scheduler command replay is idempotent.
                return True
            spec = (JobSpec.model_validate(assignment.raw_spec)
                    if self.input_stager is not None else None)
            if (
                self.input_stager is not None and (spec.assets or spec.source_ref)
                and not self.input_stager.ready(assignment.job_id, spec, assignment.checkpoint_ref)
            ):
                await self._publish_rejection(assignment, "inputs are not staged at this site")
                return False
            if (
                self._finalization_backlog()
                >= self.finalization_backlog_limit
            ):
                await self._publish_rejection(assignment, "finalization backlog is full")
                return False
            if assignment.lease.replace_idle_actors or _requires_slot_eviction(assignment):
                try:
                    await self._evict_private_slot_actors(assignment)
                except Exception as exc:
                    await self._publish_rejection(
                        assignment,
                        _safe_message(f"{type(exc).__name__}: {exc}")
                        or "idle actor eviction failed",
                    )
                    return False
            decision = await asyncio.to_thread(self.admission.admit, assignment.lease)
            if not decision.accepted:
                await self._publish_rejection(
                    assignment, decision.reason or "physical admission rejected"
                )
                return False
            if _is_adapter_training(assignment) and decision.reused_actor_id is not None:
                await asyncio.to_thread(
                    self.admission.release,
                    assignment.lease.lease_id,
                )
                await self._publish_rejection(
                    assignment,
                    "adapter training cannot reuse a resident actor",
                )
                return False
            lease = assignment.lease
            if decision.reused_actor_id is not None and lease.reuse_actor_id is None:
                lease = replace(lease, reuse_actor_id=decision.reused_actor_id)
            assignment = replace(
                assignment,
                lease=lease,
                gpu_indices=dict(decision.gpu_indices),
            )
            sentinel: CudaSentinel | None = None
            if decision.cooperative_pids:
                try:
                    sentinel = await self._handoff_cooperative_processes(assignment, decision)
                except Exception as exc:
                    await asyncio.to_thread(self.admission.release, assignment.lease.lease_id)
                    await self._publish_rejection(
                        assignment,
                        _safe_message(f"{type(exc).__name__}: {exc}")
                        or "cooperative handoff failed",
                    )
                    return False
            # The scheduler ACK is emitted only after its transaction commits.
            # Work must not start on a merely spooled or ambiguously delivered
            # acceptance because the scheduler might reassign the same GPU.
            try:
                await self.outbound.publish_confirmed(
                    "assignment_accepted",
                    {
                        "site_id": self.site_id,
                        "job_id": assignment.job_id,
                        "attempt_id": assignment.attempt_id,
                        "fencing_token": assignment.fencing_token or "",
                        "lease_id": assignment.lease.lease_id,
                        "gpu_uuids": list(assignment.lease.gpu_uuids),
                        "fence_epochs": dict(assignment.lease.fence_epochs),
                        "reused_actor_id": decision.reused_actor_id,
                    },
                )
                await asyncio.to_thread(self.admission.renew, assignment.lease)
            except asyncio.CancelledError:
                if sentinel is not None:
                    await sentinel.stop()
                    if self.process_ownership is not None:
                        for pid in sentinel.pids:
                            self.process_ownership.unregister(pid)
                await asyncio.to_thread(self.admission.release, assignment.lease.lease_id)
                raise
            except Exception:
                if sentinel is not None:
                    await sentinel.stop()
                    if self.process_ownership is not None:
                        for pid in sentinel.pids:
                            self.process_ownership.unregister(pid)
                await asyncio.to_thread(self.admission.release, assignment.lease.lease_id)
                return False
            task = asyncio.create_task(
                self._run_attempt(assignment, sentinel),
                name=f"topodrome-attempt-{assignment.attempt_id}",
            )
            self._active[assignment.attempt_id] = _ActiveAttempt(
                assignment, task, sentinel=sentinel
            )
            if self.input_stager is not None:
                self.input_stager.resolver.pin(
                    assignment.attempt_id, spec.assets.values()
                )
                self.input_stager.consumed(assignment.job_id)
        return True

    async def _evict_private_slot_actors(
        self,
        assignment: Assignment,
    ) -> tuple[str, ...]:
        """Batch-fence exact idle actors, then release every physical member."""

        if _parallelism_mode(assignment) == "model_sharded" and assignment.lease.allow_actor_reuse:
            selected_allocations = tuple(
                await asyncio.gather(
                    *(
                        asyncio.to_thread(self.fences.actor_allocation, gpu_uuid)
                        for gpu_uuid in assignment.lease.gpu_uuids
                    )
                )
            )
            if all(allocation is not None for allocation in selected_allocations):
                exact = tuple(
                    allocation for allocation in selected_allocations if allocation is not None
                )
                if (
                    len({allocation.actor_id for allocation in exact}) == 1
                    and tuple(allocation.replica_rank for allocation in exact)
                    == tuple(range(len(exact)))
                    and all(allocation.state == "idle" for allocation in exact)
                    and all(
                        allocation.model_key == assignment.model_key
                        and self.backend_is_live_allocation(allocation)
                        for allocation in exact
                    )
                ):
                    return ()

        resident_allocations = await asyncio.to_thread(self.fences.actor_allocations)
        by_gpu = {allocation.gpu_uuid: allocation for allocation in resident_allocations}
        grouped: dict[str, tuple[ActorAllocation, ...]] = {}
        for gpu_uuid in assignment.lease.gpu_uuids:
            allocation = by_gpu.get(gpu_uuid)
            if allocation is None:
                continue
            if allocation.state != "idle":
                raise RuntimeError(f"GPU {gpu_uuid} has a non-idle model actor")
            grouped.setdefault(
                allocation.actor_id,
                tuple(
                    member
                    for member in resident_allocations
                    if member.actor_id == allocation.actor_id
                ),
            )

        actor_groups = tuple(grouped.values())
        target_gpu_uuids = set(assignment.lease.gpu_uuids)
        if any(
            any(member.gpu_uuid not in target_gpu_uuids for member in group)
            for group in actor_groups
        ):
            raise RuntimeError("assignment does not lease every member of a resident actor replica")
        evict = getattr(self.backend, "evict_warm_allocation", None)
        evict_replica = getattr(self.backend, "evict_warm_replica", None)
        if actor_groups and not callable(evict) and not callable(evict_replica):
            raise RuntimeError("execution backend cannot dispose warm actors")
        for group in actor_groups:
            exact_idle_replica = bool(
                group
                and tuple(item.replica_rank for item in group) == tuple(range(len(group)))
                and len({item.model_key for item in group}) == 1
                and len({item.gpu_uuid for item in group}) == len(group)
                and all(
                    item.state == "idle" and self.backend_is_live_allocation(item) for item in group
                )
            )
            if not exact_idle_replica:
                raise RuntimeError(
                    f"actor {group[0].actor_id if group else '<unknown>'} "
                    "is not a complete live idle replica"
                )

        began = await asyncio.to_thread(
            self.fences.begin_idle_actor_eviction_batch,
            actor_groups,
        )
        if not began:
            raise RuntimeError("idle actor set changed during atomic eviction fencing")

        evicted_gpu_uuids: list[str] = []
        worker_pids: list[int] = []
        completed_actor_ids: set[str] = set()
        try:
            for group in actor_groups:
                actor_id = group[0].actor_id
                actor_members = tuple(member.gpu_uuid for member in group)
                worker_pids.extend(await self._evict_exact_warm_replica(group))
                removed = await asyncio.to_thread(
                    self.fences.complete_idle_actor_eviction,
                    actor_id,
                    expected_gpu_uuids=actor_members,
                )
                if removed != len(group):
                    raise RuntimeError("stopped actor allocation could not be durably completed")
                completed_actor_ids.add(actor_id)
                evicted_gpu_uuids.extend(actor_members)
            released = tuple(dict.fromkeys(evicted_gpu_uuids))
            if released:
                await self._wait_for_evicted_slot_release(assignment, released)
            return released
        except asyncio.CancelledError:
            # A canceled Ray control call has an ambiguous physical outcome.
            # Keep every not-yet-completed source fenced as stopping.
            raise
        except Exception:
            for group in actor_groups:
                actor_id = group[0].actor_id
                if actor_id in completed_actor_ids:
                    continue
                with suppress(Exception):
                    if all(self.backend_is_live_allocation(item) for item in group):
                        await asyncio.to_thread(
                            self.fences.restore_idle_actor_eviction,
                            actor_id,
                            expected_gpu_uuids=tuple(item.gpu_uuid for item in group),
                        )
                # Ambiguous teardown remains fenced as stopping. It is
                # reconciled only against the exact owned runtime later.
            raise
        finally:
            # Keep exact worker identities registered while NVML drains their
            # CUDA contexts. This lets the re-probe distinguish teardown lag
            # from a genuinely new foreign process even when the Ray head is
            # adopted rather than started by this agent generation.
            if self.process_ownership is not None:
                for pid in worker_pids:
                    self.process_ownership.unregister(pid)

    async def _wait_for_evicted_slot_release(
        self,
        assignment: Assignment,
        gpu_uuids: tuple[str, ...],
    ) -> None:
        """Let NVML observe exact actor teardown before fresh admission.

        Ray can prove an exact actor dead slightly before the NVIDIA driver
        removes its compute-process/VRAM accounting.  Rejecting during that
        short interval creates a needless scheduler attempt and advances the
        gang fence even though the site just made the requested slots free.
        This bounded wait applies only after Topodrome confirmed and recorded
        an exact warm-actor eviction.  Foreign/cooperative activity remains an
        immediate admission decision and is never signalled here.
        """

        loop = asyncio.get_running_loop()
        lease_remaining = (
            assignment.lease.expires_at
            - datetime.now(tz=UTC).timestamp()
            - _POST_EVICTION_OFFER_RESERVE_SECONDS
        )
        deadline = loop.time() + min(
            _POST_EVICTION_RELEASE_WAIT_SECONDS,
            max(0.0, lease_remaining),
        )
        while await asyncio.to_thread(
            self._evicted_slot_release_pending,
            assignment.lease,
            gpu_uuids,
        ):
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(_POST_EVICTION_RELEASE_POLL_SECONDS, remaining))

    def _evicted_slot_release_pending(
        self,
        lease: LeaseRequest,
        gpu_uuids: tuple[str, ...],
    ) -> bool:
        snapshots = {snapshot.uuid: snapshot for snapshot in self.inventory.sample()}
        selected: list[GPUSnapshot] = []
        for gpu_uuid in gpu_uuids:
            snapshot = snapshots.get(gpu_uuid)
            if snapshot is None or snapshot.externally_reserved:
                return False
            # A new external process is not residual state from the actor we
            # evicted.  Let normal fail-closed/cooperative admission handle it
            # immediately instead of treating it as teardown lag.
            if snapshot.external_processes:
                return False
            selected.append(snapshot)

        required = lease.min_free_vram_bytes + self.admission.safety_margin_bytes
        return any(
            snapshot.topodrome_processes
            or snapshot.free_memory_bytes < required
            or (
                snapshot.utilization_percent > self.admission.max_unattributed_utilization_percent
                and not self.admission.allow_foreign_processes_when_capacity_sufficient
            )
            for snapshot in selected
        )

    def backend_is_live_allocation(self, allocation: ActorAllocation) -> bool:
        validate = getattr(self.backend, "is_warm_allocation", None)
        return bool(callable(validate) and validate(allocation))

    async def _evict_exact_warm_replica(
        self,
        allocations: Sequence[ActorAllocation],
    ) -> tuple[int, ...]:
        """Use whole-replica runtime control, with a legacy test-backend fallback."""

        ordered = tuple(allocations)
        evict_replica = getattr(self.backend, "evict_warm_replica", None)
        if callable(evict_replica):
            return tuple(await evict_replica(ordered))
        evict_one = getattr(self.backend, "evict_warm_allocation", None)
        if not callable(evict_one) or not ordered:
            raise RuntimeError("execution backend cannot dispose warm actors")
        return tuple(await evict_one(ordered[0]))

    async def _handoff_cooperative_processes(
        self, assignment: Assignment, admission: AdmissionDecision
    ) -> CudaSentinel:
        if self.sentinel_factory is None or self.process_ownership is None:
            raise RuntimeError(
                "cooperative admission requires a CUDA sentinel and ownership registry"
            )
        sentinel = await self.sentinel_factory.start(admission.gpu_indices)
        try:
            if not sentinel.is_alive():
                raise RuntimeError("CUDA sentinel exited before cooperative handoff")
            for pid in sentinel.pids:
                self.process_ownership.register(
                    pid,
                    lease_id=assignment.lease.lease_id,
                    actor_id=f"sentinel:{assignment.lease.lease_id}",
                )
            expected_pids = frozenset(
                pid for pids in admission.cooperative_pids.values() for pid in pids
            )
            reclaimed = await self.cooperative_exit.wait(
                assignment.lease.gpu_uuids,
                expected_pids=expected_pids,
            )
            if not reclaimed.exited:
                detail = (
                    reclaimed.remaining_pids
                    or reclaimed.unexpected_foreign_pids
                    or reclaimed.missing_gpu_uuids
                )
                raise RuntimeError(f"cooperative scavenger did not yield: {detail}")
            if not sentinel.is_alive():
                raise RuntimeError("CUDA sentinel exited during cooperative handoff")
            capacity = await asyncio.to_thread(
                self.admission.verify_reserved_capacity,
                assignment.lease,
                allowed_owned_pids=frozenset(sentinel.pids),
            )
            if not capacity.accepted:
                raise RuntimeError(capacity.reason or "insufficient capacity after handoff")
            return sentinel
        except BaseException:
            await sentinel.stop()
            for pid in sentinel.pids:
                self.process_ownership.unregister(pid)
            raise

    async def _run_attempt(self, assignment: Assignment, sentinel: CudaSentinel | None) -> None:
        prepared: PreparedRuntime | None = None
        outcome: ExecutionOutcome | None = None
        workspace: AttemptWorkspace | None = None
        terminal: _PendingTerminalState | None = None
        cleanup_safe = True
        log_stop = asyncio.Event()
        log_stream = None
        try:
            await self._publish_attempt_state(assignment, "starting")
            async with self._active_lock:
                active = self._active.get(assignment.attempt_id)
                preempted_before_start = bool(active and active.preempt_requested)
            if preempted_before_start:
                raise RuntimeError("preemption requested before runtime preparation")
            if await self._local_lease_expired(assignment.attempt_id):
                raise _LocalLeaseExpired("local lease expired before workspace preparation")
            workspace = await asyncio.to_thread(
                self.workspaces.prepare, assignment.job_id, assignment.attempt_id
            )
            self._active[assignment.attempt_id].workspace = workspace
            reporter = (
                TransferLogReporter(workspace.log_dir / "attempt.log")
                if callable(getattr(self.backend, "verify_attempt_log", None))
                else lambda *_: None
            )
            if callable(getattr(self.outbound, "publish_live_request", None)):
                self._active[assignment.attempt_id].log_stream_started = True
                log_stream = asyncio.create_task(self._stream_attempt_log_safely(
                    assignment.attempt_id, workspace.log_dir / "attempt.log", log_stop
                ))
            stage_inputs = getattr(self.backend, "stage_inputs", None)
            if callable(stage_inputs):
                # A canceled await does not stop work already running through
                # asyncio.to_thread. Keep the lease and staging activity fenced
                # until that operation really exits, then propagate cancellation.
                reporter("downloading")
                with cancellable_transfers(self._active[assignment.attempt_id].transfer_cancel):
                    with observe_transfers(reporter):
                        await _finish_input_staging(stage_inputs(assignment, workspace))
            if await self._local_lease_expired(assignment.attempt_id):
                raise _LocalLeaseExpired("local lease expired while staging inputs")
            async with self._active_lock:
                active = self._active.get(assignment.attempt_id)
                canceled_after_staging = bool(active and active.cancel_requested)
                preempted_after_staging = bool(active and active.preempt_requested)
            if canceled_after_staging:
                outcome = ExecutionOutcome(False, retryable=False, keep_warm=False)
                terminal = _PendingTerminalState(
                    "canceled",
                    reason_code="cancel_requested",
                    message="canceled while staging inputs",
                )
                return
            if preempted_after_staging:
                checkpoint_ref = await self._latest_checkpoint(assignment, workspace)
                await self._publish_attempt_state(assignment, "checkpointing")
                outcome = ExecutionOutcome(
                    False,
                    retryable=True,
                    keep_warm=False,
                    metadata={"checkpoint_ref": checkpoint_ref},
                )
                terminal = _PendingTerminalState(
                    "preempted",
                    reason_code="preempted",
                    message="preempted while staging inputs",
                    checkpoint_ref=checkpoint_ref,
                )
                return
            reporter("loading")
            preparation_started = time.monotonic()
            with cancellable_transfers(self._active[assignment.attempt_id].transfer_cancel):
                with observe_transfers(reporter):
                    prepared = await self.backend.prepare(assignment, workspace)
            startup_seconds = time.monotonic() - preparation_started
            # Publish the handle into active bookkeeping immediately. All
            # validation and fence work below can now be preempted through the
            # exact prepared runtime rather than leaving it untracked.
            async with self._active_lock:
                active = self._active.get(assignment.attempt_id)
                if active is not None:
                    active.prepared = prepared
            kernel_mode = _kernel_mode(assignment)
            if (
                kernel_mode is KernelMode.RAW_MODEL or _is_adapter_training(assignment)
            ) and prepared.reused:
                raise RuntimeError("private workloads require a fresh disposable actor")
            if await self._local_lease_expired(assignment.attempt_id):
                raise _LocalLeaseExpired("local lease expired before execution")
            if self.process_ownership is not None:
                for pid in prepared.process_ids:
                    self.process_ownership.register(
                        pid,
                        lease_id=assignment.lease.lease_id,
                        actor_id=prepared.actor_id,
                    )
            if prepared.model_key != assignment.model_key:
                raise RuntimeError("runtime prepared an incompatible model identity")
            if assignment.lease.reuse_actor_id is not None:
                if prepared.actor_id != assignment.lease.reuse_actor_id or not prepared.reused:
                    raise RuntimeError("runtime did not reconnect to the leased warm actor")
            else:
                self.fences.reserve_actor_group(
                    gpu_uuids=assignment.lease.gpu_uuids,
                    actor_id=prepared.actor_id,
                    model_key=prepared.model_key,
                    state="busy",
                )
            if sentinel is not None:
                await sentinel.stop()
                if self.process_ownership is not None:
                    for pid in sentinel.pids:
                        self.process_ownership.unregister(pid)
                sentinel = None
            async with self._active_lock:
                active = self._active.get(assignment.attempt_id)
                lease_expired = bool(active and active.lease_expired)
                canceled = bool(active and active.cancel_requested)
                preempted = bool(active and active.preempt_requested)
            if lease_expired:
                raise _LocalLeaseExpired("local lease expired while preparing execution")
            if canceled:
                # Cancellation can race a slow model constructor before the
                # backend has a handle it can interrupt.  The handle is now
                # exact and fenced, so skip user code and let ``finally``
                # dispose it before releasing either lease authority.
                outcome = ExecutionOutcome(
                    False,
                    retryable=False,
                    keep_warm=False,
                )
                terminal = _PendingTerminalState(
                    "canceled",
                    reason_code="cancel_requested",
                    message="canceled before runtime invocation",
                )
                return
            if preempted:
                # The pressure/preemption grace may expire while model loading
                # is still inside Ray's constructor.  Never announce RUNNING
                # or invoke the workload after that intent has been recorded.
                checkpoint_ref = await self._latest_checkpoint(assignment, workspace)
                await self._publish_attempt_state(assignment, "checkpointing")
                outcome = ExecutionOutcome(
                    False,
                    retryable=True,
                    keep_warm=False,
                    metadata={"checkpoint_ref": checkpoint_ref},
                )
                terminal = _PendingTerminalState(
                    "preempted",
                    reason_code="preempted",
                    message="preempted before runtime invocation",
                    checkpoint_ref=checkpoint_ref,
                )
                return
            reporter("computing")
            await self._publish_attempt_state(
                assignment, "running", timings={
                    "startup_seconds": None if prepared.reused else min(startup_seconds, 86400),
                },
            )
            self._active[assignment.attempt_id].compute_started_at = time.monotonic()
            if await self._local_lease_expired(assignment.attempt_id):
                raise _LocalLeaseExpired("local lease expired before runtime invocation")
            if kernel_mode is None:
                outcome = await self._execute_batch_with_live_logs(
                    prepared,
                    assignment,
                    workspace,
                )
            else:
                outcome = await self._serve_interactive(
                    prepared,
                    assignment,
                    workspace,
                    mode=kernel_mode,
                )
            if _is_adapter_training(assignment):
                # Adapter training may mutate module state in memory even when
                # the immutable base files remain untouched.  Never advertise
                # that process as a reusable model service.
                outcome = replace(outcome, keep_warm=False)
            if self._active[assignment.attempt_id].gpu_released:
                prepared = None
            if await self._local_lease_expired(assignment.attempt_id):
                raise _LocalLeaseExpired("local lease expired during execution")
            if outcome.succeeded and not outcome.artifacts_verified:
                raise RuntimeError(
                    "execution backend returned success before required artifacts "
                    "were flushed and verified"
                )
            if outcome.dirty and prepared is not None:
                self.fences.set_actor_state(prepared.actor_id, "dirty")
            async with self._active_lock:
                active = self._active.get(assignment.attempt_id)
                canceled = bool(active and active.cancel_requested)
                preempted = bool(active and active.preempt_requested)
            if canceled:
                terminal = _PendingTerminalState(
                    "canceled",
                    reason_code="cancel_requested",
                    message=outcome.message,
                )
            elif preempted:
                outcome = replace(outcome, keep_warm=False)
                await self._publish_attempt_state(assignment, "checkpointing")
                terminal = _PendingTerminalState(
                    "preempted",
                    reason_code="preempted",
                    message=outcome.message,
                    checkpoint_ref=_optional_metadata_text(outcome.metadata, "checkpoint_ref"),
                )
            elif outcome.succeeded:
                await self._publish_attempt_state(assignment, "committing")
                terminal = _PendingTerminalState(
                    "succeeded",
                    message=outcome.message,
                    artifacts=tuple(_artifact_payload(outcome.metadata)),
                )
            else:
                checkpoint_ref = await self._latest_checkpoint(assignment, workspace)
                terminal = _PendingTerminalState(
                    "failed",
                    reason_code=(
                        "retryable_runtime_error" if outcome.retryable else "runtime_error"
                    ),
                    message=outcome.message,
                    checkpoint_ref=checkpoint_ref,
                )
        except asyncio.CancelledError:
            if self._active[assignment.attempt_id].gpu_released:
                prepared = None
            if prepared is not None:
                await self.backend.cancel(assignment.attempt_id)
            raise
        except _LocalLeaseExpired as exc:
            if self._active[assignment.attempt_id].gpu_released:
                prepared = None
            if prepared is not None:
                self.fences.set_actor_state(prepared.actor_id, "dirty")
            terminal = _PendingTerminalState(
                "lost",
                reason_code="lease_expired",
                message=str(exc),
            )
            outcome = ExecutionOutcome(
                False,
                retryable=True,
                keep_warm=False,
                dirty=prepared is not None,
            )
        except Exception as exc:
            if self._active[assignment.attempt_id].gpu_released:
                prepared = None
            cleanup_safe = not bool(getattr(exc, "cleanup_unconfirmed", False))
            if prepared is not None:
                self.fences.set_actor_state(prepared.actor_id, "dirty")
            async with self._active_lock:
                active = self._active.get(assignment.attempt_id)
                explicitly_canceled = bool(
                    active and getattr(active, "explicit_cancel_requested", False)
                )
                preempted = bool(active and active.preempt_requested)
            if explicitly_canceled:
                terminal = _PendingTerminalState(
                    "canceled",
                    reason_code="cancel_requested",
                    message=f"{type(exc).__name__}: runtime stopped after cancellation grace",
                )
                outcome = ExecutionOutcome(
                    False,
                    retryable=False,
                    keep_warm=False,
                    dirty=prepared is not None,
                )
            elif preempted:
                checkpoint_ref = await self._latest_checkpoint(assignment, workspace)
                await self._publish_attempt_state(assignment, "checkpointing")
                terminal = _PendingTerminalState(
                    "preempted",
                    reason_code="preempted",
                    message=f"{type(exc).__name__}: runtime stopped after checkpoint grace",
                    checkpoint_ref=checkpoint_ref,
                )
                outcome = ExecutionOutcome(
                    False,
                    retryable=True,
                    keep_warm=False,
                    dirty=prepared is not None,
                    metadata={"checkpoint_ref": checkpoint_ref},
                )
            else:
                checkpoint_ref = await self._latest_checkpoint(assignment, workspace)
                terminal = _PendingTerminalState(
                    "failed",
                    reason_code="runtime_error",
                    message=f"{type(exc).__name__}: {exc}",
                    checkpoint_ref=checkpoint_ref,
                )
                outcome = ExecutionOutcome(False, retryable=True, dirty=prepared is not None)
        finally:
            if log_stream is not None:
                log_stop.set()
                try:
                    await asyncio.wait_for(
                        asyncio.shield(log_stream), _LIVE_LOG_FINAL_DRAIN_SECONDS
                    )
                except (TimeoutError, asyncio.CancelledError):
                    log_stream.cancel()
                    await asyncio.gather(log_stream, return_exceptions=True)
            if self._active[assignment.attempt_id].gpu_released:
                prepared = None
            if terminal is not None:
                async with self._active_lock:
                    current = self._active.get(assignment.attempt_id)
                    explicitly_canceled = bool(current and current.explicit_cancel_requested)
                if explicitly_canceled and terminal.state in {"failed", "preempted", "lost"}:
                    terminal = replace(
                        terminal,
                        state="canceled",
                        reason_code="cancel_requested",
                        checkpoint_ref=None,
                    )
                if workspace is None:
                    # Failure before workspace creation cannot have launched
                    # user code or created an attempt log.
                    terminal = replace(terminal, log=None, log_complete=True)
                else:
                    terminal = await self._complete_terminal_log(
                        assignment,
                        workspace,
                        terminal,
                        outcome,
                    )
            if sentinel is not None:
                await sentinel.stop()
                if self.process_ownership is not None:
                    for pid in sentinel.pids:
                        self.process_ownership.unregister(pid)
            # A dirty/private/group runtime is disposed and confirmed while
            # both local and scheduler claims still fence its complete gang.
            # Only then may the local claim be released and the terminal event
            # release PostgreSQL ownership. Clean reusable actors stay warm.
            should_evict = prepared is not None and (
                outcome is None or outcome.dirty or not outcome.keep_warm
            )
            evicted = False
            if cleanup_safe and should_evict and prepared is not None:
                try:
                    self.fences.set_actor_state(prepared.actor_id, "stopping")
                except Exception as exc:
                    # Still stop the exact runtime below, but retain both
                    # lease authorities because persistent allocation state
                    # could not be advanced atomically.
                    cleanup_safe = False
                    with suppress(Exception):
                        await self.outbound.publish(
                            "log_chunk",
                            {
                                "site_id": self.site_id,
                                "job_id": assignment.job_id,
                                "attempt_id": assignment.attempt_id,
                                "level": "error",
                                "reason_code": "allocation_stopping_fence_failed",
                                "message": _safe_message(f"{type(exc).__name__}: {exc}"),
                            },
                        )
                try:
                    await self.backend.evict(prepared)
                    evicted = True
                except Exception as exc:
                    cleanup_safe = False
                    with suppress(Exception):
                        await self.outbound.publish(
                            "log_chunk",
                            {
                                "site_id": self.site_id,
                                "job_id": assignment.job_id,
                                "attempt_id": assignment.attempt_id,
                                "level": "error",
                                "reason_code": "actor_eviction_unconfirmed",
                                "message": _safe_message(f"{type(exc).__name__}: {exc}"),
                            },
                        )
                if evicted and self.process_ownership is not None:
                    for pid in prepared.process_ids:
                        self.process_ownership.unregister(pid)

            lease_released = self._active[assignment.attempt_id].gpu_released
            if cleanup_safe and not lease_released:
                try:
                    await asyncio.to_thread(
                        self.admission.release,
                        assignment.lease.lease_id,
                    )
                    lease_released = True
                except Exception as exc:
                    if prepared is not None:
                        self.fences.set_actor_state(prepared.actor_id, "dirty")
                    await self.outbound.publish(
                        "log_chunk",
                        {
                            "site_id": self.site_id,
                            "job_id": assignment.job_id,
                            "attempt_id": assignment.attempt_id,
                            "level": "error",
                            "reason_code": "local_lease_release_failed",
                            "message": _safe_message(f"{type(exc).__name__}: {exc}"),
                        },
                    )
            try:
                if lease_released and evicted and prepared is not None:
                    try:
                        await asyncio.to_thread(
                            self.fences.evict_actor,
                            prepared.actor_id,
                        )
                    except Exception as exc:
                        # The actor is confirmed gone. Keeping a conservative
                        # stopping allocation row safely blocks admission until
                        # the next reconciliation repairs bookkeeping.
                        await self.outbound.publish(
                            "log_chunk",
                            {
                                "site_id": self.site_id,
                                "job_id": assignment.job_id,
                                "attempt_id": assignment.attempt_id,
                                "level": "error",
                                "reason_code": "allocation_release_failed",
                                "message": _safe_message(f"{type(exc).__name__}: {exc}"),
                            },
                        )
                elif lease_released and prepared is not None:
                    if self.process_ownership is not None:
                        for pid in prepared.process_ids:
                            self.process_ownership.register(pid, actor_id=prepared.actor_id)

                if lease_released and terminal is not None:
                    try:
                        await self._publish_attempt_state(
                            assignment,
                            terminal.state,
                            reason_code=terminal.reason_code,
                            message=terminal.message,
                            artifacts=terminal.artifacts,
                            checkpoint_ref=terminal.checkpoint_ref,
                            attempt_log=terminal.log,
                            log_complete=terminal.log_complete,
                        )
                        if self.finalization_journal is not None:
                            self.finalization_journal.complete(assignment.attempt_id)
                    except Exception as exc:
                        # No terminal ACK means scheduler claims remain fenced
                        # until its normal retry/reconciliation path succeeds.
                        log.error(
                            "terminal_attempt_report_failed",
                            attempt_id=assignment.attempt_id,
                            error=_safe_message(f"{type(exc).__name__}: {exc}"),
                        )
            finally:
                if workspace is not None and not (
                    self.finalization_journal is not None
                    and self.finalization_journal.contains(assignment.attempt_id)
                ):
                    try:
                        await asyncio.to_thread(self.workspaces.cleanup_scratch, workspace)
                    except Exception as exc:
                        try:
                            await self.outbound.publish(
                                "log_chunk",
                                {
                                    "site_id": self.site_id,
                                    "job_id": assignment.job_id,
                                    "attempt_id": assignment.attempt_id,
                                    "level": "error",
                                    "reason_code": "scratch_cleanup_failed",
                                    "message": _safe_message(f"{type(exc).__name__}: {exc}"),
                                },
                            )
                        except Exception:
                            log.exception(
                                "scratch_cleanup_report_failed",
                                attempt_id=assignment.attempt_id,
                            )
                async with self._active_lock:
                    self._active.pop(assignment.attempt_id, None)
                if self.input_stager is not None:
                    self.input_stager.resolver.unpin(assignment.attempt_id)
                await self._publish_drain_if_ready()

    async def _complete_terminal_log(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
        terminal: _PendingTerminalState,
        outcome: ExecutionOutcome | None,
    ) -> _PendingTerminalState | None:
        raw_log = outcome.metadata.get("log") if outcome is not None else None
        if raw_log is None and getattr(self, "terminal_log_journal", None) is not None:
            # Completion of the compute lease must not wait on remote storage.
            # The fsynced log remains local, and this private outbox survives a
            # restart until authoritative verification and its scheduler ACK.
            self.terminal_log_journal.save({
                "job_id": assignment.job_id, "attempt_id": assignment.attempt_id,
            })
            return replace(terminal, log=None, log_complete=False)
        verify = getattr(self.backend, "verify_attempt_log", None)
        delay = _TERMINAL_LOG_VERIFY_INITIAL_RETRY_SECONDS
        while True:
            try:
                if raw_log is None:
                    if callable(verify):
                        raw_log = await verify(assignment, workspace)
                    elif (workspace.log_dir / "attempt.log").is_file():
                        raise RuntimeError("execution backend cannot verify its attempt log")
                descriptor = (
                    None
                    if raw_log is None
                    else AttemptLogDescriptor.model_validate(raw_log).model_dump(mode="json")
                )
                return replace(
                    terminal,
                    log=descriptor,
                    log_complete=True,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Never publish a terminal runtime report with indeterminate
                # log completion: it would release the scheduler lease while
                # leaving `submit --wait` unable to distinguish a late flush
                # from permanent absence. Retry storage verification while the
                # fenced attempt is alive. Agent shutdown suppresses the
                # terminal report and lets ordinary lease reaping take over.
                log.warning(
                    "terminal_attempt_log_verification_failed",
                    attempt_id=assignment.attempt_id,
                    error_type=type(exc).__name__,
                )
                raw_log = None
                await _wait_or_stop(self._stop, delay)
                if self._stop.is_set():
                    return None
                delay = min(delay * 2, _TERMINAL_LOG_VERIFY_MAX_RETRY_SECONDS)

    async def _execute_batch_with_live_logs(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> ExecutionOutcome:
        """Run one batch attempt while forwarding its growing log out of site."""

        async def execute() -> ExecutionOutcome:
            split_execute = getattr(self.backend, "execute_with_finalization", None)
            if callable(split_execute):
                return await split_execute(
                    prepared,
                    assignment,
                    workspace,
                    lambda reusable: self._release_compute(assignment, prepared, reusable),
                )
            return await self.backend.execute(prepared, assignment, workspace)

        publish_live_request = getattr(self.outbound, "publish_live_request", None)
        if not callable(publish_live_request) or getattr(
            self._active[assignment.attempt_id], "log_stream_started", False
        ):
            return await execute()

        stop = asyncio.Event()
        stream = asyncio.create_task(
            self._stream_attempt_log_safely(
                assignment.attempt_id,
                workspace.log_dir / "attempt.log",
                stop,
            ),
            name=f"topodrome-live-log-{assignment.attempt_id}",
        )
        try:
            return await execute()
        finally:
            stop.set()
            try:
                await asyncio.wait_for(
                    asyncio.shield(stream),
                    timeout=_LIVE_LOG_FINAL_DRAIN_SECONDS,
                )
            except TimeoutError:
                stream.cancel()
                await asyncio.gather(stream, return_exceptions=True)
            except asyncio.CancelledError:
                stream.cancel()
                await asyncio.gather(stream, return_exceptions=True)
                raise

    async def _release_compute(
        self,
        assignment: Assignment,
        prepared: PreparedRuntime,
        reusable: bool,
    ) -> bool:
        """Dispose or detach the exact idle runtime before releasing its gang.

        Local ownership is released first. Central claims remain conservative
        until the ordered, durable COMMITTING report arrives. The attempt stays
        active and renewable for logs, retry fencing and the shutdown barrier.
        """
        async with self._active_lock:
            active = self._active[assignment.attempt_id]
            if active.cancel_requested or active.preempt_requested or active.lease_expired:
                raise RuntimeError("compute completed after a stop request")
            reusable = reusable and not _is_private_workload(assignment)
            if not reusable:
                self.fences.set_actor_state(prepared.actor_id, "stopping")
                await self.backend.evict(prepared)
            await asyncio.to_thread(self.admission.release, assignment.lease.lease_id)
            # From this point no terminal/error/cancel path may affect the old
            # actor. It can be leased again as soon as central claims are freed.
            active.gpu_released = True
            active.prepared = None
            if not reusable:
                await asyncio.to_thread(self.fences.evict_actor, prepared.actor_id)
            if self.process_ownership is not None:
                for pid in prepared.process_ids:
                    if reusable:
                        self.process_ownership.register(pid, actor_id=prepared.actor_id)
                    else:
                        self.process_ownership.unregister(pid)
        active.finalization_started_at = time.monotonic()
        await self._publish_attempt_state(
            assignment, "committing", gpu_released=True, timings={
                "compute_seconds": (active.finalization_started_at - active.compute_started_at
                                    if active.compute_started_at is not None else None),
            },
        )
        return True

    async def _stream_attempt_log_safely(
        self,
        attempt_id: str,
        path: Path,
        stop: asyncio.Event,
    ) -> None:
        try:
            await stream_attempt_log(
                path,
                lambda generation, offset, data: self._publish_live_log_bytes(
                    attempt_id,
                    generation,
                    offset,
                    data,
                ),
                stop,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Live output is advisory; authoritative terminal log verification
            # and the attempt itself must not fail because this side channel did.
            log.warning(
                "attempt_live_log_stream_failed",
                attempt_id=attempt_id,
                error_type=type(exc).__name__,
            )

    async def _publish_live_log_bytes(
        self,
        attempt_id: str,
        generation: int,
        offset: int,
        data: bytes,
    ) -> LogChunkAcknowledgement | None:
        async with self._active_lock:
            active = self._active.get(attempt_id)
        if active is None:
            return None
        # Snapshot the current rotated token without holding the safety-critical
        # fencing lock across a best-effort network send. A concurrent rotation
        # makes the chunk stale; the scheduler ACKs it without ingesting it and
        # the unchanged byte offset is retried with the new token.
        async with active.fencing_lock:
            if active.scheduler_terminal or active.renewal_pending:
                return None
            assignment = active.assignment
            token = assignment.fencing_token
        if token is None:
            return None
        publish = getattr(self.outbound, "publish_live_request", None)
        if not callable(publish):
            return None
        acknowledgement = await publish(
            "attempt_log_chunk",
            {
                "site_id": self.site_id,
                "job_id": assignment.job_id,
                "attempt_id": assignment.attempt_id,
                "fencing_token": token,
                "generation": generation,
                "offset": offset,
                "data_base64": base64.b64encode(data).decode("ascii"),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )
        if acknowledgement is None:
            return None
        raw = acknowledgement.get("attempt_log_chunk")
        if not isinstance(raw, Mapping) or raw.get("stale") is True:
            return None
        accepted = raw.get("accepted")
        next_generation = raw.get("generation")
        next_offset = raw.get("next_offset")
        reset = raw.get("reset", False)
        stop = raw.get("stop", False)
        if (
            not isinstance(accepted, bool)
            or isinstance(next_generation, bool)
            or not isinstance(next_generation, int)
            or isinstance(next_offset, bool)
            or not isinstance(next_offset, int)
            or not isinstance(reset, bool)
            or not isinstance(stop, bool)
        ):
            return None
        return LogChunkAcknowledgement(
            accepted=accepted,
            generation=next_generation,
            next_offset=next_offset,
            reset=reset,
            stop=stop,
        )

    async def _serve_interactive(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
        *,
        mode: KernelMode,
    ) -> ExecutionOutcome:
        session_id = _required_kernel_value(assignment, "session_id")
        relay_id = _required_kernel_value(assignment, "relay_id")
        runtime_id = f"kernel-runtime:{assignment.attempt_id}"
        session_lease = _session_lease(
            assignment,
            prepared,
            session_id=session_id,
            mode=mode,
        )
        self._session_leases.reserve(
            session_id,
            mode,
            session_lease,
            requested_gpus=len(assignment.lease.gpu_uuids),
        )
        host = _InteractiveHost(
            assignment=assignment,
            prepared=prepared,
            workspace=workspace,
            mode=mode,
            runtime_id=runtime_id,
            sessions={session_id: relay_id},
        )
        async with self._active_lock:
            if assignment.attempt_id in self._kernel_hosts:
                self._session_leases.release(session_id)
                raise RuntimeError("interactive attempt already has a kernel host")
            self._kernel_hosts[assignment.attempt_id] = host
            self._kernel_session_hosts[session_id] = assignment.attempt_id
        try:
            await self._publish_kernel_state(host, session_id, "running")
            await host.stop.wait()
            # Lease expiry and host shutdown may race a cell.  Never evict or
            # advertise this actor as reusable while an operation still owns
            # its serialized execution slot.
            async with host.operation_lock:
                pass
            async with self._active_lock:
                active = self._active.get(assignment.attempt_id)
                lost = host.dirty or bool(
                    active and (active.lease_expired or active.preempt_requested)
                )
                remaining = tuple(host.sessions)
            for remaining_session in remaining:
                await self._publish_kernel_state(
                    host,
                    remaining_session,
                    "lost" if lost else "stopping",
                    message="kernel host ended; live state is not migratable" if lost else None,
                )
                if not lost:
                    await self._publish_kernel_state(host, remaining_session, "stopped")
            return ExecutionOutcome(
                succeeded=False,
                keep_warm=mode is KernelMode.SERVICE and not host.dirty,
                dirty=host.dirty,
                artifacts_verified=True,
                message=host.failure_message or "interactive kernel stopped",
            )
        finally:
            async with self._active_lock:
                self._kernel_hosts.pop(assignment.attempt_id, None)
                sessions = tuple(host.sessions)
                for current_session in sessions:
                    self._kernel_session_hosts.pop(current_session, None)
                host.sessions.clear()
            for current_session in sessions:
                self._session_leases.release(current_session)

    async def _attach_service_kernel(self, payload: Mapping[str, Any]) -> None:
        session_id = str(payload["session_id"])
        attempt_id = str(payload["host_attempt_id"])
        relay_id = str(payload["relay_id"])
        async with self._active_lock:
            existing_host = self._kernel_session_hosts.get(session_id)
            if existing_host is not None:
                if existing_host != attempt_id:
                    raise RuntimeError("kernel session is attached to another host")
                return
            host = getattr(self, "_kernel_hosts", {}).get(attempt_id)
            if host is None or host.mode is not KernelMode.SERVICE or host.stop.is_set():
                raise RuntimeError("service kernel host is unavailable")
            lease = SessionLease(
                workload_lease_id=f"{host.assignment.lease.lease_id}:session:{session_id}",
                actor_id=host.prepared.actor_id,
            )
            self._session_leases.reserve(
                session_id,
                KernelMode.SERVICE,
                lease,
                requested_gpus=0,
            )
            host.sessions[session_id] = relay_id
            self._kernel_session_hosts[session_id] = attempt_id
        await self._publish_kernel_state(host, session_id, "running")

    async def _queue_kernel_operation(self, payload: Mapping[str, Any]) -> None:
        operation_id = str(payload["operation_id"])
        session_id = str(payload["session_id"])
        attempt_id = str(payload["host_attempt_id"])
        relay_id = str(payload["relay_id"])
        operation = str(payload["operation"])
        raw_arguments = payload.get("arguments", {})
        if not isinstance(raw_arguments, Mapping):
            raise ValueError("kernel operation arguments must be a mapping")
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "session_id": session_id,
                    "relay_id": relay_id,
                    "operation": operation,
                    "arguments": dict(raw_arguments),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        async with self._active_lock:
            host = getattr(self, "_kernel_hosts", {}).get(attempt_id)
            if (
                host is None
                or self._kernel_session_hosts.get(session_id) != attempt_id
                or host.sessions.get(session_id) != relay_id
                or host.stop.is_set()
            ):
                raise RuntimeError("kernel session is unavailable")
            cached = host.operation_results.get(operation_id)
            if cached is not None:
                cached_fingerprint, report = cached
                if cached_fingerprint != fingerprint:
                    raise RuntimeError("kernel operation id was reused with different content")
                replay_task = asyncio.create_task(
                    self._publish_kernel_result(report),
                    name=f"topodrome-kernel-replay-{operation_id}",
                )
                self._kernel_background_tasks.add(replay_task)
                replay_task.add_done_callback(self._kernel_background_tasks.discard)
                return
            pending = host.operation_tasks.get(operation_id)
            if pending is not None:
                if pending[0] != fingerprint:
                    raise RuntimeError("kernel operation id was reused with different content")
                return
            task = asyncio.create_task(
                self._execute_kernel_operation(
                    host,
                    operation_id=operation_id,
                    session_id=session_id,
                    operation=operation,
                    arguments=dict(raw_arguments),
                    fingerprint=fingerprint,
                    maximum_output_bytes=_kernel_output_limit(payload),
                ),
                name=f"topodrome-kernel-operation-{operation_id}",
            )
            host.operation_tasks[operation_id] = (fingerprint, session_id, task)

    async def _execute_kernel_operation(
        self,
        host: _InteractiveHost,
        *,
        operation_id: str,
        session_id: str,
        operation: str,
        arguments: Mapping[str, Any],
        fingerprint: str,
        maximum_output_bytes: int,
    ) -> None:
        succeeded = False
        result: Any = None
        message: str | None = None
        allowed = {"describe", "generate", "tokenize"}
        if host.mode is KernelMode.RAW_MODEL:
            allowed.update({"call", "exec", "inspect", "jupyter_execute"})
        if operation not in allowed:
            message = f"operation {operation!r} is unavailable in {host.mode.value} mode"
        else:
            try:
                backend_arguments = dict(arguments)
                if operation == "jupyter_execute":
                    backend_arguments["_output_limit_bytes"] = maximum_output_bytes
                async with host.operation_lock:
                    if host.stop.is_set():
                        raise RuntimeError("kernel session is stopping")
                    result = await self.backend.execute_kernel(
                        host.prepared,
                        host.assignment,
                        host.workspace,
                        operation_id=operation_id,
                        session_id=session_id,
                        mode=host.mode,
                        operation=operation,
                        arguments=backend_arguments,
                    )
                if _json_payload_size(result) > maximum_output_bytes:
                    result = None
                    message = "kernel result exceeded the configured output limit"
                else:
                    succeeded = True
            except asyncio.CancelledError:
                message = "KeyboardInterrupt: kernel operation interrupted"
            except Exception as exc:
                # User-code failures from jupyter_execute are returned as a
                # structured, sanitized result by the runtime.  Other errors
                # are infrastructure failures and must not relay site-local
                # paths or backend details to the browser.
                message = f"{type(exc).__name__}: kernel runtime operation failed"
                host.dirty = True
                host.failure_message = message
                self.fences.set_actor_state(host.prepared.actor_id, "dirty")
                host.stop.set()
        report = {
            "operation_id": operation_id,
            "session_id": session_id,
            "attempt_id": host.assignment.attempt_id,
            "succeeded": succeeded,
            "result": result,
            "message": message,
        }
        async with self._active_lock:
            host.operation_tasks.pop(operation_id, None)
            host.operation_results[operation_id] = (fingerprint, report)
            host.operation_results.move_to_end(operation_id)
            while len(host.operation_results) > 256:
                host.operation_results.popitem(last=False)
        await self._publish_kernel_result(report)

    async def _publish_kernel_result(self, report: Mapping[str, Any]) -> None:
        attempt_id = str(report["attempt_id"])
        async with self._active_lock:
            active = self._active.get(attempt_id)
        if active is None:
            raise RuntimeError("kernel result has no active fenced attempt")
        async with active.fencing_lock:
            await self._finish_pending_renewal_locked(active)
            fenced_report = {
                **dict(report),
                "fencing_token": active.assignment.fencing_token or "",
            }
            publish_live = getattr(self.outbound, "publish_live", None)
            if callable(publish_live):
                await publish_live("kernel_result", fenced_report)
                return
            # Minimal injected test transports written before the live lane can
            # still exercise the agent.  Production OutboundClient always exposes
            # publish_live and never spools notebook code or results.
            await self.outbound.publish("kernel_result", fenced_report)

    async def _interrupt_kernel_operation(self, payload: Mapping[str, Any]) -> None:
        operation_id = str(payload["operation_id"])
        session_id = str(payload["session_id"])
        attempt_id = str(payload["host_attempt_id"])
        relay_id = str(payload["relay_id"])
        async with self._active_lock:
            host = self._kernel_hosts.get(attempt_id)
            if (
                host is None
                or self._kernel_session_hosts.get(session_id) != attempt_id
                or host.sessions.get(session_id) != relay_id
            ):
                raise RuntimeError("kernel session is unavailable")
            pending = host.operation_tasks.get(operation_id)
            if pending is None:
                return
            task = pending[2]
            task.cancel()
        await self.backend.cancel(f"{attempt_id}:{operation_id}")

    async def _stop_kernel(self, payload: Mapping[str, Any]) -> None:
        session_id = str(payload["session_id"])
        attempt_id = str(payload["host_attempt_id"])
        relay_id = str(payload["relay_id"])
        operations_to_cancel: list[tuple[str, asyncio.Task[None]]] = []
        async with self._active_lock:
            host = self._kernel_hosts.get(attempt_id)
            if host is None or self._kernel_session_hosts.get(session_id) != attempt_id:
                # Idempotent replay after the durable stopped event was spooled.
                return
            if host.sessions.get(session_id) != relay_id:
                raise RuntimeError("kernel stop has a stale relay identity")
            operations_to_cancel = [
                (operation_id, pending[2])
                for operation_id, pending in host.operation_tasks.items()
                if pending[1] == session_id
            ]
            for _, task in operations_to_cancel:
                task.cancel()
        for operation_id, _ in operations_to_cancel:
            await self.backend.cancel(f"{attempt_id}:{operation_id}")
        async with host.operation_lock:
            await self._publish_kernel_state(host, session_id, "stopping")
            async with self._active_lock:
                host.sessions.pop(session_id, None)
                self._kernel_session_hosts.pop(session_id, None)
                no_sessions = not host.sessions
                if no_sessions:
                    active = self._active.get(attempt_id)
                    if active is not None:
                        active.cancel_requested = True
                    host.stop.set()
            self._session_leases.release(session_id)
            await self._publish_kernel_state(
                host,
                session_id,
                "stopped",
                relay_id=relay_id,
            )

    async def _publish_kernel_state(
        self,
        host: _InteractiveHost,
        session_id: str,
        state: str,
        *,
        message: str | None = None,
        relay_id: str | None = None,
    ) -> None:
        async with self._active_lock:
            active = self._active.get(host.assignment.attempt_id)
        if active is None:
            raise RuntimeError("kernel event has no active fenced attempt")
        async with active.fencing_lock:
            await self._finish_pending_renewal_locked(active)
            await self.outbound.publish(
                "kernel_event",
                {
                    "session_id": session_id,
                    "attempt_id": host.assignment.attempt_id,
                    "fencing_token": active.assignment.fencing_token or "",
                    "state": state,
                    "relay_id": relay_id or host.sessions.get(session_id),
                    "runtime_id": host.runtime_id,
                    "actor_id": host.prepared.actor_id,
                    "message": _safe_message(message),
                },
            )

    async def cancel(self, attempt_id: str) -> None:
        async with self._active_lock:
            active = self._active.get(attempt_id)
            if active is None or active.explicit_cancel_requested:
                return
            active.cancel_requested = True
            if hasattr(active, "transfer_cancel"):
                active.transfer_cancel.set()
            active.explicit_cancel_requested = True
            if (getattr(active, "workspace", None) is not None
                and callable(getattr(self.backend, "verify_attempt_log", None))):
                phase = ("stopping-upload" if active.gpu_released else
                         "stopping-worker" if active.prepared is not None else "stopping-download")
                TransferLogReporter(active.workspace.log_dir / "attempt.log")(phase)
            if hasattr(active, "transfer_cancel"):
                active.transfer_cancel.set()
            if getattr(active, "gpu_released", False):
                self.backend.cancel_finalization(attempt_id)
                return
            host = self._kernel_hosts.get(attempt_id)
            if host is not None:
                host.stop.set()
            cancellation = asyncio.create_task(
                self._bounded_runtime_stop(
                    active,
                    control=self.backend.cancel,
                    grace=self.cancellation_grace_seconds,
                    refusal_reason_code="forced_cancellation_refused",
                ),
                name=f"topodrome-cancel-{attempt_id}",
            )
            self._cancellation_tasks.add(cancellation)
            cancellation.add_done_callback(self._cancellation_tasks.discard)
            cancellation.add_done_callback(_consume_task_result)

    async def request_preemption(self, attempt_id: str, *, explicit: bool = False) -> None:
        async with self._active_lock:
            active = self._active.get(attempt_id)
            if active is None or active.preempt_requested or getattr(active, "gpu_released", False):
                return
            active.preempt_requested = True
            if hasattr(active, "transfer_cancel"):
                active.transfer_cancel.set()
            host = self._kernel_hosts.get(attempt_id)
            if host is not None:
                # Live kernel state is non-migratable; end the host before the
                # ordinary preemption flow releases its physical lease.
                host.stop.set()
        await self._publish_attempt_state(
            active.assignment,
            "preempt_requested",
            reason_code="scheduler" if explicit else "foreign_gpu_pressure",
        )
        await self._bounded_runtime_stop(
            active,
            control=self.backend.request_preemption,
            grace=_checkpoint_grace_seconds(active.assignment),
            refusal_reason_code="forced_preemption_refused",
        )

    async def _bounded_runtime_stop(
        self,
        active: _ActiveAttempt,
        *,
        control: Callable[[str], Coroutine[Any, Any, None]],
        grace: float,
        refusal_reason_code: str,
    ) -> None:
        attempt_id = active.assignment.attempt_id
        grace = max(0.0, float(grace))
        deadline = asyncio.get_running_loop().time() + grace
        # Start the grace clock before issuing Ray control. The request itself
        # is bounded by the runtime adapter, but it must not extend the grace
        # period or delay the exact hard-stop boundary.
        request: asyncio.Task[None] = asyncio.create_task(
            control(attempt_id),
            name=f"topodrome-runtime-stop-control-{attempt_id}",
        )
        # Let an immediate cooperative request run once even when an operator
        # deliberately configures a zero-second grace (useful in tests).
        await asyncio.sleep(0)
        try:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            await asyncio.wait_for(asyncio.shield(active.task), timeout=remaining)
            return
        except TimeoutError:
            pass
        finally:
            if not request.done():
                request.cancel()
            done, _pending = await asyncio.wait(
                (request,),
                timeout=_PREEMPT_CONTROL_CANCEL_WAIT_SECONDS,
            )
            request_error: BaseException | None = None
            if request in done:
                try:
                    request.result()
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    request_error = exc
            else:
                # A non-cooperative wrapper must not delay the hard-stop past
                # its bounded cancellation allowance. Its production Ray call
                # has its own finite deadline and this callback consumes any
                # eventual exception.
                request.add_done_callback(_consume_task_result)
                request_error = TimeoutError("runtime preemption request ignored cancellation")
            if request_error is not None:
                log.warning(
                    "runtime_stop_request_failed",
                    attempt_id=attempt_id,
                    error=_safe_message(f"{type(request_error).__name__}: {request_error}"),
                )
        async with self._active_lock:
            current = self._active.get(attempt_id)
            prepared = None if current is None else current.prepared
        if prepared is None or active.task.done():
            return
        if not self._owns_runtime(active.assignment, prepared):
            await self._publish_force_stop_refusal(
                active.assignment,
                reason_code=refusal_reason_code,
            )
            return
        force_stop = getattr(self.backend, "force_stop", None)
        if not callable(force_stop):
            await self._publish_force_stop_refusal(
                active.assignment,
                reason_code=refusal_reason_code,
            )
            return
        try:
            stopped = await force_stop(prepared)
        except Exception:
            stopped = False
        if not stopped:
            await self._publish_force_stop_refusal(
                active.assignment,
                reason_code=refusal_reason_code,
            )

    def _owns_runtime(self, assignment: Assignment, prepared: PreparedRuntime) -> bool:
        if self.process_ownership is None or not prepared.process_ids:
            return False
        for pid in prepared.process_ids:
            owner = self.process_ownership.exact_owner(pid, require_live=True)
            if (
                owner is None
                or owner.lease_id != assignment.lease.lease_id
                or owner.actor_id != prepared.actor_id
            ):
                return False
        return True

    async def _publish_force_stop_refusal(
        self,
        assignment: Assignment,
        *,
        reason_code: str = "forced_preemption_refused",
    ) -> None:
        await self.outbound.publish(
            "log_chunk",
            {
                "site_id": self.site_id,
                "job_id": assignment.job_id,
                "attempt_id": assignment.attempt_id,
                "level": "error",
                "reason_code": reason_code,
                "message": "runtime ownership could not be proven; no process was signalled",
            },
        )

    async def _latest_checkpoint(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace | None,
    ) -> str | None:
        if workspace is None:
            return assignment.checkpoint_ref
        latest = getattr(self.backend, "latest_checkpoint", None)
        if not callable(latest):
            return assignment.checkpoint_ref
        async with self._active_lock:
            active = self._active.get(assignment.attempt_id)
            if active is not None and (
                active.explicit_cancel_requested or getattr(active, "gpu_released", False)
            ):
                return assignment.checkpoint_ref
        recovery = asyncio.create_task(latest(assignment, workspace))
        try:
            deadline = asyncio.get_running_loop().time() + _CHECKPOINT_RECOVERY_WAIT_SECONDS
            while not recovery.done():
                remaining = deadline - asyncio.get_running_loop().time()
                async with self._active_lock:
                    active = self._active.get(assignment.attempt_id)
                    canceled = active is not None and active.explicit_cancel_requested
                if canceled or remaining <= 0:
                    # A canceled thread await cannot stop an in-flight upload.
                    # Its checkpoint remains in persistent storage; do not
                    # advertise it as portable until verification succeeds.
                    return assignment.checkpoint_ref
                await asyncio.wait((recovery,), timeout=min(0.1, remaining))
            value = recovery.result()
            return None if value is None else str(value)
        except Exception as exc:
            await self.outbound.publish(
                "log_chunk",
                {
                    "site_id": self.site_id,
                    "job_id": assignment.job_id,
                    "attempt_id": assignment.attempt_id,
                    "level": "error",
                    "reason_code": "checkpoint_verification_failed",
                    "message": _safe_message(f"{type(exc).__name__}: {exc}"),
                },
            )
            return assignment.checkpoint_ref
        finally:
            if not recovery.done():
                recovery.cancel()
                recovery.add_done_callback(_consume_task_result)

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            if getattr(self, "input_stager", None) is not None:
                await self.input_stager.tick()
            await self._reconcile_stale_warm_liveness()
            snapshots = await asyncio.to_thread(self.inventory.sample)
            host = (
                None
                if self.host_inventory is None
                else await asyncio.to_thread(self.host_inventory.sample)
            )
            activity = await self._activity_payload()
            model_capabilities = await asyncio.to_thread(self.model_capabilities)
            model_capabilities = await asyncio.to_thread(
                self._publishable_model_capabilities,
                model_capabilities,
            )
            await self.outbound.publish_heartbeat(
                {
                    "site_id": self.site_id,
                    "site_kind": self.site_kind,
                    "state": "draining" if self._draining else "ready",
                    "gpus": [_snapshot_payload(snapshot) for snapshot in snapshots],
                    "models": [dict(value) for value in model_capabilities],
                    "cached_assets": list(self.cached_assets()),
                    "supported_workloads": (
                        []
                        if self._finalization_backlog()
                        >= getattr(self, "finalization_backlog_limit", 4)
                        else list(self.supported_workloads)
                    ),
                    "storage_available": self.storage_available(),
                    "storage_authoritative": bool(
                        getattr(getattr(self, "backend", None), "storage_authoritative", False)
                    ),
                    "queue_depth": len(self._active),
                    "gpu_safety_margin_bytes": self.admission.safety_margin_bytes,
                    "provider_instance_id": self.provider_instance_id,
                    "region": self.region,
                    **(
                        {}
                        if host is None
                        else {
                            "total_cpu": host.total_cpu,
                            "available_cpu": host.available_cpu,
                            "total_memory_bytes": host.total_memory_bytes,
                            "available_memory_bytes": host.available_memory_bytes,
                        }
                    ),
                    **activity,
                    "sent_at": datetime.now(UTC).isoformat(),
                },
            )
            await self._publish_lease_renewals()
            await _wait_or_stop(self._stop, self.heartbeat_interval_seconds)

    def _publishable_model_capabilities(
        self,
        capabilities: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        """Attach site-durable idle clocks to exact warm replicas."""

        enriched: list[dict[str, Any]] = []
        idle_by_replica: dict[tuple[str, tuple[str, ...]], float | None] = {}
        for raw in capabilities:
            capability = dict(raw)
            actor_id = capability.get("replica_id")
            raw_members = capability.get("replica_gpu_uuids")
            if (
                capability.get("availability") == "warm"
                and isinstance(actor_id, str)
                and actor_id
                and isinstance(raw_members, Sequence)
                and not isinstance(raw_members, (str, bytes))
                and raw_members
                and all(isinstance(member, str) and member for member in raw_members)
            ):
                members = tuple(raw_members)
                key = (actor_id, members)
                if key not in idle_by_replica:
                    idle_by_replica[key] = self.fences.actor_idle_since(
                        actor_id,
                        expected_gpu_uuids=members,
                    )
                idle_since = idle_by_replica[key]
                if idle_since is not None:
                    capability["idle_since"] = datetime.fromtimestamp(
                        idle_since,
                        tz=UTC,
                    ).isoformat()
            enriched.append(capability)
        return tuple(enriched)

    async def _reconcile_stale_warm_liveness(self) -> tuple[str, ...]:
        """Drop local allocation rows only after exact runtime disposal proof."""

        # A backend without warm-runtime support has nothing to reconcile.
        # Keeping this capability optional also preserves lightweight status
        # agents that only publish host inventory.
        reconcile = getattr(getattr(self, "backend", None), "reconcile_warm_liveness", None)
        if not callable(reconcile):
            return ()
        async with self._get_warm_maintenance_lock():
            actor_ids = tuple(await reconcile())
            removed: list[str] = []
            for actor_id in actor_ids:
                try:
                    await asyncio.to_thread(self.fences.evict_actor, actor_id)
                except Exception as exc:
                    log.warning(
                        "stale_warm_allocation_release_failed",
                        actor_id=actor_id,
                        error=_safe_message(f"{type(exc).__name__}: {exc}"),
                    )
                else:
                    removed.append(actor_id)
            return tuple(removed)

    def _get_warm_maintenance_lock(self) -> asyncio.Lock:
        """Return the per-agent lock for exact warm-runtime mutations."""

        lock = getattr(self, "_warm_maintenance_lock", None)
        if lock is None:
            # A few focused unit tests construct the agent without __init__.
            # Runtime construction always installs this lock eagerly.
            lock = asyncio.Lock()
            self._warm_maintenance_lock = lock
        return lock

    async def _publish_lease_renewals(self) -> None:
        async with self._active_lock:
            active_attempts = tuple(self._active.values())
        for active in active_attempts:
            if active.assignment.fencing_token is None or active.scheduler_terminal:
                continue
            try:
                async with active.fencing_lock:
                    active.renewal_pending = True
                    await self._finish_pending_renewal_locked(active)
            except asyncio.CancelledError:
                raise
            except _SchedulerLeaseTerminated:
                continue
            except Exception as exc:
                # Do not extend local ownership without a durable scheduler ACK.
                # A later heartbeat may retry while the scheduler lease remains
                # valid; crash recovery clears any leftover claim safely.
                log.warning(
                    "lease_renewal_unconfirmed",
                    attempt_id=active.assignment.attempt_id,
                    error=_safe_message(f"{type(exc).__name__}: {exc}"),
                )
                continue

    async def _finish_pending_renewal_locked(
        self,
        active: _ActiveAttempt,
        *,
        renew_local: bool = True,
    ) -> None:
        """Complete one replay-safe rotation while ``active.fencing_lock`` is held."""

        if active.scheduler_terminal:
            raise _SchedulerLeaseTerminated("scheduler already terminated this lease")
        if not active.renewal_pending:
            return
        assignment = active.assignment
        token = assignment.fencing_token
        if token is None:
            raise RuntimeError("active assignment is missing its fencing token")
        if (
            getattr(self, "finalization_journal", None) is not None
            and active.gpu_released and assignment.lease.expires_at <= time.time()
        ):
            reply = await self.outbound.publish_request(
                f"recover-finalization:{assignment.attempt_id}", "recover_finalization",
                {"job_id": assignment.job_id, "attempt_id": assignment.attempt_id},
            )
            result = reply["finalization_recovery"]
            if result.get("retry"):
                raise RuntimeError("retrying expired finalization grant")
            if not result["accepted"]:
                active.scheduler_terminal = True
                active.lease_expired = True
                self.backend.cancel_finalization(assignment.attempt_id)
                raise _SchedulerLeaseTerminated("finalization is no longer recoverable")
            recovered = Assignment.from_mapping(result["assignment"])
            self._verify_fencing_token(recovered)
            active.assignment = recovered
            active.renewal_pending = False
            return
        publish_request = getattr(self.outbound, "publish_request", None)
        if not callable(publish_request):
            raise RuntimeError("outbound transport cannot correlate renewal ACK payloads")
        ack_payload = await publish_request(
            f"lease-renewal:{assignment.attempt_id}",
            "lease_renewal",
            {
                "attempt_id": assignment.attempt_id,
                "fencing_token": token,
            },
        )
        if ack_payload.get("finalization_recovery_required"):
            # The old event was acknowledged as a no-op. Retry via the separate
            # upload-only recovery lane rather than leaving it at the spool head.
            active.assignment = replace(
                assignment, lease=replace(assignment.lease, expires_at=0)
            )
            return await self._finish_pending_renewal_locked(active, renew_local=False)
        raw_terminal = ack_payload.get("lease_renewal_terminal")
        if raw_terminal is not None:
            if not isinstance(raw_terminal, Mapping):
                raise RuntimeError("scheduler terminal renewal response is malformed")
            terminal_attempt_id = str(raw_terminal.get("attempt_id", ""))
            if terminal_attempt_id != assignment.attempt_id:
                raise RuntimeError("scheduler terminal renewal response belongs to another attempt")
            try:
                terminal_state = AttemptState(str(raw_terminal.get("state", "")))
            except ValueError as exc:
                raise RuntimeError(
                    "scheduler terminal renewal response has an invalid state"
                ) from exc
            if terminal_state not in {
                AttemptState.SUCCEEDED,
                AttemptState.REJECTED,
                AttemptState.PREEMPTED,
                AttemptState.FAILED,
                AttemptState.CANCELED,
                AttemptState.LOST,
            }:
                raise RuntimeError("scheduler terminal renewal response is not terminal")

            # The scheduler authenticated this exact historical assignment and
            # proved it is terminal.  Fence local work immediately and never
            # enqueue another renewal for an authority that cannot be revived.
            active.scheduler_terminal = True
            active.lease_expired = True
            active.cancel_requested = True
            active.renewal_pending = False
            host = getattr(self, "_kernel_hosts", {}).get(assignment.attempt_id)
            if host is not None:
                host.stop.set()
            if getattr(active, "gpu_released", False):
                self.backend.cancel_finalization(assignment.attempt_id)
            else:
                cancellation = asyncio.create_task(
                    self.backend.cancel(assignment.attempt_id),
                    name=f"topodrome-terminal-lease-cancel-{assignment.attempt_id}",
                )
                cancellation.add_done_callback(_consume_task_result)
            raise _SchedulerLeaseTerminated(
                f"scheduler attempt is terminal ({terminal_state.value})"
            )
        raw_grant = ack_payload.get("lease_renewal")
        if not isinstance(raw_grant, Mapping):
            raise RuntimeError("scheduler renewal ACK omitted the rotated fencing grant")
        grant = LeaseRenewalGrant.model_validate(raw_grant)
        if grant.attempt_id != assignment.attempt_id:
            raise RuntimeError("scheduler renewal ACK belongs to another attempt")
        if grant.fencing_token == token:
            raise RuntimeError("scheduler renewal ACK did not rotate the fencing token")
        if grant.lease_expires_at <= datetime.now(UTC):
            raise RuntimeError("scheduler renewal ACK contains an expired lease")

        rotated = replace(
            assignment,
            fencing_token=grant.fencing_token,
            # Detached finalizers have no local GPU claim/TTL row. Their
            # deadline must follow the authenticated renewal, not the original
            # offer expiry (which can precede compute completion by hours).
            lease=replace(assignment.lease, expires_at=grant.lease_expires_at.timestamp()),
        )
        self._verify_fencing_token(rotated)
        if self.fence_verifier is not None:
            verified = self.fence_verifier.verify(
                grant.fencing_token,
                attempt_id=assignment.attempt_id,
                site_id=self.site_id,
            )
            if getattr(verified, "expires_at", None) != grant.fencing_token_expires_at:
                raise RuntimeError("scheduler renewal ACK token expiry is inconsistent")

        # The new token becomes authoritative only after the correlated ACK is
        # authenticated.  A failed local TTL write leaves renewal_pending set,
        # forcing the next report to rotate again before it can be published.
        active.assignment = rotated
        if renew_local and not getattr(active, "gpu_released", False):
            await asyncio.to_thread(self.admission.renew, rotated.lease)
        active.renewal_pending = False

    async def _lease_expiry_loop(self) -> None:
        while not self._stop.is_set():
            await self._enforce_local_lease_expiry()
            await _wait_or_stop(
                self._stop,
                min(1.0, self.heartbeat_interval_seconds),
            )

    async def _enforce_local_lease_expiry(self, *, now: float | None = None) -> None:
        """Cooperatively stop attempts whose scheduler-ACKed local TTL elapsed.

        Claims remain fenced until the attempt cleanup confirms its runtime has
        stopped. This prevents a higher epoch from sharing a physical GPU with
        work that has not yet honored cancellation.
        """

        expired = set(await asyncio.to_thread(self.fences.expired_lease_ids, now=now))
        observed_at = time.time() if now is None else now
        async with self._active_lock:
            detached_expired = [
                item
                for item in self._active.values()
                if getattr(item, "gpu_released", False)
                and item.assignment.lease.expires_at <= observed_at
            ]
            for item in detached_expired:
                if getattr(self, "finalization_journal", None) is not None:
                    # Storage commands check the deadline and pause themselves.
                    # Recovery can only renew upload authority, never GPU claims.
                    continue
                item.lease_expired = True
                self.backend.cancel_finalization(item.assignment.attempt_id)
        if not expired:
            return
        async with self._active_lock:
            affected = tuple(
                (attempt_id, active)
                for attempt_id, active in self._active.items()
                if active.assignment.lease.lease_id in expired
            )
            for _, active in affected:
                active.lease_expired = True
                active.cancel_requested = True
                if hasattr(active, "transfer_cancel"):
                    active.transfer_cancel.set()
                host = getattr(self, "_kernel_hosts", {}).get(active.assignment.attempt_id)
                if host is not None:
                    host.stop.set()
        for attempt_id, _ in affected:
            try:
                await self.backend.cancel(attempt_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "expired_lease_cancel_failed",
                    attempt_id=attempt_id,
                    error=_safe_message(f"{type(exc).__name__}: {exc}"),
                )

    async def _local_lease_expired(self, attempt_id: str) -> bool:
        async with self._active_lock:
            active = self._active.get(attempt_id)
            return bool(active and active.lease_expired)

    async def _activity_payload(self) -> dict[str, bool]:
        """Return the provider idle policy's exact five local signals."""

        async with self._active_lock:
            active = tuple(self._active.values())
        queued = any(
            item.prepared is None and not getattr(item, "gpu_released", False) for item in active
        )
        running = any(item.prepared is not None for item in active)
        interactive = any(
            item.assignment.raw_spec.get("workload_kind") == "interactive" for item in active
        )
        artifact_pending = any(
            getattr(item, "gpu_released", False)
            or (
                item.prepared is not None
                and (
                    bool(item.assignment.raw_spec.get("required_outputs", ()))
                    or bool(getattr(item, "preempt_requested", False))
                )
            )
            for item in active
        ) or self._pending_io_receipts()
        # Model/source acquisition is part of prepare. Required outputs remain
        # conservatively pending for the duration of execute; the backend does
        # not return success until their authoritative verification finishes.
        return {
            "job_running": running,
            "assignment_queued_locally": queued,
            "interactive_kernel_active": interactive,
            "artifact_flush_pending": artifact_pending,
            "staging_operation_active": queued or bool(
                getattr(self, "input_stager", None) and self.input_stager.stages
            ),
        }

    async def _pressure_loop(self) -> None:
        while not self._stop.is_set():
            snapshots = await asyncio.to_thread(self.inventory.sample)
            events = self.pressure.observe(snapshots)
            rebalance_events = (
                ()
                if self.idle_actor_rebalancer is None
                else self.idle_actor_rebalancer.observe(snapshots)
            )
            if events:
                by_gpu = {event.gpu_uuid: event for event in events}
                async with self._active_lock:
                    affected = {
                        attempt_id
                        for attempt_id, active in self._active.items()
                        if not getattr(active, "gpu_released", False)
                        and set(active.assignment.lease.gpu_uuids) & set(by_gpu)
                    }
                for attempt_id in affected:
                    try:
                        await self.request_preemption(attempt_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # A transient transport/protocol failure must not stop
                        # the only monitor that yields GPUs to foreign users.
                        # ``request_preemption`` records its intent before any
                        # await, so repeated pressure remains idempotent.
                        log.warning(
                            "pressure_preemption_failed",
                            attempt_id=attempt_id,
                            error=_safe_message(f"{type(exc).__name__}: {exc}"),
                        )
            if rebalance_events:
                try:
                    await self._rebalance_idle_warm_actors(rebalance_events)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning(
                        "idle_warm_actor_rebalance_pass_failed",
                        error=_safe_message(f"{type(exc).__name__}: {exc}"),
                    )
            if getattr(self, "warm_actor_idle_timeout_seconds", None) is not None:
                try:
                    await self._release_expired_idle_warm_actor()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Expiration is opportunistic capacity recovery. A
                    # transient Ray/SQLite error must never stop pressure
                    # monitoring or the site agent itself.
                    log.warning(
                        "idle_warm_actor_expiry_pass_failed",
                        error=_safe_message(f"{type(exc).__name__}: {exc}"),
                    )
            await _wait_or_stop(self._stop, self.pressure_interval_seconds)

    async def _release_expired_idle_warm_actor(
        self,
        *,
        now: float | None = None,
    ) -> tuple[str, ...]:
        """Dispose at most one exact actor whose durable idle TTL elapsed.

        Only the short candidate/fence phase holds the active-attempt lock.
        The durable ``stopping`` rows reject subsequent offers, so potentially
        slow Ray disposal cannot starve unrelated lease renewals or admissions.
        """

        timeout = getattr(self, "warm_actor_idle_timeout_seconds", None)
        evict = getattr(self.backend, "evict_warm_allocation", None)
        evict_replica = getattr(self.backend, "evict_warm_replica", None)
        if timeout is None or (not callable(evict) and not callable(evict_replica)):
            return ()
        observed_at = time.time() if now is None else now
        idle_before = observed_at - timeout
        selected: tuple[ActorAllocation, ...] | None = None

        # Serializes exact runtime disposal with heartbeat liveness cleanup.
        async with self._get_warm_maintenance_lock():
            async with self._active_lock:
                protected_gpu_uuids = frozenset(
                    gpu_uuid
                    for active in self._active.values()
                    if not getattr(active, "gpu_released", False)
                    for gpu_uuid in active.assignment.lease.gpu_uuids
                )
                candidates = await asyncio.to_thread(
                    self.fences.expired_idle_actor_groups,
                    idle_before=idle_before,
                )
                for allocations in candidates:
                    gpu_uuids = tuple(item.gpu_uuid for item in allocations)
                    if set(gpu_uuids) & protected_gpu_uuids:
                        continue
                    if not all(
                        self.backend_is_live_allocation(allocation) for allocation in allocations
                    ):
                        continue
                    began = await asyncio.to_thread(
                        self.fences.begin_expired_idle_actor_eviction,
                        allocations[0].actor_id,
                        expected_gpu_uuids=gpu_uuids,
                        idle_before=idle_before,
                        now=observed_at,
                    )
                    if began:
                        selected = allocations
                        break

            if selected is None:
                return ()

            actor_id = selected[0].actor_id
            gpu_uuids = tuple(item.gpu_uuid for item in selected)
            worker_pids: tuple[int, ...] = ()
            try:
                worker_pids = await self._evict_exact_warm_replica(selected)
            except asyncio.CancelledError:
                # Disposal is ambiguous. Persistent stopping rows retain the
                # fence and startup reconciliation resumes exact cleanup.
                raise
            except Exception as exc:
                restored = False
                try:
                    if all(self.backend_is_live_allocation(allocation) for allocation in selected):
                        restored = await asyncio.to_thread(
                            self.fences.restore_idle_actor_eviction,
                            actor_id,
                            expected_gpu_uuids=gpu_uuids,
                            now=(time.time() if now is None else now),
                        )
                except Exception:
                    # Ambiguous liveness remains fenced as stopping.
                    restored = False
                log.warning(
                    "idle_warm_actor_expiry_failed",
                    actor_id=actor_id,
                    gpu_uuids=list(gpu_uuids),
                    allocation_restored=restored,
                    error=_safe_message(f"{type(exc).__name__}: {exc}"),
                )
                return ()

            removed = await asyncio.to_thread(
                self.fences.complete_idle_actor_eviction,
                actor_id,
                expected_gpu_uuids=gpu_uuids,
            )
            if self.process_ownership is not None:
                for pid in worker_pids:
                    self.process_ownership.unregister(pid)
            if removed != len(selected):
                log.error(
                    "idle_warm_actor_expiry_release_failed",
                    actor_id=actor_id,
                    gpu_uuids=list(gpu_uuids),
                )
                return ()
            log.info(
                "idle_warm_actor_expired",
                actor_id=actor_id,
                gpu_uuids=list(gpu_uuids),
                idle_timeout_seconds=timeout,
            )
            return (actor_id,)

    async def _rebalance_idle_warm_actors(
        self,
        events: Sequence[PressureEvent],
    ) -> tuple[str, ...]:
        async with self._get_warm_maintenance_lock():
            return await self._rebalance_idle_warm_actors_locked(events)

    async def _rebalance_idle_warm_actors_locked(
        self,
        events: Sequence[PressureEvent],
    ) -> tuple[str, ...]:
        """Dispose exact idle replicas only when a safer local gang is fresh.

        The active-attempt lock serializes this path with local offers.  The
        SQLite transition from ``idle`` to ``stopping`` is still the final
        authority because attempt cleanup and process teardown can race outside
        the asyncio lock.
        """

        rebalancer = self.idle_actor_rebalancer
        evict = getattr(self.backend, "evict_warm_allocation", None)
        evict_replica = getattr(self.backend, "evict_warm_replica", None)
        if rebalancer is None or (not callable(evict) and not callable(evict_replica)):
            return ()

        evicted_actor_ids: list[str] = []
        async with self._active_lock:
            # The triggering sample is deliberately not enough. Re-probe after
            # acquiring the offer lock so ended pressure or newly occupied
            # alternative capacity cannot justify an unnecessary model unload.
            snapshots = await asyncio.to_thread(self.inventory.sample)
            allocations = await asyncio.to_thread(self.fences.actor_allocations)
            active_claims = await asyncio.to_thread(self.fences.active_claims)
            protected_gpu_uuids = frozenset(
                gpu_uuid
                for active in self._active.values()
                if not getattr(active, "gpu_released", False)
                for gpu_uuid in active.assignment.lease.gpu_uuids
            )
            plans = rebalancer.plans(
                events,
                snapshots,
                allocations,
                active_claims,
                safety_margin_bytes=self.admission.safety_margin_bytes,
                protected_gpu_uuids=protected_gpu_uuids,
            )
            for plan in plans:
                if not all(
                    self.backend_is_live_allocation(allocation) for allocation in plan.allocations
                ):
                    continue
                began = await asyncio.to_thread(
                    self.fences.begin_idle_actor_eviction,
                    plan.actor_id,
                    expected_gpu_uuids=plan.gpu_uuids,
                    alternative_gpu_uuids=plan.alternative_gpu_uuids,
                    cooldown_seconds=rebalancer.policy.cooldown_seconds,
                )
                if not began:
                    continue

                worker_pids: tuple[int, ...] = ()
                try:
                    worker_pids = await self._evict_exact_warm_replica(plan.allocations)
                except asyncio.CancelledError:
                    # Cancellation makes disposal ambiguous. Keep the complete
                    # replica fenced as stopping for later exact reconciliation.
                    raise
                except Exception as exc:
                    restored = False
                    try:
                        actor_still_live = all(
                            self.backend_is_live_allocation(allocation)
                            for allocation in plan.allocations
                        )
                        if actor_still_live:
                            restored = await asyncio.to_thread(
                                self.fences.restore_idle_actor_eviction,
                                plan.actor_id,
                                expected_gpu_uuids=plan.gpu_uuids,
                            )
                    except Exception:
                        # Ambiguous liveness or rollback retains the stopping
                        # fence; it never makes the GPU look free.
                        restored = False
                    log.warning(
                        "idle_warm_actor_rebalance_failed",
                        actor_id=plan.actor_id,
                        gpu_uuids=list(plan.gpu_uuids),
                        allocation_restored=restored,
                        error=_safe_message(f"{type(exc).__name__}: {exc}"),
                    )
                    continue

                removed = await asyncio.to_thread(
                    self.fences.complete_idle_actor_eviction,
                    plan.actor_id,
                    expected_gpu_uuids=plan.gpu_uuids,
                )
                if self.process_ownership is not None:
                    for pid in worker_pids:
                        self.process_ownership.unregister(pid)
                if removed != len(plan.allocations):
                    # The exact runtime is gone, but conservative stopping rows
                    # continue to block admission until reconciliation repairs
                    # local bookkeeping.
                    log.error(
                        "idle_warm_actor_rebalance_release_failed",
                        actor_id=plan.actor_id,
                        gpu_uuids=list(plan.gpu_uuids),
                    )
                    continue
                evicted_actor_ids.append(plan.actor_id)
                log.info(
                    "idle_warm_actor_rebalanced",
                    actor_id=plan.actor_id,
                    gpu_uuids=list(plan.gpu_uuids),
                    alternative_gpu_uuids=list(plan.alternative_gpu_uuids),
                    alternative_band=plan.alternative_band,
                )
        return tuple(evicted_actor_ids)

    async def _publish_rejection(self, assignment: Assignment, reason: str) -> None:
        await self.outbound.publish(
            "assignment_rejected",
            {
                "site_id": self.site_id,
                "job_id": assignment.job_id,
                "attempt_id": assignment.attempt_id,
                "fencing_token": assignment.fencing_token or "",
                "lease_id": assignment.lease.lease_id,
                "reason_code": "local_admission_rejected",
                "message": _safe_message(reason),
            },
        )

    async def _publish_attempt_state(
        self,
        assignment: Assignment,
        state: str,
        *,
        reason_code: str | None = None,
        message: str | None = None,
        artifacts: Sequence[Mapping[str, Any]] = (),
        checkpoint_ref: str | None = None,
        attempt_log: Mapping[str, Any] | None = None,
        log_complete: bool = False,
        gpu_released: bool = False,
        timings: Mapping[str, float | None] | None = None,
    ) -> None:
        async with self._active_lock:
            active = self._active.get(assignment.attempt_id)
        if active is None:
            raise RuntimeError("attempt event has no active fenced attempt")
        async with active.fencing_lock:
            # Pressure can win the event-loop race immediately after assignment
            # acceptance. In that case PREEMPT_REQUESTED is the first durable
            # runtime state; never enqueue a later STARTING regression behind it.
            if state == "starting" and active.preempt_requested:
                return
            await self._finish_pending_renewal_locked(
                active,
                renew_local=state not in {"succeeded", "preempted", "failed", "canceled", "lost"},
            )
            current = active.assignment
            if state == "succeeded" and active.finalization_started_at is not None:
                timings = {
                    "finalization_seconds": time.monotonic() - active.finalization_started_at,
                }
            async def publish(kind, payload):
                if getattr(self, "finalization_journal", None) is None:
                    return await self.outbound.publish(kind, payload)
                reply = await self.outbound.publish_request(
                    f"attempt-state:{current.attempt_id}:{state}", kind, payload
                )
                if reply.get("finalization_recovery_required"):
                    active.assignment = replace(
                        current, lease=replace(current.lease, expires_at=0)
                    )
                    active.renewal_pending = True
                    raise RuntimeError("upload state report needs fresh authority")
            await publish(
                "attempt_event",
                {
                    "site_id": self.site_id,
                    "job_id": current.job_id,
                    "attempt_id": current.attempt_id,
                    "fencing_token": current.fencing_token or "",
                    "state": state,
                    "reason_code": reason_code,
                    "message": _safe_message(message),
                    "checkpoint_ref": checkpoint_ref,
                    "artifacts": [dict(value) for value in artifacts],
                    "log": None if attempt_log is None else dict(attempt_log),
                    "log_complete": log_complete,
                    "gpu_released": gpu_released,
                    "timings": None if timings is None else dict(timings),
                },
            )

    async def _publish_drain_if_ready(self) -> None:
        if (
            self._draining
            and not self._active
            and not self._pending_io_receipts()
            and not self._drain_ready_emitted
            and self._drain_command_nonce is not None
        ):
            await self.outbound.publish(
                "drain_ready",
                {
                    "site_id": self.site_id,
                    "shutdown_generation": self._shutdown_generation,
                    "command_nonce": self._drain_command_nonce,
                },
            )
            self._drain_ready_emitted = True


async def _wait_or_stop(stop: asyncio.Event, delay: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
    except TimeoutError:
        pass


async def _finish_input_staging(operation: Awaitable[None]) -> None:
    """Do not release a lease while an uncancellable staging thread is live."""

    task = asyncio.ensure_future(operation)
    was_cancelled = False
    while True:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                # Cancellation originated inside the staging operation rather
                # than from the attempt task; preserve its ordinary semantics.
                await task
            was_cancelled = True
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            continue
        break
    if was_cancelled:
        raise asyncio.CancelledError


def _snapshot_payload(snapshot: GPUSnapshot) -> dict[str, Any]:
    return {
        "uuid": snapshot.uuid,
        "index": snapshot.index,
        "name": snapshot.name,
        "total_memory_bytes": snapshot.total_memory_bytes,
        "free_memory_bytes": snapshot.free_memory_bytes,
        "utilization_percent": snapshot.utilization_percent,
        "externally_reserved": snapshot.externally_reserved,
        "observed_at": datetime.fromtimestamp(snapshot.sampled_at, tz=UTC).isoformat(),
        "processes": [
            {
                "pid": process.pid,
                "used_memory_bytes": process.used_memory_bytes or 0,
                "process_name": process.process_name or "",
                "username": process.username,
                "script_name": process.script_name,
                "owner": process.owner.value,
                "topodrome_owned": process.owner.value == "topodrome",
                "reclaimable": process.owner.value == "cooperative",
            }
            for process in snapshot.processes
        ],
    }


def _kernel_mode(assignment: Assignment) -> KernelMode | None:
    if assignment.raw_spec.get("workload_kind") != "interactive":
        return None
    try:
        return KernelMode(str(assignment.kwargs["mode"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("interactive assignment has an invalid kernel mode") from exc


def _is_adapter_training(assignment: Assignment) -> bool:
    return assignment.raw_spec.get("workload_kind") == "adapter_training"


def _is_private_workload(assignment: Assignment) -> bool:
    # Multi-GPU worker groups are deliberately private and disposable. A
    # detached single-GPU actor cannot safely be folded into their atomic
    # placement group, so exact idle allocations must be evicted beforehand.
    if _parallelism_mode(assignment) == "distributed":
        return True
    if _is_adapter_training(assignment):
        return True
    return (
        assignment.raw_spec.get("workload_kind") == "interactive"
        and assignment.kwargs.get("mode") == KernelMode.RAW_MODEL.value
    )


def _requires_slot_eviction(assignment: Assignment) -> bool:
    return _is_private_workload(assignment) or _parallelism_mode(assignment) == "model_sharded"


def _parallelism_mode(assignment: Assignment) -> str:
    model = assignment.raw_spec.get("model")
    if not isinstance(model, Mapping):
        return "single"
    parallelism = model.get("parallelism")
    if not isinstance(parallelism, Mapping):
        return "single"
    return str(parallelism.get("mode") or "single")


def _required_kernel_value(assignment: Assignment, name: str) -> str:
    value = assignment.kwargs.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"interactive assignment requires {name}")
    return value


def _session_lease(
    assignment: Assignment,
    prepared: PreparedRuntime,
    *,
    session_id: str,
    mode: KernelMode,
) -> SessionLease:
    workload_id = f"{assignment.lease.lease_id}:session:{session_id}"
    if mode is KernelMode.RAW_MODEL:
        return SessionLease(
            workload_lease_id=workload_id,
            actor_id=prepared.actor_id,
            gpu_uuids=assignment.lease.gpu_uuids,
            exclusive=True,
            private_actor=True,
        )
    return SessionLease(workload_lease_id=workload_id, actor_id=prepared.actor_id)


def _artifact_payload(metadata: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    artifacts = metadata.get("artifacts", ())
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        return ()
    return tuple(value for value in artifacts if isinstance(value, Mapping))


def _optional_metadata_text(metadata: Mapping[str, Any], name: str) -> str | None:
    value = metadata.get(name)
    return None if value is None else str(value)


def _plain_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump(mode="python")
        if isinstance(result, Mapping):
            return result
    raise ValueError("expected an object-like assignment field")


def _timestamp(value: Any) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    return float(value)


def _shutdown_coordinates(payload: Mapping[str, Any]) -> tuple[int, str]:
    generation = payload.get("shutdown_generation")
    nonce = payload.get("command_nonce")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise ValueError("shutdown_generation must be a positive integer")
    if not isinstance(nonce, str) or not 16 <= len(nonce) <= 120:
        raise ValueError("command_nonce is invalid")
    return generation, nonce


def _actor_eviction_coordinates(
    payload: Mapping[str, Any],
) -> tuple[str, str, str, tuple[str, ...], float | None]:
    if any(name in payload for name in ("pid", "pids", "worker_pids", "process_ids")):
        raise ValueError("actor eviction commands must not contain process identities")
    eviction_id = payload.get("eviction_id")
    actor_id = payload.get("actor_id")
    replica_key_hash = payload.get("replica_key_hash")
    raw_gpu_uuids = payload.get("gpu_uuids")
    raw_minimum_idle_seconds = payload.get("minimum_idle_seconds")
    if not isinstance(eviction_id, str) or not 1 <= len(eviction_id) <= 200:
        raise ValueError("eviction_id is invalid")
    if not isinstance(actor_id, str) or not 1 <= len(actor_id) <= 300:
        raise ValueError("actor_id is invalid")
    if (
        not isinstance(replica_key_hash, str)
        or len(replica_key_hash) != 64
        or any(character not in "0123456789abcdef" for character in replica_key_hash)
    ):
        raise ValueError("replica_key_hash must be a lowercase SHA-256 digest")
    if (
        not isinstance(raw_gpu_uuids, Sequence)
        or isinstance(raw_gpu_uuids, (str, bytes))
        or not 1 <= len(raw_gpu_uuids) <= 64
        or not all(isinstance(value, str) and 1 <= len(value) <= 300 for value in raw_gpu_uuids)
    ):
        raise ValueError("gpu_uuids must be a non-empty sequence of GPU identities")
    gpu_uuids = tuple(raw_gpu_uuids)
    if len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("gpu_uuids must be unique and in replica-rank order")
    if raw_minimum_idle_seconds is None:
        minimum_idle_seconds = None
    elif (
        isinstance(raw_minimum_idle_seconds, bool)
        or not isinstance(raw_minimum_idle_seconds, (int, float))
        or not 0 <= raw_minimum_idle_seconds <= 604_800
    ):
        raise ValueError("minimum_idle_seconds is invalid")
    else:
        minimum_idle_seconds = float(raw_minimum_idle_seconds)
    return eviction_id, actor_id, replica_key_hash, gpu_uuids, minimum_idle_seconds


def _checkpoint_grace_seconds(assignment: Assignment) -> float:
    raw = assignment.raw_spec.get("checkpoint", {})
    if not isinstance(raw, Mapping):
        return 30.0
    try:
        grace = float(raw.get("grace_seconds", 30.0))
    except (TypeError, ValueError):
        return 30.0
    return max(0.0, grace)


def _kernel_output_limit(payload: Mapping[str, Any]) -> int:
    try:
        requested = int(payload.get("maximum_output_bytes", 262_144))
    except (TypeError, ValueError):
        requested = 262_144
    return min(4_194_304, max(4096, requested))


def _json_payload_size(value: Any) -> int:
    try:
        return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return 4_194_305


_SECRET_VALUE = re.compile(
    r"(?i)((?:token|secret|password|authorization|api[_-]?key)\s*[=:]\s*)[^\s&,]+"
)


def _safe_message(message: str | None) -> str | None:
    if message is None:
        return None
    redacted = _SECRET_VALUE.sub(r"\1[REDACTED]", message)
    redacted = re.sub(r"(?i)bearer\s+[^\s,]+", "Bearer [REDACTED]", redacted)
    return redacted[:2000]


__all__ = [
    "Assignment",
    "ExecutionBackend",
    "ExecutionOutcome",
    "PreparedRuntime",
    "RegionalDurability",
    "SiteAgent",
]
