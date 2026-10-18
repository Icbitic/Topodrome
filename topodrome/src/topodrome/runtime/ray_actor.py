"""Lazy optional-Ray adapter for detached physical-GPU model actors."""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import queue
import re
import socket
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock, Thread
from typing import Any, cast
from uuid import uuid4

from .actor import (
    ActorState,
    IncompatibleActorBusyError,
    LocalModelActor,
    ModelLoader,
)
from .environment import (
    RAY_ALLOWED_JOB_KEYS_ENVIRONMENT,
    RAY_WORKER_ENVIRONMENT_SETUP_HOOK,
    configure_ray_driver_environment,
    encode_ray_worker_allowed_job_keys,
    validate_job_environment,
    validate_runtime_references,
)
from .model_key import ActorKey


class RayUnavailableError(RuntimeError):
    pass


class RayActorError(RuntimeError):
    pass


class RayActorTimeoutError(RayActorError):
    """A bounded Ray data/control operation exceeded its explicit deadline."""


class RayDisposalError(RayActorError):
    """Exact actor disposal could not be confirmed; its slots stay fenced."""

    cleanup_unconfirmed = True


@dataclass(frozen=True, slots=True)
class RayOperationTimeouts:
    """Finite deadlines for each class of Ray operation.

    Execution receives a deliberately generous default because user workloads
    can be long-lived. Preparation includes model construction and runtime-env
    setup. Control and disposal stay short so drain/preemption cannot wedge.
    Operators embedding the adapter can override every value explicitly.
    """

    prepare_seconds: float = 1800.0
    execute_seconds: float = 86400.0
    control_seconds: float = 10.0
    disposal_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "prepare_seconds",
            "execute_seconds",
            "control_seconds",
            "disposal_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Ray {name} must be finite and positive")


def _bounded_call(
    operation: str,
    timeout_seconds: float,
    function: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Bound Ray control APIs that do not expose their own timeout argument."""

    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result.put((True, function(*args, **kwargs)))
        except BaseException as exc:
            result.put((False, exc))

    worker = Thread(target=invoke, name="topodrome-ray-control", daemon=True)
    worker.start()
    try:
        succeeded, value = result.get(timeout=timeout_seconds)
    except queue.Empty:
        raise RayActorTimeoutError(f"{operation} exceeded {timeout_seconds:g} seconds") from None
    if not succeeded:
        raise value
    return value


def _ray_get(ray: Any, value: Any, *, timeout_seconds: float, operation: str) -> Any:
    try:
        return ray.get(value, timeout=timeout_seconds)
    except Exception as exc:
        timeout_type = getattr(getattr(ray, "exceptions", None), "GetTimeoutError", None)
        if isinstance(timeout_type, type) and isinstance(exc, timeout_type):
            raise RayActorTimeoutError(f"{operation} exceeded {timeout_seconds:g} seconds") from exc
        raise


def _actor_stopped_error(ray: Any, error: BaseException) -> bool:
    exceptions = getattr(ray, "exceptions", None)
    stopped_types = tuple(
        value
        for name in (
            "ActorDiedError",
            "ActorUnavailableError",
            "ActorUnschedulableError",
            "OwnerDiedError",
        )
        if isinstance((value := getattr(exceptions, name, None)), type)
    )
    return bool(stopped_types and isinstance(error, stopped_types))


def _confirm_actor_stopped(
    ray: Any,
    handle: Any,
    *,
    timeouts: RayOperationTimeouts,
) -> None:
    deadline = time.monotonic() + timeouts.disposal_seconds
    last_error: BaseException | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RayDisposalError("Ray actor disposal was not confirmed") from last_error
        try:
            reference = _bounded_call(
                "Ray actor disposal probe submission",
                min(timeouts.control_seconds, remaining),
                handle.describe.remote,
            )
            _ray_get(
                ray,
                reference,
                timeout_seconds=min(0.25, remaining),
                operation="Ray actor disposal probe",
            )
        except RayActorTimeoutError as exc:
            last_error = exc
        except BaseException as exc:
            if _actor_stopped_error(ray, exc):
                return
            raise RayDisposalError("Ray actor disposal probe failed ambiguously") from exc
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def _kill_and_confirm_actor(
    ray: Any,
    handle: Any,
    *,
    timeouts: RayOperationTimeouts,
) -> None:
    """Kill one exact handle and require positive proof that it is gone.

    ``ray.kill`` may itself time out after the request reached GCS, or report
    that an already-dead actor is unavailable.  Neither condition should skip
    the exact-handle liveness probe that decides whether physical claims can be
    released.
    """

    try:
        _bounded_call(
            "Ray actor kill",
            timeouts.control_seconds,
            ray.kill,
            handle,
            no_restart=True,
        )
    except BaseException as exc:
        if not isinstance(exc, RayActorTimeoutError) and not _actor_stopped_error(ray, exc):
            raise RayDisposalError("Ray actor kill failed ambiguously") from exc
    _confirm_actor_stopped(ray, handle, timeouts=timeouts)


def _confirm_placement_group_removed(
    ray: Any,
    placement_group: Any,
    *,
    timeouts: RayOperationTimeouts,
) -> None:
    table = getattr(getattr(ray, "util", None), "placement_group_table", None)
    if not callable(table):
        raise RayDisposalError("Ray cannot confirm placement-group disposal")
    deadline = time.monotonic() + timeouts.disposal_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RayDisposalError("Ray placement-group disposal was not confirmed")
        value = _bounded_call(
            "Ray placement-group disposal probe",
            min(timeouts.control_seconds, remaining),
            table,
            placement_group,
        )
        if not value or str(value.get("state", "")).upper() == "REMOVED":
            return
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def ray_gpu_resource_name(gpu_uuid: str) -> str:
    """Return the custom Ray resource used for one admitted physical GPU."""

    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        raise ValueError("gpu_uuid must not be empty")
    sanitized = _NON_ALNUM.sub("_", gpu_uuid.strip().lower()).strip("_")
    if not sanitized:
        raise ValueError("gpu_uuid must contain an alphanumeric character")
    return f"topodrome_gpu_{sanitized}"


def _slot_actor_name(key: ActorKey) -> str:
    # The name belongs to the complete physical slot set rather than the
    # current model. A site-agent restart can therefore find and reuse or evict
    # the exact allocation. The one-member encoding stays backward compatible.
    slot = "\0".join((key.site_id, *key.replica_gpu_uuids)).encode()
    return f"topodrome-model-slot-{hashlib.sha256(slot).hexdigest()[:24]}"


@dataclass(frozen=True, slots=True)
class RayActorSpec:
    key: ActorKey
    cuda_visible_device: str
    namespace: str = "topodrome"
    num_cpus: float = 1.0
    memory_bytes: int | None = None
    pip: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    working_dir: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cuda_visible_device, str) or not self.cuda_visible_device.strip():
            raise ValueError("cuda_visible_device must be an explicit non-empty value")
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("namespace must not be empty")
        if self.num_cpus < 0:
            raise ValueError("num_cpus must be non-negative")
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        visible_devices = tuple(value.strip() for value in self.cuda_visible_device.split(","))
        if any(not value for value in visible_devices):
            raise ValueError("CUDA device list contains an empty member")
        if len(set(visible_devices)) != len(visible_devices):
            raise ValueError("CUDA devices must be unique")
        if len(visible_devices) != len(self.key.replica_gpu_uuids):
            raise ValueError("one CUDA device is required per replica GPU UUID")
        requirements, working_dir = validate_runtime_references(self.pip, self.working_dir)
        object.__setattr__(self, "pip", requirements)
        object.__setattr__(self, "working_dir", working_dir)
        object.__setattr__(self, "env", validate_job_environment(self.env or {}))

    @property
    def actor_name(self) -> str:
        return _slot_actor_name(self.key)

    @property
    def resource_name(self) -> str:
        if len(self.key.replica_gpu_uuids) != 1:
            raise ValueError("multi-GPU replicas have more than one resource name")
        return ray_gpu_resource_name(self.key.gpu_uuid)

    @property
    def resource_names(self) -> tuple[str, ...]:
        return tuple(ray_gpu_resource_name(uuid) for uuid in self.key.replica_gpu_uuids)


@dataclass(frozen=True, slots=True)
class RayWorkerGroupSpec:
    """One private, atomically placed worker per admitted physical GPU."""

    keys: tuple[ActorKey, ...]
    cuda_visible_devices: tuple[str, ...]
    namespace: str = "topodrome"
    num_cpus: float = 1.0
    memory_bytes: int = 1024**3
    pip: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    working_dir: str | None = None

    def __post_init__(self) -> None:
        if len(self.keys) < 2:
            raise ValueError("a Ray worker group requires at least two GPUs")
        if len(self.keys) != len(self.cuda_visible_devices):
            raise ValueError("one CUDA device is required for every worker-group key")
        if len({key.gpu_uuid for key in self.keys}) != len(self.keys):
            raise ValueError("worker-group GPU UUIDs must be unique")
        if len(set(self.cuda_visible_devices)) != len(self.cuda_visible_devices):
            raise ValueError("worker-group CUDA devices must be unique")
        if len({key.site_id for key in self.keys}) != 1:
            raise ValueError("worker-group GPUs must belong to one site")
        if len({(key.model, key.runtime_fingerprint) for key in self.keys}) != 1:
            raise ValueError("worker-group actors require one model and runtime identity")
        if self.num_cpus <= 0 or self.memory_bytes < len(self.keys):
            raise ValueError("worker-group CPU and memory reservations must be positive")
        requirements, working_dir = validate_runtime_references(self.pip, self.working_dir)
        object.__setattr__(self, "pip", requirements)
        object.__setattr__(self, "working_dir", working_dir)
        object.__setattr__(self, "env", validate_job_environment(self.env))

    @property
    def worker_specs(self) -> tuple[RayActorSpec, ...]:
        cpu_per_worker = self.num_cpus / len(self.keys)
        base_memory, extra_workers = divmod(self.memory_bytes, len(self.keys))
        return tuple(
            RayActorSpec(
                key=key,
                cuda_visible_device=device,
                namespace=self.namespace,
                num_cpus=cpu_per_worker,
                memory_bytes=base_memory + int(rank < extra_workers),
                pip=self.pip,
                env=self.env,
                working_dir=self.working_dir,
            )
            for rank, (key, device) in enumerate(
                zip(self.keys, self.cuda_visible_devices, strict=True)
            )
        )


def _runtime_env(
    spec: RayActorSpec,
    *,
    extra_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    extra = extra_environment or {}
    environment = {**dict(spec.env), **dict(extra)}
    environment["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_device
    environment[RAY_ALLOWED_JOB_KEYS_ENVIRONMENT] = encode_ray_worker_allowed_job_keys(
        {*spec.env, *extra}
    )
    result: dict[str, Any] = {
        "env_vars": environment,
        # Ray itself adds private coordinates; the setup hook strips inherited
        # site credentials while retaining only this job's validated keys.
        "worker_process_setup_hook": RAY_WORKER_ENVIRONMENT_SETUP_HOOK,
    }
    if spec.pip:
        result["pip"] = list(spec.pip)
    if spec.working_dir is not None:
        result["working_dir"] = spec.working_dir
    return result


def _actor_options(spec: RayActorSpec, *, detached: bool) -> dict[str, Any]:
    options: dict[str, Any] = {
        # Physical admission happened before this call.  num_gpus=0 avoids Ray
        # remapping CUDA devices; exact UUID resources are the authority.
        "num_gpus": 0,
        "num_cpus": spec.num_cpus,
        # One Ray actor requesting every UUID resource is itself an atomic gang
        # reservation: Ray either schedules the complete model replica on this
        # host or starts no process at all.
        "resources": {name: 1 for name in spec.resource_names},
        "max_concurrency": 4,
        "runtime_env": _runtime_env(spec),
    }
    if spec.memory_bytes is not None:
        options["memory"] = spec.memory_bytes
    if detached:
        options.update(
            {
                "name": spec.actor_name,
                "namespace": spec.namespace,
                "lifetime": "detached",
            }
        )
    return options


def _worker_group_options(
    spec: RayActorSpec,
    *,
    placement_group: Any,
    rank: int,
    world_size: int,
    master_port: int,
) -> dict[str, Any]:
    options = _actor_options(spec, detached=False)
    options["runtime_env"] = _runtime_env(
        spec,
        extra_environment={
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
        },
    )
    # Placement groups reserve every UUID bundle before any worker starts. The
    # actor is then constrained to its corresponding bundle.
    options.update(
        {
            "placement_group": placement_group,
            "placement_group_bundle_index": rank,
            "placement_group_capture_child_tasks": True,
        }
    )
    return options


def _rendezvous_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class _RayActorProcess:
    """Implementation instantiated inside a Ray worker process.

    This class intentionally has no decorator so importing Topodrome never
    imports Ray.  The adapter wraps it with ``ray.remote`` only on demand.
    """

    def __init__(
        self,
        key: ActorKey,
        loader: ModelLoader,
        cuda_visible_device: str,
        dirty_on_experiment_error: bool,
    ) -> None:
        # runtime_env sets this before worker startup; repeat it here before the
        # loader runs to make the invariant explicit and independently testable.
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
        self._actor = LocalModelActor(
            key,
            loader,
            dirty_on_experiment_error=dirty_on_experiment_error,
        )

    def describe(self) -> dict[str, Any]:
        return self._actor.describe().as_dict()

    def run(self, experiment: Any, runtime_options: Mapping[str, Any]) -> Any:
        return self._actor.run(experiment, **dict(runtime_options))

    def request_preemption(self, reason: str) -> bool:
        return self._actor.request_preemption(reason)

    def replace_model(self, key: ActorKey, loader: ModelLoader, expected: ActorKey) -> str:
        return self._actor.replace_model(key, loader, expected=expected)

    def shutdown(self, force: bool = False) -> None:
        self._actor.evict(force=force)


def _load_ray() -> Any:
    # Ray 2.58+ detects an ancestor ``uv run`` process and, by default, builds
    # another runtime environment inside Ray's packaged working directory.
    # Site agents already run from Topodrome's locked uv environment; nesting a
    # second environment both defeats that lock and can omit Ray itself.
    configure_ray_driver_environment()
    try:
        return importlib.import_module("ray")
    except ModuleNotFoundError as exc:
        raise RayUnavailableError(
            "Ray support is optional; install Topodrome with its Ray/runtime extra"
        ) from exc


def _actor_not_found(ray: Any, error: Exception) -> bool:
    actor_not_found = getattr(getattr(ray, "exceptions", None), "ActorNotFoundError", None)
    return isinstance(error, ValueError) or (
        isinstance(actor_not_found, type) and isinstance(error, actor_not_found)
    )


class RayActorHandle:
    """Small facade that keeps ObjectRef behavior explicit to callers."""

    def __init__(
        self,
        ray: Any,
        handle: Any,
        spec: RayActorSpec,
        *,
        reused: bool,
        timeouts: RayOperationTimeouts,
    ) -> None:
        self._ray = ray
        self._handle = handle
        self.spec = spec
        self.reused = reused
        self._timeouts = timeouts
        self._lifecycle_lock = RLock()
        self._evicted = False

    @property
    def raw_handle(self) -> Any:
        return self._handle

    def submit(self, experiment: Any, **runtime_options: Any) -> Any:
        """Return a Ray ObjectRef for a serialized experiment call."""

        return _bounded_call(
            "Ray actor execution submission",
            self._timeouts.control_seconds,
            self._handle.run.remote,
            experiment,
            runtime_options,
        )

    def run(self, experiment: Any, **runtime_options: Any) -> Any:
        try:
            return _ray_get(
                self._ray,
                self.submit(experiment, **runtime_options),
                timeout_seconds=self._timeouts.execute_seconds,
                operation="Ray actor execution",
            )
        finally:
            # A cleanup failure disposes the in-process model. Also terminate
            # the detached Ray shell so it releases the per-UUID resource.
            self.reap_if_unusable()

    def describe(self) -> Mapping[str, Any]:
        reference = _bounded_call(
            "Ray actor liveness probe submission",
            self._timeouts.control_seconds,
            self._handle.describe.remote,
        )
        return cast(
            Mapping[str, Any],
            _ray_get(
                self._ray,
                reference,
                timeout_seconds=self._timeouts.control_seconds,
                operation="Ray actor liveness probe",
            ),
        )

    def request_preemption(self, reason: str = "preemption requested") -> bool:
        reference = _bounded_call(
            "Ray preemption request submission",
            self._timeouts.control_seconds,
            self._handle.request_preemption.remote,
            reason,
        )
        return bool(
            _ray_get(
                self._ray,
                reference,
                timeout_seconds=self._timeouts.control_seconds,
                operation="Ray preemption request",
            )
        )

    def reap_if_unusable(self) -> bool:
        with self._lifecycle_lock:
            if self._evicted:
                return True
        try:
            description = self.describe()
        except Exception:
            # A failed liveness probe is not proof that the actor is gone.
            # Force and confirm exact-handle disposal before releasing slots.
            self.evict(force=True)
            return True
        state = ActorState(description["state"])
        if not description.get("dirty") and state in (ActorState.IDLE, ActorState.BUSY):
            return False
        self.evict(force=True)
        return True

    def evict(self, *, force: bool = False) -> bool:
        """Evict and confirm this exact handle, never a same-name replacement."""

        with self._lifecycle_lock:
            if self._evicted:
                return False
            if force:
                _kill_and_confirm_actor(
                    self._ray,
                    self._handle,
                    timeouts=self._timeouts,
                )
            else:
                with suppress(BaseException):
                    reference = _bounded_call(
                        "Ray actor shutdown submission",
                        self._timeouts.control_seconds,
                        self._handle.shutdown.remote,
                        False,
                    )
                    _ray_get(
                        self._ray,
                        reference,
                        timeout_seconds=self._timeouts.disposal_seconds,
                        operation="Ray actor shutdown",
                    )
                # Graceful shutdown is best effort, but exact forced
                # termination and confirmation remain mandatory.
                _kill_and_confirm_actor(
                    self._ray,
                    self._handle,
                    timeouts=self._timeouts,
                )
            with self._lifecycle_lock:
                self._evicted = True
            return True


class RayWorkerGroupHandle:
    """Exact handles for a private all-ranks execution group."""

    def __init__(
        self,
        ray: Any,
        placement_group: Any,
        workers: Sequence[RayActorHandle],
        *,
        timeouts: RayOperationTimeouts,
    ) -> None:
        if len(workers) < 2:
            raise ValueError("a worker-group handle requires at least two workers")
        self._ray = ray
        self._placement_group = placement_group
        self._workers = tuple(workers)
        self._timeouts = timeouts
        self.group_id = f"actor-group-{uuid4()}"
        self._lifecycle_lock = RLock()
        self._evicted = False

    @property
    def specs(self) -> tuple[RayActorSpec, ...]:
        return tuple(worker.spec for worker in self._workers)

    def describe(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(worker.describe() for worker in self._workers)

    def run(self, experiment: Any, **runtime_options: Any) -> Any:
        references: list[Any] = []
        try:
            for rank, worker in enumerate(self._workers):
                options = _rank_runtime_options(
                    runtime_options, rank=rank, world_size=len(self._workers)
                )
                references.append(worker.submit(experiment, **options))
            results = _ray_get(
                self._ray,
                references,
                timeout_seconds=self._timeouts.execute_seconds,
                operation="Ray worker-group execution",
            )
            if not isinstance(results, Sequence) or len(results) != len(self._workers):
                raise RayActorError("Ray worker group returned an invalid result set")
            return results[0]
        except BaseException:
            with suppress(BaseException):
                self.evict(force=True)
            raise

    def request_preemption(self, reason: str = "preemption requested") -> bool:
        references = [
            _bounded_call(
                "Ray worker preemption request submission",
                self._timeouts.control_seconds,
                worker.raw_handle.request_preemption.remote,
                reason,
            )
            for worker in self._workers
        ]
        return any(
            bool(value)
            for value in _ray_get(
                self._ray,
                references,
                timeout_seconds=self._timeouts.control_seconds,
                operation="Ray worker-group preemption request",
            )
        )

    def reap_if_unusable(self) -> bool:
        with self._lifecycle_lock:
            if self._evicted:
                return True
        unusable = any(worker.reap_if_unusable() for worker in self._workers)
        if unusable:
            self.evict(force=True)
        return unusable

    def evict(self, *, force: bool = False) -> bool:
        """Stop only this group's exact workers, then release its reservation."""

        with self._lifecycle_lock:
            if self._evicted:
                return False
            error: BaseException | None = None
            for worker in self._workers:
                try:
                    worker.evict(force=force)
                except BaseException as exc:
                    error = error or exc
            try:
                _bounded_call(
                    "Ray placement-group removal",
                    self._timeouts.control_seconds,
                    self._ray.util.remove_placement_group,
                    self._placement_group,
                )
                _confirm_placement_group_removed(
                    self._ray,
                    self._placement_group,
                    timeouts=self._timeouts,
                )
            except BaseException as exc:
                error = error or exc
            if error is not None:
                raise RayDisposalError(
                    "failed to confirm disposal of the complete Ray worker group"
                ) from error
            self._evicted = True
            return True


def _rank_runtime_options(
    options: Mapping[str, Any], *, rank: int, world_size: int
) -> dict[str, Any]:
    result = dict(options)
    result.update(rank=rank, world_size=world_size, local_rank=0)
    if rank == 0:
        return result
    # Only rank zero owns durable attempt output. Other ranks still execute and
    # participate in collectives, but cannot race checkpoint/artifact commits.
    result["checkpoint_handler"] = None
    result["artifact_handler"] = None
    raw_kwargs = result.get("kwargs")
    if isinstance(raw_kwargs, Mapping):
        rank_kwargs = dict(raw_kwargs)
        log_path = rank_kwargs.get("log_path")
        if isinstance(log_path, str):
            path = Path(log_path)
            rank_kwargs["log_path"] = str(path.with_name(f"{path.stem}.rank-{rank}{path.suffix}"))
        result["kwargs"] = rank_kwargs
    return result


class DetachedRayActorAdapter:
    """Create, reconnect to, reuse, and replace detached Ray model actors."""

    def __init__(
        self,
        ray_module: Any | None = None,
        *,
        timeouts: RayOperationTimeouts | None = None,
    ) -> None:
        # Tests may inject a compatible facade.  Normal imports remain lazy.
        self._ray_module = ray_module
        self.timeouts = timeouts or RayOperationTimeouts()

    @property
    def ray(self) -> Any:
        if self._ray_module is None:
            self._ray_module = _load_ray()
        return self._ray_module

    def _find(self, spec: RayActorSpec) -> Any | None:
        try:
            return _bounded_call(
                "Ray named-actor lookup",
                self.timeouts.control_seconds,
                self.ray.get_actor,
                spec.actor_name,
                namespace=spec.namespace,
            )
        except RayActorTimeoutError as exc:
            # A lookup timeout cannot prove that the named slot is empty.  The
            # caller must retain the physical fence until reconciliation can
            # establish the actor's exact state.
            raise RayDisposalError("Ray named-actor lookup outcome is unconfirmed") from exc
        except Exception as exc:
            if _actor_not_found(self.ray, exc):
                return None
            raise

    def _create(
        self,
        spec: RayActorSpec,
        loader: ModelLoader,
        *,
        dirty_on_experiment_error: bool,
        detached: bool = True,
    ) -> Any:
        if bool(getattr(loader, "contains_credentials", False)):
            raise ValueError(
                "credential-bearing loaders cannot be sent to Ray; acquire the model "
                "in the site service and create a credential-free local loader"
            )
        remote_class = _bounded_call(
            "Ray actor-class registration",
            self.timeouts.control_seconds,
            self.ray.remote,
            _RayActorProcess,
        )
        options = _actor_options(spec, detached=detached)
        try:
            configured = _bounded_call(
                "Ray actor option preparation",
                self.timeouts.control_seconds,
                remote_class.options,
                **options,
            )
            try:
                return _bounded_call(
                    "Ray actor creation",
                    self.timeouts.prepare_seconds,
                    configured.remote,
                    spec.key,
                    loader,
                    spec.cuda_visible_device,
                    dirty_on_experiment_error,
                )
            except RayActorTimeoutError as exc:
                # The detached named actor may appear after the local call has
                # returned.  Without its exact handle, cleanup is unconfirmed.
                raise RayDisposalError("Ray actor creation outcome is unconfirmed") from exc
        except ValueError:
            if not detached:
                raise
            # A concurrent site-agent operation may have won named creation.
            existing = self._find(spec)
            if existing is None:
                raise
            return existing

    def create_private(self, spec: RayActorSpec, loader: ModelLoader) -> RayActorHandle:
        """Create an unnamed attempt-owned process, never a reusable warm shell."""
        handle = self._create(spec, loader, dirty_on_experiment_error=True, detached=False)
        return RayActorHandle(self.ray, handle, spec, reused=False, timeouts=self.timeouts)

    def get_or_create(
        self,
        spec: RayActorSpec,
        loader: ModelLoader,
        *,
        dirty_on_experiment_error: bool = True,
        expected_switch_actor_id: str | None = None,
    ) -> RayActorHandle:
        handle = self._find(spec)
        if handle is not None:
            current = RayActorHandle(
                self.ray,
                handle,
                spec,
                reused=True,
                timeouts=self.timeouts,
            )
            description = current.describe()
            if (
                expected_switch_actor_id is not None
                and description["actor_id"] != expected_switch_actor_id
            ):
                raise RayDisposalError("runtime replacement actor identity changed")
            existing_key = ActorKey.from_mapping(description["key"])
            state = ActorState(description["state"])
            dirty = bool(description.get("dirty"))
            if (
                existing_key == spec.key
                and state in (ActorState.IDLE, ActorState.BUSY)
                and not dirty
            ):
                return current
            if state == ActorState.BUSY or int(description.get("pending_runs", 0)):
                raise IncompatibleActorBusyError(
                    f"GPU {spec.key.gpu_uuid!r} has an incompatible busy Ray actor"
                )
            if (
                state is ActorState.IDLE
                and not dirty
                and expected_switch_actor_id is not None
                and description.get("supports_model_switch")
                and callable(getattr(loader, "replace", None))
                and existing_key.site_id == spec.key.site_id
                and existing_key.replica_gpu_uuids == spec.key.replica_gpu_uuids
                and existing_key.runtime_fingerprint == spec.key.runtime_fingerprint
                and existing_key.model is not None
                and spec.key.model is not None
                and existing_key.model.backend == spec.key.model.backend == "transformers"
                and existing_key.model.backend_version == spec.key.model.backend_version
                and not any(
                    (
                        existing_key.model.quantization,
                        existing_key.model.adapter,
                        spec.key.model.quantization,
                        spec.key.model.adapter,
                    )
                )
            ):
                if bool(getattr(loader, "contains_credentials", False)):
                    raise ValueError("credential-bearing loaders cannot be sent to Ray")
                try:
                    reference = _bounded_call(
                        "Ray checkpoint replacement submission",
                        self.timeouts.control_seconds,
                        handle.replace_model.remote,
                        spec.key,
                        loader,
                        existing_key,
                    )
                    _ray_get(
                        self.ray,
                        reference,
                        timeout_seconds=self.timeouts.prepare_seconds,
                        operation="Ray checkpoint replacement",
                    )
                    replaced = current.describe()
                    if (
                        ActorKey.from_mapping(replaced["key"]) != spec.key
                        or replaced["actor_id"] != description["actor_id"]
                        or replaced["state"] != "idle"
                        or replaced.get("dirty")
                    ):
                        raise RayActorError("checkpoint replacement identity was not confirmed")
                except BaseException as switch_error:
                    try:
                        current.evict(force=True)
                    except BaseException as cleanup_error:
                        raise RayDisposalError(
                            "checkpoint replacement disposal unconfirmed"
                        ) from cleanup_error
                    raise switch_error
                # This is not an exact-model warm hit: retain startup accounting.
                current.reused = False
                return current
            if expected_switch_actor_id is not None:
                raise RayDisposalError("runtime replacement worker is no longer compatible")
            current.evict(force=False)

        if expected_switch_actor_id is not None:
            raise RayDisposalError("runtime replacement worker disappeared")

        handle = self._create(
            spec,
            loader,
            dirty_on_experiment_error=dirty_on_experiment_error,
        )
        created = RayActorHandle(
            self.ray,
            handle,
            spec,
            reused=False,
            timeouts=self.timeouts,
        )
        try:
            # Resolve construction now. Loader failures surface during prepare,
            # not after the scheduler has announced the attempt as running.
            reference = _bounded_call(
                "Ray model-construction probe submission",
                self.timeouts.control_seconds,
                handle.describe.remote,
            )
            _ray_get(
                self.ray,
                reference,
                timeout_seconds=self.timeouts.prepare_seconds,
                operation="Ray model construction",
            )
        except BaseException as prepare_error:
            try:
                created.evict(force=True)
            except BaseException as cleanup_error:
                raise RayDisposalError(
                    "failed Ray model preparation left an unconfirmed actor"
                ) from cleanup_error
            raise prepare_error
        return created

    def create_worker_group(
        self,
        spec: RayWorkerGroupSpec,
        loader: ModelLoader,
        *,
        dirty_on_experiment_error: bool = True,
    ) -> RayWorkerGroupHandle:
        """Atomically reserve all UUID bundles and launch one private rank each."""

        if bool(getattr(loader, "contains_credentials", False)):
            raise ValueError(
                "credential-bearing loaders cannot be sent to Ray; acquire the model "
                "in the site service and create a credential-free local loader"
            )
        worker_specs = spec.worker_specs
        bundles: list[dict[str, float | int]] = []
        for worker_spec in worker_specs:
            bundle: dict[str, float | int] = {
                "CPU": worker_spec.num_cpus,
                worker_spec.resource_name: 1,
            }
            if worker_spec.memory_bytes is not None:
                bundle["memory"] = worker_spec.memory_bytes
            bundles.append(bundle)
        try:
            placement_group = _bounded_call(
                "Ray placement-group creation",
                self.timeouts.control_seconds,
                self.ray.util.placement_group,
                bundles,
                strategy="STRICT_PACK",
            )
        except RayActorTimeoutError as exc:
            # A late placement group has no handle through which this process
            # can prove removal, so the complete gang must remain fenced.
            raise RayDisposalError("Ray placement-group creation outcome is unconfirmed") from exc
        workers: list[RayActorHandle] = []
        try:
            ready = _bounded_call(
                "Ray placement-group readiness submission",
                self.timeouts.control_seconds,
                placement_group.ready,
            )
            _ray_get(
                self.ray,
                ready,
                timeout_seconds=self.timeouts.prepare_seconds,
                operation="Ray placement-group readiness",
            )
            remote_class = _bounded_call(
                "Ray actor-class registration",
                self.timeouts.control_seconds,
                self.ray.remote,
                _RayActorProcess,
            )
            master_port = _rendezvous_port()
            for rank, worker_spec in enumerate(worker_specs):
                configured = _bounded_call(
                    "Ray worker option preparation",
                    self.timeouts.control_seconds,
                    remote_class.options,
                    **_worker_group_options(
                        worker_spec,
                        placement_group=placement_group,
                        rank=rank,
                        world_size=len(worker_specs),
                        master_port=master_port,
                    ),
                )
                try:
                    raw_handle = _bounded_call(
                        "Ray worker creation",
                        self.timeouts.prepare_seconds,
                        configured.remote,
                        worker_spec.key,
                        loader,
                        worker_spec.cuda_visible_device,
                        dirty_on_experiment_error,
                    )
                except RayActorTimeoutError as exc:
                    # A private worker may still materialize after the local
                    # timeout. Placement-group removal is attempted below, but
                    # that does not prove the late actor never started.
                    raise RayDisposalError("Ray worker creation outcome is unconfirmed") from exc
                workers.append(
                    RayActorHandle(
                        self.ray,
                        raw_handle,
                        worker_spec,
                        reused=False,
                        timeouts=self.timeouts,
                    )
                )
            # Resolve every constructor before acknowledging runtime preparation.
            descriptions = [
                _bounded_call(
                    "Ray worker-construction probe submission",
                    self.timeouts.control_seconds,
                    worker.raw_handle.describe.remote,
                )
                for worker in workers
            ]
            _ray_get(
                self.ray,
                descriptions,
                timeout_seconds=self.timeouts.prepare_seconds,
                operation="Ray worker-group model construction",
            )
            return RayWorkerGroupHandle(
                self.ray,
                placement_group,
                workers,
                timeouts=self.timeouts,
            )
        except BaseException as prepare_error:
            cleanup_error: BaseException | None = None
            for worker_handle in workers:
                try:
                    worker_handle.evict(force=True)
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
            try:
                _bounded_call(
                    "Ray placement-group removal",
                    self.timeouts.control_seconds,
                    self.ray.util.remove_placement_group,
                    placement_group,
                )
                _confirm_placement_group_removed(
                    self.ray,
                    placement_group,
                    timeouts=self.timeouts,
                )
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            if cleanup_error is not None:
                raise RayDisposalError(
                    "failed Ray group preparation left unconfirmed resources"
                ) from cleanup_error
            raise prepare_error

    def reconnect(
        self,
        spec: RayActorSpec,
        *,
        expected_actor_id: str,
    ) -> RayActorHandle | None:
        """Reconnect only to the exact clean, idle detached actor persisted locally."""

        handle = self._find(spec)
        if handle is None:
            return None
        wrapped = RayActorHandle(
            self.ray,
            handle,
            spec,
            reused=True,
            timeouts=self.timeouts,
        )
        try:
            description = wrapped.describe()
        except BaseException as exc:
            if _actor_stopped_error(self.ray, exc):
                return None
            raise
        if str(description.get("actor_id")) != expected_actor_id:
            raise RayDisposalError(
                "named Ray slot contains an actor outside the persisted ownership record"
            )
        reusable = (
            ActorKey.from_mapping(description["key"]) == spec.key
            and ActorState(description["state"]) is ActorState.IDLE
            and not bool(description.get("dirty"))
            and not int(description.get("pending_runs", 0))
        )
        if not reusable:
            # This is the exact actor persisted by the local authority. It is
            # safe to dispose, but stale bookkeeping may be dropped only after
            # that disposal is confirmed.
            wrapped.evict(force=True)
            return None
        return wrapped

    def evict(self, spec: RayActorSpec, *, force: bool = False) -> bool:
        handle = self._find(spec)
        if handle is None:
            return False
        wrapped = RayActorHandle(
            self.ray,
            handle,
            spec,
            reused=False,
            timeouts=self.timeouts,
        )
        return wrapped.evict(force=force)
