"""High-level execution facade for site-agent integrations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any, cast
from uuid import uuid4

from .actor import ActorManager, ActorState, LocalModelActor, ModelLoader
from .checkpoint import CheckpointHandler, CheckpointRestore, PreemptionToken
from .context import ArtifactRegistration
from .model_key import ActorKey
from .ray_actor import (
    DetachedRayActorAdapter,
    RayActorHandle,
    RayActorSpec,
    RayDisposalError,
    RayWorkerGroupHandle,
    RayWorkerGroupSpec,
)


class ExecutionTransport(StrEnum):
    LOCAL = "local"
    RAY = "ray"
    RAY_SHARDED = "ray_sharded"
    RAY_GROUP = "ray_group"


class RuntimeExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedActor:
    """A warm allocation prepared before an attempt is announced as running."""

    actor_id: str
    worker_pids: tuple[int, ...]
    key: ActorKey
    reused: bool
    transport: ExecutionTransport
    _handle: LocalModelActor | RayActorHandle | RayWorkerGroupHandle = field(
        repr=False,
        compare=False,
    )
    _owner: str = field(repr=False, compare=False)
    group_keys: tuple[ActorKey, ...] = ()


@dataclass(slots=True)
class _ActiveExecution:
    prepared: PreparedActor
    preemption: PreemptionToken


def _describe_prepared_ray_handle(
    handle: RayActorHandle | RayWorkerGroupHandle,
) -> Mapping[str, Any] | tuple[Mapping[str, Any], ...]:
    """Describe a just-prepared runtime or confirm its complete disposal.

    Preparation is not complete until worker identity has been collected.  A
    bounded describe failure after actor creation must therefore not strand an
    untracked GPU owner while the site releases its lease.
    """

    try:
        return handle.describe()
    except BaseException as describe_error:
        try:
            handle.evict(force=True)
        except BaseException as cleanup_error:
            raise RayDisposalError(
                "prepared Ray runtime identity failed and disposal is unconfirmed"
            ) from cleanup_error
        raise describe_error


class RuntimeExecutionService:
    """Prepare warm actors and execute attempts without site-agent imports.

    This synchronous core is suitable for a dedicated agent worker thread.  The
    accompanying :class:`AsyncRuntimeExecutionService` uses ``asyncio.to_thread``
    and structurally matches the prepare/execute/preempt/cancel/evict shape used
    by site agents.
    """

    def __init__(
        self,
        *,
        actors: ActorManager | None = None,
        ray: DetachedRayActorAdapter | None = None,
    ) -> None:
        self.actors = actors or ActorManager()
        self.ray = ray
        self._owner = f"runtime-{uuid4()}"
        self._lock = RLock()
        self._active: dict[str, _ActiveExecution] = {}

    def prepare(
        self,
        key: ActorKey,
        loader: ModelLoader,
        *,
        use_ray: bool = False,
        cuda_visible_device: str | None = None,
        ray_namespace: str = "topodrome",
        num_cpus: float = 1.0,
        memory_bytes: int | None = None,
        pip: tuple[str, ...] = (),
        env: Mapping[str, str] | None = None,
        working_dir: str | None = None,
    ) -> PreparedActor:
        if use_ray:
            if self.ray is None:
                raise RuntimeExecutionError("Ray execution was requested without a Ray adapter")
            if cuda_visible_device is None:
                raise RuntimeExecutionError("Ray execution requires an explicit CUDA device")
            spec = RayActorSpec(
                key=key,
                cuda_visible_device=cuda_visible_device,
                namespace=ray_namespace,
                num_cpus=num_cpus,
                memory_bytes=memory_bytes,
                pip=pip,
                env=env or {},
                working_dir=working_dir,
            )
            handle = self.ray.get_or_create(spec, loader)
            description = cast(Mapping[str, Any], _describe_prepared_ray_handle(handle))
            return PreparedActor(
                actor_id=str(description["actor_id"]),
                worker_pids=(int(description["worker_pid"]),),
                key=key,
                reused=handle.reused,
                transport=ExecutionTransport.RAY,
                _handle=handle,
                _owner=self._owner,
                group_keys=(key,),
            )

        if pip or env or working_dir is not None:
            raise RuntimeExecutionError("custom runtime environments require Ray execution")

        existing = self.actors.get(*key.slot)
        reused = bool(existing and existing.key == key and existing.is_reusable)
        actor = self.actors.get_or_create(key, loader)
        return PreparedActor(
            actor_id=actor.actor_id,
            worker_pids=(actor.worker_pid,),
            key=key,
            reused=reused,
            transport=ExecutionTransport.LOCAL,
            _handle=actor,
            _owner=self._owner,
            group_keys=(key,),
        )

    def prepare_group(
        self,
        keys: tuple[ActorKey, ...],
        loader: ModelLoader,
        *,
        cuda_visible_devices: tuple[str, ...],
        ray_namespace: str = "topodrome",
        num_cpus: float = 1.0,
        memory_bytes: int = 1024**3,
        pip: tuple[str, ...] = (),
        env: Mapping[str, str] | None = None,
        working_dir: str | None = None,
    ) -> PreparedActor:
        if self.ray is None:
            raise RuntimeExecutionError("Ray worker-group execution requires a Ray adapter")
        group_spec = RayWorkerGroupSpec(
            keys=keys,
            cuda_visible_devices=cuda_visible_devices,
            namespace=ray_namespace,
            num_cpus=num_cpus,
            memory_bytes=memory_bytes,
            pip=pip,
            env=env or {},
            working_dir=working_dir,
        )
        handle = self.ray.create_worker_group(group_spec, loader)
        descriptions = cast(
            tuple[Mapping[str, Any], ...],
            _describe_prepared_ray_handle(handle),
        )
        return PreparedActor(
            actor_id=handle.group_id,
            worker_pids=tuple(int(description["worker_pid"]) for description in descriptions),
            key=keys[0],
            reused=False,
            transport=ExecutionTransport.RAY_GROUP,
            _handle=handle,
            _owner=self._owner,
            group_keys=keys,
        )

    def prepare_sharded(
        self,
        key: ActorKey,
        loader: ModelLoader,
        *,
        cuda_visible_devices: tuple[str, ...],
        strategy: str,
        max_memory_bytes_per_gpu: int,
        ray_namespace: str = "topodrome",
        num_cpus: float = 1.0,
        memory_bytes: int | None = None,
        pip: tuple[str, ...] = (),
        env: Mapping[str, str] | None = None,
        working_dir: str | None = None,
    ) -> PreparedActor:
        """Prepare one reusable actor owning an exact ordered GPU gang."""

        if self.ray is None:
            raise RuntimeExecutionError("model-sharded execution requires a Ray adapter")
        if len(key.replica_gpu_uuids) < 2:
            raise RuntimeExecutionError("model-sharded execution requires at least two GPUs")
        if len(cuda_visible_devices) != len(key.replica_gpu_uuids):
            raise RuntimeExecutionError("one CUDA device is required for every replica member")
        configure = getattr(loader, "for_model_sharding", None)
        if not callable(configure):
            raise RuntimeExecutionError(
                "the selected loader does not implement model-sharded inference"
            )
        sharded_loader = configure(
            degree=len(key.replica_gpu_uuids),
            strategy=strategy,
            max_memory_bytes_per_gpu=max_memory_bytes_per_gpu,
        )
        spec = RayActorSpec(
            key=key,
            cuda_visible_device=",".join(cuda_visible_devices),
            namespace=ray_namespace,
            num_cpus=num_cpus,
            memory_bytes=memory_bytes,
            pip=pip,
            env=env or {},
            working_dir=working_dir,
        )
        handle = self.ray.get_or_create(spec, sharded_loader)
        description = cast(Mapping[str, Any], _describe_prepared_ray_handle(handle))
        return PreparedActor(
            actor_id=str(description["actor_id"]),
            worker_pids=(int(description["worker_pid"]),),
            key=key,
            reused=handle.reused,
            transport=ExecutionTransport.RAY_SHARDED,
            _handle=handle,
            _owner=self._owner,
            group_keys=(key,),
        )

    def execute(
        self,
        prepared: PreparedActor,
        experiment: Any,
        *,
        job_id: str,
        attempt_id: str,
        kwargs: Mapping[str, Any] | None = None,
        workspace: Path | str | None = None,
        committed_inputs: Mapping[str, Path | str] | None = None,
        scratch: Path | str | None = None,
        checkpoint_handler: CheckpointHandler | None = None,
        checkpoint_interval_seconds: float | None = None,
        checkpoint_restore: CheckpointRestore | None = None,
        artifact_handler: Callable[[ArtifactRegistration], Any] | None = None,
    ) -> Any:
        self._validate_prepared(prepared)
        preemption = PreemptionToken()
        with self._lock:
            if attempt_id in self._active:
                raise RuntimeExecutionError(f"attempt is already running: {attempt_id!r}")
            self._active[attempt_id] = _ActiveExecution(prepared, preemption)

        try:
            if prepared.transport is ExecutionTransport.LOCAL:
                actor = prepared._handle
                assert isinstance(actor, LocalModelActor)
                current = self.actors.get(*prepared.key.slot)
                if current is not actor or actor.actor_id != prepared.actor_id:
                    raise RuntimeExecutionError("prepared local actor is no longer allocated")
                return actor.run(
                    experiment,
                    job_id=job_id,
                    attempt_id=attempt_id,
                    kwargs=kwargs,
                    workspace=workspace,
                    committed_inputs=committed_inputs,
                    scratch=scratch,
                    preemption=preemption,
                    checkpoint_handler=checkpoint_handler,
                    checkpoint_interval_seconds=checkpoint_interval_seconds,
                    checkpoint_restore=checkpoint_restore,
                    artifact_handler=artifact_handler,
                )
            handle = prepared._handle
            assert isinstance(handle, (RayActorHandle, RayWorkerGroupHandle))
            # The token is useful only in-process. Ray preemption uses the
            # actor's concurrent control method, so do not serialize the token.
            return handle.run(
                experiment,
                job_id=job_id,
                attempt_id=attempt_id,
                kwargs=kwargs,
                workspace=workspace,
                committed_inputs=committed_inputs,
                scratch=scratch,
                checkpoint_handler=checkpoint_handler,
                checkpoint_interval_seconds=checkpoint_interval_seconds,
                checkpoint_restore=checkpoint_restore,
                artifact_handler=artifact_handler,
            )
        finally:
            with self._lock:
                self._active.pop(attempt_id, None)
            if prepared.transport is ExecutionTransport.LOCAL:
                actor = prepared._handle
                assert isinstance(actor, LocalModelActor)
                self.actors.reap(actor)
            else:
                handle = prepared._handle
                assert isinstance(handle, (RayActorHandle, RayWorkerGroupHandle))
                handle.reap_if_unusable()

    def request_preemption(self, attempt_id: str, reason: str = "preemption requested") -> bool:
        with self._lock:
            active = self._active.get(attempt_id)
        if active is None:
            return False
        if active.prepared.transport is ExecutionTransport.LOCAL:
            return active.preemption.request(reason)
        handle = active.prepared._handle
        assert isinstance(handle, (RayActorHandle, RayWorkerGroupHandle))
        return handle.request_preemption(reason)

    def cancel(self, attempt_id: str) -> bool:
        """Request cooperative cancellation at the experiment's next boundary."""

        return self.request_preemption(attempt_id, "attempt canceled")

    def evict(self, prepared: PreparedActor, *, force: bool = False) -> bool:
        self._validate_prepared(prepared)
        if prepared.transport is ExecutionTransport.LOCAL:
            return self.actors.evict(*prepared.key.slot, force=force)
        if self.ray is None:
            return False
        handle = prepared._handle
        assert isinstance(handle, (RayActorHandle, RayWorkerGroupHandle))
        return handle.evict(force=force)

    def force_stop(self, prepared: PreparedActor) -> bool:
        """Hard-stop a tracked Ray actor after cooperative grace expires."""

        self._validate_prepared(prepared)
        if prepared.transport is ExecutionTransport.LOCAL:
            # A local actor is the site-agent process itself and must never be
            # signalled as though it were an isolated worker.
            return False
        if self.ray is None:
            return False
        handle = prepared._handle
        assert isinstance(handle, (RayActorHandle, RayWorkerGroupHandle))
        return handle.evict(force=True)

    def is_live(self, prepared: PreparedActor, *, require_idle: bool = False) -> bool:
        """Probe the exact prepared process/group before reuse or advertisement."""

        self._validate_prepared(prepared)
        if prepared.transport is ExecutionTransport.LOCAL:
            actor = prepared._handle
            assert isinstance(actor, LocalModelActor)
            current = self.actors.get(*prepared.key.slot)
            if current is not actor or actor.actor_id != prepared.actor_id:
                return False
            description = actor.describe().as_dict()
            state = ActorState(description["state"])
            return (
                not bool(description.get("dirty"))
                and (
                    state is ActorState.IDLE
                    if require_idle
                    else state in (ActorState.IDLE, ActorState.BUSY)
                )
            )
        handle = prepared._handle
        assert isinstance(handle, (RayActorHandle, RayWorkerGroupHandle))
        try:
            descriptions = handle.describe()
        except Exception:
            return False
        values: tuple[Mapping[str, Any], ...]
        if isinstance(handle, RayActorHandle):
            values = (cast(Mapping[str, Any], descriptions),)
        else:
            values = cast(tuple[Mapping[str, Any], ...], descriptions)
        if not values:
            return False
        if (
            isinstance(handle, RayActorHandle)
            and str(values[0].get("actor_id")) != prepared.actor_id
        ):
            return False
        states = tuple(ActorState(value["state"]) for value in values)
        allowed = (
            all(state is ActorState.IDLE for state in states)
            if require_idle
            else all(state in (ActorState.IDLE, ActorState.BUSY) for state in states)
        )
        return allowed and all(not bool(value.get("dirty")) for value in values) and (
            not require_idle
            or all(not int(value.get("pending_runs", 0)) for value in values)
        )

    def reconnect(
        self,
        key: ActorKey,
        *,
        actor_id: str,
        cuda_visible_device: str,
        ray_namespace: str = "topodrome",
    ) -> PreparedActor | None:
        if self.ray is None:
            return None
        spec = RayActorSpec(
            key=key,
            cuda_visible_device=cuda_visible_device,
            namespace=ray_namespace,
        )
        handle = self.ray.reconnect(spec, expected_actor_id=actor_id)
        if handle is None:
            return None
        description = handle.describe()
        return PreparedActor(
            actor_id=str(description["actor_id"]),
            worker_pids=(int(description["worker_pid"]),),
            key=key,
            reused=True,
            transport=(
                ExecutionTransport.RAY_SHARDED
                if len(key.replica_gpu_uuids) > 1
                else ExecutionTransport.RAY
            ),
            _handle=handle,
            _owner=self._owner,
            group_keys=(key,),
        )

    def _validate_prepared(self, prepared: PreparedActor) -> None:
        if not isinstance(prepared, PreparedActor) or prepared._owner != self._owner:
            raise RuntimeExecutionError("prepared actor belongs to another runtime service")


class AsyncRuntimeExecutionService:
    """Non-blocking facade for asyncio site agents."""

    def __init__(self, core: RuntimeExecutionService | None = None) -> None:
        self.core = core or RuntimeExecutionService()

    async def prepare(self, *args: Any, **kwargs: Any) -> PreparedActor:
        return await asyncio.to_thread(self.core.prepare, *args, **kwargs)

    async def prepare_group(self, *args: Any, **kwargs: Any) -> PreparedActor:
        return await asyncio.to_thread(self.core.prepare_group, *args, **kwargs)

    async def prepare_sharded(self, *args: Any, **kwargs: Any) -> PreparedActor:
        return await asyncio.to_thread(self.core.prepare_sharded, *args, **kwargs)

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self.core.execute, *args, **kwargs)

    async def request_preemption(
        self, attempt_id: str, reason: str = "preemption requested"
    ) -> bool:
        return await asyncio.to_thread(self.core.request_preemption, attempt_id, reason)

    async def cancel(self, attempt_id: str) -> bool:
        return await asyncio.to_thread(self.core.cancel, attempt_id)

    async def evict(self, prepared: PreparedActor, *, force: bool = False) -> bool:
        return await asyncio.to_thread(self.core.evict, prepared, force=force)

    async def force_stop(self, prepared: PreparedActor) -> bool:
        return await asyncio.to_thread(self.core.force_stop, prepared)

    async def is_live(
        self,
        prepared: PreparedActor,
        *,
        require_idle: bool = False,
    ) -> bool:
        return await asyncio.to_thread(
            self.core.is_live,
            prepared,
            require_idle=require_idle,
        )

    async def reconnect(self, *args: Any, **kwargs: Any) -> PreparedActor | None:
        return await asyncio.to_thread(self.core.reconnect, *args, **kwargs)


__all__ = [
    "AsyncRuntimeExecutionService",
    "ExecutionTransport",
    "PreparedActor",
    "RuntimeExecutionError",
    "RuntimeExecutionService",
]
