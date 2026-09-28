"""Serialized, reusable model actors and their site-local warm pool."""

from __future__ import annotations

import gc
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Protocol
from uuid import uuid4

from .checkpoint import CheckpointHandler, CheckpointRestore, PreemptionToken
from .cleanup import CleanupError
from .context import ArtifactRegistration, ExperimentContext
from .experiment import invoke_experiment
from .model_key import ActorKey, ModelKey


class ActorState(StrEnum):
    LOADING = "loading"
    IDLE = "idle"
    BUSY = "busy"
    DIRTY = "dirty"
    EVICTING = "evicting"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ActorError(RuntimeError):
    pass


class ActorUnavailableError(ActorError):
    pass


class IncompatibleActorBusyError(ActorError):
    pass


class ActorDisposalError(ActorError):
    pass


class ActorCleanupError(ActorError):
    """Cleanup failed, so the result was discarded and the actor was evicted."""

    def __init__(
        self,
        cleanup_error: CleanupError,
        experiment_error: BaseException | None = None,
    ) -> None:
        self.cleanup_error = cleanup_error
        self.experiment_error = experiment_error
        message = "experiment cleanup failed; warm actor was evicted"
        if experiment_error is not None:
            message += f" (experiment also raised {type(experiment_error).__name__})"
        super().__init__(message)


@dataclass(slots=True)
class LoadedModel:
    """Resources owned for the lifetime of a warm actor."""

    model: Any
    tokenizer: Any = None
    device: Any = "cuda"
    close: Callable[[], Any] | None = None


class ModelLoader(Protocol):
    def __call__(self, key: ModelKey) -> LoadedModel: ...


@dataclass(frozen=True, slots=True)
class ActorDescription:
    actor_id: str
    worker_pid: int
    key: ActorKey
    state: ActorState
    executions: int
    pending_runs: int
    dirty: bool
    dirty_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "worker_pid": self.worker_pid,
            "key": self.key.as_dict(),
            "state": self.state.value,
            "executions": self.executions,
            "pending_runs": self.pending_runs,
            "dirty": self.dirty,
            "dirty_reasons": list(self.dirty_reasons),
        }


class LocalModelActor:
    """Own one loaded model and serialize all experiment mutations.

    The class contains no Ray dependency and is useful both in local tests and
    inside a Ray actor process.  Concurrent callers queue on ``_execution_lock``.
    A warm actor survives successful attempts, while uncertain state causes an
    immediate disposal after the current call.
    """

    def __init__(
        self,
        key: ActorKey,
        loader: ModelLoader,
        *,
        dirty_on_experiment_error: bool = True,
    ) -> None:
        if not isinstance(key, ActorKey):
            raise TypeError("key must be an ActorKey")
        if not callable(loader):
            raise TypeError("loader must be callable")
        self.key = key
        self.actor_id = f"actor-{uuid4()}"
        self.worker_pid = os.getpid()
        self._loader = loader
        self._dirty_on_experiment_error = dirty_on_experiment_error
        self._execution_lock = Lock()
        self._state_lock = RLock()
        self._state = ActorState.LOADING
        self._loaded: LoadedModel | None = None
        self._active_context: ExperimentContext | None = None
        self._pending_runs = 0
        self._executions = 0
        self._dirty_reasons: list[str] = []
        try:
            loaded = loader(key.model)
            if not isinstance(loaded, LoadedModel):
                raise TypeError("model loader must return LoadedModel")
            self._loaded = loaded
        except BaseException:
            with self._state_lock:
                self._state = ActorState.STOPPED
            raise
        with self._state_lock:
            self._state = ActorState.IDLE

    @property
    def state(self) -> ActorState:
        with self._state_lock:
            return self._state

    @property
    def is_reusable(self) -> bool:
        with self._state_lock:
            return self._state in (ActorState.IDLE, ActorState.BUSY) and not self._dirty_reasons

    @property
    def has_pending_work(self) -> bool:
        with self._state_lock:
            return self._pending_runs > 0

    def describe(self) -> ActorDescription:
        with self._state_lock:
            return ActorDescription(
                actor_id=self.actor_id,
                worker_pid=self.worker_pid,
                key=self.key,
                state=self._state,
                executions=self._executions,
                pending_runs=self._pending_runs,
                dirty=bool(self._dirty_reasons),
                dirty_reasons=tuple(self._dirty_reasons),
            )

    def _reserve(self) -> None:
        with self._state_lock:
            if self._state not in (ActorState.IDLE, ActorState.BUSY):
                raise ActorUnavailableError(
                    f"actor {self.key.model.alias!r} is {self._state.value}"
                )
            if self._dirty_reasons:
                raise ActorUnavailableError("actor has uncertain model state")
            self._pending_runs += 1
            # BUSY also covers queued calls, so an incompatible model cannot
            # evict the allocation between reservation and actual execution.
            self._state = ActorState.BUSY

    def run(
        self,
        experiment: Any,
        *,
        job_id: str,
        attempt_id: str,
        kwargs: Mapping[str, Any] | None = None,
        workspace: Path | str | None = None,
        committed_inputs: Mapping[str, Path | str] | None = None,
        scratch: Path | str | None = None,
        preemption: PreemptionToken | None = None,
        preemption_probe: Callable[[], bool] | None = None,
        checkpoint_handler: CheckpointHandler | None = None,
        checkpoint_interval_seconds: float | None = None,
        checkpoint_restore: CheckpointRestore | None = None,
        artifact_handler: Callable[[ArtifactRegistration], Any] | None = None,
    ) -> Any:
        self._reserve()
        return self._run_reserved(
            experiment,
            job_id=job_id,
            attempt_id=attempt_id,
            kwargs=kwargs,
            workspace=workspace,
            committed_inputs=committed_inputs,
            scratch=scratch,
            preemption=preemption,
            preemption_probe=preemption_probe,
            checkpoint_handler=checkpoint_handler,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
            checkpoint_restore=checkpoint_restore,
            artifact_handler=artifact_handler,
        )

    # ActorManager reserves while holding the physical-slot lock and then uses
    # this method outside the manager lock, allowing other GPUs to progress.
    def _run_reserved(
        self,
        experiment: Any,
        *,
        job_id: str,
        attempt_id: str,
        kwargs: Mapping[str, Any] | None,
        workspace: Path | str | None,
        committed_inputs: Mapping[str, Path | str] | None,
        scratch: Path | str | None,
        preemption: PreemptionToken | None,
        preemption_probe: Callable[[], bool] | None,
        checkpoint_handler: CheckpointHandler | None,
        checkpoint_interval_seconds: float | None,
        checkpoint_restore: CheckpointRestore | None,
        artifact_handler: Callable[[ArtifactRegistration], Any] | None,
    ) -> Any:
        result: Any = None
        experiment_error: BaseException | None = None
        cleanup_error: CleanupError | None = None

        with self._execution_lock:
            with self._state_lock:
                if self._state not in (ActorState.IDLE, ActorState.BUSY):
                    self._pending_runs -= 1
                    raise ActorUnavailableError(
                        f"actor became {self._state.value} before execution"
                    )
                loaded = self._loaded
                if loaded is None:
                    self._pending_runs -= 1
                    raise ActorUnavailableError("actor model has been disposed")
                self._state = ActorState.BUSY
            try:
                context = ExperimentContext(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    device=loaded.device,
                    job_id=job_id,
                    attempt_id=attempt_id,
                    workspace=workspace,
                    committed_inputs=committed_inputs,
                    scratch=scratch,
                    preemption=preemption,
                    preemption_probe=preemption_probe,
                    checkpoint_handler=checkpoint_handler,
                    checkpoint_interval_seconds=checkpoint_interval_seconds,
                    checkpoint_restore=checkpoint_restore,
                    artifact_handler=artifact_handler,
                )
            except BaseException:
                with self._state_lock:
                    self._pending_runs -= 1
                    self._state = ActorState.BUSY if self._pending_runs else ActorState.IDLE
                raise
            with self._state_lock:
                self._active_context = context

            try:
                result = invoke_experiment(experiment, context, kwargs)
            except BaseException as exc:
                experiment_error = exc
                if self._dirty_on_experiment_error:
                    context.mark_dirty(
                        f"experiment raised {type(exc).__name__}; cleanup cannot be verified"
                    )
            finally:
                try:
                    context.close()
                except CleanupError as exc:
                    cleanup_error = exc
                    context.mark_dirty("one or more experiment cleanup callbacks failed")

                with self._state_lock:
                    self._active_context = None
                    self._executions += 1
                    self._pending_runs -= 1
                    if context.dirty:
                        self._dirty_reasons.extend(context.dirty_reasons)
                        self._state = ActorState.DIRTY
                    elif self._state == ActorState.EVICTING:
                        pass
                    elif self._pending_runs:
                        self._state = ActorState.BUSY
                    else:
                        self._state = ActorState.IDLE

                if context.dirty:
                    # The execution lock is still held, so no queued call can
                    # observe or use the uncertain model before disposal.
                    self._dispose(due_to_dirty=True)

        if cleanup_error is not None:
            raise ActorCleanupError(cleanup_error, experiment_error) from cleanup_error
        if experiment_error is not None:
            raise experiment_error
        return result

    def request_preemption(self, reason: str = "preemption requested") -> bool:
        with self._state_lock:
            context = self._active_context
        if context is None:
            return False
        return context.request_preemption(reason)

    def _dispose(self, *, due_to_dirty: bool = False) -> None:
        with self._state_lock:
            loaded, self._loaded = self._loaded, None
            self._state = ActorState.STOPPING
        disposal_error: BaseException | None = None
        if loaded is not None and loaded.close is not None:
            try:
                loaded.close()
            except BaseException as exc:
                disposal_error = exc
                with self._state_lock:
                    self._dirty_reasons.append(f"model disposal raised {type(exc).__name__}")
        if loaded is not None:
            # Drop the actor's strong references before asking a loaded Torch
            # runtime to return cached blocks to the device allocator.
            loaded.model = None
            loaded.tokenizer = None
            try:
                gc.collect()
                torch = sys.modules.get("torch")
                cuda = getattr(torch, "cuda", None)
                empty_cache = getattr(cuda, "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()
            except BaseException as exc:
                if disposal_error is None:
                    disposal_error = exc
                with self._state_lock:
                    self._dirty_reasons.append(f"device cache cleanup raised {type(exc).__name__}")
        with self._state_lock:
            self._state = ActorState.STOPPED
        if disposal_error is not None and not due_to_dirty:
            raise ActorDisposalError("model disposal failed") from disposal_error

    def evict(self, *, force: bool = False) -> None:
        """Dispose an idle actor, or cooperatively preempt and drain when forced."""

        with self._state_lock:
            if self._state == ActorState.STOPPED:
                return
            if self._state in (ActorState.DIRTY, ActorState.STOPPING):
                # A dirty execution owns disposal; wait below for its lock.
                pass
            elif self._pending_runs and not force:
                raise ActorUnavailableError("cannot evict an actor with pending work")
            else:
                self._state = ActorState.EVICTING
                if self._active_context is not None:
                    self._active_context.request_preemption("actor eviction requested")
        with self._execution_lock:
            with self._state_lock:
                if self._state == ActorState.STOPPED:
                    return
            self._dispose()


class ActorManager:
    """Warm actor allocations keyed separately from short-lived job leases."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._actors: dict[tuple[str, str], LocalModelActor] = {}
        self._slot_locks: dict[tuple[str, str], RLock] = {}

    def _slot_lock(self, slot: tuple[str, str]) -> RLock:
        with self._lock:
            return self._slot_locks.setdefault(slot, RLock())

    def _get_or_create_locked(self, key: ActorKey, loader: ModelLoader) -> LocalModelActor:
        with self._lock:
            actor = self._actors.get(key.slot)
        if actor is not None:
            if actor.key == key and actor.is_reusable:
                return actor
            if actor.has_pending_work:
                raise IncompatibleActorBusyError(
                    f"GPU {key.gpu_uuid!r} is busy with model {actor.key.model.alias!r}"
                )
            actor.evict()
        actor = LocalModelActor(key, loader)
        with self._lock:
            self._actors[key.slot] = actor
        return actor

    def get_or_create(self, key: ActorKey, loader: ModelLoader) -> LocalModelActor:
        slot_lock = self._slot_lock(key.slot)
        with slot_lock:
            return self._get_or_create_locked(key, loader)

    def run(
        self,
        key: ActorKey,
        loader: ModelLoader,
        experiment: Any,
        *,
        job_id: str,
        attempt_id: str,
        **runtime_options: Any,
    ) -> Any:
        experiment_kwargs = runtime_options.pop("kwargs", None)
        workspace = runtime_options.pop("workspace", None)
        committed_inputs = runtime_options.pop("committed_inputs", None)
        scratch = runtime_options.pop("scratch", None)
        preemption = runtime_options.pop("preemption", None)
        preemption_probe = runtime_options.pop("preemption_probe", None)
        checkpoint_handler = runtime_options.pop("checkpoint_handler", None)
        checkpoint_interval_seconds = runtime_options.pop("checkpoint_interval_seconds", None)
        checkpoint_restore = runtime_options.pop("checkpoint_restore", None)
        artifact_handler = runtime_options.pop("artifact_handler", None)
        if runtime_options:
            raise TypeError(f"unknown runtime options: {', '.join(sorted(runtime_options))}")
        slot_lock = self._slot_lock(key.slot)
        with slot_lock:
            actor = self._get_or_create_locked(key, loader)
            actor._reserve()
        try:
            return actor._run_reserved(
                experiment,
                job_id=job_id,
                attempt_id=attempt_id,
                kwargs=experiment_kwargs,
                workspace=workspace,
                committed_inputs=committed_inputs,
                scratch=scratch,
                preemption=preemption,
                preemption_probe=preemption_probe,
                checkpoint_handler=checkpoint_handler,
                checkpoint_interval_seconds=checkpoint_interval_seconds,
                checkpoint_restore=checkpoint_restore,
                artifact_handler=artifact_handler,
            )
        finally:
            if not actor.is_reusable:
                with slot_lock:
                    with self._lock:
                        if self._actors.get(key.slot) is actor:
                            del self._actors[key.slot]

    def evict(self, site_id: str, gpu_uuid: str, *, force: bool = False) -> bool:
        slot = (site_id, gpu_uuid)
        slot_lock = self._slot_lock(slot)
        with slot_lock:
            with self._lock:
                actor = self._actors.get(slot)
            if actor is None:
                return False
            actor.evict(force=force)
            with self._lock:
                if self._actors.get(slot) is actor:
                    del self._actors[slot]
            return True

    def request_preemption(
        self, site_id: str, gpu_uuid: str, reason: str = "preemption requested"
    ) -> bool:
        with self._lock:
            actor = self._actors.get((site_id, gpu_uuid))
        return bool(actor and actor.request_preemption(reason))

    def get(self, site_id: str, gpu_uuid: str) -> LocalModelActor | None:
        with self._lock:
            return self._actors.get((site_id, gpu_uuid))

    def reap(self, actor: LocalModelActor) -> bool:
        """Forget an actor only after it has made itself non-reusable."""

        if actor.is_reusable:
            return False
        slot_lock = self._slot_lock(actor.key.slot)
        with slot_lock, self._lock:
            if self._actors.get(actor.key.slot) is actor:
                del self._actors[actor.key.slot]
                return True
            return False

    def snapshot(self) -> tuple[ActorDescription, ...]:
        with self._lock:
            actors = tuple(self._actors.values())
        return tuple(
            sorted(
                (actor.describe() for actor in actors),
                key=lambda item: (item.key.site_id, item.key.gpu_uuid),
            )
        )

    def warm_model_keys(self) -> frozenset[ModelKey]:
        return frozenset(
            description.key.model
            for description in self.snapshot()
            if description.state in (ActorState.IDLE, ActorState.BUSY) and not description.dirty
        )

    def close(self, *, force: bool = False) -> None:
        with self._lock:
            slots = tuple(self._actors)
        failures: list[Exception] = []
        for site_id, gpu_uuid in slots:
            try:
                self.evict(site_id, gpu_uuid, force=force)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ActorDisposalError(
                f"failed to evict {len(failures)} warm actor(s)"
            ) from failures[0]
