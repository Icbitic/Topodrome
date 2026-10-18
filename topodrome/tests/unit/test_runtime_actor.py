from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from topodrome.runtime import (
    ActorCleanupError,
    ActorKey,
    ActorManager,
    ActorState,
    LoadedModel,
    ModelKey,
    RuntimeExecutionService,
)


def model_key(repo_id: str = "test/llama") -> ModelKey:
    return ModelKey(
        repo_id=repo_id,
        revision="exact-revision",
        backend="test",
        backend_version="1",
        dtype="float32",
    )


def actor_key(repo_id: str = "test/llama") -> ActorKey:
    return ActorKey(site_id="lab1", gpu_uuid="GPU-1234", model=model_key(repo_id))


def test_warm_actor_is_reused_and_concurrent_runs_are_serialized() -> None:
    manager = ActorManager()
    loads: list[ModelKey] = []
    active = 0
    maximum_active = 0
    guard = threading.Lock()

    def loader(key: ModelKey) -> LoadedModel:
        loads.append(key)
        return LoadedModel(model={"identity": key.repo_id}, tokenizer="tokenizer", device="cuda:0")

    def experiment(context, *, value: int) -> int:
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        assert context.model["identity"] == "test/llama"
        with guard:
            active -= 1
        return value * 2

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                manager.run,
                actor_key(),
                loader,
                experiment,
                job_id=f"job-{value}",
                attempt_id=f"attempt-{value}",
                kwargs={"value": value},
            )
            for value in range(4)
        ]

    assert [future.result() for future in futures] == [0, 2, 4, 6]
    assert maximum_active == 1
    assert loads == [model_key()]
    warm = manager.get("lab1", "GPU-1234")
    assert warm is not None
    assert warm.state is ActorState.IDLE
    assert warm.describe().executions == 4


def test_cleanup_failure_marks_actor_dirty_disposes_it_and_prevents_reuse() -> None:
    manager = ActorManager()
    loads = 0
    disposals = 0

    def loader(_key: ModelKey) -> LoadedModel:
        nonlocal loads, disposals
        loads += 1

        def dispose() -> None:
            nonlocal disposals
            disposals += 1

        return LoadedModel(model=object(), close=dispose)

    def bad_cleanup() -> None:
        raise RuntimeError("hook could not be removed")

    def dirty_experiment(context) -> None:
        context.add_cleanup(bad_cleanup)

    actor = manager.get_or_create(actor_key(), loader)
    with pytest.raises(ActorCleanupError):
        manager.run(
            actor_key(),
            loader,
            dirty_experiment,
            job_id="job-dirty",
            attempt_id="attempt-dirty",
        )

    assert actor.state is ActorState.STOPPED
    assert actor.describe().dirty
    assert disposals == 1
    assert manager.get("lab1", "GPU-1234") is None

    result = manager.run(
        actor_key(),
        loader,
        lambda context: context.device,
        job_id="job-clean",
        attempt_id="attempt-clean",
    )
    assert result == "cuda"
    assert loads == 2


def test_incompatible_idle_actor_is_evicted_before_replacement() -> None:
    manager = ActorManager()
    disposed: list[str] = []

    def loader(key: ModelKey) -> LoadedModel:
        return LoadedModel(
            model=key.repo_id,
            close=lambda: disposed.append(key.repo_id),
        )

    first = manager.get_or_create(actor_key("test/first"), loader)
    second = manager.get_or_create(actor_key("test/second"), loader)

    assert first.state is ActorState.STOPPED
    assert second.state is ActorState.IDLE
    assert disposed == ["test/first"]
    assert manager.get("lab1", "GPU-1234") is second


def test_explicit_dirty_mark_evicts_even_after_success() -> None:
    manager = ActorManager()

    def experiment(context) -> str:
        context.mark_dirty("temporary weight restoration was not verifiable")
        return "committed-result"

    result = manager.run(
        actor_key(),
        lambda _key: LoadedModel(model=object()),
        experiment,
        job_id="job-1",
        attempt_id="attempt-1",
    )

    assert result == "committed-result"
    assert manager.get("lab1", "GPU-1234") is None


def test_execution_service_reuses_prepared_actor_and_routes_preemption() -> None:
    service = RuntimeExecutionService()

    def loader(_key: ModelKey) -> LoadedModel:
        return LoadedModel(model="model")

    first = service.prepare(actor_key(), loader)
    second = service.prepare(actor_key(), loader)
    started = threading.Event()

    def waits_for_preemption(context) -> str:
        started.set()
        while not context.should_preempt():
            time.sleep(0.005)
        return context.preemption_reason

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            service.execute,
            second,
            waits_for_preemption,
            job_id="job-preempt",
            attempt_id="attempt-preempt",
        )
        assert started.wait(timeout=1)
        assert service.request_preemption("attempt-preempt", "foreign pressure")

    assert future.result() == "foreign pressure"
    assert second.reused
    assert first.actor_id == second.actor_id
    assert second.worker_pids == (os.getpid(),)
