from __future__ import annotations

import json
import time
from threading import Event
from typing import Any

import pytest

from topodrome.runtime import (
    ActorKey,
    DetachedRayActorAdapter,
    LoadedModel,
    ModelKey,
    RayActorSpec,
    RayActorTimeoutError,
    RayDisposalError,
    RayOperationTimeouts,
    RayWorkerGroupSpec,
    RuntimeExecutionService,
)
from topodrome.runtime.environment import (
    RAY_ALLOWED_JOB_KEYS_ENVIRONMENT,
    RAY_WORKER_ENVIRONMENT_SETUP_HOOK,
)

_EINOPS_PIN = f"einops==0.8.0 --hash=sha256:{'a' * 64}"
_ACCELERATE_PIN = f"accelerate==1.2.0 --hash=sha256:{'b' * 64}"
_RAY_WORKING_DIR = f"gcs://_ray_pkg_{'c' * 16}.zip"


class _ActorDiedError(RuntimeError):
    pass


class _GetTimeoutError(RuntimeError):
    pass


class _RemoteMethod:
    def __init__(self, handle: _Handle, method: Any) -> None:
        self._handle = handle
        self._method = method

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        if not self._handle.alive:
            raise _ActorDiedError("actor is dead")
        return self._method(*args, **kwargs)


class _Handle:
    def __init__(self, instance: Any) -> None:
        self.instance = instance
        self.alive = True

    def __getattr__(self, name: str) -> _RemoteMethod:
        return _RemoteMethod(self, getattr(self.instance, name))


class _RemoteClass:
    def __init__(self, ray: _FakeRay, actor_type: type[Any]) -> None:
        self.ray = ray
        self.actor_type = actor_type
        self.actor_options: dict[str, Any] = {}

    def options(self, **options: Any) -> _RemoteClass:
        self.actor_options = options
        self.ray.last_options = options
        self.ray.options_history.append(options)
        return self

    def remote(self, *args: Any) -> _Handle:
        handle = _Handle(self.actor_type(*args))
        if "name" in self.actor_options:
            key = (self.actor_options["namespace"], self.actor_options["name"])
            if key in self.ray.actors:
                raise ValueError("actor already exists")
            self.ray.actors[key] = handle
        else:
            self.ray.private_actors.append(handle)
        return handle


class _PlacementGroup:
    def __init__(self, bundles: list[dict[str, float | int]], strategy: str) -> None:
        self.bundles = bundles
        self.strategy = strategy

    def ready(self) -> _PlacementGroup:
        return self


class _RayUtil:
    def __init__(self) -> None:
        self.groups: list[_PlacementGroup] = []
        self.removed: list[_PlacementGroup] = []

    def placement_group(
        self,
        bundles: list[dict[str, float | int]],
        *,
        strategy: str,
    ) -> _PlacementGroup:
        group = _PlacementGroup(bundles, strategy)
        self.groups.append(group)
        return group

    def remove_placement_group(self, group: _PlacementGroup) -> None:
        if group not in self.removed:
            self.removed.append(group)

    def placement_group_table(self, group: _PlacementGroup) -> dict[str, str]:
        return {} if group in self.removed else {"state": "CREATED"}


class _FakeRay:
    def __init__(self) -> None:
        self.actors: dict[tuple[str, str], _Handle] = {}
        self.private_actors: list[_Handle] = []
        self.last_options: dict[str, Any] = {}
        self.options_history: list[dict[str, Any]] = []
        self.util = _RayUtil()
        self.get_timeouts: list[float | None] = []
        self.exceptions = type(
            "RayExceptions",
            (),
            {
                "ActorDiedError": _ActorDiedError,
                "GetTimeoutError": _GetTimeoutError,
            },
        )

    def remote(self, actor_type: type[Any]) -> _RemoteClass:
        return _RemoteClass(self, actor_type)

    def get_actor(self, name: str, *, namespace: str) -> _Handle:
        try:
            return self.actors[(namespace, name)]
        except KeyError:
            raise ValueError("actor not found") from None

    def get(self, value: Any, *, timeout: float | None = None) -> Any:
        self.get_timeouts.append(timeout)
        return value

    def kill(self, handle: _Handle, *, no_restart: bool) -> None:
        assert no_restart
        handle.alive = False
        for key, candidate in tuple(self.actors.items()):
            if candidate is handle:
                del self.actors[key]
        if handle in self.private_actors:
            self.private_actors.remove(handle)


def actor_key() -> ActorKey:
    return ActorKey(
        site_id="lab1",
        gpu_uuid="GPU-abcd:1234",
        model=ModelKey(
            alias="llama",
            revision="exact",
            backend="test",
            backend_version="1",
        ),
    )


def test_ray_actor_is_detached_explicitly_pinned_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    ray = _FakeRay()
    adapter = DetachedRayActorAdapter(ray)
    loads = 0

    def loader(_key: ModelKey) -> LoadedModel:
        nonlocal loads
        loads += 1
        return LoadedModel(model="loaded")

    spec = RayActorSpec(actor_key(), cuda_visible_device="GPU-abcd:1234")
    first = adapter.get_or_create(spec, loader)
    second = adapter.get_or_create(spec, loader)

    assert first.run(lambda context: context.model, job_id="j", attempt_id="a") == "loaded"
    assert not first.reused
    assert second.reused
    assert loads == 1
    assert ray.last_options["lifetime"] == "detached"
    assert ray.last_options["num_gpus"] == 0
    assert ray.last_options["resources"] == {"topodrome_gpu_gpu_abcd_1234": 1}
    assert ray.last_options["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == ("GPU-abcd:1234")
    runtime_env = ray.last_options["runtime_env"]
    assert runtime_env["worker_process_setup_hook"] == RAY_WORKER_ENVIRONMENT_SETUP_HOOK
    assert runtime_env["env_vars"][RAY_ALLOWED_JOB_KEYS_ENVIRONMENT] == "[]"


def test_ray_actor_shell_is_killed_after_dirty_cleanup() -> None:
    ray = _FakeRay()
    adapter = DetachedRayActorAdapter(ray)
    spec = RayActorSpec(actor_key(), cuda_visible_device="0")
    handle = adapter.get_or_create(spec, lambda _key: LoadedModel(model=object()))

    def experiment(context) -> None:
        context.mark_dirty("unknown weight state")

    handle.run(experiment, job_id="j", attempt_id="a")

    assert not ray.actors


def test_ray_adapter_refuses_to_serialize_credential_bearing_loader() -> None:
    ray = _FakeRay()
    adapter = DetachedRayActorAdapter(ray)
    spec = RayActorSpec(actor_key(), cuda_visible_device="0")

    class CredentialLoader:
        contains_credentials = True

        def __call__(self, _key: ModelKey) -> LoadedModel:
            return LoadedModel(model=object())

    with pytest.raises(ValueError, match="credential-bearing"):
        adapter.get_or_create(spec, CredentialLoader())

    assert not ray.actors


def test_fresh_runtime_service_reconnects_named_detached_actor_without_reloading() -> None:
    ray = _FakeRay()
    key = actor_key()
    spec = RayActorSpec(key, cuda_visible_device="0")
    first_adapter = DetachedRayActorAdapter(ray)
    loads = 0

    def loader(_key: ModelKey) -> LoadedModel:
        nonlocal loads
        loads += 1
        return LoadedModel(model="warm")

    original = first_adapter.get_or_create(spec, loader)
    actor_id = str(original.describe()["actor_id"])

    replacement = RuntimeExecutionService(ray=DetachedRayActorAdapter(ray))
    recovered = replacement.reconnect(
        key,
        actor_id=actor_id,
        cuda_visible_device="0",
    )

    assert recovered is not None
    assert recovered.actor_id == actor_id
    assert recovered.reused
    assert loads == 1
    assert (
        replacement.execute(
            recovered,
            lambda context: context.model,
            job_id="job-after-restart",
            attempt_id="attempt-after-restart",
        )
        == "warm"
    )


def test_force_stop_targets_prepared_handle_not_same_name_replacement() -> None:
    ray = _FakeRay()
    adapter = DetachedRayActorAdapter(ray)
    service = RuntimeExecutionService(ray=adapter)
    key = actor_key()
    spec = RayActorSpec(key, cuda_visible_device="0")
    prepared = service.prepare(
        key,
        lambda _key: LoadedModel(model="owned"),
        use_ray=True,
        cuda_visible_device="0",
    )
    replacement = _Handle(object())
    slot = (spec.namespace, spec.actor_name)
    ray.actors[slot] = replacement

    assert service.force_stop(prepared)
    assert ray.actors[slot] is replacement


def test_runtime_environment_resources_and_fingerprint_control_actor_reuse() -> None:
    ray = _FakeRay()
    adapter = DetachedRayActorAdapter(ray)
    loads = 0

    def loader(_key: ModelKey) -> LoadedModel:
        nonlocal loads
        loads += 1
        return LoadedModel(model="loaded")

    base = actor_key()
    first_key = ActorKey(
        base.site_id,
        base.gpu_uuid,
        base.model,
        runtime_fingerprint="runtime-a",
    )
    second_key = ActorKey(
        base.site_id,
        base.gpu_uuid,
        base.model,
        runtime_fingerprint="runtime-b",
    )
    first = adapter.get_or_create(
        RayActorSpec(
            first_key,
            cuda_visible_device="0",
            num_cpus=3.5,
            memory_bytes=6 * 1024**3,
            pip=(_EINOPS_PIN,),
            env={"EXPERIMENT_MODE": "strict"},
            working_dir=_RAY_WORKING_DIR,
        ),
        loader,
    )
    options = ray.last_options
    # Ray serializes per-actor RuntimeEnv mappings before worker creation.
    # Callable objects here pass fake-Ray tests but fail against real Ray 2.58.
    json.dumps(options["runtime_env"], sort_keys=True)
    assert options["num_cpus"] == 3.5
    assert options["memory"] == 6 * 1024**3
    assert options["runtime_env"]["pip"] == [_EINOPS_PIN]
    assert options["runtime_env"]["working_dir"] == _RAY_WORKING_DIR
    assert options["runtime_env"]["env_vars"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "EXPERIMENT_MODE": "strict",
        RAY_ALLOWED_JOB_KEYS_ENVIRONMENT: '["EXPERIMENT_MODE"]',
    }
    assert (
        options["runtime_env"]["worker_process_setup_hook"]
        == RAY_WORKER_ENVIRONMENT_SETUP_HOOK
    )

    second = adapter.get_or_create(
        RayActorSpec(second_key, cuda_visible_device="0"),
        loader,
    )
    assert not first.reused
    assert not second.reused
    assert loads == 2


def test_worker_group_reserves_all_uuid_bundles_and_executes_every_rank() -> None:
    ray = _FakeRay()
    adapter = DetachedRayActorAdapter(ray)
    base = actor_key()
    keys = tuple(
        ActorKey(
            base.site_id,
            f"GPU-{rank}",
            base.model,
            runtime_fingerprint="distributed-runtime",
        )
        for rank in range(2)
    )
    group = adapter.create_worker_group(
        RayWorkerGroupSpec(
            keys=keys,
            cuda_visible_devices=("2", "5"),
            num_cpus=6,
            memory_bytes=10 * 1024**3,
            pip=(_ACCELERATE_PIN,),
            env={"RUN_MODE": "distributed"},
            working_dir=_RAY_WORKING_DIR,
        ),
        lambda _key: LoadedModel(model="rank-model"),
    )

    placement = ray.util.groups[0]
    assert placement.strategy == "STRICT_PACK"
    assert placement.bundles == [
        {
            "CPU": 3,
            "memory": 5 * 1024**3,
            "topodrome_gpu_gpu_0": 1,
        },
        {
            "CPU": 3,
            "memory": 5 * 1024**3,
            "topodrome_gpu_gpu_1": 1,
        },
    ]
    worker_options = ray.options_history[-2:]
    assert [
        options["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"]
        for options in worker_options
    ] == ["2", "5"]
    assert [options["runtime_env"]["env_vars"]["RANK"] for options in worker_options] == [
        "0",
        "1",
    ]
    assert all(options["placement_group"] is placement for options in worker_options)

    calls: list[str] = []
    checkpoints: list[object] = []

    def experiment(context) -> str:
        calls.append(str(context.model))
        context.checkpoint({"calls": len(calls)})
        return "complete"

    assert (
        group.run(
            experiment,
            job_id="job-group",
            attempt_id="attempt-group",
            checkpoint_handler=lambda record: checkpoints.append(record),
        )
        == "complete"
    )
    assert calls == ["rank-model", "rank-model"]
    assert len(checkpoints) == 1
    assert group.evict(force=True)
    assert not ray.private_actors
    assert ray.util.removed == [placement]


def test_worker_group_fails_attempt_and_disposes_all_ranks_when_any_rank_fails() -> None:
    ray = _FakeRay()
    adapter = DetachedRayActorAdapter(ray)
    base = actor_key()
    keys = tuple(
        ActorKey(
            base.site_id,
            f"GPU-{rank}",
            base.model,
            runtime_fingerprint="failing-distributed-runtime",
        )
        for rank in range(2)
    )
    group = adapter.create_worker_group(
        RayWorkerGroupSpec(
            keys=keys,
            cuda_visible_devices=("0", "1"),
            num_cpus=2,
            memory_bytes=2 * 1024**3,
        ),
        lambda _key: LoadedModel(model="rank-model"),
    )
    calls = 0

    def one_rank_fails(_context: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("rank failed")
        return "rank-zero-result"

    with pytest.raises(RuntimeError, match="rank failed"):
        group.run(one_rank_fails, job_id="job", attempt_id="attempt")

    assert calls == 2
    assert ray.private_actors == []
    assert ray.util.removed == ray.util.groups


def test_model_sharded_replica_claims_gpu_gang_runs_once_and_reuses_exact_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    ray = _FakeRay()
    service = RuntimeExecutionService(ray=DetachedRayActorAdapter(ray))
    base = actor_key()
    key = ActorKey(
        site_id=base.site_id,
        gpu_uuid="GPU-A",
        model=base.model,
        runtime_fingerprint="model-sharded-runtime",
        replica_gpu_uuids=("GPU-A", "GPU-B"),
    )

    class ShardedLoader:
        def __init__(self) -> None:
            self.loads = 0
            self.configurations: list[tuple[int, str, int]] = []

        def for_model_sharding(
            self,
            *,
            degree: int,
            strategy: str,
            max_memory_bytes_per_gpu: int,
        ) -> ShardedLoader:
            self.configurations.append(
                (degree, strategy, max_memory_bytes_per_gpu)
            )
            return self

        def __call__(self, _model_key: ModelKey) -> LoadedModel:
            self.loads += 1
            return LoadedModel(model="one-sharded-model")

    loader = ShardedLoader()
    first = service.prepare_sharded(
        key,
        loader,
        cuda_visible_devices=("2", "5"),
        strategy="balanced",
        max_memory_bytes_per_gpu=10 * 1024**3,
        num_cpus=4,
        memory_bytes=8 * 1024**3,
    )
    second = service.prepare_sharded(
        key,
        loader,
        cuda_visible_devices=("2", "5"),
        strategy="balanced",
        max_memory_bytes_per_gpu=10 * 1024**3,
        num_cpus=4,
        memory_bytes=8 * 1024**3,
    )

    calls: list[str] = []

    def experiment(context: Any) -> str:
        calls.append(str(context.model))
        return str(context.model)

    assert service.execute(first, experiment, job_id="job-1", attempt_id="attempt-1") == (
        "one-sharded-model"
    )
    assert service.execute(second, experiment, job_id="job-2", attempt_id="attempt-2") == (
        "one-sharded-model"
    )
    assert not first.reused
    assert second.reused
    assert first.actor_id == second.actor_id
    assert first.worker_pids == second.worker_pids
    assert loader.loads == 1
    assert calls == ["one-sharded-model", "one-sharded-model"]
    assert loader.configurations == [
        (2, "balanced", 10 * 1024**3),
        (2, "balanced", 10 * 1024**3),
    ]
    options = ray.options_history[0]
    assert options["resources"] == {
        "topodrome_gpu_gpu_a": 1,
        "topodrome_gpu_gpu_b": 1,
    }
    assert options["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "2,5"
    assert len(ray.actors) == 1


def test_model_sharded_reconnect_requires_exact_ordered_group_identity() -> None:
    ray = _FakeRay()
    first_service = RuntimeExecutionService(ray=DetachedRayActorAdapter(ray))
    base = actor_key()
    key = ActorKey(
        base.site_id,
        "GPU-A",
        base.model,
        runtime_fingerprint="sharded-runtime",
        replica_gpu_uuids=("GPU-A", "GPU-B"),
    )

    class Loader:
        def for_model_sharding(self, **_options: Any) -> Loader:
            return self

        def __call__(self, _key: ModelKey) -> LoadedModel:
            return LoadedModel(model="sharded")

    original = first_service.prepare_sharded(
        key,
        Loader(),
        cuda_visible_devices=("0", "1"),
        strategy="balanced",
        max_memory_bytes_per_gpu=1,
    )
    replacement = RuntimeExecutionService(ray=DetachedRayActorAdapter(ray))

    assert (
        replacement.reconnect(
            key,
            actor_id=original.actor_id,
            cuda_visible_device="0,1",
        )
        is not None
    )
    reversed_key = ActorKey(
        base.site_id,
        "GPU-B",
        base.model,
        runtime_fingerprint="sharded-runtime",
        replica_gpu_uuids=("GPU-B", "GPU-A"),
    )
    assert (
        replacement.reconnect(
            reversed_key,
            actor_id=original.actor_id,
            cuda_visible_device="1,0",
        )
        is None
    )


def test_every_ray_get_uses_an_explicit_finite_timeout() -> None:
    ray = _FakeRay()
    timeouts = RayOperationTimeouts(
        prepare_seconds=11,
        execute_seconds=12,
        control_seconds=13,
        disposal_seconds=14,
    )
    adapter = DetachedRayActorAdapter(ray, timeouts=timeouts)
    spec = RayActorSpec(actor_key(), cuda_visible_device="0")
    handle = adapter.get_or_create(
        spec,
        lambda _key: LoadedModel(model="loaded"),
    )

    assert handle.describe()["state"] == "idle"
    assert handle.run(lambda context: context.model, job_id="j", attempt_id="a") == (
        "loaded"
    )
    assert not handle.request_preemption()
    assert handle.evict(force=True)

    assert ray.get_timeouts
    assert None not in ray.get_timeouts
    assert set(ray.get_timeouts) <= {11, 12, 13, 14, 0.25}


def test_timed_out_model_construction_is_disposed_before_error_returns() -> None:
    class TimeoutRay(_FakeRay):
        fail_next_get = False

        def get(self, value: Any, *, timeout: float | None = None) -> Any:
            self.get_timeouts.append(timeout)
            if self.fail_next_get:
                self.fail_next_get = False
                raise _GetTimeoutError("injected timeout")
            return value

    ray = TimeoutRay()
    adapter = DetachedRayActorAdapter(
        ray,
        timeouts=RayOperationTimeouts(
            prepare_seconds=0.1,
            execute_seconds=1,
            control_seconds=0.1,
            disposal_seconds=0.1,
        ),
    )
    spec = RayActorSpec(actor_key(), cuda_visible_device="0")
    ray.fail_next_get = True

    with pytest.raises(RayActorTimeoutError, match="model construction"):
        adapter.get_or_create(spec, lambda _key: LoadedModel(model="loaded"))

    assert ray.actors == {}


def test_unconfirmed_actor_kill_raises_and_keeps_disposal_fenced() -> None:
    class UnkillableRay(_FakeRay):
        def kill(self, handle: _Handle, *, no_restart: bool) -> None:
            del handle
            assert no_restart

    ray = UnkillableRay()
    adapter = DetachedRayActorAdapter(
        ray,
        timeouts=RayOperationTimeouts(
            prepare_seconds=1,
            execute_seconds=1,
            control_seconds=0.05,
            disposal_seconds=0.05,
        ),
    )
    handle = adapter.get_or_create(
        RayActorSpec(actor_key(), cuda_visible_device="0"),
        lambda _key: LoadedModel(model="loaded"),
    )

    with pytest.raises(RayDisposalError, match="not confirmed"):
        handle.evict(force=True)

    assert ray.actors


def test_timed_out_named_actor_creation_is_reported_as_cleanup_unconfirmed() -> None:
    gate = Event()

    class SlowRemoteClass(_RemoteClass):
        def remote(self, *args: Any) -> _Handle:
            gate.wait()
            return super().remote(*args)

    class SlowCreationRay(_FakeRay):
        def remote(self, actor_type: type[Any]) -> _RemoteClass:
            return SlowRemoteClass(self, actor_type)

    ray = SlowCreationRay()
    adapter = DetachedRayActorAdapter(
        ray,
        timeouts=RayOperationTimeouts(
            prepare_seconds=0.01,
            execute_seconds=1,
            control_seconds=0.01,
            disposal_seconds=0.05,
        ),
    )
    try:
        with pytest.raises(RayDisposalError, match="creation outcome is unconfirmed") as error:
            adapter.get_or_create(
                RayActorSpec(actor_key(), cuda_visible_device="0"),
                lambda _key: LoadedModel(model="loaded"),
            )
        assert error.value.cleanup_unconfirmed
    finally:
        gate.set()

    deadline = time.monotonic() + 1
    while not ray.actors and time.monotonic() < deadline:
        time.sleep(0.001)
    # This late actor is why the site authority must retain its GPU fence.
    assert ray.actors


def test_timed_out_placement_group_creation_keeps_complete_gang_fenced() -> None:
    gate = Event()

    class SlowRayUtil(_RayUtil):
        def placement_group(
            self,
            bundles: list[dict[str, float | int]],
            *,
            strategy: str,
        ) -> _PlacementGroup:
            gate.wait()
            return super().placement_group(bundles, strategy=strategy)

    ray = _FakeRay()
    ray.util = SlowRayUtil()
    adapter = DetachedRayActorAdapter(
        ray,
        timeouts=RayOperationTimeouts(
            prepare_seconds=1,
            execute_seconds=1,
            control_seconds=0.01,
            disposal_seconds=0.05,
        ),
    )
    base = actor_key()
    keys = tuple(
        ActorKey(base.site_id, f"GPU-{rank}", base.model) for rank in range(2)
    )
    try:
        with pytest.raises(RayDisposalError, match="group creation outcome") as error:
            adapter.create_worker_group(
                RayWorkerGroupSpec(keys=keys, cuda_visible_devices=("0", "1")),
                lambda _key: LoadedModel(model="loaded"),
            )
        assert error.value.cleanup_unconfirmed
    finally:
        gate.set()

    deadline = time.monotonic() + 1
    while not ray.util.groups and time.monotonic() < deadline:
        time.sleep(0.001)
    assert ray.util.groups


def test_post_creation_identity_timeout_disposes_exact_actor() -> None:
    class SecondGetTimeoutRay(_FakeRay):
        def get(self, value: Any, *, timeout: float | None = None) -> Any:
            self.get_timeouts.append(timeout)
            if len(self.get_timeouts) == 2:
                raise _GetTimeoutError("injected identity timeout")
            return value

    ray = SecondGetTimeoutRay()
    service = RuntimeExecutionService(
        ray=DetachedRayActorAdapter(
            ray,
            timeouts=RayOperationTimeouts(
                prepare_seconds=0.1,
                execute_seconds=1,
                control_seconds=0.1,
                disposal_seconds=0.1,
            ),
        )
    )

    with pytest.raises(RayActorTimeoutError, match="liveness probe"):
        service.prepare(
            actor_key(),
            lambda _key: LoadedModel(model="loaded"),
            use_ray=True,
            cuda_visible_device="0",
        )

    assert ray.actors == {}
