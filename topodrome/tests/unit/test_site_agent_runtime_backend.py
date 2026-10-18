from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from topodrome.domain.specs import AssetMember, JobSpec
from topodrome.runtime import (
    ActorKey,
    AsyncRuntimeExecutionService,
    ExecutionTransport,
    ExperimentContext,
    LoadedModel,
    ModelKey,
    ModelRegistration,
    ModelRegistry,
    PreparedActor,
)
from topodrome.site_agent.admission import (
    ActorAllocation,
    AdmissionController,
    LeaseFenceStore,
    LeaseRequest,
)
from topodrome.site_agent.agent import Assignment, SiteAgent
from topodrome.site_agent.inventory import GPUSnapshot
from topodrome.site_agent.pressure import PressurePolicy, SustainedPressureMonitor
from topodrome.site_agent.runtime_backend import (
    RuntimeExecutionBackend,
    _execute_kernel_operation,
    source_resolver_for_storage,
)
from topodrome.site_agent.workspace import WorkspaceManager
from topodrome.source_layout import (
    SOURCE_BUNDLE_FORMAT_VERSION,
    SOURCE_BUNDLE_METADATA_DIRECTORY,
)
from topodrome.storage import (
    AssetResolver,
    LocalStorage,
    asset_manifest_digest,
    load_published_manifest,
)

pytestmark = pytest.mark.unit


def test_checkpoint_finalization_verifies_only_latest_snapshot(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for name in ("00000001-training.pkl", "00000002-training.pkl"):
        (checkpoints / name).write_bytes(name.encode())
    verified = []

    def verify(path):
        verified.append(path)
        return SimpleNamespace(digest=SimpleNamespace(sha256="a" * 64))

    backend = SimpleNamespace(storage=SimpleNamespace(flush_and_verify=verify))
    assignment = SimpleNamespace(
        job_id="job-1",
        attempt_id="attempt-1",
        raw_spec={"entrypoint": "run.py", "model": {"repo_id": "test/model"}},
    )
    result = RuntimeExecutionBackend._verify_checkpoints(
        backend, assignment, SimpleNamespace(checkpoint_dir=checkpoints)
    )
    expected = "jobs/job-1/attempts/attempt-1/checkpoints/00000002-training.pkl"
    assert verified == [expected]
    assert result == expected + "#sha256=" + "a" * 64
    # Verification does not delete recovery files or touch training logs.
    assert len(list(checkpoints.glob("*.pkl"))) == 2


@pytest.mark.parametrize("error", [ValueError("lens failed"), SystemExit("early exit")])
def test_experiment_failure_keeps_unbuffered_output_and_full_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    from topodrome.site_agent import runtime_backend as module

    path = tmp_path / "logs" / "attempt.log"

    def fail(*args: Any) -> None:
        print("before death", end="")
        assert path.read_text() == "before death"
        raise error

    monkeypatch.setattr(module, "invoke_experiment", fail)
    with pytest.raises(type(error) if isinstance(error, Exception) else RuntimeError):
        module._execute_and_collect(
            SimpleNamespace(),
            entrypoint="experiment.py",
            experiment_kwargs={},
            log_path=str(path),
        )
    contents = path.read_text()
    assert contents.startswith("before death")
    assert "Traceback (most recent call last)" in contents
    assert str(error) in contents
    assert path.stat().st_mode & 0o777 == 0o600


def test_driver_death_diagnostic_is_saved_without_storage_upload(tmp_path: Path) -> None:
    from topodrome.site_agent.runtime_backend import _append_runtime_failure

    path = tmp_path / "attempt.log"
    path.write_text("last workload output\n")
    _append_runtime_failure(path, RuntimeError("worker killed by host memory pressure"))
    assert path.read_text().startswith("last workload output\n")
    assert "worker killed by host memory pressure" in path.read_text()


def test_abrupt_worker_exit_preserves_output_without_newline(tmp_path: Path) -> None:
    path = tmp_path / "attempt.log"
    program = f"""
import os
from topodrome.site_agent import runtime_backend as module
def die(*args):
    print('last bytes before abrupt exit', end='')
    os._exit(17)
module.invoke_experiment = die
module._execute_and_collect(
    None, entrypoint='unused.py', experiment_kwargs={{}}, log_path={str(path)!r})
"""
    result = subprocess.run(  # noqa: S603 - fixed synthetic worker, no user-supplied code
        [sys.executable, "-c", program], capture_output=True, timeout=20
    )
    assert result.returncode == 17
    assert path.read_text() == "last bytes before abrupt exit"


_EINOPS_PIN = f"einops==0.8.0 --hash=sha256:{'a' * 64}"
_RAY_WORKING_DIR = f"gcs://_ray_pkg_{'b' * 16}.zip"


@pytest.mark.asyncio
async def test_runtime_backend_disconnects_ray_driver_once_without_stopping_cluster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    ray = SimpleNamespace(
        is_initialized=lambda: True,
        shutdown=lambda: calls.append("driver-disconnected"),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setenv("RAY_DISABLE_FAILURE_SIGNAL_HANDLER", "0")
    backend = RuntimeExecutionBackend(
        site_id="lab1",
        registry=ModelRegistry(),
        runtime=AsyncRuntimeExecutionService(),
        storage=LocalStorage(tmp_path / "global"),
        use_ray=True,
    )
    backend._ray_connected = True

    assert await backend.shutdown_driver()
    assert not await backend.shutdown_driver()

    assert calls == ["driver-disconnected"]
    assert not backend._ray_connected
    assert os.environ["RAY_DISABLE_FAILURE_SIGNAL_HANDLER"] == "1"


@pytest.mark.asyncio
async def test_runtime_backend_keeps_driver_connected_when_disconnect_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise RuntimeError("driver disconnect failed")

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(is_initialized=lambda: True, shutdown=fail),
    )
    backend = RuntimeExecutionBackend(
        site_id="lab1",
        registry=ModelRegistry(),
        runtime=AsyncRuntimeExecutionService(),
        storage=LocalStorage(tmp_path / "global"),
        use_ray=True,
    )
    backend._ray_connected = True

    with pytest.raises(RuntimeError, match="driver disconnect failed"):
        await backend.shutdown_driver()

    assert backend._ray_connected


@pytest.mark.asyncio
async def test_runtime_backend_forwards_runtime_resources_to_all_admitted_gpus(
    tmp_path: Path,
) -> None:
    model_key = ModelKey(
        repo_id="test/model",
        revision="revision",
        backend="transformers",
        backend_version="1",
        dtype="bfloat16",
    )
    registry = ModelRegistry([ModelRegistration(model_key, local_path=tmp_path / "model")])
    runtime_directory = tmp_path / "runtime-source"
    runtime_directory.mkdir()
    published_working_dirs: list[tuple[Path, Path]] = []

    def publish_working_dir(directory: Path, scratch: Path) -> str:
        published_working_dirs.append((directory, scratch))
        return _RAY_WORKING_DIR

    class Runtime:
        options: dict[str, Any] | None = None
        keys: tuple[Any, ...] = ()

        async def prepare_group(
            self,
            keys: tuple[Any, ...],
            _loader: Any,
            **options: Any,
        ) -> PreparedActor:
            self.keys = keys
            self.options = options
            return PreparedActor(
                actor_id="group-actor",
                worker_pids=(101, 202),
                key=keys[0],
                reused=False,
                transport=ExecutionTransport.RAY_GROUP,
                _handle=object(),  # type: ignore[arg-type]
                _owner="test",
                group_keys=keys,
            )

    runtime = Runtime()
    backend = RuntimeExecutionBackend(
        site_id="lab1",
        registry=registry,
        runtime=runtime,  # type: ignore[arg-type]
        storage=LocalStorage(tmp_path / "global"),
        source_resolver=lambda _reference: runtime_directory,
        loader_factory=lambda _: lambda __: LoadedModel(model=object(), device="cpu"),
        use_ray=True,
        working_dir_publisher=publish_working_dir,
    )
    backend._ray_connected = True
    raw_spec = {
        "entrypoint": "experiment.py",
        "model": {
            **model_key.as_dict(),
            "parallelism": {
                "mode": "distributed",
                "degree": 2,
                "engine": "ray_worker_group",
            },
        },
        "resources": {
            "gpu": 2,
            "min_vram_gb": 15,
            "cpu": 6,
            "memory_gb": 10,
        },
        "runtime": {
            "pip": [_EINOPS_PIN],
            "env": {"EXPERIMENT_MODE": "strict"},
            "working_dir_ref": "sha256:" + "a" * 64,
        },
    }
    assignment = Assignment(
        job_id="job-group",
        attempt_id="attempt-group",
        lease=LeaseRequest(
            lease_id="lease-group",
            attempt_id="attempt-group",
            gpu_uuids=("GPU-A", "GPU-B"),
            fence_epochs={"GPU-A": 1, "GPU-B": 1},
            min_free_vram_bytes=15 * 1024**3,
            expires_at=4_000_000_000,
            cpu_cores=6,
            memory_bytes=10 * 1024**3,
        ),
        model_key="group-runtime",
        entrypoint="experiment.py",
        raw_spec=raw_spec,
        gpu_indices={"GPU-A": 2, "GPU-B": 5},
    )
    workspace = WorkspaceManager(
        storage=backend.storage,
        scratch_root=tmp_path / "scratch",
    ).prepare(assignment.job_id, assignment.attempt_id)

    prepared = await backend.prepare(assignment, workspace)

    expected_fingerprint = JobSpec.model_validate(raw_spec).runtime_fingerprint
    assert tuple(key.gpu_uuid for key in runtime.keys) == ("GPU-A", "GPU-B")
    assert {key.runtime_fingerprint for key in runtime.keys} == {expected_fingerprint}
    assert runtime.options == {
        "cuda_visible_devices": ("2", "5"),
        "ray_namespace": "topodrome",
        "num_cpus": 6.0,
        "memory_bytes": 10 * 1024**3,
        "pip": (_EINOPS_PIN,),
        "env": {"EXPERIMENT_MODE": "strict"},
        "working_dir": _RAY_WORKING_DIR,
    }
    assert published_working_dirs == [
        (
            runtime_directory.resolve(),
            workspace.scratch_dir / "ray-runtime-env",
        )
    ]
    assert prepared.process_ids == (101, 202)
    assert not prepared.reused


@pytest.mark.asyncio
async def test_runtime_backend_prepares_one_warm_model_sharded_replica(
    tmp_path: Path,
) -> None:
    model_key = ModelKey(
        repo_id="test/model",
        revision="revision",
        backend="transformers",
        backend_version="1",
        dtype="bfloat16",
    )
    registry = ModelRegistry([ModelRegistration(model_key, local_path=tmp_path / "model")])

    class Runtime:
        key: Any = None
        options: dict[str, Any] | None = None

        def __init__(self) -> None:
            self.core = SimpleNamespace(is_live=lambda _prepared, **_options: True)

        async def prepare_sharded(
            self,
            key: Any,
            _loader: Any,
            **options: Any,
        ) -> PreparedActor:
            self.key = key
            self.options = options
            return PreparedActor(
                actor_id="sharded-actor",
                worker_pids=(4242,),
                key=key,
                reused=False,
                transport=ExecutionTransport.RAY_SHARDED,
                _handle=object(),  # type: ignore[arg-type]
                _owner="test",
                group_keys=(key,),
            )

    runtime = Runtime()
    backend = RuntimeExecutionBackend(
        site_id="lab1",
        registry=registry,
        runtime=runtime,  # type: ignore[arg-type]
        storage=LocalStorage(tmp_path / "global"),
        loader_factory=lambda _: lambda __: LoadedModel(model=object()),
        use_ray=True,
    )
    backend._ray_connected = True
    raw_spec = {
        "entrypoint": "experiment.py",
        "model": {
            **model_key.as_dict(),
            "parallelism": {
                "mode": "model_sharded",
                "degree": 2,
                "engine": "transformers_device_map",
                "strategy": "balanced",
                "max_memory_gb_per_gpu": 10,
            },
        },
        "resources": {
            "gpu": 2,
            "min_vram_gb": 10,
            "cpu": 4,
            "memory_gb": 8,
        },
    }
    job_spec = JobSpec.model_validate(raw_spec)
    identity = json.dumps(
        {
            "model": job_spec.model.model_dump(mode="json"),
            "runtime_fingerprint": job_spec.runtime_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assignment = Assignment(
        job_id="job-sharded",
        attempt_id="attempt-sharded",
        lease=LeaseRequest(
            lease_id="lease-sharded",
            attempt_id="attempt-sharded",
            gpu_uuids=("GPU-A", "GPU-B"),
            fence_epochs={"GPU-A": 1, "GPU-B": 1},
            min_free_vram_bytes=10 * 1024**3,
            expires_at=4_000_000_000,
            model_key=identity,
        ),
        model_key=identity,
        entrypoint="experiment.py",
        raw_spec=raw_spec,
        gpu_indices={"GPU-A": 3, "GPU-B": 7},
    )
    workspace = WorkspaceManager(
        storage=backend.storage,
        scratch_root=tmp_path / "scratch",
    ).prepare(assignment.job_id, assignment.attempt_id)

    prepared = await backend.prepare(assignment, workspace)

    assert runtime.key.replica_gpu_uuids == ("GPU-A", "GPU-B")
    assert runtime.key.runtime_fingerprint == job_spec.runtime_fingerprint
    assert runtime.options == {
        "cuda_visible_devices": ("3", "7"),
        "strategy": "balanced",
        "max_memory_bytes_per_gpu": 10 * 1024**3,
        "ray_namespace": "topodrome",
        "num_cpus": 4.0,
        "memory_bytes": 8 * 1024**3,
        "pip": (),
        "env": {},
        "working_dir": None,
    }
    assert prepared.actor_id == "sharded-actor"
    assert prepared.process_ids == (4242,)
    warm = [item for item in backend.model_capabilities() if item["availability"] == "warm"]
    assert [item["gpu_uuid"] for item in warm] == ["GPU-A", "GPU-B"]
    assert all(item["replica_gpu_uuids"] == ["GPU-A", "GPU-B"] for item in warm)
    assert all(
        item["replica_key_hash"] == hashlib.sha256(identity.encode("utf-8")).hexdigest()
        for item in warm
    )


@pytest.mark.asyncio
async def test_stale_warm_runtime_is_not_advertised_or_removed_before_disposal(
    tmp_path: Path,
) -> None:
    model_key = ModelKey(
        repo_id="test/model",
        revision="revision",
        backend="transformers",
        backend_version="1",
    )
    key = ActorKey(
        "lab1",
        "GPU-A",
        model_key,
        replica_gpu_uuids=("GPU-A", "GPU-B"),
    )
    prepared = PreparedActor(
        actor_id="stale-sharded-actor",
        worker_pids=(4242,),
        key=key,
        reused=True,
        transport=ExecutionTransport.RAY_SHARDED,
        _handle=object(),  # type: ignore[arg-type]
        _owner="test",
        group_keys=(key,),
    )

    class Runtime:
        def __init__(self) -> None:
            self.allow_disposal = False
            self.core = SimpleNamespace(is_live=lambda _prepared, **_options: False)

        async def is_live(self, _prepared: PreparedActor) -> bool:
            return False

        async def evict(self, _prepared: PreparedActor, *, force: bool) -> bool:
            assert force
            if not self.allow_disposal:
                raise RuntimeError("disposal unconfirmed")
            return True

    runtime = Runtime()
    backend = RuntimeExecutionBackend(
        site_id="lab1",
        registry=ModelRegistry(),
        runtime=runtime,  # type: ignore[arg-type]
        storage=LocalStorage(tmp_path / "global"),
        use_ray=True,
    )
    backend._prepared[prepared.actor_id] = (prepared, "model-identity")
    allocation = ActorAllocation(
        gpu_uuid="GPU-A",
        actor_id=prepared.actor_id,
        model_key="model-identity",
        state="idle",
    )

    assert not any(item["availability"] == "warm" for item in backend.model_capabilities())
    assert not backend.is_warm_allocation(allocation)
    assert await backend.reconcile_warm_liveness() == ()
    assert prepared.actor_id in backend._prepared

    runtime.allow_disposal = True
    assert await backend.reconcile_warm_liveness() == (prepared.actor_id,)
    assert prepared.actor_id not in backend._prepared


@pytest.mark.asyncio
async def test_site_agent_keeps_and_reuses_complete_sharded_replica_between_leases(
    tmp_path: Path,
) -> None:
    storage = LocalStorage(tmp_path / "global")
    source = b"""
def run(context):
    output = context.workspace / 'result.json'
    output.write_text(context.attempt_id, encoding='utf-8')
    context.register_artifact(output, logical_path='result.json')
"""
    source_digest = hashlib.sha256(source).hexdigest()
    bundle_digest = "c" * 64
    source_root = storage.resolve(f"projects/.topodrome-sources/{bundle_digest}")
    source_root.mkdir(parents=True)
    (source_root / "experiment.py").write_bytes(source)
    (source_root / "manifest.json").write_text(
        json.dumps({"files": {"experiment.py": source_digest}}),
        encoding="utf-8",
    )
    asset_content = b"pinned input"
    asset_logical_path = "assets/private/pinned/input.bin"
    asset_source = storage.resolve(asset_logical_path)
    asset_source.parent.mkdir(parents=True)
    asset_source.write_bytes(asset_content)
    asset_member = AssetMember(
        logical_path=asset_logical_path,
        relative_path="input.bin",
        size_bytes=len(asset_content),
        sha256=hashlib.sha256(asset_content).hexdigest(),
    )
    asset_digest = asset_manifest_digest((asset_member,))
    order: list[str] = []

    class TrackingAssetResolver(AssetResolver):
        def resolve(self, reference: Any) -> Any:
            order.append("asset")
            return super().resolve(reference)

    asset_resolver = TrackingAssetResolver(storage, cache_root=tmp_path / "asset-cache")
    model_key = ModelKey(
        repo_id="test/model",
        revision="revision",
        backend="transformers",
        backend_version="1",
        dtype="bfloat16",
    )
    registry = ModelRegistry([ModelRegistration(model_key, local_path=tmp_path / "model")])

    class Runtime:
        def __init__(self) -> None:
            self.prepares = 0
            self.executions = 0
            self.evictions = 0
            self.prepared: PreparedActor | None = None
            self.core = SimpleNamespace(is_live=lambda _prepared, **_options: True)

        async def is_live(
            self,
            _prepared: PreparedActor,
            *,
            require_idle: bool = False,
        ) -> bool:
            del require_idle
            return self.prepared is not None

        async def prepare_sharded(
            self,
            key: Any,
            _loader: Any,
            **_options: Any,
        ) -> PreparedActor:
            order.append("prepare")
            self.prepares += 1
            reused = self.prepared is not None
            if self.prepared is None:
                self.prepared = PreparedActor(
                    actor_id="sharded-actor",
                    worker_pids=(4242,),
                    key=key,
                    reused=False,
                    transport=ExecutionTransport.RAY_SHARDED,
                    _handle=object(),  # type: ignore[arg-type]
                    _owner="test",
                    group_keys=(key,),
                )
            return PreparedActor(
                actor_id=self.prepared.actor_id,
                worker_pids=self.prepared.worker_pids,
                key=self.prepared.key,
                reused=reused,
                transport=self.prepared.transport,
                _handle=object(),  # type: ignore[arg-type]
                _owner="test",
                group_keys=self.prepared.group_keys,
            )

        async def execute(
            self,
            _prepared: PreparedActor,
            experiment: Any,
            **options: Any,
        ) -> Any:
            self.executions += 1
            context = ExperimentContext(
                model=SimpleNamespace(hf_device_map={"first": 0, "last": 1}),
                tokenizer=None,
                device="cuda:0",
                job_id=str(options["job_id"]),
                attempt_id=str(options["attempt_id"]),
                workspace=options["workspace"],
                committed_inputs=options["committed_inputs"],
                scratch=options["scratch"],
                checkpoint_handler=options["checkpoint_handler"],
                checkpoint_interval_seconds=options["checkpoint_interval_seconds"],
                checkpoint_restore=options["checkpoint_restore"],
            )
            try:
                return experiment(context, **dict(options["kwargs"]))
            finally:
                await context.aclose()

        async def evict(self, _prepared: PreparedActor, *, force: bool) -> bool:
            assert force
            self.evictions += 1
            self.prepared = None
            return True

    runtime = Runtime()
    backend = RuntimeExecutionBackend(
        site_id="lab1",
        registry=registry,
        runtime=runtime,  # type: ignore[arg-type]
        storage=storage,
        source_resolver=source_resolver_for_storage(storage),
        asset_resolver=asset_resolver,
        loader_factory=lambda _registration: object(),  # type: ignore[arg-type,return-value]
        use_ray=True,
    )
    backend._ray_connected = True

    class Inventory:
        def sample(self) -> tuple[GPUSnapshot, ...]:
            return tuple(
                GPUSnapshot(
                    uuid=uuid,
                    index=index,
                    name="GPU",
                    total_memory_bytes=24 * 1024**3,
                    free_memory_bytes=24 * 1024**3,
                    used_memory_bytes=0,
                    utilization_percent=0,
                )
                for uuid, index in (("GPU-A", 3), ("GPU-B", 7))
            )

    class Outbound:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        async def publish(self, kind: str, payload: Any) -> None:
            self.events.append((kind, dict(payload)))

        async def publish_confirmed(self, kind: str, payload: Any) -> None:
            self.events.append((kind, dict(payload)))

    inventory = Inventory()
    fences = LeaseFenceStore(tmp_path / "leases.sqlite3")
    admission = AdmissionController(
        inventory,  # type: ignore[arg-type]
        fences,
        safety_margin_bytes=0,
        warm_actor_validator=backend.is_warm_allocation,
    )
    outbound = Outbound()
    agent = SiteAgent(
        site_id="lab1",
        inventory=inventory,  # type: ignore[arg-type]
        admission=admission,
        fences=fences,
        ray=SimpleNamespace(),  # type: ignore[arg-type]
        workspaces=WorkspaceManager(storage=storage, scratch_root=tmp_path / "scratch"),
        backend=backend,
        outbound=outbound,  # type: ignore[arg-type]
        pressure=SustainedPressureMonitor(PressurePolicy(enabled=False)),
    )
    raw_spec = {
        "entrypoint": "experiment.py",
        "source_ref": f"sha256:{bundle_digest}",
        "model": {
            **model_key.as_dict(),
            "parallelism": {
                "mode": "model_sharded",
                "degree": 2,
                "engine": "transformers_device_map",
                "strategy": "balanced",
                "max_memory_gb_per_gpu": 10,
            },
        },
        "resources": {"gpu": 2, "min_vram_gb": 10},
        "assets": {
            "input": {
                "source_ref": "asset://private/pinned@v1",
                "kind": "file",
                "digest": asset_digest,
                "total_bytes": len(asset_content),
                "members": [asset_member.model_dump(mode="json")],
            }
        },
        "required_outputs": ["result.json"],
    }
    spec = JobSpec.model_validate(raw_spec)
    identity = json.dumps(
        {
            "model": spec.model.model_dump(mode="json"),
            "runtime_fingerprint": spec.runtime_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    def assignment(attempt: int) -> Assignment:
        attempt_id = f"attempt-{attempt}"
        return Assignment(
            job_id="job-sharded",
            attempt_id=attempt_id,
            lease=LeaseRequest(
                lease_id=f"lease-{attempt}",
                attempt_id=attempt_id,
                gpu_uuids=("GPU-A", "GPU-B"),
                fence_epochs={"GPU-A": attempt, "GPU-B": attempt},
                min_free_vram_bytes=10 * 1024**3,
                expires_at=4_000_000_000,
                model_key=identity,
            ),
            model_key=identity,
            entrypoint="experiment.py",
            raw_spec=raw_spec,
        )

    async def stage():
        from datetime import timedelta

        from topodrome.domain.specs import utc_now
        await agent.input_stager.accept({
            "job_id": "job-sharded", "stage_id": "stage-sharded",
            "assets": raw_spec["assets"], "source_ref": raw_spec["source_ref"],
            "expires_at": (utc_now() + timedelta(seconds=120)).isoformat(),
        })
        await agent.input_stager.stages["job-sharded"].task

    await stage()
    assert await agent.offer(assignment(1))
    await agent.wait_for_idle()
    assert order.index("asset") < order.index("prepare")
    first_allocations = fences.actor_allocations()
    assert len(first_allocations) == 2
    assert {item.actor_id for item in first_allocations} == {"sharded-actor"}
    assert {item.state for item in first_allocations} == {"idle"}
    assert runtime.evictions == 0

    await stage()
    assert await agent.offer(assignment(2))
    await agent.wait_for_idle()
    second_allocations = fences.actor_allocations()
    assert second_allocations == first_allocations
    assert runtime.prepares == 2
    assert runtime.executions == 2
    assert runtime.evictions == 0

    with pytest.raises(RuntimeError, match="identity is inconsistent"):
        await backend.evict_warm_replica(second_allocations[:1])
    worker_pids = await backend.evict_warm_replica(second_allocations)
    assert worker_pids == (4242,)
    assert fences.evict_actor("sharded-actor") == 2
    assert fences.actor_allocations() == ()
    assert runtime.evictions == 1
    fences.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("split_finalization", [False, True])
async def test_runtime_backend_reuses_actor_and_commits_verified_artifacts(
    tmp_path: Path,
    split_finalization: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = LocalStorage(tmp_path / "global")
    source = b"""
def run(context, value):
    print(f'experiment-output:{value}')
    output = context.workspace / 'result.json'
    input_value = context.committed_inputs['input'].read_text(encoding='utf-8')
    restored = context.restore_checkpoint()
    context.checkpoint({'value': value})
    output.write_text(f'{input_value}:{value}:{restored}', encoding='utf-8')
    context.register_artifact(output, logical_path='result.json')
"""
    source_digest = hashlib.sha256(source).hexdigest()
    bundle_digest = "a" * 64
    source_root = storage.resolve(f"projects/.topodrome-sources/{bundle_digest}")
    source_root.mkdir(parents=True)
    (source_root / "experiment.py").write_bytes(source)
    user_manifest = b'{"project":"runtime-regression"}\n'
    (source_root / "manifest.json").write_bytes(user_manifest)
    metadata = source_root / SOURCE_BUNDLE_METADATA_DIRECTORY
    metadata.mkdir()
    (metadata / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": SOURCE_BUNDLE_FORMAT_VERSION,
                "source_ref": f"sha256:{bundle_digest}",
                "files": {
                    "experiment.py": source_digest,
                    "manifest.json": hashlib.sha256(user_manifest).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    asset_path = "assets/tiny/v1/input.txt"
    input_asset = storage.resolve(asset_path)
    input_asset.parent.mkdir(parents=True)
    input_asset.write_text("input", encoding="utf-8")
    asset_member = AssetMember(
        logical_path=asset_path,
        relative_path="input.txt",
        size_bytes=5,
        sha256=hashlib.sha256(b"input").hexdigest(),
    )
    asset_digest = asset_manifest_digest((asset_member,))
    model_key = ModelKey(
        repo_id="test/model",
        revision="revision",
        backend="transformers",
        backend_version="1",
        dtype="bfloat16",
    )
    registry = ModelRegistry([ModelRegistration(model_key, local_path=tmp_path / "unused-model")])
    backend = RuntimeExecutionBackend(
        site_id="site-1",
        registry=registry,
        runtime=AsyncRuntimeExecutionService(),
        storage=storage,
        source_resolver=source_resolver_for_storage(storage),
        loader_factory=lambda _: lambda __: LoadedModel(model=object(), device="cpu"),
        use_ray=False,
    )
    workspaces = WorkspaceManager(storage=storage, scratch_root=tmp_path / "scratch")

    def assignment(
        attempt_id: str,
        epoch: int,
        *,
        checkpoint_ref: str | None = None,
    ) -> Assignment:
        raw_model = model_key.as_dict()
        raw_spec = {
            "entrypoint": "experiment.py",
            "source_ref": f"sha256:{bundle_digest}",
            "model": raw_model,
            "assets": {
                "input": {
                    "source_ref": "asset://private/tiny@v1/input.txt",
                    "kind": "file",
                    "digest": asset_digest,
                    "total_bytes": 5,
                    "members": [asset_member.model_dump(mode="json")],
                }
            },
            "kwargs": {"value": 7},
            "required_outputs": ["result.json"],
        }
        identity = json.dumps(raw_model, sort_keys=True, separators=(",", ":"))
        return Assignment(
            job_id="job-1",
            attempt_id=attempt_id,
            lease=LeaseRequest(
                lease_id=f"lease-{epoch}",
                attempt_id=attempt_id,
                gpu_uuids=("GPU-A",),
                fence_epochs={"GPU-A": epoch},
                min_free_vram_bytes=1,
                expires_at=4_000_000_000,
                model_key=identity,
            ),
            model_key=identity,
            entrypoint="experiment.py",
            kwargs={"value": 7},
            raw_spec=raw_spec,
            gpu_indices={"GPU-A": 0},
            checkpoint_ref=checkpoint_ref,
            fencing_token=secrets.token_urlsafe(),
        )

    first_assignment = assignment("attempt-1", 1)
    first_workspace = workspaces.prepare("job-1", "attempt-1")
    first = await backend.prepare(first_assignment, first_workspace)
    compute_finished = False
    original_commit = backend._commit_artifacts

    async def release_compute(reusable):
        nonlocal compute_finished
        assert reusable
        compute_finished = True
        return True

    def commit(*args):
        assert compute_finished
        return original_commit(*args)

    if split_finalization:
        monkeypatch.setattr(backend, "_commit_artifacts", commit)
        outcome = await backend.execute_with_finalization(
            first,
            first_assignment,
            first_workspace,
            release_compute,
        )
    else:
        outcome = await backend.execute(first, first_assignment, first_workspace)

    assert outcome.succeeded
    assert outcome.artifacts_verified
    assert outcome.metadata["artifacts"][0]["durable"] is True
    assert outcome.metadata["log"]["logical_path"].endswith("/logs/attempt.log")
    assert first_workspace.staging_dir.joinpath("manifest.json").read_bytes() == user_manifest
    log_bytes = first_workspace.log_dir.joinpath("attempt.log").read_bytes()
    assert log_bytes.startswith(b"experiment-output:7\n")
    assert b'"stage":"manifest"' in log_bytes
    assert b'"stage":"checkpoint"' in log_bytes
    assert b'"stage":"logs"' in log_bytes
    assert outcome.metadata["log"]["sha256"] == hashlib.sha256(log_bytes).hexdigest()
    assert outcome.metadata["log"]["end_offset"] == len(log_bytes)
    assert ".pkl#sha256=" in str(outcome.metadata["checkpoint_ref"])
    manifest = load_published_manifest(storage, "job-1")
    assert manifest.artifact("result.json").sha256 == hashlib.sha256(b"input:7:None").hexdigest()

    checkpoint_ref = str(outcome.metadata["checkpoint_ref"])
    second_assignment = assignment("attempt-2", 2, checkpoint_ref=checkpoint_ref)
    second_workspace = workspaces.prepare("job-1", "attempt-2")
    second = await backend.prepare(second_assignment, second_workspace)
    assert second.reused
    assert second.actor_id == first.actor_id
    restored = await backend.execute(second, second_assignment, second_workspace)
    assert restored.succeeded
    assert (
        second_workspace.log_dir.joinpath("attempt.log")
        .read_text()
        .startswith("experiment-output:7\n")
    )
    manifest = load_published_manifest(storage, "job-1")
    assert (
        manifest.artifact("result.json").sha256
        == hashlib.sha256(b"input:7:{'value': 7}").hexdigest()
    )
    await backend.evict(second)


def test_raw_jupyter_execution_preserves_namespace_and_bounds_normal_errors() -> None:
    context = SimpleNamespace(model=object(), tokenizer=None, device="cpu")
    first = _execute_kernel_operation(
        context,
        session_id="native-jupyter-test",
        mode="raw-model",
        operation="jupyter_execute",
        arguments={
            "code": "counter = 40\nprint('hello')\ncounter + 2",
            "_output_limit_bytes": 4096,
        },
    )
    assert first == {
        "status": "ok",
        "stdout": "hello\n",
        "stderr": "",
        "result": 42,
        "truncated": False,
    }
    second = _execute_kernel_operation(
        context,
        session_id="native-jupyter-test",
        mode="raw-model",
        operation="jupyter_execute",
        arguments={"code": "counter += 1\ncounter", "_output_limit_bytes": 4096},
    )
    assert second["result"] == 41

    failure = _execute_kernel_operation(
        context,
        session_id="native-jupyter-test",
        mode="raw-model",
        operation="jupyter_execute",
        arguments={"code": "raise ValueError('ordinary cell failure')"},
    )
    assert failure["status"] == "error"
    assert failure["ename"] == "ValueError"
    assert failure["traceback"] == []

    bounded = _execute_kernel_operation(
        context,
        session_id="native-jupyter-test",
        mode="raw-model",
        operation="jupyter_execute",
        arguments={"code": "print('x' * 10000)", "_output_limit_bytes": 4096},
    )
    assert bounded["truncated"] is True
    assert len(bounded["stdout"].encode()) < 1200
