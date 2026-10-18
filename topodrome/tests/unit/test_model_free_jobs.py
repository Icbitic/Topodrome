from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.unit.test_placement import site
from tests.unit.test_runtime_ray_actor import _FakeRay
from topodrome.domain.arrays import JobArraySpec
from topodrome.domain.enums import ActorState
from topodrome.domain.specs import JobSpec, utc_now
from topodrome.runtime import (
    ActorKey,
    CheckpointRestore,
    DetachedRayActorAdapter,
    ExecutionTransport,
    ExperimentContext,
    ModelRegistry,
    PreparedActor,
    RayWorkerGroupSpec,
    RuntimeExecutionService,
)
from topodrome.runtime.gpu_job import GPUJobLoader
from topodrome.scheduler.placement import choose_placement
from topodrome.site_agent.agent import Assignment, _is_private_workload
from topodrome.site_agent.runtime_backend import RuntimeExecutionBackend
from topodrome.site_agent.workspace import WorkspaceManager
from topodrome.storage import LocalStorage

pytestmark = pytest.mark.unit


def spec(gpus=1, execution=None):
    return JobSpec(
        entrypoint="run.py",
        model=None,
        execution=execution,
        resources={"gpu": gpus, "min_vram_gb": 2},
    )


@pytest.mark.parametrize("gpus,execution", [(1, None), (2, "single"), (2, "distributed")])
def test_null_model_round_trips_without_a_fake_model_alias(gpus, execution):
    wanted = spec(gpus, execution)
    assert wanted.model is None
    assert wanted.execution_mode == (execution or "single")
    assert JobSpec.model_validate_json(wanted.model_dump_json()) == wanted
    array = JobArraySpec(
        job=wanted,
        elements=[{"index": 0}, {"index": 1}],
        max_parallel=2,
    )
    assert array.element_spec(array.elements[0]).model is None
    assert array.element_spec(array.elements[1]).execution_mode == wanted.execution_mode


@pytest.mark.parametrize(
    "changes",
    [
        {"resources": {"gpu": 0, "min_vram_gb": 2}},
        {"resources": {"gpu": 1}},
        {"execution": "distributed"},
        {"execution": "model_sharded"},
        {"workload_kind": "interactive"},
        {"workload_kind": "adapter_training"},
        {"model": "none"},
    ],
)
def test_model_free_contract_rejects_ambiguous_or_unsafe_jobs(changes):
    payload = spec().model_dump()
    payload.update(changes)
    with pytest.raises(ValidationError):
        JobSpec.model_validate(payload)


def test_model_must_still_be_explicit_and_existing_model_identity_is_unchanged():
    with pytest.raises(ValidationError):
        JobSpec(entrypoint="run.py")
    wanted = JobSpec(entrypoint="run.py", model="test/llama")
    assert "execution" not in wanted.model_dump()
    assert "execution" not in json.loads(wanted.model_dump_json())
    old_payload = {
        "pip": [],
        "env": {},
        "working_dir_ref": None,
        "cpu": "1",
        "memory_bytes": 1024**3,
        "gpu_count": 1,
        "model_parallelism": {"mode": "single", "degree": 1, "engine": "none"},
    }
    old_fingerprint = hashlib.sha256(
        json.dumps(old_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert wanted.runtime_fingerprint == old_fingerprint
    with pytest.raises(ValidationError, match=r"model\.parallelism"):
        JobSpec(entrypoint="run.py", model="test/llama", execution="single")
    assert spec(2, "single").runtime_fingerprint != spec(2, "distributed").runtime_fingerprint


def model_free_site():
    candidate = site("test", free_gib=79)
    gpu = candidate.gpus[0]
    return replace(
        candidate,
        models=(),
        supported_workloads=frozenset({"batch", "model_free"}),
        gpus=(replace(gpu, uuid="GPU-A", index=3), replace(gpu, uuid="GPU-B", index=7)),
    )


def choose(wanted, candidate):
    return choose_placement(wanted, [candidate], heartbeat_cutoff=utc_now() - timedelta(seconds=30))


def test_model_free_placement_requires_no_model_but_requires_a_capable_agent():
    candidate = model_free_site()
    result = choose(spec(2), candidate)
    assert result.gpu_uuids == ("GPU-A", "GPU-B")
    assert not result.uses_warm_actor
    assert result.estimated_ready_seconds == 0
    assert choose(spec(), replace(candidate, supported_workloads=frozenset({"batch"}))) is None


@pytest.mark.parametrize(
    "gpu_changes",
    [
        {"active_lease": True},
        {"externally_reserved": True},
        {"healthy": False},
        {"free_memory_bytes": 1},
        {"foreign_process_count": 1},
    ],
)
def test_null_model_gangs_still_require_every_physical_gpu_to_be_safe(gpu_changes):
    candidate = model_free_site()
    candidate = replace(
        candidate, gpus=(candidate.gpus[0], replace(candidate.gpus[1], **gpu_changes))
    )
    assert choose(spec(2), candidate) is None
    assert choose(spec(), candidate) is not None


def test_model_free_jobs_replace_idle_models_but_never_reuse_them():
    candidate = model_free_site()
    gpu = replace(
        candidate.gpus[0],
        actor_state=ActorState.IDLE,
        actor_key_hash="hash",
        actor_key_data={
            "repo_id": "test/llama", "gpu_uuid": "GPU-A", "replica_gpu_uuids": ["GPU-A"],
        },
    )
    candidate = replace(candidate, gpus=(gpu,))
    result = choose(spec(), candidate)
    assert result is not None and not result.uses_warm_actor
    assert result.idle_actor_replacement_gpu_count == 1
    assert (
        choose(spec(), replace(candidate, gpus=(replace(gpu, actor_state=ActorState.BUSY),)))
        is None
    )


def test_null_assignment_claims_are_exact_private_and_not_model_reusable():
    payload = {
        "job_id": "job-test",
        "attempt_id": "attempt-test",
        "lease_id": "lease-test",
        "gpu_uuids": ["GPU-A", "GPU-B"],
        "fence_epochs": {"GPU-A": 1, "GPU-B": 2},
        "expires_at": 4_000_000_000,
        "spec": spec(2, "distributed").model_dump(),
    }
    assignment = Assignment.from_mapping(payload)
    assert not assignment.lease.allow_actor_reuse
    assert _is_private_workload(assignment)
    assert json.loads(assignment.model_key)["model"] is None
    with pytest.raises(ValueError, match="claim count"):
        Assignment.from_mapping({**payload, "gpu_uuids": ["GPU-A"]})


@pytest.fixture
def torch_devices(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    return torch


def test_gpu_loader_provides_real_worker_local_devices_without_model_loading(torch_devices):
    torch = torch_devices
    loaded = GPUJobLoader(2)(None)
    assert loaded.model is None and loaded.tokenizer is None
    assert all(isinstance(device, torch.device) for device in loaded.devices)
    assert loaded.devices == (torch.device("cuda:0"), torch.device("cuda:1"))
    assert loaded.device == loaded.devices[0]
    with pytest.raises(RuntimeError, match="CUDA visibility"):
        GPUJobLoader(1)(None)


def test_model_free_gang_context_keeps_all_attempt_services_and_is_private(
    torch_devices,
    tmp_path,
    capsys,
):
    torch = torch_devices
    ray = _FakeRay()
    service = RuntimeExecutionService(ray=DetachedRayActorAdapter(ray))
    key = ActorKey("test", "GPU-A", None, replica_gpu_uuids=("GPU-A", "GPU-B"))
    assert ActorKey.from_mapping(key.as_dict()) == key
    prepared = service.prepare(key, GPUJobLoader(2), use_ray=True, cuda_visible_device="3,7")
    assert not prepared.reused and not ray.actors and len(ray.private_actors) == 1
    assert set(ray.last_options["resources"]) == {"topodrome_gpu_gpu_a", "topodrome_gpu_gpu_b"}
    assert ray.last_options["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "3,7"
    assert "lifetime" not in ray.last_options
    checkpoint_records, artifact_records, cleanup = [], [], []
    (tmp_path / "input.txt").write_text("pinned")

    def experiment(context):
        assert context.model is None and context.tokenizer is None
        assert context.device == torch.device("cuda:0")
        assert context.devices == (torch.device("cuda:0"), torch.device("cuda:1"))
        assert (context.rank, context.world_size, context.local_rank) == (0, 1, 0)
        assert context.is_main_process and context.restore_checkpoint() == {"step": 4}
        assert context.committed_inputs["input"].read_text() == "pinned"
        assert context.scratch == tmp_path / "scratch"
        context.add_cleanup(cleanup.append, "closed")
        assert context.report_progress(5, 10, stage="training")
        context.checkpoint({"step": 5})
        context.register_artifact(context.committed_inputs["input"], logical_path="result.txt")
        assert not context.should_preempt()
        context.request_preemption("test")
        assert context.should_preempt() and context.checkpoint_due()
        return "done"

    assert (
        service.execute(
            prepared,
            experiment,
            job_id="job-test",
            attempt_id="attempt-test",
            workspace=tmp_path,
            scratch=tmp_path / "scratch",
            committed_inputs={"input": tmp_path / "input.txt"},
            checkpoint_restore=CheckpointRestore("old/checkpoint", {"step": 4}),
            checkpoint_handler=checkpoint_records.append,
            artifact_handler=artifact_records.append,
        )
        == "done"
    )
    assert cleanup == ["closed"] and len(checkpoint_records) == len(artifact_records) == 1
    assert '"stage":"training"' in capsys.readouterr().out
    assert service.evict(prepared) and not ray.private_actors


def test_model_free_distributed_context_exposes_rank_and_only_own_cuda_zero(
    torch_devices,
    monkeypatch,
):
    torch = torch_devices
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    ray = _FakeRay()
    adapter = DetachedRayActorAdapter(ray)
    keys = tuple(ActorKey("test", f"GPU-{index}", None) for index in (3, 7))
    group = adapter.create_worker_group(
        RayWorkerGroupSpec(keys, cuda_visible_devices=("3", "7")),
        GPUJobLoader(1),
    )
    calls, checkpoint_records = [], []

    def experiment(context):
        assert context.devices == (torch.device("cuda:0"),)
        assert context.device == torch.device("cuda:0") and context.local_rank == 0
        calls.append((context.rank, context.world_size, context.is_main_process))
        context.checkpoint({"rank": context.rank})
        return context.rank

    assert (
        group.run(
            experiment,
            job_id="job-test",
            attempt_id="attempt-test",
            checkpoint_handler=checkpoint_records.append,
        )
        == 0
    )
    assert calls == [(0, 2, True), (1, 2, False)]
    assert len(checkpoint_records) == 1
    assert group.evict() and not ray.private_actors
    assert len(ray.util.removed) == 1


def test_model_free_failure_runs_cleanup_and_confirms_disposal(torch_devices):
    ray = _FakeRay()
    service = RuntimeExecutionService(ray=DetachedRayActorAdapter(ray))
    prepared = service.prepare(
        ActorKey("test", "GPU-A", None, replica_gpu_uuids=("GPU-A", "GPU-B")),
        GPUJobLoader(2),
        use_ray=True,
        cuda_visible_device="3,7",
    )
    closed = []

    def fail(context):
        context.add_cleanup(closed.append, True)
        raise RuntimeError("script failure")

    with pytest.raises(RuntimeError, match="script failure"):
        service.execute(prepared, fail, job_id="job-test", attempt_id="attempt-test")
    assert closed == [True] and not ray.private_actors


@pytest.mark.parametrize(
    "identity",
    [
        {"rank": 2, "world_size": 2},
        {"rank": -1},
        {"world_size": 0},
        {"local_rank": -1},
    ],
)
def test_context_rejects_invalid_distributed_identity(identity):
    with pytest.raises(ValueError):
        ExperimentContext(
            model=None, tokenizer=None, device="cuda:0", job_id="j", attempt_id="a", **identity
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("gpus,execution", [(1, None), (2, "single"), (2, "distributed")])
async def test_backend_bypasses_model_services_and_never_advertises_empty_warm_actors(
    tmp_path,
    gpus,
    execution,
):
    observed = []

    def unexpected(*args, **kwargs):
        pytest.fail("model-free jobs must not acquire/load/advertise/reuse a model")

    class Runtime:
        core = SimpleNamespace(is_live=unexpected)

        async def prepare(self, key, loader, **options):
            observed.append((key, loader, options))
            return PreparedActor(
                actor_id="private",
                worker_pids=(42,),
                key=key,
                reused=False,
                transport=ExecutionTransport.RAY,
                _handle=object(),
                _owner="test",
                group_keys=(key,),
            )

        async def prepare_group(self, keys, loader, **options):
            observed.append((keys, loader, options))
            return PreparedActor(
                actor_id="private",
                worker_pids=(42, 43),
                key=keys[0],
                reused=False,
                transport=ExecutionTransport.RAY_GROUP,
                _handle=object(),
                _owner="test",
                group_keys=keys,
            )

    uuids = ("GPU-A", "GPU-B")[:gpus]
    assignment = Assignment.from_mapping(
        {
            "job_id": "j",
            "attempt_id": "a",
            "lease_id": "l",
            "gpu_uuids": uuids,
            "fence_epochs": {uuid: 1 for uuid in uuids},
            "expires_at": 4_000_000_000,
            "spec": spec(gpus, execution).model_dump(),
        }
    )
    assignment = replace(assignment, gpu_indices=dict(zip(uuids, (3, 7), strict=False)))
    backend = RuntimeExecutionBackend(
        site_id="test",
        registry=ModelRegistry(),
        runtime=Runtime(),
        storage=LocalStorage(tmp_path / "global"),
        use_ray=True,
        acquirer=unexpected,
        loader_factory=unexpected,
    )
    backend._ray_connected = True
    workspace = WorkspaceManager(
        storage=backend.storage,
        scratch_root=tmp_path / "scratch",
    ).prepare("j", "a")
    prepared = await backend.prepare(assignment, workspace)
    key_or_keys, loader, options = observed[0]
    assert isinstance(loader, GPUJobLoader)
    assert loader.visible_gpu_count == (1 if execution == "distributed" else gpus)
    if execution == "distributed":
        assert options["cuda_visible_devices"] == uuids
        assert all(key.model is None for key in key_or_keys)
    else:
        assert key_or_keys.model is None and key_or_keys.replica_gpu_uuids == uuids
        assert options["cuda_visible_device"] == ",".join(uuids)
    assert not prepared.reused and backend.model_capabilities() == ()
    assert backend.supports_model_free


def test_model_free_workers_cannot_execute_inside_the_site_agent_process():
    from topodrome.runtime.service import RuntimeExecutionError

    with pytest.raises(RuntimeExecutionError, match="isolated Ray worker"):
        RuntimeExecutionService().prepare(ActorKey("test", "GPU-A", None), GPUJobLoader())
