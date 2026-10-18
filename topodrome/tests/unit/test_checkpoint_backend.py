import asyncio
import json
import threading
from dataclasses import replace

from tests.unit.test_checkpoint_switch import tiny_checkpoints as _tiny_checkpoints
from tests.unit.test_runtime_ray_actor import _FakeRay
from topodrome.domain.specs import JobSpec
from topodrome.runtime import (
    AsyncRuntimeExecutionService,
    DetachedRayActorAdapter,
    ModelRegistry,
    RuntimeExecutionService,
    TransformersLoader,
)
from topodrome.site_agent.admission import ActorAllocation, LeaseRequest
from topodrome.site_agent.agent import Assignment
from topodrome.site_agent.runtime_backend import RuntimeExecutionBackend
from topodrome.site_agent.workspace import WorkspaceManager
from topodrome.storage.local import LocalStorage

tiny_checkpoints = _tiny_checkpoints


async def test_backend_uses_exact_old_handle_and_rebinds_after_switch(
    tmp_path, tiny_checkpoints, monkeypatch
):
    _, _, _, loaders = tiny_checkpoints
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    registry = ModelRegistry([loader.registration for loader in loaders])
    core = RuntimeExecutionService(ray=DetachedRayActorAdapter(_FakeRay()))
    backend = RuntimeExecutionBackend(
        site_id="test-site",
        registry=registry,
        runtime=AsyncRuntimeExecutionService(core),
        storage=LocalStorage(tmp_path / "global"),
        loader_factory=lambda reg: TransformersLoader(reg, device="cpu"),
    )
    monkeypatch.setattr(backend, "_ensure_ray_driver", lambda: None)
    workspaces = WorkspaceManager(
        global_root=tmp_path / "global", scratch_root=tmp_path / "scratch"
    )

    def assignment(index):
        spec = JobSpec(
            entrypoint="run.py",
            model={
                "repo_id": "test/olmo3",
                "revision": str(index),
                "backend_version": "configured",
                "dtype": "float32",
            },
            resources={"gpu": 1, "min_vram_gb": 1},
        )
        serialized = json.dumps(
            {
                "model": spec.model.model_dump(mode="json"),
                "runtime_fingerprint": spec.runtime_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return Assignment(
            f"job-{index}",
            f"attempt-{index}",
            LeaseRequest(
                f"lease-{index}",
                f"attempt-{index}",
                ("GPU-A",),
                {"GPU-A": index + 1},
                1024,
                4_000_000_000,
                model_key=serialized,
                replace_idle_actors=bool(index),
            ),
            serialized,
            "run.py",
            raw_spec=spec.model_dump(),
            gpu_indices={"GPU-A": 0},
        )

    first = assignment(0)
    prepared = await backend.prepare(first, workspaces.prepare(first.job_id, first.attempt_id))
    old = backend._prepared[prepared.actor_id][0]
    second = assignment(1)
    allocation = ActorAllocation("GPU-A", prepared.actor_id, first.model_key, "idle")
    source = await backend.switchable_runtime(second, (allocation,))
    assert source.actor_id == prepared.actor_id
    second = replace(
        second,
        lease=replace(
            second.lease,
            runtime_switch_actor_id=source.actor_id,
            runtime_switch_model_key=source.model_key,
            runtime_switch_process_ids=source.process_ids,
        ),
    )
    replacing = threading.Event()
    resume = threading.Event()
    original_replace = TransformersLoader.replace

    def paused_replace(loader, loaded, previous_key):
        replacing.set()
        assert resume.wait(timeout=10), "test did not release checkpoint replacement"
        return original_replace(loader, loaded, previous_key)

    monkeypatch.setattr(TransformersLoader, "replace", paused_replace)
    task = asyncio.create_task(
        backend.prepare(second, workspaces.prepare(second.job_id, second.attempt_id))
    )
    assert await asyncio.to_thread(replacing.wait, 5)
    # This is the real transitional state that the heartbeat used to mistake
    # for dead ownership. It must not be advertised as either checkpoint.
    assert not core.is_live(old)
    reconciliation = asyncio.create_task(backend.reconcile_warm_liveness())
    try:
        assert await asyncio.wait_for(reconciliation, timeout=1) == ()
        assert prepared.actor_id in backend._prepared
    finally:
        resume.set()
        new = await task
    assert await backend.reconcile_warm_liveness() == ()
    assert new.actor_id == prepared.actor_id and new.process_ids == prepared.process_ids
    assert new.model_key == second.model_key and not new.reused
    assert backend._prepared[new.actor_id][1] == second.model_key
    assert not core.is_live(old, require_idle=True)  # Never advertise old weights under new bytes.
    assert core.is_live(backend._prepared[new.actor_id][0], require_idle=True)
    await backend.evict(new)
