from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from topodrome.config import IdleComputeSettings
from topodrome.site_agent.admission import GIB, LeaseFenceStore, LeaseRequest
from topodrome.site_agent.idle_compute import IdleComputeManager, run_worker
from topodrome.site_agent.inventory import (
    CooperativeProcessMatcher,
    GPUProcess,
    GPUReservation,
    GPUSnapshot,
    ProcessOwner,
    ProcessOwnershipRegistry,
)
from topodrome.site_agent.spool import SpoolEvent

pytestmark = pytest.mark.unit


@pytest.fixture
def manager(tmp_path: Path):
    fences = LeaseFenceStore(tmp_path / "leases.sqlite3")
    result = IdleComputeManager(
        IdleComputeSettings(enabled=True), fences,
        ProcessOwnershipRegistry(), CooperativeProcessMatcher(),
    )
    result.acknowledge(event(), permission())
    yield result
    fences.close()


def event(**values):
    return SpoolEvent(1, "event", "heartbeat", {}, values.get("created_at", time.time()))


def permission(**values):
    return {"idle_compute_permission": {
        "allowed": True, "blocked_gpu_uuids": [], "valid_for_seconds": 15, **values,
    }}


def gpu(**values):
    return GPUSnapshot(
        uuid="GPU-A", index=0, name="A800", total_memory_bytes=80 * GIB,
        free_memory_bytes=80 * GIB, used_memory_bytes=0, utilization_percent=0,
        **values,
    )


def test_explicit_opt_in_and_fresh_control_are_required(manager):
    assert IdleComputeSettings().enabled is False
    assert manager.eligible(gpu()) == {}
    manager._permit_until = 0
    assert manager.eligible(gpu()) is None
    manager.acknowledge(event(created_at=time.time() - 20), permission())
    assert manager.eligible(gpu()) is None
    manager.acknowledge(event(), {**permission(), "duplicate": True})
    assert manager.eligible(gpu()) is None
    manager.acknowledge(event(), permission(allowed=False))
    assert manager.eligible(gpu()) is None


def test_reservations_foreign_processes_unknown_actors_and_pressure_block_load(manager):
    manager.acknowledge(event(), permission(blocked_gpu_uuids=["GPU-A"]))
    assert manager.eligible(gpu()) is None
    manager.acknowledge(event(), permission())
    reservation = GPUReservation(0, 123, "owner", 1, time.time(), 100)
    assert manager.eligible(gpu(external_reservation=reservation)) is None
    for owner in (ProcessOwner.FOREIGN, ProcessOwner.COOPERATIVE, ProcessOwner.TOPODROME):
        assert manager.eligible(gpu(processes=(GPUProcess(123, 1024, owner),))) is None
    assert manager.eligible(replace(gpu(), free_memory_bytes=GIB)) is None
    assert manager.eligible(replace(gpu(), utilization_percent=99)) is None


def test_active_claim_and_stopping_allocation_are_ineligible(manager):
    request = LeaseRequest(
        lease_id="lease", attempt_id="attempt", gpu_uuids=("GPU-A",),
        fence_epochs={"GPU-A": 1}, min_free_vram_bytes=GIB,
        expires_at=time.time() + 90,
    )
    manager.fences.claim(request, fresh_eligible=frozenset({"GPU-A"}),
                         pressure_safe=frozenset({"GPU-A"}))
    assert manager.eligible(gpu()) is None
    manager.fences.release_lease("lease")
    manager.fences.reserve_actor_group(
        gpu_uuids=("GPU-A",), actor_id="actor", model_key="model", state="stopping",
    )
    assert manager.eligible(gpu()) is None


def test_only_exact_live_idle_warm_replica_is_allowlisted(manager):
    pid = os.getpid()
    manager.ownership.register(pid, actor_id="actor")
    manager.fences.reserve_actor_group(
        gpu_uuids=("GPU-A", "GPU-B"), actor_id="actor", model_key="model", state="idle",
    )
    snapshot = gpu(processes=(GPUProcess(pid, GIB, ProcessOwner.TOPODROME),))
    allowed = manager.eligible(snapshot)
    assert allowed is not None and pid in allowed
    assert allowed[pid] == manager.ownership.exact_owner(pid, require_live=True).start_ticks
    manager.fences.set_actor_state("actor", "busy")
    assert manager.eligible(snapshot) is None


@pytest.mark.asyncio
async def test_own_filler_is_cooperative_reused_and_stopped_on_reservation(
    manager, monkeypatch, tmp_path,
):
    class Pipe:
        def __init__(self):
            self.records = []
            self.closed = False

        def write(self, value):
            self.records.append(value)

        async def drain(self):
            pass

        def close(self):
            self.closed = True

    pipe = Pipe()
    process = SimpleNamespace(pid=901, returncode=None, stdin=pipe)

    async def wait():
        process.returncode = 0
        return 0

    process.wait = wait
    launches = []

    async def spawn(*args, **kwargs):
        launches.append((args, kwargs))
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
    storage_paths = {
        key: str(tmp_path / key.lower()) for key in (
            "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "CUDA_CACHE_PATH",
            "TRITON_CACHE_DIR", "TORCH_HOME", "HF_HOME",
        )
    }
    for key, value in storage_paths.items():
        monkeypatch.setenv(key, value)
    excluded = ("TOPODROME_SITE_ENROLLMENT_TOKEN", "HF_TOKEN", "ARBITRARY_SETTING")
    for key in excluded:
        monkeypatch.setenv(key, "must-not-inherit")
    await manager.tick([gpu()])
    assert manager.cooperative.matches(901)
    environment = launches[0][1]["env"]
    assert all(environment[key] == value for key, value in storage_paths.items())
    assert not any(key in environment for key in excluded)
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    await manager.tick([replace(gpu(), utilization_percent=100, processes=(
        GPUProcess(901, 600 * 1024**2, ProcessOwner.COOPERATIVE),
    ))])
    assert len(launches) == 1 and len(pipe.records) == 2
    manager.acknowledge(event(), permission(blocked_gpu_uuids=["GPU-A"]))
    await manager.tick([gpu()])
    assert pipe.closed and not manager.workers and not manager.cooperative.matches(901)


@pytest.mark.asyncio
async def test_draining_never_launches_helpers(manager, monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("draining site must not spawn")

    monkeypatch.setattr("asyncio.create_subprocess_exec", forbidden)
    await manager.tick([gpu()], draining=True)
    assert not manager.workers


def test_worker_batches_compute_and_yields_to_unknown_process(monkeypatch):
    ready, closed = threading.Event(), threading.Event()
    state = {"calls": 0, "synchronizations": 0, "foreign": False, "closed": False}

    class Input:
        @property
        def buffer(self):
            return self

        def __iter__(self):
            yield json.dumps({"allowed_pids": {}, "ttl": 3}).encode() + b"\n"
            ready.set()
            closed.wait(2)

    def initialize():
        assert ready.wait(2)

    def compute(*args, **kwargs):
        state["calls"] += 1
        state["foreign"] = True

    def synchronize():
        state["synchronizations"] += 1
        assert state["calls"] % 8 == 0

    def cleanup():
        state["closed"] = True

    nvml = SimpleNamespace(
        nvmlInit=initialize, nvmlShutdown=cleanup,
        nvmlDeviceGetHandleByUUID=lambda uuid: uuid,
        nvmlDeviceGetComputeRunningProcesses=lambda _: (
            [SimpleNamespace(pid=123456789)] if state["foreign"] else []
        ),
        nvmlDeviceGetGraphicsRunningProcesses=lambda _: [],
        NVMLError=RuntimeError, NVMLError_NotSupported=NotImplementedError,
    )
    torch = SimpleNamespace(
        set_num_threads=lambda _: None, float16="fp16",
        randn=lambda *args, **kwargs: object(), randn_like=lambda _: object(),
        empty_like=lambda _: object(), mm=compute,
        cuda=SimpleNamespace(synchronize=synchronize),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(sys, "stdin", Input())
    try:
        run_worker("GPU-A", 512)
    finally:
        closed.set()
    assert state["calls"] >= 8 and state["synchronizations"] >= 1 and state["closed"]
