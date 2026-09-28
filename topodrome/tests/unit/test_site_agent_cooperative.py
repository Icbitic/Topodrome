from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from topodrome.domain.specs import AttemptReport
from topodrome.site_agent.admission import (
    GIB,
    AdmissionController,
    LeaseFenceStore,
    LeaseRequest,
)
from topodrome.site_agent.agent import (
    Assignment,
    ExecutionOutcome,
    PreparedRuntime,
    SiteAgent,
)
from topodrome.site_agent.cooperative import (
    CooperativeExitPolicy,
    CooperativeExitVerifier,
)
from topodrome.site_agent.inventory import (
    GPUProcess,
    GPUSnapshot,
    ProcessOwner,
    ProcessOwnershipRegistry,
)
from topodrome.site_agent.pressure import PressurePolicy, SustainedPressureMonitor
from topodrome.site_agent.workspace import WorkspaceManager

pytestmark = pytest.mark.unit


def snapshot(*processes: GPUProcess, free_gib: float = 20) -> GPUSnapshot:
    free_bytes = int(free_gib * 1024**3)
    return GPUSnapshot(
        uuid="GPU-A",
        index=0,
        name="GPU",
        total_memory_bytes=32 * 1024**3,
        free_memory_bytes=free_bytes,
        used_memory_bytes=32 * 1024**3 - free_bytes,
        utilization_percent=90,
        processes=processes,
    )


class SequencedInventory:
    def __init__(self, values: Sequence[Sequence[GPUSnapshot]]) -> None:
        self.values = list(values)
        self.calls = 0

    def sample(self) -> tuple[GPUSnapshot, ...]:
        index = min(self.calls, len(self.values) - 1)
        self.calls += 1
        return tuple(self.values[index])


@pytest.mark.asyncio
async def test_waits_until_cooperative_process_self_exits() -> None:
    cooperative = GPUProcess(400, 512 * 1024**2, ProcessOwner.COOPERATIVE)
    inventory = SequencedInventory([[snapshot(cooperative)], [snapshot()]])
    verifier = CooperativeExitVerifier(
        inventory,
        CooperativeExitPolicy(exit_timeout_seconds=1, poll_interval_seconds=0.001),
    )

    result = await verifier.wait(("GPU-A",))
    assert result.exited
    assert inventory.calls == 2


@pytest.mark.asyncio
async def test_persistent_cooperative_process_times_out_without_signalling() -> None:
    cooperative = GPUProcess(401, 512 * 1024**2, ProcessOwner.COOPERATIVE)
    inventory = SequencedInventory([[snapshot(cooperative)]])
    verifier = CooperativeExitVerifier(
        inventory,
        CooperativeExitPolicy(exit_timeout_seconds=0, poll_interval_seconds=0.001),
    )

    result = await verifier.wait(("GPU-A",))
    assert not result.exited
    assert result.remaining_pids == (401,)


class FakeOutbound:
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, Any]]] = []

    async def publish(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.events.append((kind, payload))

    async def publish_confirmed(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.events.append((kind, payload))


class DisconnectingOutbound(FakeOutbound):
    async def publish_confirmed(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.events.append((kind, payload))
        raise ConnectionError("scheduler ACK was not received")


class FakeRuntime:
    def __init__(self) -> None:
        self.prepared = False
        self.executed = False
        self.evicted = False

    async def prepare(self, assignment: Assignment, workspace: Any) -> PreparedRuntime:
        self.prepared = True
        return PreparedRuntime("actor-1", assignment.model_key, False, process_ids=(901,))

    async def execute(
        self, prepared: PreparedRuntime, assignment: Assignment, workspace: Any
    ) -> ExecutionOutcome:
        self.executed = True
        return ExecutionOutcome(True, artifacts_verified=True)

    async def request_preemption(self, attempt_id: str) -> None:  # pragma: no cover
        raise AssertionError

    async def cancel(self, attempt_id: str) -> None:  # pragma: no cover
        raise AssertionError

    async def evict(self, prepared: PreparedRuntime) -> None:
        self.evicted = True


class UnusedRay:
    async def ensure_running(self) -> bool:  # pragma: no cover
        return False

    async def is_running(self) -> bool:  # pragma: no cover
        return True

    async def stop(self) -> None:  # pragma: no cover
        return None


class FakeSentinel:
    def __init__(self, pid: int = 900) -> None:
        self._pids = (pid,)
        self.alive = True
        self.stopped = False

    @property
    def pids(self) -> tuple[int, ...]:
        return self._pids

    def is_alive(self) -> bool:
        return self.alive

    async def stop(self) -> None:
        self.stopped = True
        self.alive = False


class FakeSentinelFactory:
    def __init__(self, sentinel: FakeSentinel) -> None:
        self.sentinel = sentinel
        self.requests: list[Mapping[str, int]] = []

    async def start(self, gpu_indices: Mapping[str, int]) -> FakeSentinel:
        self.requests.append(dict(gpu_indices))
        return self.sentinel


@pytest.mark.asyncio
async def test_agent_does_not_prepare_or_execute_before_durable_acceptance_ack(
    tmp_path: Path,
) -> None:
    inventory = SequencedInventory([[snapshot(free_gib=25)]])
    fences = LeaseFenceStore(tmp_path / "leases.sqlite3")
    admission = AdmissionController(
        inventory,
        fences,
        safety_margin_bytes=GIB,
        allow_foreign_processes_when_capacity_sufficient=True,
    )
    outbound = DisconnectingOutbound()
    runtime = FakeRuntime()
    (tmp_path / "global").mkdir()
    agent = SiteAgent(
        site_id="lab1",
        inventory=inventory,
        admission=admission,
        fences=fences,
        ray=UnusedRay(),
        workspaces=WorkspaceManager(
            global_root=tmp_path / "global", scratch_root=tmp_path / "scratch"
        ),
        backend=runtime,
        outbound=outbound,  # type: ignore[arg-type]
        pressure=SustainedPressureMonitor(PressurePolicy()),
    )
    assignment = Assignment(
        job_id="job-ack",
        attempt_id="attempt-ack",
        lease=LeaseRequest(
            lease_id="lease-ack",
            attempt_id="attempt-ack",
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 1},
            min_free_vram_bytes=15 * GIB,
            expires_at=4_000_000_000,
            model_key="model-key",
        ),
        model_key="model-key",
        entrypoint="experiment.py",
        fencing_token=secrets.token_urlsafe(),
    )

    assert not await agent.offer(assignment)
    assert [kind for kind, _ in outbound.events] == ["assignment_accepted"]
    assert not runtime.prepared
    assert not runtime.executed
    assert fences.active_claims() == {}
    assert agent._active == {}
    fences.close()


@pytest.mark.asyncio
async def test_agent_never_reports_running_when_scavenger_does_not_exit(
    tmp_path: Path,
) -> None:
    cooperative = GPUProcess(402, 512 * 1024**2, ProcessOwner.COOPERATIVE)
    gpu_with_scavenger = snapshot(cooperative)
    inventory = SequencedInventory([[gpu_with_scavenger]])
    fences = LeaseFenceStore(tmp_path / "leases.sqlite3")
    admission = AdmissionController(inventory, fences, safety_margin_bytes=GIB)
    outbound = FakeOutbound()
    runtime = FakeRuntime()
    sentinel = FakeSentinel()
    registry = ProcessOwnershipRegistry(proc_root=tmp_path / "proc")
    (tmp_path / "global").mkdir()
    agent = SiteAgent(
        site_id="remote",
        inventory=inventory,
        admission=admission,
        fences=fences,
        ray=UnusedRay(),
        workspaces=WorkspaceManager(
            global_root=tmp_path / "global", scratch_root=tmp_path / "scratch"
        ),
        backend=runtime,
        outbound=outbound,  # type: ignore[arg-type]
        pressure=SustainedPressureMonitor(PressurePolicy()),
        cooperative_exit=CooperativeExitVerifier(
            inventory,
            CooperativeExitPolicy(exit_timeout_seconds=0, poll_interval_seconds=0.001),
        ),
        process_ownership=registry,
        sentinel_factory=FakeSentinelFactory(sentinel),
    )
    assignment = Assignment(
        job_id="job-1",
        attempt_id="attempt-1",
        lease=LeaseRequest(
            lease_id="lease-1",
            attempt_id="attempt-1",
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 1},
            min_free_vram_bytes=15 * GIB,
            expires_at=4_000_000_000,
            model_key="model-key",
        ),
        model_key="model-key",
        entrypoint="experiment.py",
    )

    assert not await agent.offer(assignment)

    states = [payload.get("state") for kind, payload in outbound.events if kind == "attempt_event"]
    assert "running" not in states
    assert not runtime.executed
    assert not runtime.evicted
    assert sentinel.stopped
    assert any(kind == "assignment_rejected" for kind, _ in outbound.events)
    assert fences.active_claims() == {}
    assert fences.actor_allocation("GPU-A") is None
    fences.close()


@pytest.mark.asyncio
async def test_agent_reclaims_with_sentinel_then_loads_before_running(
    tmp_path: Path,
) -> None:
    cooperative = GPUProcess(403, 29 * 1024**3, ProcessOwner.COOPERATIVE)
    owned_sentinel = GPUProcess(900, 128 * 1024**2, ProcessOwner.TOPODROME)
    inventory = SequencedInventory(
        [
            [snapshot(cooperative, free_gib=2)],
            [snapshot(owned_sentinel, free_gib=25)],
            [snapshot(owned_sentinel, free_gib=25)],
        ]
    )
    fences = LeaseFenceStore(tmp_path / "leases.sqlite3")
    admission = AdmissionController(inventory, fences, safety_margin_bytes=5 * GIB)
    outbound = FakeOutbound()
    runtime = FakeRuntime()
    sentinel = FakeSentinel()
    factory = FakeSentinelFactory(sentinel)
    registry = ProcessOwnershipRegistry(proc_root=tmp_path / "proc")
    process = tmp_path / "proc" / "901"
    process.mkdir(parents=True)
    fields_before_start = " ".join("0" for _ in range(17))
    (process / "stat").write_text(
        f"901 (ray worker) S 1 {fields_before_start} 12345\n",
        encoding="utf-8",
    )
    (tmp_path / "global").mkdir()
    agent = SiteAgent(
        site_id="remote",
        inventory=inventory,
        admission=admission,
        fences=fences,
        ray=UnusedRay(),
        workspaces=WorkspaceManager(
            global_root=tmp_path / "global", scratch_root=tmp_path / "scratch"
        ),
        backend=runtime,
        outbound=outbound,  # type: ignore[arg-type]
        pressure=SustainedPressureMonitor(PressurePolicy()),
        cooperative_exit=CooperativeExitVerifier(
            inventory,
            CooperativeExitPolicy(exit_timeout_seconds=1, poll_interval_seconds=0.001),
        ),
        process_ownership=registry,
        sentinel_factory=factory,
    )
    assignment = Assignment(
        job_id="job-2",
        attempt_id="attempt-2",
        lease=LeaseRequest(
            lease_id="lease-2",
            attempt_id="attempt-2",
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 2},
            min_free_vram_bytes=15 * GIB,
            expires_at=4_000_000_000,
            model_key="model-key",
        ),
        model_key="model-key",
        entrypoint="experiment.py",
    )

    assert await agent.offer(assignment)
    await agent.wait_for_idle()

    assert factory.requests == [{"GPU-A": 0}]
    assert sentinel.stopped
    assert runtime.executed
    event_kinds = [kind for kind, _ in outbound.events]
    assert event_kinds[0] == "assignment_accepted"
    states = [payload.get("state") for kind, payload in outbound.events if kind == "attempt_event"]
    assert states == ["starting", "running", "committing", "succeeded"]
    for kind, payload in outbound.events:
        if kind == "attempt_event":
            AttemptReport.model_validate(payload)
    assert not registry.owns(900)
    assert registry.owns(901)
    assert fences.active_claims() == {}
    assert fences.actor_allocation("GPU-A").state == "idle"  # type: ignore[union-attr]
    assert not (tmp_path / "scratch/jobs/job-2/attempts/attempt-2").exists()
    fences.close()


@pytest.mark.asyncio
async def test_adapter_training_uses_fresh_actor_then_always_evicts_it(
    tmp_path: Path,
) -> None:
    inventory = SequencedInventory([[snapshot(free_gib=25)]])
    fences = LeaseFenceStore(tmp_path / "adapter-leases.sqlite3")
    outbound = FakeOutbound()
    runtime = FakeRuntime()
    (tmp_path / "adapter-global").mkdir()
    agent = SiteAgent(
        site_id="lab1",
        inventory=inventory,
        admission=AdmissionController(
            inventory,
            fences,
            safety_margin_bytes=GIB,
            allow_foreign_processes_when_capacity_sufficient=True,
        ),
        fences=fences,
        ray=UnusedRay(),
        workspaces=WorkspaceManager(
            global_root=tmp_path / "adapter-global",
            scratch_root=tmp_path / "adapter-scratch",
        ),
        backend=runtime,
        outbound=outbound,  # type: ignore[arg-type]
        pressure=SustainedPressureMonitor(PressurePolicy()),
    )
    assignment = Assignment(
        job_id="job-adapter",
        attempt_id="attempt-adapter",
        lease=LeaseRequest(
            lease_id="lease-adapter",
            attempt_id="attempt-adapter",
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 1},
            min_free_vram_bytes=15 * GIB,
            expires_at=4_000_000_000,
            model_key="model-key",
        ),
        model_key="model-key",
        entrypoint="train_adapter.py",
        raw_spec={"workload_kind": "adapter_training"},
    )

    assert await agent.offer(assignment)
    await agent.wait_for_idle()

    assert runtime.executed
    assert runtime.evicted
    assert fences.active_claims() == {}
    assert fences.actor_allocation("GPU-A") is None
    assert not (
        tmp_path / "adapter-scratch/jobs/job-adapter/attempts/attempt-adapter"
    ).exists()
    fences.close()


@pytest.mark.asyncio
async def test_adapter_training_rejects_explicit_warm_actor_before_admission(
    tmp_path: Path,
) -> None:
    inventory = SequencedInventory([[snapshot(free_gib=25)]])
    fences = LeaseFenceStore(tmp_path / "adapter-reuse-leases.sqlite3")
    outbound = FakeOutbound()
    runtime = FakeRuntime()
    (tmp_path / "adapter-reuse-global").mkdir()
    agent = SiteAgent(
        site_id="lab1",
        inventory=inventory,
        admission=AdmissionController(inventory, fences, safety_margin_bytes=GIB),
        fences=fences,
        ray=UnusedRay(),
        workspaces=WorkspaceManager(
            global_root=tmp_path / "adapter-reuse-global",
            scratch_root=tmp_path / "adapter-reuse-scratch",
        ),
        backend=runtime,
        outbound=outbound,  # type: ignore[arg-type]
        pressure=SustainedPressureMonitor(PressurePolicy()),
    )
    assignment = Assignment(
        job_id="job-adapter-reuse",
        attempt_id="attempt-adapter-reuse",
        lease=LeaseRequest(
            lease_id="lease-adapter-reuse",
            attempt_id="attempt-adapter-reuse",
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 1},
            min_free_vram_bytes=15 * GIB,
            expires_at=4_000_000_000,
            reuse_actor_id="actor-warm",
            model_key="model-key",
        ),
        model_key="model-key",
        entrypoint="train_adapter.py",
        raw_spec={"workload_kind": "adapter_training"},
    )

    assert not await agent.offer(assignment)
    assert not runtime.prepared
    assert fences.active_claims() == {}
    assert any(kind == "assignment_rejected" for kind, _ in outbound.events)
    fences.close()
