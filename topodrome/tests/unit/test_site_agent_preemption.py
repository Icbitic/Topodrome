from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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
from topodrome.site_agent.inventory import GPUSnapshot, ProcessOwnershipRegistry
from topodrome.site_agent.pressure import (
    PressureEvent,
    PressurePolicy,
    SustainedPressureMonitor,
)
from topodrome.site_agent.workspace import AttemptWorkspace, WorkspaceManager

pytestmark = pytest.mark.unit


class _Inventory:
    def sample(self) -> tuple[GPUSnapshot, ...]:
        return (
            GPUSnapshot(
                uuid="GPU-A",
                index=0,
                name="GPU",
                total_memory_bytes=32 * GIB,
                free_memory_bytes=28 * GIB,
                used_memory_bytes=4 * GIB,
                utilization_percent=0,
            ),
        )


class _Outbound:
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, Any]]] = []

    async def publish(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.events.append((kind, payload))

    async def publish_confirmed(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.events.append((kind, payload))


class _Ray:
    async def stop(self) -> bool:
        return True


class _CooperativeBackend:
    def __init__(self, checkpoint_ref: str) -> None:
        self.checkpoint_ref = checkpoint_ref
        self.started = asyncio.Event()
        self.requested = asyncio.Event()
        self.forced = False
        self.evicted = False

    async def prepare(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> PreparedRuntime:
        return PreparedRuntime("actor-1", assignment.model_key, False, process_ids=(901,))

    async def execute(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> ExecutionOutcome:
        self.started.set()
        await self.requested.wait()
        return ExecutionOutcome(
            True,
            artifacts_verified=True,
            metadata={"checkpoint_ref": self.checkpoint_ref},
        )

    async def request_preemption(self, attempt_id: str) -> None:
        self.requested.set()
        await asyncio.sleep(0)

    async def cancel(self, attempt_id: str) -> None:
        self.requested.set()

    async def evict(self, prepared: PreparedRuntime) -> None:
        self.evicted = True

    async def force_stop(self, prepared: PreparedRuntime) -> bool:
        self.forced = True
        return True


class _ForcedBackend(_CooperativeBackend):
    def __init__(self, checkpoint_ref: str) -> None:
        super().__init__(checkpoint_ref)
        self.killed = asyncio.Event()

    async def execute(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> ExecutionOutcome:
        self.started.set()
        await self.killed.wait()
        raise RuntimeError("actor died")

    async def request_preemption(self, attempt_id: str) -> None:
        return None

    async def force_stop(self, prepared: PreparedRuntime) -> bool:
        assert prepared.actor_id == "actor-1"
        self.forced = True
        self.killed.set()
        return True

    async def latest_checkpoint(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> str:
        return self.checkpoint_ref


class _FailedBackend(_CooperativeBackend):
    async def execute(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> ExecutionOutcome:
        self.started.set()
        raise RuntimeError("experiment failed after saving progress")

    async def latest_checkpoint(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> str:
        return self.checkpoint_ref


@pytest.mark.asyncio
async def test_dead_worker_does_not_wait_for_large_checkpoint_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from topodrome.site_agent import agent as module

    class SlowCheckpointBackend(_FailedBackend):
        async def latest_checkpoint(self, *args: Any) -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    monkeypatch.setattr(module, "_CHECKPOINT_RECOVERY_WAIT_SECONDS", 0.01)
    backend = SlowCheckpointBackend("unverified-checkpoint")
    agent, assignment, outbound, fences, _ownership = _build_agent(tmp_path, backend)
    assert await agent.offer(assignment)
    await asyncio.wait_for(agent.wait_for_idle(), timeout=1)
    assert _attempt_states(outbound)[-1] == "failed"
    assert backend.evicted
    assert fences.active_claims() == {}
    fences.close()


@pytest.mark.asyncio
async def test_cancel_during_failure_checkpoint_recovery_stops_waiting(
    tmp_path: Path,
) -> None:
    checkpoint_started = asyncio.Event()

    class SlowCheckpointBackend(_FailedBackend):
        async def latest_checkpoint(self, *args: Any) -> str:
            checkpoint_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    backend = SlowCheckpointBackend("unverified-checkpoint")
    agent, assignment, outbound, fences, _ownership = _build_agent(tmp_path, backend)
    assert await agent.offer(assignment)
    await asyncio.wait_for(checkpoint_started.wait(), timeout=1)
    await agent.cancel(assignment.attempt_id)
    await asyncio.wait_for(agent.wait_for_idle(), timeout=1)
    assert _attempt_states(outbound)[-1] == "canceled"
    assert backend.evicted
    assert fences.active_claims() == {}
    fences.close()


class _UncancellablePreemptionBackend(_ForcedBackend):
    def __init__(self) -> None:
        super().__init__("checkpoint-stubborn-control")
        self.control_gate = asyncio.Event()

    async def request_preemption(self, attempt_id: str) -> None:
        del attempt_id
        try:
            await self.control_gate.wait()
        except asyncio.CancelledError:
            # Simulate a third-party RPC wrapper that mishandles task
            # cancellation. The site agent must still reach exact hard stop.
            await self.control_gate.wait()


class _BlockingDisposalBackend(_CooperativeBackend):
    def __init__(self) -> None:
        super().__init__("checkpoint-unused")
        self.disposal_started = asyncio.Event()
        self.allow_disposal = asyncio.Event()

    async def execute(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> ExecutionOutcome:
        self.started.set()
        return ExecutionOutcome(
            True,
            keep_warm=False,
            artifacts_verified=True,
        )

    async def evict(self, prepared: PreparedRuntime) -> None:
        self.disposal_started.set()
        await self.allow_disposal.wait()
        self.evicted = True


class _SlowPreparationBackend(_CooperativeBackend):
    def __init__(self) -> None:
        super().__init__("")
        self.prepare_started = asyncio.Event()
        self.allow_prepare = asyncio.Event()
        self.execute_called = False

    async def prepare(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> PreparedRuntime:
        self.prepare_started.set()
        await self.allow_prepare.wait()
        return PreparedRuntime(
            "actor-slow-load",
            assignment.model_key,
            False,
            process_ids=(901,),
        )

    async def execute(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> ExecutionOutcome:
        self.execute_called = True
        return ExecutionOutcome(True, artifacts_verified=True)

    async def request_preemption(self, attempt_id: str) -> None:
        return None

    async def cancel(self, attempt_id: str) -> None:
        return None


def _build_agent(
    tmp_path: Path,
    backend: Any,
    *,
    grace_seconds: float = 0,
    cancellation_grace_seconds: float = 0,
) -> tuple[SiteAgent, Assignment, _Outbound, LeaseFenceStore, ProcessOwnershipRegistry]:
    inventory = _Inventory()
    fences = LeaseFenceStore(tmp_path / "leases.sqlite3")
    proc_root = tmp_path / "proc"
    runtime_process = proc_root / "901"
    runtime_process.mkdir(parents=True)
    fields_before_start = " ".join("0" for _ in range(17))
    (runtime_process / "stat").write_text(
        f"901 (ray worker) S 1 {fields_before_start} 12345\n",
        encoding="utf-8",
    )
    ownership = ProcessOwnershipRegistry(proc_root=proc_root)
    outbound = _Outbound()
    global_root = tmp_path / "global"
    global_root.mkdir()
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
        raw_spec={"checkpoint": {"grace_seconds": grace_seconds}},
    )
    agent = SiteAgent(
        site_id="lab1",
        inventory=inventory,
        admission=AdmissionController(inventory, fences, safety_margin_bytes=GIB),
        fences=fences,
        ray=_Ray(),  # type: ignore[arg-type]
        workspaces=WorkspaceManager(
            global_root=global_root,
            scratch_root=tmp_path / "scratch",
        ),
        backend=backend,
        outbound=outbound,  # type: ignore[arg-type]
        pressure=SustainedPressureMonitor(PressurePolicy()),
        process_ownership=ownership,
        cancellation_grace_seconds=cancellation_grace_seconds,
    )
    return agent, assignment, outbound, fences, ownership


def _attempt_states(outbound: _Outbound) -> list[str]:
    return [str(payload["state"]) for kind, payload in outbound.events if kind == "attempt_event"]


@pytest.mark.asyncio
async def test_starting_report_is_suppressed_when_preemption_wins_acceptance_race(
    tmp_path: Path,
) -> None:
    agent, assignment, outbound, fences, _ = _build_agent(
        tmp_path,
        _SlowPreparationBackend(),
    )
    agent._active = {  # type: ignore[assignment]
        assignment.attempt_id: SimpleNamespace(
            assignment=assignment,
            fencing_lock=asyncio.Lock(),
            renewal_pending=False,
            scheduler_terminal=False,
            preempt_requested=True,
        )
    }

    await agent._publish_attempt_state(assignment, "starting")

    assert _attempt_states(outbound) == []
    fences.close()


@pytest.mark.asyncio
async def test_preemption_during_model_load_never_invokes_user_code(
    tmp_path: Path,
) -> None:
    backend = _SlowPreparationBackend()
    agent, assignment, outbound, fences, _ = _build_agent(
        tmp_path,
        backend,
        grace_seconds=0,
    )

    assert await agent.offer(assignment)
    await backend.prepare_started.wait()
    await agent.request_preemption(assignment.attempt_id)
    backend.allow_prepare.set()
    await agent.wait_for_idle()

    assert not backend.execute_called
    assert backend.evicted
    assert _attempt_states(outbound) == [
        "starting",
        "preempt_requested",
        "checkpointing",
        "preempted",
    ]
    assert fences.active_claims() == {}
    assert fences.actor_allocation("GPU-A") is None
    fences.close()


@pytest.mark.asyncio
async def test_cancellation_during_model_load_never_invokes_user_code(
    tmp_path: Path,
) -> None:
    backend = _SlowPreparationBackend()
    agent, assignment, outbound, fences, _ = _build_agent(tmp_path, backend)

    assert await agent.offer(assignment)
    await backend.prepare_started.wait()
    await agent.cancel(assignment.attempt_id)
    backend.allow_prepare.set()
    await agent.wait_for_idle()

    assert not backend.execute_called
    assert backend.evicted
    assert _attempt_states(outbound) == ["starting", "canceled"]
    assert fences.active_claims() == {}
    assert fences.actor_allocation("GPU-A") is None
    fences.close()


@pytest.mark.asyncio
async def test_pressure_monitor_survives_one_preemption_transport_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _SlowPreparationBackend()
    agent, assignment, _outbound, fences, _ = _build_agent(tmp_path, backend)

    class AlwaysPressure:
        def observe(self, snapshots: object) -> tuple[PressureEvent, ...]:
            del snapshots
            return (PressureEvent("GPU-A", 0, 1, ()),)

    agent.pressure = AlwaysPressure()  # type: ignore[assignment]
    agent.pressure_interval_seconds = 0.001
    agent._active = {  # type: ignore[assignment]
        assignment.attempt_id: type("Active", (), {"assignment": assignment})()
    }
    calls = 0

    async def request_preemption(attempt_id: str) -> None:
        nonlocal calls
        assert attempt_id == assignment.attempt_id
        calls += 1
        if calls == 1:
            raise ConnectionError("synthetic transport failure")
        agent._stop.set()

    monkeypatch.setattr(agent, "request_preemption", request_preemption)
    await asyncio.wait_for(agent._pressure_loop(), timeout=1)

    assert calls == 2
    fences.close()


@pytest.mark.asyncio
async def test_cooperative_preemption_commits_checkpoint_and_stops_warm_actor(
    tmp_path: Path,
) -> None:
    checkpoint_ref = "jobs/job-1/attempts/attempt-1/checkpoints/00000001-progress.pkl"
    backend = _CooperativeBackend(checkpoint_ref)
    agent, assignment, outbound, fences, _ = _build_agent(
        tmp_path,
        backend,
        grace_seconds=0.5,
    )

    assert await agent.offer(assignment)
    await backend.started.wait()
    await agent.request_preemption(assignment.attempt_id)
    await agent.wait_for_idle()

    assert not backend.forced
    assert backend.evicted
    assert _attempt_states(outbound) == [
        "starting",
        "running",
        "preempt_requested",
        "checkpointing",
        "preempted",
    ]
    preempted = next(
        payload
        for kind, payload in outbound.events
        if kind == "attempt_event" and payload["state"] == "preempted"
    )
    assert preempted["checkpoint_ref"] == checkpoint_ref
    assert fences.active_claims() == {}
    assert fences.actor_allocation("GPU-A") is None
    fences.close()


@pytest.mark.asyncio
async def test_grace_timeout_force_stops_only_exact_owned_actor_and_reports_preempted(
    tmp_path: Path,
) -> None:
    checkpoint_ref = "jobs/job-1/attempts/attempt-1/checkpoints/00000002-progress.pkl"
    backend = _ForcedBackend(checkpoint_ref)
    agent, assignment, outbound, fences, ownership = _build_agent(tmp_path, backend)
    ownership.register(7777, lease_id="foreign-lease", actor_id="foreign-actor")

    assert await agent.offer(assignment)
    await backend.started.wait()
    await agent.request_preemption(assignment.attempt_id)
    await agent.wait_for_idle()

    assert backend.forced
    assert "failed" not in _attempt_states(outbound)
    assert _attempt_states(outbound)[-2:] == ["checkpointing", "preempted"]
    preempted = next(
        payload
        for kind, payload in outbound.events
        if kind == "attempt_event" and payload["state"] == "preempted"
    )
    assert preempted["checkpoint_ref"] == checkpoint_ref
    assert ownership.exact_owner(7777) is not None
    assert ownership.exact_owner(7777).actor_id == "foreign-actor"  # type: ignore[union-attr]
    assert fences.active_claims() == {}
    fences.close()


@pytest.mark.asyncio
async def test_cancel_force_stops_only_exact_owned_actor_after_cancellation_grace(
    tmp_path: Path,
) -> None:
    backend = _ForcedBackend("checkpoint-must-not-turn-cancel-into-retry")
    agent, assignment, outbound, fences, ownership = _build_agent(
        tmp_path,
        backend,
        # Cancellation has its own operator-owned bound. A job must not be
        # able to extend it through the checkpoint policy in its raw spec.
        grace_seconds=3_600,
        cancellation_grace_seconds=0,
    )
    ownership.register(7777, lease_id="foreign-lease", actor_id="foreign-actor")

    assert await agent.offer(assignment)
    await backend.started.wait()
    await asyncio.wait_for(agent.cancel(assignment.attempt_id), timeout=1)
    await agent.wait_for_idle()

    assert backend.requested.is_set()
    assert backend.forced
    states = _attempt_states(outbound)
    assert "preempt_requested" not in states
    assert "checkpointing" not in states
    assert "preempted" not in states
    assert "failed" not in states
    assert states[-1] == "canceled"
    canceled = next(
        payload
        for kind, payload in outbound.events
        if kind == "attempt_event" and payload["state"] == "canceled"
    )
    assert canceled["reason_code"] == "cancel_requested"
    assert ownership.exact_owner(7777) is not None
    assert ownership.exact_owner(7777).actor_id == "foreign-actor"  # type: ignore[union-attr]
    assert fences.active_claims() == {}
    assert fences.actor_allocation("GPU-A") is None
    fences.close()


@pytest.mark.asyncio
async def test_preemption_grace_does_not_wait_behind_uncancellable_control_rpc(
    tmp_path: Path,
) -> None:
    backend = _UncancellablePreemptionBackend()
    agent, assignment, _outbound, fences, _ownership = _build_agent(tmp_path, backend)

    assert await agent.offer(assignment)
    await backend.started.wait()
    await asyncio.wait_for(agent.request_preemption(assignment.attempt_id), timeout=1)
    assert backend.forced
    backend.control_gate.set()
    await agent.wait_for_idle()

    assert fences.active_claims() == {}
    fences.close()


@pytest.mark.asyncio
async def test_runtime_failure_reports_latest_durable_checkpoint_for_retry(
    tmp_path: Path,
) -> None:
    checkpoint_ref = "jobs/job-1/attempts/attempt-1/checkpoints/00000003-progress.pkl"
    backend = _FailedBackend(checkpoint_ref)
    agent, assignment, outbound, fences, _ = _build_agent(tmp_path, backend)

    assert await agent.offer(assignment)
    await agent.wait_for_idle()

    failed = next(
        payload
        for kind, payload in outbound.events
        if kind == "attempt_event" and payload["state"] == "failed"
    )
    assert failed["checkpoint_ref"] == checkpoint_ref
    assert backend.evicted
    assert fences.active_claims() == {}
    fences.close()


@pytest.mark.asyncio
async def test_terminal_and_local_release_wait_for_confirmed_actor_disposal(
    tmp_path: Path,
) -> None:
    backend = _BlockingDisposalBackend()
    agent, assignment, outbound, fences, _ = _build_agent(tmp_path, backend)

    assert await agent.offer(assignment)
    await backend.disposal_started.wait()

    assert "succeeded" not in _attempt_states(outbound)
    assert fences.active_claims() == {"GPU-A": "lease-1"}
    allocation = fences.actor_allocation("GPU-A")
    assert allocation is not None
    assert allocation.state == "stopping"

    backend.allow_disposal.set()
    await agent.wait_for_idle()

    assert backend.evicted
    assert _attempt_states(outbound)[-2:] == ["committing", "succeeded"]
    assert fences.active_claims() == {}
    assert fences.actor_allocation("GPU-A") is None
    fences.close()
