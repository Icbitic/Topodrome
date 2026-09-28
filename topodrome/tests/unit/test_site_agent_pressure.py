from __future__ import annotations

from pathlib import Path

import pytest

from topodrome.site_agent.inventory import (
    CooperativeProcessMatcher,
    GPUProcess,
    GPUSnapshot,
    ProcessOwner,
    ProcessOwnershipRegistry,
)
from topodrome.site_agent.pressure import (
    MIB,
    OwnedProcessSignaller,
    PressurePolicy,
    SustainedPressureMonitor,
)

pytestmark = pytest.mark.unit


def snapshot(*processes: GPUProcess) -> GPUSnapshot:
    return GPUSnapshot(
        uuid="GPU-A",
        index=0,
        name="shared GPU",
        total_memory_bytes=32 * 1024**3,
        free_memory_bytes=20 * 1024**3,
        used_memory_bytes=12 * 1024**3,
        utilization_percent=70,
        processes=processes,
        sampled_at=1,
    )


def _write_process_identity(proc_root: Path, pid: int, start_ticks: int) -> None:
    process = proc_root / str(pid)
    process.mkdir(parents=True, exist_ok=True)
    fields_before_start = " ".join("0" for _ in range(17))
    (process / "stat").write_text(
        f"{pid} (test process) S 1 {fields_before_start} {start_ticks}\n",
        encoding="utf-8",
    )


def test_pressure_must_be_sustained_and_emits_once_until_reset() -> None:
    foreign = GPUProcess(9123, 512 * MIB, ProcessOwner.FOREIGN)
    monitor = SustainedPressureMonitor(
        PressurePolicy(sustained_seconds=10, foreign_memory_threshold_bytes=256 * MIB)
    )

    assert monitor.observe([snapshot(foreign)], now=0) == ()
    assert monitor.observe([snapshot(foreign)], now=9.9) == ()
    event = monitor.observe([snapshot(foreign)], now=10)
    assert len(event) == 1
    assert event[0].foreign_pids == (9123,)
    assert monitor.observe([snapshot(foreign)], now=20) == ()

    assert monitor.observe([snapshot()], now=21) == ()
    assert monitor.observe([snapshot(foreign)], now=22) == ()
    assert len(monitor.observe([snapshot(foreign)], now=32)) == 1


def test_pressure_preemption_can_be_disabled_for_authorized_colocation() -> None:
    foreign = GPUProcess(9123, 8 * 1024**3, ProcessOwner.FOREIGN)
    monitor = SustainedPressureMonitor(PressurePolicy(enabled=False, sustained_seconds=0))
    assert monitor.observe([snapshot(foreign)], now=0) == ()


def test_signaller_never_targets_a_foreign_process(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_process_identity(proc_root, 100, 12345)
    registry = ProcessOwnershipRegistry(proc_root=proc_root)
    registry.register(100, lease_id="lease")
    calls: list[tuple[int, int]] = []
    signaller = OwnedProcessSignaller(registry, kill=lambda pid, sig: calls.append((pid, sig)))

    with pytest.raises(PermissionError, match="foreign process"):
        signaller.terminate(200)
    assert calls == []

    signaller.terminate(100)
    assert calls and calls[0][0] == 100

    _write_process_identity(proc_root, 100, 54321)
    with pytest.raises(PermissionError, match="foreign process"):
        signaller.terminate(100)


def test_cooperative_matcher_requires_explicit_cmdline_marker_or_pid(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    process_root = proc_root / "321"
    process_root.mkdir(parents=True)
    (process_root / "comm").write_text("gpu-scavenger\n")
    (process_root / "cmdline").write_bytes(b"python\0/opt/scavenge.py\0")
    matcher = CooperativeProcessMatcher(
        command_line_patterns=(r"scavenge\.py",),
        pids=(999,),
        proc_root=proc_root,
    )

    assert matcher.matches(321)
    assert matcher.matches(999, command_line="elsewhere")
    assert not matcher.matches(777, command_line="train.py")
    assert not matcher.matches(778, command_line="ray::GenericWorker")


@pytest.mark.parametrize(
    "command_line",
    (
        "python gpu_idle_load.py",
        "python3 -u /home/jovyan/gpu_idle_load.py",
        "/opt/conda/bin/python3.11 /opt/topodrome/gpu_idle_load.py --poll 1",
    ),
)
def test_jupyter_idle_worker_pattern_matches_exact_python_script(command_line: str) -> None:
    matcher = CooperativeProcessMatcher(
        command_line_patterns=(
            r"^(?:\S*/)?python(?:3(?:\.\d+)?)?(?:\s+-u)?\s+"
            r"(?:\S*/)?gpu_idle_load\.py(?:\s|$)",
        )
    )

    assert matcher.matches(500, command_line=command_line)


@pytest.mark.parametrize(
    "command_line",
    (
        "python gpu_idle_load.py.backup",
        "python my_gpu_idle_load.py",
        "python gpu_idle_load.pyx",
        "bash gpu_idle_load.py",
        "python -c gpu_idle_load.py",
        "ray::GenericWorker gpu_idle_load.py",
        "python train.py --label gpu_idle_load.py",
    ),
)
def test_jupyter_idle_worker_pattern_rejects_similar_or_unrelated_commands(
    command_line: str,
) -> None:
    matcher = CooperativeProcessMatcher(
        command_line_patterns=(
            r"^(?:\S*/)?python(?:3(?:\.\d+)?)?(?:\s+-u)?\s+"
            r"(?:\S*/)?gpu_idle_load\.py(?:\s|$)",
        )
    )

    assert not matcher.matches(501, command_line=command_line)
