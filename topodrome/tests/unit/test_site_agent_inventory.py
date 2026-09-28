from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from topodrome.site_agent.inventory import (
    GPUReservation,
    GPUReservationLockReader,
    NVMLInventoryProvider,
    ProcessOwnershipRegistry,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/usr/bin/python3.11 -u /private/project/train.py --token SECRET", "train.py"),
        ("python -m experiments.train --password SECRET", "experiments.train"),
        ("python -c 'print(SECRET)'", None),
        ("python --token SECRET run.py", None),
        ("bash -c SECRET", None),
        ("python 'unterminated", None),
    ],
)
def test_process_script_label_never_includes_arguments(command, expected):
    from topodrome.site_agent.inventory import _script_name

    assert _script_name(command) == expected


def test_process_username_is_best_effort(tmp_path, monkeypatch):
    from topodrome.site_agent import inventory

    assert inventory._process_username(123, tmp_path) is None
    (tmp_path / "123").mkdir()
    monkeypatch.setattr(inventory.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="alice"))
    assert inventory._process_username(123, tmp_path) == "alice"


def _write_process(
    proc_root: Path,
    pid: int,
    *,
    state: str = "S",
    start_ticks: int = 12345,
    started_at: float = 100,
) -> None:
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    fields_before_start = " ".join("0" for _ in range(17))
    (process / "stat").write_text(
        f"{pid} (reservation supervisor) {state} 1 {fields_before_start} {start_ticks}\n",
        encoding="ascii",
    )
    os.utime(process, (started_at, started_at))


def _write_owned_process(
    proc_root: Path,
    pid: int,
    *,
    parent_pid: int,
    start_ticks: int,
) -> None:
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    fields_before_start = " ".join("0" for _ in range(17))
    (process / "stat").write_text(
        f"{pid} (ray process) S {parent_pid} {fields_before_start} {start_ticks}\n",
        encoding="ascii",
    )
    (process / "status").write_text(f"Name:\tray process\nPPid:\t{parent_pid}\n")


def test_verified_managed_ray_root_owns_loader_descendants(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_owned_process(proc_root, 100, parent_pid=1, start_ticks=1000)
    _write_owned_process(proc_root, 101, parent_pid=100, start_ticks=1001)
    _write_owned_process(proc_root, 102, parent_pid=101, start_ticks=1002)
    registry = ProcessOwnershipRegistry(proc_root=proc_root)

    registry.register(
        100,
        actor_id="managed-ray-head",
        start_ticks=1000,
    )

    assert registry.owns(102)
    # Reusing the recorded root PID invalidates ownership for every descendant.
    (proc_root / "100" / "stat").write_text(
        f"100 (replacement) S 1 {' '.join('0' for _ in range(17))} 9999\n",
        encoding="ascii",
    )
    assert not registry.owns(102)


def _lock_root(tmp_path: Path) -> Path:
    root = tmp_path / "gpu_runtime_locks"
    root.mkdir(mode=0o700, parents=True)
    return root


def _write_lock(
    root: Path,
    *,
    gpu_index: int = 0,
    physical_gpu: int = 0,
    pid: int = 4321,
    acquired_at: object = 150,
    **updates: object,
) -> Path:
    payload: dict[str, object] = {
        "owner": "foreign-job",
        "physical_gpu": physical_gpu,
        "supervisor_pid": pid,
        "world_size": 1,
        "acquired_at": acquired_at,
        **updates,
    }
    path = root / f"gpu_{gpu_index}.lock"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return path


def _reader(root: Path, proc_root: Path) -> GPUReservationLockReader:
    return GPUReservationLockReader(
        root,
        proc_root=proc_root,
        clock=lambda: 200,
    )


def test_live_exact_supervisor_reservation_is_honored(tmp_path: Path) -> None:
    root = _lock_root(tmp_path)
    proc_root = tmp_path / "proc"
    _write_process(proc_root, 4321)
    _write_lock(root, supervisor_start_ticks=12345)

    reservation = _reader(root, proc_root).reservation_for(0)

    assert reservation == GPUReservation(
        physical_gpu=0,
        supervisor_pid=4321,
        owner="foreign-job",
        world_size=1,
        acquired_at=150,
        supervisor_start_ticks=12345,
    )


def test_iso_timestamp_alias_is_supported(tmp_path: Path) -> None:
    root = _lock_root(tmp_path)
    proc_root = tmp_path / "proc"
    _write_process(proc_root, 4321)
    acquired = datetime.fromtimestamp(150, tz=UTC).isoformat()
    path = _write_lock(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("acquired_at")
    payload["created_at"] = acquired
    path.write_text(json.dumps(payload), encoding="utf-8")

    reservation = _reader(root, proc_root).reservation_for(0)

    assert reservation is not None
    assert reservation.acquired_at == 150


@pytest.mark.parametrize(
    "content",
    [
        b"not-json",
        b"[]",
        b'{"owner":"x","owner":"y","physical_gpu":0,"supervisor_pid":4321,'
        b'"world_size":1,"acquired_at":150}',
        json.dumps(
            {
                "physical_gpu": 0,
                "supervisor_pid": 4321,
                "world_size": 1,
                "acquired_at": 150,
            }
        ).encode(),
        json.dumps(
            {
                "owner": "x",
                "physical_gpu": 1,
                "supervisor_pid": 4321,
                "world_size": 1,
                "acquired_at": 150,
            }
        ).encode(),
        json.dumps(
            {
                "owner": "x",
                "physical_gpu": 0,
                "supervisor_pid": True,
                "world_size": 1,
                "acquired_at": 150,
            }
        ).encode(),
        json.dumps(
            {
                "owner": "x",
                "physical_gpu": 0,
                "supervisor_pid": 4321,
                "world_size": 0,
                "acquired_at": 150,
            }
        ).encode(),
        json.dumps(
            {
                "owner": "x",
                "physical_gpu": 0,
                "supervisor_pid": 4321,
                "world_size": 1,
                "acquired_at": "1970-01-01T00:02:30",
            }
        ).encode(),
        json.dumps(
            {
                "owner": "x",
                "physical_gpu": 0,
                "supervisor_pid": 4321,
                "world_size": 1,
                "acquired_at": 501,
            }
        ).encode(),
    ],
)
def test_malformed_or_mismatched_lock_is_ignored(
    tmp_path: Path,
    content: bytes,
) -> None:
    root = _lock_root(tmp_path)
    proc_root = tmp_path / "proc"
    _write_process(proc_root, 4321)
    path = root / "gpu_0.lock"
    path.write_bytes(content)
    path.chmod(0o600)

    assert _reader(root, proc_root).reservation_for(0) is None


def test_dead_zombie_and_recycled_supervisor_pids_are_ignored(tmp_path: Path) -> None:
    root = _lock_root(tmp_path)
    proc_root = tmp_path / "proc"
    _write_lock(root, pid=4001)
    reader = _reader(root, proc_root)
    assert reader.reservation_for(0) is None

    _write_process(proc_root, 4001, state="Z")
    assert reader.reservation_for(0) is None

    (proc_root / "4001" / "stat").unlink()
    (proc_root / "4001").rmdir()
    _write_process(proc_root, 4001, state="S", started_at=180)
    assert reader.reservation_for(0) is None


def test_recorded_start_ticks_reject_pid_reuse(tmp_path: Path) -> None:
    root = _lock_root(tmp_path)
    proc_root = tmp_path / "proc"
    _write_process(proc_root, 4321, start_ticks=999)
    _write_lock(root, supervisor_start_ticks=123)

    assert _reader(root, proc_root).reservation_for(0) is None


def test_symlink_hardlink_and_writable_entries_are_never_trusted(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_process(proc_root, 4321)

    symlink_root = _lock_root(tmp_path / "symlink-case")
    target = _write_lock(symlink_root, gpu_index=1)
    (symlink_root / "gpu_0.lock").symlink_to(target.name)
    assert _reader(symlink_root, proc_root).reservation_for(0) is None

    hardlink_root = _lock_root(tmp_path / "hardlink-case")
    target = _write_lock(hardlink_root, gpu_index=1)
    os.link(target, hardlink_root / "gpu_0.lock")
    assert _reader(hardlink_root, proc_root).reservation_for(0) is None

    writable_root = _lock_root(tmp_path / "writable-case")
    path = _write_lock(writable_root)
    path.chmod(0o666)
    assert _reader(writable_root, proc_root).reservation_for(0) is None


def test_symlink_or_writable_lock_directory_is_never_trusted(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_process(proc_root, 4321)
    real_root = _lock_root(tmp_path / "real")
    _write_lock(real_root)

    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    assert _reader(linked_root, proc_root).reservation_for(0) is None

    real_root.chmod(0o777)
    assert _reader(real_root, proc_root).reservation_for(0) is None


def test_oversized_lock_is_ignored_without_parsing(tmp_path: Path) -> None:
    root = _lock_root(tmp_path)
    proc_root = tmp_path / "proc"
    _write_process(proc_root, 4321)
    path = root / "gpu_0.lock"
    path.write_bytes(b" " * (16 * 1024 + 1))
    path.chmod(0o600)

    assert _reader(root, proc_root).reservation_for(0) is None


def test_nvml_inventory_attaches_validated_reservation() -> None:
    reservation = GPUReservation(0, 4321, "foreign-job", 2, 150, 12345)

    class Detector:
        def reservation_for(self, gpu_index: int) -> GPUReservation | None:
            assert gpu_index == 0
            return reservation

    class NVML:
        NVML_VALUE_NOT_AVAILABLE = -1

        @staticmethod
        def nvmlInit() -> None:
            return None

        @staticmethod
        def nvmlDeviceGetCount() -> int:
            return 1

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index: int) -> int:
            return index

        @staticmethod
        def nvmlDeviceGetMemoryInfo(_handle: int) -> SimpleNamespace:
            return SimpleNamespace(total=80, free=70, used=10)

        @staticmethod
        def nvmlDeviceGetUtilizationRates(_handle: int) -> SimpleNamespace:
            return SimpleNamespace(gpu=5)

        @staticmethod
        def nvmlDeviceGetUUID(_handle: int) -> str:
            return "GPU-A"

        @staticmethod
        def nvmlDeviceGetName(_handle: int) -> str:
            return "A800"

        @staticmethod
        def nvmlDeviceGetComputeRunningProcesses(_handle: int) -> list[object]:
            return []

        @staticmethod
        def nvmlDeviceGetGraphicsRunningProcesses(_handle: int) -> list[object]:
            return []

    provider = NVMLInventoryProvider(
        ProcessOwnershipRegistry(),
        reservation_detector=Detector(),
    )
    provider._nvml = NVML()

    snapshot = provider.sample()[0]

    assert snapshot.external_reservation is reservation
    assert snapshot.externally_reserved
