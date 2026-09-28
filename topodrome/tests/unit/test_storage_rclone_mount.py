from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from topodrome.storage import RcloneMountError, RcloneMountSupervisor

pytestmark = pytest.mark.unit


class _Reader:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy

    def healthcheck(self) -> bool:
        return self.healthy


class _Process:
    def __init__(self, mounted: dict[str, bool]) -> None:
        self._mounted = mounted
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0
        self._mounted["value"] = False

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._mounted["value"] = False

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            raise subprocess.TimeoutExpired("rclone", 1)
        return self.returncode


def _binary(tmp_path: Path) -> Path:
    binary = tmp_path / "rclone"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o700)
    return binary


def test_managed_mount_uses_full_vfs_cache_and_stops_only_owned_process(
    tmp_path: Path,
) -> None:
    mounted = {"value": False}
    calls: list[tuple[list[str], dict[str, Any]]] = []
    process = _Process(mounted)

    def popen(arguments: list[str], **kwargs: Any) -> Any:
        calls.append((arguments, kwargs))
        mounted["value"] = True
        return process

    supervisor = RcloneMountSupervisor(
        remote="lab1:topodrome",
        mountpoint=tmp_path / "mount",
        cache_dir=tmp_path / "cache",
        health_reader=_Reader(),
        binary=_binary(tmp_path),
        min_free_space_gb=0.000001,
        is_mount=lambda _path: mounted["value"],
        popen=popen,
    )

    supervisor.start()

    assert supervisor.managed
    arguments, options = calls[0]
    assert arguments[:3] == [str(_binary(tmp_path).resolve()), "mount", "lab1:topodrome"]
    assert arguments[arguments.index("--vfs-cache-mode") + 1] == "full"
    assert arguments[arguments.index("--vfs-cache-max-size") + 1] == "20G"
    assert arguments[arguments.index("--vfs-cache-min-free-space") + 1] == "1e-06G"
    assert arguments[arguments.index("--vfs-write-back") + 1] == "5s"
    assert options["start_new_session"] is True
    assert "TOPODROME_SITE_ENROLLMENT_TOKEN" not in options["env"]
    assert "TOPODROME_HF_TOKEN" not in options["env"]

    supervisor.stop()

    assert process.terminated
    assert not process.killed
    assert not mounted["value"]


def test_healthy_existing_mount_is_never_stopped_by_supervisor(tmp_path: Path) -> None:
    popen_calls: list[object] = []
    unmount_calls: list[object] = []
    supervisor = RcloneMountSupervisor(
        remote="lab1:topodrome",
        mountpoint=tmp_path / "mount",
        cache_dir=tmp_path / "cache",
        health_reader=_Reader(),
        binary=_binary(tmp_path),
        min_free_space_gb=0.000001,
        is_mount=lambda _path: True,
        popen=lambda *args, **kwargs: popen_calls.append((args, kwargs)),
        run=lambda *args, **kwargs: unmount_calls.append((args, kwargs)),
    )

    supervisor.start()
    supervisor.stop()

    assert not supervisor.managed
    assert popen_calls == []
    assert unmount_calls == []


def test_refuses_unhealthy_foreign_mount(tmp_path: Path) -> None:
    supervisor = RcloneMountSupervisor(
        remote="lab1:topodrome",
        mountpoint=tmp_path / "mount",
        cache_dir=tmp_path / "cache",
        health_reader=_Reader(healthy=False),
        binary=_binary(tmp_path),
        min_free_space_gb=0.000001,
        is_mount=lambda _path: True,
    )

    with pytest.raises(RcloneMountError, match="existing rclone mount"):
        supervisor.start()


def test_rejects_cache_nested_below_mount(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be inside"):
        RcloneMountSupervisor(
            remote="lab1:topodrome",
            mountpoint=tmp_path / "mount",
            cache_dir=tmp_path / "mount" / "cache",
            health_reader=_Reader(),
            binary=_binary(tmp_path),
        )


def test_binary_must_resolve_without_shell_lookup(tmp_path: Path) -> None:
    missing = tmp_path / "missing-rclone"
    assert not missing.exists()
    with pytest.raises(FileNotFoundError):
        RcloneMountSupervisor(
            remote="lab1:topodrome",
            mountpoint=tmp_path / "mount",
            cache_dir=tmp_path / "cache",
            health_reader=_Reader(),
            binary=missing,
        )


def test_test_binary_is_executable(tmp_path: Path) -> None:
    assert os.access(_binary(tmp_path), os.X_OK)


def test_mount_process_receives_only_rclone_scoped_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mounted = {"value": False}
    captured: dict[str, str] = {}
    process = _Process(mounted)

    def popen(_arguments: list[str], **kwargs: Any) -> Any:
        captured.update(kwargs["env"])
        mounted["value"] = True
        return process

    monkeypatch.setenv("RCLONE_CONFIG", str(tmp_path / "rclone.conf"))
    monkeypatch.setenv("RCLONE_CONFIG_LAB1_PASS", "rclone-only-canary")
    monkeypatch.setenv("TOPODROME_SITE_ENROLLMENT_TOKEN", "enrollment-canary")
    monkeypatch.setenv("TOPODROME_SITE_FENCE_VERIFICATION_KEY", "fence-canary")
    monkeypatch.setenv("TOPODROME_HF_TOKEN", "hub-canary")
    monkeypatch.setenv("TOPODROME_AUTODL_TOKEN", "provider-canary")
    supervisor = RcloneMountSupervisor(
        remote="lab1:topodrome",
        mountpoint=tmp_path / "mount",
        cache_dir=tmp_path / "cache",
        health_reader=_Reader(),
        binary=_binary(tmp_path),
        min_free_space_gb=0.000001,
        is_mount=lambda _path: mounted["value"],
        popen=popen,
    )

    supervisor.start()
    supervisor.stop()

    assert captured["RCLONE_CONFIG_LAB1_PASS"] == "rclone-only-canary"  # noqa: S105
    assert not any(
        value.endswith("canary") and value != "rclone-only-canary"
        for value in captured.values()
    )
