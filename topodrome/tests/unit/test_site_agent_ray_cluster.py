from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from pathlib import Path
from threading import Event

import pytest

import topodrome.site_agent.ray_cluster as ray_cluster
from topodrome.site_agent.ray_cluster import (
    ExistingRayLifecycle,
    RayClusterConfig,
    SubprocessRayLifecycle,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("ray_alias_tmpfs")]


class _FakeHeadProcess:
    def __init__(self, pid: int = 424_242) -> None:
        self.pid = pid
        self.returncode: int | None = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0


class _OwnershipSink:
    def __init__(self) -> None:
        self.registered: list[tuple[int, str | None, int | None]] = []
        self.unregistered: list[int] = []

    def register(
        self,
        pid: int,
        *,
        lease_id: str | None = None,
        actor_id: str | None = None,
        start_ticks: int | None = None,
    ) -> None:
        assert lease_id is None
        self.registered.append((pid, actor_id, start_ticks))

    def unregister(self, pid: int) -> None:
        self.unregistered.append(pid)


def _fake_owned_head(pid: int, *, address: str, temp_dir: Path) -> object:
    return ray_cluster._OwnedRayHead(
        pid=pid,
        process_group_id=pid,
        start_ticks=12345,
        boot_id="test-boot",
        command_sha256="a" * 64,
        address=address,
        temp_dir=temp_dir,
    )


@pytest.mark.asyncio
async def test_long_ray_temp_uses_short_alias_while_data_stays_under_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / ("long-scratch-component-" * 5) / "ray"
    ownership = _OwnershipSink()
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(
            gpu_uuids=("GPU-A",),
            ray_executable="/legacy/bin/ray",
            ray_command_prefix=("/managed/venv/bin/python", "-m", "ray.scripts.scripts"),
            temp_dir=target,
            short_temp_root=Path("/tmp"),  # noqa: S108
        ),
        process_ownership=ownership,
    )
    states = iter((False, True))
    commands: list[tuple[str, ...]] = []
    effective: Path | None = None
    alive = True
    process = _FakeHeadProcess()

    async def is_running() -> bool:
        return next(states)

    async def spawn(
        arguments: tuple[str, ...],
        *,
        log_directory: Path,
    ) -> _FakeHeadProcess:
        nonlocal effective
        commands.append(arguments)
        assert log_directory == target
        raw = next(value for value in arguments if value.startswith("--temp-dir="))
        effective = Path(raw.removeprefix("--temp-dir="))
        assert effective.is_symlink()
        assert effective.resolve() == target.resolve()
        (effective / "state-marker").write_text("under scratch", encoding="utf-8")
        return process

    async def wait_for_start(_process: object) -> None:
        return None

    async def terminate(_owned: object) -> None:
        nonlocal alive
        alive = False
        process.returncode = 0

    async def endpoint_reachable() -> bool:
        return False

    monkeypatch.setattr(lifecycle, "is_running", is_running)
    monkeypatch.setattr(lifecycle, "_spawn_owned_head", spawn)
    monkeypatch.setattr(lifecycle, "_wait_for_owned_start", wait_for_start)
    monkeypatch.setattr(lifecycle, "_terminate_owned_head", terminate)
    monkeypatch.setattr(lifecycle, "_endpoint_reachable", endpoint_reachable)
    monkeypatch.setattr(ray_cluster, "_capture_owned_head", _fake_owned_head)
    monkeypatch.setattr(ray_cluster, "_process_executable_is_python", lambda _pid: True)
    monkeypatch.setattr(ray_cluster, "_RAY_IDENTITY_STABILITY_SECONDS", 0.0)
    monkeypatch.setattr(ray_cluster, "_RAY_IDENTITY_POLL_SECONDS", 0.0)
    monkeypatch.setattr(ray_cluster, "_owned_head_matches", lambda _owned: alive)
    monkeypatch.setattr(ray_cluster, "_process_group_exists", lambda _pgid: alive)

    assert await lifecycle.ensure_running()
    assert effective is not None
    assert len(os.fsencode(effective)) <= 32
    assert target.joinpath("state-marker").read_text(encoding="utf-8") == "under scratch"
    assert commands[0][:6] == (
        "/managed/venv/bin/python",
        "-m",
        "ray.scripts.scripts",
        "start",
        "--head",
        "--block",
    )
    assert ownership.registered == [(process.pid, "managed-ray-head", 12345)]

    await lifecycle.stop()

    assert len(commands) == 1
    assert ownership.unregistered == [process.pid]
    assert not await asyncio.to_thread(os.path.lexists, effective)
    assert target.is_dir()


def test_default_ray_command_prefix_preserves_custom_executable() -> None:
    config = RayClusterConfig(
        gpu_uuids=("GPU-A",),
        ray_executable="/custom/bin/ray",
    )

    assert config.command_prefix == ("/custom/bin/ray",)


@pytest.mark.asyncio
async def test_explicit_ray_command_prefix_is_used_for_status_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(
            gpu_uuids=("GPU-A",),
            ray_executable="/legacy/bin/ray",
            ray_command_prefix=(
                "/managed/venv/bin/python",
                "-m",
                "ray.scripts.scripts",
            ),
        )
    )
    commands: list[tuple[str, ...]] = []

    async def endpoint_reachable() -> bool:
        return True

    async def run(*arguments: str, check: bool) -> int:
        assert not check
        commands.append(arguments)
        return 0

    monkeypatch.setattr(lifecycle, "_endpoint_reachable", endpoint_reachable)
    monkeypatch.setattr(lifecycle, "_run", run)

    assert await lifecycle.is_running()
    assert commands == [
        (
            "/managed/venv/bin/python",
            "-m",
            "ray.scripts.scripts",
            "status",
            "--address",
            "127.0.0.1:6379",
        )
    ]


@pytest.mark.parametrize(
    "prefix",
    [(), ("",), ("python", "-m", "bad\0module")],
)
def test_explicit_ray_command_prefix_rejects_invalid_arguments(
    prefix: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="command prefix"):
        RayClusterConfig(gpu_uuids=("GPU-A",), ray_command_prefix=prefix)


@pytest.mark.asyncio
async def test_failed_ray_start_removes_only_its_exact_short_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / ("another-long-component-" * 5) / "ray"
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(gpu_uuids=("GPU-A",), temp_dir=target)
    )
    captured_alias: Path | None = None
    alive = True
    process = _FakeHeadProcess(pid=424_243)

    async def is_running() -> bool:
        return False

    async def spawn(
        arguments: tuple[str, ...],
        *,
        log_directory: Path,
    ) -> _FakeHeadProcess:
        nonlocal captured_alias
        assert log_directory == target
        raw = next(value for value in arguments if value.startswith("--temp-dir="))
        captured_alias = Path(raw.removeprefix("--temp-dir="))
        return process

    async def wait_for_start(_process: object) -> None:
        raise RuntimeError("injected Ray start failure")

    async def terminate(_owned: object) -> None:
        nonlocal alive
        alive = False
        process.returncode = 0

    monkeypatch.setattr(lifecycle, "is_running", is_running)
    monkeypatch.setattr(lifecycle, "_spawn_owned_head", spawn)
    monkeypatch.setattr(lifecycle, "_wait_for_owned_start", wait_for_start)
    monkeypatch.setattr(lifecycle, "_terminate_owned_head", terminate)
    monkeypatch.setattr(ray_cluster, "_capture_owned_head", _fake_owned_head)
    monkeypatch.setattr(ray_cluster, "_process_executable_is_python", lambda _pid: True)
    monkeypatch.setattr(ray_cluster, "_RAY_IDENTITY_STABILITY_SECONDS", 0.0)
    monkeypatch.setattr(ray_cluster, "_RAY_IDENTITY_POLL_SECONDS", 0.0)
    monkeypatch.setattr(ray_cluster, "_owned_head_matches", lambda _owned: alive)

    with pytest.raises(RuntimeError, match="injected Ray start failure"):
        await lifecycle.ensure_running()

    assert captured_alias is not None
    assert not await asyncio.to_thread(os.path.lexists, captured_alias)
    assert target.is_dir()


def test_ray_temp_rejects_preexisting_alias_to_another_target(tmp_path: Path) -> None:
    target = tmp_path / ("long-component-" * 8) / "ray"
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(gpu_uuids=("GPU-A",), temp_dir=target)
    )
    alias = lifecycle._prepare_temp_dir()
    assert alias is not None and alias.is_symlink()
    lifecycle._cleanup_temp_alias()
    alias.parent.mkdir(mode=0o700, exist_ok=True)
    other = tmp_path / "other"
    other.mkdir()
    alias.symlink_to(other, target_is_directory=True)
    try:
        with pytest.raises(RuntimeError, match="different directory"):
            lifecycle._prepare_temp_dir()
        assert alias.resolve() == other.resolve()
    finally:
        alias.unlink()
        # Other Topodrome lifecycles for this user may legitimately have an
        # alias in the same private directory.
        with suppress(OSError):
            alias.parent.rmdir()


@pytest.mark.parametrize("unsafe_kind", ["target-symlink", "untrusted-short-root"])
def test_ray_temp_rejects_unsafe_symlink_or_shared_directory(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    target = tmp_path / ("long-component-" * 8) / "ray"
    short_root = Path("/tmp")  # noqa: S108
    if unsafe_kind == "target-symlink":
        target.parent.mkdir(mode=0o700, parents=True)
        actual = tmp_path / "actual-ray"
        actual.mkdir()
        target.symlink_to(actual, target_is_directory=True)
    else:
        short_root = tmp_path / "shared-aliases"
        short_root.mkdir(mode=0o700)
        short_root.chmod(0o777)
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(
            gpu_uuids=("GPU-A",),
            temp_dir=target,
            short_temp_root=short_root,
        )
    )

    expected = "symbolic link" if unsafe_kind == "target-symlink" else "without sticky bit"
    with pytest.raises(RuntimeError, match=expected):
        lifecycle._prepare_temp_dir()


def test_ray_temp_rejects_directory_owned_by_another_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "ray"
    target.mkdir()
    actual_uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: actual_uid + 1)
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(gpu_uuids=("GPU-A",), temp_dir=target)
    )

    with pytest.raises(RuntimeError, match="not owned"):
        lifecycle._prepare_temp_dir()


def test_long_ray_temp_fails_closed_when_short_aliases_are_disabled(
    tmp_path: Path,
) -> None:
    target = tmp_path / ("long-component-" * 8) / "ray"
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(
            gpu_uuids=("GPU-A",),
            temp_dir=target,
            short_temp_root=None,
        )
    )

    with pytest.raises(RuntimeError, match="no ray_short_temp_root"):
        lifecycle._prepare_temp_dir()


@pytest.mark.asyncio
async def test_adopted_ray_head_is_never_stopped_or_signaled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(gpu_uuids=("GPU-A",), temp_dir=tmp_path / "ray")
    )
    signals: list[tuple[int, int]] = []

    async def is_running() -> bool:
        return True

    monkeypatch.setattr(lifecycle, "is_running", is_running)
    monkeypatch.setattr(
        os,
        "killpg",
        lambda process_group, signum: signals.append((process_group, signum)),
    )

    assert not await lifecycle.ensure_running()
    assert not await lifecycle.stop()
    assert signals == []


@pytest.mark.asyncio
async def test_restarted_lifecycle_stops_only_exact_persisted_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "ray"
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(gpu_uuids=("GPU-A",), temp_dir=target)
    )
    lifecycle._prepare_temp_dir()
    owned = ray_cluster._OwnedRayHead(
        pid=424_244,
        process_group_id=424_244,
        start_ticks=9876,
        boot_id="test-boot",
        command_sha256="b" * 64,
        address=lifecycle.config.address,
        temp_dir=target,
    )
    lifecycle._write_owned_head(owned)
    terminated: list[object] = []

    async def is_running() -> bool:
        return True

    async def terminate(value: object) -> None:
        terminated.append(value)

    async def endpoint_reachable() -> bool:
        return False

    monkeypatch.setattr(lifecycle, "is_running", is_running)
    monkeypatch.setattr(lifecycle, "_terminate_owned_head", terminate)
    monkeypatch.setattr(lifecycle, "_endpoint_reachable", endpoint_reachable)
    monkeypatch.setattr(ray_cluster, "_owned_head_matches", lambda value: value == owned)

    assert await lifecycle.stop()
    assert terminated == [owned]
    assert not lifecycle._ownership_path().exists()


@pytest.mark.asyncio
async def test_recycled_pid_identity_fails_closed_without_any_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "ray"
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(gpu_uuids=("GPU-A",), temp_dir=target)
    )
    lifecycle._prepare_temp_dir()
    owned = ray_cluster._OwnedRayHead(
        pid=424_245,
        process_group_id=424_245,
        start_ticks=111,
        boot_id="old-boot",
        command_sha256="c" * 64,
        address=lifecycle.config.address,
        temp_dir=target,
    )
    lifecycle._write_owned_head(owned)
    signals: list[tuple[int, int]] = []

    async def is_running() -> bool:
        return True

    monkeypatch.setattr(lifecycle, "is_running", is_running)
    monkeypatch.setattr(ray_cluster, "_owned_head_matches", lambda _value: False)
    monkeypatch.setattr(ray_cluster, "_process_group_exists", lambda _pgid: True)
    monkeypatch.setattr(
        os,
        "killpg",
        lambda process_group, signum: signals.append((process_group, signum)),
    )

    assert not await lifecycle.stop()
    assert signals == []
    assert lifecycle._ownership_path().exists()


@pytest.mark.asyncio
async def test_owned_head_identity_waits_for_wrapper_exec_and_stable_python_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "ray"
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(
            gpu_uuids=("GPU-A",),
            temp_dir=target,
            command_timeout_seconds=1.0,
        )
    )
    lifecycle._prepare_temp_dir()
    process = _FakeHeadProcess(pid=424_246)
    spawned = ray_cluster._OwnedRayHead(
        pid=process.pid,
        process_group_id=process.pid,
        start_ticks=2468,
        boot_id="test-boot",
        command_sha256="a" * 64,
        address=lifecycle.config.address,
        temp_dir=target,
    )
    final = ray_cluster._OwnedRayHead(
        pid=process.pid,
        process_group_id=process.pid,
        start_ticks=spawned.start_ticks,
        boot_id=spawned.boot_id,
        command_sha256="b" * 64,
        address=spawned.address,
        temp_dir=spawned.temp_dir,
    )
    captures = iter((spawned, final, final))
    python_states = iter((False, True, True))
    monkeypatch.setattr(
        ray_cluster,
        "_capture_owned_head",
        lambda *_args, **_kwargs: next(captures),
    )
    monkeypatch.setattr(
        ray_cluster,
        "_process_executable_is_python",
        lambda _pid: next(python_states),
    )
    monkeypatch.setattr(ray_cluster, "_RAY_IDENTITY_STABILITY_SECONDS", 0.0)
    monkeypatch.setattr(ray_cluster, "_RAY_IDENTITY_POLL_SECONDS", 0.0)

    captured = await lifecycle._wait_for_stable_owned_head(process, spawned)

    assert captured == final


@pytest.mark.asyncio
async def test_owned_head_identity_fails_closed_when_process_exits(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ray"
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(gpu_uuids=("GPU-A",), temp_dir=target)
    )
    lifecycle._prepare_temp_dir()
    process = _FakeHeadProcess(pid=424_247)
    process.returncode = 7
    spawned = ray_cluster._OwnedRayHead(
        pid=process.pid,
        process_group_id=process.pid,
        start_ticks=1357,
        boot_id="test-boot",
        command_sha256="c" * 64,
        address=lifecycle.config.address,
        temp_dir=target,
    )

    with pytest.raises(RuntimeError, match=r"exited before.*stabilized"):
        await lifecycle._wait_for_stable_owned_head(process, spawned)


@pytest.mark.asyncio
async def test_owned_head_identity_times_out_before_publishing_shell_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "ray"
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(
            gpu_uuids=("GPU-A",),
            temp_dir=target,
            command_timeout_seconds=0.001,
        )
    )
    lifecycle._prepare_temp_dir()
    process = _FakeHeadProcess(pid=424_248)
    spawned = ray_cluster._OwnedRayHead(
        pid=process.pid,
        process_group_id=process.pid,
        start_ticks=8642,
        boot_id="test-boot",
        command_sha256="d" * 64,
        address=lifecycle.config.address,
        temp_dir=target,
    )
    monkeypatch.setattr(ray_cluster, "_capture_owned_head", lambda *_args, **_kwargs: spawned)
    monkeypatch.setattr(ray_cluster, "_process_executable_is_python", lambda _pid: False)

    with pytest.raises(
        ray_cluster._RayCommandTimeoutError,
        match="ownership identity did not stabilize",
    ):
        await lifecycle._wait_for_stable_owned_head(process, spawned)


@pytest.mark.asyncio
async def test_existing_ray_driver_connect_and_shutdown_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect_gate = Event()
    shutdown_gate = Event()

    class Ray:
        def is_initialized(self) -> bool:
            return False

        def init(self, **_options: object) -> None:
            connect_gate.wait()

        def shutdown(self) -> None:
            shutdown_gate.wait()

    ray = Ray()
    lifecycle = ExistingRayLifecycle(control_timeout_seconds=0.01)
    monkeypatch.setattr(lifecycle, "_ray", lambda: ray)
    try:
        assert not await lifecycle.is_running()
        lifecycle._connected = True
        with pytest.raises(TimeoutError):
            await lifecycle.stop()
        # An ambiguous driver shutdown must not be reported as complete.
        assert lifecycle._connected
    finally:
        connect_gate.set()
        shutdown_gate.set()
        await asyncio.sleep(0.01)


@pytest.mark.parametrize("timeout", [0.0, float("inf"), float("nan")])
def test_existing_ray_rejects_unbounded_control_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ExistingRayLifecycle(control_timeout_seconds=timeout)
