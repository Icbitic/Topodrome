from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from topodrome.deployment import jupyter_agent
from topodrome.deployment.jupyter_agent import (
    JupyterAgentBootstrapError,
    OwnershipRecord,
    ensure_jupyter_agent,
    stop_owned_jupyter_agent,
)


def _protected_file(parent: Path, name: str, content: str) -> Path:
    parent.mkdir(mode=0o700)
    path = parent / name
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return path


def _configuration_digest(config: Path, environment: Path | None = None) -> str:
    environment_bytes = b"" if environment is None else environment.read_bytes()
    return hashlib.sha256(config.read_bytes() + b"\0" + environment_bytes).hexdigest()


def _capture_started_process_identity(
    process: subprocess.Popen[bytes],
) -> jupyter_agent.ProcessIdentity:
    deadline = time.monotonic() + 2
    while True:
        try:
            return jupyter_agent._capture_process_identity(process.pid)
        except ValueError:
            if process.poll() is not None or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def test_site_environment_accepts_only_agent_runtime_keys() -> None:
    values = jupyter_agent._parse_site_environment(
        b"TOPODROME_SITE_ENROLLMENT_TOKEN='site secret'\n"
        b"HF_TOKEN=hf-secret\n"
    )
    assert values == {
        "TOPODROME_SITE_ENROLLMENT_TOKEN": "site secret",
        "HF_TOKEN": "hf-secret",
    }

    with pytest.raises(ValueError, match="unsupported key"):
        jupyter_agent._parse_site_environment(b"TOPODROME_AUTODL_TOKEN=provider-secret\n")


def test_agent_environment_does_not_inherit_scheduler_or_jupyter_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOPODROME_AUTODL_TOKEN", "provider-secret")
    monkeypatch.setenv("TOPODROME_TEST_JUPYTER_TOKEN", "terminal-secret")
    monkeypatch.setenv("JUPYTER_TOKEN", "server-secret")
    monkeypatch.setenv("TOPODROME_SITE_ENROLLMENT_TOKEN", "site-secret")

    environment = jupyter_agent._agent_environment({})

    assert environment["TOPODROME_SITE_ENROLLMENT_TOKEN"] == "site-secret"  # noqa: S105
    assert "TOPODROME_AUTODL_TOKEN" not in environment
    assert "TOPODROME_TEST_JUPYTER_TOKEN" not in environment
    assert "JUPYTER_TOKEN" not in environment


def test_protected_files_reject_group_read_and_symlinks(tmp_path: Path) -> None:
    parent = tmp_path / "protected"
    config = _protected_file(parent, "site.yaml", "site_id: remote\n")
    config.chmod(0o640)
    with pytest.raises(PermissionError, match="mode 0600"):
        jupyter_agent._read_protected_file(
            config,
            description="site configuration",
        )

    config.chmod(0o600)
    link = parent / "site-link.yaml"
    link.symlink_to(config)
    with pytest.raises(PermissionError, match="unavailable"):
        jupyter_agent._read_protected_file(
            link,
            description="site configuration",
        )


def test_existing_exact_live_generation_is_reused_without_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _protected_file(tmp_path / "config", "site.yaml", "site_id: remote\n")
    state_directory = tmp_path / "state"
    state_directory.mkdir(mode=0o700)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        identity = _capture_started_process_identity(process)
        record = OwnershipRecord(
            generation="a" * 32,
            phase="ready",
            configuration_sha256=_configuration_digest(config),
            created_at=datetime.now(UTC),
            identity=identity,
            ready_at=datetime.now(UTC),
        )
        jupyter_agent._write_record(state_directory / "agent.json", record)

        def fail_spawn(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            raise AssertionError("an exact live generation must not be spawned again")

        monkeypatch.setattr(jupyter_agent, "_spawn_child", fail_spawn)
        reused = ensure_jupyter_agent(
            config_path=config,
            site_environment_path=None,
            state_directory=state_directory,
            startup_timeout_seconds=1,
            stability_seconds=0,
        )

        assert reused == record
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_stale_or_forged_identity_is_never_signalled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _protected_file(tmp_path / "config", "site.yaml", "site_id: remote\n")
    state_directory = tmp_path / "state"
    state_directory.mkdir(mode=0o700)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        identity = _capture_started_process_identity(process)
        forged = jupyter_agent.ProcessIdentity(
            pid=identity.pid,
            process_group_id=identity.process_group_id,
            start_ticks=identity.start_ticks + 1,
            boot_id=identity.boot_id,
            command_sha256=identity.command_sha256,
        )
        record = OwnershipRecord(
            generation="b" * 32,
            phase="ready",
            configuration_sha256=_configuration_digest(config),
            created_at=datetime.now(UTC),
            identity=forged,
            ready_at=datetime.now(UTC),
        )
        jupyter_agent._write_record(state_directory / "agent.json", record)

        def stop_before_spawn(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            raise RuntimeError("stop after stale-state cleanup")

        monkeypatch.setattr(jupyter_agent, "_spawn_child", stop_before_spawn)
        with pytest.raises(RuntimeError, match="stop after stale-state cleanup"):
            ensure_jupyter_agent(
                config_path=config,
                site_environment_path=None,
                state_directory=state_directory,
                startup_timeout_seconds=1,
                stability_seconds=0,
            )

        assert process.poll() is None
        assert not (state_directory / "agent.json").exists()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_live_generation_with_changed_protected_config_fails_closed(
    tmp_path: Path,
) -> None:
    config = _protected_file(tmp_path / "config", "site.yaml", "site_id: remote\n")
    state_directory = tmp_path / "state"
    state_directory.mkdir(mode=0o700)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        record = OwnershipRecord(
            generation="c" * 32,
            phase="ready",
            configuration_sha256="d" * 64,
            created_at=datetime.now(UTC),
            identity=_capture_started_process_identity(process),
            ready_at=datetime.now(UTC),
        )
        jupyter_agent._write_record(state_directory / "agent.json", record)

        with pytest.raises(JupyterAgentBootstrapError, match="different protected"):
            ensure_jupyter_agent(
                config_path=config,
                site_environment_path=None,
                state_directory=state_directory,
                startup_timeout_seconds=1,
                stability_seconds=0,
            )
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_stop_owned_generation_signals_only_its_verified_pidfd(tmp_path: Path) -> None:
    config = _protected_file(tmp_path / "config", "site.yaml", "site_id: remote\n")
    state_directory = tmp_path / "state"
    state_directory.mkdir(mode=0o700)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        identity = _capture_started_process_identity(process)
        record = OwnershipRecord(
            generation="e" * 32,
            phase="ready",
            configuration_sha256=_configuration_digest(config),
            created_at=datetime.now(UTC),
            identity=identity,
            ready_at=datetime.now(UTC),
        )
        jupyter_agent._write_record(state_directory / "agent.json", record)

        assert stop_owned_jupyter_agent(
            config_path=config,
            site_environment_path=None,
            state_directory=state_directory,
            shutdown_timeout_seconds=5,
        )
        assert process.wait(timeout=5) == -signal.SIGTERM
        assert not (state_directory / "agent.json").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_stop_owned_stale_identity_never_signals_reused_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _protected_file(tmp_path / "config", "site.yaml", "site_id: remote\n")
    state_directory = tmp_path / "state"
    state_directory.mkdir(mode=0o700)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        identity = _capture_started_process_identity(process)
        forged = jupyter_agent.ProcessIdentity(
            pid=identity.pid,
            process_group_id=identity.process_group_id,
            start_ticks=identity.start_ticks + 1,
            boot_id=identity.boot_id,
            command_sha256=identity.command_sha256,
        )
        record = OwnershipRecord(
            generation="f" * 32,
            phase="ready",
            configuration_sha256=_configuration_digest(config),
            created_at=datetime.now(UTC),
            identity=forged,
            ready_at=datetime.now(UTC),
        )
        jupyter_agent._write_record(state_directory / "agent.json", record)
        monkeypatch.setattr(
            jupyter_agent,
            "_send_pidfd_signal",
            lambda *_args: pytest.fail("a stale identity must never be signalled"),
        )

        assert not stop_owned_jupyter_agent(
            config_path=config,
            site_environment_path=None,
            state_directory=state_directory,
            shutdown_timeout_seconds=1,
        )
        assert process.poll() is None
        assert not (state_directory / "agent.json").exists()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_stop_owned_live_generation_fails_closed_without_pidfd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _protected_file(tmp_path / "config", "site.yaml", "site_id: remote\n")
    state_directory = tmp_path / "state"
    state_directory.mkdir(mode=0o700)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        identity = _capture_started_process_identity(process)
        record = OwnershipRecord(
            generation="1" * 32,
            phase="ready",
            configuration_sha256=_configuration_digest(config),
            created_at=datetime.now(UTC),
            identity=identity,
            ready_at=datetime.now(UTC),
        )
        state_path = state_directory / "agent.json"
        jupyter_agent._write_record(state_path, record)
        monkeypatch.setattr(os, "pidfd_open", None)
        monkeypatch.setattr(signal, "pidfd_send_signal", None)
        monkeypatch.setattr(jupyter_agent, "_pidfd_syscall_numbers", lambda: None)

        with pytest.raises(JupyterAgentBootstrapError, match="requires Linux pidfd"):
            stop_owned_jupyter_agent(
                config_path=config,
                site_environment_path=None,
                state_directory=state_directory,
                shutdown_timeout_seconds=1,
            )
        assert process.poll() is None
        assert state_path.exists()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_stop_owned_uses_syscall_fallback_without_python_pidfd_wrappers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if jupyter_agent._pidfd_syscall_numbers() is None:
        pytest.skip("raw pidfd syscall numbers are unavailable on this platform")
    config = _protected_file(tmp_path / "config", "site.yaml", "site_id: remote\n")
    state_directory = tmp_path / "state"
    state_directory.mkdir(mode=0o700)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        identity = _capture_started_process_identity(process)
        record = OwnershipRecord(
            generation="3" * 32,
            phase="ready",
            configuration_sha256=_configuration_digest(config),
            created_at=datetime.now(UTC),
            identity=identity,
            ready_at=datetime.now(UTC),
        )
        jupyter_agent._write_record(state_directory / "agent.json", record)
        monkeypatch.setattr(os, "pidfd_open", None)
        monkeypatch.setattr(signal, "pidfd_send_signal", None)

        assert stop_owned_jupyter_agent(
            config_path=config,
            site_environment_path=None,
            state_directory=state_directory,
            shutdown_timeout_seconds=5,
        )
        assert process.wait(timeout=5) == -signal.SIGTERM
        assert not (state_directory / "agent.json").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_stop_owned_never_escalates_past_sigterm_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _protected_file(tmp_path / "config", "site.yaml", "site_id: remote\n")
    state_directory = tmp_path / "state"
    state_directory.mkdir(mode=0o700)
    identity = jupyter_agent.ProcessIdentity(
        pid=12345,
        process_group_id=12345,
        start_ticks=77,
        boot_id="test-boot",
        command_sha256="a" * 64,
    )
    record = OwnershipRecord(
        generation="2" * 32,
        phase="ready",
        configuration_sha256=_configuration_digest(config),
        created_at=datetime.now(UTC),
        identity=identity,
        ready_at=datetime.now(UTC),
    )
    state_path = state_directory / "agent.json"
    jupyter_agent._write_record(state_path, record)
    read_fd, write_fd = os.pipe()
    sent: list[int] = []
    monkeypatch.setattr(jupyter_agent, "_open_verified_pidfd", lambda _identity: read_fd)
    monkeypatch.setattr(
        jupyter_agent,
        "_send_pidfd_signal",
        lambda _pidfd, signum: sent.append(signum),
    )
    monkeypatch.setattr(jupyter_agent, "_pidfd_has_exited", lambda _pidfd: False)
    try:
        with pytest.raises(JupyterAgentBootstrapError, match="shutdown timeout"):
            stop_owned_jupyter_agent(
                config_path=config,
                site_environment_path=None,
                state_directory=state_directory,
                shutdown_timeout_seconds=0.001,
            )
    finally:
        os.close(write_fd)

    assert sent == [signal.SIGTERM]
    assert state_path.exists()
