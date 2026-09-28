from __future__ import annotations

import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

import topodrome.deployment.ssh_tunnel as ssh_tunnel
from topodrome.deployment.ssh_tunnel import (
    SSHLocalForwardConfig,
    SSHLocalForwardError,
    SSHLocalForwardSupervisor,
    TunnelOwnershipRecord,
    TunnelProcessIdentity,
    ssh_local_forward_arguments,
)

pytestmark = pytest.mark.unit


def _protected_file(parent: Path, name: str, content: str) -> Path:
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = parent / name
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _fake_ssh(parent: Path, *, listen: bool = True) -> Path:
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    executable = parent / ("fake-ssh" if listen else "fake-ssh-no-listener")
    body = f"""#!{sys.executable}
import signal
import socket
import sys
import time

running = True
def stop(_signum, _frame):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
listener = None
if {listen!r}:
    forward = sys.argv[sys.argv.index('-L') + 1]
    host, raw_port, _target_host, _target_port = forward.split(':')
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind((host, int(raw_port)))
    listener.listen()
while running:
    time.sleep(0.01)
if listener is not None:
    listener.close()
"""
    executable.write_text(body, encoding="utf-8")
    executable.chmod(0o700)
    return executable


def _config(
    tmp_path: Path,
    *,
    port: int | None = None,
    listen: bool = True,
    startup_timeout_seconds: float = 2,
) -> SSHLocalForwardConfig:
    credentials = tmp_path / "credentials"
    identity = _protected_file(credentials, "id_ed25519", "private identity\n")
    known_hosts = _protected_file(credentials, "known_hosts", "host ssh-ed25519 public\n")
    executable = _fake_ssh(tmp_path / "bin", listen=listen)
    return SSHLocalForwardConfig(
        identity_file=identity,
        known_hosts_file=known_hosts,
        remote_host="bastion.example",
        remote_user="topodrome",
        local_port=port or _free_port(),
        target_port=8042,
        state_directory=tmp_path / "state",
        ssh_executable=executable,
        remote_port=2222,
        startup_timeout_seconds=startup_timeout_seconds,
        shutdown_timeout_seconds=1,
        poll_interval_seconds=0.01,
        connect_timeout_seconds=7,
        server_alive_interval_seconds=11,
        server_alive_count_max=4,
    )


def test_fixed_arguments_enforce_noninteractive_host_verified_foreground_tunnel(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    arguments = ssh_local_forward_arguments(config)

    assert arguments[0] == str(config.ssh_executable)
    assert arguments[1:5] == ("-F", "/dev/null", "-N", "-T")
    assert "-f" not in arguments
    assert "BatchMode=yes" in arguments
    assert "StrictHostKeyChecking=yes" in arguments
    assert "ExitOnForwardFailure=yes" in arguments
    assert "PasswordAuthentication=no" in arguments
    assert "KbdInteractiveAuthentication=no" in arguments
    assert "IdentitiesOnly=yes" in arguments
    assert "ServerAliveInterval=11" in arguments
    assert "ServerAliveCountMax=4" in arguments
    assert "ConnectTimeout=7" in arguments
    assert f"UserKnownHostsFile={config.known_hosts_file}" in arguments
    assert (
        arguments[arguments.index("-L") + 1]
        == f"127.0.0.1:{config.local_port}:127.0.0.1:8042"
    )
    assert arguments[-2:] == ("--", "topodrome@bastion.example")


def test_starts_adopts_and_stops_only_exact_owned_process(tmp_path: Path) -> None:
    config = _config(tmp_path)
    owner = SSHLocalForwardSupervisor(config)
    adopted = SSHLocalForwardSupervisor(config)

    started = owner.ensure_running()
    try:
        assert not started.adopted
        assert os.getpgid(started.pid) == started.pid
        assert ssh_tunnel._process_owns_listener(
            started.pid,
            config.local_host,
            config.local_port,
        )
        recovered = adopted.ensure_running()
        assert recovered.adopted
        assert recovered.pid == started.pid

        state = json.loads(owner.state_path.read_text(encoding="utf-8"))
        assert state["identity"]["pid"] == started.pid
        assert stat.S_IMODE(owner.state_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(owner.log_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(config.state_directory.stat().st_mode) == 0o700
        assert "private identity" not in owner.state_path.read_text(encoding="utf-8")
    finally:
        owner.stop()

    assert not owner.state_path.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(started.pid, 0)


def test_refuses_unowned_occupied_local_port_without_spawning(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as foreign:
        foreign.bind(("127.0.0.1", 0))
        foreign.listen()
        port = int(foreign.getsockname()[1])
        config = _config(tmp_path, port=port)
        supervisor = SSHLocalForwardSupervisor(config)

        with pytest.raises(SSHLocalForwardError, match="occupied by an unowned"):
            supervisor.ensure_running()

        assert not supervisor.state_path.exists()
        assert foreign.fileno() >= 0


def test_changed_protected_configuration_never_replaces_live_tunnel(tmp_path: Path) -> None:
    config = _config(tmp_path)
    owner = SSHLocalForwardSupervisor(config)
    started = owner.ensure_running()
    try:
        config.known_hosts_file.write_text("changed host key\n", encoding="utf-8")
        config.known_hosts_file.chmod(0o600)
        replacement = SSHLocalForwardSupervisor(config)

        with pytest.raises(SSHLocalForwardError, match="different protected"):
            replacement.ensure_running()

        os.kill(started.pid, 0)
        assert ssh_tunnel._process_owns_listener(
            started.pid,
            config.local_host,
            config.local_port,
        )
    finally:
        # Restore the original configuration fingerprint so the original
        # supervisor can prove and stop precisely the process it launched.
        config.known_hosts_file.write_text("host ssh-ed25519 public\n", encoding="utf-8")
        config.known_hosts_file.chmod(0o600)
        owner.stop()


def test_forged_live_identity_is_never_signalled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.state_directory.mkdir(mode=0o700)
    foreign = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        identity = ssh_tunnel._capture_process_identity(foreign.pid)
        forged = TunnelProcessIdentity(
            pid=identity.pid,
            process_group_id=identity.process_group_id,
            uid=identity.uid,
            start_ticks=identity.start_ticks + 1,
            boot_id=identity.boot_id,
            command_sha256=identity.command_sha256,
        )
        record = TunnelOwnershipRecord(
            configuration_sha256=ssh_tunnel._configuration_sha256(config),
            created_at=time.time(),
            identity=forged,
        )
        ssh_tunnel._write_record(config.state_directory / "ssh-local-forward.json", record)
        signalled: list[tuple[int, int]] = []

        original = os.killpg

        def guarded_signal(process_group_id: int, signum: int) -> None:
            if signum != 0:
                signalled.append((process_group_id, signum))
            original(process_group_id, signum)

        monkeypatch.setattr(os, "killpg", guarded_signal)
        supervisor = SSHLocalForwardSupervisor(config)

        with pytest.raises(SSHLocalForwardError, match="changed identity"):
            supervisor.stop()

        assert signalled == []
        assert foreign.poll() is None
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


def test_readiness_timeout_stops_only_new_child_and_removes_state(tmp_path: Path) -> None:
    config = _config(tmp_path, listen=False, startup_timeout_seconds=0.15)
    supervisor = SSHLocalForwardSupervisor(config)

    with pytest.raises(SSHLocalForwardError, match="did not become ready"):
        supervisor.ensure_running()

    assert not supervisor.state_path.exists()
    assert supervisor.log_path.exists()
    assert stat.S_IMODE(supervisor.log_path.stat().st_mode) == 0o600


@pytest.mark.parametrize("unsafe", ["identity-mode", "known-hosts-link", "public-bind"])
def test_rejects_unprotected_inputs_and_non_loopback_bind(
    tmp_path: Path,
    unsafe: str,
) -> None:
    if unsafe == "public-bind":
        base = _config(tmp_path)
        with pytest.raises(ValueError, match="loopback"):
            SSHLocalForwardConfig(
                identity_file=base.identity_file,
                known_hosts_file=base.known_hosts_file,
                remote_host=base.remote_host,
                remote_user=base.remote_user,
                local_host="0.0.0.0",  # noqa: S104 - rejection is the subject of this test
                local_port=base.local_port,
                target_port=base.target_port,
                state_directory=base.state_directory,
                ssh_executable=base.ssh_executable,
            )
        return

    config = _config(tmp_path)
    if unsafe == "identity-mode":
        config.identity_file.chmod(0o640)
    else:
        actual = config.known_hosts_file
        link = actual.with_name("known_hosts_link")
        link.symlink_to(actual)
        config = SSHLocalForwardConfig(
            identity_file=config.identity_file,
            known_hosts_file=link,
            remote_host=config.remote_host,
            remote_user=config.remote_user,
            local_port=config.local_port,
            target_port=config.target_port,
            state_directory=config.state_directory,
            ssh_executable=config.ssh_executable,
        )

    with pytest.raises(PermissionError, match=r"SSH (identity|known-hosts)"):
        SSHLocalForwardSupervisor(config).ensure_running()


def test_stop_cleans_a_dead_stale_record_without_touching_occupied_port(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.state_directory.mkdir(mode=0o700)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    identity = ssh_tunnel._capture_process_identity(process.pid)
    process.terminate()
    process.wait(timeout=5)
    record = TunnelOwnershipRecord(
        configuration_sha256=ssh_tunnel._configuration_sha256(config),
        created_at=time.time(),
        identity=identity,
    )
    ssh_tunnel._write_record(config.state_directory / "ssh-local-forward.json", record)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as foreign:
        foreign.bind((config.local_host, config.local_port))
        foreign.listen()

        assert not SSHLocalForwardSupervisor(config).stop()
        assert foreign.fileno() >= 0

    assert not (config.state_directory / "ssh-local-forward.json").exists()


def test_config_rejects_option_injection(tmp_path: Path) -> None:
    base = _config(tmp_path)
    with pytest.raises(ValueError, match="remote_host"):
        SSHLocalForwardConfig(
            identity_file=base.identity_file,
            known_hosts_file=base.known_hosts_file,
            remote_host="-oProxyCommand=bad",
            remote_user=base.remote_user,
            local_port=base.local_port,
            target_port=base.target_port,
            state_directory=base.state_directory,
            ssh_executable=base.ssh_executable,
        )


def test_stop_uses_term_then_kill_only_for_still_exact_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = TunnelProcessIdentity(
        pid=424242,
        process_group_id=424242,
        uid=os.geteuid(),
        start_ticks=12,
        boot_id="boot",
        command_sha256="a" * 64,
    )
    exact = True
    exists = True
    signals: list[int] = []

    monkeypatch.setattr(ssh_tunnel, "_process_matches", lambda _identity: exact)
    monkeypatch.setattr(ssh_tunnel, "_process_group_exists", lambda _pgid: exists)

    def signal_group(_pgid: int, signum: int) -> None:
        nonlocal exists
        signals.append(signum)
        if signum == signal.SIGKILL:
            exists = False

    monkeypatch.setattr(os, "killpg", signal_group)
    monotonic_values = iter((0.0, 0.0, 1.1, 1.1, 1.1))
    monkeypatch.setattr(time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    ssh_tunnel._terminate_exact(
        identity,
        timeout_seconds=1,
        poll_interval_seconds=0.01,
        process=None,
    )

    assert signals == [signal.SIGTERM, signal.SIGKILL]
