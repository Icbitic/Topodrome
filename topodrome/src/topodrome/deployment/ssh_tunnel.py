"""Crash-safe supervision for an outbound SSH local port forward.

The site agent can use a loopback scheduler URL while this process maintains
an SSH ``-L`` forward to a scheduler listener reachable from the SSH server.
This module deliberately contains no settings or application wiring: callers
construct :class:`SSHLocalForwardConfig` from their protected configuration and
own the supervisor's lifetime explicitly.

Only a process whose boot ID, PID, process-group ID, start time, and command
line still match a private durable record is adopted or signalled.  A listener
without that complete proof is always foreign.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import stat
import subprocess
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol
from uuid import uuid4

_FORMAT_VERSION = 1
_SAFE_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")
_SAFE_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,252}")
_LOOPBACK_HOSTS = {"127.0.0.1", "::1"}


class SSHLocalForwardError(RuntimeError):
    """The tunnel could not be proven safe, started, or stopped."""


@dataclass(frozen=True, slots=True)
class SSHLocalForwardConfig:
    """Static configuration for one loopback-only SSH local forward."""

    identity_file: Path
    known_hosts_file: Path
    remote_host: str
    remote_user: str
    local_port: int
    target_port: int
    state_directory: Path
    ssh_executable: Path = Path("/usr/bin/ssh")
    remote_port: int = 22
    local_host: str = "127.0.0.1"
    target_host: str = "127.0.0.1"
    startup_timeout_seconds: float = 15.0
    shutdown_timeout_seconds: float = 10.0
    poll_interval_seconds: float = 0.05
    connect_timeout_seconds: int = 10
    server_alive_interval_seconds: int = 15
    server_alive_count_max: int = 3

    def __post_init__(self) -> None:
        for field_name in (
            "identity_file",
            "known_hosts_file",
            "state_directory",
            "ssh_executable",
        ):
            value = Path(getattr(self, field_name)).expanduser()
            if not value.is_absolute():
                raise ValueError(f"{field_name} must be absolute")
            object.__setattr__(self, field_name, value)
        if self.local_host not in _LOOPBACK_HOSTS:
            raise ValueError("SSH local forward must bind an explicit loopback address")
        _validate_host(self.remote_host, label="remote_host")
        _validate_host(self.target_host, label="target_host")
        if not _SAFE_USER.fullmatch(self.remote_user):
            raise ValueError("remote_user contains unsafe characters")
        for field_name in ("local_port", "target_port", "remote_port"):
            port_value = int(getattr(self, field_name))
            if not 1 <= port_value <= 65_535:
                raise ValueError(f"{field_name} must be between 1 and 65535")
        for field_name in (
            "connect_timeout_seconds",
            "server_alive_interval_seconds",
            "server_alive_count_max",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.startup_timeout_seconds <= 0 or self.shutdown_timeout_seconds <= 0:
            raise ValueError("tunnel lifecycle timeouts must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")


@dataclass(frozen=True, slots=True)
class TunnelProcessIdentity:
    pid: int
    process_group_id: int
    uid: int
    start_ticks: int
    boot_id: str
    command_sha256: str

    @classmethod
    def from_mapping(cls, raw: Any) -> TunnelProcessIdentity:
        if not isinstance(raw, dict):
            raise ValueError("process identity must be a mapping")
        identity = cls(
            pid=int(raw["pid"]),
            process_group_id=int(raw["process_group_id"]),
            uid=int(raw["uid"]),
            start_ticks=int(raw["start_ticks"]),
            boot_id=str(raw["boot_id"]),
            command_sha256=str(raw["command_sha256"]),
        )
        identity.validate()
        return identity

    def validate(self) -> None:
        if (
            self.pid <= 1
            or self.process_group_id != self.pid
            or self.uid < 0
            or self.start_ticks <= 0
            or not self.boot_id
            or not re.fullmatch(r"[0-9a-f]{64}", self.command_sha256)
        ):
            raise ValueError("invalid tunnel process identity")

    def as_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "uid": self.uid,
            "start_ticks": self.start_ticks,
            "boot_id": self.boot_id,
            "command_sha256": self.command_sha256,
        }


@dataclass(frozen=True, slots=True)
class TunnelOwnershipRecord:
    configuration_sha256: str
    created_at: float
    identity: TunnelProcessIdentity

    @classmethod
    def from_mapping(cls, raw: Any) -> TunnelOwnershipRecord:
        if not isinstance(raw, dict) or int(raw.get("version", 0)) != _FORMAT_VERSION:
            raise ValueError("unsupported tunnel ownership record")
        record = cls(
            configuration_sha256=str(raw["configuration_sha256"]),
            created_at=float(raw["created_at"]),
            identity=TunnelProcessIdentity.from_mapping(raw["identity"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        self.identity.validate()
        if not re.fullmatch(r"[0-9a-f]{64}", self.configuration_sha256) or self.created_at <= 0:
            raise ValueError("invalid tunnel ownership record")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": _FORMAT_VERSION,
            "configuration_sha256": self.configuration_sha256,
            "created_at": self.created_at,
            "identity": self.identity.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class SSHLocalForwardStatus:
    """Result of ensuring one forward, including whether it was adopted."""

    pid: int
    local_host: str
    local_port: int
    adopted: bool


class SSHLocalForwardSupervisor:
    """Own exactly one foreground ``ssh -L`` process and its private state."""

    def __init__(self, config: SSHLocalForwardConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def state_path(self) -> Path:
        return self.config.state_directory / "ssh-local-forward.json"

    @property
    def log_path(self) -> Path:
        return self.config.state_directory / "ssh-local-forward.log"

    def ensure_running(self) -> SSHLocalForwardStatus:
        """Adopt an exact live tunnel or start one after proving the port free."""

        state_root = _prepare_private_directory(self.config.state_directory)
        with _locked_file(state_root / "ssh-local-forward.lock"):
            self._reap_exited_child()
            configuration_sha256 = _configuration_sha256(self.config)
            existing = _read_record(self.state_path)
            if existing is not None:
                if _process_matches(existing.identity):
                    if existing.configuration_sha256 != configuration_sha256:
                        raise SSHLocalForwardError(
                            "a live owned tunnel uses different protected configuration"
                        )
                    self._wait_until_ready(existing, process=None)
                    return SSHLocalForwardStatus(
                        pid=existing.identity.pid,
                        local_host=self.config.local_host,
                        local_port=self.config.local_port,
                        adopted=True,
                    )
                if _process_group_exists(existing.identity.process_group_id) and not (
                    _only_owned_zombies_remain(existing.identity)
                ):
                    raise SSHLocalForwardError(
                        "the recorded tunnel process group is live but its identity changed"
                    )

            if not _port_is_free(self.config.local_host, self.config.local_port):
                raise SSHLocalForwardError(
                    "the SSH local-forward port is occupied by an unowned process"
                )

            arguments = ssh_local_forward_arguments(self.config)
            log_handle = _open_private_log(self.log_path)
            process: subprocess.Popen[bytes] | None = None
            record: TunnelOwnershipRecord | None = None
            try:
                process = self._spawn(arguments, log_handle)
                identity = _capture_process_identity(process.pid)
                record = TunnelOwnershipRecord(
                    configuration_sha256=configuration_sha256,
                    created_at=time.time(),
                    identity=identity,
                )
                _write_record(self.state_path, record)
                self._wait_until_ready(record, process=process)
            except BaseException:
                if record is not None:
                    with suppress(BaseException):
                        _terminate_exact(
                            record.identity,
                            timeout_seconds=self.config.shutdown_timeout_seconds,
                            poll_interval_seconds=self.config.poll_interval_seconds,
                            process=process,
                        )
                    with suppress(BaseException):
                        _remove_record_if_equal(self.state_path, record)
                elif process is not None:
                    with suppress(BaseException):
                        _terminate_spawned_process(
                            process,
                            timeout_seconds=self.config.shutdown_timeout_seconds,
                            poll_interval_seconds=self.config.poll_interval_seconds,
                        )
                raise
            finally:
                log_handle.close()

            assert process is not None and record is not None
            self._process = process
            return SSHLocalForwardStatus(
                pid=record.identity.pid,
                local_host=self.config.local_host,
                local_port=self.config.local_port,
                adopted=False,
            )

    def stop(self) -> bool:
        """Stop only the exact process group in the private ownership record."""

        state_root = _prepare_private_directory(self.config.state_directory)
        with _locked_file(state_root / "ssh-local-forward.lock"):
            self._reap_exited_child()
            record = _read_record(self.state_path)
            if record is None:
                return False
            if not _process_matches(record.identity):
                if _process_group_exists(record.identity.process_group_id) and not (
                    _only_owned_zombies_remain(record.identity)
                ):
                    raise SSHLocalForwardError(
                        "refusing to signal a tunnel process group with changed identity"
                    )
                _remove_record_if_equal(self.state_path, record)
                return False
            process = self._process
            if process is not None and process.pid != record.identity.pid:
                process = None
            _terminate_exact(
                record.identity,
                timeout_seconds=self.config.shutdown_timeout_seconds,
                poll_interval_seconds=self.config.poll_interval_seconds,
                process=process,
            )
            _remove_record_if_equal(self.state_path, record)
            self._process = None
            return True

    def _reap_exited_child(self) -> None:
        # Popen retains the exact child until waitpid reaps it. An unreaped
        # exit has an empty cmdline but still makes killpg(pgid, 0) succeed;
        # treating that zombie as changed live ownership prevents all retries.
        if self._process is not None and self._process.poll() is not None:
            self._process = None

    def _spawn(
        self,
        arguments: Sequence[str],
        log_handle: IO[bytes],
    ) -> subprocess.Popen[bytes]:
        return subprocess.Popen(  # noqa: S603 - validated absolute executable and fixed argv
            list(arguments),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env={"LANG": "C", "LC_ALL": "C"},
            start_new_session=True,
            close_fds=True,
        )

    def _wait_until_ready(
        self,
        record: TunnelOwnershipRecord,
        *,
        process: subprocess.Popen[bytes] | None,
    ) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise SSHLocalForwardError(
                    f"SSH local-forward process exited with {process.returncode}"
                )
            if not _process_matches(record.identity):
                raise SSHLocalForwardError("SSH local-forward process identity changed")
            if _process_owns_listener(
                record.identity.pid,
                self.config.local_host,
                self.config.local_port,
            ):
                return
            time.sleep(
                min(
                    self.config.poll_interval_seconds,
                    max(0.0, deadline - time.monotonic()),
                )
            )
        raise SSHLocalForwardError("SSH local forward did not become ready in time")


def ssh_local_forward_arguments(config: SSHLocalForwardConfig) -> tuple[str, ...]:
    """Build the complete fixed SSH argv without consulting user SSH config."""

    destination_host = _bracket_ipv6(config.remote_host)
    target_host = _bracket_ipv6(config.target_host)
    local_host = _bracket_ipv6(config.local_host)
    forward = f"{local_host}:{config.local_port}:{target_host}:{config.target_port}"
    return (
        str(config.ssh_executable),
        "-F",
        "/dev/null",
        "-N",
        "-T",
        "-i",
        str(config.identity_file),
        "-p",
        str(config.remote_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={config.known_hosts_file}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        f"ConnectTimeout={config.connect_timeout_seconds}",
        "-o",
        f"ServerAliveInterval={config.server_alive_interval_seconds}",
        "-o",
        f"ServerAliveCountMax={config.server_alive_count_max}",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ControlPersist=no",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "GatewayPorts=no",
        "-o",
        "LogLevel=ERROR",
        "-L",
        forward,
        "--",
        f"{config.remote_user}@{destination_host}",
    )


def _configuration_sha256(config: SSHLocalForwardConfig) -> str:
    _validate_executable(config.ssh_executable)
    identity_digest = _protected_file_digest(config.identity_file, label="SSH identity")
    known_hosts_digest = _protected_file_digest(
        config.known_hosts_file,
        label="SSH known-hosts file",
    )
    payload = {
        "argv": ssh_local_forward_arguments(config),
        "identity_sha256": identity_digest,
        "known_hosts_sha256": known_hosts_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_executable(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        raise PermissionError("SSH executable is unavailable") from None
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid not in {0, os.geteuid()}
        or metadata.st_mode & 0o022
        or not metadata.st_mode & stat.S_IXUSR
    ):
        raise PermissionError("SSH executable must be a protected executable regular file")


def _protected_file_digest(path: Path, *, label: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise PermissionError(f"{label} is unavailable") from None
    try:
        metadata = os.fstat(descriptor)
        parent = path.parent.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid not in {0, os.geteuid()}
            or stat.S_IMODE(parent.st_mode) & 0o022
        ):
            raise PermissionError(f"{label} must be owner-controlled and mode 0600 or stricter")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _prepare_private_directory(path: Path) -> Path:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise PermissionError("SSH tunnel state directory must be private and owner-controlled")
    return path


@contextmanager
def _locked_file(path: Path) -> Iterator[int]:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError:
        raise PermissionError("SSH tunnel lock is unavailable") from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise PermissionError("SSH tunnel lock must be private and owner-controlled")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        os.close(descriptor)


def _open_private_log(path: Path) -> IO[bytes]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError:
        raise PermissionError("SSH tunnel log is unavailable") from None
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        os.close(descriptor)
        raise PermissionError("SSH tunnel log must be private and owner-controlled")
    return os.fdopen(descriptor, "ab", buffering=0)


def _read_record(path: Path) -> TunnelOwnershipRecord | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError:
        raise SSHLocalForwardError("SSH tunnel ownership state is unavailable") from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise PermissionError("SSH tunnel ownership state is not private")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            raw = json.load(handle)
        return TunnelOwnershipRecord.from_mapping(raw)
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        raise SSHLocalForwardError("SSH tunnel ownership state is invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_record(path: Path, record: TunnelOwnershipRecord) -> None:
    record.validate()
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(record.as_dict(), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _remove_record_if_equal(path: Path, expected: TunnelOwnershipRecord) -> None:
    current = _read_record(path)
    if current is None:
        return
    if current != expected:
        raise SSHLocalForwardError("SSH tunnel ownership state changed")
    path.unlink()
    _fsync_directory(path.parent)


def _capture_process_identity(pid: int) -> TunnelProcessIdentity:
    # A script executable may perform one interpreter re-exec immediately
    # after ``Popen`` returns. Linux can expose an empty cmdline during that
    # narrow transition, so sample for a small bounded interval.
    deadline = time.monotonic() + 0.25
    while True:
        process_metadata = Path(f"/proc/{pid}").stat()
        if process_metadata.st_uid != os.geteuid():
            raise SSHLocalForwardError("SSH tunnel child has an unexpected owner")
        start_ticks, state = _process_stat(pid)
        process_group_id = os.getpgid(pid)
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        if process_group_id == pid and state != "Z" and command:
            break
        if state == "Z" or time.monotonic() >= deadline:
            raise SSHLocalForwardError("SSH tunnel child has an invalid process identity")
        time.sleep(0.005)
    identity = TunnelProcessIdentity(
        pid=pid,
        process_group_id=process_group_id,
        uid=process_metadata.st_uid,
        start_ticks=start_ticks,
        boot_id=_boot_id(),
        command_sha256=hashlib.sha256(command).hexdigest(),
    )
    identity.validate()
    return identity


def _process_matches(identity: TunnelProcessIdentity) -> bool:
    try:
        process_metadata = Path(f"/proc/{identity.pid}").stat()
        start_ticks, state = _process_stat(identity.pid)
        if state == "Z":
            return False
        if os.getpgid(identity.pid) != identity.process_group_id:
            return False
        command = Path(f"/proc/{identity.pid}/cmdline").read_bytes()
        return (
            _boot_id() == identity.boot_id
            and identity.uid == os.geteuid()
            and process_metadata.st_uid == identity.uid
            and start_ticks == identity.start_ticks
            and bool(command)
            and hashlib.sha256(command).hexdigest() == identity.command_sha256
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
        return False


def _process_stat(pid: int) -> tuple[int, str]:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    fields = raw[closing + 2 :].split()
    if closing < 0 or len(fields) < 20:
        raise ValueError("invalid process stat")
    return int(fields[19]), fields[0]


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _only_owned_zombies_remain(identity: TunnelProcessIdentity) -> bool:
    """Recognize a dead adopted leader without signalling or reaping strangers.

    A replacement supervisor cannot waitpid the old parent's child. A proven
    zombie has no sockets or ability to execute; still refuse recovery if any
    live member of its group remains or ownership cannot be established.
    """
    try:
        proc = Path(f"/proc/{identity.pid}")
        start, state = _process_stat(identity.pid)
        if (
            state != "Z"
            or start != identity.start_ticks
            or _boot_id() != identity.boot_id
            or identity.uid != os.geteuid()
            or proc.stat().st_uid != identity.uid
            or os.getpgid(identity.pid) != identity.process_group_id
        ):
            return False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "stat").read_text(encoding="utf-8")
                fields = raw.rsplit(")", 1)[1].split()
                if int(fields[2]) == identity.process_group_id and fields[0] != "Z":
                    return False
            except (FileNotFoundError, ProcessLookupError):
                continue
        return True
    except (OSError, ValueError, IndexError):
        return False


def _terminate_exact(
    identity: TunnelProcessIdentity,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    process: subprocess.Popen[bytes] | None,
) -> None:
    if not _process_matches(identity):
        if _process_group_exists(identity.process_group_id):
            raise SSHLocalForwardError("tunnel identity changed before termination")
        return
    os.killpg(identity.process_group_id, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while True:
        # Reap a child launched by this supervisor. Otherwise its zombie keeps
        # the process-group ID observable even though the tunnel has exited.
        if process is not None and process.pid == identity.pid:
            process.poll()
        if not _process_group_exists(identity.process_group_id):
            break
        if time.monotonic() >= deadline:
            if not _process_matches(identity):
                raise SSHLocalForwardError(
                    "tunnel leader exited but its process group remains; refusing escalation"
                )
            os.killpg(identity.process_group_id, signal.SIGKILL)
            kill_deadline = time.monotonic() + min(2.0, timeout_seconds)
            while _process_group_exists(identity.process_group_id):
                if process is not None and process.pid == identity.pid:
                    process.poll()
                if time.monotonic() >= kill_deadline:
                    raise SSHLocalForwardError("owned tunnel process group did not terminate")
                time.sleep(poll_interval_seconds)
            break
        time.sleep(poll_interval_seconds)
    if process is not None and process.pid == identity.pid:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=min(1.0, timeout_seconds))


def _terminate_spawned_process(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(poll_interval_seconds)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=min(2.0, timeout_seconds))


def _port_is_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if host == "::1" else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        # Match OpenSSH's listener semantics. A stopped forward can leave
        # accepted connections in TIME_WAIT; those are not a foreign listener
        # and must not block an immediate, idle-gated agent restart. This does
        # not enable SO_REUSEPORT or permit sharing a live listening socket.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _process_owns_listener(pid: int, host: str, port: int) -> bool:
    inodes: set[str] = set()
    try:
        for descriptor in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            match = re.fullmatch(r"socket:\[(\d+)\]", target)
            if match is not None:
                inodes.add(match.group(1))
    except (FileNotFoundError, PermissionError, OSError):
        return False
    if not inodes:
        return False
    table, address = _proc_listener_address(host)
    try:
        lines = Path(f"/proc/{pid}/net/{table}").read_text(encoding="ascii").splitlines()[1:]
    except (FileNotFoundError, PermissionError, OSError, UnicodeError):
        return False
    expected_port = f"{port:04X}"
    for line in lines:
        fields = line.split()
        if len(fields) < 10 or fields[3] != "0A" or fields[9] not in inodes:
            continue
        raw_address, separator, raw_port = fields[1].partition(":")
        if separator and raw_address.upper() == address and raw_port.upper() == expected_port:
            return True
    return False


def _proc_listener_address(host: str) -> tuple[str, str]:
    if host == "127.0.0.1":
        return "tcp", "0100007F"
    if host == "::1":
        return "tcp6", "00000000000000000000000001000000"
    raise ValueError("listener host is not an explicit loopback address")


def _validate_host(value: str, *, label: str) -> None:
    if not _SAFE_HOST.fullmatch(value) or value.startswith(":") or value.endswith(":"):
        raise ValueError(f"{label} contains unsafe characters")


def _bracket_ipv6(value: str) -> str:
    return f"[{value}]" if ":" in value else value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class LocalForwardLifecycle(Protocol):
    """Common ownership lifecycle for a fixed route or bounded route manager."""

    def ensure_running(self) -> object: ...

    def stop(self) -> object: ...


class SSHLocalForwardGroup:
    """Own independent control and bulk TCP routes for one agent lifetime."""

    def __init__(self, tunnels: Sequence[LocalForwardLifecycle]) -> None:
        self.tunnels = tuple(tunnels)

    def ensure_running(self) -> None:
        first_error: Exception | None = None
        for tunnel in self.tunnels:
            try:
                tunnel.ensure_running()
            except Exception as exc:
                # A broken bulk route must not tear down a healthy control
                # route. Initial construction owns rollback for both routes.
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def stop(self) -> None:
        first_error: BaseException | None = None
        for tunnel in reversed(self.tunnels):
            try:
                tunnel.stop()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


__all__ = [
    "SSHLocalForwardConfig",
    "SSHLocalForwardError",
    "SSHLocalForwardGroup",
    "SSHLocalForwardStatus",
    "SSHLocalForwardSupervisor",
    "TunnelOwnershipRecord",
    "TunnelProcessIdentity",
    "ssh_local_forward_arguments",
]
