"""Crash-safe launcher for an outbound agent on a terminal-only Jupyter site.

The public command contains paths and tuning values only.  It validates a
protected site configuration and optional site environment, starts one detached
process, and returns only after that process has built the local agent and
remained alive for a short stabilization interval.  A durable PID generation
record prevents concurrent or post-crash invocations from creating duplicates.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import fcntl
import hashlib
import json
import os
import re
import select
import shlex
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any
from uuid import uuid4

import yaml

from topodrome.config import SiteSettings
from topodrome.runtime.environment import minimal_ray_head_environment
from topodrome.site_agent.app import AgentApplication, build_application

READY_MARKER = "TOPODROME_AGENT_READY"
STOPPED_MARKER = "TOPODROME_AGENT_STOPPED"
ABSENT_MARKER = "TOPODROME_AGENT_ABSENT"
_CHILD_MODE = "TOPODROME_JUPYTER_AGENT_BOOTSTRAP_CHILD"
_CHILD_GENERATION = "TOPODROME_JUPYTER_AGENT_BOOTSTRAP_GENERATION"
_FORMAT_VERSION = 1
_LAUNCH_CLAIM_GRACE_SECONDS = 10.0
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Linux allocated these modern process syscalls from the common numbering
# range on the architectures Topodrome supports.  Some relocatable CPython
# builds omit their wrappers when built against older libc headers even though
# the running kernel implements them.  Unknown systems deliberately fail
# closed instead of falling back to a numeric-PID signal.
_PIDFD_SYSCALLS = {
    "aarch64": (434, 424),
    "arm64": (434, 424),
    "i386": (434, 424),
    "i686": (434, 424),
    "ppc64": (434, 424),
    "ppc64le": (434, 424),
    "riscv64": (434, 424),
    "s390x": (434, 424),
    "x86_64": (434, 424),
}


class JupyterAgentBootstrapError(RuntimeError):
    """The wrapper could not prove one healthy, owned agent generation."""


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    process_group_id: int
    start_ticks: int
    boot_id: str
    command_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "start_ticks": self.start_ticks,
            "boot_id": self.boot_id,
            "command_sha256": self.command_sha256,
        }


@dataclass(frozen=True, slots=True)
class OwnershipRecord:
    generation: str
    phase: str
    configuration_sha256: str
    created_at: datetime
    identity: ProcessIdentity | None = None
    ready_at: datetime | None = None

    @classmethod
    def from_mapping(cls, raw: Any) -> OwnershipRecord:
        if not isinstance(raw, dict) or raw.get("version") != _FORMAT_VERSION:
            raise ValueError("unsupported ownership record")
        generation = str(raw["generation"])
        phase = str(raw["phase"])
        configuration_sha256 = str(raw["configuration_sha256"])
        created_at = _parse_datetime(raw["created_at"])
        identity_raw = raw.get("identity")
        identity = None
        if identity_raw is not None:
            if not isinstance(identity_raw, dict):
                raise ValueError("invalid process identity")
            identity = ProcessIdentity(
                pid=int(identity_raw["pid"]),
                process_group_id=int(identity_raw["process_group_id"]),
                start_ticks=int(identity_raw["start_ticks"]),
                boot_id=str(identity_raw["boot_id"]),
                command_sha256=str(identity_raw["command_sha256"]),
            )
        ready_at = None if raw.get("ready_at") is None else _parse_datetime(raw["ready_at"])
        record = cls(
            generation=generation,
            phase=phase,
            configuration_sha256=configuration_sha256,
            created_at=created_at,
            identity=identity,
            ready_at=ready_at,
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{32}", self.generation):
            raise ValueError("invalid ownership generation")
        if not re.fullmatch(r"[0-9a-f]{64}", self.configuration_sha256):
            raise ValueError("invalid configuration identity")
        if self.phase not in {"launching", "starting", "ready"}:
            raise ValueError("invalid ownership phase")
        if self.phase == "launching":
            if self.identity is not None or self.ready_at is not None:
                raise ValueError("launching record contains a process identity")
            return
        identity = self.identity
        if (
            identity is None
            or identity.pid <= 1
            or identity.process_group_id != identity.pid
            or identity.start_ticks <= 0
            or not identity.boot_id
            or not re.fullmatch(r"[0-9a-f]{64}", identity.command_sha256)
        ):
            raise ValueError("invalid owned process identity")
        if (self.phase == "ready") != (self.ready_at is not None):
            raise ValueError("ready phase and timestamp disagree")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": _FORMAT_VERSION,
            "generation": self.generation,
            "phase": self.phase,
            "configuration_sha256": self.configuration_sha256,
            "created_at": self.created_at.isoformat(),
            "identity": None if self.identity is None else self.identity.as_dict(),
            "ready_at": None if self.ready_at is None else self.ready_at.isoformat(),
        }


def ensure_jupyter_agent(
    *,
    config_path: Path,
    site_environment_path: Path | None,
    state_directory: Path,
    startup_timeout_seconds: float,
    stability_seconds: float,
) -> OwnershipRecord:
    """Return a proven live generation, starting it when no owned process exists."""

    if startup_timeout_seconds <= 0 or stability_seconds < 0:
        raise ValueError("bootstrap timeouts are invalid")
    state_root = _prepare_private_directory(state_directory)
    config_bytes = _read_protected_file(config_path, description="site configuration")
    site_environment: dict[str, str] = {}
    environment_bytes = b""
    if site_environment_path is not None:
        environment_bytes = _read_protected_file(
            site_environment_path,
            description="site environment",
        )
        site_environment = _parse_site_environment(environment_bytes)
    configuration_sha256 = hashlib.sha256(config_bytes + b"\0" + environment_bytes).hexdigest()
    state_path = state_root / "agent.json"
    deadline = time.monotonic() + startup_timeout_seconds

    with _locked_file(state_root / "startup.lock"):
        existing = _read_record(state_path)
        if existing is not None:
            existing = _resolve_existing(
                existing,
                state_path=state_path,
                configuration_sha256=configuration_sha256,
                deadline=deadline,
                stability_seconds=stability_seconds,
            )
            if existing is not None:
                return existing

        generation = uuid4().hex
        launch = OwnershipRecord(
            generation=generation,
            phase="launching",
            configuration_sha256=configuration_sha256,
            created_at=datetime.now(UTC),
        )
        _write_record(state_path, launch)
        environment = _agent_environment(site_environment)
        environment[_CHILD_MODE] = "1"
        environment[_CHILD_GENERATION] = generation
        arguments = _child_arguments(
            config_path=config_path,
            site_environment_path=site_environment_path,
            state_directory=state_root,
            startup_timeout_seconds=startup_timeout_seconds,
            stability_seconds=stability_seconds,
        )
        log_handle = _open_private_log(state_root / "agent.log")
        try:
            try:
                process = _spawn_child(arguments, environment, log_handle)
            except BaseException:
                _delete_record_if_generation(state_path, generation)
                raise
        finally:
            log_handle.close()
        return _wait_for_generation(
            state_path=state_path,
            generation=generation,
            configuration_sha256=configuration_sha256,
            expected_pid=process.pid,
            deadline=deadline,
            stability_seconds=stability_seconds,
            process=process,
        )


def stop_owned_jupyter_agent(
    *,
    config_path: Path,
    site_environment_path: Path | None,
    state_directory: Path,
    shutdown_timeout_seconds: float,
) -> bool:
    """Gracefully stop only the exact detached generation recorded by this wrapper.

    The pidfd pins the kernel process object before signalling, closing the PID
    reuse race between validating the durable identity and sending SIGTERM.
    There is deliberately no numeric-PID fallback and no SIGKILL escalation.
    The agent's SIGTERM handler drains accepted work and stops only its
    independently proven, Topodrome-owned Ray process group.
    """

    if shutdown_timeout_seconds <= 0:
        raise ValueError("shutdown timeout must be positive")
    state_root = _prepare_private_directory(state_directory)
    config_bytes = _read_protected_file(config_path, description="site configuration")
    environment_bytes = b""
    if site_environment_path is not None:
        environment_bytes = _read_protected_file(
            site_environment_path,
            description="site environment",
        )
    configuration_sha256 = hashlib.sha256(config_bytes + b"\0" + environment_bytes).hexdigest()
    state_path = state_root / "agent.json"

    with _locked_file(state_root / "startup.lock"):
        record = _read_record(state_path)
        if record is None:
            return False
        if record.configuration_sha256 != configuration_sha256:
            if record.identity is not None and _process_matches(record.identity):
                raise JupyterAgentBootstrapError(
                    "an owned agent is running with a different protected configuration"
                )
            _delete_record_if_generation(state_path, record.generation)
            return False
        identity = record.identity
        if identity is None:
            # No process identity was ever claimed. Removing the abandoned
            # launch intent cannot signal a process.
            _delete_record_if_generation(state_path, record.generation)
            return False

        pidfd = _open_verified_pidfd(identity)
        if pidfd is None:
            _delete_record_if_identity(state_path, record.generation, identity)
            return False
        try:
            _send_pidfd_signal(pidfd, signal.SIGTERM)
            deadline = time.monotonic() + shutdown_timeout_seconds
            while not _pidfd_has_exited(pidfd):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise JupyterAgentBootstrapError(
                        "the owned agent did not stop within the shutdown timeout"
                    )
                time.sleep(min(0.05, remaining))
        finally:
            os.close(pidfd)
        _delete_record_if_identity(state_path, record.generation, identity)
        return True


def _resolve_existing(
    record: OwnershipRecord,
    *,
    state_path: Path,
    configuration_sha256: str,
    deadline: float,
    stability_seconds: float,
) -> OwnershipRecord | None:
    if record.configuration_sha256 != configuration_sha256:
        if record.identity is not None and _process_matches(record.identity):
            raise JupyterAgentBootstrapError(
                "an owned agent is running with a different protected configuration"
            )
        _delete_record_if_generation(state_path, record.generation)
        return None
    if record.phase == "launching":
        claim_deadline = min(
            deadline,
            time.monotonic()
            + max(
                0.0,
                _LAUNCH_CLAIM_GRACE_SECONDS
                - (datetime.now(UTC) - record.created_at).total_seconds(),
            ),
        )
        while time.monotonic() < claim_deadline:
            time.sleep(0.05)
            current = _read_record(state_path)
            if current is None or current.generation != record.generation:
                break
            if current.phase != "launching":
                record = current
                break
        if record.phase == "launching":
            current = _read_record(state_path)
            if current is not None and current.generation == record.generation:
                _delete_record_if_generation(state_path, record.generation)
            return None
    assert record.identity is not None
    if not _process_matches(record.identity):
        _delete_record_if_generation(state_path, record.generation)
        return None
    if record.phase == "ready":
        return _stabilize(record, stability_seconds=stability_seconds)
    return _wait_for_generation(
        state_path=state_path,
        generation=record.generation,
        configuration_sha256=configuration_sha256,
        expected_pid=record.identity.pid,
        deadline=deadline,
        stability_seconds=stability_seconds,
    )


def _wait_for_generation(
    *,
    state_path: Path,
    generation: str,
    configuration_sha256: str,
    expected_pid: int,
    deadline: float,
    stability_seconds: float,
    process: subprocess.Popen[bytes] | None = None,
) -> OwnershipRecord:
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise JupyterAgentBootstrapError("the detached agent exited during startup")
        record = _read_record(state_path)
        if record is None:
            time.sleep(0.05)
            continue
        if record.generation != generation or record.configuration_sha256 != configuration_sha256:
            raise JupyterAgentBootstrapError("the agent ownership generation changed")
        if record.identity is not None:
            if record.identity.pid != expected_pid or not _process_matches(record.identity):
                raise JupyterAgentBootstrapError("the owned agent process identity changed")
            if record.phase == "ready":
                return _stabilize(record, stability_seconds=stability_seconds)
        time.sleep(0.05)
    raise JupyterAgentBootstrapError("the detached agent did not become locally ready in time")


def _stabilize(record: OwnershipRecord, *, stability_seconds: float) -> OwnershipRecord:
    assert record.identity is not None
    deadline = time.monotonic() + stability_seconds
    while time.monotonic() < deadline:
        if not _process_matches(record.identity):
            raise JupyterAgentBootstrapError("the detached agent exited during stabilization")
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    if not _process_matches(record.identity):
        raise JupyterAgentBootstrapError("the detached agent exited during stabilization")
    return record


def _child_arguments(
    *,
    config_path: Path,
    site_environment_path: Path | None,
    state_directory: Path,
    startup_timeout_seconds: float,
    stability_seconds: float,
) -> list[str]:
    arguments = [
        sys.executable,
        "-m",
        "topodrome.deployment.jupyter_agent",
        "--config",
        str(config_path.expanduser().absolute()),
        "--state-dir",
        str(state_directory),
        "--startup-timeout-seconds",
        str(startup_timeout_seconds),
        "--stability-seconds",
        str(stability_seconds),
    ]
    if site_environment_path is not None:
        arguments.extend(["--site-env-file", str(site_environment_path.expanduser().absolute())])
    return arguments


def _spawn_child(
    arguments: Sequence[str],
    environment: Mapping[str, str],
    log_handle: IO[bytes],
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # noqa: S603 - fixed interpreter/module and validated paths
        list(arguments),
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=dict(environment),
        start_new_session=True,
        close_fds=True,
    )


def _run_child(
    *,
    config_path: Path,
    site_environment_path: Path | None,
    state_directory: Path,
) -> None:
    generation = os.environ.get(_CHILD_GENERATION, "")
    if not re.fullmatch(r"[0-9a-f]{32}", generation):
        raise JupyterAgentBootstrapError("the child launch generation is invalid")
    state_root = _prepare_private_directory(state_directory)
    state_path = state_root / "agent.json"
    config_bytes = _read_protected_file(config_path, description="site configuration")
    environment_bytes = b""
    site_environment: dict[str, str] = {}
    if site_environment_path is not None:
        environment_bytes = _read_protected_file(
            site_environment_path,
            description="site environment",
        )
        site_environment = _parse_site_environment(environment_bytes)
    configuration_sha256 = hashlib.sha256(config_bytes + b"\0" + environment_bytes).hexdigest()
    agent_environment = _agent_environment(site_environment)
    os.environ.clear()
    os.environ.update(agent_environment)
    os.environ["TOPODROME_SITE_CONFIG"] = str(config_path.expanduser().absolute())

    with _locked_file(state_root / "agent.lock", blocking=False):
        launch = _read_record(state_path)
        if (
            launch is None
            or launch.phase != "launching"
            or launch.generation != generation
            or launch.configuration_sha256 != configuration_sha256
        ):
            raise JupyterAgentBootstrapError("the launch intent is no longer current")
        identity = _capture_process_identity(os.getpid())
        starting = OwnershipRecord(
            generation=generation,
            phase="starting",
            configuration_sha256=configuration_sha256,
            created_at=launch.created_at,
            identity=identity,
        )
        _write_record(state_path, starting)
        application: AgentApplication | None = None
        try:
            settings = _site_settings_from_protected_bytes(config_bytes)
            application = build_application(settings)
            ready = OwnershipRecord(
                generation=generation,
                phase="ready",
                configuration_sha256=configuration_sha256,
                created_at=launch.created_at,
                identity=identity,
                ready_at=datetime.now(UTC),
            )
            _write_record(state_path, ready)
            asyncio.run(application.run())
        finally:
            _delete_record_if_identity(state_path, generation, identity)


def _agent_environment(site_environment: Mapping[str, str]) -> dict[str, str]:
    environment = minimal_ray_head_environment(os.environ)
    for key, value in os.environ.items():
        if _site_environment_key_allowed(key):
            environment[key] = value
    environment.update(site_environment)
    for forbidden in (
        "TOPODROME_AUTODL_TOKEN",
        "TOPODROME_TEST_JUPYTER_URL",
        "TOPODROME_TEST_JUPYTER_TOKEN",
        "TOPODROME_JUPYTER_BOOTSTRAP_URL",
        "TOPODROME_JUPYTER_BOOTSTRAP_TOKEN",
        "TOPODROME_MANAGED_JUPYTER_SERVER_TOKEN",
        "JUPYTER_TOKEN",
    ):
        environment.pop(forbidden, None)
    return environment


def _site_environment_key_allowed(key: str) -> bool:
    return (
        key.startswith("TOPODROME_SITE_")
        or key in {"TOPODROME_HF_TOKEN", "HF_TOKEN"}
    )


def _parse_site_environment(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError:
        raise ValueError("site environment is not UTF-8") from None
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError("site environment contains an invalid assignment")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _ENVIRONMENT_NAME.fullmatch(key) or not _site_environment_key_allowed(key):
            raise ValueError("site environment contains an unsupported key")
        if key in values:
            raise ValueError("site environment contains a duplicate key")
        lexer = shlex.shlex(raw_value, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        try:
            parts = list(lexer)
        except ValueError:
            raise ValueError("site environment contains invalid quoting") from None
        if len(parts) > 1:
            raise ValueError("site environment values containing spaces must be quoted")
        value = parts[0] if parts else ""
        if "\0" in value or "\n" in value or "\r" in value:
            raise ValueError("site environment contains an invalid value")
        values[key] = value
    return values


def _site_settings_from_protected_bytes(payload: bytes) -> SiteSettings:
    """Validate exactly the bytes whose protected identity was recorded."""

    try:
        raw = yaml.safe_load(payload)
    except (UnicodeError, yaml.YAMLError):
        raise ValueError("protected site configuration is invalid YAML") from None
    if not isinstance(raw, dict):
        raise ValueError("protected site configuration must contain a mapping")
    # Match SiteSettings.from_yaml: null identity hints are placeholders and
    # must not mask protected environment values supplied for a recreated site.
    for field_name in ("provider_instance_id", "region"):
        if raw.get(field_name) is None:
            raw.pop(field_name, None)
    return SiteSettings(**raw)


def _prepare_private_directory(path: Path) -> Path:
    target = path.expanduser().absolute()
    target.mkdir(parents=True, mode=0o700, exist_ok=True)
    metadata = target.lstat()
    if (
        target.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise PermissionError("bootstrap state directory must be private and owner-controlled")
    return target


def _read_protected_file(path: Path, *, description: str) -> bytes:
    target = path.expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError:
        raise PermissionError(f"protected {description} is unavailable") from None
    try:
        metadata = os.fstat(descriptor)
        parent = target.parent.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or parent.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or stat.S_IMODE(parent.st_mode) & 0o022
        ):
            raise PermissionError(
                f"protected {description} must be owner-controlled and mode 0600 or stricter"
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@contextmanager
def _locked_file(path: Path, *, blocking: bool = True) -> Iterator[int]:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise PermissionError("bootstrap lock must be a private owner-controlled file")
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError:
            raise JupyterAgentBootstrapError("another owned agent generation is active") from None
        yield descriptor
    finally:
        os.close(descriptor)


def _open_private_log(path: Path) -> IO[bytes]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        os.close(descriptor)
        raise PermissionError("agent log must be a private owner-controlled file")
    return os.fdopen(descriptor, "ab", buffering=0)


def _read_record(path: Path) -> OwnershipRecord | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError:
        raise JupyterAgentBootstrapError("agent ownership state is unavailable") from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise PermissionError("agent ownership state is not private")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            raw = json.load(handle)
        return OwnershipRecord.from_mapping(raw)
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        raise JupyterAgentBootstrapError("agent ownership state is invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_record(path: Path, record: OwnershipRecord) -> None:
    record.validate()
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(record.as_dict(), handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _delete_record_if_generation(path: Path, generation: str) -> None:
    current = _read_record(path)
    if current is None or current.generation != generation:
        return
    path.unlink()
    _fsync_directory(path.parent)


def _delete_record_if_identity(
    path: Path,
    generation: str,
    identity: ProcessIdentity,
) -> None:
    current = _read_record(path)
    if current is None or current.generation != generation or current.identity != identity:
        return
    path.unlink()
    _fsync_directory(path.parent)


def _capture_process_identity(pid: int) -> ProcessIdentity:
    process_path = Path("/proc") / str(pid)
    metadata = process_path.stat()
    process_group_id = os.getpgid(pid)
    if metadata.st_uid != os.geteuid() or process_group_id != pid:
        raise JupyterAgentBootstrapError("detached process ownership could not be proven")
    return ProcessIdentity(
        pid=pid,
        process_group_id=process_group_id,
        start_ticks=_process_start_ticks(pid),
        boot_id=_boot_id(),
        command_sha256=_process_command_sha256(pid),
    )


def _process_matches(identity: ProcessIdentity) -> bool:
    try:
        process_path = Path("/proc") / str(identity.pid)
        return (
            _boot_id() == identity.boot_id
            and process_path.stat().st_uid == os.geteuid()
            and os.getpgid(identity.pid) == identity.process_group_id == identity.pid
            and _process_start_ticks(identity.pid) == identity.start_ticks
            and _process_command_sha256(identity.pid) == identity.command_sha256
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def _open_verified_pidfd(identity: ProcessIdentity) -> int | None:
    """Pin and re-verify the persisted process before any signal is possible."""

    if not _process_matches(identity):
        return None
    if not _pidfd_supported():
        raise JupyterAgentBootstrapError("safe owned-agent signaling requires Linux pidfd support")
    try:
        pidfd = _pidfd_open(identity.pid)
    except ProcessLookupError:
        return None
    except OSError:
        raise JupyterAgentBootstrapError(
            "the owned agent process descriptor could not be opened"
        ) from None
    if not _process_matches(identity):
        os.close(pidfd)
        return None
    return pidfd


def _send_pidfd_signal(pidfd: int, signum: int) -> None:
    sender = getattr(signal, "pidfd_send_signal", None)
    try:
        if callable(sender):
            sender(pidfd, signum, None, 0)
        else:
            _pidfd_send_signal_syscall(pidfd, signum)
    except ProcessLookupError:
        # A pidfd cannot retarget a replacement process. The owned generation
        # already exited, so this is the desired terminal condition.
        return
    except OSError:
        raise JupyterAgentBootstrapError("the owned agent process could not be signalled") from None


def _pidfd_supported() -> bool:
    has_open = callable(getattr(os, "pidfd_open", None))
    has_send = callable(getattr(signal, "pidfd_send_signal", None))
    return (has_open and has_send) or _pidfd_syscall_numbers() is not None


def _pidfd_syscall_numbers() -> tuple[int, int] | None:
    if sys.platform != "linux":
        return None
    try:
        machine = os.uname().machine.lower()
    except OSError:
        return None
    return _PIDFD_SYSCALLS.get(machine)


def _pidfd_open(pid: int) -> int:
    opener = getattr(os, "pidfd_open", None)
    if callable(opener):
        return int(opener(pid, 0))
    numbers = _pidfd_syscall_numbers()
    if numbers is None:  # pragma: no cover - checked by _pidfd_supported
        raise OSError("pidfd_open is unavailable")
    return _linux_syscall_fd(numbers[0], pid, 0)


def _pidfd_send_signal_syscall(pidfd: int, signum: int) -> None:
    numbers = _pidfd_syscall_numbers()
    if numbers is None:  # pragma: no cover - checked by _pidfd_supported
        raise OSError("pidfd_send_signal is unavailable")
    _linux_syscall(
        numbers[1],
        ctypes.c_int(pidfd),
        ctypes.c_int(signum),
        ctypes.c_void_p(),
        ctypes.c_uint(0),
    )


def _linux_syscall_fd(number: int, *arguments: int) -> int:
    return _linux_syscall(number, *(ctypes.c_long(int(value)) for value in arguments))


def _linux_syscall(number: int, *arguments: object) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = int(syscall(ctypes.c_long(number), *arguments))
    if result == -1:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return result


def _pidfd_has_exited(pidfd: int) -> bool:
    try:
        readable, _, _ = select.select((pidfd,), (), (), 0)
    except OSError:
        raise JupyterAgentBootstrapError(
            "the owned agent process descriptor could not be polled"
        ) from None
    return bool(readable)


def _process_start_ticks(pid: int) -> int:
    raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    return int(raw.rsplit(")", 1)[1].split()[19])


def _process_command_sha256(pid: int) -> str:
    command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    if not command:
        raise ValueError("process command line is empty")
    return hashlib.sha256(command).hexdigest()


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp is not a string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks a timezone")
    return parsed.astimezone(UTC)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _default_state_directory() -> Path:
    configured = os.environ.get("TOPODROME_JUPYTER_AGENT_STATE_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".local" / "state" / "topodrome" / "jupyter-agent"


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start or recover one detached Topodrome agent on a Jupyter site"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--site-env-file", type=Path)
    parser.add_argument("--state-dir", type=Path, default=_default_state_directory())
    parser.add_argument("--startup-timeout-seconds", type=float, default=75.0)
    parser.add_argument("--stability-seconds", type=float, default=2.0)
    parser.add_argument(
        "--stop-owned",
        action="store_true",
        help="gracefully stop only the exact agent generation in the private state record",
    )
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=150.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _arguments(argv)
    try:
        if os.environ.get(_CHILD_MODE) == "1":
            if arguments.stop_owned:
                raise JupyterAgentBootstrapError(
                    "a detached agent child cannot invoke the stop operation"
                )
            _run_child(
                config_path=arguments.config,
                site_environment_path=arguments.site_env_file,
                state_directory=arguments.state_dir,
            )
            return
        if arguments.stop_owned:
            stopped = stop_owned_jupyter_agent(
                config_path=arguments.config,
                site_environment_path=arguments.site_env_file,
                state_directory=arguments.state_dir,
                shutdown_timeout_seconds=arguments.shutdown_timeout_seconds,
            )
            print(STOPPED_MARKER if stopped else ABSENT_MARKER, flush=True)
            return
        ensure_jupyter_agent(
            config_path=arguments.config,
            site_environment_path=arguments.site_env_file,
            state_directory=arguments.state_dir,
            startup_timeout_seconds=arguments.startup_timeout_seconds,
            stability_seconds=arguments.stability_seconds,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        print(
            f"TOPODROME_AGENT_BOOTSTRAP_ERROR={type(exc).__name__}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from None
    print(READY_MARKER, flush=True)


__all__ = [
    "ABSENT_MARKER",
    "READY_MARKER",
    "STOPPED_MARKER",
    "JupyterAgentBootstrapError",
    "OwnershipRecord",
    "ProcessIdentity",
    "ensure_jupyter_agent",
    "main",
    "stop_owned_jupyter_agent",
]


if __name__ == "__main__":  # pragma: no cover - module child entrypoint
    main()
