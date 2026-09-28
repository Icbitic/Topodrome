"""Lifecycle boundary for the one Ray cluster local to a physical site."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import signal
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from topodrome.runtime.environment import minimal_ray_head_environment

# Ray appends a timestamped session directory and ``sockets/plasma_store`` to
# its temp root. Keep ample space below Linux's 107-byte AF_UNIX limit rather
# than depending on the current Ray session-name implementation.
_RAY_TEMP_DIR_BYTE_BUDGET = 32
# The packaged ``ray`` launcher is a tiny shell wrapper which execs the final
# Python process in-place.  Capturing ``/proc/<pid>/cmdline`` before that exec
# makes an otherwise correct durable identity stale as soon as the wrapper
# advances.  A short, bounded dwell after the Python executable appears closes
# that launcher race without delaying normal Ray startup materially.
_RAY_IDENTITY_POLL_SECONDS = 0.02
_RAY_IDENTITY_STABILITY_SECONDS = 0.1


def gpu_resource_name(gpu_uuid: str) -> str:
    """Return the Ray custom-resource name for one exact physical GPU."""

    normalized = re.sub(r"[^a-z0-9]+", "_", gpu_uuid.lower()).strip("_")
    if not normalized:
        raise ValueError("GPU UUID must contain an alphanumeric character")
    return f"topodrome_gpu_{normalized}"


class RayLifecycle(Protocol):
    async def ensure_running(self) -> bool:
        """Ensure the cluster is reachable; return true if it was started."""

    async def is_running(self) -> bool: ...

    async def stop(self) -> bool:
        """Stop only a proven-owned head; report whether absence is confirmed."""


class ProcessOwnershipSink(Protocol):
    """Minimal process registry boundary used by the managed Ray lifecycle."""

    def register(
        self,
        pid: int,
        *,
        lease_id: str | None = None,
        actor_id: str | None = None,
        start_ticks: int | None = None,
    ) -> None: ...

    def unregister(self, pid: int) -> None: ...


@dataclass(frozen=True, slots=True)
class RayClusterConfig:
    gpu_uuids: tuple[str, ...]
    ray_executable: str = "ray"
    address: str = "127.0.0.1:6379"
    node_ip_address: str = "127.0.0.1"
    temp_dir: Path | None = None
    short_temp_root: Path | None = Path("/tmp")  # noqa: S108 -- private validated alias
    command_timeout_seconds: float = 30.0
    ray_command_prefix: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("GPU UUIDs must be unique")
        if self.temp_dir is not None:
            object.__setattr__(self, "temp_dir", Path(self.temp_dir))
        if self.short_temp_root is not None:
            object.__setattr__(self, "short_temp_root", Path(self.short_temp_root))
        if not self.ray_executable or "\0" in self.ray_executable:
            raise ValueError("Ray executable must be a non-empty command")
        prefix = self.ray_command_prefix
        if prefix is not None and (
            not prefix or any(not argument or "\0" in argument for argument in prefix)
        ):
            raise ValueError("Ray command prefix must contain non-empty arguments")
        if self.command_timeout_seconds <= 0:
            raise ValueError("Ray command timeout must be positive")

    @property
    def custom_resources(self) -> dict[str, int]:
        return {gpu_resource_name(uuid): 1 for uuid in self.gpu_uuids}

    @property
    def command_prefix(self) -> tuple[str, ...]:
        """Return an explicit launcher or the backward-compatible executable."""

        return self.ray_command_prefix or (self.ray_executable,)


class SubprocessRayLifecycle:
    """Reconnect to, start, and cleanly stop a site-local Ray head.

    Commands are constructed as argument lists, never through a shell.  Ray is
    told it has zero generic GPUs: Topodrome actors must request the explicit
    per-UUID custom resource after the site agent grants a physical lease.
    """

    def __init__(
        self,
        config: RayClusterConfig,
        *,
        process_ownership: ProcessOwnershipSink | None = None,
    ) -> None:
        self.config = config
        self._process_ownership = process_ownership
        self._lock = asyncio.Lock()
        self._started_here = False
        self._owned_head: _OwnedRayHead | None = None
        self._registered_head: _OwnedRayHead | None = None
        self._owned_process: asyncio.subprocess.Process | None = None
        self._temp_alias: Path | None = None
        self._temp_alias_target: Path | None = None

    async def is_running(self) -> bool:
        # ``ray status`` retries its GCS connection internally and can consume
        # the complete command timeout when a cold site has no head yet.  A
        # bounded socket probe makes the expected cold-start path immediate.
        if not await self._endpoint_reachable():
            return False
        try:
            result = await self._run(
                *self.config.command_prefix,
                "status",
                "--address",
                self.config.address,
                check=False,
            )
        except _RayCommandTimeoutError:
            return False
        return result == 0

    async def _endpoint_reachable(self) -> bool:
        host, separator, raw_port = self.config.address.rpartition(":")
        if not separator or not host or not raw_port.isdigit():
            raise ValueError("Ray address must be host:port when the agent manages the cluster")
        try:
            _reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, int(raw_port)),
                timeout=min(1.0, self.config.command_timeout_seconds),
            )
        except (OSError, TimeoutError):
            return False
        writer.close()
        with suppress(Exception):
            await asyncio.wait_for(
                writer.wait_closed(),
                timeout=min(1.0, self.config.command_timeout_seconds),
            )
        return True

    async def ensure_running(self) -> bool:
        async with self._lock:
            effective_temp_dir = self._prepare_temp_dir()
            owned = self._load_owned_head()
            if await self.is_running():
                if owned is not None and _owned_head_matches(owned):
                    self._owned_head = owned
                    self._register_owned_head(owned)
                else:
                    # A reachable head without an exact live ownership record
                    # is adopted for use only. It must never be stopped by us.
                    self._owned_head = None
                    self._unregister_owned_head()
                return False
            _host, separator, port = self.config.address.rpartition(":")
            if not separator or not port.isdigit():
                raise ValueError("Ray address must be host:port when the agent manages the cluster")
            if effective_temp_dir is None:
                raise RuntimeError(
                    "managed Ray startup requires a persistent temp_dir for ownership state"
                )
            if owned is not None:
                if _owned_head_matches(owned):
                    await self._terminate_owned_head(owned)
                    self._remove_owned_head(owned)
                elif _process_group_exists(owned.process_group_id):
                    raise RuntimeError(
                        "stale Ray ownership record names an unverified live process group"
                    )
                else:
                    self._remove_owned_head(owned)
            arguments = [
                *self.config.command_prefix,
                "start",
                "--head",
                "--block",
                f"--port={port}",
                "--num-gpus=0",
                "--disable-usage-stats",
                "--include-dashboard=false",
                "--dashboard-host=127.0.0.1",
                f"--node-ip-address={self.config.node_ip_address}",
                f"--resources={json.dumps(self.config.custom_resources, sort_keys=True)}",
            ]
            if effective_temp_dir is not None:
                arguments.append(f"--temp-dir={effective_temp_dir}")
            try:
                process = await self._spawn_owned_head(
                    tuple(arguments),
                    log_directory=self._temp_target(),
                )
                self._owned_process = process
                spawned = _capture_owned_head(
                    process.pid,
                    address=self.config.address,
                    temp_dir=self._temp_target(),
                )
                owned = await self._wait_for_stable_owned_head(process, spawned)
                self._write_owned_head(owned)
                self._owned_head = owned
                self._owned_process = process
                self._started_here = True
                await self._wait_for_owned_start(process)
                self._register_owned_head(owned)
            except BaseException:
                if self._owned_head is not None and _owned_head_matches(self._owned_head):
                    with suppress(BaseException):
                        await self._terminate_owned_head(self._owned_head)
                elif self._owned_process is not None and self._owned_process.returncode is None:
                    # The child was just created by this process with a fresh
                    # session, so its process group is exact even if identity
                    # capture failed before the durable record was written.
                    with suppress(ProcessLookupError):
                        os.killpg(self._owned_process.pid, signal.SIGKILL)
                    with suppress(BaseException):
                        await asyncio.wait_for(self._owned_process.wait(), timeout=1.0)
                if self._owned_head is not None:
                    with suppress(BaseException):
                        self._remove_owned_head(self._owned_head)
                self._owned_head = None
                self._owned_process = None
                self._started_here = False
                self._unregister_owned_head()
                self._cleanup_temp_alias()
                raise
            return True

    async def stop(self) -> bool:
        async with self._lock:
            # Reload the durable identity so a restarted site agent can still
            # stop the exact ``ray start --block`` process group it created.
            owned = self._owned_head or self._load_owned_head()
            running = await self.is_running()
            if owned is None:
                # No ownership proof: a reachable head is foreign/adopted, and
                # even an unreachable legacy head may have orphaned workers.
                self._unregister_owned_head()
                return False
            if not _owned_head_matches(owned):
                if _process_group_exists(owned.process_group_id):
                    return False
                self._remove_owned_head(owned)
                self._owned_head = None
                self._owned_process = None
                self._started_here = False
                self._unregister_owned_head()
                self._cleanup_temp_alias()
                return not running
            await self._terminate_owned_head(owned)
            if await self._endpoint_reachable():
                # Never escalate to the global ``ray stop`` command. A live
                # endpoint after exact group termination is treated as foreign.
                raise RuntimeError("a Ray endpoint remains after owned-head termination")
            self._remove_owned_head(owned)
            self._owned_head = None
            self._owned_process = None
            self._started_here = False
            self._unregister_owned_head()
            self._cleanup_temp_alias()
            return True

    def _register_owned_head(self, owned: _OwnedRayHead) -> None:
        """Classify managed Ray descendants before their first CUDA context.

        The root identity comes from the same private, boot-bound procfs record
        that authorizes lifecycle termination.  ``ProcessOwnershipRegistry``
        validates its start ticks again whenever it walks a GPU worker's parent
        chain, so a recycled PID can never turn a foreign process into an owned
        one.
        """

        registry = self._process_ownership
        if registry is None or self._registered_head == owned:
            return
        self._unregister_owned_head()
        registry.register(
            owned.pid,
            actor_id="managed-ray-head",
            start_ticks=owned.start_ticks,
        )
        self._registered_head = owned

    def _unregister_owned_head(self) -> None:
        registered, self._registered_head = self._registered_head, None
        if registered is not None and self._process_ownership is not None:
            self._process_ownership.unregister(registered.pid)

    async def _spawn_owned_head(
        self,
        arguments: tuple[str, ...],
        *,
        log_directory: Path,
    ) -> asyncio.subprocess.Process:
        log_path = log_directory / "topodrome-ray-head.log"
        log_handle = log_path.open("ab", buffering=0)
        try:
            log_path.chmod(0o600)
            return await asyncio.create_subprocess_exec(
                *arguments,
                stdout=log_handle,
                stderr=log_handle,
                env=minimal_ray_head_environment(),
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Ray executable was not found: {self.config.command_prefix[0]}"
            ) from exc
        finally:
            log_handle.close()

    async def _wait_for_owned_start(self, process: asyncio.subprocess.Process) -> None:
        deadline = asyncio.get_running_loop().time() + self.config.command_timeout_seconds
        while True:
            if process.returncode is not None:
                raise RuntimeError(
                    f"owned Ray head exited during startup with {process.returncode}"
                )
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise _RayCommandTimeoutError("owned Ray head startup timed out")
            if await self._endpoint_reachable():
                try:
                    result = await self._run(
                        *self.config.command_prefix,
                        "status",
                        "--address",
                        self.config.address,
                        check=False,
                        timeout_seconds=remaining,
                    )
                except _RayCommandTimeoutError:
                    result = 1
                if result == 0:
                    return
            await asyncio.sleep(min(0.1, remaining))

    async def _wait_for_stable_owned_head(
        self,
        process: asyncio.subprocess.Process,
        spawned: _OwnedRayHead,
    ) -> _OwnedRayHead:
        """Capture the durable identity after the trusted launcher has exec'd.

        ``create_subprocess_exec`` returns while a script's interpreter may
        still be running its wrapper.  PID, process group, boot ID, and procfs
        start ticks survive ``exec``, while the command line deliberately does
        not.  Keep the immutable spawn identity as the authority, then commit
        the command digest only after the same process is a Python process and
        its argv has remained unchanged for a short dwell.  No ownership record
        is published for an exited, replaced, or perpetually changing process.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.command_timeout_seconds
        stable: _OwnedRayHead | None = None
        stable_since = 0.0
        while True:
            if process.returncode is not None:
                raise RuntimeError(
                    "owned Ray head exited before its ownership identity stabilized"
                )
            try:
                current = _capture_owned_head(
                    process.pid,
                    address=self.config.address,
                    temp_dir=self._temp_target(),
                )
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError) as exc:
                raise RuntimeError(
                    "owned Ray head identity became unavailable before it stabilized"
                ) from exc
            if not _same_spawn_identity(spawned, current):
                raise RuntimeError(
                    "owned Ray head process identity changed before it stabilized"
                )

            now = loop.time()
            if _process_executable_is_python(current.pid):
                if stable is None or stable.command_sha256 != current.command_sha256:
                    stable = current
                    stable_since = now
                elif now - stable_since >= _RAY_IDENTITY_STABILITY_SECONDS:
                    return current
            else:
                stable = None
                stable_since = 0.0

            remaining = deadline - now
            if remaining <= 0:
                raise _RayCommandTimeoutError(
                    "owned Ray head ownership identity did not stabilize"
                )
            await asyncio.sleep(min(_RAY_IDENTITY_POLL_SECONDS, remaining))

    async def _terminate_owned_head(self, owned: _OwnedRayHead) -> None:
        if not _owned_head_matches(owned):
            raise RuntimeError("Ray head ownership changed before termination")
        os.killpg(owned.process_group_id, signal.SIGTERM)
        deadline = asyncio.get_running_loop().time() + self.config.command_timeout_seconds
        while _process_group_exists(owned.process_group_id):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                # Sending a second signal is safe only while the original group
                # leader still has the persisted PID/start identity.
                if not _owned_head_matches(owned):
                    raise RuntimeError(
                        "owned Ray leader exited but its process group remains"
                    )
                os.killpg(owned.process_group_id, signal.SIGKILL)
                kill_deadline = (
                    asyncio.get_running_loop().time()
                    + min(5.0, self.config.command_timeout_seconds)
                )
                while _process_group_exists(owned.process_group_id):
                    if asyncio.get_running_loop().time() >= kill_deadline:
                        raise RuntimeError("owned Ray process group did not terminate")
                    await asyncio.sleep(0.05)
                break
            await asyncio.sleep(min(0.05, remaining))
        process = self._owned_process
        if process is not None and process.pid == owned.pid:
            with suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=1.0)

    def _temp_target(self) -> Path:
        configured = self.config.temp_dir
        if configured is None:
            raise RuntimeError("Ray temp_dir is required for managed ownership")
        return _absolute_path(configured, label="Ray temp directory")

    def _ownership_path(self) -> Path:
        return self._temp_target() / ".topodrome-owned-head.json"

    def _load_owned_head(self) -> _OwnedRayHead | None:
        if self.config.temp_dir is None:
            return None
        path = self._ownership_path()
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise RuntimeError("Ray ownership record is not a user-owned regular file")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RuntimeError("Ray ownership record permissions are not private")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            owned = _OwnedRayHead.from_mapping(raw)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("Ray ownership record is invalid") from exc
        if owned.address != self.config.address or owned.temp_dir != self._temp_target():
            raise RuntimeError("Ray ownership record belongs to another managed head")
        return owned

    def _write_owned_head(self, owned: _OwnedRayHead) -> None:
        path = self._ownership_path()
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        payload = json.dumps(owned.as_dict(), sort_keys=True).encode()
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise RuntimeError("Ray ownership record was created concurrently") from exc
            temporary.unlink()
            _fsync_directory(path.parent)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()

    def _remove_owned_head(self, expected: _OwnedRayHead) -> None:
        path = self._ownership_path()
        current = self._load_owned_head()
        if current is None:
            return
        if current != expected:
            raise RuntimeError("Ray ownership record changed before cleanup")
        path.unlink()
        _fsync_directory(path.parent)

    def _prepare_temp_dir(self) -> Path | None:
        configured = self.config.temp_dir
        if configured is None:
            return None
        target = _absolute_path(configured, label="Ray temp directory")
        _ensure_owned_directory(target, label="Ray temp directory", create_parents=True)
        if len(os.fsencode(target)) <= _RAY_TEMP_DIR_BYTE_BUDGET:
            return target

        short_root = self.config.short_temp_root
        if short_root is None:
            raise RuntimeError(
                "Ray temp directory is too long for AF_UNIX sockets and no "
                "ray_short_temp_root is configured"
            )
        base = _absolute_path(short_root, label="Ray short-temp root")
        _validate_alias_base(base)
        private_root = base / f".tdr-{os.geteuid()}"
        _ensure_owned_directory(
            private_root,
            label="Ray private short-temp root",
            create_parents=False,
            repair_permissions=False,
        )
        digest = hashlib.sha256(os.fsencode(target)).hexdigest()[:10]
        alias = private_root / f"r-{digest}"
        if len(os.fsencode(alias)) > _RAY_TEMP_DIR_BYTE_BUDGET:
            raise RuntimeError(
                "configured ray_short_temp_root cannot produce a sufficiently short "
                "AF_UNIX socket path"
            )
        _ensure_exact_symlink(alias, target)
        self._temp_alias = alias
        self._temp_alias_target = target
        return alias

    def _cleanup_temp_alias(self) -> None:
        alias = self._temp_alias
        target = self._temp_alias_target
        self._temp_alias = None
        self._temp_alias_target = None
        if alias is None or target is None:
            return
        try:
            metadata = os.lstat(alias)
            if (
                stat.S_ISLNK(metadata.st_mode)
                and metadata.st_uid == os.geteuid()
                and _symlink_target(alias) == target
            ):
                alias.unlink()
        except FileNotFoundError:
            pass
        # This per-UID directory is removed only when no other Topodrome Ray
        # lifecycle has an alias in it. Never recursively delete shared state.
        with suppress(OSError):
            alias.parent.rmdir()

    async def _run(
        self,
        *arguments: str,
        check: bool,
        timeout_seconds: float | None = None,
    ) -> int:
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=minimal_ray_head_environment(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Ray executable was not found: {self.config.command_prefix[0]}"
            ) from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=(
                    self.config.command_timeout_seconds
                    if timeout_seconds is None
                    else min(self.config.command_timeout_seconds, max(0.001, timeout_seconds))
                ),
            )
        except TimeoutError:
            process.kill()
            # ``wait`` should finish promptly after SIGKILL, but a broken
            # subprocess transport must not turn a bounded Ray probe into an
            # unbounded site-agent shutdown.
            with suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=1.0)
            raise _RayCommandTimeoutError(
                f"Ray command timed out: {_ray_action(arguments)}"
            ) from None
        if check and process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            if not detail:
                detail = stdout.decode(errors="replace").strip()
            raise RuntimeError(
                f"Ray command {_ray_action(arguments)} failed with {process.returncode}: "
                f"{detail[-2000:]}"
            )
        return int(process.returncode or 0)


class _RayCommandTimeoutError(RuntimeError):
    """A bounded Ray CLI probe expired while the process itself was reachable."""


def _ray_action(arguments: tuple[str, ...]) -> str:
    """Name the managed Ray operation independent of its launcher prefix."""

    for argument in arguments:
        if argument in {"start", "status"}:
            return argument
    return arguments[0] if arguments else "unknown"


@dataclass(frozen=True, slots=True)
class _OwnedRayHead:
    pid: int
    process_group_id: int
    start_ticks: int
    boot_id: str
    command_sha256: str
    address: str
    temp_dir: Path

    @classmethod
    def from_mapping(cls, raw: Any) -> _OwnedRayHead:
        if not isinstance(raw, dict) or int(raw.get("version", 0)) != 1:
            raise ValueError("unsupported Ray ownership record")
        result = cls(
            pid=int(raw["pid"]),
            process_group_id=int(raw["process_group_id"]),
            start_ticks=int(raw["start_ticks"]),
            boot_id=str(raw["boot_id"]),
            command_sha256=str(raw["command_sha256"]),
            address=str(raw["address"]),
            temp_dir=Path(str(raw["temp_dir"])),
        )
        if (
            result.pid <= 1
            or result.process_group_id != result.pid
            or result.start_ticks <= 0
            or not result.boot_id
            or not re.fullmatch(r"[0-9a-f]{64}", result.command_sha256)
            or not result.temp_dir.is_absolute()
        ):
            raise ValueError("invalid Ray ownership identity")
        return result

    def as_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "start_ticks": self.start_ticks,
            "boot_id": self.boot_id,
            "command_sha256": self.command_sha256,
            "address": self.address,
            "temp_dir": str(self.temp_dir),
        }


def _capture_owned_head(pid: int, *, address: str, temp_dir: Path) -> _OwnedRayHead:
    process_path = Path("/proc") / str(pid)
    metadata = process_path.stat()
    if metadata.st_uid != os.geteuid():
        raise RuntimeError("spawned Ray head is not owned by the site-agent user")
    process_group_id = os.getpgid(pid)
    if process_group_id != pid:
        raise RuntimeError("spawned Ray head is not an isolated process-group leader")
    return _OwnedRayHead(
        pid=pid,
        process_group_id=process_group_id,
        start_ticks=_process_start_ticks(pid),
        boot_id=_boot_id(),
        command_sha256=_process_command_sha256(pid),
        address=address,
        temp_dir=temp_dir,
    )


def _owned_head_matches(owned: _OwnedRayHead) -> bool:
    try:
        process_path = Path("/proc") / str(owned.pid)
        return (
            _boot_id() == owned.boot_id
            and process_path.stat().st_uid == os.geteuid()
            and os.getpgid(owned.pid) == owned.process_group_id == owned.pid
            and _process_start_ticks(owned.pid) == owned.start_ticks
            and _process_command_sha256(owned.pid) == owned.command_sha256
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def _same_spawn_identity(left: _OwnedRayHead, right: _OwnedRayHead) -> bool:
    """Compare the procfs identity fields which are invariant across exec."""

    return (
        left.pid == right.pid
        and left.process_group_id == right.process_group_id
        and left.start_ticks == right.start_ticks
        and left.boot_id == right.boot_id
        and left.address == right.address
        and left.temp_dir == right.temp_dir
    )


def _process_executable_is_python(pid: int) -> bool:
    """Report whether the live process has crossed into its Python runtime."""

    try:
        executable = os.readlink(Path("/proc") / str(pid) / "exe")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return False
    name = Path(executable).name.casefold()
    return re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", name) is not None


def _process_start_ticks(pid: int) -> int:
    raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    # The command in field two is parenthesized and may itself contain spaces.
    fields_after_command = raw.rsplit(")", 1)[1].split()
    return int(fields_after_command[19])


def _process_command_sha256(pid: int) -> str:
    command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    if not command:
        raise ValueError("process command line is empty")
    return hashlib.sha256(command).hexdigest()


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ExistingRayLifecycle:
    """Connect to an externally managed cluster using a lazy Ray import."""

    def __init__(
        self,
        address: str = "auto",
        *,
        namespace: str = "topodrome",
        control_timeout_seconds: float = 30.0,
    ) -> None:
        if not math.isfinite(control_timeout_seconds) or control_timeout_seconds <= 0:
            raise ValueError("Ray control timeout must be finite and positive")
        self.address = address
        self.namespace = namespace
        self.control_timeout_seconds = control_timeout_seconds
        self._connected = False

    def _ray(self) -> Any:
        # Workers inherit the site's uv-managed interpreter.  Disable Ray's
        # automatic nested ``uv run`` environment, which otherwise packages
        # the project and creates a second incomplete virtual environment.
        os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
        try:
            import ray
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Ray lifecycle requires the optional 'ray' package") from exc
        return ray

    async def is_running(self) -> bool:
        ray = self._ray()
        if ray.is_initialized():
            return True
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    ray.init,
                    address=self.address,
                    namespace=self.namespace,
                    ignore_reinit_error=True,
                ),
                timeout=self.control_timeout_seconds,
            )
        except Exception:
            return False
        self._connected = True
        return True

    async def ensure_running(self) -> bool:
        if not await self.is_running():
            raise RuntimeError(f"site-local Ray cluster is unavailable at {self.address}")
        return False

    async def stop(self) -> bool:
        if self._connected:
            await asyncio.wait_for(
                asyncio.to_thread(self._ray().shutdown),
                timeout=self.control_timeout_seconds,
            )
            self._connected = False
        # This lifecycle never owns the externally managed cluster itself.
        return False


__all__ = [
    "ExistingRayLifecycle",
    "RayClusterConfig",
    "RayLifecycle",
    "SubprocessRayLifecycle",
    "gpu_resource_name",
]


def _absolute_path(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{label} must be absolute")
    # Normalize ``..`` without resolving a symlink and accidentally hiding it
    # from the checks below.
    return Path(os.path.abspath(os.fspath(expanded)))


def _ensure_owned_directory(
    path: Path,
    *,
    label: str,
    create_parents: bool,
    repair_permissions: bool = True,
) -> None:
    if not path.parent.exists():
        if not create_parents:
            raise RuntimeError(f"{label} parent does not exist")
        # The final parent is security-sensitive because Ray follows the
        # short alias into it. ``parents=True`` may create earlier ancestors
        # with their conventional umask, but the immediate parent receives
        # this private mode and is validated below.
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _validate_directory_parent(path.parent, label=label)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"{label} must not be a symbolic link")
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} is not a directory")
    if metadata.st_uid != os.geteuid():
        raise RuntimeError(f"{label} is not owned by the site-agent user")
    permissions = stat.S_IMODE(metadata.st_mode)
    if permissions & 0o077:
        if not repair_permissions:
            raise RuntimeError(f"{label} permissions must exclude group and other users")
        path.chmod(0o700)


def _validate_directory_parent(path: Path, *, label: str) -> None:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} parent must be a real directory")
    permissions = stat.S_IMODE(metadata.st_mode)
    if permissions & 0o022 and not permissions & stat.S_ISVTX:
        raise RuntimeError(f"{label} parent is writable by another user without sticky bit")


def _validate_alias_base(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise RuntimeError("Ray short-temp root does not exist") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("Ray short-temp root must be a real directory")
    permissions = stat.S_IMODE(metadata.st_mode)
    if permissions & 0o022 and not permissions & stat.S_ISVTX:
        raise RuntimeError(
            "Ray short-temp root is writable by another user without sticky bit"
        )


def _ensure_exact_symlink(alias: Path, target: Path) -> None:
    try:
        metadata = os.lstat(alias)
    except FileNotFoundError:
        try:
            alias.symlink_to(target, target_is_directory=True)
        except FileExistsError:
            pass
        metadata = os.lstat(alias)
    if not stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError("Ray short-temp alias is not a symbolic link")
    if metadata.st_uid != os.geteuid():
        raise RuntimeError("Ray short-temp alias is not owned by the site-agent user")
    if _symlink_target(alias) != target:
        raise RuntimeError("Ray short-temp alias points to a different directory")


def _symlink_target(alias: Path) -> Path:
    raw = Path(os.readlink(alias))
    if not raw.is_absolute():
        raw = alias.parent / raw
    return Path(os.path.abspath(os.fspath(raw)))
