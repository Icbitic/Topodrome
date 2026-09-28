"""Physical GPU inventory and Topodrome process ownership tracking.

Ray's logical resources are deliberately not used here.  The site agent takes
its admission decisions from NVML and from the process identities it has
registered for leases and model actors.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pwd
import re
import shlex
import stat
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

_MAX_GPU_RESERVATION_LOCK_BYTES = 16 * 1024
_MAX_GPU_RESERVATION_WORLD_SIZE = 1024
_PROCESS_START_CLOCK_TOLERANCE_SECONDS = 2.0


class ProcessOwner(StrEnum):
    """How a process visible to NVML is classified."""

    TOPODROME = "topodrome"
    COOPERATIVE = "cooperative"
    FOREIGN = "foreign"


@dataclass(frozen=True, slots=True)
class GPUProcess:
    pid: int
    used_memory_bytes: int | None
    owner: ProcessOwner
    process_name: str | None = None
    command_line: str | None = field(default=None, repr=False)
    username: str | None = None
    script_name: str | None = None


@dataclass(frozen=True, slots=True)
class GPUReservation:
    """Validated foreign reservation for one exact physical GPU."""

    physical_gpu: int
    supervisor_pid: int
    owner: str
    world_size: int
    acquired_at: float
    supervisor_start_ticks: int


@dataclass(frozen=True, slots=True)
class GPUSnapshot:
    uuid: str
    index: int
    name: str
    total_memory_bytes: int
    free_memory_bytes: int
    used_memory_bytes: int
    utilization_percent: int
    external_reservation: GPUReservation | None = None
    processes: tuple[GPUProcess, ...] = field(default_factory=tuple)
    sampled_at: float = field(default_factory=time.time)

    @property
    def externally_reserved(self) -> bool:
        return self.external_reservation is not None

    @property
    def foreign_processes(self) -> tuple[GPUProcess, ...]:
        return tuple(p for p in self.processes if p.owner is ProcessOwner.FOREIGN)

    @property
    def cooperative_processes(self) -> tuple[GPUProcess, ...]:
        return tuple(p for p in self.processes if p.owner is ProcessOwner.COOPERATIVE)

    @property
    def external_processes(self) -> tuple[GPUProcess, ...]:
        """All processes Topodrome does not own, including cooperative loads."""

        return tuple(p for p in self.processes if p.owner is not ProcessOwner.TOPODROME)

    @property
    def topodrome_processes(self) -> tuple[GPUProcess, ...]:
        return tuple(p for p in self.processes if p.owner is ProcessOwner.TOPODROME)

    @property
    def foreign_memory_bytes(self) -> int:
        return sum(p.used_memory_bytes or 0 for p in self.foreign_processes)


class InventoryProvider(Protocol):
    """Structural boundary used by admission and the orchestration loop."""

    def sample(self) -> Sequence[GPUSnapshot]: ...


@dataclass(frozen=True, slots=True)
class HostSnapshot:
    """Physical host capacity observed outside Ray's logical scheduler."""

    total_cpu: float
    available_cpu: float
    total_memory_bytes: int
    available_memory_bytes: int
    sampled_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.total_cpu <= 0 or self.total_memory_bytes <= 0:
            raise ValueError("host CPU and memory totals must be positive")
        if not 0 <= self.available_cpu <= self.total_cpu:
            raise ValueError("available CPU must be between zero and total CPU")
        if not 0 <= self.available_memory_bytes <= self.total_memory_bytes:
            raise ValueError("available memory must be between zero and total memory")


class HostInventoryProvider(Protocol):
    def sample(self) -> HostSnapshot: ...


class SystemHostInventory:
    """Read Linux host/cgroup capacity for heartbeat and final admission."""

    def __init__(
        self,
        *,
        proc_meminfo: Path | str = "/proc/meminfo",
        cgroup_root: Path | str = "/sys/fs/cgroup",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._proc_meminfo = Path(proc_meminfo)
        self._cgroup_root = Path(cgroup_root)
        self._clock = clock

    def sample(self) -> HostSnapshot:
        total_cpu = _cpu_capacity(self._cgroup_root)
        total_memory, available_memory = _memory_capacity(
            self._proc_meminfo,
            self._cgroup_root,
        )
        return HostSnapshot(
            total_cpu=total_cpu,
            # Linux load average includes runnable and uninterruptible tasks
            # over a trailing window; it is neither an allocation ledger nor
            # instantaneous spare capacity. Treating it as a hard reservation
            # can report zero CPU long after bootstrap I/O has finished and
            # strand an otherwise idle GPU. The scheduler and the site-local
            # lease store subtract Topodrome's durable CPU reservations from
            # this cgroup/affinity capacity, while Ray enforces the allocation.
            available_cpu=total_cpu,
            total_memory_bytes=total_memory,
            available_memory_bytes=available_memory,
            sampled_at=self._clock(),
        )


class CooperativeProcessDetector(Protocol):
    """Explicit authority for classifying an external PID as reclaimable."""

    def matches(self, pid: int, *, command_line: str | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class OwnedProcess:
    pid: int
    lease_id: str | None = None
    actor_id: str | None = None
    start_ticks: int | None = field(default=None, repr=False)


class ProcessOwnershipRegistry:
    """Thread-safe registry of processes launched by Topodrome.

    A framework worker can be a descendant of a registered Ray actor PID, so
    :meth:`owns` also walks Linux ``/proc`` parent links.  Failure to inspect a
    process is conservative: an unknown process remains foreign.
    """

    def __init__(self, proc_root: Path | str = "/proc") -> None:
        self._proc_root = Path(proc_root)
        self._owned: dict[int, OwnedProcess] = {}
        self._lock = threading.RLock()

    def register(
        self,
        pid: int,
        *,
        lease_id: str | None = None,
        actor_id: str | None = None,
        start_ticks: int | None = None,
    ) -> None:
        if pid <= 0:
            raise ValueError("pid must be positive")
        observed_start = self._start_ticks(pid) if start_ticks is None else start_ticks
        if observed_start is not None and observed_start <= 0:
            raise ValueError("process start ticks must be positive")
        with self._lock:
            self._owned[pid] = OwnedProcess(pid, lease_id, actor_id, observed_start)

    def unregister(self, pid: int) -> None:
        with self._lock:
            self._owned.pop(pid, None)

    def clear_lease(self, lease_id: str) -> None:
        with self._lock:
            self._owned = {
                pid: record for pid, record in self._owned.items() if record.lease_id != lease_id
            }

    def snapshot(self) -> tuple[OwnedProcess, ...]:
        with self._lock:
            return tuple(self._owned.values())

    def exact_owner(self, pid: int, *, require_live: bool = False) -> OwnedProcess | None:
        """Return only an explicitly registered PID, never a descendant guess."""

        with self._lock:
            record = self._owned.get(pid)
        if record is None or not require_live:
            return record
        if self._same_process(record):
            return record
        self._discard_if_current(record)
        return None

    def owns(self, pid: int) -> bool:
        with self._lock:
            owned = dict(self._owned)
        direct = owned.get(pid)
        if direct is not None:
            if self._same_process(direct):
                return True
            self._discard_if_current(direct)
            return False

        # Limit traversal both for corrupt proc data and for recycled PIDs.
        current = pid
        visited: set[int] = set()
        for _ in range(128):
            if current <= 1 or current in visited:
                return False
            visited.add(current)
            parent = self._parent_pid(current)
            if parent is None:
                return False
            owner = owned.get(parent)
            if owner is not None:
                if self._same_process(owner):
                    return True
                self._discard_if_current(owner)
                return False
            current = parent
        return False

    def _same_process(self, record: OwnedProcess) -> bool:
        return (
            record.start_ticks is not None and self._start_ticks(record.pid) == record.start_ticks
        )

    def _discard_if_current(self, record: OwnedProcess) -> None:
        with self._lock:
            if self._owned.get(record.pid) == record:
                self._owned.pop(record.pid, None)

    def _start_ticks(self, pid: int) -> int | None:
        try:
            raw = (self._proc_root / str(pid) / "stat").read_text()
            close = raw.rfind(")")
            if close < 0:
                return None
            fields = raw[close + 1 :].split()
            # Field 22 is process start time; after removing pid/comm, field 3
            # (state) is index zero.
            value = int(fields[19])
            return value if value > 0 else None
        except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
            return None

    def _parent_pid(self, pid: int) -> int | None:
        try:
            for line in (self._proc_root / str(pid) / "status").read_text().splitlines():
                if line.startswith("PPid:"):
                    return int(line.split(":", 1)[1].strip())
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            return None
        return None


class CooperativeProcessMatcher:
    """Explicitly recognize a reclaimable external GPU load process.

    Matching is opt-in. A process must be an enrolled PID or match a configured
    command-line ownership marker. Executable names are deliberately ignored:
    generic ``python`` and Ray worker names are not safe evidence of ownership.
    Command lines stay local because they may contain credentials.
    """

    def __init__(
        self,
        *,
        command_line_patterns: Sequence[str] = (),
        pids: Iterable[int] = (),
        proc_root: Path | str = "/proc",
    ) -> None:
        self._command_line_patterns = tuple(
            re.compile(pattern) for pattern in command_line_patterns
        )
        self._pids = {int(pid) for pid in pids}
        if any(pid <= 0 for pid in self._pids):
            raise ValueError("cooperative process PIDs must be positive")
        self._proc_root = Path(proc_root)
        self._lock = threading.RLock()

    def register_pid(self, pid: int) -> None:
        if pid <= 0:
            raise ValueError("pid must be positive")
        with self._lock:
            self._pids.add(pid)

    def unregister_pid(self, pid: int) -> None:
        with self._lock:
            self._pids.discard(pid)

    def matches(
        self,
        pid: int,
        *,
        command_line: str | None = None,
    ) -> bool:
        with self._lock:
            if pid in self._pids:
                return True
            command_line_patterns = self._command_line_patterns
        command = (
            command_line
            if command_line is not None
            else _process_command_line(pid, self._proc_root)
        )
        return bool(command and any(pattern.search(command) for pattern in command_line_patterns))


class GPUReservationDetector(Protocol):
    """Read a foreign reservation for one physical index, if it is still valid."""

    def reservation_for(self, gpu_index: int) -> GPUReservation | None: ...


class GPUReservationLockReader:
    """Validate cooperative ``gpu_<index>.lock`` reservation records.

    Lock records are advisory input from a foreign launcher, so none of their
    contents are trusted until the file and its live supervisor identity have
    been checked. Invalid, unsafe, concurrently modified, or stale records are
    ignored; this reader never removes a file or signals a process.

    The producer does not currently record a mandatory process start identity.
    We therefore reject a PID whose procfs directory is newer than the lock or
    its acquisition timestamp, and honor an optional ``supervisor_start_ticks``
    field when present. There remains a narrow ambiguity if a stale lock is
    deliberately rewritten after Linux reuses its PID.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        proc_root: Path | str = "/proc",
        effective_uid: int | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        root = Path(directory).expanduser()
        if not root.is_absolute():
            raise ValueError("GPU reservation lock directory must be absolute")
        self._directory = root
        self._proc_root = Path(proc_root)
        self._effective_uid = os.geteuid() if effective_uid is None else effective_uid
        if self._effective_uid < 0:
            raise ValueError("effective UID cannot be negative")
        self._clock = clock

    def reservation_for(self, gpu_index: int) -> GPUReservation | None:
        if gpu_index < 0:
            raise ValueError("GPU index cannot be negative")
        try:
            opened = self._read_lock(gpu_index)
        except OSError:
            # A racing writer, disappearing directory, or filesystem error
            # must not take down the inventory heartbeat.
            return None
        if opened is None:
            return None
        payload, lock_metadata = opened
        parsed = self._parse_payload(payload, gpu_index)
        if parsed is None:
            return None
        owner, supervisor_pid, world_size, acquired_at, recorded_start_ticks = parsed
        identity = self._live_process_identity(supervisor_pid, expected_uid=lock_metadata.st_uid)
        if identity is None:
            return None
        process_started_at, observed_start_ticks = identity
        if recorded_start_ticks is not None and recorded_start_ticks != observed_start_ticks:
            return None
        # A process created after either record timestamp proves that this is a
        # recycled PID, not the supervisor which acquired the reservation.
        tolerance = _PROCESS_START_CLOCK_TOLERANCE_SECONDS
        if process_started_at > acquired_at + tolerance:
            return None
        if process_started_at > lock_metadata.st_mtime + tolerance:
            return None
        return GPUReservation(
            physical_gpu=gpu_index,
            supervisor_pid=supervisor_pid,
            owner=owner,
            world_size=world_size,
            acquired_at=acquired_at,
            supervisor_start_ticks=observed_start_ticks,
        )

    def _read_lock(self, gpu_index: int) -> tuple[bytes, os.stat_result] | None:
        directory_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            directory_fd = os.open(self._directory, directory_flags)
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            return None
        try:
            directory_metadata = os.fstat(directory_fd)
            trusted_uids = {0, self._effective_uid}
            if (
                not stat.S_ISDIR(directory_metadata.st_mode)
                or directory_metadata.st_uid not in trusted_uids
                or stat.S_IMODE(directory_metadata.st_mode) & 0o022
            ):
                return None
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(
                    f"gpu_{gpu_index}.lock",
                    flags,
                    dir_fd=directory_fd,
                )
            except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
                return None
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_uid not in trusted_uids
                    or before.st_uid != directory_metadata.st_uid
                    or stat.S_IMODE(before.st_mode) & 0o022
                    or before.st_nlink != 1
                    or not 0 < before.st_size <= _MAX_GPU_RESERVATION_LOCK_BYTES
                ):
                    return None
                chunks: list[bytes] = []
                remaining = _MAX_GPU_RESERVATION_LOCK_BYTES + 1
                while remaining > 0:
                    chunk = os.read(descriptor, remaining)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                content = b"".join(chunks)
                after = os.fstat(descriptor)
                if len(content) > _MAX_GPU_RESERVATION_LOCK_BYTES or not _same_file(before, after):
                    return None
                return content, after
            finally:
                os.close(descriptor)
        finally:
            os.close(directory_fd)

    def _parse_payload(
        self,
        content: bytes,
        gpu_index: int,
    ) -> tuple[str, int, int, float, int | None] | None:
        try:
            payload = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        physical_gpu = payload.get("physical_gpu")
        supervisor_pid = payload.get("supervisor_pid")
        world_size = payload.get("world_size")
        owner = payload.get("owner")
        if (
            type(physical_gpu) is not int
            or physical_gpu != gpu_index
            or type(supervisor_pid) is not int
            or not 1 < supervisor_pid <= 2**31 - 1
            or type(world_size) is not int
            or not 1 <= world_size <= _MAX_GPU_RESERVATION_WORLD_SIZE
            or not isinstance(owner, str)
            or not owner.strip()
            or len(owner) > 256
            or any(ord(character) < 32 for character in owner)
        ):
            return None
        timestamp_keys = tuple(
            key for key in ("acquired_at", "created_at", "timestamp") if key in payload
        )
        if len(timestamp_keys) != 1:
            return None
        acquired_at = _timestamp_seconds(payload[timestamp_keys[0]])
        if acquired_at is None or acquired_at > self._clock() + 300:
            return None
        recorded_start_ticks = payload.get("supervisor_start_ticks")
        if recorded_start_ticks is not None and (
            type(recorded_start_ticks) is not int or recorded_start_ticks <= 0
        ):
            return None
        return owner.strip(), supervisor_pid, world_size, acquired_at, recorded_start_ticks

    def _live_process_identity(
        self,
        pid: int,
        *,
        expected_uid: int,
    ) -> tuple[float, int] | None:
        process_root = self._proc_root / str(pid)
        try:
            before = process_root.lstat()
            if not stat.S_ISDIR(before.st_mode) or before.st_uid != expected_uid:
                return None
            raw = _read_proc_stat(process_root / "stat", expected_uid=expected_uid)
            after = process_root.lstat()
        except (
            FileNotFoundError,
            NotADirectoryError,
            PermissionError,
            OSError,
            UnicodeDecodeError,
        ):
            return None
        if not _same_process_directory(before, after):
            return None
        parsed = _parse_proc_stat(raw, expected_pid=pid)
        if parsed is None:
            return None
        state, start_ticks = parsed
        if state in {"Z", "X", "x"}:
            return None
        return before.st_mtime, start_ticks


def _allocated_memory(nvml: Any, handle: Any, memory: Any) -> int:
    # NVML v1 includes driver-reserved memory; v2 separates actual allocations.
    # Preserve v1's conservative figure if the newer API is unavailable.
    try:
        value = int(nvml.nvmlDeviceGetMemoryInfo(handle, version=nvml.nvmlMemory_v2).used)
        return value if value >= 0 else int(memory.used)
    except Exception:
        return int(memory.used)


def _gpm_idle_utilization(
    nvml: Any, handle: Any, *, sleep: Callable[[float], None] = time.sleep
) -> int | None:
    """Confirm low hardware activity over two independent sampling windows.

    No stream configuration, resets, or device mutations. Unsupported metrics,
    permission errors and invalid samples retain the conservative legacy value.
    """
    samples: list[Any] = []
    try:
        if not nvml.nvmlGpmQueryDeviceSupport(handle).isSupportedDevice:
            return None
        samples.append(nvml.nvmlGpmSampleAlloc())
        samples.append(nvml.nvmlGpmSampleAlloc())
        maximum = 0.0
        for _ in range(2):
            nvml.nvmlGpmSampleGet(handle, samples[0])
            sleep(0.2)
            nvml.nvmlGpmSampleGet(handle, samples[1])
            query = nvml.c_nvmlGpmMetricsGet_t()
            query.version = nvml.NVML_GPM_METRICS_GET_VERSION
            query.numMetrics = 2
            query.sample1, query.sample2 = samples
            # NVML_GPM_METRIC_GRAPHICS_UTIL and NVML_GPM_METRIC_SM_UTIL.
            query.metrics[0].metricId = 1
            query.metrics[1].metricId = 2
            nvml.nvmlGpmMetricsGet(query)
            for metric in query.metrics[:2]:
                value = float(metric.value)
                if metric.nvmlReturn != 0 or not math.isfinite(value) or not 0 <= value <= 10:
                    return None
                maximum = max(maximum, value)
        return math.ceil(maximum)
    except Exception:
        return None
    finally:
        for sample in samples:
            try:
                nvml.nvmlGpmSampleFree(sample)
            except Exception:
                logging.getLogger(__name__).debug("GPM sample cleanup failed")


class NVMLInventoryProvider:
    """Read physical GPU state using a lazily imported NVML binding.

    ``nvidia-ml-py`` is optional for clients, schedulers, and unit tests.  It is
    imported only when a real site agent samples its inventory.
    """

    def __init__(
        self,
        ownership: ProcessOwnershipRegistry,
        *,
        cooperative_matcher: CooperativeProcessDetector | None = None,
        reservation_detector: GPUReservationDetector | None = None,
        allowed_gpu_indices: Iterable[int] | None = None,
        autodl_gpm_fallback: bool = False,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._ownership = ownership
        self._autodl_gpm_fallback = autodl_gpm_fallback
        self._cooperative_matcher = cooperative_matcher
        self._reservation_detector = reservation_detector
        self._allowed_gpu_indices = (
            None
            if allowed_gpu_indices is None
            else frozenset(int(index) for index in allowed_gpu_indices)
        )
        if self._allowed_gpu_indices is not None and any(
            index < 0 for index in self._allowed_gpu_indices
        ):
            raise ValueError("allowed GPU indices cannot be negative")
        self._clock = clock
        self._nvml: object | None = None
        self._initialized = False
        self._gpm_reported: set[str] = set()

    def _library(self) -> Any:
        if self._nvml is None:
            try:
                import pynvml  # type: ignore[import-untyped]
            except ImportError as exc:  # pragma: no cover - hardware environment
                raise RuntimeError(
                    "NVML inventory requires the optional 'nvidia-ml-py' package"
                ) from exc
            self._nvml = pynvml
        return self._nvml

    def start(self) -> None:
        if not self._initialized:
            nvml = self._library()
            nvml.nvmlInit()
            self._initialized = True

    def close(self) -> None:
        if self._initialized:
            self._library().nvmlShutdown()
            self._initialized = False

    def sample(self) -> tuple[GPUSnapshot, ...]:
        self.start()
        nvml = self._library()
        sampled_at = self._clock()
        snapshots: list[GPUSnapshot] = []
        for index in range(nvml.nvmlDeviceGetCount()):
            if self._allowed_gpu_indices is not None and index not in self._allowed_gpu_indices:
                continue
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            name = _decode(nvml.nvmlDeviceGetName(handle))
            uuid = _decode(nvml.nvmlDeviceGetUUID(handle))
            processes = self._processes(nvml, handle)
            gpu_utilization = int(utilization.gpu)
            # Blackwell bug 5446845 can latch the legacy utilization counter
            # at 100% after abnormal context teardown. Never infer idleness
            # from an empty process list alone: independently measure hardware.
            if (
                (self._autodl_gpm_fallback or "RTX 5090" in name)
                and gpu_utilization >= 99
                and _allocated_memory(nvml, handle, memory) <= 16 * 1024**2
                and not processes
            ):
                measured = _gpm_idle_utilization(nvml, handle)
                if measured is not None:
                    memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                    processes = self._processes(nvml, handle)
                    if _allocated_memory(nvml, handle, memory) <= 16 * 1024**2 and not processes:
                        gpu_utilization = measured
                        if uuid not in self._gpm_reported:
                            logging.getLogger(__name__).warning(
                                "GPU %s: latched NVML utilization %s%%; independent GPM %s%%",
                                uuid,
                                int(utilization.gpu),
                                measured,
                            )
                            self._gpm_reported.add(uuid)
            reservation = (
                None
                if self._reservation_detector is None
                else self._reservation_detector.reservation_for(index)
            )
            snapshots.append(
                GPUSnapshot(
                    uuid=uuid,
                    index=index,
                    name=name,
                    total_memory_bytes=int(memory.total),
                    free_memory_bytes=int(memory.free),
                    used_memory_bytes=int(memory.used),
                    utilization_percent=gpu_utilization,
                    external_reservation=reservation,
                    processes=processes,
                    sampled_at=sampled_at,
                )
            )
        return tuple(snapshots)

    def _processes(self, nvml: Any, handle: Any) -> tuple[GPUProcess, ...]:
        by_pid: dict[int, int | None] = {}
        getters = (
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses", None),
            getattr(nvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
        )
        for getter in getters:
            if getter is None:
                continue
            try:
                records: Iterable[Any] = getter(handle)
            except Exception as exc:  # pragma: no cover - NVML-version specific
                if _is_not_supported(nvml, exc):
                    continue
                raise
            for record in records:
                pid = int(record.pid)
                raw_memory = getattr(record, "usedGpuMemory", None)
                unavailable = getattr(nvml, "NVML_VALUE_NOT_AVAILABLE", None)
                used_memory = (
                    None if raw_memory is None or raw_memory == unavailable else int(raw_memory)
                )
                previous = by_pid.get(pid)
                if previous is None or (used_memory is not None and used_memory > previous):
                    by_pid[pid] = used_memory

        processes: list[GPUProcess] = []
        for pid, used_memory in sorted(by_pid.items()):
            process_name = _process_name(pid)
            command_line = _process_command_line(pid)
            if self._ownership.owns(pid):
                owner = ProcessOwner.TOPODROME
            elif self._cooperative_matcher is not None and self._cooperative_matcher.matches(
                pid,
                command_line=command_line,
            ):
                owner = ProcessOwner.COOPERATIVE
            else:
                owner = ProcessOwner.FOREIGN
            processes.append(
                GPUProcess(
                    pid=pid,
                    used_memory_bytes=used_memory,
                    owner=owner,
                    process_name=process_name,
                    command_line=command_line,
                    username=_process_username(pid),
                    script_name=_script_name(command_line),
                )
            )
        return tuple(processes)

    def __enter__(self) -> NVMLInventoryProvider:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _decode(value: str | bytes) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _is_not_supported(nvml: Any, exc: Exception) -> bool:
    error_type = getattr(nvml, "NVMLError_NotSupported", ())
    return bool(error_type) and isinstance(exc, error_type)


def _process_name(pid: int, proc_root: Path = Path("/proc")) -> str | None:
    try:
        return (proc_root / str(pid) / "comm").read_text().strip() or None
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _process_command_line(pid: int, proc_root: Path = Path("/proc")) -> str | None:
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    return raw.replace(b"\0", b" ").decode(errors="replace").strip() or None


def _process_username(pid: int, proc_root: Path = Path("/proc")) -> str | None:
    try:
        uid = (proc_root / str(pid)).stat().st_uid
        try:
            return pwd.getpwuid(uid).pw_name
        except KeyError:
            return f"uid:{uid}"
    except OSError:
        return None


def _script_name(command_line: str | None) -> str | None:
    """Only a Python script basename/module, never arguments or inline code."""
    try:
        parts = shlex.split((command_line or "")[:8192])
    except ValueError:
        return None
    if not parts or not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(parts[0]).name):
        return None
    args = iter(parts[1:])
    for arg in args:
        if arg in {"-u", "-B", "-E", "-I", "-O", "-OO", "-s", "-S"}:
            continue
        if arg == "-m":
            module = next(args, "")
            return module if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) else None
        if not arg.startswith("-") and arg.endswith(".py"):
            return Path(arg).name[:160]
        return None
    return None


def _same_file(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_mode,
        first.st_uid,
        first.st_nlink,
        first.st_size,
        first.st_mtime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_mode,
        second.st_uid,
        second.st_nlink,
        second.st_size,
        second.st_mtime_ns,
    )


def _same_process_directory(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_uid,
        first.st_ctime_ns,
        first.st_mtime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_uid,
        second.st_ctime_ns,
        second.st_mtime_ns,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _timestamp_seconds(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str) and 0 < len(value) <= 100:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return None
        try:
            result = parsed.astimezone(UTC).timestamp()
        except (OSError, OverflowError, ValueError):
            return None
    else:
        return None
    if not math.isfinite(result) or result <= 0:
        return None
    return result


def _read_proc_stat(path: Path, *, expected_uid: int) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != expected_uid:
            raise OSError("unsafe proc stat entry")
        raw = os.read(descriptor, 4097)
        if not raw or len(raw) > 4096:
            raise OSError("invalid proc stat entry")
        return raw.decode("ascii")
    finally:
        os.close(descriptor)


def _parse_proc_stat(raw: str, *, expected_pid: int) -> tuple[str, int] | None:
    close = raw.rfind(")")
    if close < 0 or not raw.startswith(f"{expected_pid} ("):
        return None
    fields = raw[close + 1 :].split()
    try:
        state = fields[0]
        start_ticks = int(fields[19])
    except (IndexError, ValueError):
        return None
    if len(state) != 1 or start_ticks <= 0:
        return None
    return state, start_ticks


def _cpu_capacity(cgroup_root: Path) -> float:
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux fallback
        affinity_count = os.cpu_count() or 1
    capacity = float(max(1, affinity_count))
    try:
        quota, period = (cgroup_root / "cpu.max").read_text(encoding="utf-8").split()[:2]
        if quota != "max":
            capacity = min(capacity, int(quota) / int(period))
    except (FileNotFoundError, PermissionError, OSError, ValueError, ZeroDivisionError):
        pass
    return max(capacity, 0.001)


def _memory_capacity(proc_meminfo: Path, cgroup_root: Path) -> tuple[int, int]:
    try:
        fields = {
            key.rstrip(":"): int(value) * 1024
            for key, value, *_unit in (
                line.split() for line in proc_meminfo.read_text(encoding="utf-8").splitlines()
            )
        }
        host_total = fields["MemTotal"]
        host_available = fields.get("MemAvailable", fields.get("MemFree", 0))
    except (FileNotFoundError, PermissionError, OSError, KeyError, ValueError) as exc:
        raise RuntimeError("unable to read physical host memory capacity") from exc
    total = host_total
    available = min(host_available, host_total)
    try:
        raw_limit = (cgroup_root / "memory.max").read_text(encoding="utf-8").strip()
        if raw_limit != "max":
            limit = int(raw_limit)
            current = int((cgroup_root / "memory.current").read_text(encoding="utf-8").strip())
            total = min(total, limit)
            available = min(available, max(0, limit - current))
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        pass
    if total <= 0:
        raise RuntimeError("physical host memory capacity is not positive")
    return total, max(0, min(available, total))


__all__ = [
    "CooperativeProcessDetector",
    "CooperativeProcessMatcher",
    "GPUProcess",
    "GPUReservation",
    "GPUReservationDetector",
    "GPUReservationLockReader",
    "GPUSnapshot",
    "HostInventoryProvider",
    "HostSnapshot",
    "InventoryProvider",
    "NVMLInventoryProvider",
    "OwnedProcess",
    "ProcessOwner",
    "ProcessOwnershipRegistry",
    "SystemHostInventory",
]
