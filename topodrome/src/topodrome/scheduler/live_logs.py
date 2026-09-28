"""Bounded, idempotent transport storage for live attempt output.

The final ``attempt.log`` published through the storage adapter remains the
authoritative record.  This store is only a scheduler-local side channel that
makes output visible while a remote attempt is still running.  Chunk offsets
make an acknowledgement-lost retry safe without putting user output in
PostgreSQL or the durable agent event journal.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from topodrome.domain.logs import (
    LIVE_LOG_CHUNK_BYTES,
    LOG_ANCHOR_BYTES,
    LOG_READ_LIMIT_BYTES,
    LiveLogChunkReport,
)
from topodrome.domain.progress import PROGRESS_PREFIX, PROGRESS_RECORD_BYTES, JobProgress
from topodrome.domain.specs import AttemptLogDescriptor

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")


@dataclass(frozen=True, slots=True)
class LiveLogAppendResult:
    accepted: bool
    generation: int
    next_offset: int
    duplicate: bool = False
    reset: bool = False
    stop: bool = False

    def as_payload(self) -> dict[str, bool | int]:
        return {
            "accepted": self.accepted,
            "generation": self.generation,
            "next_offset": self.next_offset,
            "duplicate": self.duplicate,
            "reset": self.reset,
            "stop": self.stop,
        }


@dataclass(frozen=True, slots=True)
class LogRange:
    data: bytes
    offset: int
    next_offset: int
    size_bytes: int
    reset: bool
    anchor_sha256: str | None
    complete: bool


class LiveLogStore:
    """Filesystem-backed live bytes, separate from authoritative job output."""

    def __init__(
        self,
        global_root: str | Path,
        *,
        max_stream_bytes: int = 64 * 1024 * 1024,
        retention_seconds: float = 24 * 60 * 60,
    ) -> None:
        if max_stream_bytes < LIVE_LOG_CHUNK_BYTES:
            raise ValueError("live log stream limit must fit one complete chunk")
        if retention_seconds <= 0:
            raise ValueError("live log retention must be positive")
        self.root = Path(global_root) / ".topodrome-live-logs"
        self.max_stream_bytes = max_stream_bytes
        self.retention_seconds = retention_seconds
        self._lock = threading.RLock()

    def append(self, report: LiveLogChunkReport) -> LiveLogAppendResult:
        """Append or acknowledge an identical replay without duplicating bytes."""

        data = report.decoded()
        with self._lock:
            with self._secure_attempt_directory(report.job_id, report.attempt_id) as (
                _,
                directory_fd,
            ):
                if _regular_entry_exists(directory_fd, "complete.json"):
                    return LiveLogAppendResult(
                        accepted=False,
                        generation=report.generation,
                        next_offset=report.offset,
                        stop=True,
                    )
                generations = self._generations_fd(directory_fd)
                latest = generations[-1] if generations else None
                if latest is not None and report.generation < latest:
                    return LiveLogAppendResult(
                        accepted=False,
                        generation=latest + 1,
                        next_offset=0,
                        reset=True,
                    )
                if latest is not None and report.generation > latest:
                    if report.offset != 0:
                        return LiveLogAppendResult(
                            accepted=False,
                            generation=report.generation,
                            next_offset=0,
                            reset=True,
                        )
                    for old_generation in generations:
                        os.unlink(f"{old_generation}.log", dir_fd=directory_fd)

                filename = f"{report.generation}.log"
                descriptor = os.open(
                    filename,
                    os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "r+b") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    if report.offset > size:
                        return LiveLogAppendResult(
                            accepted=False,
                            generation=report.generation,
                            next_offset=size,
                        )

                    overlap = min(len(data), size - report.offset)
                    if overlap:
                        handle.seek(report.offset)
                        existing = handle.read(overlap)
                        if not hmac.compare_digest(existing, data[:overlap]):
                            return LiveLogAppendResult(
                                accepted=False,
                                generation=report.generation + 1,
                                next_offset=0,
                                reset=True,
                            )
                    suffix = data[overlap:]
                    if suffix and size + len(suffix) > self.max_stream_bytes:
                        return LiveLogAppendResult(
                            accepted=False,
                            generation=report.generation,
                            next_offset=size,
                            stop=True,
                        )
                    if suffix:
                        handle.seek(0, os.SEEK_END)
                        handle.write(suffix)
                        handle.flush()
                        os.fsync(handle.fileno())
                        # Look behind the new bytes to reassemble a split record.
                        # Only records ending in the newly appended suffix count.
                        try:
                            start = max(0, size - PROGRESS_RECORD_BYTES)
                            handle.seek(start)
                            window = handle.read(size + len(suffix) - start)
                            for match in re.finditer(
                                rb"(?m)^" + PROGRESS_PREFIX.encode() + rb"([^\n]{1,4096})\n",
                                window,
                            ):
                                if start + match.end() <= size:
                                    continue
                                try:
                                    progress = JobProgress.model_validate_json(match[1])
                                except ValueError:
                                    continue
                                if progress.total is not None and progress.current > progress.total:
                                    continue
                                _write_json_at(
                                    directory_fd,
                                    "progress.json",
                                    {
                                        **progress.model_dump(),
                                        "generation": report.generation,
                                        "received_at": time.time(),
                                    },
                                )
                        except OSError:
                            pass  # Advisory telemetry cannot reject valid log bytes.
                    return LiveLogAppendResult(
                        accepted=True,
                        generation=report.generation,
                        next_offset=report.offset + len(data),
                        duplicate=not suffix,
                    )

    def read_progress(self, job_id: str, attempt_id: str) -> dict | None:
        with self._lock:
            directory = self._attempt_directory(job_id, attempt_id)
            path = directory / "progress.json"
            if not _safe_regular_file(path, directory):
                return None
            try:
                if path.stat().st_size > PROGRESS_RECORD_BYTES:
                    return None
                value = json.loads(path.read_text())
                generations = self._generations(directory)
                if generations and value.get("generation") != generations[-1]:
                    return None
                value["stale"] = time.time() - value.get("received_at", 0) > 30
                return value
            except (OSError, ValueError):
                return None

    def latest_path(self, job_id: str, attempt_id: str) -> Path | None:
        with self._lock:
            directory = self._attempt_directory(job_id, attempt_id)
            generations = self._generations(directory)
            if not generations:
                return None
            path = self._generation_path(directory, generations[-1])
            return path if _safe_regular_file(path, directory) else None

    def log_path(
        self,
        job_id: str,
        attempt_id: str,
        *,
        attempts_root: Path,
    ) -> Path | None:
        """Return authoritative bytes only after scheduler verification."""

        if self.is_complete(job_id, attempt_id):
            return self.authoritative_path(attempt_id, attempts_root=attempts_root)
        return self.latest_path(job_id, attempt_id)

    @staticmethod
    def authoritative_path(attempt_id: str, *, attempts_root: Path) -> Path | None:
        _validate_component("attempt_id", attempt_id)
        try:
            resolved_root = attempts_root.resolve()
            attempt_root = resolved_root / attempt_id
            if attempt_root.is_symlink() or not attempt_root.is_dir():
                return None
        except OSError:
            return None
        for relative in (Path("logs") / "attempt.log", Path("attempt.log")):
            candidate = attempt_root / relative
            if _safe_regular_file(candidate, attempt_root):
                return candidate
        return None

    def register_authoritative(
        self,
        job_id: str,
        attempt_id: str,
        *,
        attempts_root: Path,
        descriptor: AttemptLogDescriptor | None,
    ) -> bool:
        """Verify final bytes, write a compact completion marker, then clean live data.

        A descriptor comes only from a fenced terminal report, but the scheduler
        independently hashes the authoritative file before trusting it. The
        advisory sidecars are the only unlink targets.
        """

        with self._lock:
            authoritative = self.authoritative_path(attempt_id, attempts_root=attempts_root)
            if descriptor is None:
                if authoritative is not None:
                    return False
                marker = {
                    "absent": True,
                    "size_bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                }
            else:
                expected_path = f"jobs/{job_id}/attempts/{attempt_id}/logs/attempt.log"
                if descriptor.logical_path != expected_path or authoritative is None:
                    return False
                digest, size = _hash_regular_file(authoritative)
                if size != descriptor.end_offset or not hmac.compare_digest(
                    digest, descriptor.sha256
                ):
                    return False
                marker = {
                    "absent": False,
                    "size_bytes": size,
                    "sha256": digest,
                }
            with self._secure_attempt_directory(job_id, attempt_id) as (_, directory_fd):
                _write_json_at(directory_fd, "complete.json", marker)
                for generation in self._generations_fd(directory_fd):
                    os.unlink(f"{generation}.log", dir_fd=directory_fd)
            return True

    def is_complete(self, job_id: str, attempt_id: str) -> bool:
        with self._lock:
            directory = self._attempt_directory(job_id, attempt_id)
            marker = directory / "complete.json"
            if not _safe_regular_file(marker, directory):
                return False
            try:
                value = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            return (
                isinstance(value, dict)
                and isinstance(value.get("absent"), bool)
                and isinstance(value.get("size_bytes"), int)
                and value["size_bytes"] >= 0
                and isinstance(value.get("sha256"), str)
                and bool(re.fullmatch(r"[0-9a-f]{64}", value["sha256"]))
            )

    def prune_expired(
        self,
        *,
        now: float | None = None,
        retain_attempts: frozenset[tuple[str, str]] = frozenset(),
    ) -> int:
        with self._lock:
            return self._prune_expired_locked(
                now=time.time() if now is None else now,
                retain_attempts=retain_attempts,
            )

    def _prune_expired_locked(
        self,
        *,
        now: float,
        retain_attempts: frozenset[tuple[str, str]],
    ) -> int:
        if not self.root.is_dir() or self.root.is_symlink():
            return 0
        removed = 0
        cutoff = now - self.retention_seconds
        for job_directory in tuple(self.root.iterdir()):
            if job_directory.is_symlink() or not job_directory.is_dir():
                continue
            for attempt_directory in tuple(job_directory.iterdir()):
                if attempt_directory.is_symlink() or not attempt_directory.is_dir():
                    continue
                if (job_directory.name, attempt_directory.name) in retain_attempts:
                    continue
                entries = tuple(attempt_directory.iterdir())
                logs = tuple(
                    entry
                    for entry in entries
                    if not entry.is_symlink()
                    and entry.is_file()
                    and entry.suffix == ".log"
                    and entry.stem.isdigit()
                )
                if not logs or max(entry.stat().st_mtime for entry in logs) >= cutoff:
                    continue
                for entry in logs:
                    entry.unlink()
                    removed += 1
                try:
                    attempt_directory.rmdir()
                except OSError:
                    pass
            try:
                job_directory.rmdir()
            except OSError:
                pass
        return removed

    def read_tail(
        self,
        job_id: str,
        attempt_ids: Sequence[str],
        *,
        attempts_root: Path,
        limit_bytes: int,
    ) -> bytes:
        """Tail authorized attempts in execution order, live or verified final.

        Hold the store lock through reading so terminal publication cannot
        remove a selected sidecar between path selection and open.
        """

        if not 1 <= limit_bytes <= 10_000_000:
            raise ValueError("log tail limit is outside the supported range")
        chunks: list[bytes] = []
        remaining = limit_bytes
        with self._lock:
            for attempt_id in reversed(attempt_ids):
                if remaining <= 0:
                    break
                path = self.log_path(job_id, attempt_id, attempts_root=attempts_root)
                if path is None:
                    continue
                try:
                    with path.open("rb") as handle:
                        size = os.fstat(handle.fileno()).st_size
                        budget = remaining - bool(chunks)
                        if not size or budget <= 0:
                            continue
                        handle.seek(max(0, size - budget))
                        data = handle.read(budget)
                except FileNotFoundError:
                    continue
                if data:
                    remaining -= len(data) + bool(chunks)
                    chunks.append(data)
        return b"\n".join(reversed(chunks))

    def read_range(
        self,
        job_id: str,
        attempt_id: str,
        *,
        attempts_root: Path,
        offset: int,
        limit_bytes: int,
        anchor_sha256: str | None,
    ) -> LogRange:
        if offset < 0:
            raise ValueError("log offset cannot be negative")
        if not 1 <= limit_bytes <= LOG_READ_LIMIT_BYTES:
            raise ValueError("log read limit is outside the supported range")
        with self._lock:
            complete = self.is_complete(job_id, attempt_id)
            path = self.log_path(job_id, attempt_id, attempts_root=attempts_root)
            if path is None:
                return LogRange(b"", 0, 0, 0, offset != 0, None, complete)
            with path.open("rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                actual_offset = offset
                reset = offset > size
                if not reset and offset > 0 and anchor_sha256 is not None:
                    handle.seek(max(0, offset - LOG_ANCHOR_BYTES))
                    anchor = handle.read(min(LOG_ANCHOR_BYTES, offset))
                    reset = not hmac.compare_digest(
                        hashlib.sha256(anchor).hexdigest(), anchor_sha256
                    )
                if reset:
                    actual_offset = 0
                handle.seek(actual_offset)
                data = handle.read(min(limit_bytes, max(0, size - actual_offset)))
                next_offset = actual_offset + len(data)
                next_anchor = _anchor_digest(handle, next_offset)
                return LogRange(
                    data=data,
                    offset=actual_offset,
                    next_offset=next_offset,
                    size_bytes=size,
                    reset=reset,
                    anchor_sha256=next_anchor,
                    complete=complete,
                )

    def _attempt_directory(self, job_id: str, attempt_id: str) -> Path:
        _validate_component("job_id", job_id)
        _validate_component("attempt_id", attempt_id)
        return self.root / job_id / attempt_id

    @staticmethod
    def _generation_path(directory: Path, generation: int) -> Path:
        return directory / f"{generation}.log"

    @staticmethod
    def _generations(directory: Path) -> list[int]:
        if not directory.is_dir() or directory.is_symlink():
            return []
        values: list[int] = []
        for path in directory.glob("*.log"):
            try:
                generation = int(path.stem)
            except ValueError:
                continue
            if generation >= 0 and _safe_regular_file(path, directory):
                values.append(generation)
        return sorted(values)

    @staticmethod
    def _generations_fd(directory_fd: int) -> list[int]:
        values: list[int] = []
        for name in os.listdir(directory_fd):
            stem, suffix = os.path.splitext(name)
            if suffix != ".log" or not stem.isdigit():
                continue
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISREG(metadata.st_mode):
                values.append(int(stem))
        return sorted(values)

    @contextmanager
    def _secure_attempt_directory(self, job_id: str, attempt_id: str) -> Iterator[tuple[Path, int]]:
        directory = self._attempt_directory(job_id, attempt_id)
        global_root = self.root.parent
        global_root.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptors: list[int] = []
        try:
            current = os.open(global_root, flags)
            descriptors.append(current)
            for component in (self.root.name, job_id, attempt_id):
                try:
                    os.mkdir(component, 0o700, dir_fd=current)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=current)
                os.fchmod(child, 0o700)
                descriptors.append(child)
                current = child
            yield directory, current
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


def _anchor_digest(handle: BinaryIO, offset: int) -> str | None:
    if offset <= 0:
        return None
    handle.seek(max(0, offset - LOG_ANCHOR_BYTES))
    return hashlib.sha256(handle.read(min(LOG_ANCHOR_BYTES, offset))).hexdigest()


def _hash_regular_file(path: Path) -> tuple[str, int]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("authoritative log is not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OSError("authoritative log changed during verification")
        return digest.hexdigest(), after.st_size
    finally:
        os.close(descriptor)


def _write_json_at(directory_fd: int, name: str, value: object) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    temporary = f".{name}.{os.getpid()}.{threading.get_ident()}.{os.urandom(8).hex()}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("failed to write live-log completion marker")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _safe_regular_file(path: Path, parent: Path) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        resolved_parent = parent.resolve()
        resolved = path.resolve()
        return resolved == resolved_parent or resolved_parent in resolved.parents
    except OSError:
        return False


def _regular_entry_exists(directory_fd: int, name: str) -> bool:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(metadata.st_mode)


def _validate_component(label: str, value: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"unsafe {label}")


__all__ = [
    "LOG_READ_LIMIT_BYTES",
    "LiveLogAppendResult",
    "LiveLogChunkReport",
    "LiveLogStore",
    "LogRange",
]
