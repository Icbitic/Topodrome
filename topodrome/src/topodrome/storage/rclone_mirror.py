"""No-FUSE rclone mirror backed by an authoritative remote filesystem."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4
from weakref import WeakValueDictionary

from topodrome.storage.base import (
    AuthoritativeDigestReader,
    ObjectDigest,
    StorageUnavailableError,
    StorageVerificationError,
    VerifiedObject,
)
from topodrome.storage.local import fsync_and_digest
from topodrome.storage.paths import normalize_logical_path, resolve_beneath
from topodrome.storage.progress import (
    check_transfer_cancelled,
    observing_transfers,
    reusing_verified_outputs,
    transfer_lock,
    transfer_progress,
)
from topodrome.storage.rclone import RcloneCommandReader
from topodrome.storage.rclone_environment import rclone_process_environment
from topodrome.storage.rclone_progress import run_upload
from topodrome.storage.transfer_process import run_transfer

_RCLONE_NOT_FOUND = 3
_MAX_STAT_BYTES = 1024 * 1024
_MAX_PARTIAL_METADATA_BYTES = 16 * 1024
_RCLONE_REMOTE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PARTIAL_METADATA_SCHEMA = 1
_PARTIAL_DIRECTORY = ".topodrome-mirror-partials"


@dataclass(frozen=True, slots=True)
class _RemoteStat:
    is_directory: bool
    size_bytes: int
    modification_time: str | None
    object_id: str | None

    @property
    def resumable_identity(self) -> dict[str, str | int | None]:
        return {
            "size_bytes": self.size_bytes,
            "modification_time": self.modification_time,
            "object_id": self.object_id,
        }

    @property
    def has_stable_identity(self) -> bool:
        return self.modification_time is not None or self.object_id is not None


class RcloneMirrorStorage:
    """A local cache synchronized through direct rclone commands.

    This adapter is intended for remote sites where FUSE is unavailable.  A
    missing local object is downloaded atomically from ``remote_root``.  Files
    written into the returned local namespace become durable only after
    :meth:`flush_and_verify`, which uploads through a unique remote temporary
    name, promotes it, and reads it back through an independent command.

    Existing local objects are deliberately treated as cache hits.  Topodrome's
    source bundles, immutable assets, attempt outputs, and checkpoint
    references are immutable paths.  Mutable shared files therefore must not
    be used as an inter-site synchronization primitive with this adapter.
    """

    def __init__(
        self,
        cache_root: Path | str,
        remote_root: str,
        *,
        binary: Path | str = "rclone",
        command_timeout_seconds: float = 120,
        transfer_timeout_seconds: float = 6 * 60 * 60,
        verification_timeout_seconds: float = 120,
        poll_interval_seconds: float = 0.25,
        min_free_space_gb: float = 10,
        healthcheck_cache_seconds: float = 5,
        authoritative_reader: AuthoritativeDigestReader | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        if not remote_root or any(character in remote_root for character in "\r\n\0"):
            raise ValueError("remote_root must be a non-empty rclone remote path")
        remote_name, separator, remote_path = remote_root.rstrip("/").partition(":")
        if (
            not separator
            or not _RCLONE_REMOTE_NAME.fullmatch(remote_name)
            or (
                remote_path
                and any(part in {"", ".", ".."} for part in remote_path.lstrip("/").split("/"))
            )
        ):
            raise ValueError("remote_root must use a named rclone remote and a safe path")
        for value, label in (
            (command_timeout_seconds, "command timeout"),
            (transfer_timeout_seconds, "transfer timeout"),
            (verification_timeout_seconds, "verification timeout"),
            (min_free_space_gb, "minimum free space"),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        if poll_interval_seconds < 0:
            raise ValueError("poll interval must be non-negative")
        if healthcheck_cache_seconds < 0:
            raise ValueError("healthcheck cache duration must be non-negative")

        resolved_binary = shutil.which(str(binary))
        if resolved_binary is None:
            raise FileNotFoundError(f"rclone executable not found: {binary}")
        self.binary = str(Path(resolved_binary).resolve())
        self.root = Path(cache_root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        # Keep partials on the cache filesystem so final promotion is one atomic
        # rename even when ``root`` itself is a mount point.
        partial_root = self.root / _PARTIAL_DIRECTORY
        if partial_root.is_symlink():
            raise ValueError("rclone partial directory must not be a symbolic link")
        partial_root.mkdir(mode=0o700, exist_ok=True)
        if not partial_root.is_dir() or partial_root.is_symlink():
            raise ValueError("rclone partial directory must be a private directory")
        partial_status = partial_root.stat()
        if partial_status.st_uid != os.geteuid():
            raise ValueError("rclone partial directory must be owned by the site agent")
        if stat.S_IMODE(partial_status.st_mode) & 0o077:
            partial_root.chmod(0o700)
        self._partial_root = partial_root.resolve()
        self.remote_root = remote_root.rstrip("/")
        self._command_timeout = command_timeout_seconds
        self._transfer_timeout = transfer_timeout_seconds
        self._verification_timeout = verification_timeout_seconds
        self._poll_interval = poll_interval_seconds
        self._minimum_free_bytes = round(min_free_space_gb * 1024**3)
        self._healthcheck_cache_seconds = healthcheck_cache_seconds
        self._reader = authoritative_reader or RcloneCommandReader(
            self.remote_root,
            binary=self.binary,
            command_timeout_seconds=command_timeout_seconds,
            transfer_timeout_seconds=transfer_timeout_seconds,
        )
        self._monotonic = monotonic
        self._sleep = sleep
        self._run = run_transfer if run is subprocess.run else run
        self._path_locks: WeakValueDictionary[str, threading.RLock] = WeakValueDictionary()
        self._path_locks_guard = threading.Lock()
        self._capacity_lock = threading.Lock()
        self._reserved_download_bytes = 0
        self._health_lock = threading.Lock()
        self._health_checked_at: float | None = None
        self._health_result = False

    def healthcheck(self) -> bool:
        now = self._monotonic()
        with self._health_lock:
            if (
                self._health_checked_at is not None
                and now - self._health_checked_at < self._healthcheck_cache_seconds
            ):
                return self._health_result
            try:
                enough_space = self._has_cache_capacity()
                result = enough_space and self._reader.healthcheck()
            except (OSError, StorageUnavailableError):
                result = False
            self._health_checked_at = now
            self._health_result = result
            return result

    def resolve(self, logical_path: str) -> Path:
        """Resolve a cache path, atomically fetching a missing remote object."""

        normalized = normalize_logical_path(logical_path)
        self._reject_internal_path(normalized)
        local_path = resolve_beneath(self.root, normalized)
        if local_path.exists():
            return local_path
        lock = self._path_lock(normalized)
        with transfer_lock(lock):
            local_path = resolve_beneath(self.root, normalized)
            if local_path.exists():
                return local_path
            remote_stat = self._remote_stat(normalized)
            if remote_stat is None:
                return local_path
            self._download(normalized, local_path, remote_stat)
            return resolve_beneath(self.root, normalized)

    def resolve_pinned(self, logical_path: str, *, expected_size_bytes: int) -> Path:
        """Resolve one immutable file after checking its pinned byte size.

        Unlike :meth:`resolve`, this method never accepts a directory and
        checks authoritative metadata before starting a potentially large
        transfer.  A stale regular cache file with the wrong size is removed;
        other filesystem object types are left untouched and rejected.
        """

        if isinstance(expected_size_bytes, bool) or expected_size_bytes < 0:
            raise ValueError("expected_size_bytes must be non-negative")
        normalized = normalize_logical_path(logical_path)
        self._reject_internal_path(normalized)
        lock = self._path_lock(normalized)
        with transfer_lock(lock):
            local_path = self._lexical_cache_path(normalized)
            try:
                local_status = local_path.lstat()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise StorageUnavailableError("rclone mirror cache file is unavailable") from exc
            else:
                if not stat.S_ISREG(local_status.st_mode):
                    raise StorageVerificationError(
                        "pinned cache object was not a regular non-symbolic-link file"
                    )
                if local_status.st_size == expected_size_bytes:
                    return local_path
                try:
                    local_path.unlink()
                    self._fsync_directory(local_path.parent)
                except OSError as exc:
                    raise StorageUnavailableError(
                        "stale pinned cache file could not be removed"
                    ) from exc

            remote_stat = self._remote_stat(normalized)
            if remote_stat is None:
                raise StorageUnavailableError("pinned authoritative rclone file was missing")
            if remote_stat.is_directory:
                raise StorageVerificationError(
                    "pinned authoritative rclone object was not a regular file"
                )
            if remote_stat.size_bytes != expected_size_bytes:
                raise StorageVerificationError(
                    "pinned authoritative rclone file size did not match the expected value"
                )

            download_path = resolve_beneath(self.root, normalized)
            self._download(normalized, download_path, remote_stat)
            resolved = self._lexical_cache_path(normalized)
            try:
                downloaded_status = resolved.lstat()
            except OSError as exc:
                raise StorageUnavailableError(
                    "downloaded pinned cache file was unavailable"
                ) from exc
            if (
                not stat.S_ISREG(downloaded_status.st_mode)
                or downloaded_status.st_size != expected_size_bytes
            ):
                raise StorageVerificationError("downloaded pinned cache file was invalid")
            return resolved

    def invalidate_cached(self, logical_path: str) -> None:
        """Remove one cached regular file without touching unsafe object types."""

        normalized = normalize_logical_path(logical_path)
        self._reject_internal_path(normalized)
        lock = self._path_lock(normalized)
        with lock:
            local_path = self._lexical_cache_path(normalized)
            try:
                local_status = local_path.lstat()
            except FileNotFoundError:
                return
            except OSError as exc:
                raise StorageUnavailableError("rclone mirror cache file is unavailable") from exc
            if not stat.S_ISREG(local_status.st_mode):
                raise StorageVerificationError(
                    "cached object was not a regular non-symbolic-link file"
                )
            try:
                local_path.unlink()
                self._fsync_directory(local_path.parent)
            except OSError as exc:
                raise StorageUnavailableError("cached rclone file could not be removed") from exc

    def replica_eviction_candidate(self, member):
        """Verify authority outside the resolver's short pin/delete critical section."""
        normalized = normalize_logical_path(member.logical_path)
        self._reject_internal_path(normalized)
        with transfer_lock(self._path_lock(normalized)):
            path = self._lexical_cache_path(normalized)
            try:
                before = path.lstat()
            except FileNotFoundError:
                return None
            if not stat.S_ISREG(before.st_mode):
                return None
            digest = self._reader.digest(normalized)
            if digest.size_bytes != member.size_bytes or digest.sha256 != member.sha256:
                raise StorageVerificationError("cache eviction requires verified remote bytes")
            return before

    def evict_verified_replica(self, member, candidate) -> int:
        normalized = normalize_logical_path(member.logical_path)
        self._reject_internal_path(normalized)
        lock = self._path_lock(normalized)
        if not lock.acquire(blocking=False):
            return 0
        try:
            path = self._lexical_cache_path(normalized)
            try:
                before = path.lstat()
            except FileNotFoundError:
                return 0
            if not stat.S_ISREG(before.st_mode) or before != candidate:
                return 0
            path.unlink()
            self._fsync_directory(path.parent)
            return before.st_size
        finally:
            lock.release()

    def flush_and_verify(
        self,
        logical_path: str,
        *,
        expected_sha256: str | None = None,
    ) -> VerifiedObject:
        normalized = normalize_logical_path(logical_path)
        self._reject_internal_path(normalized)
        lock = self._path_lock(normalized)
        with transfer_lock(lock):
            local_path = resolve_beneath(self.root, normalized)
            materialized = False
            if not local_path.exists():
                remote_stat = self._remote_stat(normalized)
                if remote_stat is not None:
                    self._download(normalized, local_path, remote_stat)
                    local_path = resolve_beneath(self.root, normalized)
                    materialized = True
            size = local_path.stat().st_size
            transfer_progress("hashing", 0, size)
            local_digest = fsync_and_digest(local_path)
            if expected_sha256 is not None and local_digest.sha256 != expected_sha256:
                raise StorageVerificationError("artifact checksum did not match the expected value")
            already_verified = False
            if not materialized and reusing_verified_outputs():
                check_transfer_cancelled()
                try:
                    already_verified = self._reader.digest(normalized) == local_digest
                except StorageUnavailableError:
                    pass
            if not materialized and not already_verified:
                self._upload_atomically(normalized, local_path)
            transfer_progress("verify", size, size)
            if not already_verified:
                self._verify_authoritative(normalized, local_digest)
            return VerifiedObject.now(
                logical_path=normalized,
                local_path=local_path,
                digest=local_digest,
            )

    def _path_lock(self, normalized: str) -> threading.RLock:
        # Only operations on the exact same object may serialize. Hash stripes
        # let a slow checkpoint upload block unrelated source/asset downloads.
        # Callers retain a strong reference while waiting/holding the lock;
        # weak entries disappear once there are no holders or waiters.
        with self._path_locks_guard:
            lock = self._path_locks.get(normalized)
            if lock is None:
                lock = threading.RLock()
                self._path_locks[normalized] = lock
            return lock

    def _remote_path(self, logical_path: str) -> str:
        return f"{self.remote_root}/{logical_path}"

    def _lexical_cache_path(self, logical_path: str) -> Path:
        # Validate any existing symlink chain against the cache root, but keep
        # the lexical final component so lstat can reject a symlink itself.
        resolve_beneath(self.root, logical_path)
        return self.root.joinpath(*logical_path.split("/"))

    @staticmethod
    def _reject_internal_path(logical_path: str) -> None:
        if logical_path.split("/", 1)[0] == _PARTIAL_DIRECTORY:
            raise ValueError("logical path uses the rclone mirror's reserved namespace")

    def _remote_stat(self, logical_path: str) -> _RemoteStat | None:
        try:
            completed = self._run(
                [self.binary, "lsjson", self._remote_path(logical_path), "--stat"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=rclone_process_environment(),
                timeout=self._command_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise StorageUnavailableError("authoritative rclone stat timed out") from exc
        except OSError as exc:
            raise StorageUnavailableError("authoritative rclone stat failed") from exc
        if completed.returncode == _RCLONE_NOT_FOUND:
            return None
        if completed.returncode != 0:
            raise StorageUnavailableError("authoritative rclone stat failed")
        if len(completed.stdout) > _MAX_STAT_BYTES:
            raise StorageUnavailableError("authoritative rclone stat response was too large")
        try:
            raw = json.loads(completed.stdout)
            if not isinstance(raw, dict) or not isinstance(raw.get("IsDir"), bool):
                raise ValueError
            raw_size = raw.get("Size", 0)
            if isinstance(raw_size, bool) or not isinstance(raw_size, int):
                raise ValueError
            if not raw["IsDir"] and raw_size < 0:
                raise ValueError
            size = max(0, raw_size)
            modification_time = raw.get("ModTime")
            object_id = raw.get("ID")
            if modification_time is not None and not isinstance(modification_time, str):
                raise ValueError
            if object_id is not None and not isinstance(object_id, str):
                raise ValueError
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StorageUnavailableError("authoritative rclone stat response was invalid") from exc
        return _RemoteStat(
            is_directory=raw["IsDir"],
            size_bytes=size,
            modification_time=modification_time,
            object_id=object_id,
        )

    def _download(
        self,
        logical_path: str,
        local_path: Path,
        remote_stat: _RemoteStat,
    ) -> None:
        if not remote_stat.is_directory:
            self._download_file(logical_path, local_path, remote_stat)
            return
        self._download_directory(logical_path, local_path, remote_stat)

    def _download_directory(
        self,
        logical_path: str,
        local_path: Path,
        remote_stat: _RemoteStat,
    ) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path = resolve_beneath(self.root, logical_path)
        temporary = local_path.with_name(f".{local_path.name}.topodrome-download-{uuid4().hex}")
        try:
            self._require_cache_capacity(remote_stat.size_bytes)
            temporary.mkdir(mode=0o700)
            command = [
                self.binary,
                "copy",
                self._remote_path(logical_path),
                str(temporary),
            ]
            completed = self._run_quiet(command, timeout=self._transfer_timeout)
            if completed.returncode != 0:
                raise StorageUnavailableError("authoritative rclone download failed")
            self._validate_download(temporary, expect_directory=True)
            if local_path.exists():
                return
            os.replace(temporary, local_path)
        except subprocess.TimeoutExpired as exc:
            raise StorageUnavailableError("authoritative rclone download timed out") from exc
        except OSError as exc:
            raise StorageUnavailableError("authoritative rclone download failed") from exc
        finally:
            if temporary.is_symlink() or not temporary.is_dir():
                temporary.unlink(missing_ok=True)
            else:
                shutil.rmtree(temporary, ignore_errors=True)

    def _download_file(
        self,
        logical_path: str,
        local_path: Path,
        remote_stat: _RemoteStat,
    ) -> None:
        """Resume one immutable file outside the visible cache namespace.

        The deterministic partial survives process and attempt retries. A
        sidecar binds it to the remote size plus stable rclone identity before
        any byte is reused. Remotes that publish neither an object ID nor a
        modification time still download safely, but restart from byte zero.
        """

        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path = resolve_beneath(self.root, logical_path)
        partial, metadata, lock_path = self._partial_paths(logical_path)
        with self._partial_process_lock(lock_path):
            if local_path.exists():
                return
            offset = self._prepare_partial(
                partial,
                metadata,
                logical_path=logical_path,
                remote_stat=remote_stat,
            )
            remaining = remote_stat.size_bytes - offset
            with self._reserve_download_capacity(remaining):
                if remaining:
                    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
                    flags |= getattr(os, "O_CLOEXEC", 0)
                    flags |= getattr(os, "O_NOFOLLOW", 0)
                    try:
                        descriptor = os.open(partial, flags, 0o600)
                        with os.fdopen(descriptor, "ab", buffering=0) as output:
                            try:
                                transfer_progress("downloading", offset, remote_stat.size_bytes)
                                completed = self._run_quiet(
                                    [
                                        self.binary,
                                        "cat",
                                        self._remote_path(logical_path),
                                        "--offset",
                                        str(offset),
                                        "--count",
                                        str(remaining),
                                    ],
                                    timeout=self._transfer_timeout,
                                    stdout=output,
                                    progress=lambda: transfer_progress(
                                        "downloading", os.fstat(output.fileno()).st_size,
                                        remote_stat.size_bytes,
                                    ),
                                )
                            finally:
                                os.fsync(output.fileno())
                    except subprocess.TimeoutExpired as exc:
                        raise StorageUnavailableError(
                            "authoritative rclone download timed out"
                        ) from exc
                    except OSError as exc:
                        raise StorageUnavailableError(
                            "authoritative rclone download failed"
                        ) from exc
                    if completed.returncode != 0:
                        raise StorageUnavailableError("authoritative rclone download failed")
                    transfer_progress(
                        "download-complete", remote_stat.size_bytes, remote_stat.size_bytes,
                    )

            try:
                downloaded_size = partial.stat().st_size
            except OSError as exc:
                raise StorageUnavailableError("downloaded rclone file was invalid") from exc
            if downloaded_size != remote_stat.size_bytes:
                if downloaded_size > remote_stat.size_bytes:
                    self._discard_partial(partial, metadata)
                raise StorageUnavailableError(
                    "downloaded rclone file size did not match authoritative size"
                )
            current_stat = self._remote_stat(logical_path)
            if current_stat != remote_stat:
                self._discard_partial(partial, metadata)
                raise StorageUnavailableError("authoritative object changed during download")
            self._validate_download(partial, expect_directory=False)
            if local_path.exists():
                self._discard_partial(partial, metadata)
                return
            try:
                os.replace(partial, local_path)
                self._fsync_directory(local_path.parent)
                metadata.unlink(missing_ok=True)
            except OSError as exc:
                raise StorageUnavailableError("authoritative rclone download failed") from exc

    def _validate_download(self, path: Path, *, expect_directory: bool) -> None:
        if expect_directory:
            if not path.is_dir() or path.is_symlink():
                raise StorageUnavailableError("downloaded rclone directory was invalid")
            if any(child.is_symlink() for child in path.rglob("*")):
                raise StorageUnavailableError("downloaded rclone tree contained a symlink")
            return
        if not path.is_file() or path.is_symlink():
            raise StorageUnavailableError("downloaded rclone file was invalid")

    def _partial_paths(self, logical_path: str) -> tuple[Path, Path, Path]:
        key = hashlib.sha256(f"{self.remote_root}\0{logical_path}".encode()).hexdigest()
        return (
            self._partial_root / f"{key}.part",
            self._partial_root / f"{key}.json",
            self._partial_root / f"{key}.lock",
        )

    @contextmanager
    def _partial_process_lock(self, lock_path: Path) -> Iterator[None]:
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise StorageUnavailableError("rclone partial lock is unavailable") from exc
        try:
            try:
                while True:
                    check_transfer_cancelled()
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.2)
            except OSError as exc:
                raise StorageUnavailableError("rclone partial lock is unavailable") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _prepare_partial(
        self,
        partial: Path,
        metadata: Path,
        *,
        logical_path: str,
        remote_stat: _RemoteStat,
    ) -> int:
        expected = {
            "schema_version": _PARTIAL_METADATA_SCHEMA,
            "remote_root": self.remote_root,
            "logical_path": logical_path,
            **remote_stat.resumable_identity,
        }
        valid = False
        if partial.exists() and metadata.exists() and remote_stat.has_stable_identity:
            try:
                partial_status = partial.lstat()
                metadata_status = metadata.lstat()
                if not stat.S_ISREG(partial_status.st_mode) or not stat.S_ISREG(
                    metadata_status.st_mode
                ):
                    raise ValueError
                if metadata_status.st_size > _MAX_PARTIAL_METADATA_BYTES:
                    raise ValueError
                current = json.loads(metadata.read_text(encoding="utf-8"))
                valid = current == expected and partial_status.st_size <= remote_stat.size_bytes
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                valid = False
        if not valid:
            self._discard_partial(partial, metadata)
            self._write_partial_metadata(metadata, expected)
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(partial, flags, 0o600)
                os.close(descriptor)
            except OSError as exc:
                raise StorageUnavailableError("rclone partial file is unavailable") from exc
            return 0
        return partial.stat().st_size

    @staticmethod
    def _write_partial_metadata(metadata: Path, payload: Mapping[str, object]) -> None:
        temporary = metadata.with_name(f".{metadata.name}.tmp-{uuid4().hex}")
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, metadata)
        except OSError as exc:
            raise StorageUnavailableError("rclone partial metadata is unavailable") from exc
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _discard_partial(partial: Path, metadata: Path) -> None:
        for path in (partial, metadata):
            try:
                status = path.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise StorageUnavailableError("rclone partial state is unavailable") from exc
            if stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode):
                raise StorageUnavailableError("rclone partial state is not a regular file")
            try:
                path.unlink()
            except OSError as exc:
                raise StorageUnavailableError("rclone partial state is unavailable") from exc

    @contextmanager
    def _reserve_download_capacity(self, size_bytes: int) -> Iterator[None]:
        if size_bytes < 0:
            raise StorageUnavailableError("authoritative rclone file size was invalid")
        with self._capacity_lock:
            required = self._reserved_download_bytes + size_bytes
            if not self._has_cache_capacity(required):
                raise StorageUnavailableError("rclone mirror cache is below minimum free space")
            self._reserved_download_bytes += size_bytes
        try:
            yield
        finally:
            with self._capacity_lock:
                self._reserved_download_bytes -= size_bytes

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(directory, flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise StorageUnavailableError(
                "downloaded rclone file could not be synchronized"
            ) from exc

    def _upload_atomically(self, logical_path: str, local_path: Path) -> None:
        parent, _, name = logical_path.rpartition("/")
        temporary_name = f".{name}.topodrome-upload-{uuid4().hex}"
        temporary_logical = f"{parent}/{temporary_name}" if parent else temporary_name
        remote_temporary = self._remote_path(temporary_logical)
        promoted = False
        try:
            command = [self.binary, "copyto", str(local_path), remote_temporary]
            size = local_path.stat().st_size
            transfer_progress("upload", 0, size)
            if observing_transfers() and self._run is run_transfer:
                copied = run_upload(command, timeout=self._transfer_timeout, size=size)
            else:
                copied = self._run_quiet(command, timeout=self._transfer_timeout)
            if copied.returncode != 0:
                raise StorageVerificationError("artifact upload to rclone failed")
            transfer_progress("promote", size, size)
            moved = self._run_quiet(
                [
                    self.binary,
                    "moveto",
                    remote_temporary,
                    self._remote_path(logical_path),
                    "--ignore-times",
                ]
            )
            if moved.returncode != 0:
                raise StorageVerificationError("artifact promotion on rclone failed")
            promoted = True
        except subprocess.TimeoutExpired as exc:
            raise StorageVerificationError("artifact upload to rclone timed out") from exc
        except OSError as exc:
            raise StorageVerificationError("artifact upload to rclone failed") from exc
        finally:
            if not promoted:
                try:
                    self._run_quiet([self.binary, "deletefile", remote_temporary])
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def _verify_authoritative(
        self,
        logical_path: str,
        local_digest: ObjectDigest,
    ) -> None:
        deadline = self._monotonic() + self._verification_timeout
        last_observation: ObjectDigest | None = None
        while True:
            check_transfer_cancelled()
            try:
                last_observation = self._reader.digest(logical_path)
            except StorageUnavailableError:
                last_observation = None
            if last_observation == local_digest:
                return
            if self._monotonic() >= deadline:
                if last_observation is None:
                    raise StorageUnavailableError("authoritative verification unavailable")
                detail = "missing" if last_observation is None else "different content"
                raise StorageVerificationError(
                    f"authoritative object was {detail} after the verification timeout"
                )
            self._sleep(self._poll_interval)

    def _run_quiet(
        self,
        command: list[str],
        *,
        timeout: float | None = None,
        stdout: BinaryIO | int = subprocess.DEVNULL,
        progress: Callable[[], None] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        check_transfer_cancelled()
        return self._run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.DEVNULL,
            env=rclone_process_environment(),
            timeout=self._command_timeout if timeout is None else timeout,
            check=False,
            **({"progress": progress} if self._run is run_transfer else {}),
        )

    def _has_cache_capacity(self, additional_bytes: int = 0) -> bool:
        if not self.root.is_dir() or not os.access(self.root, os.R_OK | os.W_OK | os.X_OK):
            return False
        try:
            free = shutil.disk_usage(self.root).free
        except OSError:
            return False
        return free - max(0, additional_bytes) >= self._minimum_free_bytes

    def _require_cache_capacity(self, additional_bytes: int = 0) -> None:
        if not self._has_cache_capacity(additional_bytes):
            raise StorageUnavailableError("rclone mirror cache is below minimum free space")


__all__ = ["RcloneMirrorStorage"]
