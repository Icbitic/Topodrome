"""Verified, resumable HTTP mirror backed by the Lab1 storage authority."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from weakref import WeakValueDictionary

from topodrome.storage.base import (
    AuthoritativeDigestReader,
    ObjectDigest,
    StorageUnavailableError,
    StorageVerificationError,
    VerifiedObject,
)
from topodrome.storage.bulk import BulkFileTransport, BulkTransportError
from topodrome.storage.local import fsync_and_digest
from topodrome.storage.paths import (
    normalize_logical_path,
    resolve_beneath,
    resolve_writable_beneath,
)
from topodrome.storage.progress import (
    bounded_transfer,
    check_transfer_cancelled,
    reusing_verified_outputs,
    transfer_lock,
    transfer_progress,
)
from topodrome.storage.small_files import SchedulerSmallFileClient, SmallFileFallback
from topodrome.storage.source_archive import source_bundle_path, unpack_source_bundle

_MAX_PARTIAL_METADATA_BYTES = 16 * 1024
_PARTIAL_METADATA_SCHEMA = 1
_PARTIAL_DIRECTORY = ".topodrome-mirror-partials"
_UPLOAD_DIRECTORY = ".topodrome-mirror-uploads"


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


class HttpMirrorStorage:
    """One verified cache, with mandatory persistent HTTP storage sessions.

    This adapter is intended for remote sites where FUSE is unavailable.  A
    missing local object is downloaded atomically from ``remote_root``.  Files
    written into the returned local namespace become durable only after
    :meth:`flush_and_verify`, which uploads through a unique remote temporary
    name, promotes it, and verifies it independently at the authority.

    Writable workspaces use :meth:`resolve_writable`, never downloaded directory
    views. The canonical write path is also the source of ``flush_and_verify``;
    immutable input caches cannot redirect a log, checkpoint or output writer.

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
        transfer_timeout_seconds: float = 6 * 60 * 60,
        verification_timeout_seconds: float = 120,
        poll_interval_seconds: float = 0.25,
        min_free_space_gb: float = 10,
        healthcheck_cache_seconds: float = 5,
        authoritative_reader: AuthoritativeDigestReader,
        small_file_client: SchedulerSmallFileClient | None = None,
        bulk_transport: BulkFileTransport,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            not remote_root
            or len(remote_root) > 4096
            or any(character in remote_root for character in "\r\n\0")
        ):
            raise ValueError("remote_root must identify one credential-free authority")
        for value, label in (
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

        self.root = Path(cache_root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        # Keep partials on the cache filesystem so final promotion is one atomic
        # rename even when ``root`` itself is a mount point.
        partial_root = self.root / _PARTIAL_DIRECTORY
        if partial_root.is_symlink():
            raise ValueError("HTTP partial directory must not be a symbolic link")
        partial_root.mkdir(mode=0o700, exist_ok=True)
        if not partial_root.is_dir() or partial_root.is_symlink():
            raise ValueError("HTTP partial directory must be a private directory")
        partial_status = partial_root.stat()
        if partial_status.st_uid != os.geteuid():
            raise ValueError("HTTP partial directory must be owned by the site agent")
        if stat.S_IMODE(partial_status.st_mode) & 0o077:
            partial_root.chmod(0o700)
        self._partial_root = partial_root.resolve()
        upload_root = self.root / _UPLOAD_DIRECTORY
        if upload_root.is_symlink():
            raise ValueError("upload receipt directory must not be a symbolic link")
        upload_root.mkdir(mode=0o700, exist_ok=True)
        info = upload_root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            raise ValueError("upload receipts must be owned by the site agent")
        upload_root.chmod(0o700)
        self._upload_root = upload_root
        self.remote_root = remote_root.rstrip("/")
        self._transfer_timeout = transfer_timeout_seconds
        self._verification_timeout = verification_timeout_seconds
        self._poll_interval = poll_interval_seconds
        self._minimum_free_bytes = round(min_free_space_gb * 1024**3)
        self._healthcheck_cache_seconds = healthcheck_cache_seconds
        self._reader = authoritative_reader
        self._small_files = small_file_client
        self._bulk = bulk_transport
        self._monotonic = monotonic
        self._sleep = sleep
        self._path_locks: WeakValueDictionary[str, threading.RLock] = WeakValueDictionary()
        self._path_locks_guard = threading.Lock()
        self._capacity_lock = threading.Lock()
        self._reserved_download_bytes = 0
        self._health_lock = threading.Lock()
        self._health_checked_at: float | None = None
        self._health_result = False

    def close(self) -> None:
        try:
            self._bulk.close()
        finally:
            if self._small_files is not None:
                self._small_files.close()
            if self._reader is not self._bulk:
                close = getattr(self._reader, "close", None)
                if close is not None:
                    close()

    def publish_small_blob(
        self, source: Path, *, job_id: str, attempt_id: str, digest: ObjectDigest
    ) -> VerifiedObject | None:
        """Combine claim, transfer and authority verification into one request."""
        from topodrome.storage.blobs import blob_path, install_local_blob

        path = blob_path(digest.sha256)
        if self._small_files is None or not self._small_files.accepts(path, digest.size_bytes):
            return None
        with transfer_lock(self._path_lock(path)):
            local = install_local_blob(self.root, source, digest)
            try:
                verified = self._retry_bulk(
                    lambda: self._small_files.upload(
                        path,
                        local,
                        digest,
                        job_id=job_id,
                        attempt_id=attempt_id,
                    )
                )
            except SmallFileFallback:
                return None
            return VerifiedObject.now(logical_path=path, local_path=local, digest=verified)

    def _download_small(
        self,
        normalized: str,
        target: Path,
        *,
        expected_size_bytes: int | None = None,
        expected_sha256: str | None = None,
    ) -> bool | None:
        """False is missing, None is bulk/old server, True is verified and durable."""
        client = self._small_files
        if client is None or not client.accepts(normalized, expected_size_bytes):
            return None
        capacity = expected_size_bytes if expected_size_bytes is not None else client.max_bytes
        with self._reserve_download_capacity(capacity):
            try:
                content, digest = self._retry_bulk(
                    lambda: client.download(
                        normalized,
                        expected_size_bytes=expected_size_bytes,
                        expected=(
                            ObjectDigest(expected_size_bytes, expected_sha256)
                            if expected_sha256 is not None and expected_size_bytes is not None
                            else None
                        ),
                    )
                )
            except SmallFileFallback:
                return None
            except FileNotFoundError:
                return False
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.http-{uuid4().hex}")
            try:
                with temporary.open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                if fsync_and_digest(temporary) != digest:
                    raise StorageVerificationError(
                        "small-file cache publication failed verification"
                    )
                check_transfer_cancelled()
                os.replace(temporary, target)
                self._fsync_directory(target.parent)
            finally:
                temporary.unlink(missing_ok=True)
            return True

    def _download_source(self, normalized: str, target: Path) -> bool | None:
        if self._small_files is None or not source_bundle_path(normalized):
            return None
        with self._reserve_download_capacity(self._small_files.max_bytes):
            try:
                content = self._retry_bulk(
                    lambda: self._small_files.download_source_bundle(normalized)
                )
            except SmallFileFallback:
                return None
            except FileNotFoundError:
                return False
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.http-{uuid4().hex}")
            temporary.mkdir(mode=0o700)
            try:
                unpack_source_bundle(normalized, content, temporary, self._small_files.max_bytes)
                check_transfer_cancelled()
                os.replace(temporary, target)
                self._fsync_directory(target.parent)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return True

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
                result = enough_space and (
                    self._bulk.healthcheck()
                    if self._bulk is not None
                    else self._reader.healthcheck()
                )
            except (OSError, StorageUnavailableError):
                result = False
            self._health_checked_at = now
            self._health_result = result
            return result

    def resolve_writable(self, logical_path: str) -> Path:
        """Return a canonical local path without inspecting remote input caches."""
        normalized = normalize_logical_path(logical_path)
        self._reject_internal_path(normalized)
        return resolve_writable_beneath(self.root, normalized)

    def resolve(self, logical_path: str) -> Path:
        """Resolve a cache path, atomically fetching a missing remote object."""

        normalized = normalize_logical_path(logical_path)
        self._reject_internal_path(normalized)
        local_path = resolve_beneath(self.root, normalized)
        directory_cache = self._directory_cache_path(normalized)
        if directory_cache.exists():
            self._validate_download(directory_cache, expect_directory=True)
            return directory_cache
        if local_path.is_file():
            return local_path
        lock = self._path_lock(normalized)
        with transfer_lock(lock):
            local_path = resolve_beneath(self.root, normalized)
            if directory_cache.exists():
                self._validate_download(directory_cache, expect_directory=True)
                return directory_cache
            if local_path.is_file():
                return local_path
            fast = (
                self._download_source(normalized, directory_cache)
                if source_bundle_path(normalized)
                else self._download_small(normalized, local_path)
            )
            if fast is not None:
                return directory_cache if fast and source_bundle_path(normalized) else local_path
            remote_stat = self._remote_stat(normalized)
            if remote_stat is None:
                return local_path
            self._download(normalized, local_path, remote_stat)
            return (
                directory_cache
                if remote_stat.is_directory
                else resolve_beneath(self.root, normalized)
            )

    def resolve_pinned(
        self, logical_path: str, *, expected_size_bytes: int, expected_sha256: str | None = None
    ) -> Path:
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
                raise StorageUnavailableError("HTTP mirror cache file is unavailable") from exc
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

            fast = self._download_small(
                normalized,
                resolve_beneath(self.root, normalized),
                expected_size_bytes=expected_size_bytes,
                expected_sha256=expected_sha256,
            )
            if fast is False:
                raise StorageUnavailableError("pinned authoritative file was missing")
            if fast is True:
                return self._lexical_cache_path(normalized)
            remote_stat = self._remote_stat(normalized)
            if remote_stat is None:
                raise StorageUnavailableError("pinned authoritative HTTP file was missing")
            if remote_stat.is_directory:
                raise StorageVerificationError(
                    "pinned authoritative HTTP object was not a regular file"
                )
            if remote_stat.size_bytes != expected_size_bytes:
                raise StorageVerificationError(
                    "pinned authoritative HTTP file size did not match the expected value"
                )

            download_path = resolve_beneath(self.root, normalized)
            self._download(normalized, download_path, remote_stat, expected_sha256=expected_sha256)
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
                raise StorageUnavailableError("HTTP mirror cache file is unavailable") from exc
            if not stat.S_ISREG(local_status.st_mode):
                raise StorageVerificationError(
                    "cached object was not a regular non-symbolic-link file"
                )
            try:
                local_path.unlink()
                self._fsync_directory(local_path.parent)
            except OSError as exc:
                raise StorageUnavailableError("cached HTTP file could not be removed") from exc

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
            if not materialized and self._small_files is not None:
                try:
                    verified = self._retry_bulk(
                        lambda: self._small_files.upload(normalized, local_path, local_digest)
                    )
                except SmallFileFallback:
                    pass
                else:
                    return VerifiedObject.now(
                        logical_path=normalized,
                        local_path=local_path,
                        digest=verified,
                    )
            already_verified = False
            if not materialized and reusing_verified_outputs():
                check_transfer_cancelled()
                try:
                    already_verified = self._reader.digest(normalized) == local_digest
                except StorageUnavailableError:
                    pass
            if not materialized and not already_verified:
                already_verified = self._upload_atomically(normalized, local_path, local_digest)
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
        if logical_path.split("/", 1)[0] in {_PARTIAL_DIRECTORY, _UPLOAD_DIRECTORY}:
            raise ValueError("logical path uses the HTTP mirror's reserved namespace")

    def _remote_stat(self, logical_path: str) -> _RemoteStat | None:
        raw = self._bulk.stat(logical_path)
        return None if raw is None else self._parse_stat(raw)

    @staticmethod
    def _parse_stat(raw) -> _RemoteStat:
        try:
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
        except (TypeError, ValueError) as exc:
            raise StorageUnavailableError("authoritative HTTP stat response was invalid") from exc
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
        *,
        expected_sha256: str | None = None,
    ) -> None:
        if not remote_stat.is_directory:
            self._retry_bulk(
                lambda: self._download_file(
                    logical_path, local_path, remote_stat, expected_sha256=expected_sha256
                )
            )
            return
        self._download_directory(logical_path, local_path, remote_stat)

    def _download_directory(
        self, logical_path: str, local_path: Path, remote_stat: _RemoteStat
    ) -> None:
        # An individually cached child creates its parent directory, but that
        # does NOT prove the whole remote tree is cached. Publish whole trees
        # atomically in their own exact-key namespace without moving live files.
        local_path = self._directory_cache_path(logical_path)
        partial, _metadata, lock_path = self._partial_paths(logical_path)
        tree = partial.with_suffix(".tree")
        with self._partial_process_lock(lock_path):
            if local_path.exists():
                return
            if tree.is_symlink():
                raise StorageVerificationError("unsafe HTTP tree partial")
            tree.mkdir(mode=0o700, exist_ok=True)
            # Keep privately staged members on cancellation/link failure. Each
            # member is independently verified before a retry reuses it.
            self._retry_bulk(lambda: self._bulk.download_tree(logical_path, tree))
            if self._remote_stat(logical_path) != remote_stat:
                raise StorageVerificationError("HTTP source tree changed during download")
            self._validate_download(tree, expect_directory=True)
            for path in tree.rglob("*"):
                if path.is_dir():
                    self._fsync_directory(path)
                else:
                    fsync_and_digest(path)
            self._fsync_directory(tree)
            check_transfer_cancelled()
            os.replace(tree, local_path)
            self._fsync_directory(local_path.parent)

    def _download_file(
        self,
        logical_path: str,
        local_path: Path,
        remote_stat: _RemoteStat,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        """Resume one immutable file outside the visible cache namespace.

        The deterministic partial survives process and attempt retries. A
        sidecar binds it to the remote size plus stable authority identity before
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
                                self._bulk.read(
                                    logical_path,
                                    output,
                                    offset,
                                    remaining,
                                    expected_identity=remote_stat.object_id,
                                )
                            finally:
                                os.fsync(output.fileno())
                    except OSError as exc:
                        raise StorageUnavailableError("authoritative HTTP download failed") from exc
                    transfer_progress(
                        "download-complete",
                        remote_stat.size_bytes,
                        remote_stat.size_bytes,
                    )

            try:
                downloaded_size = partial.stat().st_size
            except OSError as exc:
                raise StorageUnavailableError("downloaded HTTP file was invalid") from exc
            if downloaded_size != remote_stat.size_bytes:
                if downloaded_size > remote_stat.size_bytes:
                    self._discard_partial(partial, metadata)
                raise StorageUnavailableError(
                    "downloaded HTTP file size did not match authoritative size"
                )
            current_stat = self._remote_stat(logical_path)
            if current_stat != remote_stat:
                self._discard_partial(partial, metadata)
                raise StorageUnavailableError("authoritative object changed during download")
            self._validate_download(partial, expect_directory=False)
            observed = fsync_and_digest(partial)
            expected = (
                ObjectDigest(remote_stat.size_bytes, expected_sha256)
                if expected_sha256 is not None
                else self._reader.digest(logical_path)
            )
            if expected is None or observed != expected:
                self._discard_partial(partial, metadata)
                raise StorageVerificationError("HTTP download failed checksum verification")
            if local_path.exists():
                self._discard_partial(partial, metadata)
                return
            try:
                os.replace(partial, local_path)
                self._fsync_directory(local_path.parent)
                metadata.unlink(missing_ok=True)
            except OSError as exc:
                raise StorageUnavailableError("authoritative HTTP download failed") from exc

    def _validate_download(self, path: Path, *, expect_directory: bool) -> None:
        if expect_directory:
            if not path.is_dir() or path.is_symlink():
                raise StorageUnavailableError("downloaded HTTP directory was invalid")
            if any(
                not (stat.S_ISREG(child.lstat().st_mode) or stat.S_ISDIR(child.lstat().st_mode))
                for child in path.rglob("*")
            ):
                raise StorageVerificationError("downloaded HTTP tree contained an unsafe member")
            return
        if not path.is_file() or path.is_symlink():
            raise StorageUnavailableError("downloaded HTTP file was invalid")

    def _partial_paths(self, logical_path: str) -> tuple[Path, Path, Path]:
        key = hashlib.sha256(f"{self.remote_root}\0{logical_path}".encode()).hexdigest()
        return (
            self._partial_root / f"{key}.part",
            self._partial_root / f"{key}.json",
            self._partial_root / f"{key}.lock",
        )

    def _directory_cache_path(self, logical_path: str) -> Path:
        partial, _, _ = self._partial_paths(logical_path)
        return partial.with_suffix(".tree-cache")

    @contextmanager
    def _partial_process_lock(self, lock_path: Path) -> Iterator[None]:
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise StorageUnavailableError("HTTP partial lock is unavailable") from exc
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
                raise StorageUnavailableError("HTTP partial lock is unavailable") from exc
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
                raise StorageUnavailableError("HTTP partial file is unavailable") from exc
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
            HttpMirrorStorage._fsync_directory(metadata.parent)
        except OSError as exc:
            raise StorageUnavailableError("HTTP partial metadata is unavailable") from exc
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
                raise StorageUnavailableError("HTTP partial state is unavailable") from exc
            if stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode):
                raise StorageUnavailableError("HTTP partial state is not a regular file")
            try:
                path.unlink()
            except OSError as exc:
                raise StorageUnavailableError("HTTP partial state is unavailable") from exc

    @contextmanager
    def _reserve_download_capacity(self, size_bytes: int) -> Iterator[None]:
        if size_bytes < 0:
            raise StorageUnavailableError("authoritative HTTP file size was invalid")
        with self._capacity_lock:
            required = self._reserved_download_bytes + size_bytes
            if not self._has_cache_capacity(required):
                raise StorageUnavailableError("HTTP mirror cache is below minimum free space")
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
            raise StorageUnavailableError("downloaded HTTP file could not be synchronized") from exc

    def _upload_atomically(self, logical_path: str, local_path: Path, digest: ObjectDigest) -> bool:
        self._retry_bulk(lambda: self._upload_resumable(logical_path, local_path, digest))
        return True

    def _retry_bulk(self, operation):
        """Reconnect/resume known link failures without retrying user computation.

        Retries re-enter the journaled mirror operation, not a blind write or
        rename. Authentication, host keys, integrity and local disk failures
        never downgrade or auto-retry. Retain partial state when the bounded
        three-attempt network budget is exhausted for the caller's later
        recovery. Explicit authority backpressure has a separate, cancellable
        two-minute waiting budget; normal concurrent transfers are not failed
        as though user computation or the network were broken.
        """
        with bounded_transfer(self._transfer_timeout, clock=self._monotonic):
            return self._retry_bulk_within_budget(operation)

    def _retry_bulk_within_budget(self, operation):
        attempt = 0
        busy_deadline = self._monotonic() + min(120, self._transfer_timeout)
        while attempt < 3:
            check_transfer_cancelled()
            try:
                return operation()
            except BulkTransportError as exc:
                if not exc.retryable:
                    raise
                if exc.retry_after_seconds is not None:
                    if self._monotonic() >= busy_deadline:
                        raise
                    pause = min(exc.retry_after_seconds, busy_deadline - self._monotonic())
                else:
                    attempt += 1
                    if attempt == 3:
                        raise
                    pause = 0.1 if attempt == 1 else 0.5
                pause_until = self._monotonic() + pause
                while self._monotonic() < pause_until:
                    check_transfer_cancelled()
                    self._sleep(min(0.1, pause_until - self._monotonic()))
        raise AssertionError("bulk retry budget did not terminate")

    def _upload_resumable(self, logical_path: str, source: Path, digest: ObjectDigest) -> None:
        """Persist ownership of one private prefix; never trust its length alone.

        The local receipt is durable before any remote byte is written. A retry
        rehashes the authority's prefix and the corresponding local prefix.
        Network failure/cancellation retains this owned state for recovery.
        """
        key = hashlib.sha256(f"{self.remote_root}\0{logical_path}".encode()).hexdigest()
        receipt = self._upload_root / f"{key}.json"
        lock_path = self._upload_root / f"{key}.lock"
        expected = {
            "schema_version": 1,
            "remote_root": self.remote_root,
            "logical_path": logical_path,
            "size_bytes": digest.size_bytes,
            "sha256": digest.sha256,
        }
        parent, _, name = logical_path.rpartition("/")
        prefix = f"{parent}/" if parent else ""
        template = prefix + f".{name}.topodrome-upload-"
        with self._partial_process_lock(lock_path):
            current = None
            try:
                info = receipt.lstat()
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_PARTIAL_METADATA_BYTES:
                    raise StorageVerificationError("unsafe upload receipt")
                try:
                    current = json.loads(receipt.read_text(encoding="utf-8"))
                except (ValueError, UnicodeError):
                    raise StorageVerificationError("invalid upload receipt") from None
                temporary = current.get("temporary") if isinstance(current, dict) else None
                if (
                    not isinstance(temporary, str)
                    or not temporary.startswith(template)
                    or not re.fullmatch(r"[a-f0-9]{32}", temporary[len(template) :])
                ):
                    raise StorageVerificationError("upload receipt does not own its prefix")
                # Remove only the exact, validated sibling owned by this receipt.
                if {k: v for k, v in current.items() if k != "temporary"} != expected:
                    self._bulk.remove(temporary)
                    current = None
            if current is None:
                current = {**expected, "temporary": template + uuid4().hex}
                self._write_partial_metadata(receipt, current)
            temporary = current["temporary"]
            remote = self._remote_stat(temporary)
            if remote is None and self._reader.digest(logical_path) == digest:
                # The earlier promotion may have succeeded despite losing its
                # acknowledgement. Do not resend a multi-GiB completed output.
                receipt.unlink(missing_ok=True)
                self._fsync_directory(receipt.parent)
                return
            offset = 0
            if remote is not None:
                if remote.is_directory:
                    raise StorageVerificationError("upload prefix was not a regular file")
                if remote.size_bytes <= digest.size_bytes:
                    observed = self._reader.digest(temporary)
                    # Independent verification may fail transiently. In that
                    # case leave the prefix untouched, rather than retransmit it.
                    if observed is None or observed.size_bytes != remote.size_bytes:
                        raise StorageUnavailableError("upload prefix verification unavailable")
                    if observed.sha256 == self._prefix_digest(source, remote.size_bytes):
                        offset = remote.size_bytes
                if offset != remote.size_bytes:
                    self._bulk.remove(temporary)
            transfer_progress("upload", offset, digest.size_bytes)
            if offset < digest.size_bytes or remote is None:
                self._bulk.write(temporary, source, offset, digest.size_bytes)
            # A source altered while being sent can never become visible.
            self._verify_authoritative(temporary, digest)
            check_transfer_cancelled()
            transfer_progress("promote", digest.size_bytes, digest.size_bytes)
            self._bulk.rename(temporary, logical_path)
            # fsync of the final authority namespace happens before the receipt
            # is retired, including after a lost rename acknowledgement.
            self._verify_authoritative(logical_path, digest)
            receipt.unlink(missing_ok=True)
            self._fsync_directory(receipt.parent)

    @staticmethod
    def _prefix_digest(source: Path, size: int) -> str:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            remaining = size
            while remaining:
                check_transfer_cancelled()
                content = stream.read(min(4 * 1024**2, remaining))
                if not content:
                    raise StorageVerificationError("upload source prefix was truncated")
                digest.update(content)
                remaining -= len(content)
        return digest.hexdigest()

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
            except BulkTransportError as exc:
                if not exc.retryable:
                    raise
                last_observation = None
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
            raise StorageUnavailableError("HTTP mirror cache is below minimum free space")


__all__ = ["HttpMirrorStorage"]
