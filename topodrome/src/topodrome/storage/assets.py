"""Lazy, content-addressed materialization of immutable global assets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

from topodrome.domain.specs import AssetMember, AssetRef
from topodrome.storage.base import StorageAdapter
from topodrome.storage.paths import normalize_logical_path, resolve_beneath
from topodrome.storage.progress import check_transfer_cancelled, transfer_lock, transfer_progress

_READ_SIZE = 4 * 1024 * 1024
_LOCK_STRIPES = 64
_MAX_CACHE_MANIFEST_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AssetNotFoundError(FileNotFoundError):
    """A pinned global asset member is unavailable at this site."""


class AssetIntegrityError(RuntimeError):
    """Materialized asset bytes do not match the scheduler's immutable pin."""


class UnpinnedAssetError(AssetIntegrityError):
    """A mutable asset name reached a site agent without scheduler pinning."""


class _RetryableAssetIntegrityError(AssetIntegrityError):
    """A cached source may be stale and can be invalidated exactly once."""

    def __init__(self, message: str, *, logical_path: str | None = None) -> None:
        super().__init__(message)
        self.logical_path = logical_path


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    reference: AssetRef
    local_path: Path
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class _SourceStat:
    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _SourceStat:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }


def asset_manifest_digest(
    members: Iterable[AssetMember | Mapping[str, object]],
) -> str:
    """Return the stable SHA-256 identity of a relative-path content tree.

    Global logical storage paths are intentionally excluded: two published
    names containing the same immutable tree share one site-local cache entry.
    """

    parsed = tuple(
        member if isinstance(member, AssetMember) else AssetMember.model_validate(member)
        for member in members
    )
    payload = {
        "schema_version": 1,
        "members": [
            {
                "relative_path": member.relative_path,
                "size_bytes": member.size_bytes,
                "sha256": member.sha256,
            }
            for member in sorted(parsed, key=lambda item: item.relative_path)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AssetResolver:
    """Materialize scheduler-pinned assets only when an assigned job needs them.

    The cache commit point is an atomic directory rename containing a private
    payload tree and a manifest marker. Concurrent jobs in this agent share a
    striped lock, and later jobs validate cheap immutable metadata rather than
    rehashing multi-GiB files.
    """

    def __init__(
        self,
        storage: StorageAdapter,
        *,
        cache_root: Path | str | None = None,
    ) -> None:
        self._storage = storage
        if cache_root is None:
            storage_root = getattr(storage, "root", None)
            if storage_root is None:
                raise ValueError("asset cache_root is required for this storage adapter")
            cache_root = Path(storage_root) / ".topodrome-asset-cache"
        self._cache_root = Path(cache_root).expanduser().resolve()
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._locks = tuple(threading.RLock() for _ in range(_LOCK_STRIPES))
        self._verified: dict[str, AssetRef] = {}
        self._verified_lock = threading.RLock()
        self._pins: dict[str, tuple[AssetRef, ...]] = {}
        self._maintenance_lock = threading.RLock()
        self._access_root = self._cache_root / "access"
        self._access_root.mkdir(mode=0o700, exist_ok=True)
        self._discover_cache()

    def pin(self, owner: str, references: Iterable[AssetRef]) -> None:
        references = tuple(references)
        if any(not ref.pinned for ref in references):
            raise UnpinnedAssetError("cache pins require immutable references")
        with self._maintenance_lock:
            self._pins[owner] = references
            for ref in references:
                (self._access_root / ref.cache_key).touch(mode=0o600)

    def unpin(self, owner: str) -> None:
        with self._maintenance_lock:
            for ref in self._pins.pop(owner, ()):
                (self._access_root / ref.cache_key).touch(mode=0o600)

    def collect(self, *, max_age_seconds: float, max_bytes: int, now=None) -> int:
        """Remove unused verified remote asset replicas, never authoritative files."""
        evict = getattr(self._storage, "evict_verified_replica", None)
        verify = getattr(self._storage, "replica_eviction_candidate", None)
        if not callable(evict) or not callable(verify):
            return 0
        now = time.time() if now is None else now
        freed = 0
        with self._maintenance_lock:
            with self._verified_lock:
                refs = tuple(self._verified.values())
            candidates = []
            for ref in refs:
                access = self._access_root / ref.cache_key
                if not access.exists():
                    access.touch(mode=0o600)  # Full grace for legacy entries.
                candidates.append((access.stat().st_mtime, ref))
            total = sum(ref.total_bytes for ref in refs)
        for touched, ref in sorted(candidates, key=lambda item: item[0]):
            check_transfer_cancelled()
            if now - touched < max_age_seconds and total <= max_bytes:
                continue
            with self._maintenance_lock:
                if any(a.cache_key == ref.cache_key for refs in self._pins.values() for a in refs):
                    continue
            # Network verification must never block pinning or lease renewals.
            verified = [(member, verify(member)) for member in ref.members]
            with self._maintenance_lock:
                protected = {a.cache_key for refs in self._pins.values() for a in refs}
                protected_paths = {m.logical_path for refs in self._pins.values()
                                   for a in refs for m in a.members}
                if ref.cache_key in protected:
                    continue
                lock = self._lock_for_cache_key(ref.cache_key)
                if not lock.acquire(blocking=False):
                    continue
                try:
                    if not self._cache_is_complete(self._cache_directory(ref), ref):
                        continue
                    for member, candidate in verified:
                        if candidate is not None and member.logical_path not in protected_paths:
                            freed += evict(member, candidate)
                    _remove_cache_tree(self._cache_directory(ref))
                    self._forget(ref.cache_key)
                    (self._access_root / ref.cache_key).unlink(missing_ok=True)
                    total -= ref.total_bytes
                finally:
                    lock.release()
        return freed

    @property
    def cache_root(self) -> Path:
        return self._cache_root

    def resolve(self, reference: AssetRef | str | dict[str, object]) -> ResolvedAsset:
        parsed = AssetRef.parse(reference)
        owner = "resolve-" + uuid4().hex
        self.pin(owner, (parsed,))
        try:
            return self._resolve(parsed)
        finally:
            self.unpin(owner)

    def _resolve(self, parsed: AssetRef) -> ResolvedAsset:
        if not parsed.pinned:
            raise UnpinnedAssetError("site agents require scheduler-pinned asset references")
        assert parsed.digest is not None
        assert parsed.total_bytes is not None
        if asset_manifest_digest(parsed.members) != parsed.digest:
            raise AssetIntegrityError("asset manifest digest does not match its members")
        _validate_relative_tree(parsed.members)

        assert parsed.kind is not None
        cache_key = parsed.cache_key
        assert cache_key is not None
        target = self._cache_directory(parsed)
        lock = self._lock_for_cache_key(cache_key)
        with transfer_lock(lock):
            if self._cache_is_complete(target, parsed):
                self._remember(parsed)
                return ResolvedAsset(
                    reference=parsed,
                    local_path=_exposed_path(target, parsed),
                    cache_hit=True,
                )
            if target.exists() or target.is_symlink():
                _remove_cache_tree(target)
            self._forget(cache_key)
            for attempt in range(2):
                temporary = target.with_name(f".{target.name}.materializing-{uuid4().hex}")
                try:
                    payload = temporary / "payload"
                    payload.mkdir(parents=True, mode=0o700)
                    transfer_progress("input-view-start", 0, parsed.total_bytes)
                    source_stats = {}
                    for member in sorted(parsed.members, key=lambda item: item.relative_path):
                        source_stats[member.relative_path] = self._materialize_member(
                            member, payload,
                        )
                        transfer_progress(
                            "input-member-ready", member.size_bytes, member.size_bytes,
                        )
                    marker = _cache_marker(parsed, source_stats)
                    marker_path = temporary / "manifest.json"
                    marker_path.write_bytes(marker)
                    marker_path.chmod(0o444)
                    if not self._cache_is_complete(temporary, parsed):
                        raise _RetryableAssetIntegrityError(
                            "asset source changed before cache publication"
                        )
                    _make_tree_read_only(payload)
                    temporary.chmod(0o555)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary, target)
                except _RetryableAssetIntegrityError as exc:
                    _remove_cache_tree(temporary)
                    if attempt == 0 and self._invalidate_cached(parsed, exc.logical_path):
                        continue
                    raise
                except BaseException:
                    _remove_cache_tree(temporary)
                    raise
                self._remember(parsed)
                return ResolvedAsset(
                    reference=parsed,
                    local_path=_exposed_path(target, parsed),
                    cache_hit=False,
                )
            raise AssertionError("asset materialization retry loop did not return")

    def cached_assets(self) -> tuple[str, ...]:
        """Return exact pinned cache keys with complete local cache markers."""

        with self._verified_lock:
            remembered = tuple(self._verified.items())
        stale: list[str] = []
        busy: set[str] = set()
        for cache_key, reference in remembered:
            lock = self._lock_for_cache_key(cache_key)
            if not lock.acquire(blocking=False):
                busy.add(cache_key)
                continue
            try:
                if not self._cache_is_complete(self._cache_directory(reference), reference):
                    stale.append(cache_key)
            finally:
                lock.release()
        if stale:
            with self._verified_lock:
                for cache_key in stale:
                    self._verified.pop(cache_key, None)
        with self._verified_lock:
            return tuple(sorted(key for key in self._verified if key not in busy))

    def _materialize_member(self, member: AssetMember, payload: Path) -> _SourceStat:
        source = self._resolve_pinned(member)
        try:
            source_stat = source.lstat()
        except FileNotFoundError as exc:
            raise AssetNotFoundError(f"asset member is unavailable: {member.logical_path}") from exc
        if source.is_symlink() or not stat.S_ISREG(source_stat.st_mode):
            raise AssetIntegrityError("asset members must be regular files, not symbolic links")
        source = source.resolve()
        storage_root = getattr(self._storage, "root", None)
        if storage_root is not None and not source.is_relative_to(Path(storage_root).resolve()):
            raise AssetIntegrityError("asset member resolved outside configured global storage")
        target = _view_path(payload, member.relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.link-{uuid4().hex}")
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb", buffering=0) as reader:
                before = os.fstat(reader.fileno())
                if before.st_size != member.size_bytes:
                    raise _RetryableAssetIntegrityError(
                        f"asset member failed size verification: {member.relative_path}",
                        logical_path=member.logical_path,
                    )
                while chunk := reader.read(_READ_SIZE):
                    check_transfer_cancelled()
                    digest.update(chunk)
                    size += len(chunk)
                    transfer_progress("verifying-input", size, member.size_bytes)
                after = os.fstat(reader.fileno())
            identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if identity_before != identity_after:
                raise _RetryableAssetIntegrityError(
                    "asset member changed while it was materialized",
                    logical_path=member.logical_path,
                )
            if size != member.size_bytes or digest.hexdigest() != member.sha256:
                raise _RetryableAssetIntegrityError(
                    f"asset member failed content verification: {member.relative_path}",
                    logical_path=member.logical_path,
                )
            # Asset bodies already live in the adapter's local or mirror
            # cache. A content-addressed symlink view avoids storing a second
            # multi-GiB copy while preserving a stable per-digest namespace.
            with suppress(OSError):
                source.chmod(0o444)
            temporary.symlink_to(source)
            os.replace(temporary, target)
            try:
                current = source.stat()
            except OSError as exc:
                raise _RetryableAssetIntegrityError(
                    "asset member changed before its cache view was created",
                    logical_path=member.logical_path,
                ) from exc
            if _SourceStat.from_stat(current) != _SourceStat.from_stat(after):
                raise _RetryableAssetIntegrityError(
                    "asset member changed before its cache view was created",
                    logical_path=member.logical_path,
                )
            return _SourceStat.from_stat(after)
        finally:
            temporary.unlink(missing_ok=True)

    def _resolve_pinned(self, member: AssetMember) -> Path:
        resolver = getattr(self._storage, "resolve_pinned", None)
        if resolver is None:
            return self._storage.resolve(member.logical_path)
        if not callable(resolver):
            raise AssetIntegrityError("storage resolve_pinned hook is not callable")
        value: object = resolver(
            member.logical_path,
            expected_size_bytes=member.size_bytes,
        )
        if not isinstance(value, Path):
            raise AssetIntegrityError("storage resolve_pinned hook returned an invalid path")
        return value

    def _invalidate_cached(self, reference: AssetRef, logical_path: str | None) -> bool:
        invalidator = getattr(self._storage, "invalidate_cached", None)
        if invalidator is None:
            return False
        if not callable(invalidator):
            raise AssetIntegrityError("storage invalidate_cached hook is not callable")
        paths = (
            (logical_path,)
            if logical_path is not None
            else tuple(member.logical_path for member in reference.members)
        )
        for path in paths:
            invalidator(path)
        return True

    def _cache_directory(self, reference: AssetRef) -> Path:
        cache_key = reference.cache_key
        assert cache_key is not None
        return resolve_beneath(
            self._cache_root,
            f"sha256/{cache_key}",
        )

    def _lock_for_cache_key(self, cache_key: str) -> threading.RLock:
        return self._locks[int(cache_key[:8], 16) % len(self._locks)]

    def _cache_is_complete(self, target: Path, reference: AssetRef) -> bool:
        if target.is_symlink() or not target.is_dir():
            return False
        marker = _read_marker(target / "manifest.json")
        parsed_marker = _parse_cache_marker(marker)
        if parsed_marker is None:
            return False
        marked_reference, source_stats = parsed_marker
        if _cache_pin_payload(marked_reference) != _cache_pin_payload(reference):
            return False
        payload = target / "payload"
        if payload.is_symlink() or not payload.is_dir():
            return False
        for member in reference.members:
            candidate = _view_path(payload, member.relative_path)
            try:
                candidate_stat = candidate.lstat()
            except FileNotFoundError:
                return False
            if not stat.S_ISLNK(candidate_stat.st_mode):
                return False
            expected = self._expected_local_path(member.logical_path)
            try:
                resolved = candidate.resolve(strict=True)
                resolved_stat = resolved.stat()
            except (FileNotFoundError, OSError):
                return False
            if (
                resolved != expected
                or not stat.S_ISREG(resolved_stat.st_mode)
                or _SourceStat.from_stat(resolved_stat) != source_stats.get(member.relative_path)
            ):
                return False
        return True

    def _expected_local_path(self, logical_path: str) -> Path:
        storage_root = getattr(self._storage, "root", None)
        if storage_root is None:
            # Custom adapters without a visible root cannot support passive
            # startup discovery, but assigned resolution remains correct.
            return self._storage.resolve(logical_path).resolve()
        return resolve_beneath(Path(storage_root), logical_path)

    def _remember(self, reference: AssetRef) -> None:
        cache_key = reference.cache_key
        assert cache_key is not None
        with self._verified_lock:
            self._verified[cache_key] = reference
        (self._access_root / cache_key).touch(mode=0o600)

    def _forget(self, cache_key: str) -> None:
        with self._verified_lock:
            self._verified.pop(cache_key, None)

    def _discover_cache(self) -> None:
        digest_root = self._cache_root / "sha256"
        if not digest_root.is_dir():
            return
        for candidate in digest_root.iterdir():
            if (
                candidate.is_symlink()
                or not candidate.is_dir()
                or not _SHA256.fullmatch(candidate.name)
            ):
                continue
            marker = _parse_cache_marker(_read_marker(candidate / "manifest.json"))
            if marker is None:
                continue
            reference, _ = marker
            if reference.cache_key != candidate.name:
                continue
            if asset_manifest_digest(reference.members) != reference.digest:
                continue
            if self._cache_is_complete(candidate, reference):
                self._verified[reference.cache_key] = reference


def _cache_marker(
    reference: AssetRef,
    source_stats: Mapping[str, _SourceStat],
) -> bytes:
    return (
        json.dumps(
            _cache_marker_payload(reference, source_stats),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _cache_marker_payload(
    reference: AssetRef,
    source_stats: Mapping[str, _SourceStat],
) -> dict[str, object]:
    assert reference.digest is not None
    assert reference.total_bytes is not None
    cache_key = reference.cache_key
    assert cache_key is not None
    return {
        "schema_version": 2,
        "cache_key": cache_key,
        "kind": reference.kind,
        "digest": reference.digest,
        "total_bytes": reference.total_bytes,
        "members": [
            {
                "relative_path": member.relative_path,
                "logical_path": member.logical_path,
                "size_bytes": member.size_bytes,
                "sha256": member.sha256,
                "source_stat": source_stats[member.relative_path].to_dict(),
            }
            for member in sorted(reference.members, key=lambda item: item.relative_path)
        ],
    }


def _cache_pin_payload(reference: AssetRef) -> dict[str, object]:
    return {
        "kind": reference.kind,
        "digest": reference.digest,
        "total_bytes": reference.total_bytes,
        "members": [
            {
                "logical_path": member.logical_path,
                "relative_path": member.relative_path,
                "size_bytes": member.size_bytes,
                "sha256": member.sha256,
            }
            for member in sorted(reference.members, key=lambda item: item.relative_path)
        ],
    }


def _parse_cache_marker(
    marker: dict[str, object] | None,
) -> tuple[AssetRef, dict[str, _SourceStat]] | None:
    if marker is None or set(marker) != {
        "schema_version",
        "cache_key",
        "kind",
        "digest",
        "total_bytes",
        "members",
    }:
        return None
    if marker.get("schema_version") != 2:
        return None
    raw_members = marker.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        return None
    members: list[dict[str, object]] = []
    source_stats: dict[str, _SourceStat] = {}
    try:
        for raw_member in raw_members:
            if not isinstance(raw_member, dict) or set(raw_member) != {
                "logical_path",
                "relative_path",
                "size_bytes",
                "sha256",
                "source_stat",
            }:
                return None
            raw_stat = raw_member["source_stat"]
            if not isinstance(raw_stat, dict) or set(raw_stat) != {
                "device",
                "inode",
                "size",
                "mtime_ns",
            }:
                return None
            stat_values = tuple(raw_stat[name] for name in ("device", "inode", "size", "mtime_ns"))
            if any(isinstance(value, bool) or not isinstance(value, int) for value in stat_values):
                return None
            device, inode, size, mtime_ns = stat_values
            if device < 0 or inode < 0 or size < 0:
                return None
            member = {
                "logical_path": raw_member["logical_path"],
                "relative_path": raw_member["relative_path"],
                "size_bytes": raw_member["size_bytes"],
                "sha256": raw_member["sha256"],
            }
            members.append(member)
            relative_path = member["relative_path"]
            if not isinstance(relative_path, str) or relative_path in source_stats:
                return None
            source_stats[relative_path] = _SourceStat(
                device=device,
                inode=inode,
                size=size,
                mtime_ns=mtime_ns,
            )
        reference = AssetRef.model_validate(
            {
                "source_ref": "cache://discovered",
                "kind": marker["kind"],
                "digest": marker["digest"],
                "total_bytes": marker["total_bytes"],
                "members": members,
            }
        )
    except (KeyError, TypeError, ValueError):
        return None
    if marker.get("cache_key") != reference.cache_key or any(
        source_stats[member.relative_path].size != member.size_bytes for member in reference.members
    ):
        return None
    return reference, source_stats


def _read_marker(path: Path) -> dict[str, object] | None:
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > _MAX_CACHE_MANIFEST_BYTES
        ):
            return None
        raw = json.loads(path.read_bytes())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _validate_relative_tree(members: tuple[AssetMember, ...]) -> None:
    paths = sorted((PurePosixPath(member.relative_path) for member in members), key=str)
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if other.is_relative_to(path):
                raise AssetIntegrityError("asset member paths contain a file/directory collision")


def _make_tree_read_only(root: Path) -> None:
    for child in root.rglob("*"):
        if child.is_dir():
            child.chmod(0o555)
        elif child.is_file() and not child.is_symlink():
            child.chmod(0o444)
    root.chmod(0o555)


def _remove_cache_tree(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
        return
    for child in path.rglob("*"):
        if child.is_dir() and not child.is_symlink():
            with suppress(OSError):
                child.chmod(0o700)
    with suppress(OSError):
        path.chmod(0o700)
    shutil.rmtree(path, ignore_errors=True)


def _exposed_path(target: Path, reference: AssetRef) -> Path:
    payload = target / "payload"
    if reference.kind == "file":
        return _view_path(payload, reference.members[0].relative_path)
    return payload


def _view_path(root: Path, relative_path: str) -> Path:
    normalized = normalize_logical_path(relative_path)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    current = root
    for component in PurePosixPath(normalized).parts[:-1]:
        current /= component
        if current.is_symlink():
            raise AssetIntegrityError("asset view contains a symbolic-link directory")
    return candidate


__all__ = [
    "AssetIntegrityError",
    "AssetNotFoundError",
    "AssetResolver",
    "ResolvedAsset",
    "UnpinnedAssetError",
    "asset_manifest_digest",
]
