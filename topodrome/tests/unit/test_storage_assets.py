from __future__ import annotations

import hashlib
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from topodrome.domain.specs import AssetMember, AssetRef
from topodrome.storage.assets import (
    AssetIntegrityError,
    AssetResolver,
    UnpinnedAssetError,
    asset_manifest_digest,
)
from topodrome.storage.base import VerifiedObject
from topodrome.storage.local import LocalStorage


class _CountingStorage:
    def __init__(self, storage: LocalStorage) -> None:
        self.storage = storage
        self.root = storage.root
        self.resolve_calls = 0

    def healthcheck(self) -> bool:
        return self.storage.healthcheck()

    def resolve(self, logical_path: str) -> Path:
        self.resolve_calls += 1
        return self.storage.resolve(logical_path)

    def resolve_writable(self, logical_path: str) -> Path:
        return self.storage.resolve_writable(logical_path)

    def flush_and_verify(
        self,
        logical_path: str,
        *,
        expected_sha256: str | None = None,
    ) -> VerifiedObject:
        return self.storage.flush_and_verify(logical_path, expected_sha256=expected_sha256)


def _member(logical_path: str, relative_path: str, content: bytes) -> AssetMember:
    return AssetMember(
        logical_path=logical_path,
        relative_path=relative_path,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _reference(
    members: tuple[AssetMember, ...],
    *,
    kind: str = "directory",
) -> AssetRef:
    return AssetRef(
        source_ref="job://job-1/output/",
        kind=kind,
        digest=asset_manifest_digest(members),
        total_bytes=sum(member.size_bytes for member in members),
        members=members,
    )


def test_asset_is_materialized_lazily_once_without_copying_payload_bytes(tmp_path: Path) -> None:
    local = LocalStorage(tmp_path / "global")
    logical_path = "jobs/job-1/attempts/attempt-1/committed/lens/params.pt"
    content = b"large immutable tensor placeholder"
    source = local.resolve(logical_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    storage = _CountingStorage(local)
    resolver = AssetResolver(storage, cache_root=tmp_path / "asset-cache")
    reference = _reference((_member(logical_path, "params.pt", content),), kind="file")

    assert storage.resolve_calls == 0
    assert resolver.cached_assets() == ()

    first = resolver.resolve(reference)
    assert not first.cache_hit
    assert first.local_path.is_symlink()
    assert first.local_path.resolve() == source.resolve()
    assert first.local_path.read_bytes() == content
    assert source.stat().st_mode & stat.S_IWUSR == 0
    assert storage.resolve_calls == 1
    assert reference.cache_key is not None
    assert reference.cache_key != reference.digest
    assert resolver.cached_assets() == (reference.cache_key,)

    marker_path = resolver.cache_root / "sha256" / reference.cache_key / "manifest.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    source_stat = source.stat()
    assert marker["schema_version"] == 2
    assert marker["cache_key"] == reference.cache_key
    assert marker["members"][0]["source_stat"] == {
        "device": source_stat.st_dev,
        "inode": source_stat.st_ino,
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
    }

    second = resolver.resolve(reference)
    assert second.cache_hit
    assert second.local_path == first.local_path
    # Cache validation computes the expected adapter path without asking a
    # mirror adapter to fetch it again.
    assert storage.resolve_calls == 1


def test_directory_asset_preserves_relative_layout_and_survives_restart(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "global")
    contents = {"config.json": b"{}", "weights/params.pt": b"weights"}
    members: list[AssetMember] = []
    for relative_path, content in contents.items():
        logical_path = f"jobs/job-1/attempts/attempt-1/committed/lens/{relative_path}"
        source = storage.resolve(logical_path)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
        members.append(_member(logical_path, relative_path, content))
    reference = _reference(tuple(members))
    cache_root = tmp_path / "asset-cache"

    resolver = AssetResolver(storage, cache_root=cache_root)
    first = resolver.resolve(reference)
    assert first.local_path.is_dir()
    assert first.local_path.joinpath("config.json").read_bytes() == b"{}"
    assert first.local_path.joinpath("weights/params.pt").read_bytes() == b"weights"

    restarted = AssetResolver(storage, cache_root=cache_root)
    assert restarted.cached_assets() == (reference.cache_key,)
    assert restarted.resolve(reference).cache_hit


def test_asset_rejects_mutable_or_incorrect_pins_without_publishing_cache(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "global")
    resolver = AssetResolver(storage, cache_root=tmp_path / "asset-cache")
    with pytest.raises(UnpinnedAssetError):
        resolver.resolve("job://job-1/output/")

    logical_path = "jobs/job-1/attempts/attempt-1/committed/result.bin"
    source = storage.resolve(logical_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"actual")
    bad_member = AssetMember(
        logical_path=logical_path,
        relative_path="result.bin",
        size_bytes=6,
        sha256=hashlib.sha256(b"wrong!").hexdigest(),
    )
    reference = _reference((bad_member,), kind="file")
    with pytest.raises(AssetIntegrityError, match="content verification"):
        resolver.resolve(reference)
    assert resolver.cached_assets() == ()
    assert not tuple(
        (resolver.cache_root / "sha256" / str(reference.cache_key)).glob("**/manifest.json")
    )


def test_asset_digest_is_location_independent_and_tree_collisions_are_rejected(
    tmp_path: Path,
) -> None:
    content = b"same"
    first = _member("jobs/a/committed/value", "value", content)
    second = _member("assets/named/v1/value", "value", content)
    assert asset_manifest_digest((first,)) == asset_manifest_digest((second,))

    storage = LocalStorage(tmp_path / "global")
    collision_members = (
        _member("jobs/a/committed/a", "nested", b"a"),
        _member("jobs/a/committed/b", "nested/child", b"b"),
    )
    reference = _reference(collision_members)
    resolver = AssetResolver(storage, cache_root=tmp_path / "cache")
    with pytest.raises(AssetIntegrityError, match="file/directory collision"):
        resolver.resolve(reference)


def test_equal_content_with_different_view_semantics_coexists(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "global")
    content = b"shared content"
    first_logical = "jobs/first/attempts/one/committed/result.bin"
    second_logical = "jobs/second/attempts/two/committed/result.bin"
    for logical_path in (first_logical, second_logical):
        source = storage.resolve(logical_path)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)

    first_member = _member(first_logical, "result.bin", content)
    second_member = _member(second_logical, "result.bin", content)
    file_reference = _reference((first_member,), kind="file")
    directory_reference = _reference((second_member,), kind="directory")
    assert file_reference.digest == directory_reference.digest
    assert file_reference.cache_key != directory_reference.cache_key

    resolver = AssetResolver(storage, cache_root=tmp_path / "cache")
    with ThreadPoolExecutor(max_workers=2) as executor:
        file_result, directory_result = executor.map(
            resolver.resolve,
            (file_reference, directory_reference),
        )
    file_view = file_result.local_path
    directory_view = directory_result.local_path

    assert file_view.is_symlink()
    assert file_view.resolve() == storage.resolve(first_logical).resolve()
    assert (
        directory_view.joinpath("result.bin").resolve() == storage.resolve(second_logical).resolve()
    )
    assert file_view.read_bytes() == content
    assert resolver.resolve(file_reference).cache_hit
    assert resolver.resolve(directory_reference).cache_hit


def test_cache_key_is_stable_but_includes_exact_pinned_coordinates() -> None:
    content = b"same"
    first = _member("jobs/first/committed/value", "value", content)
    second = _member("jobs/second/committed/value", "value", content)
    first_reference = _reference((first,), kind="file")
    same_reference = _reference((first,), kind="file")
    moved_reference = _reference((second,), kind="file")
    directory_reference = _reference((first,), kind="directory")

    assert first_reference.cache_key == same_reference.cache_key
    assert first_reference.digest == moved_reference.digest == directory_reference.digest
    assert len(str(first_reference.cache_key)) == 64
    assert first_reference.cache_key != moved_reference.cache_key
    assert first_reference.cache_key != directory_reference.cache_key
    assert AssetRef(source_ref="asset://shared/value@v1").cache_key is None


def test_restart_rejects_cache_when_source_stat_identity_changes(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "global")
    logical_path = "jobs/job-1/attempts/attempt-1/committed/result.bin"
    content = b"stable content"
    source = storage.resolve(logical_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    reference = _reference((_member(logical_path, "result.bin", content),), kind="file")
    cache_root = tmp_path / "asset-cache"

    resolver = AssetResolver(storage, cache_root=cache_root)
    first = resolver.resolve(reference)
    assert first.cache_hit is False
    old_inode = source.stat().st_ino
    replacement = source.with_name("replacement")
    replacement.write_bytes(content)
    os.replace(replacement, source)
    assert source.stat().st_ino != old_inode
    assert resolver.cached_assets() == ()

    restarted = AssetResolver(storage, cache_root=cache_root)
    assert restarted.cached_assets() == ()
    refreshed = restarted.resolve(reference)
    assert refreshed.cache_hit is False
    assert restarted.cached_assets() == (reference.cache_key,)


def test_stale_adapter_cache_is_invalidated_and_resolved_once_more(tmp_path: Path) -> None:
    local = LocalStorage(tmp_path / "global")
    logical_path = "jobs/job-1/attempts/attempt-1/committed/result.bin"
    expected = b"expected"
    source = local.resolve(logical_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"outdated")

    class RefreshingStorage(_CountingStorage):
        def __init__(self, storage: LocalStorage) -> None:
            super().__init__(storage)
            self.pinned_calls = 0
            self.invalidations: list[str] = []

        def resolve_pinned(self, path: str, *, expected_size_bytes: int) -> Path:
            self.pinned_calls += 1
            assert expected_size_bytes == len(expected)
            return self.storage.resolve(path)

        def invalidate_cached(self, path: str) -> None:
            self.invalidations.append(path)
            self.storage.resolve(path).write_bytes(expected)

    storage = RefreshingStorage(local)
    reference = _reference((_member(logical_path, "result.bin", expected),), kind="file")
    result = AssetResolver(storage, cache_root=tmp_path / "cache").resolve(reference)

    assert result.cache_hit is False
    assert result.local_path.read_bytes() == expected
    assert storage.pinned_calls == 2
    assert storage.invalidations == [logical_path]
