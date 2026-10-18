from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from topodrome.domain.errors import ConflictError, NotFoundError
from topodrome.scheduler.assets import (
    AssetCatalog,
    AssetPublishRequest,
    AssetTagUpdate,
    job_id_from_ref,
    resolve_job_asset,
)
from topodrome.storage.artifacts import CommitManifest, CommittedArtifact

pytestmark = pytest.mark.unit


def test_latest_versions_before_limit_and_tags_with_namespace_isolation(tmp_path):
    now = datetime(2026, 9, 13, tzinfo=UTC)
    catalog = AssetCatalog(tmp_path, clock=lambda: now)
    for owner, visibility, name, version, day in [
        ("alice", "private", "lens", "v99", 13),
        ("alice", "private", "other", "v1", 14),
        ("alice", "private", "lens", "v1", 15),
        ("alice", "private", "lens", "v2", 16),
        ("bob", "private", "lens", "v100", 17),
        ("bob", "shared", "lens", "v0", 18),
    ]:
        now = datetime(2026, 9, day, tzinfo=UTC)
        catalog.publish(
            AssetPublishRequest(
                name=name,
                version=version,
                job_id="job-producer",
                path="lens",
                visibility=visibility,
            ),
            owner_username=owner,
            source_manifest=_manifest(),
        )
    rows = catalog.list_visible(requester_username="alice", limit=3)
    assert [(r.visibility, r.name, r.version) for r in rows] == [
        ("shared", "lens", "v0"),
        ("private", "lens", "v2"),
        ("private", "other", "v1"),
    ]
    assert len(catalog.list_visible(requester_username="alice", all_versions=True)) == 5
    assert catalog.get("lens", "v99", requester_username="alice").version == "v99"
    catalog.update_tags("lens", "v99", AssetTagUpdate(add=("old",)), requester_username="alice")
    assert catalog.list_visible(requester_username="alice", tags=("old",)) == []
    assert [
        r.version
        for r in catalog.list_visible(requester_username="alice", tags=("old",), all_versions=True)
    ] == ["v99"]
    catalog.delete("lens", "v2", requester_username="alice")
    assert catalog.list_visible(requester_username="alice", visibility="private")[0].version == "v1"


def _manifest(
    *,
    job_id: str = "job-producer",
    attempt_id: str = "attempt-success",
    files: tuple[tuple[str, int, str], ...] = (
        ("lens/config.json", 12, "a" * 64),
        ("lens/params.pt", 5 * 1024**3, "b" * 64),
    ),
) -> CommitManifest:
    return CommitManifest(
        job_id=job_id,
        attempt_id=attempt_id,
        artifacts=tuple(
            CommittedArtifact(
                logical_path=path,
                storage_path=(f"jobs/{job_id}/attempts/{attempt_id}/committed/{path}"),
                size_bytes=size,
                sha256=sha256,
            )
            for path, size, sha256 in files
        ),
        committed_at=datetime(2026, 9, 13, tzinfo=UTC),
    )


def test_publish_is_manifest_only_and_resolves_exact_attempt_paths(tmp_path: Path) -> None:
    catalog = AssetCatalog(
        tmp_path,
        clock=lambda: datetime(2026, 9, 13, 1, 2, tzinfo=UTC),
    )
    result = catalog.publish(
        AssetPublishRequest(
            name="llama-tuned-lens",
            version="v1",
            job_id="job-producer",
            path="lens/",
            visibility="shared",
        ),
        owner_username="alice",
        source_manifest=_manifest(),
    )

    assert result.created is True
    assert result.asset.ref == "asset://shared/llama-tuned-lens@v1"
    assert result.asset.kind == "directory"
    assert result.asset.total_size_bytes == 5 * 1024**3 + 12
    assert [item.relative_path for item in result.asset.files] == [
        "config.json",
        "params.pt",
    ]
    # Publishing several GiB writes one small metadata file and no payload copy.
    catalog_files = [path for path in (tmp_path / "artifacts/catalog").rglob("*") if path.is_file()]
    assert [path.name for path in catalog_files] == ["manifest.json"]
    assert catalog_files[0].stat().st_size < 4096

    resolved = catalog.resolve(
        "asset://shared/llama-tuned-lens@v1",
        requester_username="bob",
    )
    assert resolved.source_ref == "asset://shared/llama-tuned-lens@v1"
    assert resolved.digest == result.asset.digest
    assert resolved.total_bytes == 5 * 1024**3 + 12
    assert resolved.members[1].logical_path == (
        "jobs/job-producer/attempts/attempt-success/committed/lens/params.pt"
    )


def test_immutable_version_is_idempotent_only_for_same_content(tmp_path: Path) -> None:
    catalog = AssetCatalog(tmp_path)
    request = AssetPublishRequest(
        name="lens",
        version="2026-09-13",
        job_id="job-producer",
        path="lens",
    )

    first = catalog.publish(
        request,
        owner_username="alice",
        source_manifest=_manifest(),
    )
    replay = catalog.publish(
        request,
        owner_username="alice",
        source_manifest=_manifest(),
    )
    assert first.created is True
    assert replay.created is False
    assert replay.asset.manifest_sha256 == first.asset.manifest_sha256

    changed = _manifest(
        files=(
            ("lens/config.json", 12, "a" * 64),
            ("lens/params.pt", 5 * 1024**3, "c" * 64),
        )
    )
    with pytest.raises(ConflictError, match="different content"):
        catalog.publish(
            request,
            owner_username="alice",
            source_manifest=changed,
        )


def test_private_assets_are_owner_scoped_while_shared_assets_are_visible(tmp_path: Path) -> None:
    catalog = AssetCatalog(tmp_path)
    private = AssetPublishRequest(
        name="private-lens",
        version="v1",
        job_id="job-producer",
        path="lens/params.pt",
    )
    shared = private.model_copy(update={"name": "shared-lens", "visibility": "shared"})
    catalog.publish(private, owner_username="alice", source_manifest=_manifest())
    catalog.publish(shared, owner_username="alice", source_manifest=_manifest())

    assert [item.name for item in catalog.list_visible(requester_username="bob")] == ["shared-lens"]
    assert {item.name for item in catalog.list_visible(requester_username="alice")} == {
        "private-lens",
        "shared-lens",
    }
    with pytest.raises(NotFoundError):
        catalog.get(
            "private-lens",
            "v1",
            requester_username="bob",
            visibility="private",
        )


def test_direct_job_reference_uses_same_file_and_prefix_pin_semantics() -> None:
    manifest = _manifest()

    tree = resolve_job_asset("job://job-producer/lens/", manifest)
    exact = resolve_job_asset("job://job-producer/lens/params.pt", manifest)

    assert job_id_from_ref("job://job-producer/lens/") == "job-producer"
    assert tree.source_ref == "job://job-producer/lens/"
    assert [member.relative_path for member in tree.members] == [
        "config.json",
        "params.pt",
    ]
    assert exact.source_ref == "job://job-producer/lens/params.pt"
    assert exact.kind == "file"
    assert [member.relative_path for member in exact.members] == ["params.pt"]
    assert exact.total_bytes == 5 * 1024**3


def test_missing_committed_selector_is_rejected(tmp_path: Path) -> None:
    catalog = AssetCatalog(tmp_path)
    with pytest.raises(NotFoundError, match="does not exist"):
        catalog.publish(
            AssetPublishRequest(
                name="missing",
                version="v1",
                job_id="job-producer",
                path="not-there",
            ),
            owner_username="alice",
            source_manifest=_manifest(),
        )


@pytest.mark.parametrize("visibility", ["private", "shared"])
def test_delete_is_owner_checked_durable_and_cannot_rebind(tmp_path, visibility):
    from topodrome.scheduler.assets import AssetTagUpdate

    catalog = AssetCatalog(tmp_path)
    request = AssetPublishRequest(
        name="lens", version="v1", job_id="job-producer", path="lens", visibility=visibility
    )
    published = catalog.publish(request, owner_username="alice", source_manifest=_manifest()).asset
    for operation in (
        lambda: catalog.delete("lens", "v1", requester_username="bob", visibility=visibility),
        lambda: catalog.update_tags(
            "lens",
            "v1",
            AssetTagUpdate(add=("bad",)),
            requester_username="bob",
            visibility=visibility,
        ),
    ):
        with pytest.raises(NotFoundError):
            operation()
    result = catalog.delete("lens", "v1", requester_username="alice", visibility=visibility)
    assert (
        result.ref == published.ref and not result.already_deleted and not result.payloads_deleted
    )
    assert catalog.list_visible(requester_username="alice") == []
    with pytest.raises(NotFoundError):
        catalog.resolve(published.ref, requester_username="alice")
    restarted = AssetCatalog(tmp_path)
    replay = restarted.delete("lens", "v1", requester_username="alice", visibility=visibility)
    assert replay.already_deleted and replay.deleted_at == result.deleted_at
    with pytest.raises(ConflictError, match="deleted"):
        restarted.publish(request, owner_username="alice", source_manifest=_manifest())
    with pytest.raises(NotFoundError):
        restarted.update_tags(
            "lens",
            "v1",
            AssetTagUpdate(add=("gone",)),
            requester_username="alice",
            visibility=visibility,
        )


def test_delete_shared_admin_and_missing_private_are_isolated(tmp_path):
    catalog = AssetCatalog(tmp_path)
    for visibility in ("shared", "private"):
        catalog.publish(
            AssetPublishRequest(
                name="lens", version="v1", job_id="job-producer", path="lens", visibility=visibility
            ),
            owner_username="alice",
            source_manifest=_manifest(),
        )
    catalog.delete("lens", "v1", visibility="shared", requester_username="@operator", is_admin=True)
    with pytest.raises(NotFoundError):
        catalog.delete("lens", "v1", requester_username="bob")
    assert catalog.get("lens", "v1", requester_username="alice").visibility == "private"


def test_delete_interrupted_after_tombstone_is_hidden_and_recoverable(tmp_path, monkeypatch):
    catalog = AssetCatalog(tmp_path)
    request = AssetPublishRequest(name="lens", version="v1", job_id="job-producer", path="lens")
    catalog.publish(request, owner_username="alice", source_manifest=_manifest())
    original = Path.unlink

    def fail_manifest(path, *args, **kwargs):
        if path.name == "manifest.json":
            raise OSError("simulated unlink failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_manifest)
    with pytest.raises(OSError):
        catalog.delete("lens", "v1", requester_username="alice")
    assert catalog.list_visible(requester_username="alice") == []
    assert len(list(tmp_path.rglob("manifest.json"))) == 1  # GC remains conservative.
    with pytest.raises(ConflictError):
        catalog.publish(request, owner_username="alice", source_manifest=_manifest())
    monkeypatch.setattr(Path, "unlink", original)
    assert catalog.delete("lens", "v1", requester_username="alice").already_deleted
    assert not list(tmp_path.rglob("manifest.json"))


def test_tags_are_atomic_add_remove_and_not_content_identity(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from topodrome.scheduler.assets import AssetTagUpdate

    catalog = AssetCatalog(tmp_path)
    request = AssetPublishRequest(name="lens", version="v1", job_id="job-producer", path="lens")
    original = catalog.publish(request, owner_username="alice", source_manifest=_manifest()).asset
    pin = catalog.resolve(original.ref, requester_username="alice")

    def add(tag):
        return catalog.update_tags(
            "lens", "v1", AssetTagUpdate(add=(tag,)), requester_username="alice"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(add, (" Project:Lens ", "kind:weights")))
    current = catalog.get("lens", "v1", requester_username="alice")
    assert current.tags == ("kind:weights", "project:lens")
    assert current.digest == original.digest and current.manifest_sha256 == original.manifest_sha256
    assert catalog.resolve(current.ref, requester_username="alice") == pin
    assert catalog.list_visible(
        requester_username="alice", tags=("PROJECT:lens", "kind:weights")
    ) == [current]
    assert catalog.list_visible(requester_username="alice", tags=("absent",)) == []
    updated = catalog.update_tags(
        "lens", "v1", AssetTagUpdate(remove=("project:lens",)), requester_username="alice"
    )
    assert updated.tags == ("kind:weights",)
    assert (
        catalog.publish(request, owner_username="alice", source_manifest=_manifest()).asset.tags
        == updated.tags
    )
    with pytest.raises(ValueError):
        AssetTagUpdate(add=("bad tag",))
    with pytest.raises(ValueError):
        AssetTagUpdate(add=("same",), remove=("same",))


def test_catalog_mutations_refuse_symlink_parent(tmp_path):
    from topodrome.scheduler.assets import AssetCatalogError, AssetTagUpdate

    catalog = AssetCatalog(tmp_path)
    request = AssetPublishRequest(
        name="lens", version="v1", job_id="job-producer", path="lens", visibility="shared"
    )
    catalog.publish(request, owner_username="alice", source_manifest=_manifest())
    parent = tmp_path / "artifacts/catalog/v1/shared/lens"
    saved = parent.with_name("saved")
    parent.rename(saved)
    parent.symlink_to(saved, target_is_directory=True)
    with pytest.raises(AssetCatalogError, match="symlink"):
        catalog.delete("lens", "v1", requester_username="alice", visibility="shared")
    with pytest.raises(AssetCatalogError, match="symlink"):
        catalog.update_tags(
            "lens",
            "v1",
            AssetTagUpdate(add=("tag",)),
            requester_username="alice",
            visibility="shared",
        )
    assert (saved / "v1/manifest.json").exists()


def test_failed_tombstone_write_preserves_asset(tmp_path, monkeypatch):
    import topodrome.scheduler.assets as assets

    catalog = AssetCatalog(tmp_path)
    request = AssetPublishRequest(name="lens", version="v1", job_id="job-producer", path="lens")
    asset = catalog.publish(request, owner_username="alice", source_manifest=_manifest()).asset

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(assets, "_link_immutable", fail)
    with pytest.raises(OSError, match="disk full"):
        catalog.delete("lens", "v1", requester_username="alice")
    assert catalog.get("lens", "v1", requester_username="alice") == asset


def test_tag_limit_applies_to_merged_set(tmp_path):
    from topodrome.domain.errors import AdmissionRejectedError
    from topodrome.scheduler.assets import AssetTagUpdate

    catalog = AssetCatalog(tmp_path)
    request = AssetPublishRequest(name="lens", version="v1", job_id="job-producer", path="lens")
    catalog.publish(request, owner_username="alice", source_manifest=_manifest())
    tags = tuple(f"tag{i}" for i in range(32))
    before = catalog.update_tags("lens", "v1", AssetTagUpdate(add=tags), requester_username="alice")
    with pytest.raises(AdmissionRejectedError, match="32 tags"):
        catalog.update_tags(
            "lens", "v1", AssetTagUpdate(add=("overflow",)), requester_username="alice"
        )
    assert catalog.get("lens", "v1", requester_username="alice") == before
