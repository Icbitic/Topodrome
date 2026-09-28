from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta

import pytest

from topodrome.config import StorageRetentionSettings
from topodrome.domain.errors import NotFoundError
from topodrome.scheduler.assets import AssetCatalog, AssetPublishRequest
from topodrome.scheduler.storage_retention import collect_blob_storage
from topodrome.storage.artifacts import AttemptArtifactStore, CommitManifest
from topodrome.storage.base import ObjectDigest, StorageVerificationError
from topodrome.storage.blobs import blob_path, blob_transaction, claim_blob, install_local_blob
from topodrome.storage.local import LocalStorage

pytestmark = pytest.mark.unit


def commit(root, job="j1", payload=b"shared bytes"):
    storage = LocalStorage(root)
    storage.content_addressed = True
    store = AttemptArtifactStore(storage, job_id=job, attempt_id=f"a-{job}")
    store.stage_bytes("results/out.bin", payload)
    manifest = store.commit(["results/out.bin"])
    return store, manifest


def data_for(*jobs):
    old = datetime.now(UTC) - timedelta(days=10)
    return (
        [
            dict(id=j, state="succeeded", spec={}, owner_username="alice", updated_at=old)
            for j in jobs
        ],
        [
            dict(
                id=f"a-{j}",
                job_id=j,
                state="succeeded",
                completed_at=old,
                updated_at=old,
                checkpoint_ref=None,
                restore_checkpoint_ref=None,
            )
            for j in jobs
        ],
        [],
        [],
    )


def age_blobs(root):
    stamp = (datetime.now(UTC) - timedelta(days=10)).timestamp()
    with blob_transaction(root) as db:
        db.execute("UPDATE objects SET touched=?", (stamp,))
    for path in (root / "blobs").rglob("*"):
        if path.is_file():
            os.utime(path, (stamp, stamp))


def test_identical_outputs_share_one_blob_and_retry_is_idempotent(tmp_path):
    store, first = commit(tmp_path)
    _, second = commit(tmp_path, "j2")
    assert first.schema_version == second.schema_version == 2
    assert first.artifacts[0].storage_path == second.artifacts[0].storage_path
    blobs = [p for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    assert len(blobs) == 1
    assert blobs[0].stat().st_nlink == 1
    assert not (tmp_path / store.committed_prefix / "results/out.bin").exists()
    assert store.commit(["results/out.bin"]) == first
    assert CommitManifest.from_bytes(first.to_bytes()) == first


def test_schema2_rejects_noncanonical_blob_path(tmp_path):
    _, manifest = commit(tmp_path)
    raw = manifest.to_bytes().replace(b"blobs/sha256/", b"outside/sha256/")
    with pytest.raises(Exception, match="manifest"):
        CommitManifest.from_bytes(raw)


def test_published_asset_pins_blob_after_producer_expires(tmp_path):
    _, manifest = commit(tmp_path)
    catalog = AssetCatalog(tmp_path, content_addressed=True)
    catalog.publish(
        AssetPublishRequest(name="kept", version="v1", job_id="j1", path="results"),
        owner_username="alice",
        source_manifest=manifest,
    )
    age_blobs(tmp_path)
    result = collect_blob_storage(tmp_path, StorageRetentionSettings(), data_for("j1"), apply=True)
    assert result["deleted_files"] == 0
    assert (tmp_path / manifest.artifacts[0].storage_path).exists()
    ref = catalog.resolve("asset://private/kept@v1", requester_username="alice")
    assert ref.members[0].logical_path == manifest.artifacts[0].storage_path
    with pytest.raises(NotFoundError):
        catalog.resolve("asset://private/kept@v1", requester_username="bob")


def test_deletion_only_removes_its_own_blob_reference(tmp_path):
    _, manifest = commit(tmp_path)
    catalog = AssetCatalog(tmp_path, content_addressed=True)
    for name in ("first", "second"):
        catalog.publish(
            AssetPublishRequest(name=name, version="v1", job_id="j1", path="results"),
            owner_username="alice",
            source_manifest=manifest,
        )
    age_blobs(tmp_path)
    catalog.delete("first", "v1", requester_username="alice")
    assert (
        collect_blob_storage(tmp_path, StorageRetentionSettings(), data_for("j1"), apply=True)[
            "deleted_files"
        ]
        == 0
    )
    catalog.delete("second", "v1", requester_username="alice")
    assert (
        collect_blob_storage(tmp_path, StorageRetentionSettings(), data_for("j1"), apply=True)[
            "deleted_files"
        ]
        == 1
    )
    assert len(list(tmp_path.rglob("deleted.json"))) == 2


def test_running_claim_retains_blob_shared_with_expired_job(tmp_path):
    commit(tmp_path)
    _, manifest = commit(tmp_path, "j2")
    data = data_for("j1", "j2")
    data[0][1]["state"] = "running"
    data[1][1]["state"] = "committing"
    age_blobs(tmp_path)
    r = collect_blob_storage(tmp_path, StorageRetentionSettings(), data, apply=True)
    assert r["deleted_files"] == 0
    data[0][1]["state"] = data[1][1]["state"] = "succeeded"
    r = collect_blob_storage(tmp_path, StorageRetentionSettings(), data)
    assert r["candidate_files"] == 1 and r["deleted_files"] == 0
    r = collect_blob_storage(tmp_path, StorageRetentionSettings(), data, apply=True)
    assert r["deleted_files"] == 1
    assert not (tmp_path / manifest.artifacts[0].storage_path).exists()
    with blob_transaction(tmp_path) as db:
        assert db.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0


def test_queued_consumer_retains_expired_blob(tmp_path):
    _, manifest = commit(tmp_path)
    data = data_for("j1", "consumer")
    data[0][1].update(
        state="queued",
        spec={
            "assets": {
                "input": {
                    "members": [{"logical_path": manifest.artifacts[0].storage_path}],
                }
            }
        },
    )
    age_blobs(tmp_path)
    assert collect_blob_storage(tmp_path, StorageRetentionSettings(), data)["candidate_files"] == 0


@pytest.mark.parametrize("consumer_state", ["queued", "running", "retry_wait"])
def test_deleted_asset_retains_existing_consumers_and_other_refs(tmp_path, consumer_state):
    _, manifest = commit(tmp_path)
    catalog = AssetCatalog(tmp_path, content_addressed=True)
    for name in ("first", "second"):
        catalog.publish(
            AssetPublishRequest(name=name, version="v1", job_id="j1", path="results"),
            owner_username="alice",
            source_manifest=manifest,
        )
    pin = catalog.resolve("asset://private/first@v1", requester_username="alice")
    data = data_for("j1", "consumer")
    data[0][1].update(
        state=consumer_state,
        spec={
            "assets": {
                "input": {"members": [{"logical_path": m.logical_path} for m in pin.members]}
            }
        },
    )
    age_blobs(tmp_path)
    catalog.delete("first", "v1", requester_username="alice")
    assert (
        collect_blob_storage(tmp_path, StorageRetentionSettings(), data, apply=True)[
            "deleted_files"
        ]
        == 0
    )
    catalog.delete("second", "v1", requester_username="alice")
    assert (
        collect_blob_storage(tmp_path, StorageRetentionSettings(), data, apply=True)[
            "deleted_files"
        ]
        == 0
    )
    data[0][1]["state"] = "succeeded"  # Its recorded lifetime is already expired.
    assert (
        collect_blob_storage(tmp_path, StorageRetentionSettings(), data, apply=True)[
            "deleted_files"
        ]
        == 1
    )
    assert not (tmp_path / manifest.artifacts[0].storage_path).exists()
    assert len(list(tmp_path.rglob("deleted.json"))) == 2


def test_unknown_claim_fails_closed(tmp_path):
    commit(tmp_path)
    age_blobs(tmp_path)
    assert (
        collect_blob_storage(tmp_path, StorageRetentionSettings(), ([], [], [], []))[
            "candidate_files"
        ]
        == 0
    )


def test_corrupt_existing_blob_is_not_overwritten(tmp_path):
    content = b"good"
    digest = ObjectDigest(len(content), hashlib.sha256(content).hexdigest())
    source = tmp_path / "source"
    source.write_bytes(content)
    target = tmp_path / blob_path(digest.sha256)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"evil")
    with pytest.raises(StorageVerificationError):
        install_local_blob(tmp_path, source, digest)
    assert target.read_bytes() == b"evil"


def test_claim_is_durable_before_blob_exists(tmp_path):
    digest = ObjectDigest(10, "a" * 64)
    claim_blob(tmp_path, "job", "attempt", digest)
    with blob_transaction(tmp_path) as db:
        assert db.execute("SELECT job_id,attempt_id FROM claims").fetchone() == ("job", "attempt")
    assert not (tmp_path / blob_path(digest.sha256)).exists()
