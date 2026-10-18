from datetime import UTC, datetime, timedelta

import pytest

from topodrome.domain.errors import AdmissionRejectedError, NotFoundError
from topodrome.scheduler.assets import ResolvedAssetInput, ResolvedAssetMember
from topodrome.storage.download_pins import (
    MAX_LIFETIME_SECONDS,
    TTL_SECONDS,
    DownloadPins,
    download_protected_paths,
)

pytestmark = pytest.mark.unit


def resolved():
    return ResolvedAssetInput(
        "job://job-one/result",
        "file",
        "a" * 64,
        3,
        (ResolvedAssetMember("jobs/job-one/result", "result", 3, "b" * 64),),
    )


def test_pin_survives_restart_and_expires_without_mutating_data(tmp_path):
    now = datetime.now(UTC)

    def clock():
        return now

    manifest = DownloadPins(tmp_path, clock=clock).create(resolved(), "alice")
    restarted = DownloadPins(tmp_path, clock=clock)
    assert restarted.get(manifest.download_id, "alice").manifest == manifest
    assert download_protected_paths(tmp_path, now=now.timestamp()) == {"jobs/job-one/result"}
    assert not (tmp_path / "jobs").exists()  # References never create or copy payloads.
    now += timedelta(seconds=TTL_SECONDS + 1)
    assert not download_protected_paths(tmp_path, now=now.timestamp())
    with pytest.raises(NotFoundError):
        restarted.get(manifest.download_id, "alice", renew=True)
    next_manifest = restarted.create(resolved(), "alice")
    assert next_manifest.download_id != manifest.download_id
    assert not restarted._path(manifest.download_id).exists()


def test_renewal_is_owner_scoped_bounded_and_delete_is_repeat_safe(tmp_path):
    now = datetime.now(UTC)
    pins = DownloadPins(tmp_path, clock=lambda: now)
    manifest = pins.create(resolved(), "alice")
    for operation in (
        lambda: pins.get(manifest.download_id, "bob", renew=True),
        lambda: pins.delete(manifest.download_id, "bob"),
    ):
        with pytest.raises(NotFoundError):
            operation()
    start = now
    while now < start + timedelta(seconds=MAX_LIFETIME_SECONDS - 600):
        now += timedelta(seconds=600)
        pins.get(manifest.download_id, "alice", renew=True)
    now = start + timedelta(seconds=MAX_LIFETIME_SECONDS - 1)
    assert (
        pins.get(manifest.download_id, "alice", renew=True).expires_at
        == (start + timedelta(seconds=MAX_LIFETIME_SECONDS)).timestamp()
    )
    now += timedelta(seconds=1)
    with pytest.raises(NotFoundError):
        pins.get(manifest.download_id, "alice", renew=True)
    pins.delete(manifest.download_id, "alice")
    pins.delete(manifest.download_id, "alice")


def test_owner_download_count_is_bounded_and_expiry_releases_capacity(tmp_path):
    now = datetime.now(UTC)
    pins = DownloadPins(tmp_path, clock=lambda: now)
    for _ in range(16):
        pins.create(resolved(), "alice")
    with pytest.raises(AdmissionRejectedError, match="too many"):
        pins.create(resolved(), "alice")
    pins.create(resolved(), "bob")
    now += timedelta(seconds=TTL_SECONDS + 1)
    pins.create(resolved(), "alice")
    assert len(list(pins.records())) == 1


def test_invalid_reference_storage_fails_closed(tmp_path):
    pins = DownloadPins(tmp_path)
    manifest = pins.create(resolved(), "alice")
    pins._path(manifest.download_id).write_text("invalid")
    with pytest.raises(ValueError):
        download_protected_paths(tmp_path)
    with pytest.raises(NotFoundError):
        pins.get("../../other", "alice")


def test_renewal_coalesces_writes_for_many_small_members(tmp_path, monkeypatch):
    now = datetime.now(UTC)
    pins = DownloadPins(tmp_path, clock=lambda: now)
    manifest = pins.create(resolved(), "alice")
    writes = []
    original = pins.write

    def write(pin):
        writes.append(pin)
        original(pin)

    monkeypatch.setattr(pins, "write", write)
    for _ in range(20):
        pins.get(manifest.download_id, "alice", renew=True)
    assert writes == []
    now += timedelta(seconds=60)
    assert (
        pins.get(manifest.download_id, "alice", renew=True).expires_at
        > manifest.expires_at.timestamp()
    )
    assert len(writes) == 1


def test_live_download_is_excluded_from_future_reclamation_forecast(tmp_path):
    from topodrome.config import StorageRetentionSettings
    from topodrome.scheduler.assets import resolve_job_asset
    from topodrome.scheduler.system_metrics import sample_storage
    from topodrome.storage.artifacts import AttemptArtifactStore
    from topodrome.storage.local import LocalStorage

    storage = LocalStorage(tmp_path)
    storage.content_addressed = True
    store = AttemptArtifactStore(storage, job_id="job-one", attempt_id="attempt-one")
    store.stage_bytes("result", b"payload")
    committed = store.commit(["result"])
    old = datetime.now(UTC) - timedelta(days=6, hours=12)
    data = (
        [dict(id="job-one", state="succeeded", spec={}, updated_at=old)],
        [
            dict(
                id="attempt-one",
                job_id="job-one",
                state="succeeded",
                updated_at=old,
                completed_at=old,
                checkpoint_ref=None,
                restore_checkpoint_ref=None,
            )
        ],
        [],
        [],
    )
    pins = DownloadPins(tmp_path)
    descriptor = pins.create(resolve_job_asset("job://job-one/result", committed), "alice")
    report = sample_storage(tmp_path, "blobs-v1", StorageRetentionSettings(), data)
    assert report["eligible_24h_bytes"] == 0
    pins.delete(descriptor.download_id, "alice")
    report = sample_storage(tmp_path, "blobs-v1", StorageRetentionSettings(), data)
    assert report["eligible_24h_bytes"] > 0
