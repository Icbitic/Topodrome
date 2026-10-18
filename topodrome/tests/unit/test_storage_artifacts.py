from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from topodrome.storage import LocalStorage, StorageAdapter, VerifiedObject
from topodrome.storage.artifacts import (
    ArtifactCommitError,
    AttemptArtifactStore,
    load_published_manifest,
    resolve_committed_artifact,
)

pytestmark = pytest.mark.unit


class RecordingStorage:
    def __init__(self, delegate: LocalStorage, *, fail_on: str | None = None) -> None:
        self.delegate = delegate
        self.fail_on = fail_on
        self.flushes: list[str] = []

    def healthcheck(self) -> bool:
        return self.delegate.healthcheck()

    def resolve(self, logical_path: str) -> Path:
        return self.delegate.resolve(logical_path)

    def resolve_writable(self, logical_path: str) -> Path:
        return self.delegate.resolve_writable(logical_path)

    def flush_and_verify(
        self,
        logical_path: str,
        *,
        expected_sha256: str | None = None,
    ) -> VerifiedObject:
        self.flushes.append(logical_path)
        if logical_path == self.fail_on:
            raise RuntimeError("injected durability failure")
        return self.delegate.flush_and_verify(
            logical_path,
            expected_sha256=expected_sha256,
        )


def test_attempt_commit_publishes_atomic_manifest_last(tmp_path: Path) -> None:
    recording = RecordingStorage(LocalStorage(tmp_path / "global"))
    assert isinstance(recording, StorageAdapter)
    store = AttemptArtifactStore(
        recording,
        job_id="job-1",
        attempt_id="attempt-2",
        clock=lambda: datetime(2026, 9, 7, tzinfo=UTC),
    )
    store.stage_bytes("metrics/summary.json", b'{"accuracy":0.9}\n')
    source = tmp_path / "figure.png"
    source.write_bytes(b"png")
    store.stage_file(source, "figures/figure.png")

    manifest = store.commit(["metrics/summary.json", "figures/figure.png"])

    assert [artifact.logical_path for artifact in manifest.artifacts] == [
        "figures/figure.png",
        "metrics/summary.json",
    ]
    assert recording.flushes[-2:] == [
        "jobs/job-1/attempts/attempt-2/manifest.json",
        "jobs/job-1/committed/manifest.json",
    ]
    assert recording.flushes[:-2] == [
        "jobs/job-1/attempts/attempt-2/committed/figures/figure.png",
        "jobs/job-1/attempts/attempt-2/committed/metrics/summary.json",
    ]
    loaded = load_published_manifest(recording, "job-1")
    assert loaded == manifest
    assert (
        resolve_committed_artifact(
            recording,
            job_id="job-1",
            logical_path="metrics/summary.json",
        ).read_bytes()
        == b'{"accuracy":0.9}\n'
    )


def test_failed_required_artifact_verification_never_publishes_manifest(tmp_path: Path) -> None:
    failed_path = "jobs/job-1/attempts/attempt-1/committed/result.json"
    recording = RecordingStorage(LocalStorage(tmp_path / "global"), fail_on=failed_path)
    store = AttemptArtifactStore(recording, job_id="job-1", attempt_id="attempt-1")
    store.stage_bytes("result.json", b"result")

    with pytest.raises(RuntimeError, match="injected"):
        store.commit(["result.json"])

    assert not recording.resolve("jobs/job-1/committed/manifest.json").exists()


def test_missing_required_artifact_cannot_commit(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "global")
    store = AttemptArtifactStore(storage, job_id="job-1", attempt_id="attempt-1")

    with pytest.raises(ArtifactCommitError, match="missing"):
        store.commit(["missing.json"])


def test_published_manifest_must_be_a_bounded_regular_file(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "global")
    manifest_path = storage.resolve("jobs/job-1/committed/manifest.json")
    manifest_path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    manifest_path.symlink_to(outside)

    with pytest.raises(ArtifactCommitError, match="regular file"):
        load_published_manifest(storage, "job-1")

    manifest_path.unlink()
    with manifest_path.open("wb") as handle:
        handle.seek(16 * 1024 * 1024)
        handle.write(b"x")
    with pytest.raises(ArtifactCommitError, match="metadata limit"):
        load_published_manifest(storage, "job-1")
