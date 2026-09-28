from __future__ import annotations

import asyncio
import os
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from topodrome.config import StorageRetentionSettings
from topodrome.domain.errors import AdmissionRejectedError
from topodrome.scheduler.assets import AssetCatalog, AssetPublishRequest
from topodrome.scheduler.storage_retention import collect, finish_io, plan, require_members
from topodrome.storage.artifacts import AttemptArtifactStore
from topodrome.storage.local import LocalStorage

pytestmark = pytest.mark.unit
NOW = datetime(2026, 10, 1, tzinfo=UTC)
OLD = NOW - timedelta(days=10)
PREFIX = "jobs/job-one/attempts/attempt-one"


def state(*, status="succeeded", ended=OLD, retention=None):
    job = dict(id="job-one", state=status, spec={}, owner_username="alice", updated_at=ended)
    if retention:
        job["spec"]["artifact_retention_seconds"] = retention
    attempt = dict(
        id="attempt-one",
        job_id="job-one",
        state=status,
        completed_at=ended,
        updated_at=ended,
        checkpoint_ref=None,
        restore_checkpoint_ref=None,
    )
    return [job], [attempt], [], []


def file(root, suffix, content=b"test", age=OLD):
    path = root / PREFIX / suffix
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (age.timestamp(), age.timestamp()))
    return path


def planned(root, data, **kwargs):
    return plan(root, StorageRetentionSettings(**kwargs), data, now=NOW.timestamp())


def test_success_checkpoints_expire_but_logs_and_unregistered_files_do_not(tmp_path):
    file(tmp_path, "checkpoints/00000001-aaaaaaaaaaaa.pkl")
    file(tmp_path, "checkpoints/00000002-aaaaaaaaaaaa.pkl")
    file(tmp_path, "checkpoints/user-data.pkl")
    file(tmp_path, "logs/attempt.log")
    file(tmp_path, "committed/unregistered.bin")
    candidates = planned(tmp_path, state())
    assert len(candidates) == 2
    assert all(c["reason"] == "recovery_checkpoint" for c in candidates)


def test_broken_catalog_symlink_aborts_collection(tmp_path):
    catalog = tmp_path / "artifacts" / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "v1").symlink_to(tmp_path / "missing")
    file(tmp_path, "checkpoints/00000001-aaaaaaaaaaaa.pkl")
    with pytest.raises(ValueError, match="symlink"):
        planned(tmp_path, state())


@pytest.mark.parametrize("status", ["running", "queued", "retry_wait", "canceling", "assigning"])
def test_nonterminal_job_never_collected(tmp_path, status):
    file(tmp_path, "checkpoints/00000001-aaaaaaaaaaaa.pkl")
    data = state(status=status)
    data[1][0]["state"] = "failed"
    assert planned(tmp_path, data) == []


def test_live_lease_and_unfinished_attempt_prevent_deletion(tmp_path):
    file(tmp_path, "checkpoints/00000001-aaaaaaaaaaaa.pkl")
    data = state()
    data[3].append(dict(attempt_id="attempt-one", state="releasing"))
    assert planned(tmp_path, data) == []
    data[3].clear()
    data[1][0]["state"] = "committing"
    assert planned(tmp_path, data) == []


@pytest.mark.parametrize("status", ["failed", "canceled"])
def test_recent_failure_keeps_latest_and_verified_fallback(tmp_path, status):
    for i in range(1, 4):
        file(tmp_path, f"checkpoints/{i:08d}-aaaaaaaaaaaa.pkl")
    data = state(status=status, ended=NOW - timedelta(days=1))
    data[1][0]["checkpoint_ref"] = f"{PREFIX}/checkpoints/00000001-aaaaaaaaaaaa.pkl#sha256=x"
    candidates = planned(tmp_path, data)
    assert [c["path"] for c in candidates] == [f"{PREFIX}/checkpoints/00000002-aaaaaaaaaaaa.pkl"]


def test_active_restore_in_another_job_protects_terminal_checkpoint(tmp_path):
    logical = f"{PREFIX}/checkpoints/00000001-aaaaaaaaaaaa.pkl"
    file(tmp_path, "checkpoints/00000001-aaaaaaaaaaaa.pkl")
    data = state()
    data[0].append(dict(id="consumer", state="running", spec={}, updated_at=NOW))
    data[1].append(
        dict(
            id="restore",
            job_id="consumer",
            state="starting",
            checkpoint_ref=None,
            restore_checkpoint_ref=logical + "#sha256=x",
        )
    )
    assert planned(tmp_path, data) == []


def test_artifact_retention_override_and_active_consumer_pin(tmp_path):
    file(tmp_path, "committed/a.bin")
    data = state(retention=20 * 86400)
    data[2].append(dict(attempt_id="attempt-one", logical_path="a.bin", size_bytes=4))
    assert planned(tmp_path, data) == []
    data[0][0]["spec"] = {}
    assert len(planned(tmp_path, data)) == 1
    data[0].append(
        dict(
            id="consumer",
            state="queued",
            updated_at=NOW,
            spec={
                "assets": {"input": {"members": [{"logical_path": f"{PREFIX}/committed/a.bin"}]}}
            },
        )
    )
    assert planned(tmp_path, data) == []


def test_named_asset_pins_job_payload_permanently(tmp_path):
    store = AttemptArtifactStore(LocalStorage(tmp_path), job_id="job-one", attempt_id="attempt-one")
    store.stage_bytes("a.bin", b"test")
    manifest = store.commit(["a.bin"])
    AssetCatalog(tmp_path).publish(
        AssetPublishRequest(
            name="important",
            version="v1",
            job_id="job-one",
            path="a.bin",
        ),
        owner_username="alice",
        source_manifest=manifest,
    )
    file(tmp_path, "committed/a.bin")
    data = state()
    data[2].append(dict(attempt_id="attempt-one", logical_path="a.bin", size_bytes=4))
    assert planned(tmp_path, data) == []


def test_corrupt_catalog_defers_all_gc(tmp_path):
    file(tmp_path, "checkpoints/00000001-aaaaaaaaaaaa.pkl")
    manifest = tmp_path / "artifacts/catalog/v1/shared/name/v1/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("broken")
    with pytest.raises(Exception, match="catalog"):
        planned(tmp_path, state())


def test_dry_run_then_bounded_apply_is_audited_and_repeat_safe(tmp_path):
    for i in range(1, 4):
        file(tmp_path, f"checkpoints/{i:08d}-aaaaaaaaaaaa.pkl")
    policy = StorageRetentionSettings(max_files_per_pass=1)
    candidates = plan(tmp_path, policy, state(), now=NOW.timestamp())
    assert collect(tmp_path, policy, candidates)["deleted_files"] == 0
    result = collect(tmp_path, policy, candidates, apply=True)
    assert result["deleted_files"] == 1
    assert result["candidate_files"] == 3
    assert (tmp_path / ".topodrome-retention").is_dir()
    repeated = collect(tmp_path, policy, candidates, apply=True)
    assert repeated["deleted_files"] == 0
    assert repeated["skipped_changed_files"] == 1


def test_changed_file_and_symlink_are_not_removed(tmp_path):
    path = file(tmp_path, "checkpoints/00000001-aaaaaaaaaaaa.pkl")
    candidates = planned(tmp_path, state())
    path.write_bytes(b"changed")
    result = collect(tmp_path, StorageRetentionSettings(), candidates, apply=True)
    assert result["deleted_files"] == 0
    path.unlink()
    outside = tmp_path / "unrelated"
    outside.write_bytes(b"test")
    path.symlink_to(outside)
    assert planned(tmp_path, state()) == []
    assert outside.read_bytes() == b"test"


def test_symlink_parent_and_unknown_job_are_not_collected(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "00000001-aaaaaaaaaaaa.pkl").write_bytes(b"test")
    directory = tmp_path / PREFIX
    directory.mkdir(parents=True)
    (directory / "checkpoints").symlink_to(external, target_is_directory=True)
    assert planned(tmp_path, state()) == []
    assert planned(tmp_path, ([], [], [], [])) == []


def test_recent_file_is_protected_even_in_old_terminal_attempt(tmp_path):
    file(tmp_path, "checkpoints/00000001-aaaaaaaaaaaa.pkl", age=NOW)
    assert planned(tmp_path, state()) == []


def test_expired_input_rejected_without_rehash(tmp_path):
    member = SimpleNamespace(logical_path=f"{PREFIX}/committed/a", size_bytes=4)
    with pytest.raises(AdmissionRejectedError, match="expired"):
        require_members(tmp_path, [member])
    file(tmp_path, "committed/a")
    require_members(tmp_path, [member])


@pytest.mark.asyncio
async def test_cancellation_waits_for_disk_operation_before_releasing_guard():
    started = threading.Event()
    stop = threading.Event()
    completed = threading.Event()

    def operation():
        started.set()
        stop.wait(5)
        completed.set()

    task = asyncio.create_task(finish_io(operation))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    stop.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed.is_set()
