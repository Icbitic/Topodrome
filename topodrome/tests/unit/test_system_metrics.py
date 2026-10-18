import asyncio
import os
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from topodrome.client.workbench_system import size, system_rows
from topodrome.config import StorageRetentionSettings
from topodrome.deployment.upgrade_store import UpgradeStore
from topodrome.scheduler.assets import AssetCatalog, AssetPublishRequest
from topodrome.scheduler.system_metrics import (
    ScanBudget,
    SystemMetrics,
    disk_usage,
    reclaimable,
    sample_storage,
)
from topodrome.storage.artifacts import AttemptArtifactStore
from topodrome.storage.blobs import blob_transaction
from topodrome.storage.local import LocalStorage

pytestmark = pytest.mark.unit


def test_allocated_usage_deduplicates_hardlinks_and_ignores_symlinks(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    file = root / "file"
    file.write_bytes(b"a" * 5000)
    (root / "duplicate").hardlink_to(file)
    (root / "outside").symlink_to(tmp_path.parent, target_is_directory=True)
    report = disk_usage(root, ScanBudget())
    assert report["allocated_bytes"] == file.stat().st_blocks * 512
    assert report["logical_bytes"] == 5000
    assert report["files"] == 1
    assert report["skipped_entries"] == 1
    assert reclaimable(root, [{"path": "file"}], ScanBudget()) == 0
    assert (
        reclaimable(root, [{"path": "file"}, {"path": "duplicate"}], ScanBudget())
        == report["allocated_bytes"]
    )


def test_missing_index_is_unknown_and_read_only(tmp_path):
    report = sample_storage(tmp_path, "blobs-v1", StorageRetentionSettings(), ([], [], [], []))
    assert report["allocated_bytes"] == 0
    assert report["estimate_state"] == "unavailable"
    assert "eligible_24h_bytes" not in report
    assert list(tmp_path.iterdir()) == []


def test_scan_budget_stops():
    with pytest.raises(TimeoutError):
        ScanBudget(entries=0)()
    stop = threading.Event()
    stop.set()
    with pytest.raises(TimeoutError):
        ScanBudget(stop)()


@pytest.mark.parametrize("pin", [None, "asset", "queued", "running", "retry_wait", "lease"])
def test_forecast_uses_references_and_never_deletes(tmp_path, pin):
    storage = LocalStorage(tmp_path)
    storage.content_addressed = True
    store = AttemptArtifactStore(storage, job_id="job-one", attempt_id="attempt-one")
    store.stage_bytes("out", b"payload" * 1000)
    manifest = store.commit(["out"])
    logical = manifest.artifacts[0].storage_path
    blob = tmp_path / logical
    old = datetime.now(UTC) - timedelta(days=6, hours=12)
    with blob_transaction(tmp_path) as index:
        index.execute("UPDATE objects SET touched=?", (old.timestamp(),))
    os.utime(blob, (old.timestamp(), old.timestamp()))
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
    if pin == "asset":
        AssetCatalog(tmp_path, content_addressed=True).publish(
            AssetPublishRequest(name="keep", version="v1", job_id="job-one", path="out"),
            owner_username="alice",
            source_manifest=manifest,
        )
    elif pin == "lease":
        data[3].append(dict(attempt_id="attempt-one", state="active"))
    elif pin:
        data[0].append(
            dict(
                id="consumer",
                state=pin,
                updated_at=old,
                spec={"assets": {"input": {"members": [{"logical_path": logical}]}}},
            )
        )
    report = sample_storage(tmp_path, "blobs-v1", StorageRetentionSettings(), data)
    assert report["estimate_state"] == "ready"
    assert report["eligible_now_bytes"] == 0
    assert report["eligible_24h_bytes"] == (0 if pin else blob.stat().st_blocks * 512)
    assert blob.read_bytes() == b"payload" * 1000
    assert not (tmp_path / ".topodrome-retention").exists()


@pytest.mark.asyncio
async def test_cache_singleflight_stale_failure_and_close(tmp_path, monkeypatch):
    import topodrome.scheduler.system_metrics as module

    service = SimpleNamespace(
        database=SimpleNamespace(ready=AsyncMock(return_value=True)),
        settings=SimpleNamespace(
            global_root=tmp_path,
            storage_layout="attempt-v1",
            storage_retention=StorageRetentionSettings(),
        ),
        connections=SimpleNamespace(identities=AsyncMock(return_value=[])),
        policy_status=AsyncMock(return_value={}),
        upgrades=SimpleNamespace(store=UpgradeStore(tmp_path)),
    )
    gate = asyncio.Event()
    calls = []

    async def snapshot(*args, **kwargs):
        calls.append(1)
        await gate.wait()
        return [], [], [], []

    monkeypatch.setattr(module, "snapshot", snapshot)
    metrics = SystemMetrics(service)
    await asyncio.gather(*(metrics.get() for _ in range(5)))
    assert len(calls) == 1
    gate.set()
    await metrics.task
    assert metrics.storage["allocated_bytes"] == 0
    await metrics.get()
    assert len(calls) == 1
    metrics.last_attempt -= 301
    monkeypatch.setattr(module, "snapshot", AsyncMock(side_effect=ValueError("SECRET")))
    await metrics.get()
    await metrics.task
    report = await metrics.get()
    assert report["storage_state"] == "stale"
    assert report["storage_error"] == "ValueError"
    assert "SECRET" not in str(report)
    await metrics.close()


def test_rows_distinguish_unknown_and_disabled_cleanup():
    rows = {
        r["metric_id"]: r
        for r in system_rows(
            {
                "storage": {"estimate_state": "ready", "eligible_24h_bytes": 1024},
                "retention": {"enabled": False},
            }
        )
    }
    assert rows["used"]["value"] == "—"
    assert rows["eligible-day"]["value"] == "1.0 KiB"
    assert "OFF" in rows["eligible-day"]["note"]
    assert size(0) == "0.0 B"
    assert system_rows(None, "error")[0]["value"] == "Unavailable"
