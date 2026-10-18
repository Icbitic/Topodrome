import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from topodrome.client.dashboard import _progress_text
from topodrome.domain.progress import PROGRESS_PREFIX, JobProgress
from topodrome.site_agent.transfer_progress import TransferLogReporter
from topodrome.storage import LocalStorage
from topodrome.storage.artifacts import AttemptArtifactStore
from topodrome.storage.progress import observe_transfers, transfer_progress

pytestmark = pytest.mark.unit


def test_observers_are_nested_and_thread_local():
    events = []
    with observe_transfers(lambda *args: events.append(args)):
        transfer_progress("outer", 1, 3)
        with ThreadPoolExecutor() as pool:
            pool.submit(transfer_progress, "other-thread", 1, 3).result()
        with observe_transfers(lambda *_: 1 / 0):
            transfer_progress("inner", 1, 3)
        transfer_progress("outer", 2, 3)
    transfer_progress("outside", 3, 3)
    assert events == [("outer", 1, 3), ("outer", 2, 3)]


def test_artifact_progress_aggregates_without_weakening_commit(tmp_path):
    events = []
    store = AttemptArtifactStore(LocalStorage(tmp_path), job_id="job-a", attempt_id="attempt-a")
    store.stage_bytes("a", b"abc")
    store.stage_bytes("b", b"defgh")
    result = store.commit(["b", "a"], progress=lambda *args: events.append(args))
    assert sum(a.size_bytes for a in result.artifacts) == 8
    assert events == [
        ("verify", 0, 8),
        ("verify", 3, 8),
        ("verify", 3, 8),
        ("verify", 8, 8),
        ("manifest", 8, 8),
    ]


def test_progress_sink_failure_does_not_fail_commit(tmp_path):
    store = AttemptArtifactStore(LocalStorage(tmp_path), job_id="job-a", attempt_id="attempt-a")
    store.stage_bytes("a", b"abc")
    assert store.commit(["a"], progress=lambda *_: 1 / 0).artifacts[0].size_bytes == 3


def test_reporter_records_and_dashboard(tmp_path):
    log = tmp_path / "attempt.log"
    reporter = TransferLogReporter(log)
    reporter("upload", 1024**3, 2 * 1024**3)
    reporter("upload", 1024**3, 2 * 1024**3)  # throttled
    reporter("verify", 2 * 1024**3, 2 * 1024**3)
    records = [
        JobProgress.model_validate_json(line.removeprefix(PROGRESS_PREFIX))
        for line in log.read_text().splitlines()
        if line
    ]
    assert len(records) == 2
    assert _progress_text(records[0].model_dump(), width=40) == "upload 50% 1.0GiB/2.0GiB"
    assert _progress_text(records[1].model_dump(), width=40).startswith("verify 100%")
    assert _progress_text({"unit": "bytes", "stage": "logs"}) == "logs"
    assert len(_progress_text(records[0].model_dump(), width=12)) <= 12


def test_progress_live_log_schema_is_compatible():
    raw = json.dumps({"current": 1, "stage": "training"})
    assert JobProgress.model_validate_json(raw).unit == "items"
