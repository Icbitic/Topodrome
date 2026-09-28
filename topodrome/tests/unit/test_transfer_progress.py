import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from topodrome.client.dashboard import _progress_text
from topodrome.domain.progress import PROGRESS_PREFIX, JobProgress
from topodrome.site_agent.transfer_progress import TransferLogReporter
from topodrome.storage import LocalStorage
from topodrome.storage.artifacts import AttemptArtifactStore
from topodrome.storage.progress import cancellable_transfers, observe_transfers, transfer_progress
from topodrome.storage.rclone_progress import run_upload

pytestmark = pytest.mark.unit


def test_stats_are_live_bounded_and_only_numeric(tmp_path):
    events = []
    script = (
        "import sys,json; "
        "sys.stderr.write('x'*200000+'\\n'); "
        "print(json.dumps({'stats':{'bytes':40},'msg':'SECRET'}),file=sys.stderr); "
        "print('{bad}',file=sys.stderr); "
        "print(json.dumps({'stats':{'bytes':999}}),file=sys.stderr)"
    )
    with observe_transfers(lambda *args: events.append(args)):
        result = run_upload([sys.executable, "-c", script], timeout=5, size=100)
    assert result.returncode == 0
    assert events == [("upload", 40, 100), ("upload", 100, 100)]


def test_timeout_kills_and_reaps_process():
    with pytest.raises(subprocess.TimeoutExpired):
        run_upload([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.1, size=5)


def test_cancellation_interrupts_upload_without_waiting_for_transfer_timeout():
    stopped = threading.Event()
    timer = threading.Timer(0.05, stopped.set)
    timer.start()
    try:
        with (
            cancellable_transfers(stopped),
            pytest.raises(RuntimeError, match="transfer canceled"),
        ):
            run_upload([sys.executable, "-c", "import time; time.sleep(10)"], timeout=20, size=5)
    finally:
        timer.cancel()
        timer.join()


def test_failed_upload_stays_failed():
    assert (
        run_upload([sys.executable, "-c", "raise SystemExit(2)"], timeout=5, size=5).returncode == 2
    )


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
