import base64
import hashlib
import json

import pytest

from topodrome.client.dashboard import _display_width, _job_lines, _progress_text, render_dashboard
from topodrome.domain.logs import LiveLogChunkReport
from topodrome.domain.progress import PROGRESS_PREFIX
from topodrome.runtime.context import ExperimentContext
from topodrome.scheduler.live_logs import LiveLogStore


def test_context_progress_throttles_and_survives_output_failure(monkeypatch, capsys):
    clock = [0.0]
    monkeypatch.setattr("topodrome.runtime.context.time.monotonic", lambda: clock[0])
    c = ExperimentContext(model=None, tokenizer=None, device=None, job_id="j", attempt_id="a")
    assert c.report_progress(10, 100, stage="train")
    assert not c.report_progress(11, 100)
    clock[0] = 3
    c.report_progress(20, 100, stage="train")
    clock[0] = 6
    c.report_progress(30, 100, stage="train")
    records = [
        json.loads(x.split(PROGRESS_PREFIX)[1])
        for x in capsys.readouterr().out.splitlines()
        if PROGRESS_PREFIX in x
    ]
    assert records[-1]["eta_seconds"] == 21
    clock[0] = 9
    monkeypatch.setattr("builtins.print", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert not c.report_progress(40, 100)
    with pytest.raises(ValueError):
        c.report_progress(101, 100)


def test_progress_split_records_retry_isolation_restart_and_generation(tmp_path):
    store = LiveLogStore(tmp_path)
    record = (
        "\n" + PROGRESS_PREFIX + json.dumps({"current": 12, "total": 100, "stage": "train"}) + "\n"
    ).encode()

    def append(data, offset=0, generation=0):
        return store.append(
            LiveLogChunkReport(
                site_id="s",
                job_id="j",
                attempt_id="a",
                fencing_token="test",  # noqa: S106 - inert test token
                generation=generation,
                offset=offset,
                data_base64=base64.b64encode(data).decode(),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )

    append(record[:30])
    assert store.read_progress("j", "a") is None
    append(record[30:], 30)
    assert store.read_progress("j", "a")["current"] == 12
    assert LiveLogStore(tmp_path).read_progress("j", "a")["current"] == 12
    assert store.read_progress("j", "retry") is None
    assert append(record).duplicate
    append(b"new generation\n", generation=1)
    assert store.read_progress("j", "a") is None


def test_dashboard_short_id_and_progress():
    lines = _job_lines(
        [
            {
                "job_id": "job_abcd123456",
                "state": "running",
                "progress": {"current": 12, "total": 100, "stage": "train"},
            }
        ],
        color=False,
        width=140,
    )
    assert "abcd " in lines[-1] and "abcd123456" not in lines[-1]
    assert "12% 12/100 train" in lines[-1]


@pytest.mark.parametrize("width", [40, 52, 60, 79, 80, 100, 109, 110, 123, 140, 180])
def test_verbose_progress_keeps_numbers_and_eta_visible(width):
    progress = {
        "current": 12,
        "total": 100,
        "eta_seconds": 90,
        "stage": "very long detailed training stage " * 2,
        "message": "loss=0.42 lots of verbose details " * 8,
    }
    frame = render_dashboard(
        {
            "jobs": [
                {
                    "job_id": "job_abcd1234",
                    "name": "Verbose experiment " * 4,
                    "site_id": "remote-jupyter-1",
                    "state": "running",
                    "progress": progress,
                }
            ]
        },
        width=width,
        height=25,
    )
    row = next(line for line in frame.splitlines() if line.startswith("  abcd "))
    assert "12%" in row and "ETA 1m30s" in row
    assert all(_display_width(line) <= width for line in frame.splitlines())
    assert len(frame.splitlines()) <= 25
    assert progress["stage"].startswith("very long detailed")  # Input is never rewritten.


def test_large_counters_do_not_hide_percentage_or_eta():
    text = _progress_text(
        {
            "current": 123456789012345,
            "total": 246913578024690,
            "eta_seconds": 7200,
            "stage": "verbose" * 9,
        },
        width=24,
    )
    assert "50%" in text and "ETA 2h00m" in text
    assert _display_width(text) <= 24


def test_unicode_stale_and_unknown_total_progress():
    text = _progress_text(
        {
            "current": 5,
            "total": 10,
            "stage": "训练阶段" * 16,
            "message": "loss=1\nnext\x1b[2J",
            "stale": True,
            "eta_seconds": 45,
        },
        width=30,
    )
    assert text.startswith("~ 50% 5/10") and "ETA" not in text
    assert "\n" not in text and "\x1b" not in text
    assert _display_width(text) <= 30
    unknown = _progress_text({"current": 42, "stage": "scan", "message": "file 1"}, width=30)
    assert unknown == "42 scan file 1"
