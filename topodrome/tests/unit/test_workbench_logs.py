from __future__ import annotations

import asyncio

import pytest
from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.color import Color
from textual.scrollbar import ScrollBarRender
from textual.widgets import Static

from tests.unit.test_workbench import FakeClient
from topodrome.client.workbench import TerminalDefaults, Workbench
from topodrome.client.workbench_logs import JobLogs, LogScroll, log_snapshot, progress_summary
from topodrome.domain.progress import PROGRESS_PREFIX, JobProgress

pytestmark = pytest.mark.unit


def record(**values):
    return "\n" + PROGRESS_PREFIX + JobProgress(**values).model_dump_json() + "\n"


def test_log_snapshot_collapses_only_valid_complete_progress_and_keeps_logs_literal():
    first = record(current=10, total=100, stage="upload", unit="bytes")
    last = record(current=80, total=100, stage="upload", unit="bytes")
    raw = "starting\n" + first + last + "Traceback: [bold]literal[/bold]\n\x1b]52;c;test\x07\n"
    text, progress = log_snapshot(raw)
    assert text == "starting\nTraceback: [bold]literal[/bold]\n]52;c;test\n"
    assert progress.current == 80 and progress.total == 100
    assert PROGRESS_PREFIX not in text


@pytest.mark.parametrize(
    "payload",
    [
        '{"current": 1}',  # complete JSON but no newline: not a committed record
        '{"current":',
        '{"current": -1}\n',
        '{"current": NaN}\n',
        '{"current": Infinity}\n',
        '{"total": 0}\n',
        '{"current": 1, "unknown": "error"}\n',
        '{"stage": "bad\\u001b"}\n',
        '{"message": "' + "x" * 5000 + '"}\n',
    ],
)
def test_invalid_or_partial_progress_records_remain_visible(payload):
    text, progress = log_snapshot(PROGRESS_PREFIX + payload)
    assert text == PROGRESS_PREFIX + payload
    assert progress is None


def test_progress_summary_formats_counts_eta_unknown_totals_and_overruns():
    progress = JobProgress(
        current=512 * 1024**2,
        total=1024**3,
        stage="upload",
        eta_seconds=62,
        unit="bytes",
    )
    text = progress_summary(progress).plain
    assert "[######------] 50%" in text and "0.5/1.0 GiB" in text
    assert "ETA 1m02s" in text and "Last reported" in text
    assert "total unknown" in progress_summary(JobProgress(current=0, stage="waiting-io")).plain
    overrun = progress_summary(JobProgress(current=110, total=100)).plain
    assert "100%" in overrun and "110/100" in overrun
    assert "succeeded" not in overrun
    almost = progress_summary(JobProgress(current=99.9, total=100)).plain
    assert "99%" in almost and "100%" not in almost


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("no_color", [False, True])
@pytest.mark.parametrize("background", ["white", "black"])
def test_scrollbar_thumb_stays_solid_and_draggable_without_backgrounds(
    vertical, no_color, background
):
    original = list(
        ScrollBarRender.render_bar(
            size=30, virtual_size=100, window_size=50, position=21, vertical=vertical
        ).segments
    )
    filtered = TerminalDefaults(no_color=no_color).apply(original, Color.parse(background))
    grabbed = []
    for index, (before, after) in enumerate(zip(original, filtered, strict=True)):
        assert after.style.bgcolor.is_default and not after.style.reverse
        meta = before.style.meta if before.style else {}
        assert after.style.meta == meta
        if meta.get("@mouse.down") == "grab":
            grabbed.append(index)
            assert after.text == "█" * len(before.text)
        else:
            assert after.text == before.text
    assert len(grabbed) > 10
    assert grabbed == list(range(min(grabbed), max(grabbed) + 1))
    unrelated = Segment("ordinary log output", Style(reverse=True))
    assert TerminalDefaults().apply([unrelated], Color.parse(background))[0].text == unrelated.text


class LogClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.output = "\n".join(f"line {n:04d}" for n in range(200)) + "\n"
        self.log_error = False
        self.log_reads = 0

    async def logs(self, job, **kwargs):
        self.log_reads += 1
        if self.log_error:
            raise RuntimeError("DO_NOT_DISPLAY_SECRET")
        return self.output


async def refresh_logs(app, pilot):
    app.schedule_detail()
    await pilot.pause()


@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
@pytest.mark.parametrize("navigation", ["keyboard", "mouse", "drag"])
async def test_logs_follow_pause_freeze_rolling_tail_and_resume(size, navigation):
    client = LogClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("enter", "3")
        scroll = app.widget("#scroll-logs", LogScroll)
        body = app.widget("#detail-logs", Static)
        assert scroll.following and scroll.scroll_y == scroll.max_scroll_y > 0
        client.output += "new line\n"
        await refresh_logs(app, pilot)
        assert "new line" in str(body.render())
        assert scroll.scroll_y == scroll.max_scroll_y
        if navigation == "keyboard":
            await pilot.press("pageup")
        elif navigation == "mouse":
            scroll.post_message(events.MouseScrollUp(scroll, 1, 1, 0, -1, 0, False, False, False))
            await pilot.pause()
        else:
            # Native scrollbars post ScrollTo during dragging.
            from textual.scrollbar import ScrollTo

            scroll.post_message(ScrollTo(y=10, animate=False))
            await pilot.pause()
        assert not scroll.following
        frozen, position = str(body.render()), scroll.scroll_y
        client.output = "\n".join(f"later {n:04d}" for n in range(300, 500)) + "\n"
        client.output += record(current=50, total=100, stage="upload")
        await refresh_logs(app, pilot)
        assert str(body.render()) == frozen and scroll.scroll_y == position
        assert "paused" in str(app.widget("#log-status", Static).render())
        assert "50%" in str(app.widget("#log-progress", Static).render())
        await pilot.press("end")
        assert scroll.following and scroll.scroll_y == scroll.max_scroll_y
        assert "later 0499" in str(body.render()) and "line 0001" not in str(body.render())
        assert PROGRESS_PREFIX not in str(body.render())
        assert app.focused is scroll
        assert not client.calls


async def test_logs_are_polled_without_keys_and_independently_of_slow_job_details():
    client = LogClient()
    gate = asyncio.Event()
    original = client.status

    async def delayed(job):
        await gate.wait()
        return await original(job)

    client.status = delayed
    app = Workbench(client, interval=0.1)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "3")
        reads = client.log_reads
        client.output += "arrived without a key\n"
        for _ in range(20):
            if "arrived without a key" in str(app.widget("#detail-logs", Static).render()):
                break
            await pilot.pause(0.05)
        assert "arrived without a key" in str(app.widget("#detail-logs", Static).render())
        assert client.log_reads > reads
        assert not app._detail_worker.is_finished
        assert "fetched" in str(app.widget("#log-status", Static).render())
        gate.set()


async def test_log_failure_keeps_output_and_has_own_stale_indicator_then_recovers():
    client = LogClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "3")
        before = str(app.widget("#detail-logs", Static).render())
        client.log_error = True
        await refresh_logs(app, pilot)
        status = str(app.widget("#log-status", Static).render())
        assert "Logs stale" in status and "RuntimeError" in status
        assert "DO_NOT_DISPLAY_SECRET" not in status
        assert str(app.widget("#detail-logs", Static).render()) == before
        assert "LIVE" in str(app.widget("#status", Static).render())
        client.log_error = False
        client.output += "recovered\n"
        await refresh_logs(app, pilot)
        assert "Logs stale" not in str(app.widget("#log-status", Static).render())
        assert "recovered" in str(app.widget("#detail-logs", Static).render())


async def test_late_logs_never_retarget_details_or_steal_focus_even_after_returning_to_job():
    client = LogClient()
    gate = asyncio.Event()
    entered = asyncio.Event()
    delayed_once = True

    async def logs(job, **kwargs):
        nonlocal delayed_once
        if delayed_once:
            delayed_once = False
            entered.set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                # Simulate a transport that finishes despite cancellation.
                await gate.wait()
            return "old response must not replace new logs\n"
        return f"fresh {job}\n"

    client.logs = logs
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert entered.is_set()
        await pilot.press("down")
        await pilot.press("up", "enter")
        assert app.focused.id == "scroll-summary"
        assert "fresh job_abcd1111" in str(app.widget("#detail-logs", Static).render())
        gate.set()
        await pilot.pause()
        assert "fresh job_abcd1111" in str(app.widget("#detail-logs", Static).render())
        assert app.focused.id == "scroll-summary"
        assert not client.calls


async def test_one_row_scroll_pauses_and_manual_return_to_bottom_resumes_following():
    client = LogClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "3", "up")
        scroll = app.widget("#scroll-logs", LogScroll)
        assert not scroll.following
        before = str(app.widget("#detail-logs", Static).render())
        client.output += "latest pending line\n"
        await refresh_logs(app, pilot)
        assert str(app.widget("#detail-logs", Static).render()) == before
        await pilot.press("down")
        await pilot.pause()
        assert scroll.following and scroll.scroll_y == scroll.max_scroll_y
        assert "latest pending line" in str(app.widget("#detail-logs", Static).render())


async def test_log_follow_survives_tab_changes_resize_and_resets_for_another_job():
    client = LogClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "3")
        scroll = app.widget("#scroll-logs", LogScroll)
        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        assert scroll.following and scroll.scroll_y == scroll.max_scroll_y
        await pilot.press("pageup", "1", "3")
        assert not scroll.following
        await pilot.press("escape", "down")
        client.output = "different job output\n"
        await refresh_logs(app, pilot)
        await pilot.press("enter", "3")
        assert scroll.following
        assert "different job output" in str(app.widget("#detail-logs", Static).render())
        assert not app.widget("#log-progress", Static).display


async def test_empty_and_telemetry_only_logs_have_useful_text_and_no_false_completion():
    client = LogClient()
    client.output = ""
    app = Workbench(client, interval=60)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "3")
        assert "No text output" in str(app.widget("#detail-logs", Static).render())
        client.output = record(current=100, total=100, stage="verify", unit="bytes")
        await refresh_logs(app, pilot)
        assert "No text output" in str(app.widget("#detail-logs", Static).render())
        assert "100%" in str(app.widget("#log-progress", Static).render())
        assert app.selected()["state"] == "running"
        logs = app.widget("#job-logs", JobLogs)
        logs.fetched_at -= 200
        logs.update_status()
        assert "Logs stale" in str(app.widget("#log-status", Static).render())
