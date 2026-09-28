"""Bounded, terminal-safe log snapshots with a readable progress summary."""

import time
import unicodedata

from pydantic import ValidationError
from rich.text import Text
from textual import on
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static

from topodrome.domain.progress import PROGRESS_PREFIX, PROGRESS_RECORD_BYTES, JobProgress

from .workbench_details import duration


def log_snapshot(raw):
    """Hide only complete, valid telemetry; preserve errors and partial records."""
    lines = []
    progress = None
    for line in raw[:70000].splitlines(keepends=True):
        if line.startswith(PROGRESS_PREFIX) and line.endswith("\n"):
            payload = line[len(PROGRESS_PREFIX) :].rstrip("\r\n")
            if len(payload.encode("utf-8")) <= PROGRESS_RECORD_BYTES:
                try:
                    progress = JobProgress.model_validate_json(payload)
                except ValidationError:
                    pass
                else:
                    # The reporter emits a leading newline to separate its
                    # record from workload output. Don't leave a blank row for
                    # every hidden progress update.
                    if lines and lines[-1] in {"\n", "\r\n"}:
                        lines.pop()
                    continue
        lines.append(line)
    text = "".join(lines)
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C") or c in "\n\t")
    return text, progress


def progress_summary(progress):
    """A progress meter is advisory, never proof of durable job completion."""
    result = Text("Last reported · ", style="dim")
    result.append((progress.stage or "working")[:32] + " ", style="cyan")
    if progress.total is not None:
        fraction = min(1.0, progress.current / progress.total)
        filled = int(fraction * 12)
        result.append("[" + "#" * filled + "-" * (12 - filled) + "] ")
        # Rounding 99.9% to 100% would contradict a not-yet-full meter.
        result.append(f"{int(fraction * 100)}%", style="bold")
    else:
        result.append("total unknown")
    if progress.unit == "bytes":
        scale, unit = (
            (1024**3, "GiB")
            if max(progress.current, progress.total or 0) >= 1024**3
            else (1024**2, "MiB")
        )
        result.append(f" · {progress.current / scale:.1f}")
        if progress.total is not None:
            result.append(f"/{progress.total / scale:.1f}")
        result.append(f" {unit}")
    else:
        result.append(f" · {progress.current:g}")
        if progress.total is not None:
            result.append(f"/{progress.total:g}")
    if progress.eta_seconds is not None:
        result.append(f" · ETA {duration(progress.eta_seconds)}")
    if progress.message:
        result.append(" · " + progress.message)
    return result


class LogScroll(VerticalScroll):
    """Use Textual's native bottom anchor for keys, wheel and scrollbar drag."""

    class FollowChanged(Message):
        pass

    @property
    def following(self):
        # Textual exposes anchor()/release_anchor(), but not the release state.
        # Keep this dependency isolated; never modify its internal scroll state.
        return self.is_anchored and not self._anchor_released

    def release_anchor(self):
        super().release_anchor()
        if self.is_mounted:
            self.post_message(self.FollowChanged())

    def watch_scroll_y(self, old_value, new_value):
        super().watch_scroll_y(old_value, new_value)
        if self.is_mounted:
            self.post_message(self.FollowChanged())

    def action_scroll_end(self):
        self.anchor()
        self.post_message(self.FollowChanged())

    # Terminal log navigation should settle immediately. An animated first
    # step can still be at the bottom when a refresh/resize rechecks anchoring.
    def action_scroll_up(self):
        self.scroll_up(animate=False, immediate=True)

    def action_scroll_down(self):
        self.scroll_down(animate=False, immediate=True)

    def action_page_up(self):
        self.scroll_page_up(animate=False)

    def action_page_down(self):
        self.scroll_page_down(animate=False)

    def action_scroll_home(self):
        self.scroll_home(animate=False, immediate=True)


class JobLogs(Vertical):
    DEFAULT_CSS = """
    JobLogs { height: 1fr; }
    #log-status { height: 1; padding: 0 1; }
    .narrow #log-status { height: 2; }
    #log-progress { height: auto; max-height: 2; padding: 0 1; }
    #scroll-logs { height: 1fr; }
    """

    def __init__(self, *, interval=2.0):
        super().__init__(id="job-logs")
        self.interval = interval
        self.fetched_at = None
        self.error = None
        self._raw = None
        self._latest = ""
        self._displayed = None

    def compose(self):
        yield Static("Waiting for logs…", id="log-status", markup=False)
        yield Static("", id="log-progress", markup=False)
        with LogScroll(id="scroll-logs"):
            yield Static("", id="detail-logs", classes="detail-text", markup=False)

    def on_mount(self):
        self.reset()
        self.set_interval(1, self.update_status)

    def reset(self):
        self.fetched_at, self.error, self._raw = None, None, None
        self._latest, self._displayed = "", None
        self.query_one("#log-progress", Static).display = False
        self.query_one(LogScroll).anchor()
        self.show_latest()
        self.update_status()

    def receive(self, raw):
        self.fetched_at = time.monotonic()
        self.error = None
        if raw != self._raw:
            self._raw = raw
            self._latest, progress = log_snapshot(raw)
            widget = self.query_one("#log-progress", Static)
            widget.display = progress is not None
            if progress is not None:
                widget.update(progress_summary(progress))
        if self.query_one(LogScroll).following:
            self.show_latest()
        self.update_status()

    def failed(self, error):
        self.error = error
        self.update_status()

    def show_latest(self):
        # Freeze the displayed snapshot while reading earlier output, even if
        # the server's bounded tail rolls forward. Keep only one pending tail.
        text = self._latest or (
            "No text output in the recent log tail." if self._raw is not None else ""
        )
        if text != self._displayed:
            self._displayed = text
            self.query_one("#detail-logs", Static).update(Text(text))

    @on(LogScroll.FollowChanged)
    def follow_changed(self):
        if self.query_one(LogScroll).following:
            self.show_latest()
        self.update_status()

    def update_status(self):
        if not self.is_mounted:
            return
        elapsed = int(time.monotonic() - self.fetched_at) if self.fetched_at is not None else None
        stale = self.error is not None or (
            elapsed is not None and elapsed > max(10, self.interval * 3)
        )
        status = Text("Logs stale" if stale else "Logs", style="yellow" if stale else "dim")
        if self.error:
            status.append(f" · {self.error}")
        status.append(f" · fetched {elapsed}s ago" if elapsed is not None else " · fetching…")
        status.append(
            " · following · ↑/PgUp pauses"
            if self.query_one(LogScroll).following
            else " · paused · End follows"
        )
        status.append(" · 64 KiB tail")
        self.query_one("#log-status", Static).update(status)
