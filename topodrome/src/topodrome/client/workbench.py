"""Interactive API-only workbench. No scheduler, SSH or filesystem administration."""

from __future__ import annotations

import asyncio
import math
import time
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import ClassVar
from uuid import uuid4

from rich.color import ColorType
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.filter import NoColor
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    OptionList,
    Static,
    Tab,
    TabbedContent,
    TabPane,
    Tabs,
)
from textual.widgets.option_list import Option

from topodrome.domain.scheduling import SitePreferenceRequest
from topodrome.domain.specs import GPUReservationExtensionRequest, GPUReservationRequest

from .async_client import AsyncClient
from .dashboard import _gpu_owner_color, _gpu_util_color, _progress_text
from .workbench_arrays import array_row, job_tree
from .workbench_assets import asset_cells, asset_group, asset_tree
from .workbench_autodl import autodl_rows
from .workbench_details import attempts_page, date, detail_page, models_page, timing_cell
from .workbench_gpus import gpu_cells, gpu_key, gpu_tree
from .workbench_impact import reservation_impact
from .workbench_logs import JobLogs
from .workbench_site_order import SiteOrderDraft
from .workbench_system import system_rows

VIEWS = ("jobs", "gpus", "sites", "assets", "reservations", "autodl", "system")
VIEW_LABELS = {"gpus": "GPUs", "autodl": "AutoDL"}
ACTION_LABELS = {
    "site-order": "Reorder sites",
    "reserve": "Reserve GPU…",
    "extend": "Extend reservation…",
    "unreserve": "Cancel reservation…",
    "release": "Release instance…",
    "unregister": "Remove offline registration…",
}


@dataclass
class ActionFocus:
    """A one-shot focus intent, bound to a resource rather than a row number."""

    view: str
    key: str | None
    details: bool
    operation: str | None = None
    reservation_id: str | None = None
    check_reservation: bool = False


class ChoiceDialog(ModalScreen):
    DEFAULT_CSS = """
    ChoiceDialog { align: center middle; background: ansi_default; }
    #choices { width: 82; max-width: 96%; height: auto; max-height: 95%;
               border: round ansi_default; padding: 1 2; }
    #choices Static { height: auto; margin-bottom: 1; }
    #choices OptionList { height: auto; max-height: 12; border: none; }
    OptionList > .option-list--option-highlighted { text-style: bold underline; }
    """
    BINDINGS: ClassVar = [("escape", "back", "Back")]

    def __init__(self, title, choices=(), *, searchable=False):
        super().__init__()
        self.title_text = title
        self.choices = list(choices)
        self.matches = self.choices
        self.searchable = searchable

    def compose(self):
        with VerticalScroll(id="choices"):
            yield Static(
                self.title_text if isinstance(self.title_text, Text) else Text(self.title_text)
            )
            if self.searchable:
                yield Input(placeholder="Type a command…", id="command-query")
            if self.choices:
                yield OptionList(*(Text(label) for _, label in self.choices))
            yield Static(
                "↑↓ choose · Enter select · Esc back"
                if self.choices
                else "↑↓ / PgUp / PgDn scroll · Esc back"
            )

    @on(OptionList.OptionSelected)
    def choose(self, event):
        if event.option_index < len(self.matches):
            self.dismiss(self.matches[event.option_index][0])

    def on_mount(self):
        if self.searchable:
            self.query_one(Input).focus()
        elif self.choices:
            self.query_one(OptionList).focus()
        else:
            self.set_focus(self.query_one("#choices", VerticalScroll))

    @on(Input.Changed, "#command-query")
    def filter_commands(self, event):
        self.matches = [c for c in self.choices if event.value.casefold() in c[1].casefold()]
        options = self.query_one(OptionList)
        options.clear_options()
        options.add_options(Text(label) for _, label in self.matches)
        options.highlighted = 0 if self.matches else None

    @on(Input.Submitted, "#command-query")
    def choose_first(self):
        if self.matches:
            index = self.query_one(OptionList).highlighted or 0
            self.dismiss(self.matches[index][0])

    def on_key(self, event):
        if self.searchable and isinstance(self.focused, Input) and event.key in {"up", "down"}:
            options = self.query_one(OptionList)
            if event.key == "down":
                options.action_cursor_down()
            else:
                options.action_cursor_up()
            event.prevent_default()
            event.stop()

    def action_back(self):
        self.dismiss(None)


class DetailActions(OptionList):
    """A literal selection marker remains clear without background colors."""

    def on_focus(self):
        # Native Tab/click navigation can re-enter a list whose old action was
        # removed. Give Enter a visible target on this deliberate re-entry.
        if self.highlighted is None and self.option_count:
            self.highlighted = 0

    def watch_highlighted(self, highlighted):
        super().watch_highlighted(highlighted)
        for index in range(self.option_count):
            option = self.get_option_at_index(index)
            marker = "> " if index == highlighted else "  "
            self.replace_option_prompt_at_index(index, Text(marker + ACTION_LABELS[option.id]))


class WorkbenchTabs(TabbedContent):
    @on(TabPane.Focused)
    def pane_focused(self, event):
        # Textual normally activates any pane that reports descendant focus.
        # That message is asynchronous: an earlier focus can arrive after the
        # user has switched panes, or started typing in the filter. Only the
        # pane containing the *current* focus may activate itself.
        event.stop()
        event.prevent_default()
        focused = self.screen.focused
        if focused is not None and event.tab_pane in focused.ancestors:
            self.active = event.tab_pane.id


class TerminalDefaults(NoColor):
    """Allow a small foreground palette, never painted backgrounds or inversion."""

    def __init__(self, *, no_color=False):
        super().__init__()
        self.no_color = no_color

    def apply(self, segments, background):
        return [
            Segment(
                # Textual paints scrollbar interiors using reversed spaces.
                # Stripping backgrounds/inversion otherwise erases the thumb.
                # Foreground cells preserve native terminal colors and all
                # original mouse hit/drag metadata, vertically and horizontally.
                "█" * len(text) if style and style.meta.get("@mouse.down") == "grab" else text,
                (style or Style())
                + Style(
                    color=(
                        style.color
                        if not self.no_color
                        and style
                        and style.color
                        and style.color.type == ColorType.STANDARD
                        and style.color.number in {1, 2, 3, 6}
                        else "default"
                    ),
                    bgcolor="default",
                    reverse=False,
                ),
                control,
            )
            for text, style, control in segments
        ]


def status_cell(value, column):
    """Color status fields only, never user-controlled names, paths or logs."""
    color = None
    if column == "Pref" and value.startswith("#"):
        color = "cyan"
    if column == "Owner":
        color = _gpu_owner_color(value)
    if column == "Util" and value.endswith("%"):
        try:
            color = _gpu_util_color(float(value[:-1]))
        except ValueError:
            pass
    if column in {"State", "Health", "Storage"}:
        if value in {"ready", "succeeded", "ok"}:
            color = "green"
        elif value in {"failed", "offline", "unavailable"}:
            color = "red"
        elif value in {
            "queued",
            "held",
            "retry_wait",
            "cancelling",
            "canceling",
            "stale",
            "draining",
        }:
            color = "yellow"
        elif value in {"running", "starting", "downloading", "finalizing"}:
            color = "cyan"
    return Text(value, style=color or "")


def job_columns(width):
    """Keep useful columns on one page; complete timing stays in details."""
    if width >= 100:
        fixed = [("ID", 4), ("Name", 18), ("State", 11), ("Site", 14), ("Wait / Run / I/O", 21)]
        return [*fixed, ("Progress / ETA", max(12, width - 82))]
    if width >= 80:
        return [
            ("ID", 4),
            ("Name", 14),
            ("State", 11),
            ("Wait / Run / I/O", 21),
            ("Progress / ETA", width - 61),
        ]
    if width >= 58:
        return [("ID", 4), ("Name", width - 45), ("State", 11), ("Wait / Run / I/O", 21)]
    return [("ID", 4), ("Name", max(6, width - 23)), ("State", 11)]


def plain(value, *, multiline=False, limit=4000):
    """Never interpret server/user text as markup or terminal control sequences."""
    text = str(value if value is not None else "-")[:limit]
    return "".join(
        c
        for c in text
        if not unicodedata.category(c).startswith("C") or (multiline and c in "\n\t")
    )


def mapping(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)


def reservation_active(row):
    try:
        return row.get("state") in {"pending", "ready"} and datetime.fromisoformat(
            row["expires_at"].replace("Z", "+00:00")
        ) > datetime.now(UTC)
    except (KeyError, TypeError, ValueError):
        return False


def reservation_caption(row):
    """Human-readable remaining time; the full expiry includes its timezone."""
    end = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
    seconds = max(0, int((end - datetime.now(UTC)).total_seconds()))
    minutes = (seconds + 59) // 60
    remaining = f"{minutes // 60}h {minutes % 60}m" if minutes >= 60 else f"{minutes}m"
    return f"{remaining} left · until {date(row['expires_at'])}"


def visible_snapshot(snapshot):
    """Keep offline site metadata, suppress obsolete GPU inventory only."""
    hidden = {
        s["site_id"] for s in snapshot.get("sites", []) if s.get("state") in {"offline", "stopped"}
    }
    return {
        **snapshot,
        "sites": list(snapshot.get("sites", [])),
        "gpus": [g for g in snapshot.get("gpus", []) if g.get("site_id") not in hidden],
    }


def age(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return f"{max(0, int((datetime.now(UTC) - stamp).total_seconds()))}s"
    except (ValueError, TypeError):
        return "unknown"


def duration_seconds(value):
    """Accept seconds or familiar whole-number s/m/h/d durations."""
    value = value.strip().lower()
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return int(value[:-1]) * scale[value[-1]] if value[-1:] in scale else int(value)


def safe_error(exc):
    """Expose HTTP status, never a response body, authenticated URL or token."""
    import httpx

    cause = exc.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        return f"HTTP {cause.response.status_code}"
    return type(exc).__name__


class PlainCheckbox(Checkbox):
    """The value must remain legible even with all terminal colors disabled."""

    def __init__(self, label, **kwargs):
        self.caption = label
        super().__init__(label, **kwargs)

    def render(self):
        return Text.assemble(
            ("> " if self.has_focus else "  ", "cyan"),
            ("[x]" if self.value else "[ ]", "bold"),
            " ",
            self.caption,
        )


class FormButton(Button):
    """Don't swallow deliberate keyboard presses during the click animation."""

    def action_press(self):
        # Mouse clicks retain Button's double-click debounce. Keyboard users
        # can review then confirm with separate Enter presses without a delay.
        self.press()


class ActionDialog(ModalScreen):
    """Capture the full immutable target before showing an explicit confirmation."""

    DEFAULT_CSS = """
    ActionDialog { align: center middle; background: ansi_default; }
    #dialog { width: 76; max-width: 96%; height: auto; max-height: 95%;
              border: round ansi_default; padding: 1 2; background: ansi_default; }
    #dialog Static { height: auto; }
    #dialog-body { height: auto; }
    #dialog-title { text-style: bold; }
    #dialog-target { margin-bottom: 1; }
    #fields, #reservation-info { height: auto; }
    #dialog Input { height: 2; border: none; border-bottom: solid ansi_default;
                    padding: 0 1; margin-bottom: 0; }
    #dialog Input:focus { border-bottom: solid ansi_cyan; text-style: bold; }
    #dialog PlainCheckbox { width: 100%; height: 1; border: none; padding: 0; margin: 1 0; }
    #dialog PlainCheckbox:focus { border: none; text-style: bold underline; }
    #dialog Horizontal { height: 1; margin-top: 1; }
    #dialog Button { height: 1; border: none !important; min-width: 0; padding: 0 1;
                     background: ansi_default !important; tint: transparent;
                     background-tint: transparent;
                     width: auto; margin-right: 2; text-style: underline; }
    #dialog Button:focus { text-style: bold underline; color: ansi_cyan; }
    #dialog-help { margin-top: 1; text-style: dim; }
    #dialog-error { color: ansi_red; }
    """
    BINDINGS: ClassVar = [("escape", "dismiss_dialog", "Back")]

    def __init__(self, action, target, *, client=None):
        super().__init__()
        self.operation = action
        self.target = dict(target)
        self.request_key = "workbench-" + uuid4().hex
        self.review_payload = None
        self.client = client
        self._impact_pending = False
        self._impact_checked_at = None
        self._impact_worker = None
        self._impact_generation = 0
        self._focus_confirm_when_ready = False

    def compose(self) -> ComposeResult:
        identity = (
            self.target.get("job_id")
            or self.target.get("reservation_id")
            or (self.target.get("ref") or self.target.get("site_id", ""))
        )
        if self.operation == "reserve":
            identity += " / " + self.target["uuid"]
        self.explanation = {
            "reserve": "Exclude this exact GPU for a duration; release its idle actor.",
            "unreserve": "Cancel this reservation. Already delivered preemption cannot be undone.",
            "extend": "Add time without releasing the GPU or changing the force setting.",
            "release": "Drain this AutoDL site, then shut it down after required I/O completes.",
            "register": "Inspect and install the site agent on an already-rented AutoDL instance. "
            "This does not rent or power on GPUs. Operator access required.",
            "unregister": "Remove this offline registration from active scheduling. "
            "This does NOT shut down the instance. Server safety checks apply.",
            "reload": "Validate and reload the configured scheduler policy using its safe "
            "activation rules. Does not restart services or force-stop jobs.",
        }[self.operation]
        with Vertical(id="dialog"):
            with VerticalScroll(id="dialog-body"):
                title = (
                    f"Reserve GPU {self.target.get('index', '?')} on {self.target['site_id']}"
                    if self.operation == "reserve"
                    else f"{'Extend' if self.operation == 'extend' else 'Cancel'} reservation · "
                    f"{self.target.get('site_id', '')} / GPU {self.target.get('gpu_index', '?')}"
                    if self.operation in {"extend", "unreserve"}
                    else self.operation.capitalize()
                )
                yield Static(Text(plain(title)), id="dialog-title")
                yield Static(Text(plain(identity), style="dim"), id="dialog-target")
                if self.operation in {"extend", "unreserve"} and self.target.get("expires_at"):
                    with Vertical(id="reservation-info"):
                        yield Static(Text(f"Current expiry: {date(self.target['expires_at'])}"))
                        yield Static(Text(f"Reason: {plain(self.target.get('reason'))}"))
                with Vertical(id="fields"):
                    if self.operation == "reserve":
                        yield Static("Duration  (30m / 12h / 1d / seconds)")
                        yield Input("30m", placeholder="30m", id="seconds", select_on_focus=True)
                        yield Static("Reason  (optional label)")
                        yield Input("Manual reservation", id="reason", select_on_focus=True)
                    if self.operation == "extend":
                        yield Static("Add time  (30m / 1h / 12h / seconds)")
                        yield Input("30m", placeholder="30m", id="seconds", select_on_focus=True)
                    if self.operation in {"reserve", "release"}:
                        yield PlainCheckbox("Evict running jobs (force)", id="force")
                    if self.operation == "register":
                        yield Static("SSH connection command")
                        yield Input(placeholder="ssh -p PORT root@HOST.seetacloud.com", id="ssh")
                        yield Static("Password (hidden)")
                        yield Input(password=True, id="password")
                yield Static("", id="review", markup=False)
                yield Static("", id="impact", markup=False)
                yield Static("", id="dialog-error")
            with Horizontal():
                yield FormButton("Cancel", id="back")
                yield FormButton("Review →", id="apply")
            yield Static(
                "↑/↓ or Tab move · Space toggle · Enter select · Esc cancel",
                id="dialog-help",
            )

    def on_mount(self):
        # The scroll container is not a form control; keep it out of the Tab order.
        self.query_one("#dialog-body", VerticalScroll).can_focus = False
        self.fit_body()
        self.query_one("#review", Static).display = False
        self.query_one("#impact", Static).display = False
        self.query_one("#dialog-error", Static).display = False
        fields = list(self.query(Input))
        (fields[0] if fields else self.query_one("#apply", Button)).focus()

    def fit_body(self):
        # Keep the action buttons visible even when the job-impact list is long.
        self.query_one("#dialog-body", VerticalScroll).styles.max_height = max(
            3, int(self.app.size.height * 0.95) - 9
        )

    def on_resize(self):
        if self.is_mounted:
            self.fit_body()

    def action_dismiss_dialog(self):
        self._focus_confirm_when_ready = False
        if self.review_payload is not None:
            self._impact_generation += 1
            if self._impact_worker:
                self._impact_worker.cancel()
            self._impact_pending = False
            self.query_one("#dialog-body", VerticalScroll).can_focus = False
            self.query_one("#apply", Button).disabled = False
            self.query_one("#impact", Static).display = False
            self.review_payload = None
            self.query_one("#fields", Vertical).display = True
            self.query_one("#review", Static).display = False
            self.query_one("#dialog-target", Static).display = True
            for info in self.query("#reservation-info"):
                info.display = True
            self.query_one("#back", Button).label = "Cancel"
            self.query_one("#apply", Button).label = "Review →"
            self.query_one("#dialog-help", Static).update(
                "↑/↓ or Tab move · Space toggle · Enter select · Esc cancel"
            )
            fields = list(self.query(Input))
            (fields[0] if fields else self.query_one("#apply", Button)).focus()
        else:
            self.clear_password()
            self.dismiss(None)

    def clear_password(self):
        for field in self.query("#password"):
            field.value = ""

    def on_key(self, event):
        if event.key in {"pageup", "pagedown"}:
            dialog = self.query_one("#dialog-body", VerticalScroll)
            scroll = dialog.scroll_page_up if event.key == "pageup" else dialog.scroll_page_down
            scroll(animate=False)
            event.stop()
            event.prevent_default()
            return
        if event.key in {"up", "down"}:
            move = self.focus_next if event.key == "down" else self.focus_previous
            move("Input, PlainCheckbox, Button")
            event.stop()
            event.prevent_default()
            return
        # Text fields retain their normal cursor movement. Arrows only move
        # between the two footer buttons when a button already has focus.
        if isinstance(self.focused, Button) and event.key in {"left", "right"}:
            self.query_one("#apply" if self.focused.id == "back" else "#back", Button).focus()
            event.stop()
            event.prevent_default()

    @on(Input.Submitted)
    def next_field(self, event):
        event.stop()
        if self.review_payload is None:
            fields = list(self.query(Input))
            index = fields.index(event.input)
            if index + 1 < len(fields):
                fields[index + 1].focus()
            elif self.operation in {"reserve", "release"}:
                self.query_one("#force", Checkbox).focus()
            else:
                self.query_one("#apply", Button).focus()

    @on(Button.Pressed)
    def pressed(self, event):
        if event.button.id == "back":
            self.action_dismiss_dialog()
            return
        if event.button.id == "apply":
            self.action_advance()

    def action_advance(self):
        if self._impact_pending:
            return
        if self.review_payload is not None:
            if (
                self.operation == "reserve"
                and self.review_payload[3].get("force")
                and self._impact_checked_at is not None
                and time.monotonic() - self._impact_checked_at > 15
            ):
                self.start_impact_check()
                return
            payload = self.review_payload
            self.review_payload = None
            self.clear_password()
            self.dismiss(payload)
            return
        values = {
            w.id: w.value if w.id == "password" else w.value.strip() for w in self.query(Input)
        }
        flags = {w.id: w.value for w in self.query(Checkbox)}
        try:
            if self.operation == "reserve":
                request = GPUReservationRequest(
                    site_id=self.target["site_id"],
                    gpu_uuid=self.target["uuid"],
                    duration_seconds=duration_seconds(values["seconds"]),
                    reason=values["reason"] or "Manual reservation",
                    force=flags["force"],
                )
                values = request.model_dump(mode="json")
            elif self.operation == "extend":
                values = GPUReservationExtensionRequest(
                    expected_expires_at=self.target["expires_at"],
                    additional_seconds=duration_seconds(values["seconds"]),
                ).model_dump(mode="json")
            elif self.operation == "register":
                from topodrome.deployment.manual_autodl import parse_ssh

                parse_ssh(values["ssh"])
                if not values["password"]:
                    raise ValueError("password required")
        except (ValueError, KeyError):
            self.query_one("#dialog-error", Static).display = True
            self.query_one("#dialog-error", Static).update(
                "Check required fields and duration, or SSH format: "
                "ssh -p PORT root@HOST.seetacloud.com."
            )
            return
        self.review_payload = (self.operation, self.target, values, flags, self.request_key)
        review = Text()
        if self.operation != "extend":
            review.append("Review before applying\n", style="bold")
        if self.operation not in {"reserve", "extend"}:
            review.append(self.explanation + "\n", style="")
        if self.operation == "reserve":
            review.append(
                f"Duration: {values['duration_seconds']} seconds\n"
                f"Reason: {plain(values['reason'])}\n",
                style="",
            )
        if self.operation == "register":
            review.append(f"\nSSH: {plain(values['ssh'])}\nPassword: hidden\n", style="")
        if self.operation == "extend":
            end = datetime.fromisoformat(values["expected_expires_at"].replace("Z", "+00:00"))
            end += timedelta(seconds=values["additional_seconds"])
            review.append(
                f"New expiry: {date(end.isoformat())}\n",
                style="cyan",
            )
            review.append(
                f"Add: {values['additional_seconds']} seconds\n"
                f"Was: {date(self.target['expires_at'])}\n"
                f"{self.explanation}\n"
                f"Reservation: {plain(self.target['reservation_id'])}\n"
                f"Reason: {plain(self.target.get('reason'))}\n"
                f"Force mode: {'on' if self.target.get('force') else 'off'} (unchanged)\n"
            )
            self.query_one("#dialog-target", Static).display = False
            self.query_one("#reservation-info", Vertical).display = False
        if flags.get("force"):
            review.append(
                "FORCED EVICTION: jobs retry; work since the last checkpoint may be lost. "
                "Interactive kernels may stop.\n",
                style="yellow",
            )
        elif self.operation in {"reserve", "release"}:
            review.append(
                "\nNo forced eviction. Running jobs are not interrupted.\n", style="green"
            )
        self.query_one("#fields", Vertical).display = False
        self.query_one("#dialog-error", Static).display = False
        self.query_one("#review", Static).update(review)
        self.query_one("#review", Static).display = True
        self.query_one("#dialog-body", VerticalScroll).scroll_home(animate=False)
        self.query_one("#back", Button).label = "← Edit"
        self.query_one("#apply", Button).label = {
            "reserve": "Confirm reservation",
            "unreserve": "Confirm unreserve",
            "extend": "Confirm extension",
            "release": "Confirm release",
            "unregister": "Confirm removal",
            "register": "Confirm registration",
            "reload": "Confirm reload",
        }[self.operation]
        self.query_one("#dialog-help", Static).update("Enter confirm · ←/→ choose · Esc edit")
        if self.operation == "extend":
            self.query_one("#dialog-help", Static).update(
                "Enter confirm · ←/→ choose · Esc edit\nPgUp/PgDn scroll"
            )
        if self.operation == "reserve" and flags.get("force"):
            self.query_one("#dialog-help", Static).update(
                "Enter confirm · ←/→ choose · PgUp/PgDn scroll · Esc edit"
            )
            self.start_impact_check()
        else:
            self.query_one("#apply", Button).focus()

    def start_impact_check(self):
        self._impact_generation += 1
        self._impact_pending = True
        self._focus_confirm_when_ready = True
        self.query_one("#apply", Button).disabled = True
        # Keep Enter inert during the asynchronous check. Do not default to
        # Edit merely because Confirm is temporarily unavailable.
        body = self.query_one("#dialog-body", VerticalScroll)
        body.can_focus = True
        self.set_focus(body)
        self.query_one("#impact", Static).display = True
        self.query_one("#impact", Static).update("Checking which jobs use this GPU…")
        self._impact_worker = self.load_impact(self._impact_generation)

    @work(group="impact", exclusive=True, exit_on_error=False)
    async def load_impact(self, generation):
        try:
            if self.client is None:
                raise RuntimeError("client unavailable")
            message = await asyncio.wait_for(
                reservation_impact(self.client, self.target), timeout=12
            )
        except Exception as exc:
            message = Text(
                f"Job impact unavailable ({safe_error(exc)}). This is not an idle guarantee. "
                "Forcing may interrupt jobs or kernels that could not be listed.",
                style="yellow",
            )
        if self.review_payload is not None and generation == self._impact_generation:
            self.query_one("#impact", Static).update(message)
            self._impact_pending = False
            self._impact_checked_at = time.monotonic()
            self.query_one("#apply", Button).disabled = False
            if self._focus_confirm_when_ready:
                self._focus_confirm_when_ready = False
                # Apply synchronously so a queued focus callback cannot undo
                # the user's next navigation after the check completes.
                self.set_focus(self.query_one("#apply", Button))


class Workbench(App):
    TITLE = "Topodrome"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { background: ansi_default; color: ansi_default; }
    * { background: ansi_default; color: ansi_default; }
    #status { height: 2; padding: 0 1; }
    #page-header { height: 2; padding: 0 1; border-bottom: solid ansi_default; }
    #page-heading { width: 1fr; height: 1; }
    #page-register { height: 1; min-width: 0; width: auto; border: none;
                     padding: 0 1; text-style: bold underline; color: ansi_cyan; }
    #selection { height: 1; padding: 0 1; }
    .empty-state { height: auto; padding: 2 3; }
    #notice { height: auto; max-height: 3; color: ansi_yellow; padding: 0 1; }
    #filter { height: 3; display: none; }
    .searching #filter { display: block; }
    #body { height: 1fr; }
    #views { width: 100%; height: 1fr; }
    #inspector { width: 100%; height: 1fr; display: none; }
    .details-open #inspector { display: block; }
    .details-open #views { display: none; }
    #detail-title { height: auto; max-height: 3; padding: 0 1; }
    #detail-context { height: auto; max-height: 3; padding: 0 1; margin-bottom: 1; }
    #detail-actions-title { height: 1; padding: 0 1; text-style: bold; }
    #detail-actions { height: auto; max-height: 3; border: none; padding: 0 1; }
    #detail-actions > .option-list--option-highlighted {
        text-style: bold underline; color: ansi_cyan;
    }
    #detail-tabs { height: 1fr; }
    TabPane { padding: 0; }
    Tabs { height: 2; padding-top: 1; }
    Tab { padding: 0 1; }
    Tab.-active { text-style: bold underline; color: ansi_cyan; }
    Underline { display: none; }
    DataTable { height: 1fr; }
    DataTable > .datatable--header { text-style: bold; }
    DataTable > .datatable--cursor { text-style: bold underline; }
    .detail-text { height: auto; padding: 1; }
    #actions { height: 1; padding: 0 1; }
    #actions Button { height: 1; min-width: 0; width: auto; border: none;
                       padding: 0 1; text-style: underline; }
    #hints { height: 1; padding: 0 1; }
    .narrow #body { layout: vertical; }
    .narrow #views { width: 100%; height: 1fr; }
    .narrow #inspector { width: 100%; height: 1fr; }
    .details-hidden #inspector { display: none; }
    .short #inspector { display: none; }
    .short.details-open #inspector { display: block; height: 1fr; }
    .short.details-open #views { display: none; }
    .short #filter { display: none; }
    .short.searching #filter { display: block; }
    .short #status { height: 1; }
    .short #page-header { height: 1; border: none; }
    .short Tabs { height: 1; padding-top: 0; }
    .short #selection { display: none; }
    """
    BINDINGS: ClassVar = [
        Binding("left", "adjacent_tab(-1)", "Previous tab", priority=True),
        Binding("right", "adjacent_tab(1)", "Next tab", priority=True),
        Binding("shift+left", "pan_content(-1)", "Scroll left"),
        Binding("shift+right", "pan_content(1)", "Scroll right"),
        Binding("shift+up", "move_site(-1)", "Move site up", priority=True),
        Binding("shift+down", "move_site(1)", "Move site down", priority=True),
        Binding("up", "move_picked_site(-1)", "Move picked site up", priority=True),
        Binding("down", "move_picked_site(1)", "Move picked site down", priority=True),
        Binding("enter", "save_site_order", "Save site order", priority=True),
        ("q", "safe_quit", "Quit"),
        ("ctrl+c", "safe_quit", "Quit"),
        ("ctrl+q", "safe_quit", "Quit"),
        ("r", "refresh_now", "Refresh"),
        ("slash", "search", "Filter"),
        ("h", "history", "History"),
        ("s", "sort_jobs", "Sort"),
        ("d", "details", "Details"),
        ("a", "context_action", "Action"),
        ("n", "register_instance", "Register AutoDL"),
        Binding("space", "toggle_expansion", "Expand / collapse / reorder", priority=True),
        ("escape", "back", "Back"),
        ("question_mark", "help", "Help"),
        *((str(i), f"view('{view}')", view.title()) for i, view in enumerate(VIEWS, 1)),
    ]

    def __init__(self, client=None, *, interval=2.0, no_color=False):
        super().__init__(ansi_color=True)
        if not math.isfinite(interval) or interval < 0.1:
            raise ValueError("refresh interval must be finite and >= 0.1")
        self.client = client or AsyncClient(timeout=10)
        self.interval = interval
        self.snapshot = {}
        self.asset_rows = []
        self.reservation_rows = []
        self.autodl_instances = []
        self.system_info = None
        self.system_error = None
        self.expanded_gpus = set()
        self._reservations_checked_at = None
        self._reservations_failed = False
        self._mutation_version = 0
        self.records = {v: {} for v in VIEWS}
        self._detail_records = {v: {} for v in VIEWS}
        self.selections = {}
        self.show_history = False
        self.expanded_assets = set()
        self.asset_history = []
        self.asset_history_state = "loading"
        self.asset_history_limited = False
        self.job_sort = "queue"
        self.last_success = None
        self.failure = None
        self._refresh_busy = False
        self._extra_busy = False
        self._action_busy = False
        self._site_order = None
        self._detail_worker = None
        self._logs_worker = None
        self._log_generation = 0
        self._table_cache = {}
        self._row_labels = {}
        self._column_signatures = {}
        self._detail_identity = None
        self._opened_detail = None
        self._detail_actions = ()
        self._detail_action_target = None
        self._focus_return = None
        self._navigation_generation = 0
        self._pending_focus = None
        self._empty_focus = None
        self._notice_text = ""
        self._report_busy = False
        self.expanded_arrays = set()
        self.array_cache = {}
        self.array_errors = {}
        self._array_busy = set()
        self._array_semaphore = asyncio.Semaphore(2)
        self.no_color = no_color
        self._filters.append(TerminalDefaults(no_color=no_color))

    def compose(self) -> ComposeResult:
        yield Static("Connecting…", id="status", markup=False)
        yield Static("", id="notice", markup=False)
        with Horizontal(id="page-header"):
            yield Static("Jobs", id="page-heading", markup=False)
            yield Button("n Register AutoDL", id="page-register")
        yield Input(
            placeholder="Filter visible rows by ID, name, site, model or state  (/)", id="filter"
        )
        with Horizontal(id="body"):
            with WorkbenchTabs(id="views"):
                for number, view in enumerate(VIEWS, 1):
                    with TabPane(f"{number} {VIEW_LABELS.get(view, view.title())}", id=view):
                        yield Static("", id=f"empty-{view}", classes="empty-state", markup=False)
                        yield DataTable(
                            id=f"table-{view}",
                            cursor_type="row",
                            zebra_stripes=False,
                            cursor_foreground_priority="renderable",
                        )
            with Vertical(id="inspector"):
                yield Static("Select a row", id="detail-title", markup=False)
                yield Static("", id="detail-context", markup=False)
                yield Static("Actions", id="detail-actions-title")
                yield DetailActions(id="detail-actions")
                with WorkbenchTabs(id="detail-tabs"):
                    for number, name in enumerate(("summary", "attempts", "logs"), 1):
                        with TabPane(f"{number} {name.title()}", id=f"pane-{name}"):
                            if name == "logs":
                                yield JobLogs(interval=self.interval)
                            else:
                                with VerticalScroll(id=f"scroll-{name}"):
                                    yield Static(
                                        "", id=f"detail-{name}", classes="detail-text", markup=False
                                    )
        yield Static("", id="selection", markup=False)
        with Horizontal(id="actions"):
            yield FormButton("↑ Up", id="order-up")
            yield FormButton("↓ Down", id="order-down")
            yield FormButton("Auto", id="order-automatic")
            yield FormButton("Save", id="order-save")
            yield FormButton("Cancel", id="order-cancel")
            yield Button("GPU reservations", id="manage-reservations")
            yield Button("Models", id="page-models")
            yield Button("Reload policy", id="page-reload")
            for name, label in (
                ("reserve", "Reserve GPU"),
                ("extend", "Extend reservation"),
                ("unreserve", "Unreserve"),
                ("release", "Release"),
                ("unregister", "Remove offline"),
            ):
                yield Button(label, id=f"action-{name}", disabled=True)
        yield Static("←→ tabs · ↑↓ select · Enter details · / filter · q quit", id="hints")

    def widget(self, selector, kind):
        return self._main.query_one(selector, kind)

    def on_mount(self):
        self._main = self.screen
        self._main.set_class(self.size.width < 100, "narrow")
        self._main.set_class(self.size.height < 22, "short")
        self.theme = "textual-dark"
        self.screen_change_signal.subscribe(self, self.screen_changed)
        self.refresh_status()
        self.set_interval(self.interval, self.tick)
        self.set_interval(1, self.update_status)
        self.focus_primary()

    async def on_unmount(self):
        await self.client.close()

    def screen_changed(self, screen):
        if screen is self._main:
            # The previously focused row/action can disappear while a modal
            # is open. Check it again after Textual resumes the main screen.
            self.call_after_refresh(self.repair_focus)

    def on_resize(self, event):
        if hasattr(self, "_main"):
            self._main.set_class(event.size.width < 100, "narrow")
            self._main.set_class(event.size.height < 22, "short")
            if self._table_cache:
                self.render_tables()

    def tick(self):
        self.refresh_status()
        self.schedule_detail()
        if self.active_view in {"assets", "reservations", "gpus", "autodl", "system"}:
            self.refresh_extras()

    @property
    def active_view(self):
        return self.widget("#views", TabbedContent).active

    @work(group="status", exit_on_error=False)
    async def refresh_status(self):
        if self._refresh_busy:
            return
        self._refresh_busy = True
        version = self._mutation_version
        try:
            snapshot = visible_snapshot(await self.client.system_status())
            if version != self._mutation_version:
                return
            self.snapshot = snapshot
            self.last_success = time.monotonic()
            self.failure = None
            self.render_tables()
            self.refresh_expanded_arrays()
        except Exception as exc:
            if version == self._mutation_version:
                self.failure = safe_error(exc)
                if self.is_running:
                    self.update_actions()
        finally:
            self._refresh_busy = False
            self.update_status()
            if self.is_running and version != self._mutation_version:
                self.refresh_status()

    def refresh_expanded_arrays(self):
        visible = {r["array_id"] for r in self.snapshot.get("arrays", [])}
        self.expanded_arrays.intersection_update(visible)
        self.array_cache = {k: v for k, v in self.array_cache.items() if k in visible}
        self.array_errors = {k: v for k, v in self.array_errors.items() if k in visible}
        if self.active_view == "jobs":
            for identity in self.expanded_arrays:
                if identity not in self._array_busy:
                    self.refresh_array(identity)

    @work(group="array-members", exit_on_error=False)
    async def refresh_array(self, identity):
        if identity in self._array_busy:
            return
        self._array_busy.add(identity)
        try:
            async with self._array_semaphore:
                if identity not in self.expanded_arrays:
                    return
                detail = await self.client.array(identity, limit=1000)
            if identity in self.expanded_arrays:
                self.array_cache[identity] = detail
                self.array_errors.pop(identity, None)
        except Exception as exc:
            if identity in self.expanded_arrays:
                self.array_errors[identity] = safe_error(exc)
        finally:
            self._array_busy.discard(identity)
            # Never change focus, reopen a collapsed group or mark all sites stale.
            if self.is_running:
                self.render_tables()

    @work(group="extras", exit_on_error=False)
    async def refresh_extras(self):
        if self._extra_busy:
            return
        self._extra_busy = True
        view = self.active_view
        version = self._mutation_version
        try:
            if view == "assets":
                rows = await self.client.assets(limit=100, all_versions=False)
                if version != self._mutation_version:
                    return
                self.asset_rows = [mapping(a) for a in rows]
                self.expanded_assets.intersection_update(
                    asset_group(row) for row in self.asset_rows
                )
                self.render_tables()
                if self.expanded_assets:
                    try:
                        rows = await self.client.assets(limit=1000, all_versions=True)
                        if version != self._mutation_version:
                            return
                        self.asset_history = [mapping(a) for a in rows]
                        self.asset_history_limited = len(rows) >= 1000
                        self.asset_history_state = "ready"
                    except Exception:
                        self.asset_history_state = "error"
            elif view == "system":
                self.system_info = await self.client.system_metrics()
                self.system_error = None
            elif view == "autodl":
                rows = await self.client.autodl_instances()
                if version != self._mutation_version:
                    return
                self.autodl_instances = [mapping(row) for row in rows]
            elif view in {"reservations", "gpus"}:
                rows = [
                    mapping(r)
                    for r in await self.client.gpu_reservations(limit=1000, active_only=True)
                ]
                if version != self._mutation_version:
                    return
                self.reservation_rows = rows
                self._reservations_checked_at = time.monotonic()
                self._reservations_failed = False
            self.render_tables()
            if self.active_view == view and self._notice_text.startswith(
                f"{view.title()} unavailable"
            ):
                self.set_notice("")
        except Exception as exc:
            if version != self._mutation_version:
                return
            if view == "system":
                self.system_error = safe_error(exc)
                self.render_tables()
            if view in {"gpus", "reservations"}:
                self._reservations_failed = True
                self.update_actions()
            self.set_notice(
                f"{view.title()} unavailable ({safe_error(exc)}); "
                "previous rows may be stale. Management information requires operator access."
            )
        finally:
            self._extra_busy = False
            if (
                self.is_running
                and (version != self._mutation_version or view != self.active_view)
                and self.active_view in {"assets", "gpus", "reservations", "autodl", "system"}
            ):
                self.refresh_extras()

    def update_status(self):
        # A queued timer can fire after the screen's children are unmounted.
        try:
            status = self.widget("#status", Static)
        except NoMatches:
            return
        elapsed = int(time.monotonic() - self.last_success) if self.last_success else None
        health = (
            (f"STALE / {self.failure}" if self.failure else "LIVE")
            if elapsed is not None
            else "CONNECTING"
        )
        title = Text("TOPO", style="bold")
        title.append("DROME", style="bold cyan")
        title.append("  /  workbench  ", style="dim")
        title.append("● " + health, style="yellow" if self.failure or elapsed is None else "green")
        title.append(f"  {elapsed if elapsed is not None else '-'}s ago", style="dim")
        if self.size.height >= 22:
            jobs = self.snapshot.get("jobs", [])
            running = sum(j.get("state") in {"running", "starting", "finalizing"} for j in jobs)
            queued = sum(j.get("state") in {"queued", "retry_wait"} for j in jobs)
            ready = sum(s.get("state") == "ready" for s in self.snapshot.get("sites", []))
            title.append(
                f"\n{ready}/{len(self.snapshot.get('sites', []))} sites ready  ·  "
                f"{len(self.snapshot.get('gpus', []))} GPUs  ·  "
                f"{running} active  ·  {queued} queued",
                style="dim",
            )
        status.update(title)

    def set_notice(self, message):
        self._notice_text = message
        self.widget("#notice", Static).update(Text(message))

    def render_tables(self):
        focused_table = self.focused if self.screen is self._main else None
        query = self.widget("#filter", Input).value.casefold()
        jobs, known_jobs = job_tree(
            self.snapshot,
            self.expanded_arrays,
            self.array_cache,
            self.array_errors,
            history=self.show_history,
            sort=self.job_sort,
            query=query,
        )
        # Queue mode preserves the scheduler's aged/priority/runtime order;
        # an array occupies the position of its first visible child.
        policy = self.snapshot.get("policy") or {}
        draft = self._site_order
        preference = list(draft.preferences if draft else policy.get("preferred_site_ids", []))
        site_rows = self.snapshot.get("sites", [])
        if draft:
            site_rows = draft.rows(site_rows)
        sites = [
            {
                **site,
                "preference_rank": (
                    preference.index(site["site_id"]) + 1 if site["site_id"] in preference else None
                ),
            }
            for site in site_rows
        ]
        if not draft:
            sites.sort(key=lambda row: row["preference_rank"] or len(preference) + 1)
        gpus = self.snapshot.get("gpus", [])
        arrays = [
            array_row(row, self.expanded_arrays, self.array_cache, self.array_errors)
            for row in self.snapshot.get("arrays", [])
        ]
        self.expanded_gpus.intersection_update(gpu_key(row) for row in gpus)
        tables = {
            "system": (
                system_rows(self.system_info, self.system_error),
                lambda r: r["metric_id"],
                ("Metric", "Value", "Notes"),
                lambda r: (r["name"], r["value"], r["note"]),
            ),
            "jobs": (
                jobs,
                lambda r: r["job_id"],
                ("ID", "Name", "State", "Site", "Wait / Run / I/O", "Progress / ETA"),
                lambda r: (
                    r["job_id"].removeprefix("job_").removeprefix("array_")[:4],
                    (
                        f"  [{r['array_index']}] {r.get('name') or 'Element'}"
                        if r.get("array_id")
                        else r.get("name")
                    ),
                    r.get("state"),
                    r.get("site_id"),
                    timing_cell(r),
                    _progress_text(r.get("progress"), width=60),
                ),
            ),
            "gpus": (
                gpu_tree(gpus, self.expanded_gpus, query),
                gpu_key,
                ("Site", "GPU", "Model", "Used / total GiB", "Util", "Owner", "Health"),
                gpu_cells,
            ),
            "sites": (
                sites,
                lambda r: r["site_id"],
                ("Site", "Pref", "Kind", "State", "Heartbeat age", "Queue", "Storage"),
                lambda r: (
                    ("↕ " if draft and draft.grabbed == r["site_id"] else "") + r["site_id"],
                    f"#{r['preference_rank']}" if r.get("preference_rank") else "—",
                    r.get("kind"),
                    r.get("state"),
                    age(r.get("heartbeat_at")),
                    r.get("queue_depth"),
                    "ok" if r.get("storage_available") else "unavailable",
                ),
            ),
            "assets": (
                asset_tree(
                    self.asset_rows,
                    self.asset_history,
                    self.expanded_assets,
                    query=query,
                    state=self.asset_history_state,
                    limited=self.asset_history_limited,
                ),
                lambda r: r["ref"],
                ("Name", "Version", "Visibility", "GiB", "Files", "Tags"),
                asset_cells,
            ),
            "reservations": (
                self.reservation_rows,
                lambda r: r["reservation_id"],
                ("Site", "GPU", "State", "Expires", "Reason"),
                lambda r: (r["site_id"], r["gpu_index"], r["state"], r["expires_at"], r["reason"]),
            ),
            "autodl": (
                autodl_rows(sites, self.autodl_instances),
                lambda r: r["site_id"],
                ("Site", "State", "GPUs", "Agent", "Region", "Queue"),
                lambda r: (
                    r["site_id"],
                    r["state"],
                    r.get("gpu_count"),
                    r.get("site_state", r["state"]),
                    r.get("region"),
                    r.get("queue_depth"),
                ),
            ),
        }
        for view, (rows, key, columns, cells) in tables.items():
            # Open details outlive list filters, including a job becoming terminal.
            detail_rows = (
                [*known_jobs.values(), *arrays, *jobs]
                if view == "jobs"
                else [*gpus, *rows]
                if view == "gpus"
                else [*self.asset_history, *self.asset_rows, *rows]
                if view == "assets"
                else rows
            )
            self._detail_records[view] = {key(row): row for row in detail_rows}
            if view not in {"jobs", "gpus", "assets"} and not (view == "sites" and draft):
                rows = [r for r in rows if not query or query in str(r).casefold()]
            records = {key(r): r for r in rows}
            old_record = self.records[view].get(self.selections.get(view), {})
            self.records[view] = records
            table = self.widget(f"#table-{view}", DataTable)
            labels = {key(row): self.disclosure_label(view, row) for row in rows}
            table.show_row_labels = any(label is not None for label in labels.values())
            layouts = (
                job_columns(self.size.width - (3 if table.show_row_labels else 0))
                if view == "jobs"
                else [
                    (
                        label,
                        {
                            "ID": 4,
                            "Name": 20,
                            "State": 12,
                            "Site": 17,
                            "Progress / ETA": 32,
                            "Wait / Run / I/O": 22,
                            "Model": 24,
                            "Reason": 30,
                            "Tags": 28,
                            "Metric": 26,
                            "Value": 22,
                            "Notes": max(18, self.size.width - 58),
                        }.get(label),
                    )
                    for label in columns
                ]
            )
            indices = [columns.index(label) for label, _ in layouts]
            columns = tuple(label for label, _ in layouts)
            if self._column_signatures.get(view) != layouts:
                table.clear(columns=True)
                for label, width in layouts:
                    table.add_column(label, width=width)
                self._column_signatures[view] = layouts
                self._table_cache.pop(view, None)
            desired = [
                (key(r), tuple(plain(cells(r)[index], limit=120) for index in indices))
                for r in rows
            ]
            previous = self._table_cache.get(view, [])
            old_selection = self.selections.get(view)
            scroll = (table.scroll_x, table.scroll_y)
            if [r[0] for r in desired] == [
                r[0] for r in previous
            ] and labels == self._row_labels.get(view):
                for (identity, values), (_, old_values) in zip(desired, previous, strict=True):
                    for col, label, value, old in zip(
                        table.columns, columns, values, old_values, strict=True
                    ):
                        if value != old:
                            table.update_cell(
                                identity, col, status_cell(value, label), update_width=True
                            )
            else:
                table.clear()
                for identity, values in desired:
                    table.add_row(
                        *(status_cell(v, c) for v, c in zip(values, columns, strict=True)),
                        key=identity,
                        label=labels[identity],
                    )
                ids = list(records)
                selected = old_selection if old_selection in records else (ids[0] if ids else None)
                if old_selection not in records and old_record.get("array_id") in records:
                    selected = old_record["array_id"]
                if old_selection not in records and old_record.get("_gpu_parent") in records:
                    selected = old_record["_gpu_parent"]
                if old_selection not in records and old_record.get("_asset_parent") in records:
                    selected = old_record["_asset_parent"]
                self.selections[view] = selected
                if selected:
                    table.move_cursor(row=ids.index(selected), animate=False, scroll=False)
                table.scroll_to(x=scroll[0], y=scroll[1], animate=False)
            self._table_cache[view] = desired
            self._row_labels[view] = labels
            empty = self.widget(f"#empty-{view}", Static)
            empty.display = not bool(desired)
            table.display = bool(desired)
            if not desired:
                message = {
                    "jobs": "The queue is clear.\nAI agents submit work through the CLI / SDK. "
                    "Press h for recent history.",
                    "gpus": "No online GPUs to show.\nOffline sites remain visible under 3 Sites.",
                    "sites": "No sites reported yet.\nPress r to refresh the connection.",
                    "assets": "No assets listed.\n"
                    "AI agents manage reusable inputs through the CLI / SDK.",
                    "reservations": "No active reservations.\n"
                    "Select a GPU under 2 GPUs, then Enter for details and actions.",
                    "autodl": "No registered AutoDL sites.\n"
                    "Press n or choose Register AutoDL above.\n"
                    "Paste its SSH command and password; review before installing the agent.",
                    "system": "No matching system metrics.",
                }[view]
                if query:
                    message = "No matching rows.\nPress Esc to clear the filter."
                elif view == "jobs" and self.show_history:
                    message = (
                        "No jobs in the recent snapshot.\n"
                        "AI agents submit work through the CLI / SDK."
                    )
                empty.update(Text(message))
        if isinstance(focused_table, DataTable) and not focused_table.display:
            # Hiding the last row may let Textual move focus to a different
            # (valid) control. Remember that this is an automatic fallback,
            # so new rows can take focus back unless the user has moved on.
            self.focus_primary()
        self.update_actions()
        self.schedule_detail()

    @on(Input.Changed, "#filter")
    def filter_changed(self):
        self.render_tables()

    @on(DataTable.RowHighlighted)
    def row_selected(self, event):
        # Empty tables (and transient clears during refresh) have no row key.
        # Ignore these queued events rather than clearing a newer selection.
        view = (event.data_table.id or "").removeprefix("table-")
        key = event.row_key.value if event.row_key is not None else None
        if view in VIEWS and key in self.records[view]:
            self.selections[view] = key
            self.update_actions()
            self.schedule_detail()

    @on(DataTable.RowSelected)
    def row_opened(self, event):
        key = event.row_key.value if event.row_key is not None else None
        if (
            self.screen is self._main
            and not self._main.has_class("details-open")
            and event.data_table.id == f"table-{self.active_view}"
            and key in self.records[self.active_view]
        ):
            self.selections[self.active_view] = key
            if self.active_view == "sites" and self._site_order:
                # A mouse click selects another row while reordering. Only
                # explicit Save / Enter commits, never a row click.
                self._site_order.grabbed = None
                self.render_tables()
                self.focus_primary()
            else:
                self.show_details()

    @on(DataTable.RowLabelSelected)
    def disclosure_selected(self, event):
        view = (event.data_table.id or "").removeprefix("table-")
        key = event.row_key.value if event.row_key is not None else None
        row = self.records.get(view, {}).get(key)
        if (
            self.screen is self._main
            and self.active_view == view
            and row
            and self.disclosure_label(view, row) is not None
        ):
            self.selections[view] = key
            self.action_toggle_expansion()

    @on(Input.Submitted, "#filter")
    def filter_submitted(self):
        self.focus_primary()

    @on(TabbedContent.TabActivated, "#views")
    def tab_changed(self, event):
        if event.pane.id != self.active_view:
            return
        if self.active_view != "sites" and self._site_order and not self._action_busy:
            self._site_order = None
            self.set_notice("Unapplied site order discarded.")
            self.render_tables()
        self.update_actions()
        self.schedule_detail()
        if not self._main.has_class("details-open") and (
            isinstance(self.focused, Tabs) or not self.visible_focus(self.focused)
        ):
            self.focus_primary()
        if self.active_view in {"assets", "reservations", "gpus", "autodl", "system"}:
            self.refresh_extras()
        elif self.active_view == "jobs":
            self.refresh_expanded_arrays()

    @on(TabbedContent.TabActivated, "#detail-tabs")
    def detail_tab_changed(self, event):
        if (
            self._main.has_class("details-open")
            and event.pane.id == self.widget("#detail-tabs", TabbedContent).active
            and (isinstance(self.focused, Tabs) or not self.visible_focus(self.focused))
        ):
            self.focus_primary()

    def on_click(self, event):
        # Clicking an already-selected tab emits no TabActivated event. It
        # should still hand arrows to its content, not strand them on Tabs.
        if self.screen is not self._main or not isinstance(event.widget, Tab):
            return
        tabs = self.widget(
            "#detail-tabs" if self._main.has_class("details-open") else "#views",
            TabbedContent,
        )
        if event.widget is tabs.get_tab(tabs.active):
            self.focus_primary()

    def selected(self):
        view = self.active_view
        if self._main.has_class("details-open") and self._opened_detail:
            opened_view, key, cached = self._opened_detail
            if opened_view == view:
                row = self._detail_records[view].get(key)
                if row is not None:
                    self._opened_detail = (view, key, dict(row))
                    return row
                if view == "gpus":
                    return {**cached, "processes_status": "stale"}
                return cached
        return self.records[view].get(self.selections.get(view))

    def selected_reservation(self):
        row = self.selected()
        if not row:
            return None
        if self.active_view == "reservations":
            return row if reservation_active(row) else None
        if self.active_view == "gpus":
            return next(
                (
                    r
                    for r in self.reservation_rows
                    if r.get("site_id") == row["site_id"]
                    and r.get("gpu_uuid") == row["uuid"]
                    and reservation_active(r)
                ),
                None,
            )
        return None

    def reservations_fresh(self):
        return (
            not self._reservations_failed
            and self._reservations_checked_at is not None
            and time.monotonic() - self._reservations_checked_at < max(10, self.interval * 3)
        )

    def update_detail_context(self):
        row = self.selected()
        text = Text()
        if row and self.active_view in {"gpus", "reservations"}:
            reservation = self.selected_reservation()
            if reservation:
                text.append("Reserved · " + reservation_caption(reservation), style="yellow")
                text.append("\n" + plain(reservation.get("reason")), style="dim")
                if not self.reservations_fresh():
                    text.append(
                        "\nReservation info is stale; refresh before changing it.", style="yellow"
                    )
            elif row.get("operator_reserved"):
                text.append(
                    "Reserved · loading reservation details…"
                    if self._extra_busy and not self._reservations_failed
                    else "Reserved · details unavailable; press r to refresh (operator only).",
                    style="yellow",
                )
            elif self.active_view == "gpus":
                text.append("No Topodrome reservation", style="dim")
        context = self.widget("#detail-context", Static)
        context.display = bool(text.plain)
        context.update(text)

    def schedule_detail(self):
        view = self.active_view
        key = (
            self._opened_detail[1]
            if self._main.has_class("details-open") and self._opened_detail
            else self.selections.get(view)
        )
        identity = (view, key)
        changed = identity != self._detail_identity
        if changed:
            if self._detail_worker:
                self._detail_worker.cancel()
            if self._logs_worker:
                self._logs_worker.cancel()
            self._log_generation += 1
            self._detail_identity = identity
            for name in ("summary", "attempts"):
                self.widget(f"#detail-{name}", Static).update("")
                self.widget(f"#scroll-{name}", VerticalScroll).scroll_home(animate=False)
            self.widget("#job-logs", JobLogs).reset()
        row = self.selected()
        if (
            view == "jobs"
            and row
            and not row.get("is_array")
            and (changed or not self._logs_worker or self._logs_worker.is_finished)
        ):
            self._logs_worker = self.run_worker(
                partial(self.load_logs, identity, row["job_id"], self._log_generation),
                group="logs",
                exit_on_error=False,
            )
        if (
            not (row and row.get("is_array"))
            and not changed
            and self._detail_worker
            and not self._detail_worker.is_finished
        ):
            return
        title = (
            f"{plain(row['site_id'])} / GPU {plain(row.get('index'))}"
            if view == "gpus" and row
            else plain(key or "No matching rows")
        )
        if self._main.has_class("details-open") and key and key not in self._detail_records[view]:
            title += "\nNot in the latest snapshot; cached details only. Actions unavailable."
        self.widget("#detail-title", Static).update(Text(title, style="bold"))
        self.update_detail_context()
        if row and row.get("is_array"):
            # Aggregate fields already belong to this status snapshot. Render
            # synchronously so an older detail worker cannot delay a terminal
            # update after the parent disappears from the active-only list.
            self.widget("#detail-summary", Static).update(detail_page(view, row))
        elif row:
            self._detail_worker = self.run_worker(
                partial(self.load_detail, identity, dict(row)), group="detail", exit_on_error=False
            )

    async def load_logs(self, identity, job_id, generation):
        # Slow summary/attempt reads must not hold up live output. Neither the
        # global LIVE badge nor these reads attest to the freshness of logs.
        try:
            logs = await self.client.logs(job_id, tail_bytes=65536)
            if identity == self._detail_identity and generation == self._log_generation:
                self.widget("#job-logs", JobLogs).receive(logs)
        except Exception as exc:
            if identity == self._detail_identity and generation == self._log_generation:
                self.widget("#job-logs", JobLogs).failed(safe_error(exc))

    async def load_detail(self, identity, row):
        try:
            if identity[0] == "jobs" and not row.get("is_array"):
                summary, attempts = await asyncio.gather(
                    self.client.status(row["job_id"]),
                    self.client.attempts(row["job_id"]),
                    return_exceptions=True,
                )
                if identity != self._detail_identity:
                    return
                if isinstance(summary, Exception):
                    raise summary
                if isinstance(attempts, Exception):
                    raise attempts
                summary_text = detail_page("jobs", row, mapping(summary))
                attempt_text = attempts_page(attempts)
            else:
                summary_text = detail_page(identity[0], row)
                attempt_text = ""
            if identity != self._detail_identity:
                return
            for name, value in (
                ("summary", summary_text),
                ("attempts", attempt_text),
            ):
                self.widget(f"#detail-{name}", Static).update(
                    value
                    if isinstance(value, Text)
                    else Text(plain(value, multiline=True, limit=70000))
                )
        except Exception as exc:
            if identity == self._detail_identity:
                self.widget("#detail-title", Static).update(
                    Text(f"{plain(identity[1])}\nDetails stale/unavailable ({safe_error(exc)})")
                )

    def available_actions(self):
        row = self.selected()
        view = self.active_view
        if not row or (
            self._main.has_class("details-open")
            and self._opened_detail
            and self._opened_detail[1] not in self._detail_records[view]
        ):
            return set()
        if view in {"gpus", "reservations"}:
            if self.selected_reservation():
                return {"extend", "unreserve"} if self.reservations_fresh() else set()
            if view == "gpus" and not row.get("operator_reserved"):
                return {"reserve"}
            return set()
        if view == "sites":
            return {"site-order"} if self.site_order_available() else set()
        return {
            "autodl": {"release", "unregister"}
            if row.get("state") in {"offline", "stopped"}
            else {"release"},
        }.get(view, set())

    def site_order_available(self):
        policy = self.snapshot.get("policy") or {}
        return (
            not self.failure
            and self.last_success is not None
            and time.monotonic() - self.last_success <= max(5, self.interval * 3)
            and isinstance(policy.get("generation"), int)
            and "preferred_site_ids" in policy
            and not policy.get("pending_generation")
        )

    def site_order_save_available(self):
        draft = self._site_order
        policy = self.snapshot.get("policy") or {}
        registered = {row["site_id"] for row in self.snapshot.get("sites", [])}
        return (
            draft is not None
            and draft.changed
            and self.site_order_available()
            and draft.generation == policy.get("generation")
            and set(draft.preferences) <= registered
        )

    def begin_site_order(self):
        if self.screen is not self._main or self.active_view != "sites" or self._action_busy:
            return False
        if self._site_order is None:
            if not self.site_order_available():
                self.set_notice("Site order is unavailable or stale. Refresh before editing.")
                return False
            self._site_order = SiteOrderDraft.capture(
                self.snapshot["policy"], self.snapshot.get("sites", [])
            )
            self._main.remove_class("searching")
            self.widget("#filter", Input).value = ""
            self.set_notice("")
            self.render_tables()
        self.focus_primary()
        return True

    def action_move_site(self, direction):
        if not self.check_action("move_site", (direction,)) or not self.begin_site_order():
            return
        selected = self.selections.get("sites")
        if self._site_order.move(selected, direction):
            if self._site_order.grabbed is not None:
                self._site_order.grabbed = selected
            self.render_tables()
            table = self.widget("#table-sites", DataTable)
            table.move_cursor(row=list(self.records["sites"]).index(selected), animate=False)
        self.focus_primary()

    def action_pick_site(self):
        if not self.begin_site_order():
            return
        draft = self._site_order
        selected = self.selections.get("sites")
        if draft.grabbed is not None:
            draft.grabbed = None
        elif selected in draft.order:
            draft.grabbed = selected
        else:
            self.set_notice("Cancel and reorder again to include newly registered sites.")
        self.render_tables()
        self.focus_primary()

    def action_move_picked_site(self, direction):
        if not self.check_action("move_picked_site", (direction,)) or self._action_busy:
            return
        # A refresh may queue transient highlights; moving is bound to the
        # exact picked-up site, not a cursor position or a queued highlight.
        self.selections["sites"] = self._site_order.grabbed
        self.action_move_site(direction)

    def action_save_site_order(self):
        draft = self._site_order
        if draft is None or self._action_busy:
            return
        if not draft.changed:
            self.set_notice("No site order changes. Esc returns to browsing.")
            return
        if not self.site_order_save_available():
            self.set_notice(
                "Site order changed or is stale. Esc cancels; refresh and reorder again."
            )
            return
        try:
            request = SitePreferenceRequest(
                preferred_site_ids=draft.preferences, expected_generation=draft.generation
            )
        except ValueError:
            self.set_notice("Order must contain at most 64 unique registered sites.")
            return
        self.action_confirmed(
            (
                "site-order",
                {"site_id": "Scheduler placement"},
                request.model_dump(mode="json"),
                {},
                "workbench-" + uuid4().hex,
            ),
            origin=ActionFocus("sites", self.selections.get("sites"), False),
        )

    def cancel_site_order(self):
        if self._action_busy:
            self.set_notice("Site order is being saved; wait for its result.")
            return
        self._site_order = None
        self.set_notice("")
        self.render_tables()
        self.focus_primary()

    @on(Button.Pressed, "#order-up, #order-down, #order-automatic, #order-save, #order-cancel")
    def order_button_pressed(self, event):
        event.stop()
        if event.button.disabled or self._site_order is None or self._action_busy:
            return
        if event.button.id in {"order-up", "order-down"}:
            self.action_move_site(-1 if event.button.id == "order-up" else 1)
        elif event.button.id == "order-automatic":
            self._site_order.automatic()
            self.render_tables()
            self.focus_primary()
        elif event.button.id == "order-save":
            self.action_save_site_order()
        elif event.button.id == "order-cancel":
            self.cancel_site_order()

    def update_actions(self):
        row = self.selected()
        view = self.active_view
        valid = self.available_actions()
        details_open = self._main.has_class("details-open")
        ordering = view == "sites" and self._site_order is not None and not details_open
        self.widget("#actions", Horizontal).display = not details_open
        register = self.widget("#page-register", Button)
        register.display = not details_open and view in {"sites", "autodl"} and not ordering
        register.disabled = self._action_busy
        actions = tuple(operation for operation in ACTION_LABELS if operation in valid)
        options = self.widget("#detail-actions", OptionList)
        reservation = self.selected_reservation()
        key = (
            self._opened_detail[1]
            if details_open and self._opened_detail
            else self.selections.get(view)
        )
        target_identity = (view, key, (reservation or {}).get("reservation_id"))
        if (
            not actions
            and options.highlighted is not None
            and self.focused is options
            and details_open
            and not self._action_busy
            and self._focus_return is None
            and self._detail_action_target
            and target_identity[:2] == self._detail_action_target[:2]
        ):
            # A temporary stale/failed read must not forget the selected action.
            # Its exact reservation must still match when data becomes usable.
            self._focus_return = ActionFocus(
                view,
                key,
                True,
                options.get_option_at_index(options.highlighted).id,
                self._detail_action_target[2],
                view in {"gpus", "reservations"},
            )
        if actions != self._detail_actions or target_identity != self._detail_action_target:
            highlighted = (
                options.get_option_at_index(options.highlighted).id
                if options.highlighted is not None and target_identity == self._detail_action_target
                else None
            )
            options.clear_options()
            options.add_options(Option(Text(ACTION_LABELS[op]), id=op) for op in actions)
            options.highlighted = actions.index(highlighted) if highlighted in actions else None
            self._detail_actions = actions
            self._detail_action_target = target_identity
        options.display = bool(actions)
        options.disabled = self._action_busy
        self.widget("#detail-actions-title", Static).display = bool(actions)
        self.update_detail_context()
        for button in self._main.query("#actions Button"):
            if button.id.startswith("order-"):
                button.display = ordering
                disabled = self._action_busy or not self.site_order_available()
                if ordering:
                    draft = self._site_order
                    selected = self.selections.get("sites")
                    index = draft.order.index(selected) if selected in draft.order else None
                    if button.id == "order-up":
                        disabled |= index is None or index == 0
                    elif button.id == "order-down":
                        disabled |= index is None or index >= len(draft.order) - 1
                    elif button.id == "order-automatic":
                        disabled |= not draft.preferences
                    elif button.id == "order-save":
                        disabled |= not self.site_order_save_available()
                    elif button.id == "order-cancel":
                        disabled = self._action_busy
                button.disabled = disabled
                continue
            if ordering:
                button.display = False
                continue
            if button.id == "manage-reservations":
                button.display = True
            if button.id.startswith("page-"):
                operation = button.id.removeprefix("page-")
                button.display = (
                    view == "autodl"
                    if operation == "register"
                    else view in {"sites", "gpus"}
                    if operation == "models"
                    else view == "sites"
                )
                button.disabled = self._action_busy
            if not button.id.startswith("action-"):
                continue
            allowed = button.id.removeprefix("action-") in valid
            button.display = allowed
            button.disabled = self._action_busy or not allowed
        count = sum(
            not (row.get("_gpu_parent") or row.get("_asset_parent"))
            for row in self.records[view].values()
        )
        heading = Text(VIEW_LABELS.get(view, view.title()), style="bold")
        heading.append(f"  {count} shown", style="dim")
        if view == "jobs":
            heading.append(
                f"  /  {'recent history' if self.show_history else 'active only'}"
                f" · {self.job_sort}",
                style="dim",
            )
        query = self.widget("#filter", Input).value
        if view == "assets":
            heading.append(
                "  /  latest versions · Space for history",
                style="dim",
            )
        if view == "system":
            heading.append("  /  central storage · read-only estimates", style="dim")
        if view == "sites":
            if ordering:
                picked = self._site_order.grabbed
                heading.append(
                    "  /  " + ("move " + plain(picked) if picked else "reorder") + " · not saved",
                    style="yellow",
                )
            else:
                order = " → ".join(
                    (self.snapshot.get("policy") or {}).get("preferred_site_ids", [])
                )
                heading.append(
                    "  /  " + plain(order or "automatic placement", limit=100), style="cyan"
                )
        if query:
            heading.append(f"  /  filter: {plain(query)}", style="cyan")
        self.widget("#page-heading", Static).update(heading)
        identity = (
            (
                row.get("job_id")
                or row.get("ref")
                or row.get("reservation_id")
                or row.get("uuid")
                or row.get("site_id")
                or row.get("metric_id")
            )
            if row
            else None
        )
        self.widget("#selection", Static).update(
            Text(
                (
                    "> "
                    + plain(identity)
                    + ("  ·  " + plain(row.get("name")) if row.get("name") else "")
                )
                if row
                else "",
                style="dim",
            )
        )
        self.widget("#hints", Static).update(
            Text(
                (
                    (
                        "↑↓ actions · Enter open · Tab details · Esc back · "
                        if valid
                        else "←→ tabs · Esc back · "
                        if len(self.detail_tab_names()) > 1
                        else "Esc back · "
                    )
                    if details_open
                    else "←→ tabs · Space expand/collapse · Enter details · / filter · "
                    if self.expansion_target(view, row) is not None
                    else "←→ tabs · n register · Enter details · / filter · "
                    if view == "autodl"
                    else "←→ tabs · ↑↓ select · Enter details/actions · / filter · "
                )
                + "? help · q quit"
            )
        )
        if ordering:
            mode = "↑↓ move · Space drop" if self._site_order.grabbed else "↑↓ select · Space pick"
            ending = " · Enter save · Esc" + (" cancel" if self.size.width >= 60 else "")
            self.widget("#hints", Static).update(Text(mode + ending, style="cyan"))
        elif view == "sites" and not details_open:
            self.widget("#hints", Static).update(
                Text("↑↓ select · Space pick/move · Enter details · ←→ tabs · ? help")
            )
        self.call_after_refresh(self.restore_action_focus)
        self.call_after_refresh(self.repair_focus)

    @on(OptionList.OptionSelected, "#detail-actions")
    def detail_action_selected(self, event):
        event.stop()
        operation = event.option.id
        # OptionSelected may have been queued just before a refresh replaced
        # the actions/target. Do not open a form for its replacement.
        self.update_actions()
        if (
            self.screen is self._main
            and self._main.has_class("details-open")
            and operation in self.available_actions()
            and event.option is self.widget("#detail-actions", OptionList).get_option(operation)
        ):
            self.open_action(operation)

    def focus_detail_actions(self):
        options = self.widget("#detail-actions", OptionList)
        if options.option_count and not options.disabled:
            if options.highlighted is None:
                options.highlighted = 0
            self.focus_main(options)

    @staticmethod
    def visible_focus(widget):
        return (
            widget is not None
            and widget.focusable
            and all(node.display for node in widget.ancestors_with_self)
        )

    def editing_input(self):
        return isinstance(self.focused, Input) and self.visible_focus(self.focused)

    def focus_context(self):
        details = self._main.has_class("details-open")
        return (
            self._navigation_generation,
            self.active_view,
            details,
            self.widget("#detail-tabs", TabbedContent).active if details else None,
        )

    def focus_main(self, widget, *, empty=False):
        """Focus after layout, but never after the user has moved elsewhere.

        Widget.focus() queues an unconditional callback. Use Screen.set_focus
        with a navigation-bound intent instead, including a post-layout pass
        for newly visible tabs/controls. A newer intent supersedes this one.
        """
        if self.screen is not self._main:
            return
        context = self.focus_context()
        intent = (context, widget)
        self._pending_focus = intent
        self._empty_focus = context if empty else None

        def apply(*, after_layout=False):
            if self._pending_focus is not intent:
                return
            if self.screen is self._main and context == self.focus_context():
                if self.visible_focus(widget):
                    self._main.set_focus(widget)
            if after_layout:
                self._pending_focus = None

        apply()
        self.call_after_refresh(partial(apply, after_layout=True))

    def focus_primary(self):
        """Keep pending/empty views keyboard-usable without hidden controls."""
        if self._main.has_class("details-open"):
            pane = self.widget("#detail-tabs", TabbedContent).active.removeprefix("pane-")
            self.focus_main(self.widget(f"#scroll-{pane}", VerticalScroll))
        else:
            table = self.widget(f"#table-{self.active_view}", DataTable)
            if table.display:
                self.focus_main(table)
            elif self.active_view == "autodl" and not self._action_busy:
                self.focus_main(self.widget("#page-register", Button), empty=True)
            else:
                self.focus_main(self.widget("#views", TabbedContent).query_one(Tabs), empty=True)

    def repair_focus(self):
        if self.screen is not self._main or self._pending_focus is not None:
            return
        if (
            self._empty_focus == self.focus_context()
            and isinstance(self.focused, Tabs)
            and not self._main.has_class("details-open")
            and self.widget(f"#table-{self.active_view}", DataTable).display
        ) or not self.visible_focus(self.focused):
            self.focus_primary()

    def action_focus(self, operation=None, target=None):
        details = self._main.has_class("details-open")
        view = self.active_view
        if operation in {"register", "reload"} and not details:
            return ActionFocus(view, None, False, operation)
        row = self.selected()
        if view not in {"gpus", "reservations", "autodl", "sites"} or not row:
            return None
        if target is not None:
            if target.get("site_id") != row.get("site_id"):
                return None
            if view == "gpus" and (target.get("uuid") or target.get("gpu_uuid")) != row["uuid"]:
                return None
            if view == "reservations" and target.get("reservation_id") != row["reservation_id"]:
                return None
        key = (
            self._opened_detail[1] if details and self._opened_detail else self.selections.get(view)
        )
        reservation = (
            target if operation in {"extend", "unreserve"} else self.selected_reservation()
        )
        return ActionFocus(
            view,
            key,
            details,
            operation,
            (reservation or {}).get("reservation_id"),
            bool(reservation),
        )

    def restore_action_focus(self):
        """Restore only the requested control on the same resource after reads."""
        intent = self._focus_return
        if intent is None or self._action_busy:
            return
        details = self._main.has_class("details-open")
        key = (
            self._opened_detail[1]
            if details and self._opened_detail
            else self.selections.get(self.active_view)
        )
        if (
            self.screen is not self._main
            or (self.active_view, details) != (intent.view, intent.details)
            or (intent.key is not None and intent.key != key)
            or (intent.key is not None and intent.key not in self._detail_records[intent.view])
        ):
            self._focus_return = None
            return
        if intent.view in {"gpus", "reservations"}:
            if not self.reservations_fresh():
                return
            reservation = self.selected_reservation()
            if (
                intent.check_reservation
                and (reservation or {}).get("reservation_id") != intent.reservation_id
            ):
                # A replacement is not the reservation the user was acting on.
                self._focus_return = None
                return
        if intent.details:
            if not self._detail_actions:
                return  # e.g. reservation GET completed before the inventory GET
            if intent.operation is not None and intent.operation not in self._detail_actions:
                self._focus_return = None
                return
            self.widget("#detail-actions", OptionList).highlighted = (
                self._detail_actions.index(intent.operation) if intent.operation else 0
            )
            self._focus_return = None
            self.focus_detail_actions()
        elif intent.operation:
            prefix = "page" if intent.operation in {"register", "reload"} else "action"
            button = self.widget(f"#{prefix}-{intent.operation}", Button)
            self._focus_return = None
            if button.display and not button.disabled:
                self.focus_main(button)
            else:
                self.focus_primary()
        else:
            self._focus_return = None
            self.focus_primary()

    async def on_event(self, event):
        # Input may be consumed by widgets or priority bindings before on_key.
        # Once the user navigates while a request/read is pending, don't steal
        # their focus when it completes. Mouse motion alone is not navigation.
        if (
            not event.is_forwarded
            and isinstance(event, (events.InputEvent, events.Paste))
            and not isinstance(event, (events.MouseMove, events.MouseUp))
        ):
            self._focus_return = None
            self._navigation_generation += 1
            self._pending_focus = None
            self._empty_focus = None
            if isinstance(self.screen, ActionDialog) and not (
                isinstance(event, events.Key)
                and event.key == "enter"
                and self.screen._impact_pending
            ):
                # Checking job impact may finish after a user chooses Edit or
                # starts reading. Its completion must not steal that choice.
                self.screen._focus_confirm_when_ready = False
        await super().on_event(event)

    def on_key(self, event):
        if self.screen is not self._main:
            return
        if isinstance(self.focused, Tabs) and event.key in {"down", "enter"}:
            self.focus_primary()
            event.prevent_default()
            event.stop()
            return
        if not self._main.has_class("details-open"):
            return
        options = self.widget("#detail-actions", OptionList)
        name = self.widget("#detail-tabs", TabbedContent).active.removeprefix("pane-")
        content = self.widget(f"#scroll-{name}", VerticalScroll)
        if event.key == "tab" and self.focused is options:
            self.focus_main(content)
        elif event.key == "shift+tab" and self.focused is content and self.available_actions():
            self.focus_detail_actions()
        else:
            return
        event.prevent_default()
        event.stop()

    @on(Button.Pressed, "#actions Button, #page-register")
    def action_pressed(self, event):
        if event.button.id.startswith("page-"):
            self.command_chosen(event.button.id.removeprefix("page-"))
        elif event.button.id == "manage-reservations":
            self._main.remove_class("details-open")
            self.action_view("reservations")
        elif not event.button.disabled:
            self.open_action(event.button.id.removeprefix("action-"))

    def open_action(self, operation, target=None):
        if operation not in {*ACTION_LABELS, "register", "reload"}:
            return
        row = target if target is not None else self.selected()
        if operation == "site-order":
            if self._main.has_class("details-open"):
                self.action_back()
            self.action_pick_site()
            return
        if target is None and operation in {"extend", "unreserve"}:
            if operation not in self.available_actions():
                self.set_notice(
                    "Reservation details changed or are unavailable. Refresh before acting."
                )
                return
            row = self.selected_reservation()
        if row and not self._action_busy and self.screen is self._main:
            self.push_screen(
                ActionDialog(operation, row, client=self.client),
                partial(self.action_confirmed, origin=self.action_focus(operation, row)),
            )

    def action_confirmed(self, result, *, origin=None):
        if result is None:
            self._focus_return = origin
            self.call_after_refresh(self.restore_action_focus)
            self.call_after_refresh(self.repair_focus)
            return
        if result is not None and not self._action_busy:
            operation, target, *_ = result
            self._focus_return = origin or self.action_focus(operation, target)
            self._action_busy = True
            self.update_actions()
            self.focus_primary()
            self.perform_action(result)

    @work(group="action", exit_on_error=False)
    async def perform_action(self, result):
        operation, target, values, flags, key = result
        intent = self._focus_return
        self.set_notice(f"◌ {operation}: request in progress. No automatic mutation retries.")
        detail = ""
        try:
            if operation == "reserve":
                response = await self.client.reserve_gpu(
                    GPUReservationRequest(**values), idempotency_key=key
                )
                if intent is not None and self._focus_return is intent:
                    intent.operation = "extend"
                    intent.reservation_id = mapping(response)["reservation_id"]
                    intent.check_reservation = True
            elif operation == "unreserve":
                await self.client.cancel_gpu_reservation(target["reservation_id"])
                if intent is not None and self._focus_return is intent:
                    intent.operation = "reserve"
                    intent.reservation_id = None
                    intent.check_reservation = True
            elif operation == "extend":
                response = await self.client.extend_gpu_reservation(
                    target["reservation_id"], GPUReservationExtensionRequest(**values)
                )
                detail = " Reserved until " + date(mapping(response)["expires_at"]) + "."
            elif operation == "release":
                await self.client.release_autodl(target["site_id"], force=flags.get("force", False))
                detail = " Release requested; shutdown completes after draining and required I/O."
            elif operation == "unregister":
                await self.client.unregister_autodl(target["site_id"])
                detail = " Registration removed; this is not a shutdown command."
            elif operation == "register":
                response = await self.client.register_autodl(values["ssh"], values["password"])
                # Only curated non-secret fields enter the view. Show accepted
                # startup immediately, before the first site-agent heartbeat.
                if response.get("site_id"):
                    self.autodl_instances = [
                        row
                        for row in self.autodl_instances
                        if row.get("site_id") != response["site_id"]
                    ] + [
                        {
                            "site_id": response["site_id"],
                            "state": response.get("state", "starting"),
                        }
                    ]
                    self.render_tables()
                detail = " Registration accepted: " + plain(
                    response.get("site_id", "inspect AutoDL view")
                )
            elif operation == "reload":
                response = await self.client.reload_policy()
                detail = " Policy request accepted; " + plain(
                    response.get("status", "inspect policy status")
                )
            elif operation == "site-order":
                response = await self.client.set_site_preference(SitePreferenceRequest(**values))
                self.snapshot["policy"] = response
                self._site_order = None
                self.render_tables()
                order = " → ".join(response.get("preferred_site_ids", []))
                detail = " New placements: " + (order or "automatic") + ". Running jobs unchanged."
            else:
                raise ValueError("unsupported management action")
            self.set_notice(f"✓ {operation}: request completed.{detail}")
        except Exception as exc:
            self._focus_return = None
            self.set_notice(
                f"{operation}: not confirmed ({safe_error(exc)}). "
                "Refresh and inspect the target before retrying; the server may have acted."
            )
        finally:
            if operation == "register":
                values.pop("password", None)
            self._action_busy = False
            # A response from before any mutation must not overwrite its new
            # state. Busy readers coalesce a fresh read when they finish.
            self._mutation_version += 1
            if operation in {"extend", "unreserve", "reserve"}:
                # Discard reads started before a mutation (including an uncertain
                # response); only a fresh GET may re-enable reservation actions.
                self._reservations_checked_at = None
            self.refresh_status()
            self.refresh_extras()
            self.update_actions()

    def action_safe_quit(self):
        if self._action_busy:
            self.set_notice("A management action is pending; wait for its result before quitting.")
        else:
            self.exit()

    def action_register_instance(self):
        if (
            self.screen is self._main
            and not self.editing_input()
            and not self._main.has_class("details-open")
            and self.active_view in {"sites", "autodl"}
        ):
            self.command_chosen("register")

    def expansion_target(self, view, row):
        if not row:
            return None
        if view == "gpus":
            return row.get("_gpu_parent") or gpu_key(row), self.expanded_gpus
        if view == "assets":
            return asset_group(row), self.expanded_assets
        if view == "jobs":
            identity = row["job_id"] if row.get("is_array") else row.get("array_id")
            if self.records[view].get(identity, {}).get("is_array"):
                return identity, self.expanded_arrays
        return None

    def disclosure_label(self, view, row):
        parent = (
            (view == "gpus" and not row.get("_gpu_parent"))
            or (view == "assets" and not row.get("_asset_parent"))
            or row.get("is_array")
        )
        target = self.expansion_target(view, row) if parent else None
        if target is not None:
            identity, expanded = target
            return "▾" if identity in expanded else "▸"
        return None

    def action_toggle_expansion(self):
        if (
            self.screen is not self._main
            or self.editing_input()
            or self._main.has_class("details-open")
        ):
            return
        if self.active_view == "sites":
            self.action_pick_site()
            return
        target = self.expansion_target(self.active_view, self.selected())
        if target is not None:
            identity, expanded = target
            if identity in expanded:
                expanded.remove(identity)
            else:
                expanded.add(identity)
                if self.active_view == "jobs":
                    self.refresh_array(identity)
                elif self.active_view == "assets":
                    self.asset_history_state = "loading"
                    self.refresh_extras()
            self.render_tables()
            self.focus_primary()

    def action_refresh_now(self):
        self.tick()

    def action_search(self):
        if self.active_view == "sites" and self._site_order:
            self.set_notice("Enter saves the site order; Esc cancels before filtering.")
            return
        self._main.add_class("searching")
        self.focus_main(self.widget("#filter", Input))

    def action_clear_filter(self):
        self._main.remove_class("searching")
        self.widget("#filter", Input).value = ""
        self.focus_primary()

    def action_history(self):
        if self.screen is not self._main or self.editing_input():
            return
        if self.active_view != "jobs":
            return
        self.show_history = not self.show_history
        self.render_tables()
        self.update_status()
        if not self._main.has_class("details-open"):
            # An empty Jobs list focuses the tab bar. Revealing history makes
            # the table visible again, but generic focus repair sees that bar
            # as valid. This explicit navigation must hand arrows to the rows.
            self.focus_primary()

    def action_sort_jobs(self):
        choices = ("queue", "name", "site")
        self.job_sort = choices[(choices.index(self.job_sort) + 1) % len(choices)]
        self.render_tables()
        self.update_status()

    def action_details(self):
        if self.screen is not self._main:
            return
        if self._main.has_class("details-open"):
            self.action_back()
        else:
            self.show_details()

    def show_details(self):
        if self.active_view == "sites" and self._site_order:
            self.set_notice("Enter saves the site order; Esc cancels before opening details.")
            return
        row = self.selected()
        if not self._main.has_class("details-open"):
            row = self.selected()
            if self.active_view == "assets" and row and row.get("_asset_info"):
                self.selections["assets"] = row["_asset_parent"]
                row = self.records["assets"][row["_asset_parent"]]
            if self.active_view == "gpus" and row and row.get("_gpu_parent"):
                self.selections["gpus"] = row["_gpu_parent"]
                row = self.records["gpus"][row["_gpu_parent"]]
            self._opened_detail = (
                (self.active_view, self.selections.get(self.active_view), dict(row))
                if row
                else None
            )
        self._main.add_class("details-open")
        self._focus_return = self.action_focus()
        self.widget("#detail-tabs", TabbedContent).active = "pane-summary"
        tabs = self.widget("#detail-tabs", TabbedContent)
        for name in ("attempts", "logs"):
            if name in self.detail_tab_names():
                tabs.show_tab(f"pane-{name}")
            else:
                tabs.hide_tab(f"pane-{name}")
        self.update_actions()
        self.schedule_detail()
        if self.available_actions() and not self._action_busy:
            options = self.widget("#detail-actions", OptionList)
            options.highlighted = 0
            self.focus_detail_actions()
        else:
            self.focus_primary()

    def action_view(self, view):
        if self.screen is not self._main or self.editing_input():
            return
        if self._main.has_class("details-open"):
            # Numbered navigation is local to the visible page. Esc returns
            # to the main views; 4-6 must not unexpectedly leave details.
            index = VIEWS.index(view)
            names = self.detail_tab_names()
            if index < len(names):
                name = names[index]
                self.widget("#detail-tabs", TabbedContent).active = f"pane-{name}"
                self.focus_primary()
            return
        self._main.remove_class("details-open")
        self.action_clear_filter()
        self.widget("#views", TabbedContent).active = view
        self.focus_primary()

    def check_action(self, action, parameters):
        if action in {"move_site", "move_picked_site", "save_site_order"}:
            available = (
                hasattr(self, "_main")
                and self.screen is self._main
                and self.active_view == "sites"
                and not self._main.has_class("details-open")
                and not self.editing_input()
            )
            if action == "save_site_order":
                return (
                    available
                    and self._site_order is not None
                    and isinstance(self.focused, DataTable)
                )
            if action == "move_picked_site":
                return (
                    available
                    and isinstance(self.focused, DataTable)
                    and self._site_order is not None
                    and self._site_order.grabbed is not None
                )
            return available
        if action == "toggle_expansion":
            return (
                hasattr(self, "_main")
                and self.screen is self._main
                and not self._main.has_class("details-open")
                and isinstance(self.focused, DataTable)
                and self.focused.id == f"table-{self.active_view}"
                and (
                    bool(self.selected())
                    if self.active_view == "sites"
                    else self.expansion_target(self.active_view, self.selected()) is not None
                )
            )
        if action in {"adjacent_tab", "pan_content"}:
            # Priority arrows must fall through to native editing and modal
            # controls, rather than silently consuming their key events.
            return hasattr(self, "_main") and self.screen is self._main and not self.editing_input()
        return super().check_action(action, parameters)

    def detail_tab_names(self):
        row = self.selected()
        return (
            ("summary", "attempts", "logs")
            if self.active_view == "jobs" and row and not row.get("is_array")
            else ("summary",)
        )

    def action_adjacent_tab(self, direction):
        if not self.check_action("adjacent_tab", (direction,)):
            return
        if self._main.has_class("details-open"):
            names = self.detail_tab_names()
            tabs = self.widget("#detail-tabs", TabbedContent)
            index = names.index(tabs.active.removeprefix("pane-"))
            name = names[(index + direction) % len(names)]
            tabs.active = f"pane-{name}"
            self.focus_primary()
        else:
            index = (VIEWS.index(self.active_view) + direction) % len(VIEWS)
            self.action_view(VIEWS[index])

    def action_pan_content(self, direction):
        if not self.check_action("pan_content", (direction,)):
            return
        if self._main.has_class("details-open"):
            name = self.widget("#detail-tabs", TabbedContent).active.removeprefix("pane-")
            content = self.widget(f"#scroll-{name}", VerticalScroll)
        else:
            content = self.widget(f"#table-{self.active_view}", DataTable)
        content.scroll_relative(x=direction * 5, animate=False)

    def action_back(self):
        if self.screen is not self._main:
            return
        if self.active_view == "sites" and self._site_order:
            self.cancel_site_order()
            return
        if self._main.has_class("details-open"):
            self._focus_return = None
            self._main.remove_class("details-open")
            self._opened_detail = None
            self.focus_primary()
            self.update_actions()
            self.schedule_detail()
        else:
            self.action_clear_filter()

    def action_help(self):
        if self.screen is self._main:
            self.push_screen(
                ChoiceDialog(
                    "SHORTCUTS\n\n1 Jobs · 2 GPUs · 3 Sites\n"
                    "4 Assets · 5 Reservations · 6 AutoDL · 7 System\n\n"
                    "Job details: 1 Summary · 2 Attempts · 3 Logs\n"
                    "Array and resource details: Summary\n"
                    "Esc returns to the main views.\n\n"
                    "↑↓ Select row   Enter Details   Esc Back\n/ Filter   Enter Finish filtering\n"
                    "Enter opens details with available actions shown directly.\n"
                    "On arrays, GPUs and assets, Space or the disclosure arrow expands/collapses\n"
                    "members, processes or older versions. Enter/click the row opens details.\n"
                    "Enter shows the same read-only process list in GPU details.\n"
                    "n registers an already-rented instance from AutoDL or Sites.\n"
                    "On Sites, ↑↓ selects; Space picks up a site, then ↑↓ moves it.\n"
                    "Space drops it to select another · Enter saves · Esc cancels.\n"
                    "Site details → Reorder sites offers the same inline mouse controls.\n"
                    "Shift+↑↓ is an optional shortcut; no modifier keys are required.\n"
                    "Auto restores automatic placement. No separate screen.\n"
                    "Site order saves live without moving running work.\n"
                    "↑↓ chooses an action; Enter opens its review form.\n"
                    "a Focus detail actions   d Toggle details\n"
                    "Disruptive management actions require review and confirmation.\n"
                    "Page-wide actions remain in the list's bottom toolbar.\n"
                    "h Toggle job history   s Sort jobs\nr Refresh   q Quit\n\n"
                    "←/→ switches tabs directly; Tab moves focus.\n"
                    "On a tab label, ↓ or Enter moves into its content.\n"
                    "Logs follow new output; scroll up to pause, End to follow again.\n"
                    "Shift+←/→ scrolls wide content horizontally.\n"
                    "Shortcuts do not replace characters while typing in fields."
                )
            )

    def action_context_action(self):
        if self.screen is not self._main or self._action_busy:
            return
        row = self.selected()
        if not row:
            self.set_notice("Select a row first.")
            return
        if not self._main.has_class("details-open"):
            self.show_details()
        elif self.available_actions():
            self.focus_detail_actions()
        else:
            self._focus_return = self.action_focus()
            self.refresh_extras()

    def command_chosen(self, command):
        if command in {"gpus", "reservations"}:
            self._main.remove_class("details-open")
            self.action_view(command)
        elif command in {"register", "reload"}:
            self.open_action(
                command,
                {"site_id": "New AutoDL instance" if command == "register" else "Scheduler policy"},
            )
        elif command == "models":
            self.show_report(command)

    @work(group="reports", exit_on_error=False)
    async def show_report(self, command):
        if self._report_busy:
            return
        self._report_busy = True
        origin = self.focus_context()
        self.set_notice(f"Loading {command}…")
        try:
            result = await self.client.models()
            if self.screen is self._main and origin == self.focus_context():
                self.push_screen(ChoiceDialog(models_page(result)))
            self.set_notice(
                f"{command}: loaded"
                if self.screen is not self._main
                else f"{command}: loaded; reopen Models to inspect"
            )
        except Exception as exc:
            self.set_notice(
                f"{command}: unavailable ({safe_error(exc)}). Operator access may be required."
            )
        finally:
            self._report_busy = False


def run_workbench(*, interval_seconds=2.0, no_color=False):
    Workbench(interval=interval_seconds, no_color=no_color).run()
