from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from textual import events
from textual.containers import Vertical
from textual.widgets import Checkbox, DataTable, Input, OptionList, Static, TabbedContent, Tabs

from topodrome.client.workbench import (
    ActionDialog,
    ChoiceDialog,
    Workbench,
    job_columns,
    plain,
    visible_snapshot,
)

pytestmark = pytest.mark.unit


async def tab_to(pilot, app, widget_id):
    for _ in range(10):
        if app.screen.focused and app.screen.focused.id == widget_id:
            return
        await pilot.press("tab")
    raise AssertionError(f"Cannot reach {widget_id} with Tab")


async def confirm_action(pilot, app):
    before = len(app.client.calls)
    await tab_to(pilot, app, "apply")
    await pilot.press("enter")
    assert isinstance(app.screen, ActionDialog)
    assert app.screen.review_payload is not None
    assert len(app.client.calls) == before
    for _ in range(30):
        if not app.screen._impact_pending:
            break
        await pilot.pause(0.05)
    assert not app.screen._impact_pending
    assert app.screen.focused.id == "apply"
    await pilot.press("enter")
    await pilot.pause()


class FakeClient:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.fail = False
        self.block = None
        self.reservations = []
        self.reservations_error = False
        self.reservation_gate = None
        self.array_gate = None
        self.array_error = False
        self.array_members = []
        self.instances = []
        self.snapshot = {
            "sites": [
                {"site_id": "lab1", "kind": "lab", "state": "ready"},
                {"site_id": "remote-jupyter-1", "kind": "jupyter", "state": "offline"},
                {"site_id": "autodl-1", "kind": "autodl", "state": "ready"},
            ],
            "gpus": [
                {"site_id": "lab1", "uuid": "GPU-exact", "index": 3, "healthy": True},
                {"site_id": "remote-jupyter-1", "uuid": "GPU-stale", "index": 0},
            ],
            "jobs": [
                {"job_id": "job_abcd1111", "name": "Alpha", "state": "running", "site_id": "lab1"},
                {"job_id": "job_abcd2222", "name": "Beta", "state": "retry_wait"},
                {"job_id": "job_done", "name": "Finished", "state": "succeeded"},
            ],
        }

    async def system_status(self):
        if self.block:
            await self.block.wait()
        if self.fail:
            raise RuntimeError("DO_NOT_DISPLAY_SECRET")
        return deepcopy(self.snapshot)

    async def array(self, array_id, **kwargs):
        if self.array_gate:
            await self.array_gate.wait()
        if self.array_error:
            raise RuntimeError("DO_NOT_DISPLAY_SECRET")
        summary = next(r for r in self.snapshot["arrays"] if r["array_id"] == array_id)
        return deepcopy({**summary, "members": self.array_members})

    async def status(self, job):
        return {
            "job_id": job,
            "spec": {"name": job, "required_outputs": ["out.txt"]},
            "attempts": 1,
        }

    async def attempts(self, job):
        return [{"attempt_id": "attempt_1", "state": "running", "site_id": "lab1"}]

    async def logs(self, job, **kwargs):
        return f"logs for {job}\n[bold]literal[/bold]\n\x1b]52;c;bad\x07"

    async def assets(self, **kwargs):
        return [
            {
                "ref": "asset://private/a@v1",
                "name": "a",
                "version": "v1",
                "visibility": "private",
                "total_size_bytes": 2,
                "files": [],
            }
        ]

    async def gpu_reservations(self, **kwargs):
        if self.reservation_gate:
            await self.reservation_gate.wait()
        if self.reservations_error:
            raise RuntimeError("DO_NOT_DISPLAY_SECRET")
        return deepcopy(self.reservations)

    async def cancel(self, job):
        self.calls.append(("cancel", job))

    async def reserve_gpu(self, request, **kwargs):
        self.calls.append(("reserve", request.model_dump(), kwargs))
        row = active_reservation(f"reservation_{len(self.calls)}", request.site_id)
        row.update(
            gpu_uuid=request.gpu_uuid,
            reason=request.reason,
            force=request.force,
            expires_at=(
                datetime.now(UTC) + timedelta(seconds=request.duration_seconds)
            ).isoformat(),
        )
        self.reservations.append(row)
        for gpu in self.snapshot["gpus"]:
            if (gpu["site_id"], gpu["uuid"]) == (row["site_id"], row["gpu_uuid"]):
                gpu["operator_reserved"] = True
        return deepcopy(row)

    async def release_autodl(self, site, **kwargs):
        self.calls.append(("release", site, kwargs))

    async def register_autodl(self, ssh, password):
        self.calls.append(("register", ssh, password))
        self.instances = [{"site_id": "autodl-new", "state": "starting"}]
        return {"site_id": "autodl-new", "password": "MUST_NOT_RENDER"}

    async def autodl_instances(self):
        return deepcopy(self.instances)

    async def unregister_autodl(self, site):
        self.calls.append(("unregister", site))
        self.snapshot["sites"] = [r for r in self.snapshot["sites"] if r["site_id"] != site]

    async def reload_policy(self):
        self.calls.append(("reload",))
        return {"status": "staged"}

    async def models(self):
        return [{"alias": "llama", "gpu": 1}]

    async def storage_gc(self, **kwargs):
        self.calls.append(("storage", kwargs))
        return {"apply": False, "candidates": 0}

    async def publish_asset(self, request):
        self.calls.append(("publish", request.model_dump()))

    async def download_asset(self, *args, **kwargs):
        self.calls.append(("download", args, kwargs))

    async def cancel_gpu_reservation(self, reservation):
        self.calls.append(("unreserve", reservation))
        removed = [r for r in self.reservations if r["reservation_id"] == reservation]
        self.reservations = [r for r in self.reservations if r not in removed]
        for gpu in self.snapshot["gpus"]:
            if any(
                r["site_id"] == gpu["site_id"] and r["gpu_uuid"] == gpu["uuid"] for r in removed
            ):
                gpu["operator_reserved"] = False

    async def extend_gpu_reservation(self, reservation_id, extension):
        self.calls.append(("extend", reservation_id, extension.model_dump(mode="json")))
        row = next(r for r in self.reservations if r["reservation_id"] == reservation_id)
        row["expires_at"] = (
            extension.expected_expires_at + timedelta(seconds=extension.additional_seconds)
        ).isoformat()
        return deepcopy(row)

    async def close(self):
        self.closed = True


def test_offline_inventory_is_hidden_not_site_or_jobs():
    original = FakeClient().snapshot
    result = visible_snapshot(original)
    assert len(result["sites"]) == 3 and len(result["gpus"]) == 1
    assert len(original["gpus"]) == 2
    assert result["jobs"] == original["jobs"]
    assert "\x1b" not in plain("\x1b[31m[bold]x\x07")


async def test_navigation_filter_selection_resize_and_history():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        table = app.widget("#table-jobs", DataTable)
        assert table.row_count == 2
        table.move_cursor(row=1)
        await pilot.pause()
        assert app.selected()["job_id"] == "job_abcd2222"
        client.snapshot["jobs"][0]["name"] = "[red]Literal"
        app.refresh_status()
        await pilot.pause()
        assert app.selected()["job_id"] == "job_abcd2222"
        assert "job_abcd2222" in str(app.widget("#detail-logs", Static).render())
        app.action_history()
        assert table.row_count == 3
        app.widget("#filter", Input).value = "finished"
        await pilot.pause()
        assert table.row_count == 1
        await pilot.resize_terminal(70, 24)
        await pilot.pause()
        assert app._main.has_class("narrow")
        app.action_clear_filter()
        await pilot.press("q")
    assert client.closed


@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
async def test_first_history_toggle_from_empty_jobs_focuses_rows_for_arrow_navigation(size):
    client = FakeClient()
    for row in client.snapshot["jobs"]:
        row["state"] = "succeeded"
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        table = app.widget("#table-jobs", DataTable)
        assert table.row_count == 0 and not table.display
        for _ in range(2):
            await pilot.press("h", "down")
            assert table.row_count == 3 and table.display
            assert app.focused is table
            assert app.selected()["job_id"] == "job_abcd2222"
            app.refresh_status()
            await pilot.pause()
            assert app.focused is table
            assert app.selected()["job_id"] == "job_abcd2222"
            await pilot.press("enter")
            assert app._main.has_class("details-open")
            assert app.selected()["job_id"] == "job_abcd2222"
            await pilot.press("escape", "h")
            assert not table.display and table.row_count == 0
            assert app.focused is not None and app.focused.focusable
        assert not client.calls


@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
async def test_mouse_main_and_detail_tabs_focus_their_content(size):
    client = FakeClient()
    client.snapshot["gpus"].append(
        {"site_id": "lab1", "uuid": "GPU-second", "index": 4, "healthy": True}
    )
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        tabs = app.widget("#views", TabbedContent)
        await pilot.click(tabs.get_tab("gpus"))
        assert app.active_view == "gpus"
        assert app.focused.id == "table-gpus"
        await pilot.press("down")
        assert app.selected()["uuid"] == "GPU-second"
        await pilot.click(tabs.get_tab("gpus"))  # already selected
        assert app.focused.id == "table-gpus"
        await pilot.press("up")
        assert app.selected()["uuid"] == "GPU-exact"
        await pilot.click(tabs.get_tab("jobs"))
        await pilot.press("enter")
        details = app.widget("#detail-tabs", TabbedContent)
        await pilot.click(details.get_tab("pane-logs"))
        assert details.active == "pane-logs"
        assert app.focused.id == "scroll-logs"
        await pilot.click(details.get_tab("pane-logs"))
        assert app.focused.id == "scroll-logs"
        await pilot.press("escape")
        assert app.focused.id == "table-jobs"


@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
async def test_finishing_empty_filter_and_clearing_it_keeps_list_navigation(size):
    app = Workbench(FakeClient(), interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("slash")
        app.widget("#filter", Input).value = "no-such-job"
        await pilot.pause()
        await pilot.press("enter")
        assert not isinstance(app.focused, Input)
        assert app.focused is not None and app.focused.focusable
        await pilot.press("escape", "down")
        assert app.focused.id == "table-jobs"
        assert app.selected()["job_id"] == "job_abcd2222"


async def test_delayed_asset_rows_return_focus_from_automatic_empty_fallback():
    client = FakeClient()
    gate = asyncio.Event()
    original = client.assets

    async def delayed(**kwargs):
        await gate.wait()
        assets = await original(**kwargs)
        return [*assets, {**assets[0], "name": "b", "ref": "asset://private/b@v1"}]

    client.assets = delayed
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("4")
        assert not app.widget("#table-assets", DataTable).display
        gate.set()
        await pilot.pause()
        assert app.focused.id == "table-assets"
        await pilot.press("down")
        assert app.selected()["ref"] == "asset://private/b@v1"


@pytest.mark.parametrize("navigation", ["filter", "tab", "page", "help"])
async def test_delayed_rows_do_not_steal_focus_after_deliberate_navigation(navigation):
    client = FakeClient()
    gate = asyncio.Event()
    original = client.assets

    async def delayed(**kwargs):
        await gate.wait()
        return await original(**kwargs)

    client.assets = delayed
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("4")
        await pilot.press(
            {"filter": "slash", "tab": "tab", "page": "2", "help": "question_mark"}[navigation]
        )
        screen, focused = app.screen, app.focused
        gate.set()
        await pilot.pause()
        assert app.screen is screen and app.focused is focused
        assert not client.calls


async def test_last_row_disappearing_then_new_rows_arriving_restores_list_focus():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        original = deepcopy(client.snapshot["jobs"])
        client.snapshot["jobs"] = []
        app.refresh_status()
        await pilot.pause()
        assert isinstance(app.focused, Tabs)
        client.snapshot["jobs"] = original
        app.refresh_status()
        await pilot.pause()
        assert app.focused.id == "table-jobs"
        await pilot.press("down")
        assert app.selected()["job_id"] == "job_abcd2222"


async def test_help_scrolls_and_returns_to_usable_focus_after_rows_disappear():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        await pilot.press("question_mark")
        assert app.focused.id == "choices"
        await pilot.press("pagedown")
        await pilot.pause()
        assert app.focused.scroll_y > 0
        client.snapshot["jobs"] = []
        app.refresh_status()
        await pilot.pause()
        assert app.focused.id == "choices"
        await pilot.press("escape")
        assert app.visible_focus(app.focused)
        await pilot.press("right")
        assert app.focused.id == "table-gpus"


@pytest.mark.parametrize("details", [False, True])
@pytest.mark.parametrize("key", ["down", "enter"])
async def test_tab_strip_hands_arrows_to_content_without_opening_actions(details, key):
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        if details:
            await pilot.press("enter")
        app.widget("#detail-tabs" if details else "#views", TabbedContent).query_one(Tabs).focus()
        await pilot.pause()
        assert isinstance(app.focused, Tabs)
        await pilot.press(key)
        assert app.focused.id == ("scroll-summary" if details else "table-jobs")
        assert app._main.has_class("details-open") == details
        assert not client.calls


async def test_rapid_navigation_and_resize_keep_latest_visible_focus():
    app = Workbench(FakeClient(), interval=60)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.action_view("gpus")
        app.action_view("jobs")
        app.action_search()
        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        assert app.focused.id == "filter"
        await pilot.press("escape", "enter", "right", "right")
        await pilot.resize_terminal(120, 40)
        await pilot.pause()
        assert app.focused.id == "scroll-logs"
        await pilot.press("escape", "down")
        assert app.focused.id == "table-jobs"
        assert app.selected()["job_id"] == "job_abcd2222"


async def test_main_navigation_shortcuts_cannot_change_focus_behind_a_modal():
    app = Workbench(FakeClient(), interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("question_mark")
        modal = app.screen
        await pilot.press("slash", "s", "d", "a", "1")
        assert app.screen is modal
        assert not app._main.has_class("searching")
        assert not app._main.has_class("details-open")
        assert app.job_sort == "queue"
        await pilot.press("escape", "down")
        assert app.focused.id == "table-jobs"
        assert app.selected()["job_id"] == "job_abcd2222"


async def test_delayed_models_report_does_not_open_over_later_navigation():
    client = FakeClient()
    gate = asyncio.Event()
    original = client.models

    async def delayed():
        await gate.wait()
        return await original()

    client.models = delayed
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("3")
        await pilot.click("#page-models")
        await pilot.press("2")
        gate.set()
        await pilot.pause()
        assert app.screen is app._main
        assert app.active_view == "gpus"
        assert app.focused.id == "table-gpus"


async def test_history_toggle_preserves_details_and_ignores_other_pages_fields_and_modals():
    app = Workbench(FakeClient(), interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        detail_focus = app.focused
        await pilot.press("h")
        assert app.show_history
        assert app._main.has_class("details-open")
        assert app.focused is detail_focus
        assert app.selected()["job_id"] == "job_abcd1111"
        await pilot.press("escape", "slash", "h")
        assert app.widget("#filter", Input).value == "h"
        assert app.focused.id == "filter"
        assert app.show_history
        await pilot.press("escape", "2", "h")
        assert app.active_view == "gpus"
        assert app.focused.id == "table-gpus"
        assert app.show_history
        await pilot.press("1", "question_mark")
        modal = app.screen
        await pilot.press("h")
        assert app.screen is modal and app.screen is not app._main
        assert app.show_history


def test_job_columns_fit_terminal_and_keep_timing_when_space_allows():
    for width in (40, 58, 65, 80, 99, 100, 120, 160):
        columns = job_columns(width)
        assert sum(size + 2 for _, size in columns) <= width
        assert ("Wait / Run / I/O" in dict(columns)) is (width >= 58)


async def test_array_inline_refresh_resize_and_back_preserve_identity():
    client = FakeClient()
    array_id = "array_sweep"
    client.snapshot["arrays"] = [
        {
            "array_id": array_id,
            "name": "Sweep",
            "state": "running",
            "total": 2,
            "terminal": 1,
            "counts": {"running": 1, "succeeded": 1},
            "max_parallel": 1,
        }
    ]
    client.snapshot["jobs"] = [
        {
            "job_id": "job_element0",
            "array_id": array_id,
            "array_index": 0,
            "name": "Sweep",
            "state": "running",
            "site_id": "lab1",
            "timing": {"seconds": {"queued": 10, "compute": 50}},
        }
    ]
    client.array_members = [
        *deepcopy(client.snapshot["jobs"]),
        {
            "job_id": "job_element1",
            "array_id": array_id,
            "array_index": 1,
            "name": "Sweep",
            "state": "succeeded",
            "site_id": "lab1",
        },
    ]
    client.snapshot["jobs"].append({"job_id": "job_other", "name": "Other", "state": "queued"})
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        assert list(app.records["jobs"]) == [array_id, "job_other"]
        await pilot.press("space")
        await pilot.pause()
        assert app.expanded_arrays == {array_id}
        assert app.focused.id == "table-jobs"
        assert list(app.records["jobs"]) == [array_id, "job_element0", "job_other"]
        await pilot.press("h")
        assert list(app.records["jobs"]) == [array_id, "job_element0", "job_element1", "job_other"]
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert app._main.has_class("details-open")
        assert app.selected()["job_id"] == "job_element1"
        await pilot.press("escape")
        await pilot.resize_terminal(65, 24)
        await pilot.pause()
        assert app.selected()["job_id"] == "job_element1"
        app.refresh_status()
        await pilot.pause()
        assert app.selected()["job_id"] == "job_element1"
        await pilot.press("up", "up", "space")
        assert not app.expanded_arrays
        assert list(app.records["jobs"]) == [array_id, "job_other"]
        assert app.selected()["job_id"] == array_id
        assert app.focused.id == "table-jobs"


async def test_array_slow_fetch_collapse_and_tab_navigation_never_steal_focus():
    client = FakeClient()
    client.snapshot["arrays"] = [
        {
            "array_id": "array_slow",
            "name": "Slow",
            "state": "queued",
            "counts": {"queued": 1},
            "total": 1,
            "terminal": 0,
        }
    ]
    client.array_members = [
        {
            "job_id": "job_child",
            "array_id": "array_slow",
            "array_index": 0,
            "state": "queued",
        }
    ]
    client.snapshot["jobs"] = client.array_members
    client.array_gate = asyncio.Event()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        await pilot.press("space")
        assert "array_slow" in app._array_busy
        await pilot.press("space", "right")
        client.array_gate.set()
        await pilot.pause()
        assert app.active_view == "gpus"
        assert app.focused.id == "table-gpus"
        assert not app.expanded_arrays and not app.array_cache
        assert list(app.records["jobs"]) == ["array_slow"]
        await pilot.press("left")
        client.array_error = True
        await pilot.press("space")
        await pilot.pause()
        assert app.failure is None and app.array_errors == {"array_slow": "RuntimeError"}
        assert "refresh unavailable" in str(app.records["jobs"]["array_slow"])
        assert "DO_NOT_DISPLAY_SECRET" not in str(app.records)
        client.array_error = False
        app.refresh_status()
        await pilot.pause()
        assert not app.array_errors
        assert list(app.records["jobs"]) == ["array_slow", "job_child"]


def test_array_tree_sort_and_filter_keep_children_with_parent():
    from topodrome.client.workbench_arrays import job_tree

    parents = [
        {
            "array_id": identity,
            "name": name,
            "state": "running",
            "total": 2,
            "terminal": 1,
            "counts": {"running": 1, "succeeded": 1},
        }
        for identity, name in [("array_a", "Zulu"), ("array_b", "Alpha")]
    ]
    children = [
        {
            "job_id": f"job_{group}{index}",
            "array_id": f"array_{group}",
            "array_index": index,
            "name": f"child-{group}-{index}",
            "state": "running" if index == 0 else "succeeded",
        }
        for group in ("a", "b")
        for index in (1, 0)
    ]
    snapshot = {"arrays": parents, "jobs": children}
    expanded = {"array_a", "array_b"}
    rows, _ = job_tree(snapshot, expanded, {}, {}, sort="name", history=True)
    assert [r["job_id"] for r in rows] == [
        "array_b",
        "job_b0",
        "job_b1",
        "array_a",
        "job_a0",
        "job_a1",
    ]
    rows, _ = job_tree(snapshot, expanded, {}, {}, query="child-b-0")
    assert [r["job_id"] for r in rows] == ["array_b", "job_b0"]
    rows, _ = job_tree(snapshot, expanded, {}, {}, query="alpha")
    assert [r["job_id"] for r in rows] == ["array_b", "job_b0"]


async def test_empty_row_highlights_and_refresh_do_not_crash_or_retarget():
    client = FakeClient()
    jobs = client.snapshot["jobs"]
    client.snapshot["jobs"] = []
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        table = app.widget("#table-jobs", DataTable)
        table.post_message(DataTable.RowHighlighted(table, -1, None))
        await pilot.press("up", "down", "a")
        assert app.selected() is None
        assert not isinstance(app.screen, ActionDialog)
        client.snapshot["jobs"] = jobs
        app.refresh_status()
        await pilot.pause()
        table.move_cursor(row=1)
        await pilot.pause()
        table.post_message(DataTable.RowHighlighted(table, -1, None))
        await pilot.pause()
        assert app.selected()["job_id"] == "job_abcd2222"
        app.widget("#filter", Input).value = "no-matching-job"
        await pilot.pause()
        table.post_message(DataTable.RowHighlighted(table, -1, None))
        await pilot.press("up", "down", "a")
        assert app.selected() is None
        assert not client.calls
        app.action_clear_filter()
        await pilot.pause()
        assert table.row_count == 2
        await pilot.press("q")


async def test_simple_layout_and_terminal_default_colors():
    from rich.segment import Segment
    from rich.style import Style
    from textual.color import Color

    from topodrome.client.workbench import TerminalDefaults

    segments = TerminalDefaults(no_color=True).apply(
        [Segment("selected", Style(color="red", bgcolor="blue", reverse=True, underline=True))],
        Color(0, 0, 0),
    )
    style = segments[0].style
    assert style.color.is_default and style.bgcolor.is_default
    assert not style.reverse and style.underline
    colored = TerminalDefaults().apply(
        [
            Segment("failed", Style(color="red", bgcolor="blue", reverse=True)),
            Segment("plain", Style(color="#abcdef", bgcolor="green")),
        ],
        Color(0, 0, 0),
    )
    assert colored[0].style.color.name == "red"
    assert colored[1].style.color.is_default
    assert all(s.style.bgcolor.is_default and not s.style.reverse for s in colored)
    from topodrome.client.workbench import status_cell

    assert status_cell("failed", "State").style == "red"
    assert status_cell("failed", "Name").style == ""
    app = Workbench(FakeClient(), interval=60)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        assert not app.widget("#inspector", Vertical).display
        assert not app.widget("#filter", Input).display
        assert app.query_one("#actions").size.height == 1
        assert app.widget("#table-jobs", DataTable).size.width >= 138
        await pilot.press("d")
        assert app.widget("#inspector", Vertical).display
        assert not app.widget("#views", TabbedContent).display
        await pilot.press("d")
        assert app.widget("#views", TabbedContent).display
        await pilot.press("a")
        assert app.screen is app._main and app._main.has_class("details-open")
        assert not app.widget("#detail-actions", OptionList).display
        await pilot.press("escape", "q")


async def test_numbered_navigation_enter_escape_help_and_inline_actions():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        for key, view in enumerate(
            ("jobs", "gpus", "sites", "assets", "reservations", "autodl"), 1
        ):
            await pilot.press(str(key))
            assert app.active_view == view
        await pilot.press("1", "enter")
        assert app._main.has_class("details-open")
        for key, pane in (("3", "logs"), ("2", "attempts"), ("1", "summary")):
            await pilot.press(key)
            assert app.widget("#detail-tabs", TabbedContent).active == f"pane-{pane}"
            assert app.active_view == "jobs"
            assert app.focused.id == f"scroll-{pane}"
        await pilot.press("6")
        assert app._main.has_class("details-open")
        assert app.active_view == "jobs"
        await pilot.press("escape", "2")
        assert not app._main.has_class("details-open")
        await pilot.press("question_mark")
        assert isinstance(app.screen, ChoiceDialog)
        await pilot.press("6")
        assert app.active_view == "gpus"
        await pilot.press("escape", "slash", "1", "2")
        assert app.active_view == "gpus"
        assert app.widget("#filter", Input).value == "12"
        await pilot.press("escape", "a", "down", "enter")
        assert isinstance(app.screen, ActionDialog)
        assert app.screen.operation == "reserve"
        assert app.screen.target["uuid"] == "GPU-exact"
        assert not client.calls
        await pilot.press("escape", "escape", "1", "a")
        assert app.screen is app._main and app._main.has_class("details-open")
        assert app.active_view == "jobs"
        assert not app.widget("#detail-actions", OptionList).display
        await pilot.press("escape", "q")


@pytest.mark.parametrize("size", [(120, 35), (80, 24), (45, 14)])
async def test_enter_shows_gpu_action_inline_without_executing_it(size):
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter")
        actions = app.widget("#detail-actions", OptionList)
        assert app.screen is app._main and app._main.has_class("details-open")
        assert actions.option_count == 1
        assert actions.get_option_at_index(0).id == "reserve"
        assert str(actions.get_option_at_index(0).prompt).startswith("> ")
        assert app.focused is actions
        assert 0 <= actions.region.y < actions.region.bottom <= size[1]
        assert not app.query_one("#actions").display
        assert not client.calls
        await pilot.press("tab")
        assert app.focused.id == "scroll-summary"
        await pilot.press("shift+tab")
        assert app.focused is actions
        await pilot.press("enter")
        assert isinstance(app.screen, ActionDialog)
        assert app.screen.operation == "reserve"
        assert app.screen.target["uuid"] == "GPU-exact"
        assert app.screen.review_payload is None and not client.calls
        await pilot.press("escape")
        assert app._main.has_class("details-open")
        await pilot.press("escape")
        assert not app._main.has_class("details-open")
        assert app.focused.id == "table-gpus"


async def test_inline_autodl_actions_use_arrows_and_mouse_and_keep_confirmation():
    client = FakeClient()
    client.snapshot["sites"][2]["state"] = "offline"
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("6", "enter")
        actions = app.widget("#detail-actions", OptionList)
        assert [actions.get_option_at_index(i).id for i in range(actions.option_count)] == [
            "release",
            "unregister",
        ]
        await pilot.press("down", "enter")
        assert isinstance(app.screen, ActionDialog)
        assert app.screen.operation == "unregister"
        assert not client.calls
        await pilot.press("escape")
        await pilot.click("#detail-actions", offset=(2, 0))
        assert isinstance(app.screen, ActionDialog)
        assert app.screen.operation == "release"
        assert not client.calls
        await confirm_action(pilot, app)
        assert client.calls == [("release", "autodl-1", {"force": False})]


async def test_detail_action_target_does_not_follow_background_selection_or_disappearances():
    client = FakeClient()
    client.snapshot["gpus"].append(
        {"site_id": "lab1", "uuid": "GPU-other", "index": 4, "healthy": True}
    )
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter")
        # Reordering and a queued table highlight must not retarget open details.
        client.snapshot["gpus"].reverse()
        app.refresh_status()
        await pilot.pause()
        app.widget("#table-gpus", DataTable).move_cursor(row=0)
        await pilot.pause()
        assert app.selected()["uuid"] == "GPU-exact"
        await pilot.press("a", "enter")
        assert app.screen.target["uuid"] == "GPU-exact"
        await pilot.press("escape")
        client.snapshot["gpus"] = [client.snapshot["gpus"][0]]
        app.refresh_status()
        await pilot.pause()
        assert app.selected()["uuid"] == "GPU-exact"
        assert not app.widget("#detail-actions", OptionList).display
        assert "Actions unavailable" in str(app.widget("#detail-title", Static).render())
        await pilot.press("a", "enter")
        assert app.screen is app._main and not client.calls
        await pilot.press("escape")
        assert app.selected()["uuid"] == "GPU-other"


async def test_job_details_remain_on_same_job_after_it_finishes():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "3")
        client.snapshot["jobs"][0]["state"] = "succeeded"
        app.refresh_status()
        await pilot.pause()
        assert app.selected()["job_id"] == "job_abcd1111"
        assert app.selected()["state"] == "succeeded"
        assert "job_abcd1111" in str(app.widget("#detail-logs", Static).render())
        assert not app.widget("#detail-actions", OptionList).display
        await pilot.press("escape")
        assert app.selected()["job_id"] == "job_abcd2222"


async def test_inline_action_refresh_does_not_replace_highlighted_operation():
    client = FakeClient()
    client.snapshot["sites"][2]["state"] = "offline"
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("6", "enter", "down")
        actions = app.widget("#detail-actions", OptionList)
        assert actions.get_option_at_index(actions.highlighted).id == "unregister"
        app.refresh_status()
        await pilot.pause()
        assert actions.get_option_at_index(actions.highlighted).id == "unregister"
        client.snapshot["sites"][2]["state"] = "ready"
        app.refresh_status()
        await pilot.pause()
        assert actions.highlighted is None
        await pilot.press("enter")
        assert app.screen is app._main and not client.calls
        await pilot.press("down", "enter")
        assert isinstance(app.screen, ActionDialog)
        assert app.screen.operation == "release"


@pytest.mark.parametrize("size", [(120, 35), (60, 20)])
async def test_left_right_switch_main_tabs_from_tables_and_tab_strip(size):
    app = Workbench(FakeClient(), interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        views = ("jobs", "gpus", "sites", "assets", "reservations", "autodl")
        for view in (*views[1:], "jobs"):
            await pilot.press("right")
            assert app.active_view == view
            table = app.widget(f"#table-{view}", DataTable)
            if table.display:
                assert app.focused is table
            else:
                assert isinstance(app.focused, Tabs)
        for view in (*reversed(views[1:]), "jobs"):
            await pilot.press("left")
            assert app.active_view == view
            table = app.widget(f"#table-{view}", DataTable)
            if table.display:
                assert app.focused is table
            else:
                assert isinstance(app.focused, Tabs)
        # Native Tabs also handles arrows: a single key must not switch twice.
        app.widget("#views", TabbedContent).query_one(Tabs).focus()
        await pilot.press("right")
        assert app.active_view == "gpus"
        await pilot.press("question_mark", "left", "right")
        assert isinstance(app.screen, ChoiceDialog)
        assert app.active_view == "gpus"
        await pilot.press("escape", "q")


async def test_left_right_switch_only_visible_detail_tabs():
    app = Workbench(FakeClient(), interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        for key, name in (
            ("right", "attempts"),
            ("right", "logs"),
            ("right", "summary"),
            ("left", "logs"),
            ("left", "attempts"),
            ("left", "summary"),
        ):
            await pilot.press(key)
            assert app.widget("#detail-tabs", TabbedContent).active == f"pane-{name}"
            assert app.focused.id == f"scroll-{name}"
            assert app.active_view == "jobs"
        await pilot.press("escape", "right", "enter", "left", "right")
        assert app.active_view == "gpus"
        assert app._main.has_class("details-open")
        assert app.widget("#detail-tabs", TabbedContent).active == "pane-summary"
        await pilot.press("escape", "right")
        assert app.active_view == "sites"


async def test_tab_arrows_preserve_filter_editing_and_keyboard_horizontal_scroll():
    app = Workbench(FakeClient(), interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("slash", "a", "b", "c")
        field = app.widget("#filter", Input)
        field.cursor_position = 1
        await pilot.press("left")
        assert field.cursor_position == 0
        await pilot.press("right", "shift+right")
        assert field.cursor_position == 2
        assert app.focused is field and app.active_view == "jobs"
        await pilot.press("escape")
        table = app.widget("#table-jobs", DataTable)
        table.add_column("Wide content", width=200)
        await pilot.pause()
        await pilot.press("shift+right")
        assert table.scroll_x > 0 and app.active_view == "jobs"
        await pilot.press("shift+left")
        assert table.scroll_x == 0 and app.active_view == "jobs"


async def test_release_requires_confirmation_and_pins_full_id():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.action_view("autodl")
        await pilot.pause()
        app.open_action("release")
        await pilot.pause()
        assert isinstance(app.screen, ActionDialog)
        assert app.screen.review_payload is None
        app.selections["autodl"] = "another-site"
        await confirm_action(pilot, app)
        assert client.calls == [("release", "autodl-1", {"force": False})]
        app.open_action("release", {"site_id": "autodl-1"})
        await pilot.pause()
        await pilot.press("escape")
        assert len(client.calls) == 1


async def test_forced_reservation_uses_uuid_and_explicit_form():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 45)) as pilot:
        await pilot.pause()
        app.widget("#views", TabbedContent).active = "gpus"
        await pilot.pause()
        app.open_action("reserve")
        await pilot.pause()
        app.screen.query_one("#seconds", Input).value = "bad"
        await pilot.pause()
        await pilot.click("#apply")
        await pilot.pause()
        assert not client.calls and isinstance(app.screen, ActionDialog)
        app.screen.query_one("#seconds", Input).value = "1800"
        app.screen.query_one("#reason", Input).value = "testing"
        app.screen.query_one("#force", Checkbox).value = True
        await pilot.pause(0.3)
        await confirm_action(pilot, app)
        assert client.calls, str(app.screen.query_one("#dialog-error", Static).render())
        call = client.calls[0]
        assert call[1]["gpu_uuid"] == "GPU-exact" and call[1]["gpu_index"] is None
        assert call[1]["force"] is True
        assert call[2]["idempotency_key"].startswith("workbench-")


async def test_network_failure_retains_rows_sanitizes_error_and_ui_stays_responsive():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        client.fail = True
        app.refresh_status()
        await pilot.pause()
        status = str(app.widget("#status", Static).render())
        assert "STALE" in status and "DO_NOT_DISPLAY_SECRET" not in status
        assert app.widget("#table-jobs", DataTable).row_count == 2
        client.block = asyncio.Event()
        app.refresh_status()
        await pilot.pause()
        app.widget("#filter", Input).value = "Beta"
        await pilot.pause()
        assert app.widget("#table-jobs", DataTable).row_count == 1
        await pilot.press("escape")
        await pilot.press("q")


@pytest.mark.parametrize(
    ("column", "value", "color"),
    [
        ("Owner", "free", "green"),
        ("Owner", "warm", "cyan"),
        ("Owner", "leased", "cyan"),
        ("Owner", "reserved", "yellow"),
        ("Owner", "foreign", "yellow"),
        ("Owner", "leased+reserved", "yellow"),
        ("Owner", "warm+foreign", "yellow"),
        ("Owner", "leased+reserved+foreign", "yellow"),
        ("Owner", "inactive", ""),
        ("Owner", "unknown", ""),
        ("Util", "0%", "green"),
        ("Util", "29.9%", "green"),
        ("Util", "30%", "yellow"),
        ("Util", "79.9%", "yellow"),
        ("Util", "80%", "red"),
        ("Util", "100%", "red"),
        ("Util", "101%", ""),
        ("Util", "-1%", ""),
        ("Util", "nan%", ""),
        ("Util", "-%", ""),
        ("Name", "foreign", ""),
        ("Name", "100%", ""),
    ],
)
def test_gpu_owner_and_utilization_colors(column, value, color):
    from rich.segment import Segment
    from rich.style import Style
    from textual.color import Color

    from topodrome.client.workbench import TerminalDefaults, status_cell

    cell = status_cell(value, column)
    assert cell.plain == value and cell.style == color
    segments = TerminalDefaults(no_color=True).apply(
        [Segment(cell.plain, Style.parse(cell.style))], Color(0, 0, 0)
    )
    assert segments[0].style.color.is_default
    assert segments[0].style.bgcolor.is_default


async def test_gpu_colors_update_in_place_after_refresh():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("2")
        table = app.widget("#table-gpus", DataTable)
        row = table.get_row("lab1/GPU-exact")
        assert row[4].style == "green" and row[5].style == "green"
        client.snapshot["gpus"][0].update(utilization_percent=95, active_lease=True)
        app.refresh_status()
        await pilot.pause()
        row = table.get_row("lab1/GPU-exact")
        assert row[4].style == "red" and row[5].style == "cyan"
        # The selected row's cursor CSS must not overwrite semantic colors.
        assert app.focused is table
        rendered = table.render_line(table.header_height)
        for value, color in (("95%", "red"), ("leased", "cyan")):
            assert any(
                value in segment.text and segment.style.color.name == color for segment in rendered
            )
        client.snapshot["gpus"][0].update(utilization_percent=50, foreign_process_count=1)
        app.refresh_status()
        await pilot.pause()
        row = table.get_row("lab1/GPU-exact")
        assert row[4].style == "yellow" and row[5].style == "yellow"


@pytest.mark.parametrize("force", [False, True])
async def test_autodl_release_is_confirmed_and_force_defaults_off(force):
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.widget("#views", TabbedContent).active = "autodl"
        await pilot.pause()
        app.open_action("release")
        await pilot.pause()
        assert not app.screen.query_one("#force", Checkbox).value
        app.screen.query_one("#force", Checkbox).value = force
        await confirm_action(pilot, app)
        assert client.calls == [("release", "autodl-1", {"force": force})]


async def test_unreserve_dispatches_exact_target():
    client = FakeClient()

    async def reservations(**kwargs):
        return [
            {
                "reservation_id": "reservation_exact",
                "site_id": "lab1",
                "gpu_uuid": "GPU-exact",
                "gpu_index": 3,
                "state": "ready",
                "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                "reason": "Maintenance",
            }
        ]

    client.gpu_reservations = reservations
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 45)) as pilot:
        await pilot.pause()
        await pilot.press("5")
        await pilot.pause()
        await pilot.press("enter")
        assert app.widget("#detail-actions", OptionList).get_option_at_index(1).id == "unreserve"
        assert not client.calls
        await pilot.press("down", "enter")
        assert isinstance(app.screen, ActionDialog)
        await confirm_action(pilot, app)
        assert client.calls == [("unreserve", "reservation_exact")]


def active_reservation(reservation_id="reservation_exact", site="lab1"):
    return {
        "reservation_id": reservation_id,
        "site_id": site,
        "gpu_uuid": "GPU-exact",
        "gpu_index": 3,
        "state": "ready",
        "reason": "Maintenance [literal]",
        "requested_by": "operator",
        "force": True,
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }


@pytest.mark.parametrize("size", [(100, 30), (45, 14)])
@pytest.mark.parametrize("interaction", ["keyboard", "mouse"])
async def test_gpu_reserve_extend_cancel_cycle_stays_usable(size, interaction):
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter", "enter")
        assert app.screen.operation == "reserve"
        await confirm_action(pilot, app)
        actions = app.widget("#detail-actions", OptionList)
        assert app.available_actions() == {"extend", "unreserve"}
        assert app.focused is actions and actions.highlighted is not None
        reservation_id = client.reservations[0]["reservation_id"]
        for operation, row in (("extend", 0), ("unreserve", 1)):
            if interaction == "mouse":
                await pilot.click("#detail-actions", offset=(2, row))
            else:
                if row:
                    await pilot.press("down")
                await pilot.press("enter")
            assert isinstance(app.screen, ActionDialog)
            assert app.screen.operation == operation
            assert app.screen.target["reservation_id"] == reservation_id
            await confirm_action(pilot, app)
            assert app.focused is actions and actions.highlighted is not None
        assert [call[0] for call in client.calls] == ["reserve", "extend", "unreserve"]
        assert app.available_actions() == {"reserve"}
        await pilot.press("enter")
        assert app.screen.operation == "reserve"
        assert len(client.calls) == 3


async def test_open_reserved_gpu_before_metadata_arrives_focuses_cancel_actions():
    client = FakeClient()
    client.snapshot["gpus"][0]["operator_reserved"] = True
    client.reservations = [active_reservation()]
    client.reservation_gate = asyncio.Event()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter")
        assert not app.available_actions()
        client.reservation_gate.set()
        await pilot.pause()
        actions = app.widget("#detail-actions", OptionList)
        assert app.focused is actions and actions.highlighted == 0
        await pilot.press("down", "enter")
        assert app.screen.operation == "unreserve"
        assert app.screen.target["reservation_id"] == "reservation_exact"
        assert not client.calls


async def test_reservations_tab_extend_cancel_and_empty_list_remain_navigable():
    client = FakeClient()
    client.reservations = [active_reservation()]
    app = Workbench(client, interval=60)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        await pilot.press("5", "enter", "enter")
        assert app.screen.operation == "extend"
        await confirm_action(pilot, app)
        assert app.focused.id == "detail-actions"
        await pilot.press("down", "enter")
        assert app.screen.operation == "unreserve"
        await confirm_action(pilot, app)
        assert not app.available_actions()
        assert app.focused.id == "scroll-summary"
        await pilot.press("escape")
        assert isinstance(app.focused, Tabs)
        await pilot.press("2", "enter", "enter")
        assert app.screen.operation == "reserve"
        assert [call[0] for call in client.calls] == ["extend", "unreserve"]


@pytest.mark.parametrize("operation", ["reserve", "extend", "unreserve", "release", "unregister"])
async def test_management_toolbar_keeps_a_visible_keyboard_target(operation):
    client = FakeClient()
    if operation in {"extend", "unreserve"}:
        client.reservations = [active_reservation()]
        client.snapshot["gpus"][0]["operator_reserved"] = True
    if operation == "unregister":
        client.snapshot["sites"][2]["state"] = "offline"
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        await pilot.press("6" if operation in {"release", "unregister"} else "2")
        await pilot.click(f"#action-{operation}")
        assert app.screen.operation == operation
        await confirm_action(pilot, app)
        if operation == "unregister":
            assert app.focused.id == "page-register"
        else:
            next_action = {"reserve": "extend", "unreserve": "reserve"}.get(operation, operation)
            assert app.focused.id == f"action-{next_action}"
            await pilot.press("enter")
            assert app.screen.operation == next_action
        assert len(client.calls) == 1


async def test_autodl_details_release_and_removal_remain_navigable():
    client = FakeClient()
    client.snapshot["sites"][2]["state"] = "offline"
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("6", "enter", "enter")
        await confirm_action(pilot, app)
        assert app.focused.id == "detail-actions"
        await pilot.press("down", "enter")
        assert app.screen.operation == "unregister"
        await confirm_action(pilot, app)
        assert app.focused.id == "scroll-summary"
        assert not app.available_actions()
        await pilot.press("escape", "2")
        assert app.active_view == "gpus"
        assert [call[0] for call in client.calls] == ["release", "unregister"]


@pytest.mark.parametrize("replace", [False, True])
async def test_stale_actions_recover_only_for_the_same_reservation(replace):
    client = FakeClient()
    client.snapshot["gpus"][0]["operator_reserved"] = True
    client.reservations = [active_reservation()]
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter", "down")
        actions = app.widget("#detail-actions", OptionList)
        assert actions.get_option_at_index(actions.highlighted).id == "unreserve"
        client.reservations_error = True
        app.refresh_extras()
        await pilot.pause()
        assert not app.available_actions()
        assert app.focused.id == "scroll-summary"
        client.reservations_error = False
        if replace:
            client.reservations = [active_reservation("replacement")]
        app.refresh_extras()
        await pilot.pause()
        if replace:
            assert actions.highlighted is None
            assert app.focused.id == "scroll-summary"
        else:
            assert app.focused is actions
            assert actions.get_option_at_index(actions.highlighted).id == "unreserve"
            await pilot.press("enter")
            assert app.screen.target["reservation_id"] == "reservation_exact"
        assert not client.calls


async def test_queued_action_event_cannot_target_a_replacement_reservation():
    client = FakeClient()
    client.reservations = [active_reservation()]
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter", "down")
        actions = app.widget("#detail-actions", OptionList)
        event = OptionList.OptionSelected(actions, actions.get_option_at_index(1), 1)
        client.reservations = [active_reservation("replacement")]
        app.refresh_extras()
        await pilot.pause()
        app.detail_action_selected(event)
        await pilot.pause()
        assert app.screen is app._main and not client.calls
        await pilot.press("down", "down", "enter")
        assert app.screen.target["reservation_id"] == "replacement"


async def test_old_inflight_status_is_discarded_and_refetched_after_cancel():
    client = FakeClient()
    client.snapshot["gpus"][0]["operator_reserved"] = True
    client.reservations = [active_reservation()]
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter", "down", "enter")
        gate = asyncio.Event()
        original_status = client.system_status
        reads = []

        async def delayed_status():
            snapshot = await original_status()
            reads.append(snapshot)
            if len(reads) == 1:
                await gate.wait()
            return snapshot

        client.system_status = delayed_status
        app.refresh_status()
        await pilot.pause()
        await confirm_action(pilot, app)
        assert "reserve" not in app.available_actions()
        gate.set()
        await pilot.pause()
        assert len(reads) == 2
        assert app.available_actions() == {"reserve"}
        assert app.focused.id == "detail-actions"
        await pilot.press("enter")
        assert app.screen.operation == "reserve"
        assert client.calls == [("unreserve", "reservation_exact")]


async def test_switching_away_from_slow_assets_fetches_gpu_reservations_immediately():
    client = FakeClient()
    client.snapshot["gpus"][0]["operator_reserved"] = True
    client.reservations = [active_reservation()]
    gate = asyncio.Event()

    async def slow_assets(**kwargs):
        await gate.wait()
        return []

    client.assets = slow_assets
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("4", "2", "enter")
        assert not app.available_actions()
        gate.set()
        await pilot.pause()
        assert app.available_actions() == {"extend", "unreserve"}
        assert app.focused.id == "detail-actions"
        await pilot.press("down", "enter")
        assert app.screen.operation == "unreserve"
        assert not client.calls


@pytest.mark.parametrize("operation", ["reserve", "extend", "release"])
async def test_pending_management_request_does_not_steal_focus_after_leaving(operation):
    client = FakeClient()
    if operation == "extend":
        client.reservations = [active_reservation()]
        client.snapshot["gpus"][0]["operator_reserved"] = True
    gate = asyncio.Event()
    method = {
        "reserve": "reserve_gpu",
        "extend": "extend_gpu_reservation",
        "release": "release_autodl",
    }[operation]
    original = getattr(client, method)

    async def delayed(*args, **kwargs):
        await gate.wait()
        return await original(*args, **kwargs)

    setattr(client, method, delayed)
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("6" if operation == "release" else "2", "enter", "enter")
        await confirm_action(pilot, app)
        assert app._action_busy
        await pilot.press("escape")
        focus = app.focused
        assert isinstance(focus, DataTable)
        gate.set()
        await pilot.pause()
        assert not app._action_busy
        assert app.focused is focus and app._focus_return is None
        assert len(client.calls) == 1


async def test_page_reload_button_remains_usable_after_confirmation():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("3")
        await pilot.click("#page-reload")
        await confirm_action(pilot, app)
        assert app.focused.id == "page-reload"
        await pilot.press("enter")
        assert app.screen.operation == "reload"
        await pilot.press("escape")
        assert app.focused.id == "page-reload"
        assert client.calls == [("reload",)]


@pytest.mark.parametrize("operation", ["extend", "unreserve"])
@pytest.mark.parametrize("size", [(80, 24), (45, 14)])
async def test_click_reserved_gpu_can_extend_or_cancel_without_other_targets(operation, size):
    client = FakeClient()
    client.snapshot["gpus"][0]["operator_reserved"] = True
    original = active_reservation()
    client.reservations = [active_reservation("other_site", "another-lab"), original]
    old_expiry = original["expires_at"]
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("2")
        await pilot.pause()
        # The first three cells are now the process disclosure label; click
        # the GPU's data cell to open its management details.
        await pilot.click("#table-gpus", offset=(5, 1))
        assert app._main.has_class("details-open")
        assert app.available_actions() == {"extend", "unreserve"}
        assert "left" in str(app.widget("#detail-context", Static).render())
        assert "Maintenance [literal]" in str(app.widget("#detail-context", Static).render())
        actions = app.widget("#detail-actions", OptionList)
        assert actions.region.bottom <= size[1]
        if operation == "unreserve":
            await pilot.press("down")
        await pilot.press("enter")
        assert isinstance(app.screen, ActionDialog)
        assert app.screen.target["reservation_id"] == "reservation_exact"
        assert not list(app.screen.query("#force"))
        if operation == "extend":
            app.screen.query_one("#seconds", Input).value = "1h"
            await tab_to(pilot, app, "apply")
            await pilot.press("enter")
            await pilot.pause()
            assert str(app.screen.query_one("#review", Static).render()).startswith("New expiry")
            assert not app.screen.query_one("#reservation-info").display
            assert app.screen.query_one("#apply").region.bottom <= size[1]
            assert app.screen.query_one("#apply").content_region.height == 1
            assert not client.calls and app.screen.focused.id == "apply"
            await pilot.press("enter")
            await pilot.pause()
            assert client.calls == [
                (
                    "extend",
                    "reservation_exact",
                    {
                        "expected_expires_at": old_expiry.replace("+00:00", "Z"),
                        "additional_seconds": 3600,
                    },
                )
            ]
            assert (
                original["expires_at"]
                == (datetime.fromisoformat(old_expiry) + timedelta(hours=1)).isoformat()
            )
            assert original["force"] is True
        else:
            await confirm_action(pilot, app)
            assert client.calls == [("unreserve", "reservation_exact")]
            assert app.available_actions() == {"reserve"}
            assert app.focused is actions
            assert actions.highlighted == 0
            assert str(actions.get_option_at_index(0).prompt).startswith("> ")
            await pilot.press("enter")
            assert isinstance(app.screen, ActionDialog)
            assert app.screen.operation == "reserve"
            assert app.screen.target["uuid"] == "GPU-exact"
            assert client.calls == [("unreserve", "reservation_exact")]
        assert client.reservations[0]["reservation_id"] == "other_site"


@pytest.mark.parametrize("delayed_read", ["status", "reservations"])
async def test_cancel_gpu_restores_actions_only_after_both_reads(delayed_read):
    client = FakeClient()
    client.snapshot["gpus"][0]["operator_reserved"] = True
    client.reservations = [active_reservation()]
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter", "down", "enter")
        gate = asyncio.Event()
        if delayed_read == "status":
            client.block = gate
        else:
            client.reservation_gate = gate
        await confirm_action(pilot, app)
        actions = app.widget("#detail-actions", OptionList)
        assert client.calls == [("unreserve", "reservation_exact")]
        assert not app._action_busy
        assert app._focus_return is not None
        assert "reserve" not in app.available_actions()
        gate.set()
        await pilot.pause()
        assert app._focus_return is None
        assert app.focused is actions and actions.highlighted == 0
        assert actions.get_option_at_index(0).id == "reserve"
        await pilot.press("enter")
        assert app.screen.operation == "reserve"
        assert app.screen.target["uuid"] == "GPU-exact"
        assert client.calls == [("unreserve", "reservation_exact")]


@pytest.mark.parametrize("navigation", ["summary", "mouse", "list", "help"])
async def test_cancel_gpu_refresh_does_not_steal_focus_after_navigation(navigation):
    client = FakeClient()
    client.snapshot["gpus"][0]["operator_reserved"] = True
    client.reservations = [active_reservation()]
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter", "down", "enter")
        client.reservation_gate = asyncio.Event()
        await confirm_action(pilot, app)
        if navigation == "mouse":
            # Pilot.click bypasses App.on_event; send real driver input here
            # to cover the focus-return cancellation before mouse forwarding.
            region = app.query_one("#scroll-summary").region
            for event_type in (events.MouseDown, events.MouseUp):
                app.post_message(
                    event_type(None, region.x + 2, region.y + 1, 0, 0, 1, False, False, False)
                )
                await pilot.pause()
            assert app.focused.id == "scroll-summary"
        else:
            await pilot.press(
                {"summary": "right", "list": "escape", "help": "question_mark"}[navigation]
            )
        if navigation == "help":
            assert isinstance(app.screen, ChoiceDialog)
        assert app._focus_return is None
        focus = app.focused
        screen = app.screen
        client.reservation_gate.set()
        await pilot.pause()
        assert app._focus_return is None
        assert app.screen is screen and app.focused is focus
        assert client.calls == [("unreserve", "reservation_exact")]
        if navigation in {"summary", "mouse"}:
            actions = app.widget("#detail-actions", OptionList)
            assert actions.highlighted is None
            # Returning deliberately must select an actionable row, not an
            # invisible cursor left over from clearing the old reservation.
            await pilot.press("shift+tab")
            assert app.focused is actions and actions.highlighted == 0
            await pilot.press("enter")
            assert app.screen.operation == "reserve"


async def test_reserved_gpu_does_not_guess_actions_when_reservation_reads_fail():
    client = FakeClient()
    client.snapshot["gpus"][0]["operator_reserved"] = True
    client.reservations = [active_reservation()]
    client.reservation_gate = asyncio.Event()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter")
        assert not app.available_actions()
        assert "loading" in str(app.widget("#detail-context", Static).render())
        client.reservations_error = True
        client.reservation_gate.set()
        await pilot.pause()
        assert not app.available_actions()
        assert "DO_NOT_DISPLAY_SECRET" not in app._notice_text
        client.reservations_error = False
        app.refresh_extras()
        await pilot.pause()
        assert app.available_actions() == {"extend", "unreserve"}
        client.reservations_error = True
        app.refresh_extras()
        await pilot.pause()
        assert not app.available_actions()
        assert "stale" in str(app.widget("#detail-context", Static).render())
        assert not client.calls


async def test_reservation_replacement_never_retargets_an_open_cancel_form():
    client = FakeClient()
    client.snapshot["gpus"][0]["operator_reserved"] = True
    client.reservations = [active_reservation("old_reservation")]
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("2")
        await pilot.pause()
        await pilot.press("enter", "down", "enter")
        assert app.screen.target["reservation_id"] == "old_reservation"
        client.reservations = [active_reservation("new_reservation")]
        app.refresh_extras()
        await pilot.pause()
        await confirm_action(pilot, app)
        assert client.calls == [("unreserve", "old_reservation")]
        assert client.reservations[0]["reservation_id"] == "new_reservation"
        assert app._focus_return is None
        assert app.widget("#detail-actions", OptionList).highlighted is None


async def test_uncertain_cancel_does_not_restore_reserve_focus():
    client = FakeClient()
    client.snapshot["gpus"][0]["operator_reserved"] = True
    client.reservations = [active_reservation()]
    cancel = client.cancel_gpu_reservation

    async def uncertain_cancel(reservation):
        await cancel(reservation)
        raise RuntimeError("DO_NOT_DISPLAY_SECRET")

    client.cancel_gpu_reservation = uncertain_cancel
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter", "down", "enter")
        await confirm_action(pilot, app)
        assert app.available_actions() == {"reserve"}
        assert app._focus_return is None
        assert app.widget("#detail-actions", OptionList).highlighted is None
        assert "not confirmed" in app._notice_text
        assert "DO_NOT_DISPLAY_SECRET" not in app._notice_text
        assert client.calls == [("unreserve", "reservation_exact")]


async def test_action_failure_is_not_retried_or_exposed_and_quit_waits():
    client = FakeClient()
    pending = asyncio.Event()

    async def fail(site, **kwargs):
        client.calls.append(("release", site))
        await pending.wait()
        raise RuntimeError("DO_NOT_DISPLAY_SECRET")

    client.release_autodl = fail
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.action_confirmed(("release", {"site_id": "site_exact"}, {}, {}, "key"))
        await pilot.pause()
        app.action_safe_quit()
        assert app._action_busy and app.is_running
        pending.set()
        await pilot.pause()
        assert not app._action_busy
        assert "not confirmed" in app._notice_text
        assert "DO_NOT_DISPLAY_SECRET" not in app._notice_text
        app.action_refresh_now()
        await pilot.pause()
        assert client.calls == [("release", "site_exact")]


async def test_compact_terminal_keeps_table_and_can_switch_to_details():
    app = Workbench(FakeClient(), interval=60, no_color=True)
    async with app.run_test(size=(45, 14)) as pilot:
        await pilot.pause()
        assert app.widget("#table-jobs", DataTable).size.height >= 3
        assert not app.widget("#inspector", Vertical).display
        app.action_details()
        await pilot.pause()
        assert app.widget("#views", TabbedContent).display is False
        assert app._main.has_class("details-open")
        app.action_search()
        await pilot.pause()
        assert app.widget("#filter", Input).display


async def test_direct_register_masks_password_and_no_job_asset_writes():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause()
        for view in ("jobs", "assets"):
            app.action_view(view)
            await pilot.pause()
            app.action_context_action()
            await pilot.pause()
            assert app.screen is app._main and app._main.has_class("details-open")
            assert not app.widget("#detail-actions", OptionList).display
            await pilot.press("escape")
        app.open_action("cancel")
        assert app.screen is app._main
        await pilot.press("ctrl+p")
        assert app.screen is app._main
        await pilot.press("6")
        await pilot.click("#page-register")
        assert isinstance(app.screen, ActionDialog)
        assert app.screen.query_one("#password", Input).password
        assert app.screen.review_payload is None
        app.screen.query_one("#ssh", Input).value = "ssh -p 12345 root@connect.bjb2.seetacloud.com"
        app.screen.query_one("#password", Input).value = " secret with spaces "
        await confirm_action(pilot, app)
        assert client.calls == [
            ("register", "ssh -p 12345 root@connect.bjb2.seetacloud.com", " secret with spaces ")
        ]
        assert "autodl-new" in app._notice_text
        assert "MUST_NOT_RENDER" not in app._notice_text
        assert "secret" not in app._notice_text
        assert app.focused.id == "page-register"
        await pilot.press("enter")
        assert app.screen.operation == "register"
        await pilot.press("escape")
        assert app.focused.id == "page-register"
        assert len(client.calls) == 1


async def test_policy_reload_removal_and_models_report():
    client = FakeClient()
    client.snapshot["sites"][2]["state"] = "offline"
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.action_view("autodl")
        await pilot.pause()
        await pilot.click("#action-unregister")
        assert isinstance(app.screen, ActionDialog)
        await confirm_action(pilot, app)
        assert client.calls == [("unregister", "autodl-1")]
        app.command_chosen("reload")
        await pilot.pause()
        assert isinstance(app.screen, ActionDialog)
        assert not any(call[0] == "reload" for call in client.calls)
        await confirm_action(pilot, app)
        assert client.calls[-1] == ("reload",)
        app.command_chosen("models")
        await pilot.pause()
        assert client.calls[-1] == ("reload",)
        assert isinstance(app.screen, ChoiceDialog)
        assert "llama" in app.screen.title_text.plain
        assert '{"' not in app.screen.title_text.plain


@pytest.mark.parametrize(
    "value, expected", [("30m", 1800), ("12h", 43200), ("1d", 86400), ("30", 30)]
)
def test_management_duration_units(value, expected):
    from topodrome.client.workbench import duration_seconds

    assert duration_seconds(value) == expected


async def test_empty_states_keep_management_accessible_without_palette():
    client = FakeClient()
    client.snapshot["jobs"] = []
    app = Workbench(client, interval=60)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert app.widget("#empty-jobs", Static).display
        assert not app.widget("#table-jobs", DataTable).display
        await pilot.press("ctrl+p")
        assert app.screen is app._main
        await pilot.press("2")
        assert app.active_view == "gpus"
        assert app.widget("#table-gpus", DataTable).display
        await pilot.press("q")


@pytest.mark.parametrize("size", [(80, 24), (60, 20)])
async def test_compact_reservation_keyboard_review_edit_and_confirm(size):
    from topodrome.client.workbench import PlainCheckbox

    client = FakeClient()
    app = Workbench(client, interval=60, no_color=True)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("2")
        app.open_action("reserve")
        await pilot.pause()
        assert app.screen.focused.id == "seconds"
        await pilot.press("1", "2", "h", "enter")
        assert app.screen.query_one("#seconds", Input).value == "12h"
        assert app.screen.focused.id == "reason"
        await pilot.press("enter")
        force = app.screen.query_one("#force", PlainCheckbox)
        assert app.screen.focused is force
        assert "[ ]" in force.render().plain
        await pilot.press("space")
        assert "[x]" in force.render().plain
        await tab_to(pilot, app, "apply")
        await pilot.press("enter")
        assert not client.calls
        assert app.screen.review_payload[2]["duration_seconds"] == 43200
        assert "FORCED EVICTION" in str(app.screen.query_one("#review", Static).render())
        assert app.screen.focused.id == "apply"
        # Editing remains an explicit option; confirmation is the default.
        await pilot.press("escape")
        assert app.screen.review_payload is None
        assert not client.calls
        assert app.screen.query_one("#seconds", Input).value == "12h"
        await tab_to(pilot, app, "force")
        await pilot.press("space")
        assert not force.value
        await confirm_action(pilot, app)
        assert client.calls[0][1]["force"] is False
        assert client.calls[0][1]["reason"] == "Manual reservation"


def test_detail_pages_are_curated_and_terminal_safe():
    from topodrome.client.workbench_details import attempts_page, detail_page

    gpu = {
        "index": 3,
        "site_id": "lab1",
        "uuid": "GPU-exact",
        "healthy": True,
        "free_memory_bytes": 60 * 1024**3,
        "total_memory_bytes": 80 * 1024**3,
        "internal_payload": {"DO_NOT_DUMP": True},
    }
    text = detail_page("gpus", gpu).plain
    assert "Current resources" in text and "20.0 / 80.0 GiB" in text
    assert "GPU UUID" in text and "GPU-exact" in text
    assert "internal_payload" not in text and "DO_NOT_DUMP" not in text
    asset = {
        "name": "[red]literal",
        "tags": ["project:interp", "kind:lens"],
        "files": [{"relative_path": "out.pt", "size_bytes": 1024, "sha256": "DO_NOT_DUMP"}],
    }
    text = detail_page("assets", asset).plain
    assert "out.pt  (1,024 bytes)" in text and "DO_NOT_DUMP" not in text
    assert "[red]literal" in text
    assert "project:interp, kind:lens" in text
    attempts = attempts_page(
        [
            {
                "attempt_id": "attempt_full",
                "attempt_number": 2,
                "state": "failed",
                "reason_code": "user_error",
                "checkpoint_ref": "large/raw/path",
            }
        ]
    ).plain
    assert "Attempt 2" in attempts and "Outcome reason" in attempts
    assert "Available" in attempts and "large/raw/path" not in attempts


async def test_gpu_detail_hides_irrelevant_attempts_and_logs():
    app = Workbench(FakeClient(), interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter", "3")
        assert app.widget("#detail-tabs", TabbedContent).active == "pane-summary"
        assert "Topodrome allocation" in str(app.widget("#detail-summary", Static).render())
        await pilot.press("escape", "1", "enter", "3")
        assert app.widget("#detail-tabs", TabbedContent).active == "pane-logs"


async def test_arrow_only_form_navigation_preserves_text_editing_and_affirmative_default():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("2")
        app.open_action("reserve")
        await pilot.pause()
        await pilot.press("ctrl+enter")
        assert app.screen.review_payload is None and not client.calls
        field = app.screen.query_one("#seconds", Input)
        field.cursor_position = 1
        await pilot.press("left")
        assert app.screen.focused is field and field.cursor_position == 0
        await pilot.press("right")
        assert field.cursor_position == 1
        await pilot.press("down")
        assert app.screen.focused.id == "reason"
        await pilot.press("down", "space")
        assert app.screen.query_one("#force", Checkbox).value
        await pilot.press("down")
        assert app.screen.focused.id == "back"
        await pilot.press("right", "enter")
        assert app.screen.review_payload is not None and not client.calls
        assert app.screen.focused.id == "apply"
        await pilot.press("up")
        assert app.screen.focused.id == "back"
        await pilot.press("right", "enter")
        await pilot.pause()
        assert len(client.calls) == 1 and client.calls[0][1]["force"] is True
        assert app.active_view == "gpus"


async def test_forced_review_waits_for_job_impact_and_can_be_abandoned():
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.press("2")
        app.open_action("reserve")
        await pilot.pause()
        app.screen.query_one("#force", Checkbox).value = True
        client.block = asyncio.Event()
        await tab_to(pilot, app, "apply")
        await pilot.press("enter")
        assert app.screen._impact_pending
        assert app.screen.query_one("#apply").disabled
        app.screen.action_advance()
        assert not client.calls
        await pilot.press("escape")
        assert app.screen.review_payload is None
        client.block.set()
        await pilot.pause()
        assert not app.screen.query_one("#impact", Static).display
        assert not client.calls
        await pilot.press("escape", "q")


@pytest.mark.parametrize("choose_edit", [False, True])
async def test_forced_review_defaults_to_confirm_only_after_check_without_stealing_focus(
    choose_edit,
):
    client = FakeClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("2", "enter", "enter")
        app.screen.query_one("#force", Checkbox).value = True
        client.block = asyncio.Event()
        await tab_to(pilot, app, "apply")
        await pilot.press("enter")
        assert app.screen._impact_pending and app.screen.query_one("#apply").disabled
        assert app.screen.focused.id == "dialog-body"
        await pilot.press("enter")
        assert app.screen.review_payload is not None and not client.calls
        if choose_edit:
            await pilot.press("tab")
            assert app.screen.focused.id == "back"
        client.block.set()
        await pilot.pause()
        assert not app.screen._impact_pending
        assert app.screen.focused.id == ("back" if choose_edit else "apply")
        await pilot.press("enter")
        await pilot.pause()
        if choose_edit:
            assert app.screen.review_payload is None and not client.calls
        else:
            assert len(client.calls) == 1 and client.calls[0][1]["force"] is True


async def test_impact_matches_latest_attempt_exact_gpu_and_warns_for_gang():
    from topodrome.client.workbench_impact import reservation_impact

    client = FakeClient()
    client.snapshot["jobs"] = [
        {
            "job_id": "job_match",
            "attempt_id": "attempt_new",
            "name": "lens training",
            "state": "running",
            "site_id": "lab1",
            "model": "llama",
        },
        {
            "job_id": "job_other",
            "attempt_id": "other_new",
            "name": "NOT_THIS_GPU",
            "state": "running",
            "site_id": "lab1",
        },
        {"job_id": "job_upload", "state": "finalizing", "site_id": "lab1", "gpu_count": 0},
    ]

    async def attempts(job):
        if job == "job_match":
            return [
                {
                    "attempt_id": "attempt_new",
                    "state": "running",
                    "site_id": "lab1",
                    "gpu_uuids": ["GPU-exact", "GPU-other"],
                    "checkpoint_ref": "private/path",
                }
            ]
        assert job == "job_other"
        return [
            {"attempt_id": "old", "state": "failed", "site_id": "lab1", "gpu_uuids": ["GPU-exact"]},
            {
                "attempt_id": "other_new",
                "state": "running",
                "site_id": "lab1",
                "gpu_uuids": ["GPU-not-target"],
            },
        ]

    client.attempts = attempts
    text = (await reservation_impact(client, {"site_id": "lab1", "uuid": "GPU-exact"})).plain
    assert "lens training" in text and "job_match" in text and "running" in text
    assert "whole job" in text and "Checkpoint: reported" in text
    assert "NOT_THIS_GPU" not in text and "private/path" not in text


async def test_impact_failure_does_not_claim_idle_or_leak_errors():
    from topodrome.client.workbench_impact import reservation_impact

    client = FakeClient()
    client.snapshot["gpus"][0]["active_lease"] = True

    async def attempts(_):
        raise RuntimeError("DO_NOT_DISPLAY_SECRET")

    client.attempts = attempts
    text = (await reservation_impact(client, {"site_id": "lab1", "uuid": "GPU-exact"})).plain
    assert "could not be identified" in text and "unavailable" in text
    assert "DO_NOT_DISPLAY_SECRET" not in text
