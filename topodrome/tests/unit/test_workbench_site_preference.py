import asyncio
from copy import deepcopy

import pytest
from textual import events
from textual._xterm_parser import XTermParser
from textual.widgets import Button, DataTable, Input, OptionList, TabbedContent

from tests.unit.test_workbench import FakeClient
from topodrome.client.workbench import ChoiceDialog, Workbench
from topodrome.client.workbench_site_order import SiteOrderDraft

pytestmark = pytest.mark.unit


class PreferenceClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.writes = 0
        self.preference_gate = None
        self.snapshot["policy"] = {
            "status": "active",
            "generation": 7,
            "pending_generation": None,
            "preferred_site_ids": ["sqz-128"],
        }
        self.snapshot["sites"].append({"site_id": "sqz-128", "kind": "lab", "state": "ready"})

    async def set_site_preference(self, request):
        self.writes += 1
        if self.preference_gate:
            await self.preference_gate.wait()
        policy = self.snapshot["policy"]
        if request.expected_generation != policy["generation"]:
            raise RuntimeError("DO_NOT_DISPLAY_SECRET")
        self.calls.append(("site-preference", request.model_dump(mode="json")))
        policy.update(
            generation=policy["generation"] + 1, preferred_site_ids=list(request.preferred_site_ids)
        )
        return deepcopy(policy)


@pytest.mark.parametrize("size", [(120, 40), (60, 20), (45, 14)])
async def test_inline_site_order_keyboard_save_reset_live_rank_and_focus(size):
    client = PreferenceClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("3")
        table = app.widget("#table-sites", DataTable)
        assert app.focused is table
        assert next(iter(app.records["sites"])) == "sqz-128"
        await pilot.press("shift+down")
        expected = ["lab1", "sqz-128", "remote-jupyter-1", "autodl-1"]
        assert app.screen is app._main and app.active_view == "sites"
        assert app._site_order.order == expected
        assert list(app.records["sites"]) == expected
        assert app.selections["sites"] == "sqz-128" and table.cursor_row == 1
        await pilot.press("down", "shift+down", "shift+up", "up")
        assert app._site_order.order == expected
        assert app.selections["sites"] == "sqz-128"
        assert not client.writes
        await pilot.press("enter")
        await pilot.pause()
        assert client.calls == [
            ("site-preference", {"preferred_site_ids": expected, "expected_generation": 7})
        ]
        assert app._site_order is None and app.screen is app._main
        assert app.records["sites"]["lab1"]["preference_rank"] == 1
        assert app.focused is table
        # Enter resumes its normal details behavior after the save.
        await pilot.press("enter")
        assert app._main.has_class("details-open")
        assert app.focused is app.widget("#detail-actions", OptionList)
        await pilot.press("enter")
        assert not app._main.has_class("details-open")
        assert app.screen is app._main and app._site_order is not None
        await pilot.click("#order-automatic")
        assert app._site_order.preferences == () and len(client.calls) == 1
        await pilot.press("enter")
        await pilot.pause()
        assert client.calls[-1] == (
            "site-preference",
            {"preferred_site_ids": [], "expected_generation": 8},
        )
        assert app.focused is table and app._site_order is None
        assert all(row["preference_rank"] is None for row in app.records["sites"].values())


@pytest.mark.parametrize("open_with", ["space", "mouse"])
async def test_inline_order_clears_filter_uses_all_sites_cancel_never_writes(open_with):
    client = PreferenceClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("3", "slash", "o")
        assert isinstance(app.focused, Input) and app.focused.value == "o"
        assert app._site_order is None
        app.focused.value = "no-match"
        await pilot.press("enter")
        await pilot.pause()
        assert not app.records["sites"]
        await pilot.press("space")
        assert app._site_order is None
        await pilot.press("slash")
        app.focused.value = "sqz"
        await pilot.press("enter")
        await pilot.pause()
        assert list(app.records["sites"]) == ["sqz-128"]
        if open_with == "space":
            await pilot.press("space")
        else:
            await pilot.click("#table-sites", offset=(3, 1))
            await pilot.click("#detail-actions", offset=(3, 0))
        assert app.screen is app._main
        assert set(app.records["sites"]) == {r["site_id"] for r in client.snapshot["sites"]}
        assert not app.widget("#filter", Input).value
        assert app.widget("#order-save", Button).disabled
        await pilot.press("shift+up", "enter")
        assert not client.writes
        await pilot.press("shift+down", "escape")
        assert app._site_order is None and not client.writes
        assert next(iter(app.records["sites"])) == "sqz-128"
        assert app.focused is app.widget("#table-sites", DataTable)
        await pilot.press("enter")
        assert app._main.has_class("details-open")


async def test_inline_order_refresh_keeps_draft_and_rejects_concurrent_edit_without_rebasing():
    client = PreferenceClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("3", "shift+down")
        draft = app._site_order
        client.snapshot["policy"].update(generation=8, preferred_site_ids=["autodl-1"])
        client.snapshot["sites"].append({"site_id": "new-site", "state": "ready"})
        app.refresh_status()
        await pilot.pause()
        assert app._site_order is draft and draft.generation == 7
        assert next(iter(app.records["sites"])) == "lab1"
        assert "new-site" not in draft.order and "new-site" in app.records["sites"]
        assert app.widget("#order-save", Button).disabled
        await pilot.press("enter")
        assert not client.writes and "changed or is stale" in app._notice_text
        await pilot.press("escape")
        assert next(iter(app.records["sites"])) == "autodl-1"
        await pilot.press("shift+down")
        assert app._site_order.generation == 8
        # A race after the last GET must still be rejected by server-side CAS.
        client.snapshot["policy"]["generation"] = 9
        await pilot.press("enter")
        await pilot.pause()
        assert not client.calls and client.writes == 1
        assert app._site_order is not None and app.widget("#order-save", Button).disabled
        assert "not confirmed" in app._notice_text and "SECRET" not in app._notice_text
        await pilot.press("enter")
        assert client.writes == 1
        client.fail = True
        app.refresh_status()
        await pilot.pause()
        assert not app.available_actions()
        assert app.widget("#order-save", Button).disabled
        assert not app.widget("#order-cancel", Button).disabled
        await pilot.press("escape")
        assert app._site_order is None


@pytest.mark.parametrize("problem", ["pending", "old-server", "stale"])
async def test_inline_order_rejects_unavailable_policy(problem):
    client = PreferenceClient()
    policy = client.snapshot["policy"]
    if problem == "pending":
        policy["pending_generation"] = 8
    elif problem == "old-server":
        del policy["preferred_site_ids"]
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("3")
        if problem == "stale":
            app.last_success -= 200
            app.update_actions()
        assert not app.available_actions()
        await pilot.press("space", "shift+down")
        assert app.screen is app._main and app._site_order is None and not client.writes


@pytest.mark.parametrize("size", [(80, 24), (45, 14)])
async def test_inline_order_mouse_buttons_scrolling_resizing_and_cancel(size):
    client = PreferenceClient()
    client.snapshot["sites"].extend(
        {"site_id": f"extra-{i}", "state": "offline"} for i in range(15)
    )
    app = Workbench(client, interval=60, no_color=True)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("3")
        await pilot.click("#table-sites", offset=(3, 1))
        await pilot.click("#detail-actions", offset=(3, 0))
        await pilot.click("#order-down")
        assert app._site_order.order[1] == "sqz-128"
        await pilot.click("#order-up")
        assert app._site_order.order[0] == "sqz-128"
        table = app.widget("#table-sites", DataTable)
        table.move_cursor(row=table.row_count - 1, animate=False)
        await pilot.pause()
        selected = app.selections["sites"]
        await pilot.press("shift+up")
        assert app._site_order.order[-2] == selected
        await pilot.resize_terminal(60, 20)
        assert app.screen is app._main and app.focused is table
        await pilot.click("#order-cancel")
        assert not client.writes and app._site_order is None
        assert app.focused is table


@pytest.mark.parametrize("navigation", ["right", "2", "mouse"])
async def test_inline_order_tab_navigation_discards_unsaved_changes(navigation):
    client = PreferenceClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("3", "shift+down")
        if navigation == "mouse":
            await pilot.click(app.widget("#views", TabbedContent).get_tab("gpus"))
        else:
            await pilot.press(navigation)
        await pilot.pause()
        assert app.active_view != "sites" and app._site_order is None
        assert not client.writes
        await pilot.press("3")
        assert next(iter(app.records["sites"])) == "sqz-128"
        assert app.focused is app.widget("#table-sites", DataTable)


async def test_inline_order_busy_save_is_single_request_and_does_not_steal_navigation():
    client = PreferenceClient()
    client.preference_gate = asyncio.Event()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("3", "shift+down", "enter")
        await pilot.pause()
        assert app._action_busy and client.writes == 1
        draft = list(app._site_order.order)
        await pilot.press("enter", "shift+down", "escape")
        assert app._site_order.order == draft and client.writes == 1
        assert app.widget("#order-save", Button).disabled
        await pilot.press("right")
        assert app.active_view == "assets"
        client.preference_gate.set()
        await pilot.pause()
        assert not app._action_busy and app._site_order is None
        assert len(client.calls) == 1 and app.active_view == "assets"
        assert app.focused is app.widget("#table-assets", DataTable)


async def test_inline_order_help_and_row_click_do_not_save_or_open_details():
    client = PreferenceClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("3", "shift+down")
        draft = app._site_order
        await pilot.press("question_mark")
        assert isinstance(app.screen, ChoiceDialog)
        await pilot.press("shift+down")
        assert app._site_order is draft and not client.writes
        await pilot.press("escape")
        await pilot.click("#table-sites", offset=(3, 1))
        assert not app._main.has_class("details-open") and not client.writes
        await pilot.press("slash", "d")
        assert not app._main.has_class("searching") and not app._main.has_class("details-open")
        await pilot.press("escape")
        assert app._site_order is None


def test_site_order_draft_is_captured_and_tracks_registration_loss_without_mutation():
    policy = {"generation": 1, "preferred_site_ids": ["b"]}
    sites = [{"site_id": "a", "state": "ready"}, {"site_id": "b", "state": "offline"}]
    draft = SiteOrderDraft.capture(policy, sites)
    assert draft.order == ["b", "a"] and not draft.changed
    assert not draft.move("b", -1) and not draft.move("missing", 1)
    assert draft.move("b", 1) and draft.preferences == ("a", "b")
    assert sites[0]["site_id"] == "a" and policy["preferred_site_ids"] == ["b"]
    assert draft.rows([sites[0]])[1]["state"] == "unregistered"
    draft.automatic()
    assert draft.preferences == () and draft.changed


@pytest.mark.parametrize("size", [(120, 40), (60, 20), (45, 14)])
async def test_modifier_free_space_pick_arrows_move_drop_select_and_save(size):
    client = PreferenceClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("3", "down")
        assert app.selections["sites"] == "lab1" and app._site_order is None
        await pilot.press("space", "up")
        assert app.screen is app._main and app.active_view == "sites"
        assert app._site_order.grabbed == "lab1"
        assert next(iter(app.records["sites"])) == "lab1"
        assert app.selections["sites"] == "lab1"
        assert "↕" in app.widget("#table-sites", DataTable).get_cell_at((0, 0)).plain
        # Boundary arrows cannot wrap or drop the picked-up site.
        await pilot.press("up")
        assert next(iter(app.records["sites"])) == "lab1"
        await pilot.press("space")
        assert app._site_order.grabbed is None
        await pilot.press("down")
        assert app.selections["sites"] == "sqz-128"
        assert next(iter(app.records["sites"])) == "lab1"
        await pilot.press("space", "down", "space")
        expected = ["lab1", "remote-jupyter-1", "sqz-128", "autodl-1"]
        assert list(app.records["sites"]) == expected
        assert app._site_order.grabbed is None and not client.writes
        await pilot.press("enter")
        await pilot.pause()
        assert client.calls == [
            ("site-preference", {"preferred_site_ids": expected, "expected_generation": 7})
        ]
        assert app._site_order is None and app.focused is app.widget("#table-sites", DataTable)


@pytest.mark.parametrize("open_with", ["space", "mouse"])
async def test_modifier_free_reorder_space_or_details_arrows_cancel_and_mouse_save(open_with):
    client = PreferenceClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("3")
        if open_with == "space":
            await pilot.press("space")
        else:
            await pilot.click("#table-sites", offset=(3, 1))
            await pilot.click("#detail-actions", offset=(3, 0))
        assert app._site_order.grabbed == "sqz-128"
        await pilot.press("down")
        assert list(app.records["sites"])[1] == "sqz-128" and not client.writes
        await pilot.press("escape")
        assert app._site_order is None and next(iter(app.records["sites"])) == "sqz-128"
        await pilot.press("space", "down")
        await pilot.click("#order-save")
        await pilot.pause()
        assert client.writes == 1 and app._site_order is None
        assert client.calls[0][1]["preferred_site_ids"][1] == "sqz-128"


async def test_redundant_reorder_shortcut_and_header_button_are_absent():
    client = PreferenceClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("3", "o")
        table = app.widget("#table-sites", DataTable)
        assert not app.query("#page-site-order")
        assert app._site_order is None and app.focused is table
        assert not client.writes and not app._main.has_class("details-open")
        await pilot.press("space", "down", "escape")
        assert app._site_order is None and app.focused is table and not client.writes


@pytest.mark.parametrize("split", [False, True])
async def test_actual_terminal_sequences_decode_and_move_without_synthetic_keys(split):
    client = PreferenceClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        parser = XTermParser()

        async def receive(sequence):
            for chunk in sequence if split else [sequence]:
                for message in parser.feed(chunk):
                    if isinstance(message, events.Key):
                        app._driver.process_message(message)
            await pilot.pause()

        await receive("3")
        await receive(" ")
        assert app._site_order.grabbed == "sqz-128"
        await receive("\x1b[B")  # ordinary Down, no modifier protocol required
        assert list(app.records["sites"])[1] == "sqz-128"
        await receive("\x1b[A")
        assert next(iter(app.records["sites"])) == "sqz-128"
        await receive("\x1b[1;2B")  # standard Shift+Down remains optional
        assert list(app.records["sites"])[1] == "sqz-128"
        await receive("\r")
        assert client.writes == 1 and app._site_order is None


async def test_pick_is_scoped_to_table_and_never_intercepts_text_details_or_modals():
    client = PreferenceClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("3", "slash", "space", "down")
        assert isinstance(app.focused, Input) and app._site_order is None
        await pilot.press("escape", "enter")
        assert app._main.has_class("details-open")
        await pilot.press("space", "down")
        assert app._site_order is None
        await pilot.press("escape", "space")
        draft = app._site_order
        original = list(draft.order)
        await pilot.press("question_mark", "down", "space")
        assert isinstance(app.screen, ChoiceDialog) and draft.order == original
        await pilot.press("escape", "down", "escape")
        assert app._site_order is None and not client.writes
