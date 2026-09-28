from datetime import UTC, datetime, timedelta

import pytest
from textual.widgets import Button, DataTable, Input, Static

from tests.unit.test_workbench import FakeClient, confirm_action
from topodrome.client.workbench import ActionDialog, Workbench
from topodrome.client.workbench_autodl import autodl_rows
from topodrome.client.workbench_details import detail_page
from topodrome.client.workbench_gpus import gpu_tree

pytestmark = pytest.mark.unit


def process_sample():
    return {
        "processes_status": "current",
        "processes_received_at": datetime.now(UTC).isoformat(),
        "processes_ttl_seconds": 30,
        "processes": [
            {
                "pid": 123,
                "username": "alice",
                "process_name": "python",
                "script_name": "train.py",
                "used_memory_bytes": 1024**3,
                "kind": "foreign",
            },
            {
                "pid": 456,
                "username": "bob",
                "process_name": "idle_loader",
                "used_memory_bytes": 2 * 1024**3,
                "kind": "cooperative",
            },
        ],
    }


@pytest.mark.parametrize("size", [(120, 40), (60, 20), (45, 14)])
async def test_empty_autodl_has_visible_primary_registration_and_keyboard_flow(size):
    client = FakeClient()
    client.snapshot["sites"] = client.snapshot["sites"][:2]
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("6")
        await pilot.pause()
        button = app.widget("#page-register", Button)
        assert app.focused is button
        assert button.label.plain == "n Register AutoDL"
        assert button.region.right <= size[0] and button.region.height == 1
        await pilot.press("enter")
        assert isinstance(app.screen, ActionDialog)
        assert app.screen.focused.id == "ssh"
        app.screen.query_one("#ssh", Input).value = "ssh -p 12345 root@connect.bjb2.seetacloud.com"
        await pilot.press("enter")
        assert app.screen.focused.id == "password"
        app.screen.query_one("#password", Input).value = "no-display-secret"
        await confirm_action(pilot, app)
        assert len(client.calls) == 1 and client.calls[0][0] == "register"
        assert app.records["autodl"]["autodl-new"]["state"] == "starting"
        assert "secret" not in app._notice_text and "MUST_NOT_RENDER" not in app._notice_text
        await pilot.press("n")
        assert isinstance(app.screen, ActionDialog)
        await pilot.press("escape")
        assert len(client.calls) == 1


@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
async def test_gpu_inline_processes_and_enter_summary_refresh_without_moving_focus(size):
    client = FakeClient()
    client.snapshot["gpus"][0].update(process_sample())
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("2", "space")
        await pilot.pause()
        table = app.widget("#table-gpus", DataTable)
        assert table.row_count == 3
        assert list(app.records["gpus"])[1].endswith("/456")
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app._opened_detail[1] == "lab1/GPU-exact"
        summary = str(app.widget("#detail-summary", Static).render())
        assert "alice" in summary and "train.py" in summary and "PID 456" in summary
        focused = app.focused
        client.snapshot["gpus"][0]["processes"] = []
        app.tick()
        await pilot.pause()
        assert app.focused is focused
        assert "No foreign processes" in str(app.widget("#detail-summary", Static).render())
        await pilot.press("escape", "space")
        assert app.widget("#table-gpus", DataTable).row_count == 1
        assert not client.calls  # Displaying a PID never creates a management operation.


async def test_gpu_disclosure_is_clickable_and_does_not_open_actions():
    client = FakeClient()
    client.snapshot["gpus"][0].update(process_sample())
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("2")
        await pilot.click("#table-gpus", offset=(1, 1))
        assert app.expanded_gpus == {"lab1/GPU-exact"}
        assert not app._main.has_class("details-open")
        await pilot.click("#table-gpus", offset=(1, 1))
        assert not app.expanded_gpus
        assert not client.calls


def test_process_details_distinguish_missing_stale_empty_and_truncated_samples():
    gpu = {"site_id": "lab1", "uuid": "GPU", "index": 0}
    assert "unavailable" in detail_page("gpus", gpu).plain.lower()
    gpu.update(process_sample())
    gpu["processes_received_at"] = (datetime.now(UTC) - timedelta(seconds=35)).isoformat()
    text = detail_page("gpus", gpu).plain
    assert "Stale" in text and "PID 123" in text
    gpu["processes_truncated"] = True
    gpu["processes_total"] = 100
    assert "Showing 2 of 100" in detail_page("gpus", gpu).plain
    assert len(gpu_tree([gpu], {"lab1/GPU"}, "alice")) == 5


def test_autodl_startups_do_not_require_site_rows_or_expose_raw_metadata():
    rows = autodl_rows(
        [],
        [
            {
                "site_id": "new",
                "state": "starting",
                "gpu_count": 2,
                "metadata": {"password": "SECRET"},
            }
        ],
    )
    assert rows[0]["site_state"] == "awaiting heartbeat"
    assert "SECRET" not in repr(rows)
    assert "installation" in detail_page("autodl", rows[0]).plain
    assert autodl_rows(rows, [{"site_id": "new", "state": "released"}]) == []
