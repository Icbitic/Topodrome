"""One disclosure interaction for arrays and GPUs; Enter is always details."""

from copy import deepcopy

import pytest
from textual.widgets import DataTable, Input, Static, TabbedContent

from tests.unit.test_workbench import FakeClient
from topodrome.client.workbench import Workbench

pytestmark = pytest.mark.unit


class ArrayClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.snapshot["arrays"] = [
            {
                "array_id": "array_sweep",
                "name": "Sweep",
                "state": "running",
                "total": 2,
                "terminal": 0,
                "counts": {"running": 1, "queued": 1},
                "active_compute": 1,
                "max_parallel": 2,
                "held": False,
                "owner_username": "alice",
                "created_at": "2026-09-28T00:00:00Z",
            }
        ]
        self.snapshot["jobs"] = [
            {
                "job_id": f"job_{i}",
                "array_id": "array_sweep",
                "array_index": i,
                "state": state,
                "name": "Sweep",
            }
            for i, state in enumerate(("running", "queued"))
        ]
        self.array_members = deepcopy(self.snapshot["jobs"])
        self.reads = []

    async def status(self, job):
        assert not job.startswith("array_"), "An array is not a single-job resource"
        self.reads.append(("status", job))
        return await super().status(job)

    async def attempts(self, job):
        assert not job.startswith("array_")
        self.reads.append(("attempts", job))
        return await super().attempts(job)

    async def logs(self, job, **kwargs):
        assert not job.startswith("array_")
        self.reads.append(("logs", job))
        return await super().logs(job, **kwargs)


@pytest.mark.parametrize("size", [(120, 35), (65, 24), (45, 14)])
async def test_array_enter_opens_live_summary_and_never_expands_or_reads_job_endpoints(size):
    client = ArrayClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app._main.has_class("details-open")
        assert app._opened_detail[1] == "array_sweep"
        assert not app.expanded_arrays
        summary = str(app.widget("#detail-summary", Static).render())
        assert "Parallel limit" in summary and "alice" in summary and "0 / 2" in summary
        tabs = app.widget("#detail-tabs", TabbedContent)
        for key in ("left", "right", "2", "3"):
            await pilot.press(key)
            assert tabs.active == "pane-summary"
        focus = app.focused
        client.snapshot["arrays"][0].update(
            state="succeeded",
            terminal=2,
            counts={"succeeded": 2},
            active_compute=0,
        )
        for job in client.snapshot["jobs"]:
            job["state"] = "succeeded"
        app.tick()
        await pilot.pause()
        assert "2 / 2" in str(app.widget("#detail-summary", Static).render())
        assert app.focused is focus
        assert app.selected()["state"] == "succeeded"
        assert not client.reads and not client.calls
        await pilot.press("escape", "h")
        assert app.focused.id == "table-jobs"
        assert app.selected()["job_id"] == "array_sweep"


@pytest.mark.parametrize(
    "view,tab,parent,child",
    [
        ("jobs", "1", "array_sweep", "job_0"),
        ("gpus", "2", "lab1/GPU-exact", "lab1/GPU-exact/process/info-0"),
    ],
)
async def test_space_and_native_disclosure_share_behavior_on_parents_and_children(
    view, tab, parent, child
):
    client = ArrayClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        await pilot.press(tab)
        table = app.widget(f"#table-{view}", DataTable)
        assert table.rows[parent].label.plain == "▸"
        await pilot.click(f"#table-{view}", offset=(1, 1))
        await pilot.pause()
        assert table.rows[parent].label.plain == "▾"
        assert child in app.records[view]
        assert not app._main.has_class("details-open")
        assert app.focused is table
        await pilot.press("down", "space")
        await pilot.pause()
        assert child not in app.records[view]
        assert app.selections[view] == parent
        assert table.rows[parent].label.plain == "▸"
        await pilot.press("space")
        await pilot.pause()
        await pilot.click(f"#table-{view}", offset=(1, 1))
        await pilot.pause()
        assert child not in app.records[view]
        await pilot.click(f"#table-{view}", offset=(5, 1))
        await pilot.pause()
        assert app._main.has_class("details-open")
        assert app._opened_detail[1] == parent
        assert not client.calls


async def test_array_disclosure_updates_when_no_visible_members_and_keeps_filter_editing():
    client = ArrayClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # A partial status snapshot and an empty fetch leave no child rows;
        # the disclosure must still update when the row identities don't.
        client.snapshot["jobs"] = []
        client.array_members = []
        app.tick()
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        table = app.widget("#table-jobs", DataTable)
        assert table.row_count == 1
        assert table.rows["array_sweep"].label.plain == "▾"
        await pilot.press("space")
        assert table.rows["array_sweep"].label.plain == "▸"
        await pilot.press("slash", "s", "space", "w")
        assert app.widget("#filter", Input).value == "s w"
        assert not app.expanded_arrays


async def test_array_member_details_have_normal_tabs_and_return_to_same_child():
    app = Workbench(ArrayClient(), interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "escape", "space", "down", "enter")
        await pilot.pause()
        assert app._opened_detail[1] == "job_0"
        await pilot.press("right")
        assert app.widget("#detail-tabs", TabbedContent).active == "pane-attempts"
        await pilot.press("right")
        assert app.widget("#detail-tabs", TabbedContent).active == "pane-logs"
        await pilot.press("escape")
        assert app.selected()["job_id"] == "job_0"
        assert app.expanded_arrays == {"array_sweep"}
        assert app.focused.id == "table-jobs"
