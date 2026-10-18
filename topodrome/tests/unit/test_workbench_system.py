import pytest
from textual.widgets import DataTable, Static

from topodrome.client.workbench import Workbench

pytestmark = pytest.mark.unit


class Client:
    fail = False
    value = 1024

    async def system_status(self):
        return {"sites": [], "gpus": [], "jobs": []}

    async def system_metrics(self):
        if self.fail:
            raise RuntimeError("DO_NOT_DISPLAY_SECRET")
        return {
            "storage_state": "ready",
            "database_ready": True,
            "storage": {"allocated_bytes": self.value},
        }

    async def close(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 35), (60, 20)])
async def test_system_tab_refresh_details_focus_and_stale_failure(size):
    client = Client()
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("7")
        await pilot.pause()
        table = app.widget("#table-system", DataTable)
        assert app.active_view == "system"
        assert app.focused is table
        assert app.records["system"]["used"]["value"] == "1.0 KiB"
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert not app.available_actions()
        assert "Topodrome used" in str(app.widget("#detail-summary", Static).render())
        client.value = 2048
        app.refresh_extras()
        await pilot.pause()
        app.schedule_detail()
        await pilot.pause()
        assert "2.0 KiB" in str(app.widget("#detail-summary", Static).render())
        client.fail = True
        app.refresh_extras()
        await pilot.pause()
        assert "stale" in app.records["system"]["sample"]["value"]
        assert "SECRET" not in str(app.records)
        await pilot.press("escape")
        assert app.focused is table
        await pilot.press("right")
        assert app.active_view == "jobs"
