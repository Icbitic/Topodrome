import asyncio
from copy import deepcopy

import pytest
from textual.widgets import DataTable, Input, Static

from tests.unit.test_workbench import FakeClient
from topodrome.client.workbench import Workbench
from topodrome.client.workbench_assets import asset_group, asset_tree

pytestmark = pytest.mark.unit


def asset(name="lens", version="v2", visibility="private", published="2026-09-28T00:00:00Z"):
    return {
        "ref": f"asset://{visibility}/{name}@{version}",
        "name": name,
        "version": version,
        "visibility": visibility,
        "published_at": published,
        "total_size_bytes": 42,
        "files": [],
        "tags": [],
    }


class AssetClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.latest = [asset(), asset(visibility="shared"), asset(name="other")]
        self.old = asset(version="v1", published="2026-09-27T00:00:00Z")
        self.history = [
            *self.latest,
            self.old,
            {**self.old, "visibility": "shared", "ref": "asset://shared/lens@v1"},
        ]
        self.history_gate = None
        self.history_error = False
        self.reads = []

    async def assets(self, **kwargs):
        self.reads.append(kwargs)
        if kwargs.get("all_versions"):
            if self.history_gate:
                await self.history_gate.wait()
            if self.history_error:
                raise RuntimeError("SECRET")
            return deepcopy(self.history)
        return deepcopy(self.latest)


@pytest.mark.parametrize("size", [(120, 35), (65, 24), (45, 14)])
async def test_assets_space_history_and_exact_version_details(size):
    client = AssetClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("4")
        await pilot.pause()
        table = app.widget("#table-assets", DataTable)
        assert table.row_count == 3
        assert not any(read.get("all_versions") for read in client.reads)
        await pilot.press("space")
        await pilot.pause()
        assert table.row_count == 4
        assert app.expanded_assets == {("private", "lens")}
        assert table.rows[client.latest[0]["ref"]].label.plain == "▾"
        assert table.rows[client.latest[1]["ref"]].label.plain == "▸"
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app._opened_detail[1] == client.old["ref"]
        assert "v1" in str(app.widget("#detail-summary", Static).render())
        # Publishing another latest version cannot retarget an open old one.
        client.latest[0] = asset(version="v3", published="2026-09-29T00:00:00Z")
        client.history.append(client.latest[0])
        focus = app.focused
        app.tick()
        await pilot.pause()
        assert app._opened_detail[1] == client.old["ref"]
        assert app.focused is focus
        await pilot.press("escape", "space")
        await pilot.pause()
        assert table.row_count == 3
        assert app.selected()["ref"] == client.latest[0]["ref"]
        assert app.focused is table
        await pilot.press("h")
        assert not app.expanded_assets and not app.show_history
        assert not client.calls


async def test_asset_arrow_click_expands_but_row_click_opens_details():
    client = AssetClient()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("4")
        await pilot.pause()
        await pilot.click("#table-assets", offset=(1, 1))
        await pilot.pause()
        assert app.expanded_assets == {("private", "lens")}
        assert not app._main.has_class("details-open")
        await pilot.press("down")  # Native table clicks open an already selected row.
        await pilot.click("#table-assets", offset=(5, 2))
        await pilot.pause()
        assert app._opened_detail[1] == client.old["ref"]
        await pilot.press("escape", "space")
        assert not app.expanded_assets


async def test_slow_asset_history_does_not_reopen_collapsed_group_or_steal_focus():
    client = AssetClient()
    client.history_gate = asyncio.Event()
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("4", "space")
        await pilot.pause()
        assert any(
            row.get("_asset_info", "").startswith("Loading")
            for row in app.records["assets"].values()
        )
        await pilot.press("space", "left")
        client.history_gate.set()
        await pilot.pause()
        assert not app.expanded_assets
        assert len(app.records["assets"]) == 3
        assert app.active_view == "sites" and app.focused.id == "table-sites"
        await pilot.press("right", "space")
        await pilot.pause()
        assert client.old["ref"] in app.records["assets"]


async def test_asset_history_failure_retry_and_filter_keep_parent_group():
    client = AssetClient()
    client.history_error = True
    app = Workbench(client, interval=60)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("4", "space")
        await pilot.pause()
        assert "History unavailable" in str(app.records["assets"])
        assert "SECRET" not in str(app.records["assets"])
        client.history_error = False
        await pilot.press("r")
        await pilot.pause()
        assert client.old["ref"] in app.records["assets"]
        await pilot.press("slash")
        app.widget("#filter", Input).value = "@v1"
        await pilot.press("enter")
        assert list(app.records["assets"]) == [client.latest[0]["ref"], client.old["ref"]]
        await pilot.press("down", "enter")
        assert app._opened_detail[1] == client.old["ref"]


def test_asset_tree_empty_truncated_and_publication_order():
    latest = asset()
    older = asset(version="v100", published="2026-09-26T00:00:00Z")
    newer = asset(version="v1", published="2026-09-27T00:00:00Z")
    expanded = {asset_group(latest)}
    rows = asset_tree([latest], [older, newer], expanded, state="ready")
    assert [row["version"] for row in rows] == ["v2", "v1", "v100"]
    rows = asset_tree([latest], [], expanded, state="ready")
    assert rows[-1]["_asset_info"] == "No older versions"
    rows = asset_tree([latest], [], expanded, state="ready", limited=True)
    assert "limited" in rows[-1]["_asset_info"]
    assert "No older" not in str(rows)
