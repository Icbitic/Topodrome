from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from topodrome.client import cli


class _AssetClient:
    published: Any = None

    def __enter__(self) -> _AssetClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def publish_asset(self, request: Any) -> dict[str, Any]:
        type(self).published = request
        return {"ref": f"asset://{request.visibility}/{request.name}@{request.version}"}

    def assets(self, *, visibility: str | None, limit: int, tags=()) -> list[dict[str, Any]]:
        return [{"visibility": visibility, "limit": limit, "tags": tags}]

    def delete_asset(self, name, version, *, visibility):
        type(self).published = (name, version, visibility)
        return {"state": "deleted", "payloads_deleted": False}

    def update_asset_tags(self, name, version, **values):
        type(self).published = {"name": name, "version": version, **values}
        return type(self).published

    def upload_asset(self, path: Any, **values: Any) -> dict[str, Any]:
        type(self).published = {"path": str(path), **values}
        return type(self).published

    def download_asset(self, name: str, version: str, destination: Path, **values: Any) -> Path:
        type(self).published = {"name": name, "version": version, **values}
        return destination

    def asset(self, name: str, version: str, *, visibility: str) -> dict[str, Any]:
        return {"name": name, "version": version, "visibility": visibility}


def test_asset_publish_cli_builds_shared_prefix_request(monkeypatch: Any) -> None:
    _AssetClient.published = None
    monkeypatch.setattr(cli, "Client", _AssetClient)

    result = CliRunner().invoke(
        cli.app,
        [
            "asset-publish",
            "lens",
            "v1",
            "--job",
            "job-1",
            "--path",
            "outputs/lens/",
            "--shared",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _AssetClient.published is not None
    assert _AssetClient.published.path == "outputs/lens"
    assert _AssetClient.published.visibility == "shared"


def test_asset_list_and_detail_cli(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "Client", _AssetClient)

    listed = CliRunner().invoke(
        cli.app,
        ["assets", "--visibility", "private", "--limit", "4"],
    )
    detail = CliRunner().invoke(cli.app, ["asset", "lens", "v2", "--shared"])

    assert listed.exit_code == 0, listed.output
    assert '"visibility": "private"' in listed.output
    assert '"limit": 4' in listed.output
    assert detail.exit_code == 0, detail.output
    assert '"visibility": "shared"' in detail.output


def test_asset_upload_cli_defaults_private_and_accepts_shared(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "Client", _AssetClient)
    runner = CliRunner()
    private = runner.invoke(cli.app, ["asset-upload", "lens", "v2", "./edited"])
    assert private.exit_code == 0, private.output
    assert _AssetClient.published["visibility"] == "private"
    shared = runner.invoke(cli.app, ["asset-upload", "lens", "v3", "./edited", "--shared"])
    assert shared.exit_code == 0, shared.output
    assert _AssetClient.published["visibility"] == "shared"


def test_asset_download_cli(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "Client", _AssetClient)
    result = CliRunner().invoke(cli.app, ["asset-download", "lens", "v1", "./local", "--shared"])
    assert result.exit_code == 0, result.output
    assert _AssetClient.published == {"name": "lens", "version": "v1", "visibility": "shared"}


def test_asset_delete_confirmation_and_exact_visibility(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "Client", _AssetClient)
    _AssetClient.published = None
    runner = CliRunner()
    declined = runner.invoke(cli.app, ["asset-delete", "lens", "v1"], input="n\n")
    assert declined.exit_code != 0
    assert _AssetClient.published is None
    accepted = runner.invoke(cli.app, ["asset-delete", "lens", "v1", "--shared", "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert _AssetClient.published == ("lens", "v1", "shared")
    assert '"payloads_deleted": false' in accepted.output


def test_asset_tags_cli_and_filter_validation(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "Client", _AssetClient)
    runner = CliRunner()
    tagged = runner.invoke(
        cli.app,
        ["asset-tags", "lens", "v1", "--add", " Project:Test ", "--remove", "old", "--shared"],
    )
    assert tagged.exit_code == 0, tagged.output
    assert _AssetClient.published == {
        "name": "lens",
        "version": "v1",
        "add": ("project:test",),
        "remove": ("old",),
        "visibility": "shared",
    }
    listed = runner.invoke(cli.app, ["assets", "--tag", "Project:Test", "--tag", "kind:lens"])
    assert listed.exit_code == 0, listed.output
    assert '"project:test"' in listed.output and '"kind:lens"' in listed.output
    _AssetClient.published = None
    for args in (
        ["asset-tags", "lens", "v1", "--add", "bad tag"],
        ["asset-tags", "lens", "v1", "--add", "same", "--remove", "Same"],
        ["assets", "--tag", "bad tag"],
    ):
        result = runner.invoke(cli.app, args)
        assert result.exit_code != 0, result.output
    assert _AssetClient.published is None
