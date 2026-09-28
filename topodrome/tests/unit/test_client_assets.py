from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from topodrome.client import AsyncClient, Client

pytestmark = pytest.mark.unit


def test_sync_upload_streams_local_file(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"local content")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/asset-uploads/lens/v1"
        assert request.url.params["visibility"] == "private"
        raw = request.read()
        length = int.from_bytes(raw[:8], "big")
        assert json.loads(raw[8 : 8 + length])["files"][0]["size_bytes"] == len(b"local content")
        assert raw[8 + length :] == b"local content"
        return httpx.Response(201, json=_asset())

    with Client("http://test", transport=httpx.MockTransport(handler)) as client:
        assert client.upload_asset(path, name="lens", version="v1").name == "lens"


def _asset() -> dict[str, object]:
    return {
        "schema_version": 1,
        "ref": "asset://shared/lens@v1",
        "name": "lens",
        "version": "v1",
        "visibility": "shared",
        "owner_username": "alice",
        "producer_job_id": "job-1",
        "producer_attempt_id": "attempt-1",
        "source_path": "lens",
        "kind": "directory",
        "files": [{"relative_path": "params.pt", "size_bytes": 123, "sha256": "a" * 64}],
        "total_size_bytes": 123,
        "digest": "b" * 64,
        "manifest_sha256": "c" * 64,
        "published_at": datetime(2026, 9, 13, tzinfo=UTC).isoformat(),
    }


def test_sync_asset_sdk_publishes_lists_and_reads() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            assert json.loads(request.content) == {
                "name": "lens",
                "version": "v1",
                "job_id": "job-1",
                "path": "lens",
                "visibility": "shared",
            }
            return httpx.Response(201, json=_asset())
        if request.url.path == "/api/v1/assets":
            assert request.url.params["visibility"] == "shared"
            assert request.url.params["limit"] == "7"
            return httpx.Response(200, json=[_asset()])
        assert request.url.path == "/api/v1/assets/lens/v1"
        assert request.url.params["visibility"] == "shared"
        return httpx.Response(200, json=_asset())

    with Client(transport=httpx.MockTransport(handler)) as client:
        published = client.publish_asset(
            name="lens",
            version="v1",
            job_id="job-1",
            path="lens/",
            visibility="shared",
        )
        listed = client.assets(visibility="shared", limit=7)
        detail = client.asset("lens", "v1", visibility="shared")

    assert listed == [published]
    assert detail == published
    assert [request.url.path for request in requests] == [
        "/api/v1/assets",
        "/api/v1/assets",
        "/api/v1/assets/lens/v1",
    ]


async def test_async_asset_sdk_publishes_lists_and_reads() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json=_asset())
        if request.url.path == "/api/v1/assets":
            return httpx.Response(200, json=[_asset()])
        return httpx.Response(200, json=_asset())

    async with AsyncClient(transport=httpx.MockTransport(handler)) as client:
        published = await client.publish_asset(
            name="lens",
            version="v1",
            job_id="job-1",
            path="lens",
            visibility="shared",
        )
        listed = await client.assets(visibility="shared")
        detail = await client.asset("lens", "v1", visibility="shared")

    assert published == listed[0] == detail


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_asset_mutation_sdk_and_repeated_tag_params(asynchronous):
    from contextlib import AsyncExitStack
    from inspect import isawaitable

    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "PATCH":
            assert request.url.path == "/api/v1/assets/lens/v1/tags"
            assert request.url.params["visibility"] == "shared"
            assert json.loads(request.content) == {"add": ["project:test"], "remove": ["old"]}
            return httpx.Response(200, json={**_asset(), "tags": ["project:test"]})
        if request.method == "DELETE":
            assert request.url.path == "/api/v1/assets/lens/v1"
            assert request.url.params["visibility"] == "shared"
            return httpx.Response(
                200,
                json={
                    "ref": _asset()["ref"],
                    "deleted_at": datetime.now(UTC).isoformat(),
                    "already_deleted": False,
                },
            )
        assert request.url.params.get_list("tag") == ["kind:lens", "project:test"]
        return httpx.Response(200, json=[_asset()])

    async def invoke(method, *args, **kwargs):
        result = method(*args, **kwargs)
        return await result if isawaitable(result) else result

    async with AsyncExitStack() as stack:
        cls = AsyncClient if asynchronous else Client
        context = cls("http://test", transport=httpx.MockTransport(handler))
        client = (
            await stack.enter_async_context(context)
            if asynchronous
            else stack.enter_context(context)
        )
        tagged = await invoke(
            client.update_asset_tags,
            "lens",
            "v1",
            add=("Project:Test",),
            remove=("old",),
            visibility="shared",
        )
        assert tagged.tags == ("project:test",)
        assert len(await invoke(client.assets, tags=("PROJECT:TEST", "kind:lens"))) == 1
        assert not (
            await invoke(client.delete_asset, "lens", "v1", visibility="shared")
        ).payloads_deleted
        with pytest.raises(ValueError):
            await invoke(client.update_asset_tags, "lens", "v1", add=("bad tag",))
        with pytest.raises(ValueError):
            await invoke(client.delete_asset, "../lens", "v1")
    assert len(requests) == 3
