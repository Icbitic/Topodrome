from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from topodrome.client import AsyncClient, Client
from topodrome.client.asset_download import AssetDownload
from topodrome.domain.errors import NotFoundError
from topodrome.scheduler.asset_download import open_member
from topodrome.scheduler.assets import AssetSummary, ResolvedAssetMember
from topodrome.storage.assets import asset_manifest_digest

pytestmark = pytest.mark.unit


def summary(files: dict[str, bytes], *, kind: str = "directory") -> AssetSummary:
    members = [
        {
            "logical_path": name,
            "relative_path": name,
            "size_bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for name, body in sorted(files.items())
    ]
    return AssetSummary(
        ref="asset://private/data@v1",
        name="data",
        version="v1",
        visibility="private",
        owner_username="alice",
        source_path="files",
        kind=kind,
        files=[{k: v for k, v in item.items() if k != "logical_path"} for item in members],
        total_size_bytes=sum(len(body) for body in files.values()),
        digest=asset_manifest_digest(members),
        manifest_sha256="a" * 64,
        published_at=datetime.now(UTC),
    )


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_sync_download_verifies_editable_copy_without_overwriting(
    tmp_path: Path, kind: str
) -> None:
    files = (
        {"data.bin": b"payload"} if kind == "file" else {"empty": b"", "sub/data.bin": b"payload"}
    )
    asset = summary(files, kind=kind)

    def handler(request: httpx.Request) -> httpx.Response:
        if "/files/" not in request.url.path:
            return httpx.Response(200, json=asset.model_dump(mode="json"))
        index = int(request.url.path.rsplit("/", 1)[1])
        return httpx.Response(200, content=files[asset.files[index].relative_path])

    destination = tmp_path / "download"
    with Client("http://test", transport=httpx.MockTransport(handler)) as client:
        assert client.download_asset("data", "v1", destination) == destination
        with pytest.raises(FileExistsError):
            client.download_asset("data", "v1", destination)
    target = destination if kind == "file" else destination / "sub/data.bin"
    assert target.read_bytes() == b"payload"
    target.write_bytes(b"local edits")
    assert not list(tmp_path.glob(".topodrome-download-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"wrong!!", b"short", b"payload plus extra"])
async def test_corrupt_download_never_exposes_destination(tmp_path: Path, body: bytes) -> None:
    asset = summary({"data": b"payload"}, kind="file")

    def handler(request: httpx.Request) -> httpx.Response:
        return (
            httpx.Response(200, content=body)
            if "/files/" in request.url.path
            else httpx.Response(200, json=asset.model_dump(mode="json"))
        )

    async with AsyncClient("http://test", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            await client.download_asset("data", "v1", tmp_path / "result")
    assert not (tmp_path / "result").exists()
    assert not await asyncio.to_thread(lambda: list(tmp_path.glob(".topodrome-download-*")))


@pytest.mark.asyncio
async def test_cancelled_stream_leaves_no_partial_destination(tmp_path: Path) -> None:
    asset = summary({"data": b"payload"}, kind="file")

    class Interrupted(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"part"
            raise asyncio.CancelledError

    def handler(request: httpx.Request) -> httpx.Response:
        return (
            httpx.Response(200, stream=Interrupted())
            if "/files/" in request.url.path
            else httpx.Response(200, json=asset.model_dump(mode="json"))
        )

    async with AsyncClient("http://test", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.download_asset("data", "v1", tmp_path / "result")
    assert not (tmp_path / "result").exists()
    assert not await asyncio.to_thread(lambda: list(tmp_path.glob(".topodrome-download-*")))


def test_destination_created_during_download_is_preserved(tmp_path: Path) -> None:
    asset = summary({"data": b"x"})
    destination = tmp_path / "result"
    with AssetDownload(asset, destination) as download:
        with download.open_file(asset.files[0]) as output:
            output.write(b"x")
        destination.mkdir()
        (destination / "mine").write_text("preserve")
        with pytest.raises(FileExistsError):
            download.commit()
    assert (destination / "mine").read_text() == "preserve"


def test_server_refuses_symlinks_and_descriptor_pins_file(tmp_path: Path) -> None:
    member = ResolvedAssetMember("data/file", "file", 3, hashlib.sha256(b"old").hexdigest())
    (tmp_path / "data").mkdir()
    (tmp_path / "data/file").write_bytes(b"old")
    with open_member(tmp_path, member) as opened:
        (tmp_path / "data/file").rename(tmp_path / "data/previous")
        (tmp_path / "data/file").write_bytes(b"new")
        assert opened.read() == b"old"
    (tmp_path / "linked").symlink_to(tmp_path / "data", target_is_directory=True)
    with pytest.raises(NotFoundError):
        open_member(tmp_path, ResolvedAssetMember("linked/file", "file", 3, "a" * 64))
