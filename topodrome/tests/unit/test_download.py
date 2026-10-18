from __future__ import annotations

import asyncio
import hashlib
import threading
from datetime import UTC, datetime

import httpx
import pytest

from topodrome.client import AsyncClient, Client
from topodrome.client import download as downloads
from topodrome.domain.downloads import DownloadManifest
from topodrome.storage.assets import asset_manifest_digest

pytestmark = pytest.mark.unit
IDENTITY = "download_" + "1" * 32


def manifest(files, *, kind="directory", reference="job://job-one/results/"):
    members = [
        {
            "logical_path": name,
            "relative_path": name,
            "size_bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for name, body in sorted(files.items())
    ]
    return DownloadManifest(
        download_id=IDENTITY,
        ref=reference,
        kind=kind,
        digest=asset_manifest_digest(members),
        total_size_bytes=sum(len(body) for body in files.values()),
        files=[{k: v for k, v in m.items() if k != "logical_path"} for m in members],
        expires_at=datetime.now(UTC),
    )


def handler_for(descriptor, files, requests, serve=None):
    def handler(request):
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.url.path.endswith("/renew"):
            return httpx.Response(200, json={"download_id": IDENTITY})
        if request.method == "POST":
            return httpx.Response(201, json=descriptor.model_dump(mode="json"))
        item = descriptor.files[int(request.url.path.rsplit("/", 1)[1])]
        body = files[item.relative_path]
        offset = int(request.headers.get("Range", "bytes=0-")[6:-1])
        headers = {"ETag": f'"{item.sha256}"', "Content-Length": str(len(body) - offset)}
        if offset:
            headers["Content-Range"] = f"bytes {offset}-{len(body) - 1}/{len(body)}"
        if serve:
            return serve(request, body, offset, headers)
        return httpx.Response(206 if offset else 200, headers=headers, content=body[offset:])

    return handler


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("reference", ["job://job-one/results/", "asset://private/data@v1"])
def test_sync_download_editable_verified_copy_and_no_overwrite(tmp_path, kind, reference):
    files = {"data": b"payload"} if kind == "file" else {"empty": b"", "sub/data": b"payload"}
    descriptor = manifest(files, kind=kind, reference=reference)
    requests = []
    with Client(
        "http://test", transport=httpx.MockTransport(handler_for(descriptor, files, requests))
    ) as c:
        destination = tmp_path / "result"
        assert c.download(reference, destination) == destination
        target = destination if kind == "file" else destination / "sub/data"
        assert target.read_bytes() == b"payload"
        target.write_bytes(b"local edits")
        with pytest.raises(FileExistsError):
            c.download(reference, destination)
        assert target.read_bytes() == b"local edits"
    assert not list(tmp_path.glob(".topodrome-download-*"))
    assert sum(r.method == "DELETE" for r in requests) == 2
    assert all("assets" not in r.url.path for r in requests)


@pytest.mark.asyncio
async def test_async_download_directory(tmp_path):
    files = {"empty": b"", "sub/data": b"payload"}
    descriptor = manifest(files)
    requests = []
    async with AsyncClient(
        "http://test", transport=httpx.MockTransport(handler_for(descriptor, files, requests))
    ) as c:
        destination = await c.download(descriptor.ref, tmp_path / "result")
    assert (destination / "sub/data").read_bytes() == b"payload"
    assert (destination / "empty").read_bytes() == b""
    assert requests[-1].method == "DELETE"


def test_interrupted_download_resumes_from_verified_prefix(tmp_path, monkeypatch):
    body = b"a" * downloads.CHUNK_BYTES + b"b" * downloads.CHUNK_BYTES + b"tail"
    files = {"data": body}
    descriptor = manifest(files, kind="file")
    requests = []
    monkeypatch.setattr(downloads.time, "sleep", lambda _: None)

    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield body[: downloads.CHUNK_BYTES]
            raise httpx.ReadError("connection lost")

    def broken(request, body, offset, headers):
        if offset:
            raise httpx.ConnectError("offline")
        return httpx.Response(200, headers=headers, stream=Broken())

    destination = tmp_path / "result"
    with Client(
        "http://test",
        transport=httpx.MockTransport(handler_for(descriptor, files, requests, broken)),
    ) as c:
        with pytest.raises(httpx.TransportError):
            c.download(descriptor.ref, destination)
    assert not destination.exists()
    assert (
        next(tmp_path.glob(".topodrome-download-*/payload")).stat().st_size == downloads.CHUNK_BYTES
    )
    assert requests[-1].method == "DELETE"
    with Client(
        "http://test", transport=httpx.MockTransport(handler_for(descriptor, files, requests))
    ) as c:
        assert c.download(descriptor.ref, destination).read_bytes() == body
    resumed = [r for r in requests if r.method == "GET"][-1]
    assert resumed.headers["Range"] == f"bytes={downloads.CHUNK_BYTES}-"
    assert not list(tmp_path.glob(".topodrome-download-*"))


@pytest.mark.parametrize("failure", ["checksum", "range", "etag", "length"])
def test_invalid_response_never_exposes_result(tmp_path, failure):
    files = {"data": b"payload"}
    descriptor = manifest(files, kind="file")
    destination = tmp_path / "result"
    if failure == "range":
        with downloads.ResumableDownload(descriptor, destination, "http://test") as d:
            with d.open_partial(descriptor.files[0])[0] as output:
                output.write(b"pay")

    def corrupt(request, body, offset, headers):
        if failure == "etag":
            headers["ETag"] = '"different"'
        if failure == "length":
            headers["Content-Length"] = "99"
        return httpx.Response(
            200, headers=headers, content=b"wrong!!" if failure == "checksum" else body
        )

    with Client(
        "http://test", transport=httpx.MockTransport(handler_for(descriptor, files, [], corrupt))
    ) as c:
        with pytest.raises(ValueError):
            c.download(descriptor.ref, destination)
    assert not destination.exists()
    if failure == "checksum":
        assert next(tmp_path.glob(".topodrome-download-*/payload")).stat().st_size == 0


def test_partial_content_is_bound_to_manifest_and_server(tmp_path):
    descriptor = manifest({"data": b"payload"}, kind="file")
    with downloads.ResumableDownload(descriptor, tmp_path / "result", "http://test"):
        pass
    for changed, origin in [
        (manifest({"data": b"changed"}, kind="file"), "http://test"),
        (descriptor, "http://other"),
    ]:
        with pytest.raises(ValueError, match="different content or server"):
            downloads.ResumableDownload(changed, tmp_path / "result", origin)


def test_staging_rejects_symlinks_and_concurrent_downloads(tmp_path):
    descriptor = manifest({"sub/data": b"payload"})
    with downloads.ResumableDownload(descriptor, tmp_path / "result", "http://test") as d:
        with pytest.raises(BlockingIOError):
            downloads.ResumableDownload(descriptor, tmp_path / "result", "http://test")
        (d.payload / "sub").symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            d.open_partial(descriptor.files[0])
    assert not (tmp_path / "data").exists()


def test_complete_corrupt_partial_is_restarted_and_extra_files_are_rejected(tmp_path):
    files = {"data": b"payload"}
    descriptor = manifest(files)
    destination = tmp_path / "result"
    with downloads.ResumableDownload(descriptor, destination, "http://test") as d:
        with d.open_partial(descriptor.files[0])[0] as output:
            output.write(b"wrong!!")
        (d.payload / "unexpected").write_bytes(b"not in manifest")
    requests = []
    with Client(
        "http://test", transport=httpx.MockTransport(handler_for(descriptor, files, requests))
    ) as c:
        with pytest.raises(ValueError, match="unrecognized"):
            c.download(descriptor.ref, destination)
    assert not destination.exists()
    assert "Range" not in next(r for r in requests if r.method == "GET").headers


@pytest.mark.asyncio
async def test_cancellation_retains_partial_and_releases_pin_and_local_lock(tmp_path):
    files = {"data": b"x" * (downloads.CHUNK_BYTES + 3)}
    descriptor = manifest(files, kind="file")
    requests = []

    class Canceled(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield files["data"][: downloads.CHUNK_BYTES]
            raise asyncio.CancelledError

    def serve(request, body, offset, headers):
        return httpx.Response(200, headers=headers, stream=Canceled())

    destination = tmp_path / "result"
    async with AsyncClient(
        "http://test",
        transport=httpx.MockTransport(handler_for(descriptor, files, requests, serve)),
    ) as c:
        with pytest.raises(asyncio.CancelledError):
            await c.download(descriptor.ref, destination)
    assert not destination.exists()
    assert requests[-1].method == "DELETE"
    async with AsyncClient(
        "http://test", transport=httpx.MockTransport(handler_for(descriptor, files, requests))
    ) as c:
        assert (await c.download(descriptor.ref, destination)).read_bytes() == files["data"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["constructor", "open_partial"])
async def test_cancel_during_resource_creation_closes_resource(tmp_path, monkeypatch, operation):
    files = {"data": b"payload"}
    descriptor = manifest(files, kind="file")
    entered, release = threading.Event(), threading.Event()
    original = (
        downloads.ResumableDownload
        if operation == "constructor"
        else downloads.ResumableDownload.open_partial
    )

    def delayed(*args):
        result = original(*args)
        entered.set()
        assert release.wait(5)
        return result

    if operation == "constructor":
        monkeypatch.setattr(downloads, "ResumableDownload", delayed)
    else:
        monkeypatch.setattr(downloads.ResumableDownload, "open_partial", delayed)
    requests = []
    async with AsyncClient(
        "http://test", transport=httpx.MockTransport(handler_for(descriptor, files, requests))
    ) as c:
        task = asyncio.create_task(c.download(descriptor.ref, tmp_path / "result"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    monkeypatch.undo()
    with downloads.ResumableDownload(descriptor, tmp_path / "result", "http://test"):
        pass  # A leaked descriptor would still hold the flock.
    assert requests[-1].method == "DELETE"


@pytest.mark.asyncio
async def test_keepalive_runs_during_local_io(tmp_path, monkeypatch):
    files = {"data": b"payload"}
    descriptor = manifest(files, kind="file")
    requests = []
    original = downloads.ResumableDownload.open_partial
    monkeypatch.setattr(downloads, "RENEW_SECONDS", 0.005)

    def slow(*args):
        threading.Event().wait(0.03)
        return original(*args)

    monkeypatch.setattr(downloads.ResumableDownload, "open_partial", slow)
    async with AsyncClient(
        "http://test", transport=httpx.MockTransport(handler_for(descriptor, files, requests))
    ) as c:
        await c.download(descriptor.ref, tmp_path / "result")
    get_index = next(i for i, r in enumerate(requests) if r.method == "GET")
    assert sum(r.url.path.endswith("/renew") for r in requests[:get_index]) >= 2
