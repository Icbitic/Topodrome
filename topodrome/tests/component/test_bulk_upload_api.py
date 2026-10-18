from __future__ import annotations

import gzip
import hashlib
import os
import threading

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from topodrome.scheduler.bulk_upload import bulk_upload_router
from topodrome.scheduler.storage_digest import hash_beneath
from topodrome.storage.bulk_upload import (
    MAX_UPLOAD_CHUNK_BYTES,
    UPLOAD_CHUNK_BYTES,
    SchedulerBulkUploadClient,
)
from topodrome.storage.progress import TransferCancelled, cancellable_transfers, observe_transfers

pytestmark = pytest.mark.component
URL = "/api/v1/agents/storage/upload-prefix"
PREFIX = "jobs/job/attempts/attempt/.output.topodrome-upload-" + "a" * 32
TARGET = "jobs/job/attempts/attempt/output"
AUTH = {"Authorization": "Bearer enrolled"}


@pytest.fixture
def api(tmp_path):
    root = tmp_path / "authority"
    root.mkdir()
    app = FastAPI()
    app.include_router(
        bulk_upload_router(root, lambda site: "enrolled" if site == "site" else None)
    )
    with TestClient(app) as client:
        yield client, root


def put(client, content, *, offset=0, total=None, path=PREFIX, headers=None):
    return client.put(
        URL,
        params={
            "site_id": "site",
            "logical_path": path,
            "offset": offset,
            "total": len(content) if total is None else total,
        },
        headers=headers
        or {
            **AUTH,
            "Content-Length": str(len(content)),
            "X-Topodrome-Sha256": hashlib.sha256(content).hexdigest(),
        },
        content=content,
    )


def test_chunks_are_contiguous_durable_and_lost_ack_is_idempotent(api, monkeypatch):
    c, root = api
    synced = []
    original = os.fsync

    def fsync(fd):
        synced.append(os.fstat(fd).st_mode)
        return original(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    assert put(c, b"abc", total=6).json() == {
        "offset": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
    }
    assert synced
    assert put(c, b"abc", total=6).status_code == 200
    assert put(c, b"xyz", total=6).status_code == 409
    assert (root / PREFIX).read_bytes() == b"abc"
    assert not (root / TARGET).exists()
    assert put(c, b"def", offset=3, total=6).status_code == 200
    params = {"site_id": "site", "logical_path": PREFIX}
    assert c.get(URL, params=params, headers=AUTH).json()["Size"] == 6
    assert c.post(URL, params={**params, "destination": TARGET}, headers=AUTH).status_code == 204
    assert not (root / PREFIX).exists()
    assert (root / TARGET).read_bytes() == b"abcdef"
    assert c.delete(URL, params={**params, "logical_path": TARGET}, headers=AUTH).status_code == 422


@pytest.mark.parametrize(
    "case",
    [
        "auth",
        "path",
        "corrupt",
        "gap",
        "oversize",
        "short",
        "symlink",
        "parent",
        "hardlink",
        "promote",
    ],
)
def test_unsafe_requests_never_expose_or_overwrite_an_output(api, case):
    c, root = api
    content = b"abc"
    kwargs = {}
    if case == "auth":
        kwargs["headers"] = {
            "Authorization": "Bearer wrong",
            "Content-Length": "3",
            "X-Topodrome-Sha256": hashlib.sha256(content).hexdigest(),
        }
        expected = 403
    elif case == "path":
        kwargs["path"], expected = TARGET, 422
    elif case == "corrupt":
        kwargs["headers"] = {**AUTH, "Content-Length": "3", "X-Topodrome-Sha256": "0" * 64}
        expected = 409
    elif case == "gap":
        kwargs.update(offset=2, total=5)
        expected = 409
    elif case == "oversize":
        content, expected = b"a" * (MAX_UPLOAD_CHUNK_BYTES + 1), 422
    elif case == "short":
        kwargs["headers"] = {
            **AUTH,
            "Content-Length": "4",
            "X-Topodrome-Sha256": hashlib.sha256(content).hexdigest(),
        }
        kwargs["total"], expected = 4, 400
    elif case in {"symlink", "hardlink"}:
        (root / PREFIX).parent.mkdir(parents=True)
        protected = root / "protected"
        protected.write_bytes(b"untouched")
        if case == "symlink":
            (root / PREFIX).symlink_to(protected)
        else:
            os.link(protected, root / PREFIX)
        expected = 409
    elif case == "parent":
        elsewhere = root / "elsewhere"
        elsewhere.mkdir()
        (root / "jobs").symlink_to(elsewhere)
        expected = 409
    else:
        assert put(c, content).status_code == 200
        response = c.post(
            URL,
            params={"site_id": "site", "logical_path": PREFIX, "destination": "jobs/other/output"},
            headers=AUTH,
        )
        assert response.status_code == 422
        return
    assert put(c, content, **kwargs).status_code == expected
    if (root / "protected").exists():
        assert (root / "protected").read_bytes() == b"untouched"
    assert not (root / TARGET).exists()


def test_empty_upload_and_prefix_removal_are_durable(api):
    c, root = api
    assert put(c, b"").status_code == 200
    assert (root / PREFIX).stat().st_size == 0
    params = {"site_id": "site", "logical_path": PREFIX}
    assert c.delete(URL, params=params, headers=AUTH).status_code == 204
    assert c.get(URL, params=params, headers=AUTH).status_code == 404


def test_real_client_restarts_at_committed_nonzero_offset(api, tmp_path):
    c, root = api
    source = tmp_path / "source"
    content = os.urandom(2 * UPLOAD_CHUNK_BYTES + 31)
    source.write_bytes(content)
    canceled = threading.Event()
    offsets = []

    def transport(request):
        if request.method == "PUT":
            offsets.append(int(request.url.params["offset"]))
        response = c.request(
            request.method, str(request.url), headers=dict(request.headers), content=request.read()
        )
        return httpx.Response(
            response.status_code, headers=response.headers, content=response.content
        )

    def client():
        return SchedulerBulkUploadClient(
            "http://testserver", "site", "enrolled", transport=httpx.MockTransport(transport)
        )

    first = client()
    try:
        with cancellable_transfers(canceled), observe_transfers(lambda *args: canceled.set()):
            with pytest.raises(TransferCancelled):
                first.write(PREFIX, source, 0, len(content))
    finally:
        first.close()
    assert not (root / TARGET).exists()
    assert (root / PREFIX).stat().st_size == UPLOAD_CHUNK_BYTES
    assert (
        hash_beneath(root, PREFIX).sha256
        == hashlib.sha256(content[:UPLOAD_CHUNK_BYTES]).hexdigest()
    )
    resumed = client()
    try:
        assert resumed.stat(PREFIX)["Size"] == UPLOAD_CHUNK_BYTES
        resumed.write(PREFIX, source, UPLOAD_CHUNK_BYTES, len(content))
        resumed.rename(PREFIX, TARGET)
        assert resumed.stat(PREFIX) is None
        assert hash_beneath(root, TARGET).sha256 == hashlib.sha256(content).hexdigest()
    finally:
        resumed.close()
    assert offsets == [0, UPLOAD_CHUNK_BYTES, 2 * UPLOAD_CHUNK_BYTES]


def test_immutable_target_is_deduplicated_not_replaced(api):
    c, root = api
    assert put(c, b"abc").status_code == 200
    target = root / TARGET
    target.write_bytes(b"different")
    inode = target.stat().st_ino
    params = {"site_id": "site", "logical_path": PREFIX, "destination": TARGET}
    assert c.post(URL, params=params, headers=AUTH).status_code == 409
    assert target.read_bytes() == b"different" and target.stat().st_ino == inode
    target.write_bytes(b"abc")
    assert c.post(URL, params=params, headers=AUTH).status_code == 204
    assert target.stat().st_ino == inode
    assert not (root / PREFIX).exists()


def gzip_put(client, content, *, path=PREFIX, offset=0, total=None, encoded=None, raw_length=None):
    encoded = gzip.compress(content, compresslevel=1, mtime=0) if encoded is None else encoded
    return client.put(
        URL,
        params={
            "site_id": "site",
            "logical_path": path,
            "offset": offset,
            "total": offset + len(content) if total is None else total,
        },
        headers={
            **AUTH,
            "Content-Length": str(len(encoded)),
            "Content-Encoding": "gzip",
            "X-Topodrome-Uncompressed-Length": str(
                len(content) if raw_length is None else raw_length
            ),
            "X-Topodrome-Sha256": hashlib.sha256(content).hexdigest(),
        },
        content=encoded,
    )


def test_gzip_acknowledges_raw_offsets_and_recovers_lost_ack(api):
    c, root = api
    first, second = b'{"text":"hello"}\n' * 8192, b"more text\n" * 4096
    total = len(first) + len(second)
    params = {"site_id": "site", "logical_path": PREFIX}
    before = c.get(URL, params=params, headers=AUTH)
    assert before.status_code == 404 and "gzip" in before.headers["accept-encoding"]
    response = gzip_put(c, first, total=total)
    assert response.status_code == 200
    assert response.json() == {"offset": len(first), "sha256": hashlib.sha256(first).hexdigest()}
    assert "gzip" in response.headers["accept-encoding"]
    assert (root / PREFIX).read_bytes() == first
    assert gzip_put(c, first, total=total).json() == response.json()
    assert (root / PREFIX).stat().st_size == len(first)
    assert gzip_put(c, second, offset=len(first), total=total).status_code == 200
    assert c.post(URL, params={**params, "destination": TARGET}, headers=AUTH).status_code == 204
    assert (root / TARGET).read_bytes() == first + second
    assert hash_beneath(root, TARGET).sha256 == hashlib.sha256(first + second).hexdigest()


@pytest.mark.parametrize(
    "case", ["truncated", "crc", "trailer", "member", "bomb", "too-large", "offset"]
)
def test_unsafe_compressed_bodies_do_not_create_a_prefix(api, case):
    c, root = api
    content = b"result text\n" * 8192
    encoded = gzip.compress(content)
    kwargs = {}
    expected = 409
    if case == "truncated":
        encoded = encoded[:-8]
    elif case == "crc":
        encoded = encoded[:-8] + bytes([encoded[-8] ^ 1]) + encoded[-7:]
    elif case == "trailer":
        encoded += b"trailing data"
    elif case == "member":
        encoded += gzip.compress(b"second member")
    elif case == "bomb":
        kwargs["raw_length"] = 100
    elif case == "too-large":
        kwargs["raw_length"] = MAX_UPLOAD_CHUNK_BYTES + 1
        expected = 422
    else:
        kwargs["total"] = len(content) - 1
        expected = 422
    response = gzip_put(c, content, encoded=encoded, **kwargs)
    assert response.status_code == expected
    assert not (root / PREFIX).exists() and not (root / TARGET).exists()


def test_unsupported_or_undeclared_encoding_is_not_retried_as_raw(api):
    c, root = api
    raw = b"hello" * 100
    encoded = gzip.compress(raw)
    headers = {
        **AUTH,
        "Content-Length": str(len(encoded)),
        "X-Topodrome-Sha256": hashlib.sha256(raw).hexdigest(),
    }
    for encoding, status in (("gzip", 422), ("deflate", 415)):
        response = put(
            c, encoded, total=len(raw), headers={**headers, "Content-Encoding": encoding}
        )
        assert response.status_code == status
        assert not (root / PREFIX).exists()


@pytest.mark.asyncio
async def test_slow_uploads_do_not_starve_prefix_metadata(tmp_path):
    import asyncio

    root = tmp_path / "authority"
    root.mkdir()
    app = FastAPI()
    app.include_router(
        bulk_upload_router(root, lambda site: "enrolled" if site == "site" else None)
    )
    (root / PREFIX).parent.mkdir(parents=True)
    (root / PREFIX).write_bytes(b"already acknowledged")
    released = asyncio.Event()
    entered = [asyncio.Event(), asyncio.Event()]
    raw = b"xyz"

    async def body(index):
        yield raw[:1]
        entered[index].set()
        await released.wait()
        yield raw[1:]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        tasks = [
            asyncio.create_task(
                client.put(
                    URL,
                    params={
                        "site_id": "site",
                        "logical_path": PREFIX.replace(".output", f".slow-{i}"),
                        "offset": 0,
                        "total": len(raw),
                    },
                    headers={
                        **AUTH,
                        "Content-Length": str(len(raw)),
                        "X-Topodrome-Sha256": hashlib.sha256(raw).hexdigest(),
                    },
                    content=body(i),
                )
            )
            for i in range(2)
        ]
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 5)
            response = await asyncio.wait_for(
                client.get(
                    URL,
                    params={"site_id": "site", "logical_path": PREFIX},
                    headers=AUTH,
                ),
                5,
            )
            assert response.status_code == 200
            assert response.json()["Size"] == len(b"already acknowledged")
            assert "gzip" in response.headers["accept-encoding"]
            third = await client.put(
                URL,
                params={
                    "site_id": "site",
                    "logical_path": PREFIX.replace(".output", ".third"),
                    "offset": 0,
                    "total": len(raw),
                },
                headers={**AUTH, "X-Topodrome-Sha256": hashlib.sha256(raw).hexdigest()},
                content=raw,
            )
            assert third.status_code == 503
        finally:
            released.set()
            completed = await asyncio.gather(*tasks)
        assert [response.status_code for response in completed] == [200, 200]


@pytest.mark.asyncio
async def test_disconnected_body_does_not_write_and_releases_reception_capacity(tmp_path):
    from starlette.requests import ClientDisconnect

    root = tmp_path / "authority"
    root.mkdir()
    app = FastAPI()
    app.include_router(
        bulk_upload_router(root, lambda site: "enrolled" if site == "site" else None)
    )

    async def disconnected():
        yield b"x"
        raise ClientDisconnect()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        params = {"site_id": "site", "logical_path": PREFIX, "offset": 0, "total": 3}
        headers = {
            **AUTH,
            "Content-Length": "3",
            "X-Topodrome-Sha256": hashlib.sha256(b"xyz").hexdigest(),
        }
        for _ in range(3):
            response = await client.put(URL, params=params, headers=headers, content=disconnected())
            assert response.status_code == 408 and not (root / PREFIX).exists()
        response = await client.put(URL, params=params, headers=headers, content=b"xyz")
        assert response.status_code == 200 and (root / PREFIX).read_bytes() == b"xyz"


def test_compression_capability_is_not_advertised_to_unauthenticated_clients(api):
    c, _root = api
    response = c.get(URL, params={"site_id": "site", "logical_path": PREFIX})
    assert response.status_code == 403 and "accept-encoding" not in response.headers
