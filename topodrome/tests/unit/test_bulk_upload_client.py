from __future__ import annotations

import gzip
import hashlib
import os

import httpx
import pytest

from topodrome.storage.base import StorageVerificationError
from topodrome.storage.bulk import BulkTransportError
from topodrome.storage.bulk_upload import SchedulerBulkUploadClient, private_upload_path

pytestmark = pytest.mark.unit
PREFIX = "jobs/job/attempts/attempt/.result.topodrome-upload-" + "f" * 32


@pytest.mark.parametrize(
    "status,retryable", [(403, False), (404, False), (503, True), (408, True), (507, False)]
)
def test_only_transient_failures_are_retryable(tmp_path, status, retryable):
    source = tmp_path / "source"
    source.write_bytes(b"data")
    client = SchedulerBulkUploadClient(
        "http://test",
        "site",
        "private",
        transport=httpx.MockTransport(lambda r: httpx.Response(status)),
    )
    try:
        with pytest.raises(BulkTransportError) as exc:
            client.write(PREFIX, source, 0, 4)
        assert exc.value.retryable is retryable
        assert "private" not in str(exc.value)
    finally:
        client.close()


def test_transport_errors_and_corrupt_ack_do_not_report_success(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"data")
    for handler, expected in [
        (
            lambda r: httpx.Response(200, json={"offset": 4, "sha256": "0" * 64}),
            StorageVerificationError,
        ),
        (lambda r: httpx.Response(200, content=b"x" * 4097), StorageVerificationError),
        (lambda r: (_ for _ in ()).throw(httpx.ReadError("private details")), BulkTransportError),
    ]:
        client = SchedulerBulkUploadClient(
            "http://test", "site", "private", transport=httpx.MockTransport(handler)
        )
        try:
            with pytest.raises(expected):
                client.write(PREFIX, source, 0, 4)
        finally:
            client.close()


@pytest.mark.parametrize(
    "path",
    [
        "jobs/job/result",
        "../private",
        "metadata/.x.topodrome-upload-" + "a" * 32,
        "jobs/.x.topodrome-upload-" + "a" * 32,
    ],
)
def test_only_owned_private_prefixes_are_uploadable(path):
    with pytest.raises(ValueError):
        private_upload_path(path)


def test_empty_file_is_created_and_acknowledged(tmp_path):
    source = tmp_path / "empty"
    source.touch()
    seen = []

    def handle(request):
        seen.append(request.read())
        return httpx.Response(200, json={"offset": 0, "sha256": hashlib.sha256(b"").hexdigest()})

    client = SchedulerBulkUploadClient(
        "http://test", "site", "private", transport=httpx.MockTransport(handle)
    )
    try:
        client.write(PREFIX, source, 0, 0)
        assert seen == [b""]
    finally:
        client.close()


@pytest.mark.parametrize("advertised", [False, True])
@pytest.mark.parametrize("compressible", [False, True])
def test_body_compression_requires_authority_negotiation(tmp_path, advertised, compressible):
    content = b"json text\n" * 13108 if compressible else os.urandom(128 * 1024)
    source = tmp_path / "source"
    source.write_bytes(content)
    requests = []

    def handle(request):
        encoding_headers = {"Accept-Encoding": "gzip, identity"} if advertised else {}
        if request.method == "GET":
            return httpx.Response(404, headers=encoding_headers)
        raw = request.read()
        encoding = request.headers.get("content-encoding", "identity")
        decoded = gzip.decompress(raw) if encoding == "gzip" else raw
        assert int(request.headers["content-length"]) == len(raw)
        assert request.headers["x-topodrome-sha256"] == hashlib.sha256(decoded).hexdigest()
        if encoding == "gzip":
            assert int(request.headers["x-topodrome-uncompressed-length"]) == len(decoded)
            assert b"private" not in gzip.decompress(raw)
        offset = int(request.url.params["offset"])
        assert decoded == content[offset : offset + len(decoded)]
        requests.append(encoding)
        return httpx.Response(
            200,
            headers=encoding_headers,
            json={"offset": offset + len(decoded), "sha256": hashlib.sha256(decoded).hexdigest()},
        )

    client = SchedulerBulkUploadClient(
        "http://test", "site", "private", transport=httpx.MockTransport(handle)
    )
    try:
        assert client.stat(PREFIX) is None
        client.write(PREFIX, source, 0, len(content))
        assert requests[0] == ("gzip" if advertised and compressible else "identity")
    finally:
        client.close()


def test_server_rollback_resets_compression_capability(tmp_path):
    content = b"data\n" * 30000
    source = tmp_path / "source"
    source.write_bytes(content)
    advertise = True
    encodings = []

    def handle(request):
        headers = {"Accept-Encoding": "gzip, identity"} if advertise else {}
        if request.method == "GET":
            return httpx.Response(404, headers=headers)
        encoding = request.headers.get("content-encoding", "identity")
        encodings.append(encoding)
        decoded = gzip.decompress(request.read()) if encoding == "gzip" else request.read()
        return httpx.Response(
            200,
            headers=headers,
            json={
                "offset": int(request.url.params["offset"]) + len(decoded),
                "sha256": hashlib.sha256(decoded).hexdigest(),
            },
        )

    client = SchedulerBulkUploadClient(
        "http://test", "site", "private", transport=httpx.MockTransport(handle)
    )
    try:
        client.stat(PREFIX)
        assert client._gzip_supported
        advertise = False
        client.stat(PREFIX)
        client.write(PREFIX, source, 0, len(content))
        assert set(encodings) == {"identity"}
    finally:
        client.close()


def test_encoded_body_still_requires_raw_acknowledgement(tmp_path):
    content = b"json text\n" * 16000
    source = tmp_path / "source"
    source.write_bytes(content)
    seen = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(404, headers={"Accept-Encoding": "gzip, identity"})
        assert request.headers["content-encoding"] == "gzip"
        seen.append(gzip.decompress(request.read()))
        return httpx.Response(200, json={"offset": len(request.read()), "sha256": "0" * 64})

    client = SchedulerBulkUploadClient(
        "http://test", "site", "private", transport=httpx.MockTransport(handle)
    )
    try:
        client.stat(PREFIX)
        with pytest.raises(StorageVerificationError):
            client.write(PREFIX, source, 0, len(content))
        assert seen == [content[: 128 * 1024]]
    finally:
        client.close()
