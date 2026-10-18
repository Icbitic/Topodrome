from __future__ import annotations

import hashlib
import threading

import httpx
import pytest

from tests.unit.test_storage_http_mirror import FakeAuthority
from topodrome.storage.base import ObjectDigest, StorageUnavailableError, StorageVerificationError
from topodrome.storage.blobs import blob_path, publish_artifact_blob
from topodrome.storage.http_mirror import HttpMirrorStorage
from topodrome.storage.progress import TransferCancelled, cancellable_transfers
from topodrome.storage.small_files import SchedulerSmallFileClient, SmallFileFallback

pytestmark = pytest.mark.unit


def identity(content):
    return ObjectDigest(len(content), hashlib.sha256(content).hexdigest())


def client(handler, **kwargs):
    return SchedulerSmallFileClient(
        "wss://scheduler/prefix/api/v1/agents/control",
        "site",
        "secret",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def response(content):
    return httpx.Response(
        200,
        content=content,
        headers={
            "ETag": f'"{identity(content).sha256}"',
            "Content-Length": str(len(content)),
        },
    )


def test_download_is_one_authenticated_request_with_exact_pin():
    calls = []

    def serve(request):
        calls.append(request)
        assert request.method == "GET"
        assert str(request.url).startswith(
            "https://scheduler/prefix/api/v1/agents/storage/small-file?"
        )
        assert request.headers["authorization"] == "Bearer secret"
        assert request.headers["if-match"] == f'"{identity(b"x").sha256}"'
        return response(b"x")

    c = client(serve)
    try:
        assert c.download("blobs/file", expected=identity(b"x")) == (b"x", identity(b"x"))
        assert c.download("blobs/file", expected=identity(b"x")) == (b"x", identity(b"x"))
        assert len(calls) == 2
    finally:
        c.close()
    assert c._client.is_closed


@pytest.mark.parametrize("failure", ["sha", "size", "etag", "length", "oversized"])
def test_download_never_exposes_unverified_bytes(failure):
    def serve(_):
        r = response(b"x" if failure != "oversized" else b"x" * 17)
        if failure == "etag":
            r.headers["etag"] = '"wrong"'
        if failure == "length":
            r.headers["content-length"] = "3"
        return r

    c = client(serve, max_bytes=16)
    expected = identity(b"y" if failure == "sha" else b"x")
    try:
        with pytest.raises(StorageVerificationError):
            c.download(
                "jobs/result",
                expected=expected,
                expected_size_bytes=2 if failure == "size" else None,
            )
    finally:
        c.close()


@pytest.mark.parametrize("code", [403, 409, 412, 500, 503])
def test_no_integrity_or_authentication_failure_downgrade(code):
    c = client(lambda _: httpx.Response(code, json={"detail": "rejected"}))
    try:
        with pytest.raises(StorageUnavailableError):
            c.download("jobs/result")
        assert c.accepts("jobs/result")
    finally:
        c.close()


def test_missing_route_is_remembered_but_missing_file_is_not():
    c = client(lambda _: httpx.Response(404, json={"detail": "Not Found"}))
    try:
        with pytest.raises(SmallFileFallback):
            c.download("jobs/result")
        assert not c.accepts("jobs/result")
    finally:
        c.close()
    c = client(lambda _: httpx.Response(404, json={"detail": "object not found"}))
    try:
        with pytest.raises(FileNotFoundError):
            c.download("jobs/result")
        assert c.accepts("jobs/result")
    finally:
        c.close()


def test_small_blob_combines_claim_upload_and_verify(tmp_path):
    calls = []

    def serve(request):
        calls.append(request)
        assert request.method == "PUT"
        assert request.url.params["logical_path"] == blob_path(identity(b"small").sha256)
        assert request.url.params["job_id"] == "job"
        assert request.url.params["attempt_id"] == "attempt"
        assert request.content == b"small"
        return httpx.Response(200, json={"size_bytes": 5, "sha256": identity(b"small").sha256})

    remote = FakeAuthority()
    c = client(serve)
    mirror = HttpMirrorStorage(
        tmp_path / "cache",
        remote.remote_root,
        bulk_transport=remote,
        authoritative_reader=remote,
        small_file_client=c,
        min_free_space_gb=0.001,
    )
    source = tmp_path / "result"
    source.write_bytes(b"small")
    try:
        verified = publish_artifact_blob(mirror, source, job_id="job", attempt_id="attempt")
        assert verified.digest == identity(b"small")
        assert verified.local_path.read_bytes() == b"small"
        assert len(calls) == 1
        assert not remote.calls
        assert not (mirror.root / ".blob-index").exists()
    finally:
        mirror.close()


def test_pinned_download_avoids_stat_and_caches_only_verified_bytes(tmp_path):
    calls = []
    c = client(lambda request: calls.append(request) or response(b"small"))
    remote = FakeAuthority()
    mirror = HttpMirrorStorage(
        tmp_path / "cache",
        remote.remote_root,
        bulk_transport=remote,
        authoritative_reader=remote,
        small_file_client=c,
        min_free_space_gb=0.001,
    )
    try:
        path = mirror.resolve_pinned(
            "jobs/result", expected_size_bytes=5, expected_sha256=identity(b"small").sha256
        )
        assert path.read_bytes() == b"small"
        assert mirror.resolve_pinned("jobs/result", expected_size_bytes=5) == path
        assert len(calls) == 1
        assert not remote.calls
    finally:
        mirror.close()


def test_bad_pin_and_cancellation_leave_no_visible_file(tmp_path):
    c = client(lambda _: response(b"bad"))
    remote = FakeAuthority()
    mirror = HttpMirrorStorage(
        tmp_path / "cache",
        remote.remote_root,
        bulk_transport=remote,
        authoritative_reader=remote,
        small_file_client=c,
        min_free_space_gb=0.001,
    )
    try:
        with pytest.raises(StorageVerificationError):
            mirror.resolve_pinned(
                "jobs/result", expected_size_bytes=3, expected_sha256=identity(b"pin").sha256
            )
        assert not (mirror.root / "jobs/result").exists()
        event = threading.Event()
        event.set()
        with pytest.raises(TransferCancelled), cancellable_transfers(event):
            mirror.resolve("jobs/result")
        assert not remote.calls
    finally:
        mirror.close()


def test_large_objects_use_streaming_without_a_tiny_request(tmp_path):
    def unexpected(_):
        raise AssertionError("bulk object must not use the tiny-file endpoint")

    c = client(unexpected, max_bytes=1)
    remote = FakeAuthority()
    remote.objects["jobs/result"] = b"large"
    mirror = HttpMirrorStorage(
        tmp_path / "cache",
        remote.remote_root,
        bulk_transport=remote,
        authoritative_reader=remote,
        small_file_client=c,
        min_free_space_gb=0.001,
    )
    try:
        assert mirror.resolve_pinned("jobs/result", expected_size_bytes=5).read_bytes() == b"large"
        assert remote.calls
    finally:
        mirror.close()


def test_upload_ack_mismatch_is_not_success(tmp_path):
    source = tmp_path / "file"
    source.write_bytes(b"x")
    c = client(
        lambda _: httpx.Response(
            200,
            json={
                "size_bytes": 1,
                "sha256": identity(b"y").sha256,
            },
        )
    )
    try:
        with pytest.raises(StorageVerificationError):
            c.upload("jobs/job/attempts/attempt/committed/result", source, identity(b"x"))
    finally:
        c.close()


def test_unsupported_and_oversized_uploads_do_not_read_file(tmp_path):
    c = client(lambda _: pytest.fail("no HTTP request"), max_bytes=1)
    try:
        with pytest.raises(SmallFileFallback):
            c.upload("jobs/job/attempts/attempt/result", tmp_path / "missing", identity(b"xx"))
        assert not c.accepts(".blob-index/index.sqlite3")
    finally:
        c.close()
