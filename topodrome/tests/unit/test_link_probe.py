from __future__ import annotations

import hashlib

import httpx
import pytest

from topodrome.storage.base import StorageVerificationError
from topodrome.storage.link_probe import LINK_PROBE_BYTES, LinkProbeDeferred, probe_link

pytestmark = pytest.mark.unit


def good_response(request):
    if request.url.params.get("size") == "0":
        return httpx.Response(200, content=b"")
    if request.method == "POST":
        return httpx.Response(
            200,
            json={"bytes": LINK_PROBE_BYTES, "sha256": hashlib.sha256(request.content).hexdigest()},
        )
    content = b"x" * LINK_PROBE_BYTES
    return httpx.Response(
        200, content=content, headers={"x-topodrome-sha256": hashlib.sha256(content).hexdigest()}
    )


@pytest.mark.parametrize("status", [404, 408, 429, 500, 502, 503, 504])
def test_unsupported_or_busy_probe_is_not_a_speed_measurement(status):
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status))) as client:
        with pytest.raises(LinkProbeDeferred):
            probe_link(client, "http://127.0.0.1", "site")


@pytest.mark.parametrize("status", [301, 401, 403])
def test_redirect_or_auth_rejection_fails_closed(status):
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status))) as client:
        with pytest.raises(StorageVerificationError, match="authorization/protocol"):
            probe_link(client, "http://127.0.0.1", "site")


@pytest.mark.parametrize("fault", ["oversize", "corrupt", "encoding", "ack", "malformed"])
def test_response_memory_and_integrity_are_bounded(fault):
    def handle(request):
        response = good_response(request)
        if request.url.params.get("size") == "0":
            return response
        if fault in {"ack", "malformed"} and request.method == "POST":
            return httpx.Response(200, content=b"{}" if fault == "ack" else b"not json")
        if request.method == "GET":
            if fault == "oversize":
                return httpx.Response(200, content=b"x" * (LINK_PROBE_BYTES + 1))
            if fault == "corrupt":
                response.headers["x-topodrome-sha256"] = "0" * 64
            if fault == "encoding":
                response.headers["content-encoding"] = "gzip"
        return response

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(StorageVerificationError):
            probe_link(client, "http://127.0.0.1", "site")


def test_observation_deadline_spans_all_samples_and_stops_a_dribbling_body():
    now = [0.0]
    requests = []

    class Dribble(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(10):
                now[0] += 1
                yield b"x"

    def handle(request):
        requests.append(request)
        if request.url.params.get("size") == "0":
            return good_response(request)
        return httpx.Response(200, stream=Dribble())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert (
            probe_link(client, "http://127.0.0.1", "site", timeout_seconds=3, clock=lambda: now[0])
            is None
        )
    assert now[0] == 3
    assert len(requests) == 2


def test_network_error_declines_without_changing_authority():
    def handle(request):
        raise httpx.ConnectError("synthetic network failure")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert probe_link(client, "http://127.0.0.1", "site") is None
