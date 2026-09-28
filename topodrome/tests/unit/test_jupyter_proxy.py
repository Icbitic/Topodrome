from __future__ import annotations

import asyncio
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from topodrome.deployment.jupyter_proxy import (
    JupyterLoopbackProxy,
    ProxyConfigurationError,
    ProxyLimits,
    ProxySettings,
    UpstreamWebSocket,
)

pytestmark = pytest.mark.unit


def test_proxy_configuration_requires_loopback_and_credential_free_upstream() -> None:
    with pytest.raises(ProxyConfigurationError, match="loopback"):
        ProxySettings(
            upstream_url="https://jupyter.invalid",
            bind_host="0.0.0.0",  # noqa: S104 - rejection test
        )
    with pytest.raises(ProxyConfigurationError, match="credential-free"):
        ProxySettings(upstream_url="https://user:password@jupyter.invalid")
    with pytest.raises(ProxyConfigurationError, match="credential-free"):
        ProxySettings(upstream_url="https://jupyter.invalid/?token=secret")

    settings = ProxySettings(
        upstream_url="http://jupyter.invalid:7798/base/",
        bind_host="::1",
    )
    assert settings.upstream_url.endswith("/")


async def test_http_proxy_rewrites_authority_preserves_auth_and_strips_hop_headers() -> None:
    secret = "authorization-must-not-be-logged"  # noqa: S105 - forwarding canary
    observed: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            201,
            content=b'{"name":"terminal"}',
            headers={
                "Content-Type": "application/json",
                "Connection": "x-upstream-hop",
                "X-Upstream-Hop": "remove-me",
                "X-Keep": "yes",
            },
        )

    proxy = JupyterLoopbackProxy(
        ProxySettings(upstream_url="http://jupyter.invalid:7798/base"),
        http_transport=httpx.MockTransport(upstream),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy),
            base_url="http://127.0.0.1:17798",
        ) as client:
            response = await client.post(
                "/api/terminals?fresh=1",
                content=b"payload",
                headers={
                    "Authorization": f"token {secret}",
                    "Connection": "x-client-hop",
                    "X-Client-Hop": "remove-me",
                    "Proxy-Authorization": "remove-me",
                    "X-Forwarded-For": "203.0.113.9",
                    "X-Keep": "client",
                },
            )
    finally:
        await proxy.aclose()

    assert response.status_code == 201
    assert response.content == b'{"name":"terminal"}'
    assert response.headers["x-keep"] == "yes"
    assert "x-upstream-hop" not in response.headers
    assert len(observed) == 1
    request = observed[0]
    assert str(request.url) == "http://jupyter.invalid:7798/base/api/terminals?fresh=1"
    assert request.headers["host"] == "jupyter.invalid:7798"
    assert request.headers["authorization"] == f"token {secret}"
    assert request.headers["x-keep"] == "client"
    assert "x-client-hop" not in request.headers
    assert "proxy-authorization" not in request.headers
    assert "x-forwarded-for" not in request.headers


async def test_http_proxy_bounds_request_and_response_bodies() -> None:
    calls = 0

    def upstream(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"response-too-large")

    proxy = JupyterLoopbackProxy(
        ProxySettings(
            upstream_url="https://jupyter.invalid",
            limits=ProxyLimits(
                max_request_body_bytes=4,
                max_response_body_bytes=4,
            ),
        ),
        http_transport=httpx.MockTransport(upstream),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy),
            base_url="http://127.0.0.1:17798",
        ) as client:
            request_rejected = await client.post("/api/terminals", content=b"12345")
            response_rejected = await client.get("/api/status")
    finally:
        await proxy.aclose()

    assert request_rejected.status_code == 413
    assert calls == 1
    assert response_rejected.status_code == 502
    assert response_rejected.text == "upstream response too large"


async def test_http_proxy_returns_generic_timeout_without_upstream_details() -> None:
    secret = "must-not-escape-in-errors"  # noqa: S105 - redaction canary

    def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"authorization={secret}", request=request)

    proxy = JupyterLoopbackProxy(
        ProxySettings(upstream_url="https://jupyter.invalid"),
        http_transport=httpx.MockTransport(upstream),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy),
            base_url="http://127.0.0.1:17798",
        ) as client:
            response = await client.get(
                "/api/status",
                headers={"Authorization": f"token {secret}"},
            )
    finally:
        await proxy.aclose()

    assert response.status_code == 504
    assert response.text == "upstream timeout"
    assert secret not in response.text


@dataclass(slots=True)
class _ConnectCall:
    url: str
    headers: tuple[tuple[str, str], ...]
    origin: str
    subprotocols: tuple[str, ...]
    open_timeout_seconds: float
    close_timeout_seconds: float
    max_message_bytes: int


_UPSTREAM_CLOSED = object()


class _FakeUpstream:
    def __init__(self, *, subprotocol: str | None = None) -> None:
        self._subprotocol = subprotocol
        self.incoming: asyncio.Queue[str | bytes | object] = asyncio.Queue()
        self.sent: list[str | bytes] = []
        self.sent_event = asyncio.Event()
        self.closed = False
        self.close_code: int | None = None

    @property
    def subprotocol(self) -> str | None:
        return self._subprotocol

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)
        self.sent_event.set()

    async def recv(self) -> str | bytes | None:
        value = await self.incoming.get()
        if value is _UPSTREAM_CLOSED:
            return None
        assert isinstance(value, (str, bytes))
        return value

    async def aclose(self, *, code: int = 1_000) -> None:
        self.closed = True
        self.close_code = code


class _FakeConnector:
    def __init__(
        self,
        upstream: _FakeUpstream | None = None,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.upstream = upstream
        self.failure = failure
        self.calls: list[_ConnectCall] = []

    async def connect(
        self,
        url: str,
        headers: Sequence[tuple[str, str]],
        *,
        origin: str,
        subprotocols: Sequence[str],
        open_timeout_seconds: float,
        close_timeout_seconds: float,
        max_message_bytes: int,
    ) -> UpstreamWebSocket:
        self.calls.append(
            _ConnectCall(
                url=url,
                headers=tuple(headers),
                origin=origin,
                subprotocols=tuple(subprotocols),
                open_timeout_seconds=open_timeout_seconds,
                close_timeout_seconds=close_timeout_seconds,
                max_message_bytes=max_message_bytes,
            )
        )
        if self.failure is not None:
            raise self.failure
        assert self.upstream is not None
        return self.upstream


class _ASGIWebSocket:
    def __init__(
        self,
        app: JupyterLoopbackProxy,
        path: str,
        query: str = "",
        *,
        headers: dict[str, str] | None = None,
        subprotocols: Sequence[str] = (),
    ) -> None:
        raw_headers = [(b"host", b"127.0.0.1:17798")]
        raw_headers.extend(
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in (headers or {}).items()
        )
        self.incoming: asyncio.Queue[MutableMapping[str, Any]] = asyncio.Queue()
        self.outgoing: asyncio.Queue[MutableMapping[str, Any]] = asyncio.Queue()
        scope: MutableMapping[str, Any] = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "scheme": "ws",
            "http_version": "1.1",
            "server": ("127.0.0.1", 17_798),
            "client": ("127.0.0.1", 40_000),
            "root_path": "",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": query.encode("ascii"),
            "headers": raw_headers,
            "subprotocols": list(subprotocols),
        }
        self.task = asyncio.create_task(app(scope, self.incoming.get, self.outgoing.put))

    async def connect(self) -> MutableMapping[str, Any]:
        await self.incoming.put({"type": "websocket.connect"})
        return await asyncio.wait_for(self.outgoing.get(), timeout=2)

    async def send_text(self, value: str) -> None:
        await self.incoming.put({"type": "websocket.receive", "text": value})

    async def disconnect(self) -> None:
        await self.incoming.put({"type": "websocket.disconnect", "code": 1_000})
        await asyncio.wait_for(self.task, timeout=2)


async def _wait_for_sent(upstream: _FakeUpstream, expected: str | bytes) -> None:
    await asyncio.wait_for(upstream.sent_event.wait(), timeout=2)
    assert expected in upstream.sent


async def test_terminal_websocket_is_bidirectional_and_rewrites_origin() -> None:
    secret = "websocket-authorization-canary"  # noqa: S105 - forwarding canary
    protocol = "v1.kernel.websocket.jupyter.org"
    upstream = _FakeUpstream(subprotocol=protocol)
    connector = _FakeConnector(upstream)
    proxy = JupyterLoopbackProxy(
        ProxySettings(upstream_url="http://jupyter.invalid:7798/base"),
        websocket_connector=connector,
    )
    socket = _ASGIWebSocket(
        proxy,
        "/terminals/websocket/owned_terminal",
        "fresh=1",
        headers={
            "Authorization": f"token {secret}",
            "Cookie": "session=allowed",
            "Origin": "http://127.0.0.1:17798",
            "Connection": "Upgrade, x-hop",
            "X-Hop": "remove-me",
            "Sec-WebSocket-Key": "remove-me",
            "X-Forwarded-For": "203.0.113.10",
        },
        subprotocols=(protocol,),
    )
    try:
        accepted = await socket.connect()
        assert accepted == {"type": "websocket.accept", "subprotocol": protocol}
        await socket.send_text("from-client")
        await _wait_for_sent(upstream, "from-client")
        await upstream.incoming.put(b"from-upstream")
        relayed = await asyncio.wait_for(socket.outgoing.get(), timeout=2)
        assert relayed == {"type": "websocket.send", "bytes": b"from-upstream"}
        await socket.disconnect()
    finally:
        if not socket.task.done():
            socket.task.cancel()
            await asyncio.gather(socket.task, return_exceptions=True)
        await proxy.aclose()

    assert upstream.closed
    assert len(connector.calls) == 1
    call = connector.calls[0]
    assert call.url == (
        "ws://jupyter.invalid:7798/base/terminals/websocket/owned_terminal?fresh=1"
    )
    assert call.origin == "http://jupyter.invalid:7798"
    assert call.subprotocols == (protocol,)
    forwarded = {name.casefold(): value for name, value in call.headers}
    assert forwarded["authorization"] == f"token {secret}"
    assert forwarded["cookie"] == "session=allowed"
    assert "host" not in forwarded
    assert "origin" not in forwarded
    assert "connection" not in forwarded
    assert "x-hop" not in forwarded
    assert "sec-websocket-key" not in forwarded
    assert "x-forwarded-for" not in forwarded


async def test_websocket_message_limit_and_connect_failure_close_safely() -> None:
    upstream = _FakeUpstream()
    proxy = JupyterLoopbackProxy(
        ProxySettings(
            upstream_url="https://jupyter.invalid",
            limits=ProxyLimits(max_websocket_message_bytes=4),
        ),
        websocket_connector=_FakeConnector(upstream),
    )
    socket = _ASGIWebSocket(proxy, "/terminals/websocket/owned")
    try:
        assert (await socket.connect())["type"] == "websocket.accept"
        await socket.send_text("12345")
        closed = await asyncio.wait_for(socket.outgoing.get(), timeout=2)
        assert closed == {"type": "websocket.close", "code": 1_009}
        await asyncio.wait_for(socket.task, timeout=2)
    finally:
        if not socket.task.done():
            socket.task.cancel()
            await asyncio.gather(socket.task, return_exceptions=True)
        await proxy.aclose()
    assert upstream.close_code == 1_009

    failed_proxy = JupyterLoopbackProxy(
        ProxySettings(upstream_url="https://jupyter.invalid"),
        websocket_connector=_FakeConnector(failure=RuntimeError("authorization=secret")),
    )
    failed = _ASGIWebSocket(failed_proxy, "/terminals/websocket/owned")
    try:
        assert await failed.connect() == {"type": "websocket.close", "code": 1_013}
        await asyncio.wait_for(failed.task, timeout=2)
    finally:
        if not failed.task.done():
            failed.task.cancel()
            await asyncio.gather(failed.task, return_exceptions=True)
        await failed_proxy.aclose()
