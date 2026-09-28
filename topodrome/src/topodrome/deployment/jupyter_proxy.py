"""Loopback-only HTTP/WebSocket reverse proxy for Jupyter management.

The terminal bootstrap controller must use a loopback HTTP origin when an
upstream Jupyter Server is available only over plaintext HTTP.  A byte-for-byte
TCP forwarder is insufficient: it leaves the loopback ``Host`` and ``Origin``
headers intact, which causes some Jupyter frontends to reject or intermittently
drop requests.  This small ASGI proxy dials one fixed operator-configured
upstream and rewrites those authority headers without ever inspecting or
logging authorization values.

It is intentionally not a general-purpose forward proxy.  Incoming absolute
targets cannot select another authority, redirects aren't followed, hop-by-hop
headers are removed in both directions, and all buffered data has explicit
limits.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx

ASGIScope = MutableMapping[str, Any]
ASGIMessage = MutableMapping[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]

_ALLOWED_HTTP_METHODS = frozenset(
    {"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"}
)
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_FORWARDED_IDENTITY_HEADERS = frozenset(
    {"forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto"}
)


class ProxyConfigurationError(ValueError):
    """The proxy configuration would weaken its loopback-only boundary."""


class _RequestBodyTooLarge(Exception):
    pass


class _RequestBodyTimedOut(Exception):
    pass


class _ClientDisconnected(Exception):
    pass


class _UpstreamResponseTooLarge(Exception):
    pass


class _WebSocketMessageTooLarge(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ProxyLimits:
    """Resource and time bounds for one proxy process."""

    max_header_bytes: int = 65_536
    max_path_bytes: int = 8_192
    max_request_body_bytes: int = 1_048_576
    max_response_body_bytes: int = 2_097_152
    max_websocket_message_bytes: int = 1_048_576
    request_body_timeout_seconds: float = 15.0
    upstream_connect_timeout_seconds: float = 10.0
    upstream_request_timeout_seconds: float = 30.0
    websocket_open_timeout_seconds: float = 15.0
    websocket_close_timeout_seconds: float = 5.0
    websocket_max_lifetime_seconds: float = 1_800.0

    def __post_init__(self) -> None:
        byte_limits = (
            self.max_header_bytes,
            self.max_path_bytes,
            self.max_request_body_bytes,
            self.max_response_body_bytes,
            self.max_websocket_message_bytes,
        )
        time_limits = (
            self.request_body_timeout_seconds,
            self.upstream_connect_timeout_seconds,
            self.upstream_request_timeout_seconds,
            self.websocket_open_timeout_seconds,
            self.websocket_close_timeout_seconds,
            self.websocket_max_lifetime_seconds,
        )
        if any(value < 1 for value in byte_limits):
            raise ProxyConfigurationError("proxy byte limits must be positive")
        if any(value <= 0 for value in time_limits):
            raise ProxyConfigurationError("proxy timeouts must be positive")


@dataclass(frozen=True, slots=True)
class ProxySettings:
    """Validated listener and fixed-upstream configuration."""

    upstream_url: str
    bind_host: str = "127.0.0.1"
    bind_port: int = 17_798
    max_concurrency: int = 64
    limits: ProxyLimits = ProxyLimits()

    def __post_init__(self) -> None:
        if not _is_loopback_host(self.bind_host):
            raise ProxyConfigurationError("Jupyter proxy bind_host must be loopback")
        if not 1_024 <= self.bind_port <= 65_535:
            raise ProxyConfigurationError("Jupyter proxy bind_port must be unprivileged")
        if self.max_concurrency < 1:
            raise ProxyConfigurationError("Jupyter proxy max_concurrency must be positive")
        _validate_upstream(self.upstream_url)


class UpstreamWebSocket(Protocol):
    """Small client-WebSocket surface used by the relay and its tests."""

    @property
    def subprotocol(self) -> str | None: ...

    async def send(self, message: str | bytes) -> None: ...

    async def recv(self) -> str | bytes | None: ...

    async def aclose(self, *, code: int = 1_000) -> None: ...


class UpstreamWebSocketConnector(Protocol):
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
    ) -> UpstreamWebSocket: ...


class _WebsocketsAdapter:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @property
    def subprotocol(self) -> str | None:
        value = self._connection.subprotocol
        return value if isinstance(value, str) else None

    async def send(self, message: str | bytes) -> None:
        await self._connection.send(message)

    async def recv(self) -> str | bytes | None:
        from websockets.exceptions import ConnectionClosed

        try:
            value = await self._connection.recv()
        except ConnectionClosed:
            return None
        if not isinstance(value, (str, bytes)):
            raise TypeError("upstream WebSocket returned an invalid message")
        return value

    async def aclose(self, *, code: int = 1_000) -> None:
        await self._connection.close(code=code)


class WebsocketsUpstreamConnector:
    """Connect directly to the fixed upstream without environment proxies."""

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
        from websockets.asyncio.client import connect
        from websockets.typing import Origin, Subprotocol

        connection = await connect(
            url,
            additional_headers=list(headers),
            origin=cast(Origin, origin),
            subprotocols=[cast(Subprotocol, value) for value in subprotocols] or None,
            open_timeout=open_timeout_seconds,
            close_timeout=close_timeout_seconds,
            max_size=max_message_bytes,
            max_queue=16,
            compression=None,
            proxy=None,
        )
        return _WebsocketsAdapter(connection)


class JupyterLoopbackProxy:
    """Minimal ASGI reverse proxy for Jupyter REST and terminal WebSockets."""

    def __init__(
        self,
        settings: ProxySettings,
        *,
        http_transport: httpx.AsyncBaseTransport | None = None,
        websocket_connector: UpstreamWebSocketConnector | None = None,
    ) -> None:
        self.settings = settings
        self._upstream = _Upstream(settings.upstream_url)
        timeout = httpx.Timeout(
            settings.limits.upstream_request_timeout_seconds,
            connect=settings.limits.upstream_connect_timeout_seconds,
        )
        self._http = httpx.AsyncClient(
            timeout=timeout,
            transport=http_transport,
            follow_redirects=False,
            trust_env=False,
        )
        self._websocket_connector = websocket_connector or WebsocketsUpstreamConnector()
        self._closed = False

    async def __call__(
        self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await self._lifespan(receive, send)
            return
        if self._closed:
            if scope_type == "http":
                await _http_error(send, 503, b"proxy unavailable")
            elif scope_type == "websocket":
                await _websocket_close(send, 1_013)
            return
        if scope_type == "http":
            await self._http_request(scope, receive, send)
            return
        if scope_type == "websocket":
            await self._websocket(scope, receive, send)
            return
        raise RuntimeError("unsupported ASGI scope")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._http.aclose()

    async def _lifespan(self, receive: ASGIReceive, send: ASGISend) -> None:
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message_type == "lifespan.shutdown":
                await self.aclose()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _http_request(
        self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        method = str(scope.get("method", "")).upper()
        if method not in _ALLOWED_HTTP_METHODS:
            await _http_error(send, 405, b"method not allowed")
            return
        try:
            raw_path, query = _request_target(scope, self.settings.limits)
            headers = _forward_request_headers(scope, self._upstream.authority)
            _require_header_limit(scope, self.settings.limits.max_header_bytes)
            body = await _read_body(receive, self.settings.limits)
        except _RequestBodyTooLarge:
            await _http_error(send, 413, b"request body too large")
            return
        except _RequestBodyTimedOut:
            await _http_error(send, 408, b"request body timed out")
            return
        except _ClientDisconnected:
            return
        except ValueError as exc:
            status = 431 if str(exc) == "headers too large" else 414
            await _http_error(send, status, b"request metadata rejected")
            return

        target = self._upstream.http_target(raw_path, query)
        try:
            async with self._http.stream(
                method,
                target,
                headers=headers,
                content=body,
            ) as response:
                response_body = await _bounded_response_body(
                    response,
                    maximum=self.settings.limits.max_response_body_bytes,
                )
                response_headers = _forward_response_headers(
                    response.headers.multi_items(),
                    content_length=len(response_body),
                )
                await send(
                    {
                        "type": "http.response.start",
                        "status": response.status_code,
                        "headers": response_headers,
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": bytes(response_body),
                        "more_body": False,
                    }
                )
        except httpx.TimeoutException:
            await _http_error(send, 504, b"upstream timeout")
        except _UpstreamResponseTooLarge:
            await _http_error(send, 502, b"upstream response too large")
        except Exception:
            await _http_error(send, 502, b"upstream unavailable")

    async def _websocket(
        self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        try:
            raw_path, query = _request_target(scope, self.settings.limits)
            _require_header_limit(scope, self.settings.limits.max_header_bytes)
            headers = _forward_websocket_headers(scope)
            subprotocols = _subprotocols(scope)
        except ValueError:
            await _websocket_close(send, 1_008)
            return

        initial = await receive()
        if initial.get("type") != "websocket.connect":
            await _websocket_close(send, 1_008)
            return
        try:
            upstream = await self._websocket_connector.connect(
                self._upstream.websocket_target(raw_path, query),
                headers,
                origin=self._upstream.origin,
                subprotocols=subprotocols,
                open_timeout_seconds=self.settings.limits.websocket_open_timeout_seconds,
                close_timeout_seconds=self.settings.limits.websocket_close_timeout_seconds,
                max_message_bytes=self.settings.limits.max_websocket_message_bytes,
            )
        except Exception:
            await _websocket_close(send, 1_013)
            return

        selected = upstream.subprotocol
        if selected is not None and selected not in subprotocols:
            await upstream.aclose(code=1_002)
            await _websocket_close(send, 1_002)
            return
        close_code = 1_000
        downstream_connected = True
        relay_tasks: tuple[asyncio.Task[bool], ...] = ()
        try:
            await send({"type": "websocket.accept", "subprotocol": selected})
            async with asyncio.timeout(
                self.settings.limits.websocket_max_lifetime_seconds
            ):
                to_upstream = asyncio.create_task(
                    self._relay_to_upstream(receive, upstream),
                    name="jupyter-proxy-to-upstream",
                )
                to_downstream = asyncio.create_task(
                    self._relay_to_downstream(upstream, send),
                    name="jupyter-proxy-to-downstream",
                )
                relay_tasks = (to_upstream, to_downstream)
                done, pending = await asyncio.wait(
                    relay_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    try:
                        downstream_connected = downstream_connected and task.result()
                    except _WebSocketMessageTooLarge:
                        close_code = 1_009
                    except Exception:
                        close_code = 1_011
        except TimeoutError:
            close_code = 1_001
        finally:
            for task in relay_tasks:
                if not task.done():
                    task.cancel()
            if relay_tasks:
                await asyncio.gather(*relay_tasks, return_exceptions=True)
            with suppress(Exception):
                await upstream.aclose(code=close_code)
            if downstream_connected:
                await _websocket_close(send, close_code)

    async def _relay_to_upstream(
        self,
        receive: ASGIReceive,
        upstream: UpstreamWebSocket,
    ) -> bool:
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                return False
            if message_type != "websocket.receive":
                raise TypeError("invalid downstream WebSocket message")
            text = message.get("text")
            payload: str | bytes
            if isinstance(text, str):
                payload = text
            else:
                raw = message.get("bytes")
                if not isinstance(raw, bytes):
                    raise TypeError("downstream WebSocket message has no payload")
                payload = raw
            if _message_size(payload) > self.settings.limits.max_websocket_message_bytes:
                raise _WebSocketMessageTooLarge
            await upstream.send(payload)

    async def _relay_to_downstream(
        self,
        upstream: UpstreamWebSocket,
        send: ASGISend,
    ) -> bool:
        while True:
            payload = await upstream.recv()
            if payload is None:
                return True
            if _message_size(payload) > self.settings.limits.max_websocket_message_bytes:
                raise _WebSocketMessageTooLarge
            if isinstance(payload, str):
                await send({"type": "websocket.send", "text": payload})
            else:
                await send({"type": "websocket.send", "bytes": payload})


@dataclass(frozen=True, slots=True)
class _Upstream:
    base_url: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _validate_upstream(self.base_url))

    @property
    def parts(self) -> SplitResult:
        return urlsplit(self.base_url)

    @property
    def authority(self) -> str:
        return self.parts.netloc

    @property
    def origin(self) -> str:
        return urlunsplit((self.parts.scheme, self.parts.netloc, "", "", ""))

    def http_target(self, raw_path: str, query: str) -> str:
        suffix = raw_path + (f"?{query}" if query else "")
        return f"{self.base_url}{suffix}"

    def websocket_target(self, raw_path: str, query: str) -> str:
        parts = self.parts
        scheme = "wss" if parts.scheme == "https" else "ws"
        path = f"{parts.path.rstrip('/')}{raw_path}"
        return urlunsplit((scheme, parts.netloc, path, query, ""))


def _validate_upstream(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized or any(character in normalized for character in ("\0", "\r", "\n")):
        raise ProxyConfigurationError("Jupyter proxy upstream must be a single-line URL")
    parts = urlsplit(normalized)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.hostname is None
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ProxyConfigurationError(
            "Jupyter proxy upstream must be a credential-free HTTP(S) URL"
        )
    try:
        _ = parts.port
    except ValueError:
        raise ProxyConfigurationError("Jupyter proxy upstream has an invalid port") from None
    return normalized


def _is_loopback_host(value: str) -> bool:
    normalized = value.strip().strip("[]").casefold()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _request_target(scope: ASGIScope, limits: ProxyLimits) -> tuple[str, str]:
    raw_value = scope.get("raw_path")
    if not isinstance(raw_value, bytes):
        path = scope.get("path")
        if not isinstance(path, str):
            raise ValueError("invalid request path")
        try:
            raw_value = path.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError("invalid request path") from None
    if len(raw_value) > limits.max_path_bytes:
        raise ValueError("path too long")
    try:
        raw_path = raw_value.decode("ascii")
    except UnicodeDecodeError:
        raise ValueError("invalid request path") from None
    if not raw_path.startswith("/") or any(value in raw_path for value in ("\0", "\r", "\n")):
        raise ValueError("invalid request path")
    query_value = scope.get("query_string", b"")
    if not isinstance(query_value, bytes):
        raise ValueError("invalid query string")
    if len(raw_value) + len(query_value) > limits.max_path_bytes:
        raise ValueError("path too long")
    try:
        query = query_value.decode("ascii")
    except UnicodeDecodeError:
        raise ValueError("invalid query string") from None
    if any(value in query for value in ("\0", "\r", "\n")):
        raise ValueError("invalid query string")
    return raw_path, query


def _raw_headers(scope: ASGIScope) -> list[tuple[bytes, bytes]]:
    value = scope.get("headers", [])
    if not isinstance(value, list) or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], bytes)
        or not isinstance(item[1], bytes)
        for item in value
    ):
        raise ValueError("invalid headers")
    return value


def _require_header_limit(scope: ASGIScope, maximum: int) -> None:
    if sum(len(name) + len(value) + 4 for name, value in _raw_headers(scope)) > maximum:
        raise ValueError("headers too large")


def _connection_header_names(headers: Sequence[tuple[bytes, bytes]]) -> set[str]:
    names: set[str] = set()
    for name, value in headers:
        if name.decode("latin-1").casefold() != "connection":
            continue
        names.update(
            part.strip().casefold()
            for part in value.decode("latin-1").split(",")
            if part.strip()
        )
    return names


def _forward_request_headers(
    scope: ASGIScope,
    upstream_authority: str,
) -> list[tuple[str, str]]:
    raw = _raw_headers(scope)
    blocked = (
        _HOP_BY_HOP_HEADERS
        | _FORWARDED_IDENTITY_HEADERS
        | _connection_header_names(raw)
        | {"content-length", "host"}
    )
    forwarded = [
        (name.decode("latin-1"), value.decode("latin-1"))
        for name, value in raw
        if name.decode("latin-1").casefold() not in blocked
    ]
    forwarded.append(("Host", upstream_authority))
    return forwarded


def _forward_websocket_headers(scope: ASGIScope) -> list[tuple[str, str]]:
    raw = _raw_headers(scope)
    blocked = (
        _HOP_BY_HOP_HEADERS
        | _FORWARDED_IDENTITY_HEADERS
        | _connection_header_names(raw)
        | {"content-length", "host", "origin"}
    )
    return [
        (name.decode("latin-1"), value.decode("latin-1"))
        for name, value in raw
        if name.decode("latin-1").casefold() not in blocked
        and not name.decode("latin-1").casefold().startswith("sec-websocket-")
    ]


def _forward_response_headers(
    headers: Sequence[tuple[str, str]],
    *,
    content_length: int,
) -> list[tuple[bytes, bytes]]:
    encoded = [(name.encode("latin-1"), value.encode("latin-1")) for name, value in headers]
    blocked = _HOP_BY_HOP_HEADERS | _connection_header_names(encoded) | {"content-length"}
    forwarded = [
        (name, value)
        for name, value in encoded
        if name.decode("latin-1").casefold() not in blocked
    ]
    forwarded.append((b"content-length", str(content_length).encode("ascii")))
    return forwarded


async def _read_body(receive: ASGIReceive, limits: ProxyLimits) -> bytes:
    body = bytearray()
    try:
        async with asyncio.timeout(limits.request_body_timeout_seconds):
            while True:
                message = await receive()
                message_type = message.get("type")
                if message_type == "http.disconnect":
                    raise _ClientDisconnected
                if message_type != "http.request":
                    raise ValueError("invalid HTTP body message")
                chunk = message.get("body", b"")
                if not isinstance(chunk, bytes):
                    raise ValueError("invalid HTTP body")
                if len(body) + len(chunk) > limits.max_request_body_bytes:
                    raise _RequestBodyTooLarge
                body.extend(chunk)
                if not bool(message.get("more_body", False)):
                    return bytes(body)
    except TimeoutError:
        raise _RequestBodyTimedOut from None


async def _bounded_response_body(response: httpx.Response, *, maximum: int) -> bytes:
    # Mock/custom transports are allowed to return an already materialized
    # response. Network transports leave it streaming. Supporting both keeps
    # the limit at this trust boundary independently of the transport.
    if response.is_stream_consumed:
        if len(response.content) > maximum:
            raise _UpstreamResponseTooLarge
        return response.content
    body = bytearray()
    async for chunk in response.aiter_raw():
        if len(body) + len(chunk) > maximum:
            raise _UpstreamResponseTooLarge
        body.extend(chunk)
    return bytes(body)


def _subprotocols(scope: ASGIScope) -> tuple[str, ...]:
    values = scope.get("subprotocols", [])
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("invalid WebSocket subprotocols")
    return tuple(values)


def _message_size(message: str | bytes) -> int:
    return len(message.encode("utf-8")) if isinstance(message, str) else len(message)


async def _http_error(send: ASGISend, status: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


async def _websocket_close(send: ASGISend, code: int) -> None:
    with suppress(Exception):
        await send({"type": "websocket.close", "code": code})


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a loopback-only reverse proxy for a fixed Jupyter Server"
    )
    parser.add_argument("--upstream", required=True, help="credential-free HTTP(S) origin")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--bind-port", type=int, default=17_798)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--max-request-body-bytes", type=int, default=1_048_576)
    parser.add_argument("--max-response-body-bytes", type=int, default=2_097_152)
    parser.add_argument("--max-websocket-message-bytes", type=int, default=1_048_576)
    parser.add_argument("--upstream-request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--websocket-max-lifetime-seconds", type=float, default=1_800.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _arguments(argv)
    limits = ProxyLimits(
        max_request_body_bytes=arguments.max_request_body_bytes,
        max_response_body_bytes=arguments.max_response_body_bytes,
        max_websocket_message_bytes=arguments.max_websocket_message_bytes,
        upstream_request_timeout_seconds=arguments.upstream_request_timeout_seconds,
        websocket_max_lifetime_seconds=arguments.websocket_max_lifetime_seconds,
    )
    settings = ProxySettings(
        upstream_url=arguments.upstream,
        bind_host=arguments.bind_host,
        bind_port=arguments.bind_port,
        max_concurrency=arguments.max_concurrency,
        limits=limits,
    )

    import uvicorn

    proxy = JupyterLoopbackProxy(settings)
    uvicorn.run(
        proxy,
        host=settings.bind_host,
        port=settings.bind_port,
        access_log=False,
        proxy_headers=False,
        server_header=False,
        date_header=False,
        ws_max_size=settings.limits.max_websocket_message_bytes,
        ws_max_queue=16,
        ws_per_message_deflate=False,
        limit_concurrency=settings.max_concurrency,
        backlog=settings.max_concurrency,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=10,
        h11_max_incomplete_event_size=settings.limits.max_header_bytes,
        log_level="warning",
    )


__all__ = [
    "JupyterLoopbackProxy",
    "ProxyConfigurationError",
    "ProxyLimits",
    "ProxySettings",
    "UpstreamWebSocket",
    "UpstreamWebSocketConnector",
    "WebsocketsUpstreamConnector",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised by systemd/CLI
    main()
