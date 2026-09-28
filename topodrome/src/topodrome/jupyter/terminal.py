"""Bootstrap an outbound site agent through Jupyter's terminal API."""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import re
import secrets
from contextlib import suppress
from dataclasses import dataclass
from ipaddress import ip_address
from types import TracebackType
from typing import Any, Protocol, cast, runtime_checkable
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from pydantic import SecretStr

from topodrome.logging import redact, redact_url


class JupyterTerminalError(RuntimeError):
    """A credential-safe terminal bootstrap failure."""


class JupyterDependencyUnavailable(JupyterTerminalError):
    """An optional Jupyter/WebSocket dependency is not installed."""


class TerminalChannel(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...


@runtime_checkable
class TerminalSocket(TerminalChannel, Protocol):
    async def aclose(self) -> None: ...


@runtime_checkable
class TerminalSocketConnector(Protocol):
    async def connect(
        self,
        url: str,
        headers: dict[str, str],
        *,
        origin: str,
    ) -> TerminalSocket: ...


class _WebsocketAdapter:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def send(self, message: str) -> None:
        await self._connection.send(message)

    async def recv(self) -> str | bytes:
        value = await self._connection.recv()
        if not isinstance(value, (str, bytes)):
            raise JupyterTerminalError("terminal WebSocket returned an invalid message")
        return value

    async def aclose(self) -> None:
        await self._connection.close()


class WebsocketsConnector:
    """Lazy adapter for the optional ``websockets`` package."""

    async def connect(
        self,
        url: str,
        headers: dict[str, str],
        *,
        origin: str,
    ) -> TerminalSocket:
        try:
            module = importlib.import_module("websockets")
        except ImportError:
            raise JupyterDependencyUnavailable(
                "install the 'jupyter' extra to use terminal WebSockets"
            ) from None
        try:
            connection = await module.connect(
                url,
                additional_headers=headers,
                origin=origin,
            )
        except Exception:
            raise JupyterTerminalError("unable to connect to the Jupyter terminal") from None
        return _WebsocketAdapter(connection)


@dataclass(frozen=True, slots=True)
class RemoteTerminal:
    name: str

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError("invalid Jupyter terminal name")


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    terminal: RemoteTerminal
    output: str


class JupyterTerminalClient:
    """Minimal REST/WebSocket client for terminal-only remote sites.

    The bearer token is sent only in request headers. It is never embedded in a
    URL, returned output, exception text, or object representation.
    """

    def __init__(
        self,
        base_url: str,
        token: str | SecretStr,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        connector: TerminalSocketConnector | None = None,
        timeout_seconds: float = 30,
        max_output_chars: int = 65_536,
    ) -> None:
        secret = token if isinstance(token, SecretStr) else SecretStr(token)
        if not secret.get_secret_value():
            raise ValueError("Jupyter token must not be empty")
        if timeout_seconds <= 0 or max_output_chars < 1:
            raise ValueError("timeouts and output limits must be positive")
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("Jupyter base URL must be HTTP or HTTPS")
        if parts.scheme == "http" and not _is_loopback_host(parts.hostname):
            raise ValueError("non-loopback Jupyter base URL must use HTTPS")
        self._base_url = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
        self._token = secret
        self._timeout = timeout_seconds
        self._max_output_chars = max_output_chars
        self._connector = connector or WebsocketsConnector()
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
            headers={"Authorization": f"token {secret.get_secret_value()}"},
        )

    def __repr__(self) -> str:
        return f"JupyterTerminalClient(base_url={redact_url(self._base_url)!r})"

    async def __aenter__(self) -> JupyterTerminalClient:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def create_terminal(self, *, name: str | None = None) -> RemoteTerminal:
        if name is not None and _SAFE_TERMINAL_NAME.fullmatch(name) is None:
            raise ValueError("terminal name must contain only safe ASCII characters")
        response = await self._request(
            "POST",
            "/api/terminals",
            json_body={"name": name} if name is not None else None,
        )
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError
            terminal = RemoteTerminal(name=str(payload["name"]))
            if name is not None and terminal.name != name:
                raise ValueError
            return terminal
        except (KeyError, TypeError, ValueError):
            raise JupyterTerminalError("Jupyter returned an invalid terminal response") from None

    async def delete_terminal(self, terminal: RemoteTerminal) -> None:
        path = f"/api/terminals/{quote(terminal.name, safe='')}"
        # Exact-name deletion is idempotent. A scheduler may crash after the
        # delete was committed but before clearing its local ownership record.
        await self._request("DELETE", path, allowed_statuses=frozenset({404}))

    async def bootstrap(
        self,
        command: str,
        *,
        ready_marker: str,
        timeout_seconds: float | None = None,
        cleanup_terminal: bool = True,
        keep_terminal_on_success: bool = False,
        terminal_name: str | None = None,
    ) -> BootstrapResult:
        if not command.strip() or "\0" in command:
            raise ValueError("bootstrap command must not be empty or contain NUL")
        if not ready_marker:
            raise ValueError("ready_marker must not be empty")
        if ready_marker in command:
            raise ValueError(
                "ready_marker must not occur in the submitted command; "
                "terminal echo is not readiness"
            )
        timeout = self._timeout if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise ValueError("bootstrap timeout must be positive")

        terminal = await self.create_terminal(name=terminal_name)
        socket: TerminalSocket | None = None
        output = ""
        succeeded = False
        try:
            socket = await self._connector.connect(
                self._terminal_websocket_url(terminal),
                {"Authorization": f"token {self._token.get_secret_value()}"},
                origin=self._base_url,
            )
            try:
                async with asyncio.timeout(timeout):
                    await synchronize_terminal(socket)
                    await send_terminal_command(socket, command)
                    marker_window = ""
                    while ready_marker not in marker_window:
                        message = await socket.recv()
                        value = _terminal_text(message)
                        output += value
                        if len(output) > self._max_output_chars:
                            output = output[-self._max_output_chars :]
                        marker_window += value
                        if ready_marker not in marker_window:
                            overlap = len(ready_marker) - 1
                            marker_window = marker_window[-overlap:] if overlap else ""
            except TimeoutError:
                raise JupyterTerminalError("Jupyter terminal bootstrap timed out") from None
            except JupyterTerminalError:
                raise
            except Exception:
                raise JupyterTerminalError("Jupyter terminal bootstrap failed") from None
            safe_output = cast(
                str,
                redact(output, (self._token.get_secret_value(),)),
            )
            succeeded = True
            return BootstrapResult(terminal=terminal, output=safe_output)
        finally:
            if socket is not None:
                with suppress(Exception):
                    await socket.aclose()
            if cleanup_terminal and not (succeeded and keep_terminal_on_success):
                try:
                    await self.delete_terminal(terminal)
                except JupyterTerminalError:
                    pass

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, str] | None = None,
        allowed_statuses: tuple[int, ...] | frozenset[int] = (),
    ) -> httpx.Response:
        try:
            if json_body is None:
                response = await self._http.request(method, self._http_url(path))
            else:
                response = await self._http.request(
                    method,
                    self._http_url(path),
                    json=json_body,
                )
            if response.status_code in allowed_statuses:
                return response
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            raise JupyterTerminalError("Jupyter terminal API request failed") from None

    def _http_url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _terminal_websocket_url(self, terminal: RemoteTerminal) -> str:
        parts = urlsplit(self._base_url)
        scheme = "wss" if parts.scheme == "https" else "ws"
        path = f"{parts.path}/terminals/websocket/{quote(terminal.name, safe='')}"
        return urlunsplit((scheme, parts.netloc, path, "", ""))


def _is_loopback_host(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().strip("[]").casefold()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _terminal_text(message: str | bytes) -> str:
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return ""
    if (
        not isinstance(payload, list)
        or len(payload) < 2
        or payload[0] not in {"stdout", "stderr"}
        or not isinstance(payload[1], str)
    ):
        return ""
    return payload[1]


async def synchronize_terminal(
    socket: TerminalChannel,
    *,
    retry_interval_seconds: float = 0.25,
) -> None:
    """Wait until the terminal shell executes input, then disable PTY echo.

    A newly-created Jupyter terminal can accept a WebSocket before its child
    shell is ready to consume input. Sending the real command immediately can
    therefore lose the only copy of a state-changing command. The harmless
    probe is retried until the shell executes it. Its marker is split in the
    submitted text, so terminal echo cannot be mistaken for execution.

    Echo remains disabled while the encoded command envelope is submitted.
    This also prevents a ready marker hidden in encoded input from satisfying
    the caller before the command actually runs.
    """

    if retry_interval_seconds <= 0:
        raise ValueError("terminal readiness retry interval must be positive")
    nonce = secrets.token_hex(16)
    marker = f"__TOPODROME_TERMINAL_SYNC_{nonce}__"
    split = len(marker) // 2
    probe = (
        "stty -echo && printf '%s%s\\n' "
        f"'{marker[:split]}' '{marker[split:]}'"
    )
    if marker in probe:  # pragma: no cover - construction invariant
        raise AssertionError("terminal readiness marker must not be echoable")
    encoded_probe = json.dumps(["stdin", f"{probe}\n"])
    marker_window = ""
    while marker not in marker_window:
        await socket.send(encoded_probe)
        try:
            async with asyncio.timeout(retry_interval_seconds):
                while marker not in marker_window:
                    value = _terminal_text(await socket.recv())
                    marker_window += value
                    if marker not in marker_window:
                        marker_window = marker_window[-(len(marker) - 1) :]
        except TimeoutError:
            continue

    # A timed-out probe can execute immediately after a later probe. Put one
    # final barrier behind every already-submitted probe and consume through
    # that barrier before the caller sends its real command.
    barrier = f"__TOPODROME_TERMINAL_BARRIER_{nonce}__"
    split = len(barrier) // 2
    command = f"printf '%s%s\\n' '{barrier[:split]}' '{barrier[split:]}'"
    await socket.send(json.dumps(["stdin", f"{command}\n"]))
    marker_window = ""
    while barrier not in marker_window:
        value = _terminal_text(await socket.recv())
        marker_window += value
        if barrier not in marker_window:
            marker_window = marker_window[-(len(barrier) - 1) :]


async def send_terminal_command(
    socket: TerminalChannel,
    command: str,
    *,
    chunk_chars: int = 1_024,
    chunk_delay_seconds: float = 0.01,
) -> None:
    """Send one command without relying on the PTY's canonical line capacity.

    Linux terminal canonical input is commonly limited to 4096 bytes. A single
    WebSocket stdin message longer than that can be truncated before the shell
    parses it. The command is therefore base64 encoded and carried in short
    here-document lines, then evaluated by the already-running terminal shell.
    No partial payload can execute before the unique closing delimiter arrives.
    """

    if not command.strip() or "\0" in command:
        raise ValueError("terminal command must not be empty or contain NUL")
    if chunk_chars < 64 or chunk_chars > 2_048:
        raise ValueError("terminal command chunk size must be between 64 and 2048")
    if chunk_delay_seconds < 0 or chunk_delay_seconds > 1:
        raise ValueError("terminal chunk delay must be between zero and one second")
    payload = base64.b64encode(f"{command.rstrip()}\n".encode()).decode("ascii")
    delimiter = f"__TOPODROME_COMMAND_{secrets.token_hex(16)}__"
    decoder = (
        "python3 -c 'import base64,sys;"
        "sys.stdout.buffer.write(base64.b64decode(sys.stdin.buffer.read()))'"
    )
    await socket.send(json.dumps(["stdin", f'eval "$({decoder} <<\'{delimiter}\'\n']))
    await asyncio.sleep(chunk_delay_seconds)
    for offset in range(0, len(payload), chunk_chars):
        await socket.send(json.dumps(["stdin", f"{payload[offset : offset + chunk_chars]}\n"]))
        await asyncio.sleep(chunk_delay_seconds)
    await socket.send(json.dumps(["stdin", f"{delimiter}\n"]))
    await asyncio.sleep(chunk_delay_seconds)
    await socket.send(
        json.dumps(
            [
                "stdin",
                ')"; _topodrome_status=$?; stty echo; (exit "$_topodrome_status")\n',
            ]
        )
    )


# Jupyter Server's terminal WebSocket route is ``/terminals/websocket/(\w+)``
# on supported 2.x releases.  Its REST endpoint will accept hyphens in an
# explicitly requested name, but the corresponding WebSocket then returns 404.
# Keep generated/requested names to the portable intersection accepted by both
# endpoints.
_SAFE_TERMINAL_NAME = re.compile(r"[A-Za-z0-9_]{1,128}")


__all__ = [
    "BootstrapResult",
    "JupyterDependencyUnavailable",
    "JupyterTerminalClient",
    "JupyterTerminalError",
    "RemoteTerminal",
    "TerminalChannel",
    "TerminalSocket",
    "TerminalSocketConnector",
    "WebsocketsConnector",
    "send_terminal_command",
    "synchronize_terminal",
]
