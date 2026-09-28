from __future__ import annotations

import asyncio
import base64
import json
import re
from collections import deque

import httpx
import pytest

from topodrome.jupyter.terminal import (
    JupyterTerminalClient,
    JupyterTerminalError,
    TerminalSocket,
    send_terminal_command,
    synchronize_terminal,
)

pytestmark = pytest.mark.unit


class FakeSocket:
    def __init__(self, messages: list[str | bytes]) -> None:
        self.messages = deque(messages)
        self.sent: list[str] = []
        self.closed = False
        self.readiness_acknowledged = False

    async def send(self, message: str) -> None:
        self.sent.append(message)
        value = json.loads(message)[1]
        if not self.readiness_acknowledged and value.startswith("stty -echo"):
            quoted = re.findall(r"'([^']*)'", value)
            assert len(quoted) == 3
            self.messages.appendleft(json.dumps(["stdout", quoted[1] + quoted[2] + "\n"]))
            self.readiness_acknowledged = True
        elif value.startswith("printf '%s%s"):
            quoted = re.findall(r"'([^']*)'", value)
            self.messages.appendleft(json.dumps(["stdout", quoted[1] + quoted[2] + "\n"]))

    async def recv(self) -> str | bytes:
        if not self.messages:
            raise RuntimeError("fake terminal has no queued output")
        return self.messages.popleft()

    async def aclose(self) -> None:
        self.closed = True


class FakeConnector:
    def __init__(self, socket: FakeSocket) -> None:
        self.socket = socket
        self.calls: list[tuple[str, dict[str, str], str]] = []

    async def connect(
        self,
        url: str,
        headers: dict[str, str],
        *,
        origin: str,
    ) -> TerminalSocket:
        self.calls.append((url, headers, origin))
        return self.socket


async def test_terminal_bootstrap_uses_headers_and_redacts_output() -> None:
    secret = "jupyter-super-secret"  # noqa: S105 - redaction canary
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(201, json={"name": "term-1"})
        if request.method == "DELETE":
            return httpx.Response(204)
        raise AssertionError(request.method)

    socket = FakeSocket(
        [
            json.dumps(["stdout", f"diagnostic token={secret}\n"]),
            json.dumps(["stdout", "TOPODROME_AGENT_READY\n"]),
        ]
    )
    connector = FakeConnector(socket)
    client = JupyterTerminalClient(
        "https://jupyter.invalid/base?token=must-not-propagate",
        secret,
        transport=httpx.MockTransport(handler),
        connector=connector,
    )
    try:
        result = await client.bootstrap(
            "nohup topodrome-agent",
            ready_marker="TOPODROME_AGENT_READY",
        )
    finally:
        await client.aclose()

    assert secret not in result.output
    assert "<redacted>" in result.output
    assert secret not in repr(client)
    assert "must-not-propagate" not in repr(client)
    assert [request.method for request in requests] == ["POST", "DELETE"]
    assert all(request.headers["Authorization"] == f"token {secret}" for request in requests)
    websocket_url, headers, origin = connector.calls[0]
    assert websocket_url == "wss://jupyter.invalid/base/terminals/websocket/term-1"
    assert "token=" not in websocket_url
    assert headers == {"Authorization": f"token {secret}"}
    assert origin == "https://jupyter.invalid/base"
    readiness = json.loads(socket.sent[0])[1]
    assert readiness.startswith("stty -echo && printf")
    quoted = re.findall(r"'([^']*)'", readiness)
    assert quoted[1] + quoted[2] not in readiness
    assert _decode_enveloped_command(socket.sent[2:]) == b"nohup topodrome-agent\n"
    assert socket.closed


async def test_token_authenticated_http_is_allowed_only_on_loopback() -> None:
    with pytest.raises(ValueError, match=r"non-loopback.*HTTPS"):
        JupyterTerminalClient("http://192.0.2.10:8888", "secret")
    with pytest.raises(ValueError, match=r"non-loopback.*HTTPS"):
        JupyterTerminalClient("http://jupyter.invalid", "secret")

    client = JupyterTerminalClient(
        "http://[::1]:8888/base",
        "secret",
        transport=httpx.MockTransport(lambda _: httpx.Response(204)),
    )
    await client.aclose()


async def test_terminal_errors_never_echo_token_or_response_body() -> None:
    secret = "never-log-this-token"  # noqa: S105 - redaction canary

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"authorization={secret}")

    client = JupyterTerminalClient(
        "https://jupyter.invalid",
        secret,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(JupyterTerminalError) as raised:
            await client.create_terminal()
    finally:
        await client.aclose()

    assert secret not in str(raised.value)
    assert secret not in repr(client)


async def test_named_terminal_creation_is_exact_safe_and_idempotently_deleted() -> None:
    requests: list[httpx.Request] = []
    name = "topodrome_bootstrap_0123456789abcdef_0123456789abcdef01234567"

    assert len(name) <= 128

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            assert json.loads(request.content) == {"name": name}
            return httpx.Response(201, json={"name": name})
        if request.method == "DELETE":
            return httpx.Response(404)
        raise AssertionError(request.method)

    client = JupyterTerminalClient(
        "https://jupyter.invalid",
        "secret",
        transport=httpx.MockTransport(handler),
    )
    try:
        terminal = await client.create_terminal(name=name)
        await client.delete_terminal(terminal)
        with pytest.raises(ValueError, match="safe ASCII"):
            await client.create_terminal(name="../../foreign")
        with pytest.raises(ValueError, match="safe ASCII"):
            await client.create_terminal(name="rest-valid-but-websocket-invalid")
    finally:
        await client.aclose()

    assert [request.method for request in requests] == ["POST", "DELETE"]


async def test_named_terminal_response_must_match_requested_identity() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"name": "unexpected_terminal"})

    client = JupyterTerminalClient(
        "https://jupyter.invalid",
        "secret",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(JupyterTerminalError, match="invalid terminal response"):
            await client.create_terminal(name="owned_terminal")
    finally:
        await client.aclose()


async def test_terminal_bootstrap_rejects_echoable_ready_marker_before_creation() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"name": "term-1"})

    client = JupyterTerminalClient(
        "https://jupyter.invalid",
        "secret",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ValueError, match="terminal echo"):
            await client.bootstrap(
                "echo TOPODROME_AGENT_READY",
                ready_marker="TOPODROME_AGENT_READY",
            )
    finally:
        await client.aclose()

    assert not requests


async def test_successful_bootstrap_can_retain_terminal_but_failure_never_does() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.method)
        if request.method == "POST":
            return httpx.Response(201, json={"name": "term-1"})
        return httpx.Response(204)

    success_socket = FakeSocket([json.dumps(["stdout", "READY\n"])])
    client = JupyterTerminalClient(
        "https://jupyter.invalid",
        "secret",
        transport=httpx.MockTransport(handler),
        connector=FakeConnector(success_socket),
    )
    result = await client.bootstrap(
        "start-agent",
        ready_marker="READY",
        keep_terminal_on_success=True,
    )
    assert requests == ["POST"]
    await client.delete_terminal(result.terminal)
    assert requests == ["POST", "DELETE"]
    await client.aclose()

    requests.clear()
    failed_socket = FakeSocket([json.dumps(["stdout", "not ready"])])
    client = JupyterTerminalClient(
        "https://jupyter.invalid",
        "secret",
        transport=httpx.MockTransport(handler),
        connector=FakeConnector(failed_socket),
    )
    with pytest.raises(JupyterTerminalError):
        await client.bootstrap(
            "start-agent",
            ready_marker="READY",
            keep_terminal_on_success=True,
        )
    assert requests == ["POST", "DELETE"]
    await client.aclose()


async def test_terminal_readiness_probe_retries_without_replaying_real_command() -> None:
    class DelayedReadinessSocket:
        def __init__(self) -> None:
            self.probes = 0
            self.messages: asyncio.Queue[str] = asyncio.Queue()

        async def send(self, message: str) -> None:
            value = json.loads(message)[1]
            if value.startswith("stty -echo"):
                self.probes += 1
            if self.probes == 2 or value.startswith("printf '%s%s"):
                quoted = re.findall(r"'([^']*)'", value)
                await self.messages.put(
                    json.dumps(["stdout", quoted[1] + quoted[2] + "\n"])
                )

        async def recv(self) -> str:
            return await self.messages.get()

    socket = DelayedReadinessSocket()
    await asyncio.wait_for(
        synchronize_terminal(socket, retry_interval_seconds=0.001),
        timeout=1,
    )

    assert socket.probes == 2


async def test_long_terminal_command_is_framed_below_canonical_line_limit() -> None:
    class CollectingSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, message: str) -> None:
            self.sent.append(message)

        async def recv(self) -> str:
            raise AssertionError("command framing does not receive output")

    command = "printf 'one very long line' # " + ("x" * 12_000)
    socket = CollectingSocket()
    await send_terminal_command(socket, command)

    stdin_messages = [json.loads(message)[1] for message in socket.sent]
    assert max(len(message.encode("utf-8")) for message in stdin_messages) < 2_048
    assert _decode_enveloped_command(socket.sent) == f"{command}\n".encode()


@pytest.mark.parametrize("delay", [-0.001, 1.001])
async def test_terminal_command_rejects_unsafe_chunk_delay(delay: float) -> None:
    with pytest.raises(ValueError, match="chunk delay"):
        await send_terminal_command(FakeSocket([]), "true", chunk_delay_seconds=delay)


def _decode_enveloped_command(messages: list[str]) -> bytes:
    values = [json.loads(message)[1] for message in messages]
    delimiter_match = re.search(r"<<'([^']+)'", values[0])
    assert delimiter_match is not None
    delimiter = delimiter_match.group(1)
    delimiter_index = values.index(f"{delimiter}\n")
    encoded = "".join(value.rstrip("\n") for value in values[1:delimiter_index])
    assert values[delimiter_index + 1].startswith(')"; _topodrome_status=')
    return base64.b64decode(encoded, validate=True)
