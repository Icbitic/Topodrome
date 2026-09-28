from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from jupyter_client.session import Session
from jupyter_server.services.kernels.connection.base import (
    deserialize_binary_message,
    deserialize_msg_from_ws_v1,
    serialize_binary_message,
    serialize_msg_to_ws_v1,
)

from topodrome.domain.enums import KernelMode
from topodrome.domain.specs import KernelOperationRequest, KernelOperationResult
from topodrome.jupyter.auth import KernelConnectSigner
from topodrome.jupyter.protocol import (
    JUPYTER_WEBSOCKET_PROTOCOL,
    JupyterChannelSession,
    JupyterMessage,
    JupyterProtocolError,
    code_is_complete,
    decode_websocket_message,
    encode_websocket_message,
    service_operation_from_code,
)

pytestmark = pytest.mark.unit


class _Backend:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.operations: list[KernelOperationRequest] = []
        self.interrupted = False
        self.stopped = False

    async def prepare_kernel_operation(
        self,
        session_id: str,
        access: str,
        operation: KernelOperationRequest,
        *,
        request_id: str,
    ) -> int:
        del session_id, access, operation, request_id
        return 7

    async def execute_kernel(
        self,
        session_id: str,
        access: str,
        operation: KernelOperationRequest,
        *,
        request_id: str | None = None,
    ) -> KernelOperationResult:
        del access, request_id
        self.operations.append(operation)
        return KernelOperationResult(
            operation_id="operation-1",
            session_id=session_id,
            result=self.result,
            execution_count=7,
        )

    async def interrupt_kernel(self, session_id: str, access: str) -> bool:
        del session_id, access
        self.interrupted = True
        return True

    async def stop_kernel(self, session_id: str) -> None:
        del session_id
        self.stopped = True

    async def relay_info(self, session_id: str, access: str) -> dict[str, Any]:
        del session_id, access
        return {}


def _request(
    message_type: str,
    content: dict[str, Any] | None = None,
    *,
    channel: str = "shell",
) -> JupyterMessage:
    return JupyterMessage(
        channel=channel,
        header={
            "msg_id": f"{message_type}-id",
            "msg_type": message_type,
            "username": "test-client",
            "session": "client-session",
            "date": "2026-09-08T00:00:00Z",
            "version": "5.4",
        },
        parent_header={},
        metadata={},
        content=content or {},
    )


def _collect_into(messages: list[JupyterMessage]) -> Any:
    async def collect(message: JupyterMessage) -> None:
        messages.append(message)

    return collect


@pytest.mark.parametrize("subprotocol", [None, JUPYTER_WEBSOCKET_PROTOCOL])
def test_jupyter_codec_round_trips_client_messages_and_binary_buffers(
    subprotocol: str | None,
) -> None:
    request = _request("comm_msg", {"comm_id": "one"})
    request = JupyterMessage(
        request.channel,
        request.header,
        request.parent_header,
        request.metadata,
        request.content,
        (b"\x00\x01tensor",),
    )

    encoded = encode_websocket_message(
        request,
        subprotocol=subprotocol,
        maximum_bytes=4096,
    )
    decoded = decode_websocket_message(
        encoded,
        subprotocol=subprotocol,
        maximum_bytes=4096,
    )

    assert decoded == request
    if subprotocol == JUPYTER_WEBSOCKET_PROTOCOL:
        assert isinstance(encoded, bytes)
        offset_count = int.from_bytes(encoded[:8], "little")
        assert offset_count == 7  # channel + four JSON parts + one buffer + terminator


def test_jupyter_codec_rejects_malformed_and_oversized_frames() -> None:
    with pytest.raises(JupyterProtocolError, match="offset"):
        decode_websocket_message(
            b"\x06\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 48,
            subprotocol=JUPYTER_WEBSOCKET_PROTOCOL,
            maximum_bytes=4096,
        )
    encoded = json.dumps(
        {
            "channel": "shell",
            "header": _request("kernel_info_request").header,
            "parent_header": {},
            "metadata": {},
            "content": {"value": "x" * 5000},
        }
    )
    with pytest.raises(JupyterProtocolError, match="request limit"):
        decode_websocket_message(encoded, subprotocol=None, maximum_bytes=1024)


@pytest.mark.parametrize("subprotocol", [None, JUPYTER_WEBSOCKET_PROTOCOL])
def test_codec_interoperates_with_jupyter_client_and_server_wire_format(
    subprotocol: str | None,
) -> None:
    client = Session(key=b"")
    native = client.msg(
        "comm_msg",
        content={"comm_id": "official-client", "data": {"value": 1}},
    )
    buffer = memoryview(b"\x00\x01official-buffer")

    if subprotocol == JUPYTER_WEBSOCKET_PROTOCOL:
        native_parts = [
            client.pack(native[part])
            for part in ("header", "parent_header", "metadata", "content")
        ]
        official_frame = serialize_msg_to_ws_v1(
            [*native_parts, buffer], "shell"
        )
    else:
        official_frame = serialize_binary_message(
            {**native, "channel": "shell", "buffers": [buffer]}
        )

    decoded = decode_websocket_message(
        official_frame,
        subprotocol=subprotocol,
        maximum_bytes=4096,
    )
    assert decoded.msg_id == native["msg_id"]
    assert decoded.msg_type == "comm_msg"
    assert decoded.header["session"] == native["header"]["session"]
    assert decoded.content == native["content"]
    assert decoded.buffers == (bytes(buffer),)

    encoded = encode_websocket_message(
        decoded,
        subprotocol=subprotocol,
        maximum_bytes=4096,
    )
    if subprotocol == JUPYTER_WEBSOCKET_PROTOCOL:
        channel, official_parts = deserialize_msg_from_ws_v1(encoded)
        assert channel == "shell"
        unpacked = [client.unpack(part) for part in official_parts[:4]]
        assert unpacked[0]["msg_id"] == native["msg_id"]
        assert unpacked[3] == native["content"]
        assert official_parts[4:] == [bytes(buffer)]
    else:
        official = deserialize_binary_message(encoded)
        assert official["channel"] == "shell"
        assert official["msg_id"] == native["msg_id"]
        assert official["content"] == native["content"]
        assert official["buffers"] == [bytes(buffer)]


def test_service_cells_are_literal_only_and_never_evaluate_scheduler_python() -> None:
    assert service_operation_from_code("describe()") == KernelOperationRequest(
        operation="describe"
    )
    assert service_operation_from_code(
        'generate("hello", max_new_tokens=4)'
    ) == KernelOperationRequest(
        operation="generate",
        arguments={"prompt": "hello", "generation": {"max_new_tokens": 4}},
    )
    assert service_operation_from_code(
        '%topodrome tokenize {"text":"hello"}'
    ) == KernelOperationRequest(operation="tokenize", arguments={"text": "hello"})
    with pytest.raises(JupyterProtocolError, match="one structured operation"):
        service_operation_from_code("import os\nos.listdir('/')")
    with pytest.raises(JupyterProtocolError, match="credentials"):
        service_operation_from_code(
            '{"operation":"generate","arguments":{"api_key":"secret"}}'
        )
    assert code_is_complete("for value in range(3):") == ("incomplete", "    ")


@pytest.mark.asyncio
async def test_service_channel_emits_native_busy_input_result_reply_idle_sequence() -> None:
    backend = _Backend({"model_type": "FakeModel"})
    channel = JupyterChannelSession(
        backend=backend,
        kernel_session="kernel-1",
        access="opaque-capability",
        mode=KernelMode.SERVICE,
        maximum_output_bytes=4096,
    )
    request = _request(
        "execute_request",
        {
            "code": "describe()",
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
    )
    messages: list[JupyterMessage] = []

    assert not await channel.handle(request, _collect_into(messages))

    assert [message.msg_type for message in messages] == [
        "status",
        "execute_input",
        "execute_result",
        "execute_reply",
        "status",
    ]
    assert [message.channel for message in messages] == [
        "iopub",
        "iopub",
        "iopub",
        "shell",
        "iopub",
    ]
    assert all(message.parent_header == request.header for message in messages)
    assert messages[-1].content == {"execution_state": "idle"}
    assert backend.operations == [KernelOperationRequest(operation="describe")]


@pytest.mark.asyncio
async def test_silent_execution_suppresses_all_iopub_output_except_status() -> None:
    backend = _Backend({"private": "output"})
    channel = JupyterChannelSession(
        backend=backend,
        kernel_session="kernel-1",
        access="opaque-capability",
        mode=KernelMode.SERVICE,
        maximum_output_bytes=4096,
    )
    request = _request(
        "execute_request",
        {
            "code": "describe()",
            "silent": True,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
        },
    )
    messages: list[JupyterMessage] = []

    await channel.handle(request, _collect_into(messages))

    assert [message.msg_type for message in messages] == [
        "status",
        "execute_reply",
        "status",
    ]
    assert messages[0].content == {"execution_state": "busy"}
    assert messages[-1].content == {"execution_state": "idle"}


@pytest.mark.asyncio
async def test_raw_cell_error_is_a_normal_reply_and_control_bypasses_shell_api() -> None:
    backend = _Backend(
        {
            "status": "error",
            "stdout": "before failure\n",
            "stderr": "",
            "ename": "ValueError",
            "evalue": "bad cell",
            "traceback": [],
        }
    )
    channel = JupyterChannelSession(
        backend=backend,
        kernel_session="kernel-1",
        access="opaque-capability",
        mode=KernelMode.RAW_MODEL,
        maximum_output_bytes=4096,
    )
    messages: list[JupyterMessage] = []
    request = _request("execute_request", {"code": "raise ValueError('bad cell')"})

    await channel.handle(request, _collect_into(messages))

    assert [message.msg_type for message in messages] == [
        "status",
        "execute_input",
        "stream",
        "error",
        "execute_reply",
        "status",
    ]
    assert messages[-2].content["status"] == "error"
    control: list[JupyterMessage] = []
    await channel.handle(
        _request("interrupt_request", channel="control"),
        _collect_into(control),
    )
    assert backend.interrupted
    assert control[0].msg_type == "interrupt_reply"


def test_kernel_capability_generation_rotation_and_expiry() -> None:
    now = datetime(2026, 9, 8, tzinfo=UTC)
    signer = KernelConnectSigner(
        "stable-kernel-signing-key-with-32-bytes",
        public_url="https://scheduler.invalid",
        ttl_seconds=60,
    )
    url, expiry = signer.issue(
        session_id="kernel-1",
        relay_id="relay-1",
        generation=4,
        session_expires_at=now + timedelta(minutes=5),
        now=now,
    )
    token = url.split("token=", 1)[1]

    claims = signer.verify(
        token,
        session_id="kernel-1",
        relay_id="relay-1",
        generation=4,
        now=now,
    )
    assert claims.expires_at == expiry
    with pytest.raises(ValueError, match="scope"):
        signer.verify(
            token,
            session_id="kernel-1",
            relay_id="relay-1",
            generation=5,
            now=now,
        )
    with pytest.raises(ValueError, match="expired"):
        signer.verify(
            token,
            session_id="kernel-1",
            relay_id="relay-1",
            generation=4,
            now=now + timedelta(seconds=61),
        )
