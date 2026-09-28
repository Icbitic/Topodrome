"""Bounded Jupyter messaging over a scheduler-owned WebSocket.

The codec implements both the classic JSON channels transport used by Kernel
Gateway clients and ``v1.kernel.websocket.jupyter.org`` binary framing used by
modern JupyterLab.  It deliberately carries no ZMQ connection information or
site-local paths: messages terminate at the scheduler and become authenticated
Topodrome actor operations on the existing outbound control plane.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import codeop
import json
import struct
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any, Protocol

from topodrome.domain.enums import KernelMode
from topodrome.domain.specs import KernelOperationRequest, KernelOperationResult

JUPYTER_WEBSOCKET_PROTOCOL = "v1.kernel.websocket.jupyter.org"
JUPYTER_PROTOCOL_VERSION = "5.4"
_CHANNELS = frozenset({"shell", "control", "stdin", "iopub", "hb"})
_SERVICE_OPERATIONS = frozenset({"describe", "generate", "tokenize"})


class JupyterProtocolError(ValueError):
    """A client frame does not satisfy the bounded Jupyter wire contract."""


@dataclass(frozen=True, slots=True)
class JupyterMessage:
    channel: str
    header: dict[str, Any]
    parent_header: dict[str, Any]
    metadata: dict[str, Any]
    content: dict[str, Any]
    buffers: tuple[bytes, ...] = ()

    @property
    def msg_id(self) -> str:
        return str(self.header["msg_id"])

    @property
    def msg_type(self) -> str:
        return str(self.header["msg_type"])


class KernelChannelBackend(Protocol):
    async def prepare_kernel_operation(
        self,
        session_id: str,
        access: str,
        operation: KernelOperationRequest,
        *,
        request_id: str,
    ) -> int: ...

    async def execute_kernel(
        self,
        session_id: str,
        access: str,
        operation: KernelOperationRequest,
        *,
        request_id: str | None = None,
    ) -> KernelOperationResult: ...

    async def interrupt_kernel(self, session_id: str, access: str) -> bool: ...

    async def stop_kernel(self, session_id: str) -> object: ...

    async def relay_info(self, session_id: str, access: str) -> dict[str, Any]: ...


SendMessage = Callable[[JupyterMessage], Awaitable[None]]


class JupyterChannelSession:
    """Translate one native channels socket to bounded Topodrome operations."""

    def __init__(
        self,
        *,
        backend: KernelChannelBackend,
        kernel_session: str,
        access: str,
        mode: KernelMode,
        maximum_output_bytes: int,
        shell_lock: asyncio.Lock | None = None,
    ) -> None:
        self.backend = backend
        self.kernel_session = kernel_session
        self.access = access
        self.mode = mode
        self.maximum_output_bytes = maximum_output_bytes
        # A scheduler process supplies one lock shared by every browser socket
        # attached to the logical kernel.  Keeping the fallback makes the
        # protocol adapter independently useful in unit tests and SDKs.
        self._shell_lock = shell_lock or asyncio.Lock()

    async def handle(self, request: JupyterMessage, send: SendMessage) -> bool:
        """Handle one message; return true when the socket should close."""

        if request.channel == "stdin":
            return False
        if request.channel == "control":
            return await self._handle_control(request, send)
        async with self._shell_lock:
            return await self._handle_shell(request, send)

    async def _handle_control(self, request: JupyterMessage, send: SendMessage) -> bool:
        if request.msg_type == "shutdown_request":
            await send(
                reply(
                    request,
                    "shutdown_reply",
                    {"status": "ok", "restart": bool(request.content.get("restart", False))},
                    kernel_session=self.kernel_session,
                )
            )
            await self.backend.stop_kernel(self.kernel_session)
            return True
        if request.msg_type == "interrupt_request":
            interrupted = await self.backend.interrupt_kernel(
                self.kernel_session, self.access
            )
            await send(
                reply(
                    request,
                    "interrupt_reply",
                    {"status": "ok", "interrupted": interrupted},
                    kernel_session=self.kernel_session,
                )
            )
            return False
        await self._send_unsupported(request, send)
        return False

    async def _handle_shell(self, request: JupyterMessage, send: SendMessage) -> bool:
        message_type = request.msg_type
        if message_type == "kernel_info_request":
            await send(
                reply(
                    request,
                    "kernel_info_reply",
                    kernel_info_content(self.mode),
                    kernel_session=self.kernel_session,
                )
            )
            return False
        if message_type == "execute_request":
            await self._execute(request, send)
            return False
        if message_type == "complete_request":
            code = str(request.content.get("code", ""))
            cursor = int(request.content.get("cursor_pos", len(code)))
            matches = (
                [
                    name
                    for name in ("describe", "generate", "tokenize")
                    if name.startswith(code[:cursor])
                ]
                if self.mode is KernelMode.SERVICE
                else []
            )
            await send(
                reply(
                    request,
                    "complete_reply",
                    {
                        "matches": matches,
                        "cursor_start": 0 if matches else cursor,
                        "cursor_end": cursor,
                        "metadata": {},
                        "status": "ok",
                    },
                    kernel_session=self.kernel_session,
                )
            )
            return False
        if message_type == "is_complete_request":
            state, indent = code_is_complete(str(request.content.get("code", "")))
            await send(
                reply(
                    request,
                    "is_complete_reply",
                    {"status": state, "indent": indent},
                    kernel_session=self.kernel_session,
                )
            )
            return False
        simple_replies: dict[str, tuple[str, dict[str, Any]]] = {
            "history_request": ("history_reply", {"status": "ok", "history": []}),
            "comm_info_request": ("comm_info_reply", {"status": "ok", "comms": {}}),
            "inspect_request": (
                "inspect_reply",
                {"status": "ok", "found": False, "data": {}, "metadata": {}},
            ),
        }
        candidate = simple_replies.get(message_type)
        if candidate is not None:
            response_type, content = candidate
            await send(
                reply(
                    request,
                    response_type,
                    content,
                    kernel_session=self.kernel_session,
                )
            )
            return False
        if message_type == "shutdown_request":
            return await self._handle_control(request, send)
        await self._send_unsupported(request, send)
        return False

    async def _execute(self, request: JupyterMessage, send: SendMessage) -> None:
        silent = bool(request.content.get("silent", False))
        code = request.content.get("code", "")
        await send(self._status(request, "busy"))
        if not isinstance(code, str):
            await self._send_execute_error(
                request,
                send,
                execution_count=0,
                name="TypeError",
                value="execute_request code must be a string",
                silent=silent,
            )
            await send(self._status(request, "idle"))
            return
        try:
            operation = (
                service_operation_from_code(code)
                if self.mode is KernelMode.SERVICE
                else KernelOperationRequest(
                    operation="jupyter_execute", arguments={"code": code}
                )
            )
            if operation is None:
                execution_count = 0
                result = KernelOperationResult(
                    operation_id=f"empty_{request.msg_id}",
                    session_id=self.kernel_session,
                    result=None,
                    execution_count=0,
                )
            else:
                execution_count = await self.backend.prepare_kernel_operation(
                    self.kernel_session,
                    self.access,
                    operation,
                    request_id=request.msg_id,
                )
            if not silent:
                await send(
                    reply(
                        request,
                        "execute_input",
                        {"code": code, "execution_count": execution_count},
                        channel="iopub",
                        kernel_session=self.kernel_session,
                    )
                )
            if operation is not None:
                result = await self.backend.execute_kernel(
                    self.kernel_session,
                    self.access,
                    operation,
                    request_id=request.msg_id,
                )
                execution_count = result.execution_count or execution_count
            raw_result = result.result
            if (
                self.mode is KernelMode.RAW_MODEL
                and isinstance(raw_result, dict)
                and raw_result.get("status") == "error"
            ):
                await self._send_execute_error(
                    request,
                    send,
                    execution_count=execution_count,
                    name=str(raw_result.get("ename", "Error")),
                    value=str(raw_result.get("evalue", "kernel execution failed")),
                    stdout=raw_result.get("stdout"),
                    stderr=raw_result.get("stderr"),
                    silent=silent,
                )
                return
            display_result = raw_result
            if self.mode is KernelMode.RAW_MODEL and isinstance(raw_result, dict):
                if not silent:
                    await self._send_stream(
                        request, send, "stdout", raw_result.get("stdout")
                    )
                    await self._send_stream(
                        request, send, "stderr", raw_result.get("stderr")
                    )
                display_result = raw_result.get("result")
            if not silent and display_result is not None:
                await send(
                    reply(
                        request,
                        "execute_result",
                        {
                            "execution_count": execution_count,
                            "data": display_data(display_result, self.maximum_output_bytes),
                            "metadata": {},
                        },
                        channel="iopub",
                        kernel_session=self.kernel_session,
                    )
                )
            await send(
                reply(
                    request,
                    "execute_reply",
                    {
                        "status": "ok",
                        "execution_count": execution_count,
                        "payload": [],
                        "user_expressions": {},
                    },
                    kernel_session=self.kernel_session,
                )
            )
        except Exception as exc:
            await self._send_execute_error(
                request,
                send,
                execution_count=0,
                name=type(exc).__name__,
                value=str(exc),
                silent=silent,
            )
            return
        finally:
            await send(self._status(request, "idle"))

    async def _send_execute_error(
        self,
        request: JupyterMessage,
        send: SendMessage,
        *,
        execution_count: int,
        name: str,
        value: str,
        stdout: object = None,
        stderr: object = None,
        silent: bool = False,
    ) -> None:
        if not silent:
            await self._send_stream(request, send, "stdout", stdout)
            await self._send_stream(request, send, "stderr", stderr)
        safe_name, _ = bounded_text(name, 200)
        safe_value, _ = bounded_text(value, min(4000, self.maximum_output_bytes // 4))
        error = {"ename": safe_name, "evalue": safe_value, "traceback": []}
        if not silent:
            await send(
                reply(
                    request,
                    "error",
                    error,
                    channel="iopub",
                    kernel_session=self.kernel_session,
                )
            )
        await send(
            reply(
                request,
                "execute_reply",
                {"status": "error", "execution_count": execution_count, **error},
                kernel_session=self.kernel_session,
            )
        )

    async def _send_stream(
        self,
        request: JupyterMessage,
        send: SendMessage,
        name: str,
        value: object,
    ) -> None:
        if value is None or value == "":
            return
        text, _ = bounded_text(value, self.maximum_output_bytes // 2)
        await send(
            reply(
                request,
                "stream",
                {"name": name, "text": text},
                channel="iopub",
                kernel_session=self.kernel_session,
            )
        )

    async def _send_unsupported(
        self,
        request: JupyterMessage,
        send: SendMessage,
    ) -> None:
        response_type = (
            f"{request.msg_type[:-8]}_reply"
            if request.msg_type.endswith("_request")
            else "error"
        )
        await send(
            reply(
                request,
                response_type,
                {
                    "status": "error",
                    "ename": "NotImplementedError",
                    "evalue": f"unsupported Jupyter message: {request.msg_type}",
                    "traceback": [],
                },
                kernel_session=self.kernel_session,
            )
        )

    def _status(self, request: JupyterMessage, state: str) -> JupyterMessage:
        return reply(
            request,
            "status",
            {"execution_state": state},
            channel="iopub",
            kernel_session=self.kernel_session,
        )


def kernel_info_content(mode: KernelMode) -> dict[str, Any]:
    language_name = "python" if mode is KernelMode.RAW_MODEL else "topodrome-service"
    return {
        "status": "ok",
        "protocol_version": JUPYTER_PROTOCOL_VERSION,
        "implementation": "topodrome",
        "implementation_version": "0.1.0",
        "language_info": {
            "name": language_name,
            "version": "1",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "python", "version": 3},
            "pygments_lexer": "python3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
        "banner": (
            "Topodrome raw-model kernel"
            if mode is KernelMode.RAW_MODEL
            else "Topodrome service kernel: describe(), generate(), tokenize()"
        ),
        "help_links": [],
    }


def display_data(value: Any, maximum_bytes: int) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = repr(value)
        portable: Any = None
    else:
        portable = value
    text, truncated = bounded_text(encoded, maximum_bytes)
    data: dict[str, Any] = {"text/plain": text}
    if not truncated and portable is not None:
        data["application/json"] = portable
    return data


def decode_websocket_message(
    value: str | bytes,
    *,
    subprotocol: str | None,
    maximum_bytes: int,
) -> JupyterMessage:
    size = len(value.encode("utf-8")) if isinstance(value, str) else len(value)
    if size > maximum_bytes:
        raise JupyterProtocolError("Jupyter message exceeds the configured request limit")
    try:
        if subprotocol == JUPYTER_WEBSOCKET_PROTOCOL:
            if not isinstance(value, bytes):
                raise JupyterProtocolError("Jupyter v1 messages must use binary frames")
            channel, parts = _decode_v1(value)
            if len(parts) < 4:
                raise JupyterProtocolError("Jupyter v1 message has too few parts")
            header, parent, metadata, content = (
                _json_object(part.decode("utf-8")) for part in parts[:4]
            )
            buffers = tuple(parts[4:])
            return _validate_message(channel, header, parent, metadata, content, buffers)
        if isinstance(value, bytes):
            raw, buffers = _decode_legacy_binary(value)
        else:
            raw = _json_object(value)
            raw_buffers = raw.pop("buffers", ())
            if raw_buffers:
                if not isinstance(raw_buffers, list):
                    raise JupyterProtocolError("Jupyter buffers must be a list")
                buffers = tuple(_decode_json_buffer(item) for item in raw_buffers)
            else:
                buffers = ()
        channel = str(raw.pop("channel", "shell"))
        return _validate_message(
            channel,
            _object(raw.get("header"), "header"),
            _object(raw.get("parent_header", {}), "parent_header"),
            _object(raw.get("metadata", {}), "metadata"),
            _object(raw.get("content", {}), "content"),
            buffers,
        )
    except JupyterProtocolError:
        raise
    except Exception as exc:
        raise JupyterProtocolError("invalid Jupyter WebSocket message") from exc


def encode_websocket_message(
    message: JupyterMessage,
    *,
    subprotocol: str | None,
    maximum_bytes: int,
) -> str | bytes:
    if subprotocol == JUPYTER_WEBSOCKET_PROTOCOL:
        parts = [
            _json_bytes(message.header),
            _json_bytes(message.parent_header),
            _json_bytes(message.metadata),
            _json_bytes(message.content),
            *message.buffers,
        ]
        encoded: str | bytes = _encode_v1(message.channel, parts)
    else:
        raw = message_as_dict(message)
        if message.buffers:
            body = dict(raw)
            body.pop("buffers", None)
            parts = [_json_bytes(body), *message.buffers]
            encoded = _encode_legacy_binary(parts)
        else:
            encoded = json.dumps(raw, separators=(",", ":"), ensure_ascii=False)
    size = len(encoded.encode("utf-8")) if isinstance(encoded, str) else len(encoded)
    if size > maximum_bytes:
        raise JupyterProtocolError("Jupyter reply exceeds the configured output limit")
    return encoded


def message_as_dict(message: JupyterMessage) -> dict[str, Any]:
    return {
        "header": message.header,
        "msg_id": message.msg_id,
        "msg_type": message.msg_type,
        "parent_header": message.parent_header,
        "metadata": message.metadata,
        "content": message.content,
        "channel": message.channel,
        "buffers": [base64.b64encode(item).decode("ascii") for item in message.buffers],
    }


def reply(
    request: JupyterMessage,
    msg_type: str,
    content: dict[str, Any],
    *,
    channel: str | None = None,
    kernel_session: str,
) -> JupyterMessage:
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    header = {
        "msg_id": uuid.uuid4().hex,
        "msg_type": msg_type,
        "username": "topodrome",
        "session": kernel_session,
        "date": now,
        "version": JUPYTER_PROTOCOL_VERSION,
    }
    return JupyterMessage(
        channel=channel or request.channel,
        header=header,
        parent_header=dict(request.header),
        metadata={},
        content=content,
    )


def service_operation_from_code(code: str) -> KernelOperationRequest | None:
    """Parse the service kernel's intentionally small, literal-only cell API.

    Accepted forms are ``describe()``, ``generate(...)``, ``tokenize(...)``, a
    ``%topodrome <operation> <json-object>`` line, or an operation JSON object.
    Python expressions are never evaluated at the scheduler.
    """

    stripped = code.strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        raw = _json_object(stripped)
        operation = str(raw.get("operation", ""))
        arguments = _object(raw.get("arguments", {}), "arguments")
        return _service_request(operation, arguments)
    if stripped.startswith("%topodrome"):
        prefix, _, remainder = stripped.partition(" ")
        if prefix != "%topodrome" or not remainder:
            raise JupyterProtocolError("expected %topodrome <operation> [json arguments]")
        operation, _, encoded_arguments = remainder.strip().partition(" ")
        arguments = _json_object(encoded_arguments) if encoded_arguments else {}
        return _service_request(operation, arguments)

    try:
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError as exc:
        raise JupyterProtocolError("service cells must contain one structured operation") from exc
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise JupyterProtocolError("service cells must call describe, generate, or tokenize")
    operation = call.func.id
    if operation not in _SERVICE_OPERATIONS:
        raise JupyterProtocolError(f"operation {operation!r} is unavailable in service mode")
    try:
        positional = [ast.literal_eval(item) for item in call.args]
        keywords = {
            item.arg: ast.literal_eval(item.value)
            for item in call.keywords
            if item.arg is not None
        }
    except (TypeError, ValueError) as exc:
        raise JupyterProtocolError("service operation arguments must be Python literals") from exc
    if any(item.arg is None for item in call.keywords):
        raise JupyterProtocolError("expanded keyword arguments are not supported")
    if operation == "describe":
        if positional or keywords:
            raise JupyterProtocolError("describe() takes no arguments")
        arguments = {}
    elif operation == "generate":
        if len(positional) > 1:
            raise JupyterProtocolError("generate() accepts one positional prompt")
        prompt = positional[0] if positional else keywords.pop("prompt", None)
        if not isinstance(prompt, str):
            raise JupyterProtocolError("generate() requires a string prompt")
        explicit_generation = keywords.pop("generation", {})
        generation = _object(explicit_generation, "generation options")
        generation.update(keywords)
        arguments = {"prompt": prompt, "generation": generation}
    else:
        if len(positional) > 1:
            raise JupyterProtocolError("tokenize() accepts one positional string")
        text = positional[0] if positional else keywords.pop("text", None)
        if not isinstance(text, str):
            raise JupyterProtocolError("tokenize() requires a string")
        explicit_options = keywords.pop("options", {})
        options = _object(explicit_options, "tokenizer options")
        options.update(keywords)
        arguments = {"text": text, "options": options}
    return _service_request(operation, arguments)


def code_is_complete(code: str) -> tuple[str, str]:
    try:
        compiled = codeop.compile_command(code, symbol="exec")
    except (OverflowError, SyntaxError, ValueError):
        return "invalid", ""
    if compiled is None:
        return "incomplete", "    "
    return "complete", ""


def bounded_text(value: object, maximum_bytes: int) -> tuple[str, bool]:
    text = str(value)
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= maximum_bytes:
        return text, False
    suffix = b"\n... <Topodrome output truncated>"
    prefix = encoded[: max(0, maximum_bytes - len(suffix))]
    while prefix:
        try:
            return (prefix + suffix).decode("utf-8"), True
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return suffix[:maximum_bytes].decode("utf-8", errors="ignore"), True


def browser_client_html() -> str:
    """Return a dependency-free notebook cell surface using native channels."""

    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Topodrome kernel</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:14px system-ui;margin:2rem;max-width:70rem}
textarea{width:100%;height:14rem}button{margin:.6rem .4rem .6rem 0}
pre{white-space:pre-wrap;background:#111;color:#eee;padding:1rem}</style>
</head><body><h1>Topodrome kernel</h1><p id="state">Connecting…</p>
<textarea id="code">describe()</textarea><br><button id="run" disabled>Run cell</button>
<button id="info" disabled>Kernel info</button><pre id="output"></pre>
<script>
const q=new URLSearchParams(location.search);
const access=q.get('token')||q.get('access'),sid=crypto.randomUUID();
const scheme=location.protocol==='https:'?'wss:':'ws:';
const auth=access?`token=${encodeURIComponent(access)}&`:'';
const channelUrl=`${scheme}//${location.host}${location.pathname}/channels`;
const ws=new WebSocket(`${channelUrl}?${auth}session_id=${sid}`);
const state=document.getElementById('state'),out=document.getElementById('output');
const code=document.getElementById('code'),run=document.getElementById('run');
const info=document.getElementById('info');
function send(type,content={}){const id=crypto.randomUUID();
ws.send(JSON.stringify({channel:'shell',header:{msg_id:id,msg_type:type,
username:'browser',session:sid,date:new Date().toISOString(),version:'5.4'},
parent_header:{},metadata:{},content}));}
ws.onopen=()=>{state.textContent='Connected';run.disabled=info.disabled=false;send('kernel_info_request')};
ws.onclose=e=>{state.textContent=`Disconnected (${e.code})`;run.disabled=info.disabled=true};
ws.onmessage=e=>{if(typeof e.data!=='string')return;const m=JSON.parse(e.data),c=m.content||{};
if(m.msg_type==='stream')out.textContent+=c.text||'';
if(m.msg_type==='execute_result'||m.msg_type==='display_data')out.textContent+=`${(c.data||{})['text/plain']||''}\n`;
if(m.msg_type==='error')out.textContent+=`${c.ename}: ${c.evalue}\n`;
if(m.msg_type==='kernel_info_reply')state.textContent=`Connected — ${c.banner||'Topodrome'}`};
run.onclick=()=>{out.textContent='';send('execute_request',{code:code.value,silent:false,store_history:true,user_expressions:{},allow_stdin:false,stop_on_error:true})};
info.onclick=()=>send('kernel_info_request');
</script></body></html>"""


def _service_request(operation: str, arguments: dict[str, Any]) -> KernelOperationRequest:
    if operation not in _SERVICE_OPERATIONS:
        raise JupyterProtocolError(f"operation {operation!r} is unavailable in service mode")
    try:
        return KernelOperationRequest(operation=operation, arguments=arguments)
    except ValueError as exc:
        raise JupyterProtocolError(str(exc)) from exc


def _validate_message(
    channel: str,
    header: dict[str, Any],
    parent_header: dict[str, Any],
    metadata: dict[str, Any],
    content: dict[str, Any],
    buffers: tuple[bytes, ...],
) -> JupyterMessage:
    if channel not in _CHANNELS:
        raise JupyterProtocolError(f"unknown Jupyter channel: {channel}")
    for key in ("msg_id", "msg_type", "session"):
        value = header.get(key)
        if not isinstance(value, str) or not value or len(value) > 200:
            raise JupyterProtocolError(f"Jupyter header {key} is invalid")
    if len(buffers) > 16 or sum(len(item) for item in buffers) > 65_536:
        raise JupyterProtocolError("Jupyter buffers exceed the supported bounded subset")
    return JupyterMessage(channel, header, parent_header, metadata, content, buffers)


def _decode_v1(value: bytes) -> tuple[str, list[bytes]]:
    if len(value) < 24:
        raise JupyterProtocolError("truncated Jupyter v1 frame")
    count = int.from_bytes(value[:8], "little")
    if count < 6 or count > 22 or len(value) < 8 * (count + 1):
        raise JupyterProtocolError("invalid Jupyter v1 offset table")
    offsets = [
        int.from_bytes(value[8 * (index + 1) : 8 * (index + 2)], "little")
        for index in range(count)
    ]
    if offsets != sorted(offsets) or offsets[0] != 8 * (count + 1) or offsets[-1] != len(value):
        raise JupyterProtocolError("invalid Jupyter v1 offsets")
    channel = value[offsets[0] : offsets[1]].decode("utf-8")
    parts = [value[offsets[index] : offsets[index + 1]] for index in range(1, count - 1)]
    return channel, parts


def _encode_v1(channel: str, parts: list[bytes]) -> bytes:
    channel_bytes = channel.encode("utf-8")
    count = len(parts) + 2
    offsets = [8 * (count + 1)]
    offsets.append(offsets[-1] + len(channel_bytes))
    for part in parts:
        offsets.append(offsets[-1] + len(part))
    return b"".join(
        [count.to_bytes(8, "little"), *(item.to_bytes(8, "little") for item in offsets),
         channel_bytes, *parts]
    )


def _decode_legacy_binary(value: bytes) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    if len(value) < 8:
        raise JupyterProtocolError("truncated Jupyter binary frame")
    count = struct.unpack("!I", value[:4])[0]
    if count < 1 or count > 17 or len(value) < 4 * (count + 1):
        raise JupyterProtocolError("invalid Jupyter binary buffer table")
    offsets = list(struct.unpack(f"!{count}I", value[4 : 4 * (count + 1)]))
    offsets.append(len(value))
    if offsets[:-1] != sorted(offsets[:-1]) or offsets[0] != 4 * (count + 1):
        raise JupyterProtocolError("invalid Jupyter binary offsets")
    parts = tuple(value[start:stop] for start, stop in pairwise(offsets))
    return _json_object(parts[0].decode("utf-8")), parts[1:]


def _encode_legacy_binary(parts: list[bytes]) -> bytes:
    count = len(parts)
    offsets = [4 * (count + 1)]
    for part in parts[:-1]:
        offsets.append(offsets[-1] + len(part))
    return struct.pack(f"!{count + 1}I", count, *offsets) + b"".join(parts)


def _json_object(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise JupyterProtocolError("Jupyter JSON is invalid") from exc
    return _object(decoded, "JSON value")


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JupyterProtocolError(f"Jupyter {label} must be an object")
    return dict(value)


def _decode_json_buffer(value: object) -> bytes:
    if not isinstance(value, str):
        raise JupyterProtocolError("Jupyter JSON buffers must be base64 strings")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise JupyterProtocolError("Jupyter JSON buffer is invalid") from exc


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


__all__ = [
    "JUPYTER_PROTOCOL_VERSION",
    "JUPYTER_WEBSOCKET_PROTOCOL",
    "JupyterChannelSession",
    "JupyterMessage",
    "JupyterProtocolError",
    "bounded_text",
    "browser_client_html",
    "code_is_complete",
    "decode_websocket_message",
    "display_data",
    "encode_websocket_message",
    "kernel_info_content",
    "message_as_dict",
    "reply",
    "service_operation_from_code",
]
