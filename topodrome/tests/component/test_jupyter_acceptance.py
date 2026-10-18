from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from sqlalchemy import select

from topodrome.config import SchedulerSettings
from topodrome.db.models import AgentEventAckRow, KernelChannelRequestRow
from topodrome.db.session import Database
from topodrome.domain.enums import KernelMode, KernelState, ModelAvailability, SiteKind
from topodrome.domain.errors import ConflictError
from topodrome.domain.events import AgentEnvelope
from topodrome.domain.specs import (
    GPUInventory,
    KernelOperationRequest,
    KernelRequest,
    ModelCapability,
    SiteHeartbeat,
    utc_now,
)
from topodrome.jupyter.protocol import (
    JUPYTER_WEBSOCKET_PROTOCOL,
    JupyterMessage,
    decode_websocket_message,
    encode_websocket_message,
)
from topodrome.protocol.fencing import FencingAuthority
from topodrome.scheduler.app import create_app
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
from topodrome.scheduler.service import SchedulerService
from topodrome.site_agent.admission import ActorAllocation, AdmissionController, LeaseFenceStore
from topodrome.site_agent.agent import (
    Assignment,
    ExecutionOutcome,
    PreparedRuntime,
    SiteAgent,
)
from topodrome.site_agent.inventory import GPUSnapshot
from topodrome.site_agent.pressure import PressurePolicy, SustainedPressureMonitor
from topodrome.site_agent.workspace import AttemptWorkspace, WorkspaceManager

pytestmark = pytest.mark.component
GIB = 1024**3


class _ASGIWebSocket:
    def __init__(
        self,
        app: Any,
        path: str,
        query: str,
        *,
        subprotocol: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.app = app
        self.path = path
        self.query = query
        self.subprotocol = subprotocol
        self.headers = dict(headers or {})
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> _ASGIWebSocket:
        headers = [(b"host", b"test")]
        headers.extend(
            (key.lower().encode("ascii"), value.encode("ascii"))
            for key, value in self.headers.items()
        )
        subprotocols: list[str] = []
        if self.subprotocol is not None:
            encoded = self.subprotocol.encode("ascii")
            headers.append((b"sec-websocket-protocol", encoded))
            subprotocols.append(self.subprotocol)
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("test", 80),
            "client": ("127.0.0.1", 12345),
            "root_path": "",
            "path": self.path,
            "raw_path": self.path.encode("ascii"),
            "query_string": self.query.encode("ascii"),
            "headers": headers,
            "subprotocols": subprotocols,
            "state": {},
        }
        await self.incoming.put({"type": "websocket.connect"})
        self.task = asyncio.create_task(self.app(scope, self.incoming.get, self.outgoing.put))
        accepted = await asyncio.wait_for(self.outgoing.get(), timeout=2)
        assert accepted["type"] == "websocket.accept", accepted
        assert accepted.get("subprotocol") == self.subprotocol
        return self

    async def __aexit__(self, *_: object) -> None:
        if self.task is None:
            return
        if not self.task.done():
            await self.incoming.put({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(asyncio.gather(self.task, return_exceptions=True), timeout=2)

    async def send(self, message: JupyterMessage) -> None:
        encoded = encode_websocket_message(
            message,
            subprotocol=self.subprotocol,
            maximum_bytes=262_144,
        )
        event = (
            {"type": "websocket.receive", "bytes": encoded}
            if isinstance(encoded, bytes)
            else {"type": "websocket.receive", "text": encoded}
        )
        await self.incoming.put(event)

    async def receive(self) -> JupyterMessage | dict[str, Any]:
        event = await asyncio.wait_for(self.outgoing.get(), timeout=3)
        if event["type"] == "websocket.close":
            return event
        assert event["type"] == "websocket.send", event
        raw = event.get("bytes") if event.get("bytes") is not None else event.get("text")
        assert isinstance(raw, (str, bytes))
        return decode_websocket_message(
            raw,
            subprotocol=self.subprotocol,
            maximum_bytes=262_144,
        )


def _jupyter_request(
    message_id: str,
    message_type: str,
    content: dict[str, Any] | None = None,
    *,
    channel: str = "shell",
) -> JupyterMessage:
    return JupyterMessage(
        channel=channel,
        header={
            "msg_id": message_id,
            "msg_type": message_type,
            "username": "component-test",
            "session": "browser-session",
            "date": utc_now().isoformat(),
            "version": "5.4",
        },
        parent_header={},
        metadata={},
        content=content or {},
    )


async def _jupyter_execution(
    socket: _ASGIWebSocket,
    request: JupyterMessage,
) -> list[JupyterMessage]:
    await socket.send(request)
    messages: list[JupyterMessage] = []
    while True:
        message = await socket.receive()
        assert isinstance(message, JupyterMessage), message
        messages.append(message)
        if message.msg_type == "status" and message.content == {"execution_state": "idle"}:
            return messages


class _Inventory:
    def sample(self) -> tuple[GPUSnapshot, ...]:
        return tuple(
            GPUSnapshot(
                uuid=f"GPU-{index}",
                index=index,
                name="Test GPU",
                total_memory_bytes=32 * GIB,
                free_memory_bytes=31 * GIB,
                used_memory_bytes=GIB,
                utilization_percent=0,
                sampled_at=time.time(),
            )
            for index in range(2)
        )


class _Ray:
    async def ensure_running(self) -> bool:
        return False

    async def is_running(self) -> bool:
        return True

    async def stop(self) -> None:
        return None


class _Backend:
    def __init__(self) -> None:
        self.prepared: list[tuple[str, str, KernelMode]] = []
        self.evicted: list[str] = []
        self.operations: list[tuple[str, str, Mapping[str, Any]]] = []
        self.inflight = 0
        self.maximum_inflight = 0
        self.block_started = asyncio.Event()

    async def prepare(
        self,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> PreparedRuntime:
        del workspace
        mode = KernelMode(str(assignment.kwargs["mode"]))
        actor_id = (
            "actor-service-shared"
            if mode is KernelMode.SERVICE
            else f"actor-private-{assignment.attempt_id}"
        )
        self.prepared.append((assignment.attempt_id, actor_id, mode))
        return PreparedRuntime(actor_id, assignment.model_key, reused=False)

    async def execute(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
    ) -> ExecutionOutcome:
        del prepared, assignment, workspace
        raise AssertionError("interactive assignments must not enter the batch source path")

    async def execute_kernel(
        self,
        prepared: PreparedRuntime,
        assignment: Assignment,
        workspace: AttemptWorkspace,
        *,
        operation_id: str,
        session_id: str,
        mode: KernelMode,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        del assignment, workspace, operation_id
        self.operations.append((session_id, operation, dict(arguments)))
        self.inflight += 1
        self.maximum_inflight = max(self.maximum_inflight, self.inflight)
        try:
            if operation == "jupyter_execute" and arguments.get("code") == "BLOCK":
                self.block_started.set()
                await asyncio.Event().wait()
            await asyncio.sleep(0.02)
            if operation == "jupyter_execute":
                return {
                    "status": "ok",
                    "stdout": "native stdout\n",
                    "stderr": "",
                    "result": 42,
                    "truncated": False,
                }
            return {
                "actor_id": prepared.actor_id,
                "session_id": session_id,
                "mode": mode.value,
                "operation": operation,
                "arguments": dict(arguments),
            }
        finally:
            self.inflight -= 1

    async def request_preemption(self, attempt_id: str) -> None:
        del attempt_id

    async def cancel(self, attempt_id: str) -> None:
        del attempt_id

    async def evict(self, prepared: PreparedRuntime) -> None:
        self.evicted.append(prepared.actor_id)

    def is_warm_allocation(self, allocation: ActorAllocation) -> bool:
        return allocation.actor_id not in self.evicted

    async def evict_warm_replica(
        self,
        allocations: tuple[ActorAllocation, ...],
    ) -> tuple[int, ...]:
        assert allocations
        assert len({allocation.actor_id for allocation in allocations}) == 1
        self.evicted.append(allocations[0].actor_id)
        return ()


class _LoopbackOutbound:
    def __init__(self, service: SchedulerService) -> None:
        self.service = service
        self.sequence = 0
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.live_events: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.sequence += 1
        self.events.append((kind, dict(payload)))
        await self.service.handle_agent_envelope(
            AgentEnvelope(
                message_id=f"site-event-{self.sequence}",
                site_id="lab1",
                agent_session_id="site-session",
                sequence=self.sequence,
                kind=kind,  # type: ignore[arg-type]
                payload=dict(payload),
            )
        )

    async def publish_confirmed(self, kind: str, payload: Mapping[str, Any]) -> None:
        await self.publish(kind, payload)

    async def publish_request(
        self,
        request_key: str,
        kind: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del request_key
        self.sequence += 1
        self.events.append((kind, dict(payload)))
        reply = await self.service.handle_agent_envelope(
            AgentEnvelope(
                message_id=f"site-event-{self.sequence}",
                site_id="lab1",
                agent_session_id="site-session",
                sequence=self.sequence,
                kind=kind,  # type: ignore[arg-type]
                payload=dict(payload),
            )
        )
        return reply.payload

    async def publish_live(self, kind: str, payload: Mapping[str, Any]) -> bool:
        self.live_events.append((kind, dict(payload)))
        await self.service.handle_agent_envelope(
            AgentEnvelope(
                message_id=f"site-live-{len(self.live_events)}",
                site_id="lab1",
                agent_session_id="site-session",
                sequence=0,
                kind=kind,  # type: ignore[arg-type]
                payload=dict(payload),
            )
        )
        return True

    async def disconnect(self) -> None:
        return None


async def _wait_state(
    service: SchedulerService,
    session_id: str,
    state: KernelState,
) -> Any:
    summary = await service.get_kernel(session_id)
    for _ in range(500):
        if summary.state is state:
            return summary
        await asyncio.sleep(0.01)
        summary = await service.get_kernel(session_id)
    raise AssertionError(f"kernel remained {summary.state}; expected {state}")


def _heartbeat(*, sequence: int, warm_service: bool = False) -> SiteHeartbeat:
    models = [
        ModelCapability(
            repo_id="test/model",
            revision="rev",
            backend="transformers",
            backend_version="configured",
            dtype="bfloat16",
            availability=(ModelAvailability.WARM if warm_service else ModelAvailability.CACHED),
            gpu_uuid="GPU-0" if warm_service else None,
        )
    ]
    return SiteHeartbeat(
        site_id="lab1",
        site_kind=SiteKind.LAB,
        agent_session_id="site-session",
        sequence=sequence,
        gpus=tuple(
            GPUInventory(
                uuid=f"GPU-{index}",
                index=index,
                name="Test GPU",
                total_memory_bytes=32 * GIB,
                free_memory_bytes=31 * GIB,
                utilization_percent=0,
            )
            for index in range(2)
        ),
        models=tuple(models),
    )


@pytest.mark.asyncio
async def test_production_kernel_path_modes_auth_idle_and_restart(
    database: Database,
    tmp_path: Path,
    pre_staged_inputs,
) -> None:
    global_root = tmp_path / "global"
    global_root.mkdir()
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        public_url="http://127.0.0.1:8042",
        global_root=global_root,
        fence_signing_key="stable-test-fence-and-relay-key-000000",
        kernel_connect_ttl_seconds=2,
        kernel_operation_timeout_seconds=3,
    )
    connections = AgentConnectionRegistry()
    service = SchedulerService(database=database, settings=settings, connections=connections)
    inventory = _Inventory()
    fences = LeaseFenceStore(tmp_path / "site-leases.sqlite3")
    backend = _Backend()
    outbound = _LoopbackOutbound(service)
    agent = SiteAgent(
        site_id="lab1",
        inventory=inventory,
        admission=AdmissionController(
            inventory,
            fences,
            safety_margin_bytes=0,
            warm_actor_validator=lambda allocation: allocation.actor_id == "actor-service-shared",
        ),
        fences=fences,
        ray=_Ray(),
        workspaces=WorkspaceManager(
            global_root=global_root,
            scratch_root=tmp_path / "scratch",
        ),
        backend=backend,
        outbound=outbound,  # type: ignore[arg-type]
        pressure=SustainedPressureMonitor(PressurePolicy(enabled=False)),
        fence_verifier=FencingAuthority(settings.fence_signing_key.get_secret_value()),
    )
    command_tasks: set[asyncio.Task[None]] = set()
    delivered_commands: list[dict[str, object]] = []

    async def deliver(message: dict[str, object]) -> None:
        delivered_commands.append(message)
        task = asyncio.create_task(agent.handle_command(message))
        command_tasks.add(task)
        task.add_done_callback(command_tasks.discard)

    await connections.register(AgentConnection("lab1", "site-session", deliver))
    await outbound.publish("heartbeat", _heartbeat(sequence=1).model_dump(mode="json"))
    app = create_app(
        settings,
        database=database,
        service=service,
        run_dispatcher=False,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8042",
    ) as client:
        request = KernelRequest(
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
            mode=KernelMode.SERVICE,
            idle_timeout_seconds=60,
        )
        first = await service.create_kernel(request)
        assert first.state is KernelState.REQUESTED
        assert await service.dispatch_once()
        await asyncio.sleep(0.05)
        assert not (
            outbound.events
            and outbound.events[-1][0] == "command_ack"
            and outbound.events[-1][1].get("accepted") is False
        ), json.dumps(outbound.events, default=str, indent=2)
        first = await _wait_state(service, first.session_id, KernelState.RUNNING)

        second = await service.create_kernel(request)
        second = await _wait_state(service, second.session_id, KernelState.RUNNING)
        async with database.session() as session:
            first_row = await service.repository.get_kernel(session, first.session_id)
            second_row = await service.repository.get_kernel(session, second.session_id)
            assert first_row.job_id == second_row.job_id
            assert first_row.attempt_id == second_row.attempt_id
            assert first_row.actor_id == second_row.actor_id == "actor-service-shared"
            assert "connect_url" not in first_row.__table__.columns
            assert first.connect_url is second.connect_url is None
            assert "access" not in json.dumps(first_row.request)

        first_url = await service.connect_kernel_url(first.session_id)
        second_url = await service.connect_kernel_url(second.session_id)
        assert (await client.get(first_url)).status_code == 200
        parsed = urlsplit(first_url)
        access = parse_qs(parsed.query)["token"][0]
        tampered = f"{access[:-1]}{'A' if access[-1] != 'A' else 'B'}"
        assert (await client.get(parsed.path, params={"access": tampered})).status_code == 403
        assert (
            await client.get(
                parsed.path.replace(first.session_id, second.session_id),
                params={"access": access},
            )
        ).status_code == 403
        async with database.session() as session:
            row = await service.repository.get_kernel(session, first.session_id)
            assert row.relay_id is not None and row.expires_at is not None
            expired_url, _ = service.kernel_signer.issue(
                session_id=row.id,
                relay_id=row.relay_id,
                generation=row.access_generation,
                session_expires_at=utc_now() + timedelta(minutes=1),
                now=utc_now() - timedelta(seconds=10),
            )
        assert (await client.get(expired_url)).status_code == 403

        gateway_headers = {"Authorization": f"token {access}"}
        gateway_base = f"{parsed.path}/gateway"
        specs = await client.get(f"{gateway_base}/api/kernelspecs", headers=gateway_headers)
        assert specs.status_code == 200
        assert specs.json()["default"] == "topodrome"
        specific_spec = await client.get(
            f"{gateway_base}/api/kernelspecs/topodrome", headers=gateway_headers
        )
        assert specific_spec.status_code == 200
        assert specific_spec.json()["name"] == "topodrome"
        assert specific_spec.json()["spec"]["argv"] == [
            "topodrome-kernel",
            "-f",
            "{connection_file}",
        ]
        attached = await client.post(
            f"{gateway_base}/api/kernels",
            headers=gateway_headers,
            json={
                "name": "topodrome",
                "env": {
                    "KERNEL_LAUNCH_TIMEOUT": "40",
                    "KERNEL_WORKING_DIR": "/central/notebook/path",
                },
            },
        )
        assert attached.status_code == 201
        gateway_kernel_id = attached.json()["id"]
        assert str(uuid.UUID(gateway_kernel_id)) == gateway_kernel_id
        assert gateway_kernel_id != first.session_id
        assert attached.headers["location"].endswith(gateway_kernel_id)
        assert (
            await client.post(
                f"{gateway_base}/api/kernels",
                headers=gateway_headers,
                json={"name": "topodrome", "env": {"TOKEN": "must-not-cross"}},
            )
        ).status_code == 400
        browser_launch = await client.get(
            first_url,
            headers={"Accept": "text/html"},
            follow_redirects=False,
        )
        assert browser_launch.status_code == 303
        assert "token=" not in browser_launch.headers["location"]
        browser = await client.get(
            browser_launch.headers["location"], headers={"Accept": "text/html"}
        )
        assert browser.status_code == 200
        assert access not in browser.text
        assert browser.headers["cache-control"] == "no-store"
        cookie_header = "; ".join(f"{cookie.name}={cookie.value}" for cookie in client.cookies.jar)
        async with _ASGIWebSocket(
            app,
            parsed.path + "/channels",
            "session_id=browser-cookie-client",
            headers={"Cookie": cookie_header, "Origin": "http://127.0.0.1:8042"},
        ) as cookie_socket:
            cookie_info_request = _jupyter_request("cookie-info-1", "kernel_info_request")
            await cookie_socket.send(cookie_info_request)
            cookie_info = await cookie_socket.receive()
            assert isinstance(cookie_info, JupyterMessage)
            assert cookie_info.msg_type == "kernel_info_reply"

        operations = await asyncio.gather(
            client.post(
                first_url,
                json={"operation": "describe", "arguments": {"caller": 1}},
            ),
            client.post(
                second_url,
                json={"operation": "describe", "arguments": {"caller": 2}},
            ),
        )
        assert all(response.status_code == 200 for response in operations)
        assert {response.json()["result"]["actor_id"] for response in operations} == {
            "actor-service-shared"
        }
        assert backend.maximum_inflight == 1

        executed_before_channel = len(backend.operations)
        async with _ASGIWebSocket(app, parsed.path + "/channels", parsed.query) as socket:
            kernel_info_request = _jupyter_request("info-1", "kernel_info_request")
            await socket.send(kernel_info_request)
            kernel_info = await socket.receive()
            assert isinstance(kernel_info, JupyterMessage)
            assert kernel_info.msg_type == "kernel_info_reply"
            assert kernel_info.parent_header == kernel_info_request.header

            execute_request = _jupyter_request(
                "service-cell-1",
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
            first_messages = await _jupyter_execution(socket, execute_request)
            assert [message.msg_type for message in first_messages] == [
                "status",
                "execute_input",
                "execute_result",
                "execute_reply",
                "status",
            ]
            assert all(
                message.parent_header == execute_request.header for message in first_messages
            )
            first_count = next(
                message.content["execution_count"]
                for message in first_messages
                if message.msg_type == "execute_reply"
            )
            duplicate_messages = await _jupyter_execution(socket, execute_request)
            duplicate_count = next(
                message.content["execution_count"]
                for message in duplicate_messages
                if message.msg_type == "execute_reply"
            )
            assert duplicate_count == first_count
            assert len(backend.operations) == executed_before_channel + 1

            # Shell replies are private to the requester, while IOPub follows
            # the native kernel broadcast contract for every attached client.
            async with _ASGIWebSocket(app, parsed.path + "/channels", parsed.query) as peer_socket:
                shared_request = _jupyter_request(
                    "service-cell-broadcast",
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
                requester_messages = await _jupyter_execution(socket, shared_request)
                peer_messages = [await peer_socket.receive() for _ in range(4)]
                assert all(isinstance(message, JupyterMessage) for message in peer_messages)
                assert [
                    message.msg_type
                    for message in peer_messages
                    if isinstance(message, JupyterMessage)
                ] == ["status", "execute_input", "execute_result", "status"]
                assert all(
                    message.channel == "iopub"
                    for message in peer_messages
                    if isinstance(message, JupyterMessage)
                )
                assert [message.msg_type for message in requester_messages].count(
                    "execute_reply"
                ) == 1

            # Jupyter Server's GatewayClient authenticates its upstream REST
            # and WebSocket requests with the same Authorization header.
            gateway_channel_path = f"{gateway_base}/api/kernels/{gateway_kernel_id}/channels"
            async with _ASGIWebSocket(
                app,
                gateway_channel_path,
                "session_id=jupyter-server-client",
                headers=gateway_headers,
            ) as gateway_socket:
                gateway_info_request = _jupyter_request("gateway-info-1", "kernel_info_request")
                await gateway_socket.send(gateway_info_request)
                gateway_info = await gateway_socket.receive()
                assert isinstance(gateway_info, JupyterMessage)
                assert gateway_info.msg_type == "kernel_info_reply"
                assert gateway_info.parent_header == gateway_info_request.header
            outbound_control = json.dumps(delivered_commands, default=str)
            assert access not in outbound_control
            assert "/central/notebook/path" not in outbound_control
            assert "{connection_file}" not in outbound_control

            # Issuing a new connection generation immediately revokes the old
            # URL and its established channel on its next frame.
            replacement_url = await service.connect_kernel_url(first.session_id)
            await socket.send(_jupyter_request("revoked-info", "kernel_info_request"))
            closed = await socket.receive()
            assert isinstance(closed, dict) and closed["code"] == 4403
        assert (await client.get(first_url)).status_code == 403
        assert (await client.get(replacement_url)).status_code == 200

        await service.stop_kernel(first.session_id)
        await _wait_state(service, first.session_id, KernelState.STOPPED)
        assert second.session_id in agent._kernel_session_hosts
        await service.stop_kernel(second.session_id)
        await _wait_state(service, second.session_id, KernelState.STOPPED)
        async with asyncio.timeout(5):
            await agent.wait_for_idle()
        assert "actor-service-shared" not in backend.evicted

        # Re-advertise the retained service actor. Raw placement must skip it
        # and acquire the other GPU for a new private actor.
        await outbound.publish(
            "heartbeat",
            _heartbeat(sequence=2, warm_service=True).model_dump(mode="json"),
        )
        raw_request = request.model_copy(update={"mode": KernelMode.RAW_MODEL})
        raw = await service.create_kernel(raw_request)
        assert await service.dispatch_once()
        raw = await _wait_state(service, raw.session_id, KernelState.RUNNING)
        async with database.session() as session:
            raw_row = await service.repository.get_kernel(session, raw.session_id)
            assert raw_row.attempt_id is not None
            assert raw_row.actor_id is not None
            raw_actor = raw_row.actor_id
            assert raw_actor.startswith("actor-private-")
            local_claims = fences.active_claims()
            assert set(local_claims) == {"GPU-1"}
            assert local_claims["GPU-1"]

        queued_raw = await service.create_kernel(raw_request)
        # Real queued demand may replace the now-idle service actor on GPU-0.
        # The offer leases that exact target first; the site then batch-fences
        # the complete source actor before creating a second private runtime.
        assert await service.dispatch_once()
        queued_raw = await _wait_state(
            service,
            queued_raw.session_id,
            KernelState.RUNNING,
        )
        assert "actor-service-shared" in backend.evicted
        assert set(fences.active_claims()) == {"GPU-0", "GPU-1"}

        raw_url = await service.connect_kernel_url(raw.session_id)
        raw_result = await client.post(
            raw_url,
            json={
                "operation": "exec",
                "arguments": {"code": "counter = 1", "result": "counter"},
            },
        )
        assert raw_result.status_code == 200
        assert raw_result.json()["result"]["mode"] == KernelMode.RAW_MODEL.value
        async with database.session() as session:
            requests = list(await session.scalars(select(KernelChannelRequestRow)))
            live_acks = list(
                await session.scalars(
                    select(AgentEventAckRow).where(AgentEventAckRow.message_id.like("site-live-%"))
                )
            )
        assert requests and all(row.result_digest is not None for row in requests)
        assert not live_acks
        assert "counter = 1" not in json.dumps(
            [
                {
                    "request_id": row.request_id,
                    "state": row.state,
                    "error_code": row.error_code,
                }
                for row in requests
            ]
        )

        raw_parsed = urlsplit(raw_url)
        async with _ASGIWebSocket(
            app,
            raw_parsed.path + "/channels",
            raw_parsed.query,
            subprotocol=JUPYTER_WEBSOCKET_PROTOCOL,
        ) as raw_socket:
            raw_request_message = _jupyter_request(
                "raw-cell-1",
                "execute_request",
                {
                    "code": "print('native stdout')\n6 * 7",
                    "silent": False,
                    "store_history": True,
                    "user_expressions": {},
                    "allow_stdin": False,
                    "stop_on_error": True,
                },
            )
            raw_messages = await _jupyter_execution(raw_socket, raw_request_message)
            assert [message.msg_type for message in raw_messages] == [
                "status",
                "execute_input",
                "stream",
                "execute_result",
                "execute_reply",
                "status",
            ]
            stream = next(message for message in raw_messages if message.msg_type == "stream")
            display = next(
                message for message in raw_messages if message.msg_type == "execute_result"
            )
            assert stream.content == {"name": "stdout", "text": "native stdout\n"}
            assert display.content["data"]["application/json"] == 42

            await raw_socket.send(
                _jupyter_request("raw-blocking-cell", "execute_request", {"code": "BLOCK"})
            )
            initial = [await raw_socket.receive(), await raw_socket.receive()]
            assert all(isinstance(message, JupyterMessage) for message in initial)
            initial_types = [
                message.msg_type for message in initial if isinstance(message, JupyterMessage)
            ]
            assert initial_types == [
                "status",
                "execute_input",
            ]
            await asyncio.wait_for(backend.block_started.wait(), timeout=2)
            await raw_socket.send(
                _jupyter_request(
                    "interrupt-1",
                    "interrupt_request",
                    channel="control",
                )
            )
            interrupted_messages: list[JupyterMessage] = []
            while True:
                message = await raw_socket.receive()
                assert isinstance(message, JupyterMessage), message
                interrupted_messages.append(message)
                if message.msg_type == "status" and message.content == {"execution_state": "idle"}:
                    break
            assert {message.msg_type for message in interrupted_messages} >= {
                "interrupt_reply",
                "error",
                "execute_reply",
                "status",
            }
            assert raw.session_id in agent._kernel_session_hosts
            assert raw_actor not in backend.evicted

        # A scheduler restart recovers the durable binding and can mint a new
        # credential without any token or URL having been persisted.
        restarted = SchedulerService(
            database=database,
            settings=settings,
            connections=connections,
        )
        recovered = await restarted.get_kernel(raw.session_id)
        assert recovered.state is KernelState.RUNNING
        assert recovered.site_id == "lab1"
        restarted_url = await restarted.connect_kernel_url(raw.session_id)
        restarted_access = parse_qs(urlsplit(restarted_url).query)["token"][0]
        operations_before_replay = len(backend.operations)
        completed_native_operation = KernelOperationRequest(
            operation="jupyter_execute",
            arguments={"code": "print('native stdout')\n6 * 7"},
        )
        with pytest.raises(ConflictError, match="ephemeral output"):
            await restarted.execute_kernel(
                raw.session_id,
                restarted_access,
                completed_native_operation,
                request_id="raw-cell-1",
            )
        with pytest.raises(ConflictError, match="different operation content"):
            await restarted.execute_kernel(
                raw.session_id,
                restarted_access,
                KernelOperationRequest(
                    operation="jupyter_execute",
                    arguments={"code": "print('different cell')"},
                ),
                request_id="raw-cell-1",
            )
        assert len(backend.operations) == operations_before_replay

        async with database.transaction() as session:
            raw_row = await service.repository.get_kernel(session, raw.session_id, lock=True)
            raw_row.expires_at = utc_now() - timedelta(seconds=1)
        assert await service.reap_kernels_once() == 1
        await asyncio.sleep(0.05)
        assert not (
            outbound.events
            and outbound.events[-1][0] == "command_ack"
            and outbound.events[-1][1].get("accepted") is False
        ), json.dumps(outbound.events[-5:], default=str, indent=2)
        assert any(
            kind == "kernel_event"
            and payload.get("session_id") == raw.session_id
            and payload.get("state") == "stopped"
            for kind, payload in outbound.events[-8:]
        ), json.dumps(
            {
                "commands": delivered_commands[-4:],
                "events": outbound.events[-8:],
                "active": list(agent._active),
                "hosts": list(agent._kernel_hosts),
                "session_hosts": dict(agent._kernel_session_hosts),
            },
            default=str,
            indent=2,
        )
        await _wait_state(service, raw.session_id, KernelState.STOPPED)
        assert raw_actor in backend.evicted
        assert set(fences.active_claims()) == {"GPU-0"}

        stopped_queued = await service.stop_kernel(queued_raw.session_id)
        if stopped_queued.state is not KernelState.STOPPED:
            stopped_queued = await _wait_state(
                service,
                queued_raw.session_id,
                KernelState.STOPPED,
            )
        assert stopped_queued.state is KernelState.STOPPED
        async with asyncio.timeout(5):
            await agent.wait_for_idle()
        assert fences.active_claims() == {}

    if command_tasks:
        await asyncio.gather(*command_tasks)
    fences.close()
