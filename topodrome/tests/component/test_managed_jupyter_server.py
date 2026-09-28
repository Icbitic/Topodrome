from __future__ import annotations

import asyncio
import json
import shutil
import socket
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx
import pytest
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import SecretStr

from topodrome.config import ManagedJupyterSettings
from topodrome.domain.enums import KernelMode, KernelState
from topodrome.domain.errors import ConflictError, NotFoundError
from topodrome.domain.specs import KernelSummary, ModelRequirement
from topodrome.scheduler.managed_jupyter import (
    JupyterServerProbe,
    ManagedJupyterController,
)

pytestmark = pytest.mark.component

_NOW = datetime(2026, 9, 8, tzinfo=UTC)


class _KernelService:
    def __init__(self, session_id: str) -> None:
        self.summary = KernelSummary(
            session_id=session_id,
            state=KernelState.RUNNING,
            mode=KernelMode.SERVICE,
            model=ModelRequirement(alias="synthetic-model"),
            site_id="synthetic-site",
            created_at=_NOW,
        )
        self.stop_calls: list[str] = []
        self.refresh_calls: list[str] = []

    async def get_kernel(self, session_id: str) -> KernelSummary:
        if session_id != self.summary.session_id:
            raise NotFoundError(f"kernel {session_id} does not exist")
        return self.summary

    async def stop_kernel(self, session_id: str) -> KernelSummary:
        self.stop_calls.append(session_id)
        self.summary = self.summary.model_copy(update={"state": KernelState.STOPPED})
        return self.summary

    async def refresh_managed_kernel_connection(
        self, session_id: str
    ) -> KernelSummary:
        if session_id != self.summary.session_id:
            raise NotFoundError(f"kernel {session_id} does not exist")
        if self.summary.state not in {KernelState.RUNNING, KernelState.IDLE}:
            raise ConflictError("kernel is not ready")
        self.refresh_calls.append(session_id)
        return self.summary


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._returncode: int | None = None

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self._returncode = 0

    def kill(self) -> None:
        self._returncode = -9

    async def wait(self) -> int:
        assert self._returncode is not None
        return self._returncode


class _RecordingLauncher:
    def __init__(self) -> None:
        self.launches: list[tuple[tuple[str, ...], dict[str, str], Path, _FakeProcess]] = []

    async def launch(
        self,
        argv: Sequence[str],
        *,
        environment: Mapping[str, str],
        cwd: Path,
    ) -> _FakeProcess:
        process = _FakeProcess(50_000 + len(self.launches))
        self.launches.append((tuple(argv), dict(environment), cwd, process))
        return process


class _GatewayStub(ThreadingHTTPServer):
    authorization_headers: list[str]
    requested_paths: list[str]
    deleted_paths: list[str]
    attach_payloads: list[dict[str, Any]]
    kernel_id: str | None

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _GatewayHandler)
        self.authorization_headers = []
        self.requested_paths = []
        self.deleted_paths = []
        self.attach_payloads = []
        self.kernel_id = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class _GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *args: object) -> None:
        del args

    def do_GET(self) -> None:
        server = cast(_GatewayStub, self.server)
        path = self._record_request(server)
        if path.endswith("/api/kernelspecs"):
            self._send_json(
                {
                    "default": "topodrome",
                    "kernelspecs": {
                        "topodrome": {
                            "name": "topodrome",
                            "spec": {
                                "argv": [],
                                "display_name": "Topodrome",
                                "env": {},
                                "interrupt_mode": "signal",
                                "language": "python",
                                "metadata": {},
                            },
                            "resources": {},
                        }
                    },
                }
            )
            return
        if path.endswith("/api/kernels"):
            self._send_json(
                [] if server.kernel_id is None else [self._kernel_model(server.kernel_id)]
            )
            return
        if "/api/kernels/" in path:
            kernel_id = unquote(path.rsplit("/", 1)[-1])
            if server.kernel_id == kernel_id:
                self._send_json(self._kernel_model(kernel_id))
                return
        if path.endswith("/api"):
            self._send_json({"version": "1.0", "protocol_version": "5.4"})
            return
        self._not_found()

    def do_POST(self) -> None:
        server = cast(_GatewayStub, self.server)
        path = self._record_request(server)
        if path.endswith("/api/kernels"):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            assert isinstance(payload, dict)
            server.attach_payloads.append(payload)
            prefix = path.removesuffix("/gateway/api/kernels")
            durable_session_id = unquote(prefix.rsplit("/", 1)[-1])
            server.kernel_id = str(
                uuid5(NAMESPACE_URL, f"topodrome-kernel:{durable_session_id}")
            )
            self._send_json(
                self._kernel_model(server.kernel_id),
                status=201,
                location=f"{path}/{server.kernel_id}",
            )
            return
        self._not_found()

    def do_DELETE(self) -> None:
        server = cast(_GatewayStub, self.server)
        path = self._record_request(server)
        server.deleted_paths.append(path)
        if server.kernel_id is not None and path.endswith(f"/api/kernels/{server.kernel_id}"):
            server.kernel_id = None
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._not_found()

    def _record_request(self, server: _GatewayStub) -> str:
        path = urlsplit(self.path).path
        server.requested_paths.append(path)
        if authorization := self.headers.get("Authorization"):
            server.authorization_headers.append(authorization)
        return path

    def _not_found(self) -> None:
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_json(
        self,
        payload: object,
        *,
        status: int = 200,
        location: str | None = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if location is not None:
            self.send_header("Location", location)
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _kernel_model(kernel_id: str) -> dict[str, object]:
        return {
            "id": kernel_id,
            "name": "topodrome",
            "last_activity": "2026-09-08T00:00:00.000000Z",
            "execution_state": "idle",
            "connections": 0,
        }


class _NativeGatewayStub:
    """WebSocket half of the loopback gateway used by the stock server."""

    def __init__(self) -> None:
        self.port = _available_port()
        self.authorization_headers: list[str] = []
        self.requested_paths: list[str] = []
        self.received_types: list[str] = []
        self._message_sequence = 0
        app = FastAPI()
        app.add_api_websocket_route("/{path:path}", self._channels)
        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self.port,
                lifespan="off",
                access_log=False,
                log_config=None,
                log_level="critical",
            )
        )
        self._thread = threading.Thread(
            target=self._server.run,
            name="topodrome-test-native-gateway",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 5
        while not self._server.started and self._thread.is_alive():
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        if not self._server.started:
            raise RuntimeError("native gateway stub failed to start")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("native gateway stub failed to stop")

    async def _channels(self, websocket: WebSocket, path: str) -> None:
        request_path = f"/{path}"
        self.requested_paths.append(request_path)
        authorization = websocket.headers.get("authorization", "")
        self.authorization_headers.append(authorization)
        if not request_path.endswith("/channels") or not authorization.startswith("token "):
            await websocket.close(code=4403)
            return
        await websocket.accept()
        try:
            while True:
                request = json.loads(await websocket.receive_text())
                message_type = str(request["header"]["msg_type"])
                self.received_types.append(message_type)
                if message_type == "kernel_info_request":
                    await websocket.send_json(
                        self._reply(
                            request,
                            channel="shell",
                            message_type="kernel_info_reply",
                            content={
                                "status": "ok",
                                "protocol_version": "5.4",
                                "implementation": "topodrome-test",
                                "implementation_version": "1.0",
                                "language_info": {
                                    "name": "python",
                                    "version": "3.11",
                                    "mimetype": "text/x-python",
                                    "file_extension": ".py",
                                },
                                "banner": "Topodrome test kernel",
                                "help_links": [],
                            },
                        )
                    )
                elif message_type == "execute_request":
                    for channel, reply_type, content in (
                        ("iopub", "status", {"execution_state": "busy"}),
                        (
                            "iopub",
                            "execute_input",
                            {"code": request["content"]["code"], "execution_count": 1},
                        ),
                        (
                            "iopub",
                            "execute_result",
                            {
                                "data": {"text/plain": "42"},
                                "metadata": {},
                                "execution_count": 1,
                            },
                        ),
                        (
                            "shell",
                            "execute_reply",
                            {
                                "status": "ok",
                                "execution_count": 1,
                                "payload": [],
                                "user_expressions": {},
                            },
                        ),
                        ("iopub", "status", {"execution_state": "idle"}),
                    ):
                        await websocket.send_json(
                            self._reply(
                                request,
                                channel=channel,
                                message_type=reply_type,
                                content=content,
                            )
                        )
        except WebSocketDisconnect:
            pass

    def _reply(
        self,
        request: dict[str, Any],
        *,
        channel: str,
        message_type: str,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        self._message_sequence += 1
        return {
            "header": {
                "msg_id": f"gateway-{self._message_sequence}-{uuid4().hex}",
                "msg_type": message_type,
                "username": "topodrome-test",
                "session": "topodrome-gateway-stub",
                "date": datetime.now(UTC).isoformat(),
                "version": "5.4",
            },
            "parent_header": request["header"],
            "metadata": {},
            "content": content,
            "channel": channel,
            "buffers": [],
        }


@pytest.fixture
def gateway_stub() -> Iterator[_GatewayStub]:
    server = _GatewayStub()
    thread = threading.Thread(
        target=server.serve_forever,
        name="topodrome-test-gateway",
        daemon=True,
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def native_gateway_stub() -> Iterator[_NativeGatewayStub]:
    server = _NativeGatewayStub()
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def _settings(
    tmp_path: Path,
    port: int,
    *,
    executable: str = "topodrome-managed-jupyter",
) -> ManagedJupyterSettings:
    return ManagedJupyterSettings(
        enabled=True,
        executable=executable,
        bind_host="127.0.0.1",
        public_url_template="http://127.0.0.1:{port}",
        notebooks_root=tmp_path / "notebooks",
        runtime_root=tmp_path / "runtime",
        port_start=port,
        port_end=port,
        max_servers=1,
        startup_timeout_seconds=15,
        shutdown_timeout_seconds=5,
        idle_timeout_seconds=60,
    )


@pytest.mark.asyncio
async def test_bridge_credential_is_ephemeral_and_does_not_rotate_after_user_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "security-session"
    ambient_canary = "ambient-provider-canary"
    monkeypatch.setenv("TOPODROME_AUTODL_TOKEN", ambient_canary)
    monkeypatch.setenv("HOME", str(tmp_path / "ambient-home"))
    clock = [_NOW]
    launcher = _RecordingLauncher()
    monkeypatch.setattr(
        "topodrome.scheduler.managed_jupyter._pid_start_ticks",
        lambda _pid: 77,
    )
    kernels = _KernelService(session_id)
    controller = ManagedJupyterController(
        settings=_settings(tmp_path, _available_port()),
        global_root=tmp_path / "global",
        scheduler_public_url="http://scheduler.invalid",
        server_token_key=SecretStr("s" * 32),
        kernels=kernels,
        launcher=launcher,
        now=lambda: clock[0],
    )

    async def ready_probe(_state: Any, *, verify_gateway: bool) -> JupyterServerProbe:
        del verify_gateway
        return JupyterServerProbe(
            ready=True,
            gateway_ready=True,
            last_activity_at=clock[0],
            connections=1,
        )

    monkeypatch.setattr(controller, "_probe", ready_probe)
    try:
        first = await controller.ensure_server(session_id)
        assert len(launcher.launches) == 1
        argv, environment, _cwd, process = launcher.launches[0]
        server_token = cast(str, first["token"])
        bridge_token = environment["JUPYTER_GATEWAY_AUTH_TOKEN"]
        assert controller.verify_bridge_credential(session_id, bridge_token)
        assert server_token not in "\0".join(argv)
        assert bridge_token not in "\0".join(argv)
        assert "--ServerApp.terminals_enabled=False" in argv
        assert "--ServerApp.no_browser_open_file=True" in argv
        assert ambient_canary not in environment.values()
        assert "TOPODROME_AUTODL_TOKEN" not in environment
        assert "HOME" not in environment

        state_files = list((tmp_path / "runtime" / "state").glob("*.json"))
        assert len(state_files) == 1
        state_text = state_files[0].read_text(encoding="utf-8")
        persisted = json.loads(state_text)
        assert persisted["bridge_generation"]
        assert bridge_token not in state_text
        assert server_token not in state_text
        assert not {"token", "credential", "secret"} & set(persisted)

        # User connect credentials normally expire quickly. Advancing the
        # controller far beyond that horizon must not rotate the private bridge
        # credential or replace a healthy IDE server.
        clock[0] += timedelta(hours=2)
        await controller.reconcile_once()
        assert len(launcher.launches) == 1
        assert launcher.launches[0][3] is process
        assert controller._states[session_id].pid == process.pid
        assert controller.verify_bridge_credential(session_id, bridge_token)
        assert kernels.refresh_calls == [session_id]

        stopped = await controller.stop_server(session_id, stop_kernel=False)
        assert stopped["state"] == "stopped"
        assert not controller.verify_bridge_credential(session_id, bridge_token)

        replacement = await controller.ensure_server(session_id)
        replacement_environment = launcher.launches[-1][1]
        replacement_bridge = replacement_environment["JUPYTER_GATEWAY_AUTH_TOKEN"]
        assert replacement["token"] == server_token
        assert replacement_bridge != bridge_token
        assert controller.verify_bridge_credential(session_id, replacement_bridge)
        assert not controller.verify_bridge_credential(session_id, bridge_token)
    finally:
        with suppress(NotFoundError):
            await controller.stop_server(session_id, stop_kernel=False)
        await controller.close()


@pytest.mark.asyncio
async def test_managed_servers_use_owner_roots_and_reap_revoked_children_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summaries = {
        owner: KernelSummary(
            session_id=f"kernel-{owner.lstrip('@')}",
            owner_username=owner,
            state=KernelState.RUNNING,
            mode=KernelMode.SERVICE,
            model=ModelRequirement(alias="synthetic-model"),
            site_id="synthetic-site",
            created_at=_NOW,
        )
        for owner in ("alice", "bob", "@operator")
    }

    class _OwnerKernels:
        def __init__(self) -> None:
            self.authorized = {"alice": True, "bob": True}
            self.revisions = {"alice": "alice-r1", "bob": "bob-r1"}
            self.stop_calls: list[str] = []

        async def get_kernel(self, session_id: str) -> KernelSummary:
            for summary in summaries.values():
                if summary.session_id == session_id:
                    return summary
            raise NotFoundError(f"kernel {session_id} does not exist")

        async def stop_kernel(self, session_id: str) -> KernelSummary:
            self.stop_calls.append(session_id)
            return await self.get_kernel(session_id)

        async def refresh_managed_kernel_connection(
            self,
            session_id: str,
        ) -> KernelSummary:
            return await self.get_kernel(session_id)

        def is_owner_authorized(self, owner_username: str) -> bool:
            return self.authorized.get(owner_username, False)

        def owner_access_revision(self, owner_username: str) -> str | None:
            return self.revisions.get(owner_username)

    kernels = _OwnerKernels()
    launcher = _RecordingLauncher()
    port = _available_port()
    settings = _settings(tmp_path, port).model_copy(
        update={"port_end": port + 10, "max_servers": 3}
    )
    monkeypatch.setattr(
        "topodrome.scheduler.managed_jupyter._pid_start_ticks",
        lambda _pid: 77,
    )
    controller = ManagedJupyterController(
        settings=settings,
        global_root=tmp_path / "global",
        scheduler_public_url="http://scheduler.invalid",
        server_token_key=SecretStr("s" * 32),
        kernels=kernels,
        launcher=launcher,
        owner_authorizer=kernels.is_owner_authorized,
        owner_revision_provider=kernels.owner_access_revision,
        now=lambda: _NOW,
    )

    async def ready_probe(_state: Any, *, verify_gateway: bool) -> JupyterServerProbe:
        del verify_gateway
        return JupyterServerProbe(
            ready=True,
            gateway_ready=True,
            last_activity_at=_NOW,
            connections=0,
        )

    monkeypatch.setattr(controller, "_probe", ready_probe)
    try:
        infos: dict[str, dict[str, Any]] = {}
        for summary in summaries.values():
            infos[summary.owner_username] = await controller.ensure_server(
                summary.session_id
            )

        launches = {
            state.owner_username: launch
            for state, launch in zip(controller._states.values(), launcher.launches, strict=True)
        }
        notebooks = tmp_path / "notebooks"
        assert launches["alice"][2] == notebooks / "users" / "alice"
        assert launches["bob"][2] == notebooks / "users" / "bob"
        assert launches["@operator"][2] == notebooks
        for owner in ("alice", "bob"):
            root = launches[owner][2]
            assert root.stat().st_mode & 0o777 == 0o700
            assert f"--ServerApp.root_dir={root}" in launches[owner][0]
        persisted_owners = {
            json.loads(path.read_text(encoding="utf-8"))["owner_username"]
            for path in (tmp_path / "runtime" / "state").glob("*.json")
        }
        assert persisted_owners == {"alice", "bob", "@operator"}

        alice_process = launches["alice"][3]
        bob_process = launches["bob"][3]
        operator_process = launches["@operator"][3]
        kernels.revisions["alice"] = "alice-r2"
        await controller.reconcile_once()
        assert alice_process.returncode == 0
        assert bob_process.returncode is None
        assert operator_process.returncode is None
        replacement = await controller.ensure_server(summaries["alice"].session_id)
        assert replacement["token"] != infos["alice"]["token"]
        assert await controller.revoke_owner("alice") == 1
        assert launcher.launches[-1][3].returncode == 0

        kernels.authorized["bob"] = False
        await controller.reconcile_once()
        assert bob_process.returncode == 0
        assert operator_process.returncode is None
        assert kernels.stop_calls == []
        assert summaries["bob"].state is KernelState.RUNNING
    finally:
        await controller.close(terminate=True)


@pytest.mark.asyncio
async def test_startup_fails_and_reaps_child_without_exact_process_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "unprovable-process-session"
    launcher = _RecordingLauncher()
    controller = ManagedJupyterController(
        settings=_settings(tmp_path, _available_port()),
        global_root=tmp_path / "global",
        scheduler_public_url="http://127.0.0.1:8042",
        server_token_key=SecretStr("s" * 32),
        kernels=_KernelService(session_id),
        launcher=launcher,
    )
    monkeypatch.setattr(
        "topodrome.scheduler.managed_jupyter._pid_start_ticks",
        lambda _pid: None,
    )

    with pytest.raises(ConflictError, match="identity could not be proven"):
        await controller.ensure_server(session_id)

    assert len(launcher.launches) == 1
    process = launcher.launches[0][3]
    assert process.returncode == 0
    state = controller._states[session_id]
    assert state.process is None
    assert state.pid is None
    assert state.process_start_ticks is None
    assert state.bridge_generation is None
    persisted = json.loads(
        next((tmp_path / "runtime" / "state").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert persisted["pid"] is None
    assert persisted["process_start_ticks"] is None
    assert persisted["bridge_generation"] is None
    assert not any((tmp_path / "runtime" / "servers").iterdir())


@pytest.mark.asyncio
async def test_stock_server_exposes_contents_and_sessions_through_managed_bridge(
    tmp_path: Path,
    gateway_stub: _GatewayStub,
    native_gateway_stub: _NativeGatewayStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = shutil.which("topodrome-managed-jupyter")
    if executable is None:
        pytest.skip("topodrome-managed-jupyter is not installed")
    session_id = "stock-server-session"
    port = _available_port()
    kernels = _KernelService(session_id)
    controller = ManagedJupyterController(
        settings=_settings(tmp_path, port, executable=executable),
        global_root=tmp_path / "global",
        scheduler_public_url=gateway_stub.base_url,
        server_token_key=SecretStr("j" * 32),
        kernels=kernels,
    )
    regular_gateway_urls = controller._gateway_urls

    def split_transport_gateway_urls(current_session_id: str) -> tuple[str, str]:
        gateway_url, _ = regular_gateway_urls(current_session_id)
        gateway_path = (
            f"/api/v1/kernel-relay/{quote(current_session_id, safe='')}/gateway"
        )
        return gateway_url, f"{native_gateway_stub.base_url}{gateway_path}"

    # The production scheduler serves HTTP and WebSocket routes on one origin.
    # Separate loopback stubs keep this component fixture small while exercising
    # the exact URLs and transports consumed by stock Jupyter Server.
    monkeypatch.setattr(controller, "_gateway_urls", split_transport_gateway_urls)
    try:
        info = await controller.ensure_server(session_id)
        token = cast(str, info["token"])
        state = controller._states[session_id]
        assert state.pid is not None
        process_cmdline = await asyncio.to_thread(
            Path(f"/proc/{state.pid}/cmdline").read_bytes
        )
        assert token.encode() not in process_cmdline

        headers = {"Authorization": f"token {token}"}
        async with httpx.AsyncClient(
            base_url=cast(str, info["url"]),
            headers=headers,
            timeout=5,
        ) as client:
            sessions = await client.get("/api/sessions")
            assert sessions.status_code == 200
            assert sessions.json() == []

            # The managed IDE surface deliberately excludes local Lab1 shells.
            # Kernel execution still goes through the scoped Topodrome gateway.
            terminals = await client.get("/api/terminals")
            assert terminals.status_code == 404

            notebook = {
                "type": "notebook",
                "format": "json",
                "content": {
                    "nbformat": 4,
                    "nbformat_minor": 5,
                    "metadata": {},
                    "cells": [],
                },
            }
            created = await client.put("/api/contents/bridge.ipynb", json=notebook)
            assert created.status_code == 201
            assert (tmp_path / "notebooks" / "bridge.ipynb").is_file()

            fetched = await client.get("/api/contents/bridge.ipynb")
            assert fetched.status_code == 200
            assert fetched.json()["content"]["cells"] == []

            attached = await client.post(
                "/api/sessions",
                json={
                    "path": "bridge.ipynb",
                    "type": "notebook",
                    "name": "",
                    "kernel": {"name": "topodrome"},
                },
            )
            assert attached.status_code == 201
            attached_payload = attached.json()
            assert attached_payload["path"] == "bridge.ipynb"
            gateway_kernel_id = attached_payload["kernel"]["id"]
            assert gateway_kernel_id == str(
                uuid5(NAMESPACE_URL, f"topodrome-kernel:{session_id}")
            )
            assert gateway_kernel_id != session_id
            session_id_from_server = attached_payload["id"]
            listed_sessions = await client.get("/api/sessions")
            assert listed_sessions.status_code == 200
            assert [item["id"] for item in listed_sessions.json()] == [
                session_id_from_server
            ]
            assert gateway_stub.attach_payloads
            assert gateway_stub.attach_payloads[-1]["name"] == "topodrome"

            execute_message_id = "ide-execute-1"
            execute_request = {
                "header": {
                    "msg_id": execute_message_id,
                    "msg_type": "execute_request",
                    "username": "ide-test",
                    "session": "ide-test-session",
                    "date": datetime.now(UTC).isoformat(),
                    "version": "5.4",
                },
                "parent_header": {},
                "metadata": {},
                "content": {
                    "code": "6 * 7",
                    "silent": False,
                    "store_history": True,
                    "user_expressions": {},
                    "allow_stdin": False,
                    "stop_on_error": True,
                },
                "channel": "shell",
                "buffers": [],
            }
            websocket_url = urlunsplit(
                (
                    "ws",
                    urlsplit(cast(str, info["url"])).netloc,
                    f"/api/kernels/{gateway_kernel_id}/channels",
                    "session_id=ide-test-session",
                    "",
                )
            )
            native_replies: list[dict[str, Any]] = []
            async with websockets.connect(
                websocket_url,
                origin=cast(str, info["url"]),
                additional_headers=headers,
                proxy=None,
                open_timeout=5,
                close_timeout=2,
            ) as channel:
                await channel.send(json.dumps(execute_request, separators=(",", ":")))
                async with asyncio.timeout(5):
                    while True:
                        raw_reply = await channel.recv()
                        if isinstance(raw_reply, bytes):
                            raw_reply = raw_reply.decode()
                        reply = json.loads(raw_reply)
                        if reply.get("parent_header", {}).get("msg_id") != execute_message_id:
                            continue
                        native_replies.append(reply)
                        if (
                            reply["header"]["msg_type"] == "status"
                            and reply["content"] == {"execution_state": "idle"}
                        ):
                            break
            reply_types = {reply["header"]["msg_type"] for reply in native_replies}
            assert reply_types >= {
                "status",
                "execute_input",
                "execute_result",
                "execute_reply",
            }
            result = next(
                reply
                for reply in native_replies
                if reply["header"]["msg_type"] == "execute_result"
            )
            assert result["content"]["data"]["text/plain"] == "42"

            detached = await client.delete(f"/api/sessions/{session_id_from_server}")
            assert detached.status_code == 204
            assert gateway_stub.deleted_paths == []
            assert gateway_stub.kernel_id == gateway_kernel_id
            assert kernels.stop_calls == []

            remaining_sessions = await client.get("/api/sessions")
            assert remaining_sessions.status_code == 200
            assert remaining_sessions.json() == []
            remaining_kernels = await client.get("/api/kernels")
            assert remaining_kernels.status_code == 200
            assert [item["id"] for item in remaining_kernels.json()] == [
                gateway_kernel_id
            ]

            # An IDE may close and reopen a notebook without asking Topodrome
            # to end the leased kernel.  Stock session attachment to that same
            # kernel must continue to work and must not create a replacement.
            reattached = await client.post(
                "/api/sessions",
                json={
                    "path": "bridge.ipynb",
                    "type": "notebook",
                    "name": "",
                    "kernel": {"id": gateway_kernel_id},
                },
            )
            assert reattached.status_code == 201
            assert reattached.json()["kernel"]["id"] == gateway_kernel_id
            assert len(gateway_stub.attach_payloads) == 1
            second_detach = await client.delete(
                f"/api/sessions/{reattached.json()['id']}"
            )
            assert second_detach.status_code == 204
            assert gateway_stub.deleted_paths == []
            assert gateway_stub.kernel_id == gateway_kernel_id
            assert state.process is not None
            assert state.process.returncode is None
            deleted = await client.delete("/api/contents/bridge.ipynb")
            assert deleted.status_code == 204
            assert not (tmp_path / "notebooks" / "bridge.ipynb").exists()

        bridge_headers = tuple(gateway_stub.authorization_headers)
        assert bridge_headers
        bridge_tokens = {header.partition(" ")[2] for header in bridge_headers}
        assert len(bridge_tokens) == 1
        bridge_token = bridge_tokens.pop()
        assert controller.verify_bridge_credential(session_id, bridge_token)
        assert native_gateway_stub.authorization_headers
        assert set(native_gateway_stub.authorization_headers) == {
            f"token {bridge_token}"
        }
        assert "execute_request" in native_gateway_stub.received_types
        assert any(
            path.endswith(f"/api/kernels/{gateway_kernel_id}/channels")
            for path in native_gateway_stub.requested_paths
        )
        assert bridge_token.encode() not in process_cmdline
        runtime_files = tuple(
            path for path in controller._runtime_dir(session_id).rglob("*") if path.is_file()
        )
        assert runtime_files
        runtime_contents = {path: path.read_bytes() for path in runtime_files}
        ide_token_files = [
            path.relative_to(controller._runtime_dir(session_id))
            for path, body in runtime_contents.items()
            if token.encode() in body
        ]
        bridge_token_files = [
            path.relative_to(controller._runtime_dir(session_id))
            for path, body in runtime_contents.items()
            if bridge_token.encode() in body
        ]
        assert ide_token_files == []
        assert bridge_token_files == []
        server_info_files = tuple(
            path for path in runtime_files if path.name.startswith("jpserver-")
        )
        assert len(server_info_files) == 1
        assert json.loads(runtime_contents[server_info_files[0]])["token"] == ""
        assert not any(
            "cookie" in path.name.casefold() and "secret" in path.name.casefold()
            for path in runtime_files
        )
        state_text = "".join(
            path.read_text(encoding="utf-8")
            for path in (tmp_path / "runtime" / "state").glob("*.json")
        )
        assert bridge_token not in state_text
        assert token not in state_text
        assert any(
            path.endswith("/gateway/api/kernelspecs")
            for path in gateway_stub.requested_paths
        )

        # A graceful scheduler shutdown preserves the detached child. A fresh
        # controller with the same protected key must recover that exact PID,
        # port, stable IDE token, and non-secret bridge generation without
        # launching a replacement process.
        original_pid = state.pid
        original_port = state.port
        original_generation = state.bridge_generation
        await controller.close()
        recovery_launcher = _RecordingLauncher()
        recovered = ManagedJupyterController(
            settings=_settings(tmp_path, port, executable=executable),
            global_root=tmp_path / "global",
            scheduler_public_url=gateway_stub.base_url,
            server_token_key=SecretStr("j" * 32),
            kernels=kernels,
            launcher=recovery_launcher,
        )
        monkeypatch.setattr(recovered, "_gateway_urls", split_transport_gateway_urls)
        recovered_info = await recovered.ensure_server(session_id)
        recovered_state = recovered._states[session_id]
        assert recovery_launcher.launches == []
        assert recovered_state.process is None
        assert recovered_state.pid == original_pid
        assert recovered_state.port == original_port
        assert recovered_state.bridge_generation == original_generation
        assert recovered_info["token"] == token
        assert controller.verify_bridge_credential(session_id, bridge_token) is False
        assert recovered.verify_bridge_credential(session_id, bridge_token)
        controller = recovered
    finally:
        with suppress(NotFoundError):
            await controller.stop_server(session_id, stop_kernel=False)
        await controller.close()
