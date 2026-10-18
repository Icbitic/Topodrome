from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from typer.testing import CliRunner

from topodrome.client import cli as cli_module
from topodrome.client.async_client import AsyncClient
from topodrome.client.client import Client
from topodrome.config import SchedulerSettings
from topodrome.db.models import KernelSessionRow
from topodrome.db.session import Database
from topodrome.domain.enums import KernelMode, KernelState
from topodrome.domain.specs import KernelRequest, utc_now
from topodrome.scheduler.app import create_app
from topodrome.scheduler.service import SchedulerService

pytestmark = pytest.mark.unit

_INFO = {
    "session_id": "kernel-1",
    "state": "running",
    "url": "https://notebooks.example:8888",
    "token": "ide-token",
    "token_url": "https://notebooks.example:8888/?token=ide-token",
    "started_at": "2026-09-08T00:00:00Z",
    "last_activity_at": "2026-09-08T00:00:00Z",
}


class _Controller:
    def __init__(self) -> None:
        self.stop_kernel: bool | None = None

    async def ensure_server(self, session_id: str) -> dict[str, Any]:
        assert session_id == "kernel-1"
        return dict(_INFO)

    async def server_info(self, session_id: str) -> dict[str, Any]:
        assert session_id == "kernel-1"
        return dict(_INFO)

    async def stop_server(
        self,
        session_id: str,
        *,
        stop_kernel: bool = True,
    ) -> dict[str, Any]:
        assert session_id == "kernel-1"
        self.stop_kernel = stop_kernel
        return {**_INFO, "state": "stopped", "token": None, "token_url": None}

    def verify_bridge_credential(self, _session_id: str, _credential: str) -> bool:
        return False


@pytest.mark.asyncio
async def test_authenticated_api_and_async_sdk_return_ide_connection(
    database: Database,
    tmp_path: Path,
) -> None:
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path,
        api_token="api-token",  # noqa: S106 - synthetic test credential
        fence_signing_key="f" * 32,
    )
    controller = _Controller()
    service = SchedulerService(
        database=database,
        settings=settings,
        managed_jupyter_controller=cast(Any, controller),
    )
    now = utc_now()
    request = KernelRequest(model="test/model", resources={"gpu": 1})
    async with database.transaction() as session:
        session.add(
            KernelSessionRow(
                id="kernel-1",
                owner_username="@operator",
                state=KernelState.RUNNING,
                mode=KernelMode.SERVICE,
                request=request.model_dump(mode="json"),
                relay_id="relay-1",
                created_at=now,
                last_activity_at=now,
                expires_at=now,
            )
        )
    app = create_app(
        settings,
        database=database,
        service=service,
        run_dispatcher=False,
    )
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(
        "http://test",
        token="api-token",  # noqa: S106 - synthetic test credential
        transport=transport,
    ) as client:
        assert await client.kernel_notebook_server("kernel-1") == _INFO
        assert await client.kernel_notebook_server_info("kernel-1") == _INFO
        stopped = await client.stop_kernel_notebook_server(
            "kernel-1",
            stop_kernel=False,
        )
        assert stopped["state"] == "stopped"
        assert controller.stop_kernel is False
        raw = await client._http.get("/api/v1/kernels/kernel-1/notebook-server")
        assert raw.headers["cache-control"] == "no-store"
        assert raw.headers["referrer-policy"] == "no-referrer"

    async with httpx.AsyncClient(
        base_url="http://test",
        transport=transport,
    ) as unauthenticated:
        assert (
            await unauthenticated.post(
                "/api/v1/kernels/kernel-1/notebook-server"
            )
        ).status_code == 401


def test_sync_sdk_and_handle_use_managed_server_endpoints() -> None:
    requests: list[tuple[str, str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        payload = (
            {**_INFO, "state": "stopped", "token": None, "token_url": None}
            if request.method == "DELETE"
            else _INFO
        )
        return httpx.Response(200, json=payload)

    with Client(
        "https://scheduler.example",
        transport=httpx.MockTransport(respond),
    ) as client:
        assert client.kernel_notebook_server("kernel-1")["token"] == _INFO["token"]
        assert client.kernel_notebook_server_info("kernel-1")["url"] == _INFO["url"]
        assert (
            client.stop_kernel_notebook_server("kernel-1", stop_kernel=False)["state"]
            == "stopped"
        )

    assert requests == [
        ("POST", "https://scheduler.example/api/v1/kernels/kernel-1/notebook-server"),
        ("GET", "https://scheduler.example/api/v1/kernels/kernel-1/notebook-server"),
        (
            "DELETE",
            "https://scheduler.example/api/v1/kernels/kernel-1/notebook-server"
            "?stop_kernel=false",
        ),
    ]


def test_cli_prints_ide_url_and_separate_token(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Client:
        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def kernel_notebook_server(self, session_id: str) -> dict[str, Any]:
            assert session_id == "kernel-1"
            return dict(_INFO)

    monkeypatch.setattr(cli_module, "Client", _Client)
    result = CliRunner().invoke(cli_module.app, ["notebook-server", "kernel-1"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["url"] == _INFO["url"]
    assert payload["token"] == _INFO["token"]
    assert payload["token_url"] == _INFO["token_url"]


def test_cli_explains_workstation_tunnel_for_loopback_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Client:
        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def kernel_notebook_server(self, session_id: str) -> dict[str, Any]:
            assert session_id == "kernel-1"
            return {
                **_INFO,
                "url": "http://127.0.0.1:8891",
                "token_url": "http://127.0.0.1:8891/?token=ide-token",
            }

    monkeypatch.setattr(cli_module, "Client", _Client)
    result = CliRunner().invoke(cli_module.app, ["notebook-server", "kernel-1"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ide_connection"] == {
        "local_url": "http://127.0.0.1:8891",
        "note": (
            "Start the SSH tunnel on the IDE workstation, then configure the IDE "
            "with local_url and the separately returned token."
        ),
        "ssh_tunnel": "ssh -N -L 8891:127.0.0.1:8891 lab1",
    }
    assert "ide-token" not in payload["ide_connection"]["ssh_tunnel"]
