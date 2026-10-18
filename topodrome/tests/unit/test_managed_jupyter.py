from __future__ import annotations

import asyncio
import json
import os
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from jupyter_server.serverapp import ServerApp
from pydantic import SecretStr, ValidationError

from topodrome.config import ManagedJupyterSettings, SchedulerSettings
from topodrome.db.models import KernelSessionRow
from topodrome.db.session import Database
from topodrome.domain.enums import KernelMode, KernelState
from topodrome.domain.errors import ConflictError
from topodrome.domain.specs import KernelRequest, KernelSummary, ModelRequirement, utc_now
from topodrome.jupyter.managed_server import ManagedJupyterServerApp
from topodrome.jupyter.server_bridge import ManagedGatewaySessionManager
from topodrome.scheduler import managed_jupyter as managed_module
from topodrome.scheduler.managed_jupyter import (
    ManagedJupyterController,
    build_managed_jupyter_controller,
)
from topodrome.scheduler.service import SchedulerService

pytestmark = pytest.mark.unit


def test_reference_scheduler_unit_preserves_managed_children_across_restart() -> None:
    unit = (
        Path(__file__).parents[2] / "configs" / "systemd" / "topodrome-scheduler.service"
    ).read_text(encoding="utf-8")

    assert "\nKillMode=process\n" in unit
    assert "KillMode=control-group" not in unit


def test_managed_server_disables_terminals_and_redacts_discovery_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ServerApp,
        "server_info",
        lambda _self: {"token": "plaintext-ide-token", "port": 18888},
    )
    application = ManagedJupyterServerApp()

    assert application.terminals_enabled is False
    assert application.no_browser_open_file is True
    assert application.session_manager_class is ManagedGatewaySessionManager
    assert application.server_info() == {"token": "", "port": 18888}


class _KernelService:
    def __init__(self) -> None:
        self.state = KernelState.RUNNING
        self.refresh_calls: list[str] = []

    async def get_kernel(self, session_id: str) -> KernelSummary:
        return KernelSummary(
            session_id=session_id,
            state=self.state,
            mode=KernelMode.SERVICE,
            model=ModelRequirement(repo_id="test/model"),
            created_at=datetime(2026, 9, 8, tzinfo=UTC),
        )

    async def stop_kernel(self, session_id: str) -> KernelSummary:
        summary = await self.get_kernel(session_id)
        self.state = KernelState.STOPPED
        return summary.model_copy(update={"state": self.state})

    async def refresh_managed_kernel_connection(
        self, session_id: str
    ) -> KernelSummary:
        if self.state not in {KernelState.RUNNING, KernelState.IDLE}:
            raise ConflictError("kernel is not ready")
        self.refresh_calls.append(session_id)
        return await self.get_kernel(session_id)


def _settings(tmp_path: Path, *, enabled: bool) -> SchedulerSettings:
    return SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        public_url="http://127.0.0.1:8042/base",
        global_root=tmp_path / "global",
        secret_file=tmp_path / "secrets" / "scheduler.env",
        fence_signing_key="f" * 32,
        managed_jupyter={
            "enabled": enabled,
            "notebooks_root": tmp_path / "notebooks",
            "runtime_root": tmp_path / "runtime",
        },
    )


def test_managed_jupyter_is_disabled_without_reading_secret_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=False)

    assert (
        build_managed_jupyter_controller(
            settings,
            cast(Any, _KernelService()),
        )
        is None
    )


def test_enabled_managed_jupyter_loads_only_protected_production_secret(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, enabled=True)
    settings.secret_file.parent.mkdir(mode=0o700)
    settings.secret_file.write_text(
        "TOPODROME_TEST_JUPYTER_TOKEN=test-only-must-not-be-used\n"
        f"TOPODROME_MANAGED_JUPYTER_SERVER_TOKEN={'m' * 32}\n",
        encoding="utf-8",
    )
    settings.secret_file.chmod(0o600)

    controller = build_managed_jupyter_controller(
        settings,
        cast(Any, _KernelService()),
    )

    assert isinstance(controller, ManagedJupyterController)
    assert controller.scheduler_public_url == "http://127.0.0.1:8042/base"


def test_enabled_managed_jupyter_rejects_missing_or_exposed_secret(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, enabled=True)
    settings.secret_file.parent.mkdir(mode=0o700)
    settings.secret_file.write_text(
        "TOPODROME_MANAGED_JUPYTER_SERVER_TOKEN=missing",
        encoding="utf-8",
    )
    settings.secret_file.chmod(0o600)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        build_managed_jupyter_controller(
            settings,
            cast(Any, _KernelService()),
        )

    settings.secret_file.write_text(
        f"TOPODROME_MANAGED_JUPYTER_SERVER_TOKEN={'m' * 32}\n",
        encoding="utf-8",
    )
    settings.secret_file.chmod(0o644)
    with pytest.raises(PermissionError, match="0600"):
        build_managed_jupyter_controller(
            settings,
            cast(Any, _KernelService()),
        )


def test_managed_jupyter_configuration_rejects_overlapping_roots(tmp_path: Path) -> None:
    settings = ManagedJupyterSettings(
        enabled=True,
        notebooks_root=tmp_path / "shared",
        runtime_root=tmp_path / "shared" / "runtime",
    )
    with pytest.raises(ValueError, match="must not overlap"):
        ManagedJupyterController(
            settings=settings,
            global_root=tmp_path,
            scheduler_public_url="https://scheduler.example",
            server_token_key=SecretStr("m" * 32),
            kernels=_KernelService(),
        )

    with pytest.raises(ValidationError, match="credential-free HTTP origin"):
        ManagedJupyterSettings(
            enabled=True,
            public_url_template="https://user:password@example.test:{port}",
        )


def test_scheduler_accepts_managed_bridge_beyond_user_generation_and_ttl(
    tmp_path: Path,
) -> None:
    class _Bridge:
        def verify_bridge_credential(self, session_id: str, credential: str) -> bool:
            return session_id == "kernel-1" and credential == "private-bridge"

    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path,
        fence_signing_key="f" * 32,
        kernel_connect_ttl_seconds=1,
    )
    service = SchedulerService(
        database=Database(settings.database_dsn),
        settings=settings,
        managed_jupyter_controller=cast(Any, _Bridge()),
    )
    kernel = KernelSessionRow(
        id="kernel-1",
        owner_username="@operator",
        state=KernelState.RUNNING,
        mode=KernelMode.SERVICE,
        request={},
        relay_id="relay-1",
        access_generation=99,
    )

    claims = service._verify_kernel_access(
        kernel,
        access="private-bridge",
        now=datetime(2099, 1, 1, tzinfo=UTC),
    )

    assert claims.session_id == "kernel-1"
    assert claims.generation == 99
    assert claims.expires_at.year == 9999


def _stop_controller(tmp_path: Path) -> ManagedJupyterController:
    return ManagedJupyterController(
        settings=ManagedJupyterSettings(
            enabled=True,
            notebooks_root=tmp_path / "notebooks",
            runtime_root=tmp_path / "runtime",
            shutdown_timeout_seconds=0.1,
        ),
        global_root=tmp_path / "global",
        scheduler_public_url="https://scheduler.example",
        server_token_key=SecretStr("m" * 32),
        kernels=_KernelService(),
    )


def test_managed_jupyter_loads_legacy_state_without_owner(tmp_path: Path) -> None:
    first = _stop_controller(tmp_path)
    state = managed_module._ServerState(
        session_id="kernel-legacy-owner",
        port=8888,
    )
    first._save_state(state)
    state_path = next((tmp_path / "runtime" / "state").glob("*.json"))
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload.pop("owner_username")
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    state_path.chmod(0o600)

    recovered = _stop_controller(tmp_path)

    assert recovered._states["kernel-legacy-owner"].owner_username is None


async def test_recovered_process_is_signalled_only_through_verified_pidfd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _stop_controller(tmp_path)
    state = managed_module._ServerState(
        session_id="kernel-1",
        port=8888,
        pid=12345,
        process_start_ticks=77,
        bridge_generation="old-generation",
    )
    monkeypatch.setattr(managed_module, "_pid_start_ticks", lambda _pid: 77)
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(
        managed_module,
        "_open_verified_pidfd",
        lambda pid, ticks: read_fd if (pid, ticks) == (12345, 77) else None,
    )
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        managed_module,
        "_send_pidfd_signal",
        lambda pidfd, signum: sent.append((pidfd, signum)),
    )
    monkeypatch.setattr(managed_module, "_pidfd_has_exited", lambda _pidfd: True)
    monkeypatch.setattr(
        os,
        "kill",
        lambda *_args: pytest.fail("numeric-PID signaling is forbidden"),
    )
    try:
        await controller._stop_process(state, preserve_desired=True)
    finally:
        os.close(write_fd)

    assert sent == [(read_fd, signal.SIGTERM)]
    assert state.pid is None
    assert state.process_start_ticks is None
    assert state.bridge_generation is None


async def test_recovered_process_fails_closed_without_pidfd_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _stop_controller(tmp_path)
    state = managed_module._ServerState(
        session_id="kernel-1",
        port=8888,
        pid=12345,
        process_start_ticks=77,
        bridge_generation="old-generation",
    )
    monkeypatch.setattr(managed_module, "_pid_start_ticks", lambda _pid: 77)
    monkeypatch.setattr(os, "pidfd_open", None)
    monkeypatch.setattr(
        os,
        "kill",
        lambda *_args: pytest.fail("numeric-PID signaling is forbidden"),
    )

    with pytest.raises(RuntimeError, match="requires Linux pidfd support"):
        await controller._stop_process(state, preserve_desired=False)

    assert state.pid == 12345
    assert state.process_start_ticks == 77
    assert state.bridge_generation is None
    assert state.desired
    state_files = list((tmp_path / "runtime" / "state").glob("*.json"))
    assert len(state_files) == 1
    persisted = json.loads(state_files[0].read_text(encoding="utf-8"))
    assert persisted["pid"] == 12345
    assert persisted["bridge_generation"] is None


def test_pidfd_verification_refuses_a_recycled_numeric_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(os, "pidfd_open", lambda _pid, _flags: read_fd)
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda *_args: None)
    monkeypatch.setattr(managed_module, "_pid_start_ticks", lambda _pid: 88)

    try:
        assert managed_module._open_verified_pidfd(12345, 77) is None
        with pytest.raises(OSError):
            os.fstat(read_fd)
    finally:
        os.close(write_fd)


async def test_restart_reaps_a_legacy_pending_stop_by_exact_pidfd_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _stop_controller(tmp_path)
    state = managed_module._ServerState(
        session_id="kernel-legacy-stop",
        port=8888,
        desired=False,
        pid=12345,
        process_start_ticks=77,
        bridge_generation=None,
    )
    first._save_state(state)

    recovered = _stop_controller(tmp_path)
    monkeypatch.setattr(managed_module, "_pid_start_ticks", lambda _pid: 77)
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(
        managed_module,
        "_open_verified_pidfd",
        lambda _pid, _ticks: read_fd,
    )
    sent: list[int] = []
    monkeypatch.setattr(
        managed_module,
        "_send_pidfd_signal",
        lambda _pidfd, signum: sent.append(signum),
    )
    monkeypatch.setattr(managed_module, "_pidfd_has_exited", lambda _pidfd: True)
    try:
        await recovered.reconcile_once()
    finally:
        os.close(write_fd)

    assert sent == [signal.SIGTERM]
    assert not recovered._states
    assert not list((tmp_path / "runtime" / "state").glob("*.json"))


async def test_stopping_kernel_reconciliation_revokes_and_reaps_exact_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Process:
        pid = 12345

        def __init__(self) -> None:
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

    kernels = _KernelService()
    controller = ManagedJupyterController(
        settings=ManagedJupyterSettings(
            enabled=True,
            notebooks_root=tmp_path / "notebooks",
            runtime_root=tmp_path / "runtime",
        ),
        global_root=tmp_path / "global",
        scheduler_public_url="https://scheduler.example",
        server_token_key=SecretStr("m" * 32),
        kernels=kernels,
    )
    process = _Process()
    state = managed_module._ServerState(
        session_id="kernel-stopping",
        port=8888,
        pid=process.pid,
        process_start_ticks=77,
        bridge_generation="stopping-generation",
        process=process,
    )
    controller._states[state.session_id] = state
    controller._save_state(state)
    monkeypatch.setattr(managed_module, "_pid_start_ticks", lambda _pid: 77)
    bridge = controller._bridge_token(state.session_id, "stopping-generation")
    assert controller.verify_bridge_credential(state.session_id, bridge)

    kernels.state = KernelState.STOPPING
    await controller.reconcile_once()

    assert process.returncode == 0
    assert not controller.verify_bridge_credential(state.session_id, bridge)
    assert state.session_id not in controller._states
    assert not list((tmp_path / "runtime" / "state").glob("*.json"))
    await controller.close()


async def test_connected_managed_kernel_refresh_precedes_expiry_reaping(
    database: Database,
    tmp_path: Path,
) -> None:
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path,
        fence_signing_key="f" * 32,
        dispatch_interval_seconds=0.01,
    )
    service = SchedulerService(database=database, settings=settings)
    now = utc_now()
    request = KernelRequest(
        model=ModelRequirement(repo_id="test/model"),
        idle_timeout_seconds=60,
    )
    async with database.transaction() as session:
        session.add(
            KernelSessionRow(
                id="kernel-connected",
                state=KernelState.IDLE,
                mode=KernelMode.SERVICE,
                request=request.model_dump(mode="json"),
                relay_id="relay-connected",
                created_at=now - timedelta(minutes=2),
                last_activity_at=now - timedelta(minutes=2),
                expires_at=now - timedelta(seconds=1),
            )
        )

    class _ConnectedController:
        def __init__(self) -> None:
            self.reconcile_calls = 0
            self.stop_calls: list[str] = []

        async def reconcile_once(self) -> None:
            self.reconcile_calls += 1
            service.stop_background()
            await service.refresh_managed_kernel_connection("kernel-connected")

        async def stop_server(
            self,
            session_id: str,
            *,
            stop_kernel: bool = True,
        ) -> dict[str, Any]:
            del stop_kernel
            self.stop_calls.append(session_id)
            return {}

    controller = _ConnectedController()
    service.managed_jupyter_controller = cast(Any, controller)

    async with asyncio.timeout(2):
        await service.run_background()

    async with database.session() as session:
        row = await session.get(KernelSessionRow, "kernel-connected")
        assert row is not None
        assert row.state is KernelState.IDLE
        assert row.expires_at is not None
        expiry = (
            row.expires_at.replace(tzinfo=UTC)
            if row.expires_at.tzinfo is None
            else row.expires_at.astimezone(UTC)
        )
        assert expiry > now
    assert controller.reconcile_calls == 1
    assert controller.stop_calls == []


async def test_kernel_stop_reaps_managed_server_when_agent_delivery_fails(
    database: Database,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class _FailingConnections:
        async def send(self, *_args: object, **_kwargs: object) -> bool:
            events.append("agent-send")
            raise RuntimeError("synthetic delivery failure")

    class _StopController:
        def __init__(self) -> None:
            self.stop_calls: list[tuple[str, bool]] = []

        async def stop_server(
            self,
            session_id: str,
            *,
            stop_kernel: bool = True,
        ) -> dict[str, Any]:
            events.append("managed-stop")
            self.stop_calls.append((session_id, stop_kernel))
            return {}

    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path,
        fence_signing_key="f" * 32,
    )
    controller = _StopController()
    service = SchedulerService(
        database=database,
        settings=settings,
        connections=cast(Any, _FailingConnections()),
        managed_jupyter_controller=cast(Any, controller),
    )
    now = utc_now()
    request = KernelRequest(model=ModelRequirement(repo_id="test/model"))
    async with database.transaction() as session:
        session.add(
            KernelSessionRow(
                id="kernel-delivery-failure",
                state=KernelState.RUNNING,
                mode=KernelMode.SERVICE,
                request=request.model_dump(mode="json"),
                attempt_id="attempt-synthetic",
                site_id="site-synthetic",
                relay_id="relay-delivery-failure",
                runtime_id="runtime-synthetic",
                actor_id="actor-synthetic",
                created_at=now,
                last_activity_at=now,
                expires_at=now + timedelta(minutes=30),
            )
        )

    with pytest.raises(RuntimeError, match="synthetic delivery failure"):
        await service.stop_kernel("kernel-delivery-failure")

    assert controller.stop_calls == [("kernel-delivery-failure", False)]
    assert events == ["managed-stop", "agent-send"]
    async with database.session() as session:
        row = await session.get(KernelSessionRow, "kernel-delivery-failure")
        assert row is not None
        assert row.state is KernelState.STOPPING


async def test_kernel_stop_delivery_continues_when_managed_cleanup_fails(
    database: Database,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class _Connections:
        async def send(self, *_args: object, **_kwargs: object) -> bool:
            events.append("agent-send")
            return True

    class _FailingStopController:
        async def stop_server(
            self,
            _session_id: str,
            *,
            stop_kernel: bool = True,
        ) -> dict[str, Any]:
            del stop_kernel
            events.append("managed-stop")
            raise RuntimeError("details must not escape through the API")

    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path,
        fence_signing_key="f" * 32,
    )
    service = SchedulerService(
        database=database,
        settings=settings,
        connections=cast(Any, _Connections()),
        managed_jupyter_controller=cast(Any, _FailingStopController()),
    )
    now = utc_now()
    request = KernelRequest(model=ModelRequirement(repo_id="test/model"))
    async with database.transaction() as session:
        session.add(
            KernelSessionRow(
                id="kernel-cleanup-failure",
                state=KernelState.RUNNING,
                mode=KernelMode.SERVICE,
                request=request.model_dump(mode="json"),
                attempt_id="attempt-cleanup-failure",
                site_id="site-cleanup-failure",
                relay_id="relay-cleanup-failure",
                runtime_id="runtime-cleanup-failure",
                actor_id="actor-cleanup-failure",
                created_at=now,
                last_activity_at=now,
                expires_at=now + timedelta(minutes=30),
            )
        )

    result = await service.stop_kernel("kernel-cleanup-failure")

    assert result.state is KernelState.STOPPING
    assert events == ["managed-stop", "agent-send"]
