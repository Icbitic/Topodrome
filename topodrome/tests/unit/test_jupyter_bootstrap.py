from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from asgi_lifespan import LifespanManager
from pydantic import SecretStr, ValidationError
from structlog.testing import capture_logs

from topodrome.config import JupyterBootstrapSettings, SchedulerSettings
from topodrome.db.models import SiteRow
from topodrome.db.session import Database
from topodrome.domain.enums import SiteKind, SiteState
from topodrome.jupyter.terminal import BootstrapResult, RemoteTerminal
from topodrome.scheduler import app as app_module
from topodrome.scheduler import jupyter_bootstrap as bootstrap_module
from topodrome.scheduler.app import create_app
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
from topodrome.scheduler.jupyter_bootstrap import (
    JupyterBootstrapController,
    build_jupyter_bootstrap_controller,
)

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 9, 8, tzinfo=UTC)
        self.monotonic = 0.0

    def now(self) -> datetime:
        return self.wall

    def monotonic_now(self) -> float:
        return self.monotonic

    def advance(self, seconds: float) -> None:
        self.monotonic += seconds
        self.wall += timedelta(seconds=seconds)


class FakeTerminalClient:
    def __init__(self, outcomes: list[BaseException | str] | None = None) -> None:
        self.outcomes = list(outcomes or ["terminal-1"])
        self.calls: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.closed = False

    async def bootstrap(self, command: str, **kwargs: Any) -> BootstrapResult:
        self.calls.append({"command": command, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        terminal = RemoteTerminal(kwargs.get("terminal_name") or outcome)
        return BootstrapResult(terminal=terminal, output="ready")

    async def delete_terminal(self, terminal: RemoteTerminal) -> None:
        self.deleted.append(terminal.name)

    async def aclose(self) -> None:
        self.closed = True


class BlockingTerminalClient(FakeTerminalClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def bootstrap(self, command: str, **kwargs: Any) -> BootstrapResult:
        self.calls.append({"command": command, **kwargs})
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")


def _settings(**overrides: Any) -> JupyterBootstrapSettings:
    return JupyterBootstrapSettings(
        enabled=True,
        site_id="remote-jupyter-1",
        command="/opt/topodrome/bin/start-outbound-agent",
        initial_retry_seconds=5,
        max_retry_seconds=12,
        **overrides,
    )


def _controller(
    database: Database,
    connections: AgentConnectionRegistry,
    client: FakeTerminalClient,
    clock: FakeClock,
    tmp_path: Path,
) -> JupyterBootstrapController:
    return JupyterBootstrapController(
        database=database,
        connections=connections,
        settings=_settings(),
        heartbeat_ttl_seconds=30,
        client=client,
        ownership_path=tmp_path / "bootstrap-ownership.json",
        terminal_origin="https://jupyter.invalid/base",
        now=clock.now,
        monotonic=clock.monotonic_now,
    )


async def _finish_attempt(controller: JupyterBootstrapController) -> None:
    for _ in range(20):
        if not controller.attempt_in_flight:
            return
        await asyncio.sleep(0)
    raise AssertionError("bootstrap attempt did not finish")


async def _register_connection(
    connections: AgentConnectionRegistry,
    *,
    session_id: str,
) -> tuple[str, str]:
    async def send(_: dict[str, object]) -> None:
        return None

    connection = AgentConnection(
        site_id="remote-jupyter-1",
        session_id=session_id,
        send=send,
    )
    await connections.register(connection)
    identity = await connections.identity(connection.site_id)
    assert identity is not None
    return identity


async def _store_heartbeat(
    database: Database,
    *,
    identity: tuple[str, str],
    heartbeat_at: datetime,
) -> None:
    async with database.transaction() as session:
        session.add(
            SiteRow(
                id="remote-jupyter-1",
                display_name="Remote Jupyter",
                kind=SiteKind.JUPYTER,
                state=SiteState.READY,
                agent_session_id=identity[0],
                agent_connection_generation=identity[1],
                heartbeat_at=heartbeat_at,
            )
        )


def test_bootstrap_configuration_is_disabled_and_credential_free_by_default() -> None:
    assert not JupyterBootstrapSettings().enabled

    with pytest.raises(ValidationError, match=r"command.*required"):
        JupyterBootstrapSettings(enabled=True)
    for command in (
        "agent --token=secret-value",
        "agent --password secret-value",
        "agent Authorization:Bearer-secret",
        "agent https://user:secret@example.invalid/start",
        "agent https://example.invalid/start?token=secret",
    ):
        with pytest.raises(ValidationError, match="credential"):
            JupyterBootstrapSettings(enabled=True, command=command)
    with pytest.raises(ValidationError, match="ready marker"):
        JupyterBootstrapSettings(
            enabled=True,
            command="echo TOPODROME_AGENT_READY",
        )


def test_disabled_builder_never_reads_secret_file(tmp_path: Path) -> None:
    settings = SchedulerSettings(
        fence_signing_key="f" * 32,
        secret_file=tmp_path / "does-not-exist.env",
    )
    database = Database("sqlite+aiosqlite://")
    assert build_jupyter_bootstrap_controller(
        settings,
        database,
        AgentConnectionRegistry(),
    ) is None


async def test_builder_uses_only_production_secrets_from_protected_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir(mode=0o700)
    secret_dir.chmod(0o700)
    secret_file = secret_dir / "scheduler.env"
    secret_file.write_text(
        "TOPODROME_JUPYTER_BOOTSTRAP_URL=https://jupyter.invalid/base\n"
        "TOPODROME_JUPYTER_BOOTSTRAP_TOKEN=production-secret\n"
        "TOPODROME_TEST_JUPYTER_URL=https://test.invalid/?token=test-secret\n"
        "TOPODROME_TEST_JUPYTER_TOKEN=test-secret\n",
        encoding="utf-8",
    )
    secret_file.chmod(0o600)
    captured: dict[str, str] = {}
    client = FakeTerminalClient()

    def client_factory(url: str, token: SecretStr) -> FakeTerminalClient:
        captured["url"] = url
        captured["token"] = token.get_secret_value()
        return client

    monkeypatch.setattr(bootstrap_module, "JupyterTerminalClient", client_factory)
    settings = SchedulerSettings(
        fence_signing_key="f" * 32,
        secret_file=secret_file,
        jupyter_bootstrap=_settings(),
    )
    database = Database("sqlite+aiosqlite://")
    controller = build_jupyter_bootstrap_controller(
        settings,
        database,
        AgentConnectionRegistry(),
    )
    assert controller is not None
    assert captured == {
        "url": "https://jupyter.invalid/base",
        "token": "production-secret",
    }
    await controller.close()
    await database.close()


def test_builder_rejects_test_fallback_credentials_and_plaintext_remote_url(
    tmp_path: Path,
) -> None:
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir(mode=0o700)
    secret_dir.chmod(0o700)
    secret_file = secret_dir / "scheduler.env"
    settings = SchedulerSettings(
        fence_signing_key="f" * 32,
        secret_file=secret_file,
        jupyter_bootstrap=_settings(),
    )
    database = Database("sqlite+aiosqlite://")

    secret_file.write_text(
        "TOPODROME_TEST_JUPYTER_URL=https://test.invalid\n"
        "TOPODROME_TEST_JUPYTER_TOKEN=must-not-fallback\n",
        encoding="utf-8",
    )
    secret_file.chmod(0o600)
    with pytest.raises(ValueError, match="TOPODROME_JUPYTER_BOOTSTRAP_URL"):
        build_jupyter_bootstrap_controller(settings, database, AgentConnectionRegistry())

    secret = "must-not-appear-in-error"  # noqa: S105 - redaction canary
    secret_file.write_text(
        f"TOPODROME_JUPYTER_BOOTSTRAP_URL=https://jupyter.invalid/?token={secret}\n"
        f"TOPODROME_JUPYTER_BOOTSTRAP_TOKEN={secret}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as raised:
        build_jupyter_bootstrap_controller(settings, database, AgentConnectionRegistry())
    assert secret not in str(raised.value)

    secret_file.write_text(
        "TOPODROME_JUPYTER_BOOTSTRAP_URL=http://192.0.2.10:8888\n"
        f"TOPODROME_JUPYTER_BOOTSTRAP_TOKEN={secret}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"non-loopback.*HTTPS") as raised:
        build_jupyter_bootstrap_controller(settings, database, AgentConnectionRegistry())
    assert secret not in str(raised.value)


def test_enabled_builder_rejects_overly_permissive_secret_file(tmp_path: Path) -> None:
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir(mode=0o700)
    secret_dir.chmod(0o700)
    secret_file = secret_dir / "scheduler.env"
    secret_file.write_text(
        "TOPODROME_JUPYTER_BOOTSTRAP_URL=https://jupyter.invalid\n"
        "TOPODROME_JUPYTER_BOOTSTRAP_TOKEN=protected-value\n",
        encoding="utf-8",
    )
    secret_file.chmod(0o644)
    settings = SchedulerSettings(
        fence_signing_key="f" * 32,
        secret_file=secret_file,
        jupyter_bootstrap=_settings(),
    )

    with pytest.raises(PermissionError, match="mode 0600"):
        build_jupyter_bootstrap_controller(
            settings,
            Database("sqlite+aiosqlite://"),
            AgentConnectionRegistry(),
        )


async def test_controller_is_single_flight_and_close_cancels_and_closes(
    database: Database,
    tmp_path: Path,
) -> None:
    client = BlockingTerminalClient()
    controller = _controller(
        database,
        AgentConnectionRegistry(),
        client,
        FakeClock(),
        tmp_path,
    )

    await asyncio.gather(*(controller.reconcile_once() for _ in range(5)))
    await asyncio.wait_for(client.started.wait(), timeout=1)
    assert len(client.calls) == 1
    assert client.calls[0] == {
        "command": "/opt/topodrome/bin/start-outbound-agent",
        "ready_marker": "TOPODROME_AGENT_READY",
        "timeout_seconds": 90.0,
        "cleanup_terminal": True,
        "keep_terminal_on_success": True,
        "terminal_name": client.calls[0]["terminal_name"],
    }

    await controller.close()
    assert client.cancelled.is_set()
    assert client.closed


async def test_fresh_exact_heartbeat_cancels_an_in_flight_recovery(
    database: Database,
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    connections = AgentConnectionRegistry()
    client = BlockingTerminalClient()
    controller = _controller(database, connections, client, clock, tmp_path)
    await controller.reconcile_once()
    await asyncio.wait_for(client.started.wait(), timeout=1)

    identity = await _register_connection(connections, session_id="connected-session")
    await _store_heartbeat(database, identity=identity, heartbeat_at=clock.now())
    await controller.reconcile_once()

    assert client.cancelled.is_set()
    assert not controller.attempt_in_flight
    assert controller.consecutive_attempts == 0
    clock.advance(60)
    await controller.reconcile_once()
    assert len(client.calls) == 1
    await controller.close()


async def test_controller_uses_capped_backoff_and_replaces_stale_terminal(
    database: Database,
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    secret = "response-body-secret"  # noqa: S105 - redaction canary
    client = FakeTerminalClient(
        [RuntimeError(f"remote said {secret}"), "terminal-2", "terminal-3"]
    )
    controller = _controller(
        database,
        AgentConnectionRegistry(),
        client,
        clock,
        tmp_path,
    )

    with capture_logs() as logs:
        await controller.reconcile_once()
        await _finish_attempt(controller)
    assert secret not in repr(logs)
    first_terminal = str(client.calls[0]["terminal_name"])
    assert controller.consecutive_attempts == 1
    assert controller.next_attempt_at == 5

    clock.advance(4)
    await controller.reconcile_once()
    assert len(client.calls) == 1
    clock.advance(1)
    await controller.reconcile_once()
    await _finish_attempt(controller)
    assert len(client.calls) == 2
    assert client.deleted == [first_terminal]
    assert controller.next_attempt_at == 15

    clock.advance(10)
    await controller.reconcile_once()
    await _finish_attempt(controller)
    second_terminal = str(client.calls[1]["terminal_name"])
    assert client.deleted == [first_terminal, second_terminal]
    assert len(client.calls) == 3
    assert controller.next_attempt_at == 27
    await controller.close()
    assert client.deleted == [first_terminal, second_terminal]


async def test_exact_fresh_connection_suppresses_bootstrap_but_stale_or_replaced_does_not(
    database: Database,
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    connections = AgentConnectionRegistry()
    identity = await _register_connection(connections, session_id="session-1")
    await _store_heartbeat(database, identity=identity, heartbeat_at=clock.now())
    client = FakeTerminalClient(["stale", "replaced"])
    controller = _controller(database, connections, client, clock, tmp_path)

    await controller.reconcile_once()
    assert not client.calls

    clock.advance(31)
    await controller.reconcile_once()
    await _finish_attempt(controller)
    assert len(client.calls) == 1

    clock.advance(5)
    async with database.transaction() as session:
        row = await session.get(SiteRow, "remote-jupyter-1")
        assert row is not None
        row.heartbeat_at = clock.now()
    replacement = await _register_connection(connections, session_id="session-2")
    assert replacement != identity
    await controller.reconcile_once()
    await _finish_attempt(controller)
    assert len(client.calls) == 2
    await controller.close()


async def test_controller_recovers_exact_owned_terminal_across_restart(
    database: Database,
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    ownership_path = tmp_path / "state" / "bootstrap.json"
    first_client = FakeTerminalClient(["success"])
    first = JupyterBootstrapController(
        database=database,
        connections=AgentConnectionRegistry(),
        settings=_settings(),
        heartbeat_ttl_seconds=30,
        client=first_client,
        ownership_path=ownership_path,
        terminal_origin="https://jupyter.invalid/base",
        now=clock.now,
        monotonic=clock.monotonic_now,
    )
    await first.reconcile_once()
    await _finish_attempt(first)
    first_name = str(first_client.calls[0]["terminal_name"])
    persisted = ownership_path.read_text(encoding="utf-8")
    assert first_name in persisted
    assert ownership_path.stat().st_mode & 0o777 == 0o600
    await first.close()

    clock.advance(5)
    second_client = FakeTerminalClient(["success"])
    replacement = JupyterBootstrapController(
        database=database,
        connections=AgentConnectionRegistry(),
        settings=_settings(),
        heartbeat_ttl_seconds=30,
        client=second_client,
        ownership_path=ownership_path,
        terminal_origin="https://jupyter.invalid/base",
        now=clock.now,
        monotonic=clock.monotonic_now,
    )
    await replacement.reconcile_once()
    await _finish_attempt(replacement)

    assert second_client.deleted == [first_name]
    assert len(second_client.calls) == 1
    replacement_name = str(second_client.calls[0]["terminal_name"])
    assert replacement_name != first_name
    assert replacement_name in ownership_path.read_text(encoding="utf-8")
    await replacement.close()


async def test_controller_never_replays_ownership_against_a_different_server(
    database: Database,
    tmp_path: Path,
) -> None:
    ownership_path = tmp_path / "state" / "bootstrap.json"
    first_client = FakeTerminalClient(["success"])
    first = JupyterBootstrapController(
        database=database,
        connections=AgentConnectionRegistry(),
        settings=_settings(),
        heartbeat_ttl_seconds=30,
        client=first_client,
        ownership_path=ownership_path,
        terminal_origin="https://first-jupyter.invalid/base",
    )
    await first.reconcile_once()
    await _finish_attempt(first)
    await first.close()

    second_client = FakeTerminalClient(["success"])
    with pytest.raises(ValueError, match="invalid Jupyter bootstrap ownership"):
        JupyterBootstrapController(
            database=database,
            connections=AgentConnectionRegistry(),
            settings=_settings(),
            heartbeat_ttl_seconds=30,
            client=second_client,
            ownership_path=ownership_path,
            terminal_origin="https://different-jupyter.invalid/base",
        )
    assert not second_client.deleted


def test_controller_fails_closed_on_unowned_persisted_terminal(
    database: Database,
    tmp_path: Path,
) -> None:
    ownership_path = tmp_path / "state" / "bootstrap.json"
    ownership_path.parent.mkdir(mode=0o700)
    ownership_path.write_text(
        '{"version":1,"site_id":"remote-jupyter-1",'
        '"terminal_name":"foreign-terminal"}',
        encoding="utf-8",
    )
    ownership_path.chmod(0o600)

    with pytest.raises(ValueError, match="invalid Jupyter bootstrap ownership"):
        JupyterBootstrapController(
            database=database,
            connections=AgentConnectionRegistry(),
            settings=_settings(),
            heartbeat_ttl_seconds=30,
            client=FakeTerminalClient(),
            ownership_path=ownership_path,
            terminal_origin="https://jupyter.invalid/base",
        )


async def test_scheduler_lifespan_runs_and_closes_bootstrap_controller(
    database: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LifecycleController:
        def __init__(self) -> None:
            self.reconciled = asyncio.Event()
            self.closed = False

        async def reconcile_once(self) -> None:
            self.reconciled.set()

        async def close(self) -> None:
            self.closed = True

    controller = LifecycleController()

    def build(*_: Any, **__: Any) -> LifecycleController:
        return controller

    monkeypatch.setattr(app_module, "build_jupyter_bootstrap_controller", build)
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path,
        fence_signing_key="f" * 32,
        dispatch_interval_seconds=0.01,
    )
    app = create_app(settings, database=database)
    async with LifespanManager(app):
        await asyncio.wait_for(controller.reconciled.wait(), timeout=1)

    assert controller.closed
