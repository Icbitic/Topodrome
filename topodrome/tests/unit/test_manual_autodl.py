# ruff: noqa: S105, S106
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr
from sqlalchemy import select

from topodrome.autodl_identity import manual_token
from topodrome.config import SchedulerSettings
from topodrome.db.models import ProviderInstanceRow
from topodrome.deployment.manual_autodl import (
    ManualRegistration,
    parse_ssh,
    private_read,
    private_write,
)
from topodrome.domain.enums import ProviderInstanceState as State
from topodrome.domain.specs import utc_now
from topodrome.providers.autodl.lifecycle import IdleActivity
from topodrome.scheduler import manual_autodl as module
from topodrome.scheduler.control import AgentConnectionRegistry
from topodrome.scheduler.manual_autodl import ManualAutoDLController

SITE = "autodl-manual-" + "a" * 24


@pytest.mark.parametrize(
    "command",
    [
        "ssh -p 22 root@x.seetacloud.com; shutdown",
        "ssh -p 0 root@x.seetacloud.com",
        "ssh -p 70000 root@x.seetacloud.com",
        "ssh -p 22 root@localhost",
        "ssh -o ProxyCommand=bad root@x.seetacloud.com",
        "ssh -p 22 root@x.seetacloud.com true",
    ],
)
def test_reject_shell_and_options(command):
    with pytest.raises(ValueError):
        parse_ssh(command)


def test_parse_and_private_credentials(tmp_path):
    assert parse_ssh("ssh -p 23505 root@connect.bjb2.seetacloud.com") == (
        "connect.bjb2.seetacloud.com",
        23505,
        "root",
    )
    path = tmp_path / "private" / "instance.json"
    private_write(path, {"password": "test-password"})
    assert path.stat().st_mode & 0o777 == 0o600
    assert private_read(path)["password"] == "test-password"
    path.chmod(0o644)
    with pytest.raises(PermissionError):
        private_read(path)
    assert "test-password" not in repr(
        ManualRegistration(ssh="ssh -p 22 root@x.seetacloud.com", password="test-password")
    )


def controller(tmp_path, database):
    settings = SchedulerSettings(
        fence_signing_key=SecretStr("f" * 64),
        insecure_development_mode=True,
        global_root=tmp_path / "global",
        agent_tokens={"autodl": SecretStr("base-test-token")},
        autodl={
            "manual_ssh": True,
            "manual_state_dir": tmp_path / "private",
            "manual_bootstrap_dir": tmp_path / "bundle",
            "idle_timeout_seconds": 300,
        },
    )
    return ManualAutoDLController(settings, database, AgentConnectionRegistry())


@pytest.mark.asyncio
async def test_registration_private_idempotent_no_api(tmp_path, database, monkeypatch):
    c = controller(tmp_path, database)
    record = {
        "site_id": SITE,
        "host_key": "pinned",
        "gpus": ["GPU-1"],
        "architecture": "x86_64",
        "libc": "glibc 2.35",
        "password": "test-password",
    }
    monkeypatch.setattr(module, "inspect", AsyncMock(return_value=record.copy()))
    install = AsyncMock()
    monkeypatch.setattr(module, "install", install)
    request = ManualRegistration(ssh="ssh -p 22 root@x.seetacloud.com", password="test-password")
    result = await c.register(request)
    await c.tasks[SITE]
    assert "test-password" not in str(result)
    assert c.enrollment_token(SITE) == manual_token("base-test-token", SITE)
    assert (await c.register(request))["existing"] is True
    assert install.await_count == 1
    async with database.session() as s:
        row = await s.scalar(select(ProviderInstanceRow))
        assert "test-password" not in str(row.metadata_json)
    await c.close()


@pytest.mark.asyncio
async def test_unregister_disconnected_preserves_audit_and_disables_token(tmp_path, database):
    c = controller(tmp_path, database)
    private_write(c.root / (SITE + ".json"), {"site_id": SITE})
    async with database.transaction() as s:
        s.add(
            ProviderInstanceRow(
                id="removed",
                provider="autodl",
                external_id=SITE,
                site_id=SITE,
                state=State.RUNNING,
                gpu_spec="manual",
                image_uuid="manual",
            )
        )
    c.connections.identity = AsyncMock(return_value=("session", 1))
    with pytest.raises(ValueError, match="still connected"):
        await c.unregister(SITE)
    c.connections.identity = AsyncMock(return_value=None)
    result = await c.unregister(SITE)
    assert result["provider_power_state"] == "unknown"
    assert c.enrollment_token(SITE) is None
    assert (await c.unregister(SITE))["state"] == "unregistered"
    async with database.session() as s:
        row = await s.get(ProviderInstanceRow, "removed")
        assert row.state is State.RELEASED
        assert row.metadata_json["unregistered_at"]
    with pytest.raises(ValueError):
        await c.unregister("../../bad")
    await c.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("age,removed", [(100, False), (601, True)])
async def test_auto_unregister_offline_grace(tmp_path, database, age, removed):
    from topodrome.db.models import SiteRow
    from topodrome.domain.enums import SiteKind, SiteState

    c = controller(tmp_path, database)
    private_write(c.root / (SITE + ".json"), {"site_id": SITE})
    async with database.transaction() as s:
        s.add(
            SiteRow(
                id=SITE,
                display_name=SITE,
                kind=SiteKind.AUTODL,
                state=SiteState.OFFLINE,
                heartbeat_at=utc_now() - timedelta(seconds=age),
            )
        )
        s.add(
            ProviderInstanceRow(
                id="offline",
                provider="autodl",
                external_id=SITE,
                site_id=SITE,
                state=State.RUNNING,
                gpu_spec="manual",
                image_uuid="manual",
            )
        )
    await c.reconcile_once()
    async with database.session() as s:
        row = await s.get(ProviderInstanceRow, "offline")
        assert (row.state is State.RELEASED) == removed
    assert (c.enrollment_token(SITE) is None) == removed
    await c.close()


@pytest.mark.asyncio
async def test_busy_or_stale_activity_resets_idle(tmp_path, database):
    c = controller(tmp_path, database)
    async with database.transaction() as s:
        s.add(
            ProviderInstanceRow(
                id="i",
                provider="autodl",
                external_id=SITE,
                site_id=SITE,
                state=State.RUNNING,
                gpu_spec="manual",
                image_uuid="manual",
            )
        )
    # Real bridge treats missing/stale telemetry as busy, never shutdown.
    c._drain = AsyncMock()
    await c.reconcile_once()
    c._drain.assert_not_awaited()
    async with database.session() as s:
        assert (await s.get(ProviderInstanceRow, "i")).idle_since is None
    c.bridge.current_activity = AsyncMock(return_value=IdleActivity())
    c.bridge.record_idle_observation = AsyncMock(return_value=utc_now() - timedelta(seconds=301))
    await c.reconcile_once()
    await c.tasks[SITE]
    c._drain.assert_awaited_once_with(SITE)
    await c.close()


@pytest.mark.asyncio
async def test_durability_failure_never_sends_shutdown(tmp_path, database, monkeypatch):
    c = controller(tmp_path, database)
    c.bridge.mark_draining = AsyncMock()
    c.bridge.current_activity = AsyncMock(return_value=IdleActivity())
    c.bridge.flush_and_verify = AsyncMock(return_value=False)
    c.bridge.cancel_draining = AsyncMock()
    c.bridge.stop_runtime = AsyncMock()
    stop = AsyncMock()
    monkeypatch.setattr(module, "shutdown", stop)
    await c._drain(SITE)
    c.bridge.cancel_draining.assert_awaited_once()
    c.bridge.stop_runtime.assert_not_awaited()
    stop.assert_not_awaited()
    await c.close()


@pytest.mark.asyncio
async def test_shutdown_journal_and_no_replay_after_disconnect(tmp_path, database, monkeypatch):
    c = controller(tmp_path, database)
    private_write(c.root / (SITE + ".json"), {"password": "test-password"})
    async with database.transaction() as s:
        s.add(
            ProviderInstanceRow(
                id="i",
                provider="autodl",
                external_id=SITE,
                site_id=SITE,
                state=State.STOPPING,
                gpu_spec="manual",
                image_uuid="manual",
            )
        )
    c.bridge.mark_draining = AsyncMock()
    c.bridge.current_activity = AsyncMock(return_value=IdleActivity())
    for name in ("flush_and_verify", "persist_regional_state", "is_durable", "mark_powered_off"):
        setattr(c.bridge, name, AsyncMock(return_value=True))
    c.bridge.stop_runtime = AsyncMock(return_value=1)

    async def guarded(*args, **kwargs):
        await kwargs["action"]()
        return True

    c.bridge.run_power_off_if_authorized = guarded

    async def disconnect(record, *, before_submit):
        await before_submit()
        raise ConnectionError("ambiguous connection")

    stop = AsyncMock(side_effect=disconnect)
    monkeypatch.setattr(module, "shutdown", stop)
    await c._drain(SITE)
    assert stop.await_count == 1
    c.bridge.mark_powered_off.assert_not_awaited()
    # A scheduler retry cannot submit the same journaled command twice.
    await c._drain(SITE)
    assert stop.await_count == 1
    await c.close()


@pytest.mark.asyncio
async def test_manual_api_operator_only_and_sdk(tmp_path, database, monkeypatch):
    import httpx

    from topodrome.client.async_client import AsyncClient
    from topodrome.scheduler.app import create_app

    c = controller(tmp_path, database)
    settings = c.settings.model_copy(update={"api_token": SecretStr("operator-test")})
    app = create_app(settings, database=database, run_dispatcher=False)
    registered = AsyncMock(return_value={"site_id": SITE, "state": "starting"})
    app.state.service.provider_controller.register = registered
    unregistered = AsyncMock(return_value={"site_id": SITE, "state": "unregistered"})
    app.state.service.provider_controller.unregister = unregistered
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient("http://test", transport=transport) as client:
        response = await client._http.post(
            "/api/v1/providers/autodl/register",
            json={"ssh": "ssh -p 22 root@x.seetacloud.com", "password": "test-password"},
        )
        assert response.status_code in {401, 403}
        response = await client._http.delete(f"/api/v1/providers/autodl/registrations/{SITE}")
        assert response.status_code in {401, 403}
    unregistered.assert_not_awaited()
    registered.assert_not_awaited()
    async with AsyncClient("http://test", token="operator-test", transport=transport) as client:
        result = await client.register_autodl("ssh -p 22 root@x.seetacloud.com", "test-password")
        assert result["site_id"] == SITE
        assert (await client.unregister_autodl(SITE))["state"] == "unregistered"
    assert registered.await_count == 1


@pytest.mark.asyncio
async def test_inactive_autodl_hidden_but_audit_retained(tmp_path, database):
    from topodrome.db.models import SiteRow
    from topodrome.db.repositories import SchedulerRepository
    from topodrome.domain.enums import SiteKind, SiteState
    from topodrome.scheduler.status import system_status_snapshot

    async with database.transaction() as s:
        s.add(
            SiteRow(
                id="autodl-old", display_name="Old", kind=SiteKind.AUTODL, state=SiteState.STOPPED
            )
        )
        s.add(
            ProviderInstanceRow(
                id="old",
                provider="autodl",
                external_id="released",
                site_id="autodl-old",
                state=State.RELEASED,
                gpu_spec="5090",
                image_uuid="image",
            )
        )
    async with database.session() as s:
        assert await SchedulerRepository().list_sites(s) == []
        snapshot = await system_status_snapshot(s)
        assert snapshot["sites"] == [] and snapshot["autodl_instances"] == []
        assert await s.get(ProviderInstanceRow, "old") is not None


@pytest.mark.asyncio
async def test_stopping_reconciles_and_preflight_retry_is_bounded(tmp_path, database):
    from topodrome.deployment.manual_autodl import ShutdownPreflightBusy

    c = controller(tmp_path, database)
    async with database.transaction() as s:
        s.add(
            ProviderInstanceRow(
                id="i",
                provider="autodl",
                external_id=SITE,
                site_id=SITE,
                state=State.STOPPING,
                gpu_spec="manual",
                image_uuid="manual",
                shutdown_generation=2,
            )
        )
    c._resume_stop = AsyncMock()
    await c.reconcile_once()
    await c.tasks[SITE]
    c._resume_stop.assert_awaited_once_with(SITE, 2)
    for _ in range(3):
        await c._stop_error(SITE, ShutdownPreflightBusy(), generation=2)
    async with database.session() as s:
        assert (await s.get(ProviderInstanceRow, "i")).state is State.STOPPING
    await c._stop_error(SITE, ShutdownPreflightBusy(), generation=2)
    async with database.session() as s:
        assert (await s.get(ProviderInstanceRow, "i")).state is State.ERROR
    await c.close()


@pytest.mark.asyncio
async def test_old_shutdown_error_cannot_fence_replacement(tmp_path, database):
    c = controller(tmp_path, database)
    async with database.transaction() as s:
        s.add(
            ProviderInstanceRow(
                id="i",
                provider="autodl",
                external_id=SITE,
                site_id=SITE,
                state=State.RUNNING,
                gpu_spec="manual",
                image_uuid="manual",
                shutdown_generation=2,
            )
        )
    await c._stop_error(SITE, RuntimeError(), generation=1)
    async with database.session() as s:
        assert (await s.get(ProviderInstanceRow, "i")).state is State.RUNNING
    await c.close()


@pytest.mark.asyncio
async def test_transport_claims_only_after_idle_preflight(monkeypatch):
    from types import SimpleNamespace

    from topodrome.deployment import manual_autodl as transport

    record = {"hostname": "test-container", "identity": "test-identity"}
    connection = AsyncMock()
    connection.__aenter__.return_value = connection
    monkeypatch.setattr(transport, "connect", lambda _: connection)
    before = AsyncMock()
    connection.run.side_effect = [
        SimpleNamespace(stdout="test-container\ntest-identity"),
        SimpleNamespace(stdout="1234\n"),
    ]
    with pytest.raises(transport.ShutdownPreflightBusy):
        await transport.shutdown(record, before_submit=before)
    before.assert_not_awaited()
    assert connection.run.await_count == 2
    connection.run.reset_mock()
    connection.run.side_effect = [
        SimpleNamespace(stdout="test-container\ntest-identity"),
        SimpleNamespace(stdout=""),
        SimpleNamespace(stdout=""),
    ]
    await transport.shutdown(record, before_submit=before)
    before.assert_awaited_once()
    assert connection.run.call_args.args == ("/usr/bin/shutdown",)
