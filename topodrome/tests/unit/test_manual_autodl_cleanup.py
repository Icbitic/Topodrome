from datetime import timedelta
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from tests.unit.test_manual_autodl import SITE, controller
from tests.unit.test_manual_autodl_release import ready_site, running_job
from topodrome.db.models import KernelSessionRow, ProviderInstanceRow, SiteRow
from topodrome.deployment.manual_autodl import private_read, private_write
from topodrome.domain.enums import KernelMode, KernelState, SiteKind, SiteState
from topodrome.domain.enums import ProviderInstanceState as State
from topodrome.domain.specs import utc_now
from topodrome.scheduler.control import AgentConnection


async def failed_instance(c, database, *, state, age, telemetry):
    old = utc_now() - timedelta(seconds=age)
    private_write(c.root / (SITE + ".json"), {"site_id": SITE})
    async with database.transaction() as session:
        if telemetry != "no-site":
            session.add(
                SiteRow(
                    id=SITE,
                    display_name=SITE,
                    kind=SiteKind.AUTODL,
                    state=SiteState.OFFLINE,
                    heartbeat_at=old if telemetry == "heartbeat" else None,
                )
            )
        session.add(
            ProviderInstanceRow(
                id="failed",
                provider="autodl",
                external_id=SITE,
                site_id=SITE,
                state=state,
                gpu_spec="manual",
                image_uuid="manual",
                created_at=old,
                updated_at=old,
                metadata_json={"registration_at": old.isoformat(), "manual_error": "TestFailure"},
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [State.STARTING_FAILED, State.ERROR, State.RUNNING])
@pytest.mark.parametrize("telemetry", ["heartbeat", "no-heartbeat", "no-site"])
@pytest.mark.parametrize("age,removed", [(100, False), (601, True)])
async def test_cleanup_all_disconnected_states(tmp_path, database, state, telemetry, age, removed):
    c = controller(tmp_path, database)
    await failed_instance(c, database, state=state, age=age, telemetry=telemetry)
    # A new controller simulates restart; no in-memory timer is needed.
    restarted = controller(tmp_path, database)
    await restarted.reconcile_once()
    await restarted.reconcile_once()
    async with database.session() as session:
        row = await session.get(ProviderInstanceRow, "failed")
        assert (row.state is State.RELEASED) == removed
        assert row.metadata_json["manual_error"] == "TestFailure"
        if removed:
            assert row.metadata_json["provider_power_state"] == "unknown"
            site = await session.get(SiteRow, SITE)
            assert site is None or site.state is SiteState.STOPPED
    assert bool(private_read(c.root / (SITE + ".json")).get("disabled")) == removed
    assert not restarted.tasks  # Cleanup never invokes SSH shutdown or provisioning.
    await restarted.close()
    await c.close()


@pytest.mark.asyncio
async def test_connected_failure_is_not_removed(tmp_path, database):
    c = controller(tmp_path, database)
    await failed_instance(c, database, state=State.ERROR, age=700, telemetry="heartbeat")
    await c.connections.register(AgentConnection(SITE, "reconnected", AsyncMock()))
    await c.reconcile_once()
    async with database.session() as session:
        assert (await session.get(ProviderInstanceRow, "failed")).state is State.ERROR
    assert c.enrollment_token(SITE) is not None
    await c.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["registration", "heartbeat", "failure"])
async def test_automatic_cleanup_rechecks_grace_under_lock(tmp_path, database, change):
    c = controller(tmp_path, database)
    await failed_instance(c, database, state=State.ERROR, age=700, telemetry="heartbeat")
    async with database.transaction() as session:
        row = await session.get(ProviderInstanceRow, "failed")
        if change == "registration":
            row.metadata_json = {**row.metadata_json, "registration_at": utc_now().isoformat()}
        elif change == "failure":
            row.updated_at = utc_now()
        else:
            site = await session.get(SiteRow, SITE)
            # Older than heartbeat TTL, but still within automatic cleanup grace.
            site.heartbeat_at = utc_now() - timedelta(seconds=100)
    with pytest.raises(ValueError, match="cleanup grace"):
        await c.unregister(SITE, automatic=True)
    assert c.enrollment_token(SITE) is not None
    await c.close()


@pytest.mark.asyncio
async def test_failed_site_with_live_lease_waits_for_job_recovery(tmp_path, database):
    c = controller(tmp_path, database)
    repository, _ = await ready_site(c, database)
    await running_job(database, repository)
    await c.connections.unregister(SITE, "test-session")
    old = utc_now() - timedelta(seconds=700)
    private_write(c.root / (SITE + ".json"), {"site_id": SITE})
    async with database.transaction() as session:
        row = await session.scalar(select(ProviderInstanceRow))
        row.state, row.created_at, row.updated_at = State.ERROR, old, old
        site = await session.get(SiteRow, SITE)
        site.heartbeat_at = old
    await c.reconcile_once()
    async with database.session() as session:
        assert (await session.scalar(select(ProviderInstanceRow))).state is State.ERROR
    assert c.enrollment_token(SITE) is not None
    await c.close()


@pytest.mark.asyncio
async def test_failed_site_with_kernel_waits_until_recovered(tmp_path, database):
    c = controller(tmp_path, database)
    await failed_instance(c, database, state=State.ERROR, age=700, telemetry="heartbeat")
    async with database.transaction() as session:
        session.add(
            KernelSessionRow(
                id="kernel",
                state=KernelState.RUNNING,
                mode=KernelMode.SERVICE,
                site_id=SITE,
                request={},
            )
        )
    await c.reconcile_once()
    assert c.enrollment_token(SITE) is not None
    async with database.transaction() as session:
        kernel = await session.get(KernelSessionRow, "kernel")
        kernel.state = KernelState.LOST
    await c.reconcile_once()
    assert c.enrollment_token(SITE) is None
    await c.close()
