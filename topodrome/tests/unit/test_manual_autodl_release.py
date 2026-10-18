# ruff: noqa: S106 -- synthetic test credentials
from datetime import timedelta
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import select
from typer.testing import CliRunner

from tests.unit.test_manual_autodl import SITE, controller
from topodrome.client.async_client import AsyncClient
from topodrome.client.cli import app as cli
from topodrome.db.models import AttemptRow, GPUReservationRow, ProviderInstanceRow
from topodrome.db.repositories import SchedulerRepository
from topodrome.domain.enums import (
    AttemptState,
    GPUReservationState,
    JobState,
    ProviderInstanceState,
)
from topodrome.domain.specs import GPUInventory, GPUReservationRequest, JobSpec, SiteHeartbeat
from topodrome.protocol.fencing import FencingAuthority
from topodrome.providers.autodl.lifecycle import IdleActivity
from topodrome.scheduler.app import create_app
from topodrome.scheduler.control import AgentConnection
from topodrome.scheduler.reservations import GPUReservationCoordinator


async def ready_site(c, database):
    messages = []

    async def send(message):
        messages.append(message)

    connection = AgentConnection(SITE, "test-session", send)
    await c.connections.register(connection)
    repository = SchedulerRepository()
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            SiteHeartbeat(
                site_id=SITE,
                site_kind="autodl",
                agent_session_id="test-session",
                sequence=1,
                gpus=(
                    GPUInventory(
                        uuid="gpu0",
                        index=0,
                        name="5090",
                        total_memory_bytes=32 * 1024**3,
                        free_memory_bytes=32 * 1024**3,
                        utilization_percent=0,
                    ),
                ),
            ),
            connection_generation=connection.generation,
        )
        session.add(
            ProviderInstanceRow(
                id="instance",
                provider="autodl",
                external_id=SITE,
                site_id=SITE,
                state=ProviderInstanceState.RUNNING,
                gpu_spec="manual",
                image_uuid="manual",
            )
        )
    return repository, messages


async def running_job(database, repository):
    authority = FencingAuthority(b"f" * 32)
    async with database.transaction() as session:
        job = await repository.submit_job(
            session,
            JobSpec(
                entrypoint="run.py",
                model={"repo_id": "test/llama"},
                retry={"max_attempts": 1, "retry_on": [], "backoff_seconds": 0},
            ),
        )
        await repository.claim_locked_job(session, job)
        offer = await repository.create_offer(
            session,
            job=job,
            site_id=SITE,
            gpu_uuids=("gpu0",),
            authority=authority,
            offer_ttl=timedelta(minutes=5),
            token_ttl=timedelta(hours=1),
        )
    async with database.transaction() as session:
        await repository.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id=SITE,
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(minutes=5),
        )
        for state in (AttemptState.STARTING, AttemptState.RUNNING):
            await repository.update_attempt(
                session,
                attempt_id=offer.attempt_id,
                site_id=SITE,
                token=offer.fencing_token,
                authority=authority,
                target=state,
            )
    return offer, authority


@pytest.mark.asyncio
async def test_release_waits_for_work_then_drains_without_idle_delay(tmp_path, database):
    c = controller(tmp_path, database)
    repository, messages = await ready_site(c, database)
    offer, _ = await running_job(database, repository)
    result = await c.request_release(SITE)
    assert result["state"] == "release_pending" and result["force"] is False
    assert await c.request_release(SITE) == result
    async with database.session() as session:
        assert await repository.placement_inventory(session) == []
        assert (await session.get(AttemptRow, offer.attempt_id)).state is AttemptState.RUNNING
        reservations = list(await session.scalars(select(GPUReservationRow)))
        assert len(reservations) == 1 and not reservations[0].force
    c._drain = AsyncMock()
    c.bridge.current_activity = AsyncMock(return_value=IdleActivity(job_running=True))
    await c.reconcile_once()
    c._drain.assert_not_awaited()
    assert not messages
    # Simulate a scheduler restart: the new controller sees persisted intent.
    restarted = controller(tmp_path, database)
    restarted._drain = AsyncMock()
    restarted.bridge.current_activity = AsyncMock(return_value=IdleActivity())
    await restarted.reconcile_once()
    await restarted.tasks[SITE]
    restarted._drain.assert_awaited_once_with(SITE)
    await restarted.close()
    await c.close()


@pytest.mark.asyncio
async def test_completed_release_journals_shutdown_and_clears_only_its_reservations(
    tmp_path, database, monkeypatch
):
    from topodrome.db.models import ProviderOperationRow
    from topodrome.deployment.manual_autodl import private_read, private_write
    from topodrome.domain.enums import ProviderOperationState

    c = controller(tmp_path, database)
    await ready_site(c, database)
    credentials = c.root / (SITE + ".json")
    private_write(credentials, {"identity": "test-only"})
    await c.request_release(SITE)
    c.bridge.mark_draining = AsyncMock()
    c.bridge.current_activity = AsyncMock(return_value=IdleActivity())
    for method in ("flush_and_verify", "persist_regional_state", "is_durable", "mark_powered_off"):
        setattr(c.bridge, method, AsyncMock(return_value=True))
    c.bridge.stop_runtime = AsyncMock(return_value=1)

    async def guarded(*args, **kwargs):
        await kwargs["action"]()
        return True

    async def shutdown(record, *, before_submit):
        await before_submit()

    stop = AsyncMock(side_effect=shutdown)
    monkeypatch.setattr("topodrome.scheduler.manual_autodl.shutdown", stop)
    c.bridge.run_power_off_if_authorized = guarded
    await c._drain(SITE)
    stop.assert_awaited_once()
    c.bridge.mark_powered_off.assert_awaited_once_with(SITE, expected_generation=1)
    assert private_read(credentials)["disabled"] is True
    async with database.session() as session:
        reservations = list(await session.scalars(select(GPUReservationRow)))
        assert reservations and all(r.state is GPUReservationState.CANCELED for r in reservations)
        operation = await session.scalar(select(ProviderOperationRow))
        assert operation.state is ProviderOperationState.APPLIED
        assert operation.request_redacted["transport"] == "ssh"
    await c.close()


@pytest.mark.asyncio
async def test_force_release_preempts_and_retries_same_job_despite_retry_budget(tmp_path, database):
    c = controller(tmp_path, database)
    repository, messages = await ready_site(c, database)
    offer, authority = await running_job(database, repository)
    original = await c.request_release(SITE)
    forced = await c.request_release(SITE, force=True)
    assert forced["force"] and forced["reservation_ids"] == original["reservation_ids"]
    coordinator = GPUReservationCoordinator(
        database=database,
        repository=repository,
        connections=c.connections,
        heartbeat_ttl_seconds=30,
    )
    assert await coordinator.reconcile_once() == 1
    assert messages[-1]["kind"] == "request_checkpoint"
    assert messages[-1]["payload"]["attempt_id"] == offer.attempt_id
    assert messages[-1]["payload"]["cancel_requested"] is False
    async with database.transaction() as session:
        for state in (AttemptState.PREEMPT_REQUESTED, AttemptState.PREEMPTED):
            await repository.update_attempt(
                session,
                attempt_id=offer.attempt_id,
                site_id=SITE,
                token=offer.fencing_token,
                authority=authority,
                target=state,
                reason_code="preempted",
            )
        assert (await repository.get_job(session, offer.job_id)).state is JobState.RETRY_WAIT
        assert await repository.placement_inventory(session) == []
    await c.close()


@pytest.mark.asyncio
async def test_release_preserves_preexisting_reservations(tmp_path, database):
    c = controller(tmp_path, database)
    repository, _ = await ready_site(c, database)
    async with database.transaction() as session:
        prior = await repository.create_gpu_reservation(
            session,
            GPUReservationRequest(
                site_id=SITE, gpu_uuid="gpu0", duration_seconds=3600, reason="existing maintenance"
            ),
        )
    await c.request_release(SITE, force=True)
    async with database.transaction() as session:
        row = await session.get(ProviderInstanceRow, "instance")
        assert (await session.get(GPUReservationRow, prior.id)).force
        await c._clear_release_reservations(session, row)
    async with database.session() as session:
        restored = await session.get(GPUReservationRow, prior.id)
        assert restored.state is GPUReservationState.READY and not restored.force
        assert restored.reason == "existing maintenance"
    await c.close()


@pytest.mark.asyncio
async def test_release_api_auth_sdk_and_removed_rental_endpoint(tmp_path, database):
    c = controller(tmp_path, database)
    settings = c.settings.model_copy(update={"api_token": SecretStr("operator-test")})
    app = create_app(settings, database=database, run_dispatcher=False)
    release = AsyncMock(return_value={"site_id": SITE, "state": "release_pending", "force": True})
    app.state.service.provider_controller.request_release = release
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient("http://test", transport=transport) as client:
        response = await client._http.post(
            f"/api/v1/providers/autodl/registrations/{SITE}/release", json={"force": True}
        )
        assert response.status_code in {401, 403}
    release.assert_not_awaited()
    async with AsyncClient("http://test", token="operator-test", transport=transport) as client:
        assert (await client.release_autodl(SITE, force=True))["state"] == "release_pending"
        response = await client._http.put(
            "/api/v1/providers/autodl/control", json={"enabled": True}
        )
        assert response.status_code == 405
    release.assert_awaited_once_with(SITE, force=True)


@pytest.mark.parametrize(
    "args",
    [
        ["autodl", "on"],
        ["autodl", "off"],
        ["autodl", "--one-gpu", "1"],
        ["autodl", "release"],
        ["autodl", "--force"],
    ],
)
def test_removed_rental_commands_and_invalid_release_arguments(args):
    assert CliRunner().invoke(cli, args).exit_code != 0
