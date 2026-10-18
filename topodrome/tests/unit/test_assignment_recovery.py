from __future__ import annotations

import base64
from datetime import timedelta

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.dialects import postgresql

from topodrome.config import SchedulerSettings
from topodrome.db.models import LeaseGPUClaimRow
from topodrome.db.repositories import SchedulerRepository
from topodrome.domain.enums import AttemptState, JobState, ModelAvailability, SiteKind
from topodrome.domain.specs import (
    GPUInventory,
    GPUReservationRequest,
    JobSpec,
    ModelCapability,
    SiteHeartbeat,
)
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
from topodrome.scheduler.service import SchedulerService
from topodrome.scheduler.source_store import SourceBundle

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("pre_staged_inputs")]


def heartbeat(sequence=1):
    return SiteHeartbeat(
        site_id="lab1",
        site_kind=SiteKind.LAB,
        agent_session_id="assignment-test",
        sequence=sequence,
        gpus=(
            GPUInventory(
                uuid="gpu-0",
                index=0,
                name="test-gpu",
                total_memory_bytes=80 * 1024**3,
                free_memory_bytes=80 * 1024**3,
                utilization_percent=0,
            ),
        ),
        models=(
            ModelCapability(
                repo_id="test/model",
                revision="rev",
                backend="transformers",
                backend_version="configured",
                dtype="bfloat16",
                availability=ModelAvailability.CACHED,
            ),
        ),
    )


@pytest.fixture
async def dispatch_ready(database, tmp_path):
    deliveries = []

    async def capture(message):
        deliveries.append(message)

    connections = AgentConnectionRegistry()
    connection = AgentConnection("lab1", "assignment-test", capture)
    await connections.register(connection)
    repository = SchedulerRepository()
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            heartbeat(),
            connection_generation=connection.generation,
        )
    service = SchedulerService(
        database=database,
        repository=repository,
        connections=connections,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
            offer_ttl_seconds=180,
        ),
    )
    source = service.sources.put(
        SourceBundle(
            files={
                "experiment.py": base64.b64encode(b"def run(context): return None\n").decode(),
            }
        )
    )
    submitted = await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=source.source_ref,
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
            retry={"max_attempts": 1},
        )
    )
    return service, connection, deliveries, submitted.job_id


async def active_claims(database):
    async with database.session() as session:
        return await session.scalar(
            select(func.count())
            .select_from(
                LeaseGPUClaimRow,
            )
            .where(LeaseGPUClaimRow.active.is_(True))
        )


@pytest.mark.parametrize("race", ["disconnect", "replacement", "cancel"])
async def test_proven_unsent_offer_recovers_immediately(dispatch_ready, monkeypatch, race):
    service, connection, deliveries, job_id = dispatch_ready
    original = service.connections.offer

    async def unsent(offer, **kwargs):
        if race == "cancel":
            await service.cancel(job_id)
        if race == "replacement":
            await service.connections.register(
                AgentConnection(
                    "lab1",
                    "assignment-test",
                    connection.send,
                )
            )
        else:
            await service.connections.unregister(
                "lab1",
                connection.session_id,
                generation=connection.generation,
            )
        return await original(offer, **kwargs)

    monkeypatch.setattr(service.connections, "offer", unsent)
    assert not await service.dispatch_once()
    expected = JobState.CANCELED if race == "cancel" else JobState.QUEUED
    assert (await service.status(job_id)).state is expected
    attempts = await service.attempts(job_id)
    assert attempts[0]["state"] == AttemptState.REJECTED
    assert attempts[0]["reason_code"] == "assignment_undelivered"
    assert await active_claims(service.database) == 0
    assert attempts[0]["attempt_id"] not in service._live_tokens
    assert not any(message["kind"] == "offer_assignment" for message in deliveries)
    async with service.database.transaction() as session:
        assert not await service.repository.withdraw_undelivered_offer(
            session,
            attempt_id=attempts[0]["attempt_id"],
            site_id="lab1",
        )

    if race == "cancel":
        return
    # Recovery keeps the logical job and doesn't consume its failure budget.
    monkeypatch.setattr(service.connections, "offer", original)
    await service.connections.register(connection)
    async with service.database.transaction() as session:
        await service.repository.upsert_heartbeat(
            session,
            heartbeat(2),
            connection_generation=connection.generation,
        )
    assert await service.dispatch_once()
    retry = (await service.attempts(job_id))[1]
    assert retry["attempt_number"] == 2
    assert retry["state"] == AttemptState.OFFERED
    assert await active_claims(service.database) == 1


@pytest.mark.parametrize("delivered", [False, True])
async def test_ambiguous_send_keeps_offer_fenced(dispatch_ready, monkeypatch, delivered):
    service, connection, deliveries, job_id = dispatch_ready
    capture = connection.send

    async def ambiguous(message):
        if delivered:
            await capture(message)
        raise ConnectionError("test socket failed; receipt is unknown")

    monkeypatch.setattr(connection, "send", ambiguous)
    assert not await service.dispatch_once()
    assert (await service.status(job_id)).state is JobState.ASSIGNING
    attempt = (await service.attempts(job_id))[0]
    assert attempt["state"] == AttemptState.OFFERED
    assert attempt["attempt_id"] in service._live_tokens
    assert await active_claims(service.database) == 1
    assert bool(deliveries) == delivered


async def test_recovery_never_revokes_a_racing_acceptance(dispatch_ready, monkeypatch):
    service, _, _, job_id = dispatch_ready

    async def accepted(offer, **kwargs):
        async with service.database.transaction() as session:
            await service.repository.accept_offer(
                session,
                attempt_id=offer.attempt_id,
                site_id=offer.site_id,
                token=offer.fencing_token,
                authority=service.fencing,
                lease_ttl=timedelta(seconds=90),
            )
        return False

    monkeypatch.setattr(service.connections, "offer", accepted)
    assert not await service.dispatch_once()
    assert (await service.status(job_id)).state is JobState.RUNNING
    attempt = (await service.attempts(job_id))[0]
    assert attempt["state"] == AttemptState.ACCEPTED
    assert attempt["attempt_id"] in service._live_tokens
    assert await active_claims(service.database) == 1


@pytest.mark.parametrize("operation", ["offer", "reservation"])
async def test_inventory_claim_locks_site_before_gpu(dispatch_ready, operation):
    service, _, _, _ = dispatch_ready
    statements = []

    def record(connection, statement, multiparams, params, execution_options):
        if getattr(statement, "_for_update_arg", None) is not None:
            statements.append(str(statement.compile(dialect=postgresql.dialect())))

    engine = service.database.engine.sync_engine
    event.listen(engine, "before_execute", record)
    try:
        if operation == "offer":
            assert await service.dispatch_once()
        else:
            async with service.database.transaction() as session:
                await service.repository.create_gpu_reservation(
                    session,
                    GPUReservationRequest(
                        site_id="lab1",
                        gpu_uuid="gpu-0",
                        duration_seconds=600,
                        reason="inventory lock ordering test",
                    ),
                )
    finally:
        event.remove(engine, "before_execute", record)
    site = next(i for i, statement in enumerate(statements) if "FROM sites" in statement)
    gpu = next(i for i, statement in enumerate(statements) if "FROM gpus" in statement)
    assert site < gpu
    assert "FOR UPDATE" in statements[site]
    assert "FOR UPDATE" in statements[gpu]
