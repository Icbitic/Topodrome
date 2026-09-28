from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import topodrome.db.repositories as repositories_module
from topodrome.config import SchedulerSettings
from topodrome.db.repositories import SchedulerRepository
from topodrome.db.session import Database
from topodrome.domain.enums import AttemptState, JobState, ModelAvailability, SiteKind
from topodrome.domain.errors import StaleFenceError
from topodrome.domain.events import AgentEnvelope
from topodrome.domain.specs import (
    GPUInventory,
    JobSpec,
    LeaseRenewalGrant,
    ModelCapability,
    SiteHeartbeat,
)
from topodrome.protocol.fencing import FencingAuthority
from topodrome.scheduler.service import SchedulerService
from topodrome.site_agent.agent import Assignment

pytestmark = pytest.mark.component

GIB = 1024**3


def _heartbeat() -> SiteHeartbeat:
    return SiteHeartbeat(
        site_id="lab1",
        site_kind=SiteKind.LAB,
        agent_session_id="retry-recovery",
        sequence=1,
        gpus=(
            GPUInventory(
                uuid="GPU-A",
                index=0,
                name="GPU",
                total_memory_bytes=32 * GIB,
                free_memory_bytes=24 * GIB,
                utilization_percent=0,
            ),
        ),
        models=(
            ModelCapability(
                alias="model",
                revision="rev",
                backend="transformers",
                backend_version="configured",
                dtype="bfloat16",
                availability=ModelAvailability.CACHED,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_retry_deadline_and_checkpoint_lineage_survive_scheduler_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "scheduler.sqlite3"
    dsn = f"sqlite+aiosqlite:///{database_path}"
    authority = FencingAuthority(b"r" * 32)
    started_at = datetime(2035, 1, 1, tzinfo=UTC)
    checkpoint_ref = (
        "jobs/job-placeholder/attempts/attempt-placeholder/"
        "checkpoints/00000001-state.pkl"
    )

    first_database = Database(dsn)
    await first_database.create_schema()
    first_repository = SchedulerRepository()
    try:
        async with first_database.transaction() as session:
            await first_repository.upsert_heartbeat(session, _heartbeat())
            job = await first_repository.submit_job(
                session,
                JobSpec(
                    entrypoint="experiment.py",
                    model={"alias": "model", "revision": "rev"},
                    resources={"gpu": 1, "min_vram_gb": 15},
                    retry={
                        "max_attempts": 3,
                        "backoff_seconds": 60,
                        "retry_on": {"runtime_error"},
                    },
                ),
            )
            job_id = job.id
        async with first_database.transaction() as session:
            job = await first_repository.claim_next_job(session, now=started_at)
            assert job is not None
            first = await first_repository.create_offer(
                session,
                job=job,
                site_id="lab1",
                gpu_uuids=("GPU-A",),
                authority=authority,
                offer_ttl=timedelta(minutes=5),
                token_ttl=timedelta(hours=1),
                now=started_at,
            )
        checkpoint_ref = checkpoint_ref.replace("job-placeholder", job_id).replace(
            "attempt-placeholder", first.attempt_id
        )
        async with first_database.transaction() as session:
            await first_repository.accept_offer(
                session,
                attempt_id=first.attempt_id,
                site_id="lab1",
                token=first.fencing_token,
                authority=authority,
                lease_ttl=timedelta(minutes=5),
                now=started_at,
            )
            for state in (AttemptState.STARTING, AttemptState.RUNNING):
                await first_repository.update_attempt(
                    session,
                    attempt_id=first.attempt_id,
                    site_id="lab1",
                    token=first.fencing_token,
                    authority=authority,
                    target=state,
                    now=started_at,
                )
            await first_repository.update_attempt(
                session,
                attempt_id=first.attempt_id,
                site_id="lab1",
                token=first.fencing_token,
                authority=authority,
                target=AttemptState.FAILED,
                reason_code="retryable_runtime_error",
                checkpoint_ref=checkpoint_ref,
                now=started_at,
            )
    finally:
        await first_database.close()

    second_database = Database(dsn)
    second_repository = SchedulerRepository()
    try:
        async with second_database.session() as session:
            recovered = await second_repository.get_job(session, job_id)
            assert recovered.state is JobState.RETRY_WAIT
            retry_not_before = recovered.retry_not_before
            assert retry_not_before is not None
            if retry_not_before.tzinfo is None:
                retry_not_before = retry_not_before.replace(tzinfo=UTC)
            assert retry_not_before == started_at + timedelta(seconds=60)

        async with second_database.transaction() as session:
            assert (
                await second_repository.claim_next_job(
                    session,
                    now=started_at + timedelta(seconds=59),
                )
                is None
            )

        async with second_database.transaction() as session:
            claimed = await second_repository.claim_next_job(
                session,
                now=started_at + timedelta(seconds=60),
            )
            assert claimed is not None and claimed.id == job_id
            retry = await second_repository.create_offer(
                session,
                job=claimed,
                site_id="lab1",
                gpu_uuids=("GPU-A",),
                authority=authority,
                offer_ttl=timedelta(minutes=5),
                token_ttl=timedelta(hours=1),
                now=started_at + timedelta(seconds=60),
            )
            assert retry.job_id == first.job_id
            assert retry.attempt_id != first.attempt_id
            assert retry.checkpoint_ref == checkpoint_ref
            assignment = Assignment.from_mapping(retry.model_dump(mode="python"))
            assert assignment.checkpoint_ref == checkpoint_ref

        async with second_database.session() as session:
            attempts = await second_repository.list_attempts(session, job_id)
            assert attempts[-1].restore_checkpoint_ref == checkpoint_ref
    finally:
        await second_database.close()


@pytest.mark.asyncio
async def test_scheduler_rotates_long_running_gang_fence_beyond_three_hours(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'scheduler.sqlite3'}")
    await database.create_schema()
    started_at = datetime(2035, 1, 1, tzinfo=UTC)
    clock = started_at
    monkeypatch.setattr(repositories_module, "utc_now", lambda: clock)
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        fence_signing_key="component-rotating-fence-key-000000000",
        lease_ttl_seconds=90,
        fencing_token_ttl_seconds=3 * 60 * 60,
    )
    settings.global_root.mkdir()
    service = SchedulerService(database=database, settings=settings)
    gang_heartbeat = _heartbeat().model_copy(
        update={
            "gpus": (
                *_heartbeat().gpus,
                GPUInventory(
                    uuid="GPU-B",
                    index=1,
                    name="GPU",
                    total_memory_bytes=32 * GIB,
                    free_memory_bytes=24 * GIB,
                    utilization_percent=0,
                ),
            )
        }
    )
    gang_spec = JobSpec(
        entrypoint="experiment.py",
        model={
            "alias": "model",
            "revision": "rev",
            "parallelism": {
                "mode": "model_sharded",
                "degree": 2,
                "engine": "transformers_device_map",
                "strategy": "balanced",
                "max_memory_gb_per_gpu": 15,
            },
        },
        resources={"gpu": 2, "min_vram_gb": 15},
    )
    try:
        async with database.transaction() as session:
            await service.repository.upsert_heartbeat(session, gang_heartbeat)
            submitted = await service.repository.submit_job(session, gang_spec)
        async with database.transaction() as session:
            job = await service.repository.claim_next_job(session, now=clock)
            assert job is not None
            offer = await service.repository.create_offer(
                session,
                job=job,
                site_id="lab1",
                gpu_uuids=("GPU-A", "GPU-B"),
                authority=service.fencing,
                offer_ttl=timedelta(seconds=30),
                token_ttl=timedelta(hours=3),
                lease_ttl=timedelta(seconds=90),
                now=clock,
            )
        async with database.transaction() as session:
            await service.repository.accept_offer(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=service.fencing,
                lease_ttl=timedelta(seconds=90),
                now=clock,
            )

        token = offer.fencing_token
        first_reply = None
        for sequence in range(1, 182):
            clock = started_at + timedelta(seconds=60 * sequence)
            renewal = AgentEnvelope(
                message_id=f"renewal-{sequence}",
                site_id="lab1",
                agent_session_id="long-running-site-session",
                sequence=sequence,
                kind="lease_renewal",
                payload={
                    "attempt_id": offer.attempt_id,
                    "fencing_token": token,
                },
                sent_at=clock,
            )
            reply = await service.handle_agent_envelope(renewal)
            grant = LeaseRenewalGrant.model_validate(reply.payload["lease_renewal"])
            if sequence == 1:
                first_reply = reply
                duplicate = await service.handle_agent_envelope(renewal)
                assert duplicate.payload["duplicate"] is True
                assert duplicate.payload["lease_renewal"] == reply.payload["lease_renewal"]
            token = grant.fencing_token

        assert first_reply is not None
        beyond_original_expiry = started_at + timedelta(seconds=60 * 181 + 1)
        final_payload = service.fencing.verify(
            token,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            now=beyond_original_expiry,
        )
        assert tuple(claim.gpu_uuid for claim in final_payload.claims) == (
            "GPU-A",
            "GPU-B",
        )
        with pytest.raises(StaleFenceError, match="expired"):
            service.fencing.verify(offer.fencing_token, now=beyond_original_expiry)

        clock = beyond_original_expiry
        state_reply = await service.handle_agent_envelope(
            AgentEnvelope(
                message_id="attempt-starting-after-three-hours",
                site_id="lab1",
                agent_session_id="long-running-site-session",
                sequence=182,
                kind="attempt_event",
                payload={
                    "attempt_id": offer.attempt_id,
                    "job_id": submitted.id,
                    "site_id": "lab1",
                    "fencing_token": token,
                    "state": "starting",
                },
                sent_at=clock,
            )
        )
        assert state_reply.kind == "ack"
        async with database.session() as session:
            attempts = await service.repository.list_attempts(session, submitted.id)
            assert attempts[0].state is AttemptState.STARTING
    finally:
        await database.close()
