from datetime import timedelta

import pytest
from sqlalchemy import select

from topodrome.config import SchedulerSettings
from topodrome.db.models import AttemptRow, LeaseRow
from topodrome.domain.enums import LeaseState, SiteKind
from topodrome.domain.events import AgentEnvelope
from topodrome.domain.specs import GPUInventory, JobSpec, SiteHeartbeat
from topodrome.scheduler.service import SchedulerService

pytestmark = pytest.mark.component


@pytest.mark.asyncio
async def test_scheduler_restart_preserves_live_lease_and_agent_renewal(database, tmp_path):
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://", global_root=tmp_path / "global",
        fence_signing_key="scheduler-restart-live-fence-key-0000", lease_ttl_seconds=300,
    )
    first = SchedulerService(database=database, settings=settings)
    heartbeat = SiteHeartbeat(site_id="sqz", site_kind=SiteKind.LAB,
        agent_session_id="same-agent", sequence=1, gpus=(GPUInventory(
            uuid="GPU-A", index=0, name="A800", total_memory_bytes=80 * 1024**3,
            free_memory_bytes=70 * 1024**3, utilization_percent=0,
        ),))
    async with database.transaction() as session:
        await first.repository.upsert_heartbeat(session, heartbeat)
        submitted = await first.repository.submit_job(session, JobSpec(
            entrypoint="experiment.py", model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        ))
        job = await first.repository.claim_next_job(session)
        offer = await first.repository.create_offer(session, job=job, site_id="sqz",
            gpu_uuids=("GPU-A",), authority=first.fencing,
            offer_ttl=timedelta(seconds=180), token_ttl=timedelta(hours=3),
            lease_ttl=timedelta(seconds=300))
        await first.repository.accept_offer(session, attempt_id=offer.attempt_id,
            site_id="sqz", token=offer.fencing_token, authority=first.fencing,
            lease_ttl=timedelta(seconds=300))
    for sequence, state in enumerate(("starting", "running"), start=1):
        await first.handle_agent_envelope(AgentEnvelope(
            site_id="sqz", agent_session_id="same-agent", sequence=sequence, kind="attempt_event",
            payload={"job_id": submitted.id, "attempt_id": offer.attempt_id,
                     "site_id": "sqz", "state": state, "fencing_token": offer.fencing_token},
        ))
    renewal = AgentEnvelope(site_id="sqz", agent_session_id="same-agent", sequence=3,
        kind="lease_renewal", payload={"attempt_id": offer.attempt_id,
                                        "fencing_token": offer.fencing_token})
    acknowledged = await first.handle_agent_envelope(renewal)
    # New process has no connection registry or in-memory live-token cache.
    second = SchedulerService(database=database, settings=settings)
    assert second._live_tokens == {}
    replay = await second.handle_agent_envelope(renewal)
    assert replay.payload["lease_renewal"] == acknowledged.payload["lease_renewal"]
    async with database.session() as session:
        latest = await session.scalar(select(AttemptRow).where(
            AttemptRow.job_id == submitted.id,
        ).order_by(AttemptRow.number.desc()))
        lease = await session.scalar(select(LeaseRow).where(
            LeaseRow.attempt_id == offer.attempt_id,
        ))
        assert latest.id == offer.attempt_id
        assert lease.state is LeaseState.ACTIVE
