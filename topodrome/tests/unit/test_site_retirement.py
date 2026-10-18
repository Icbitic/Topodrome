from __future__ import annotations

from datetime import timedelta

import pytest

from tests.unit.test_scheduler_repository import heartbeat, job_spec
from topodrome.db.models import (
    AttemptRow,
    GPUFenceCounterRow,
    InputStageRow,
    JobRow,
    KernelSessionRow,
    LeaseGPUClaimRow,
    LeaseRow,
    SiteRow,
)
from topodrome.db.repositories import SchedulerRepository
from topodrome.domain.enums import (
    AttemptState,
    JobState,
    KernelMode,
    LeaseState,
    SiteKind,
    SiteState,
)
from topodrome.domain.errors import ConflictError
from topodrome.domain.specs import utc_now
from topodrome.scheduler.status import system_status_snapshot


async def seed(database, *, fresh=False):
    repo = SchedulerRepository()
    now = utc_now()
    async with database.transaction() as session:
        await repo.upsert_heartbeat(
            session, heartbeat(), received_at=now if fresh else now - timedelta(minutes=2)
        )
        session.add(
            JobRow(id="job-old", state=JobState.SUCCEEDED, spec=job_spec().model_dump(mode="json"))
        )
        await session.flush()
        session.add(
            AttemptRow(
                id="attempt-old",
                job_id="job-old",
                number=1,
                site_id="lab1",
                state=AttemptState.SUCCEEDED,
            )
        )
        await session.flush()
        session.add(
            LeaseRow(
                id="lease-old",
                attempt_id="attempt-old",
                site_id="lab1",
                state=LeaseState.RELEASED,
                fencing_token_digest="a" * 64,
                expires_at=now - timedelta(seconds=10),
            )
        )
        session.add(GPUFenceCounterRow(site_id="lab1", gpu_uuid="gpu-0", epoch=7))
        await session.flush()
        session.add(
            LeaseGPUClaimRow(
                lease_id="lease-old", site_id="lab1", gpu_uuid="gpu-0", epoch=7, active=False
            )
        )
    return repo, now


@pytest.mark.asyncio
async def test_retirement_hides_inventory_but_preserves_history(database):
    repo, now = await seed(database)
    async with database.transaction() as session:
        await repo.retire_site(session, "lab1", now=now)
        await repo.retire_site(session, "lab1", now=now + timedelta(seconds=1))
    async with database.session() as session:
        assert (await session.get(SiteRow, "lab1")).state is SiteState.RETIRED
        assert (await session.get(AttemptRow, "attempt-old")).site_id == "lab1"
        assert (await session.get(LeaseRow, "lease-old")).state is LeaseState.RELEASED
        counter = await session.get(GPUFenceCounterRow, {"site_id": "lab1", "gpu_uuid": "gpu-0"})
        assert counter.epoch == 7
        assert await repo.list_sites(session) == []
        assert await repo.placement_inventory(session) == []
        snapshot = await system_status_snapshot(session)
        assert snapshot["sites"] == [] and snapshot["gpus"] == []
        assert snapshot["jobs"][0]["site_id"] == "lab1"


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence,agent_session", [(1, "session"), (2, "session"), (1, "new")])
async def test_retired_identity_cannot_be_revived_by_any_heartbeat(
    database, sequence, agent_session
):
    repo, now = await seed(database)
    async with database.transaction() as session:
        await repo.retire_site(session, "lab1", now=now)
    async with database.transaction() as session:
        with pytest.raises(ConflictError, match="retired"):
            await repo.upsert_heartbeat(
                session, heartbeat(sequence).model_copy(update={"agent_session_id": agent_session})
            )
        assert (await session.get(SiteRow, "lab1")).state is SiteState.RETIRED


@pytest.mark.asyncio
async def test_fresh_site_and_provider_site_cannot_be_retired(database):
    repo, now = await seed(database, fresh=True)
    async with database.transaction() as session:
        with pytest.raises(ConflictError, match="fresh heartbeat"):
            await repo.retire_site(session, "lab1", now=now)
        site = await session.get(SiteRow, "lab1")
        site.kind = SiteKind.AUTODL
        with pytest.raises(ConflictError, match="provider registration"):
            await repo.retire_site(session, "lab1", now=now + timedelta(minutes=5))


@pytest.mark.asyncio
@pytest.mark.parametrize("blocker", ["attempt", "finalization", "claim", "kernel", "staging"])
async def test_retirement_refuses_active_work_without_modification(database, blocker):
    repo, now = await seed(database)
    async with database.transaction() as session:
        if blocker == "attempt":
            (await session.get(AttemptRow, "attempt-old")).state = AttemptState.RUNNING
        elif blocker == "finalization":
            (await session.get(LeaseRow, "lease-old")).state = LeaseState.RELEASING
        elif blocker == "claim":
            claim = await session.get(
                LeaseGPUClaimRow, {"lease_id": "lease-old", "site_id": "lab1", "gpu_uuid": "gpu-0"}
            )
            claim.active = True
        elif blocker == "kernel":
            session.add(
                KernelSessionRow(
                    id="kernel",
                    site_id="lab1",
                    state="requested",
                    mode=KernelMode.RAW_MODEL,
                    request={},
                )
            )
        else:
            session.add(
                InputStageRow(
                    job_id="job-old",
                    site_id="lab1",
                    stage_id="stage",
                    state="downloading",
                    expires_at=now + timedelta(minutes=5),
                )
            )
        await session.flush()
        with pytest.raises(ConflictError, match=r"active|claim|staging"):
            await repo.retire_site(session, "lab1", now=now)
        assert (await session.get(SiteRow, "lab1")).state is SiteState.READY


@pytest.mark.asyncio
async def test_offline_unretired_sites_remain_visible(database):
    repo, _ = await seed(database)
    async with database.transaction() as session:
        (await session.get(SiteRow, "lab1")).state = SiteState.OFFLINE
    async with database.session() as session:
        assert [s.id for s in await repo.list_sites(session)] == ["lab1"]
        assert [s["site_id"] for s in (await system_status_snapshot(session))["sites"]] == ["lab1"]


@pytest.mark.asyncio
async def test_stale_placement_cannot_offer_retired_site(database):
    from topodrome.protocol.fencing import FencingAuthority

    repo, now = await seed(database)
    async with database.transaction() as session:
        job = await repo.submit_job(session, job_spec())
        job.state = JobState.ASSIGNING
        await repo.retire_site(session, "lab1", now=now)
        with pytest.raises(ConflictError, match="retired"):
            await repo.create_offer(
                session,
                job=job,
                site_id="lab1",
                gpu_uuids=("gpu-0",),
                authority=FencingAuthority(b"f" * 32),
                offer_ttl=timedelta(seconds=30),
            )
