import asyncio
from dataclasses import replace
from datetime import timedelta

import pytest

from tests.unit.test_scheduler_repository import heartbeat, job_spec
from tests.unit.test_site_agent_preemption import _build_agent, _CooperativeBackend
from topodrome.db.repositories import SchedulerRepository
from topodrome.domain.enums import AttemptState, JobState, LeaseState
from topodrome.domain.errors import StaleFenceError
from topodrome.domain.specs import JobSpec, utc_now
from topodrome.protocol.fencing import FencingAuthority
from topodrome.scheduler.status import system_status_snapshot
from topodrome.site_agent.agent import ExecutionOutcome, PreparedRuntime

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["succeeded", "failed", "canceled", "expired"])
@pytest.mark.parametrize("degree", [1, 2])
async def test_finalization_lease_survives_gpu_reuse_without_owning_it(database, ending, degree):
    repository = SchedulerRepository()
    authority = FencingAuthority(b"f" * 32)
    now = utc_now()
    spec = (
        job_spec()
        if degree == 1
        else JobSpec(
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
    )
    inventory = heartbeat()
    inventory = inventory.model_copy(
        update={
            "gpus": tuple(
                inventory.gpus[0].model_copy(update={"uuid": f"gpu-{i}", "index": i})
                for i in range(degree)
            )
        }
    )
    gpu_uuids = tuple(g.uuid for g in inventory.gpus)
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, inventory)
        await repository.submit_job(session, spec)
        job = await repository.claim_next_job(session)
        offer = await repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=gpu_uuids,
            authority=authority,
            offer_ttl=timedelta(seconds=30),
            token_ttl=timedelta(hours=1),
            now=now,
        )
        await repository.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(seconds=90),
            now=now,
        )
        for state in (AttemptState.STARTING, AttemptState.RUNNING, AttemptState.COMMITTING):
            await repository.update_attempt(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=authority,
                target=state,
                gpu_released=state is AttemptState.COMMITTING,
                now=now,
            )
    async with database.transaction() as session:
        snapshot = await system_status_snapshot(session)
        assert snapshot["jobs"][0]["state"] == "finalizing"
        assert snapshot["jobs"][0]["gpu_count"] == 0
        assert not snapshot["gpus"][0]["active_lease"]
        await repository.submit_job(session, spec)
        next_job = await repository.claim_next_job(session)
        second = await repository.create_offer(
            session,
            job=next_job,
            site_id="lab1",
            gpu_uuids=gpu_uuids,
            authority=authority,
            offer_ttl=timedelta(seconds=300),
            token_ttl=timedelta(hours=1),
            now=now,
        )
    async with database.transaction() as session:
        grant = await repository.renew_lease(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(seconds=90),
            token_ttl=timedelta(hours=1),
            renewal_id="finalizer-renewal",
            renewal_issued_at=now + timedelta(seconds=5),
            now=now + timedelta(seconds=5),
        )
        attempt = await repository.authenticate_active_attempt_message(
            session,
            job_id=offer.job_id,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=grant.fencing_token,
            authority=authority,
            now=now + timedelta(seconds=5),
        )
        assert attempt.lease.state is LeaseState.RELEASING
        assert not any(c.active for c in attempt.lease.claims)
    with pytest.raises(StaleFenceError):
        async with database.transaction() as session:
            await repository.authenticate_active_attempt_message(
                session,
                job_id=offer.job_id,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=authority,
                now=now + timedelta(seconds=5),
            )
    async with database.transaction() as session:
        if ending == "expired":
            assert (
                await repository.reap_expired_leases(session, now=now + timedelta(seconds=96)) == 1
            )
        else:
            await repository.update_attempt(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=grant.fencing_token,
                authority=authority,
                target=AttemptState(ending),
                now=now + timedelta(seconds=6),
            )
    async with database.session() as session:
        second_job = await repository.get_job(session, second.job_id)
        assert second_job.state is JobState.ASSIGNING
        assert second_job.attempts[0].lease.claims[0].active


class SplitBackend(_CooperativeBackend):
    def __init__(self, fail=False):
        super().__init__("")
        self.uploading = asyncio.Event()
        self.finish_upload = asyncio.Event()
        self.second_running = asyncio.Event()
        self.finish_second = asyncio.Event()
        self.fail = fail

    async def prepare(self, assignment, workspace):
        return PreparedRuntime(
            "actor-1",
            assignment.model_key,
            assignment.attempt_id == "attempt-2",
            process_ids=(901,),
        )

    async def execute_with_finalization(self, prepared, assignment, workspace, callback):
        if assignment.attempt_id == "attempt-1":
            assert await callback(True)
            self.uploading.set()
            await self.finish_upload.wait()
            if self.fail:
                raise RuntimeError("upload failed")
        else:
            self.second_running.set()
            await self.finish_second.wait()
            await callback(True)
        return ExecutionOutcome(True, keep_warm=True, artifacts_verified=True)

    def cancel_finalization(self, attempt_id):
        assert attempt_id == "attempt-1"
        self.finish_upload.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["success", "failure", "cancel"])
async def test_old_finalizer_cannot_touch_reused_actor(tmp_path, ending):
    backend = SplitBackend(fail=ending == "failure")
    agent, first, outbound, fences, ownership = _build_agent(tmp_path, backend)
    try:
        assert await agent.offer(first)
        first_task = agent._active[first.attempt_id].task
        await asyncio.wait_for(backend.uploading.wait(), 2)
        assert fences.active_claims() == {}
        activity = await agent._activity_payload()
        assert activity["artifact_flush_pending"] and not activity["job_running"]
        assert not activity["assignment_queued_locally"]
        second = replace(
            first,
            job_id="job-2",
            attempt_id="attempt-2",
            lease=replace(
                first.lease,
                lease_id="lease-2",
                attempt_id="attempt-2",
                fence_epochs={"GPU-A": 2},
                reuse_actor_id="actor-1",
            ),
        )
        assert await agent.offer(second)
        await asyncio.wait_for(backend.second_running.wait(), 2)
        # Pressure/reservation affecting the former GPU does not preempt uploads.
        await agent.request_preemption(first.attempt_id, explicit=True)
        assert not backend.requested.is_set()
        if ending == "cancel":
            await agent.cancel(first.attempt_id)
        else:
            backend.finish_upload.set()
        await asyncio.wait_for(first_task, 2)
        assert fences.active_claims() == {"GPU-A": "lease-2"}
        assert fences.actor_allocation("GPU-A").state == "busy"
        assert ownership.exact_owner(901).lease_id == "lease-2"
        assert not backend.evicted and not backend.forced
        states = [
            p["state"]
            for kind, p in outbound.events
            if kind == "attempt_event" and p["attempt_id"] == first.attempt_id
        ]
        assert (
            states[-1]
            == {"success": "succeeded", "failure": "failed", "cancel": "canceled"}[ending]
        )
    finally:
        backend.finish_upload.set()
        backend.finish_second.set()
        await asyncio.wait_for(agent.wait_for_idle(), 3)
        fences.close()


@pytest.mark.asyncio
async def test_backpressure_and_shutdown_wait_for_finalizers(tmp_path):
    backend = SplitBackend()
    agent, first, outbound, fences, _ = _build_agent(tmp_path, backend)
    agent.finalization_backlog_limit = 1
    try:
        assert await agent.offer(first)
        await asyncio.wait_for(backend.uploading.wait(), 2)
        second = replace(first, job_id="job-2", attempt_id="attempt-2")
        assert not await agent.offer(second)
        assert "finalization backlog" in str(outbound.events[-1])
        shutdown = asyncio.create_task(agent.shutdown())
        await asyncio.sleep(0)
        assert not shutdown.done()
        backend.finish_upload.set()
        await asyncio.wait_for(shutdown, 3)
    finally:
        backend.finish_upload.set()
        await asyncio.wait_for(agent.wait_for_idle(), 3)
        fences.close()
