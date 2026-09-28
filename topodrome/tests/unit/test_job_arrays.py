from datetime import timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from tests.unit.test_scheduler_dispatch import _heartbeat, _source_ref
from topodrome.config import SchedulerSettings
from topodrome.db.job_ordering import queue_cursor
from topodrome.db.models import AttemptRow, JobArrayRow, JobRow, LeaseGPUClaimRow, LeaseRow
from topodrome.db.repositories import SchedulerRepository
from topodrome.domain.arrays import ArrayUpdate, JobArraySpec
from topodrome.domain.enums import AttemptState, JobState, LeaseState, SiteKind
from topodrome.domain.errors import ConflictError, NotFoundError
from topodrome.domain.specs import JobSpec, utc_now
from topodrome.scheduler.arrays import array_detail, cancel_array, update_array
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
from topodrome.scheduler.service import SchedulerService
from topodrome.scheduler.timing import job_timing, set_phase


@pytest.fixture
async def array_service(database, tmp_path, pre_staged_inputs):
    connections = AgentConnectionRegistry()
    messages = []

    async def send(message):
        messages.append(message)

    connection = AgentConnection("lab1", "lab1-session", send)
    await connections.register(connection)
    repo = SchedulerRepository()
    heartbeat = _heartbeat("lab1", SiteKind.LAB)
    heartbeat = heartbeat.model_copy(
        update={
            "gpus": tuple(
                heartbeat.gpus[0].model_copy(update={"uuid": f"gpu-{i}", "index": i})
                for i in range(4)
            )
        }
    )
    async with database.transaction() as session:
        await repo.upsert_heartbeat(session, heartbeat, connection_generation=connection.generation)
    service = SchedulerService(
        database=database,
        repository=repo,
        connections=connections,
        settings=SchedulerSettings(global_root=tmp_path, fence_signing_key="f" * 64),
    )
    spec = JobSpec(
        entrypoint="experiment.py",
        source_ref=_source_ref(service),
        model={"alias": "model", "revision": "rev"},
        resources={"gpu": 1, "min_vram_gb": 15},
        name="sweep",
    )
    return service, spec, messages


def request(spec, *, count=5, parallel=2):
    return JobArraySpec(
        job=spec,
        elements=[{"index": i, "kwargs": {"seed": i}} for i in range(count)],
        max_parallel=parallel,
    )


async def test_atomic_array_owner_idempotency_and_pagination(array_service, database):
    service, spec, _ = array_service
    submitted = await service.submit_array(
        request(spec), owner_username="alice", idempotency_key="a"
    )
    repeated = await service.submit_array(
        request(spec), owner_username="alice", idempotency_key="a"
    )
    assert repeated["array_id"] == submitted["array_id"]
    assert repeated["total"] == 5
    with pytest.raises(ConflictError):
        await service.submit_array(
            request(spec, parallel=3), owner_username="alice", idempotency_key="a"
        )
    other = await service.submit_array(request(spec), owner_username="bob", idempotency_key="a")
    assert other["array_id"] != submitted["array_id"]
    with pytest.raises(NotFoundError):
        await array_detail(service, submitted["array_id"], owner_username="bob")
    with pytest.raises(NotFoundError):
        await update_array(
            service, submitted["array_id"], ArrayUpdate(held=True), owner_username="bob"
        )
    with pytest.raises(NotFoundError):
        await cancel_array(service, submitted["array_id"], owner_username="bob")
    first = await array_detail(service, submitted["array_id"], owner_username="alice", limit=2)
    second = await array_detail(service, submitted["array_id"], after=first["next_after"], limit=10)
    assert [m["index"] for m in first["members"] + second["members"]] == list(range(5))
    async with database.session() as session:
        jobs = list(
            await session.scalars(select(JobRow).where(JobRow.array_id == submitted["array_id"]))
        )
        assert {j.spec["source_ref"] for j in jobs} == {spec.source_ref}
        assert all(j.spec["kwargs"]["array_index"] == j.array_index for j in jobs)
        assert all(j.owner_username == "alice" for j in jobs)


async def test_array_rollback_on_database_failure(array_service, database, monkeypatch):
    service, spec, _ = array_service
    original = service.repository.submit_job
    counter = 0

    async def fail(*args, **kwargs):
        nonlocal counter
        counter += 1
        if counter == 3:
            raise RuntimeError("simulated failed transaction")
        return await original(*args, **kwargs)

    monkeypatch.setattr(service.repository, "submit_job", fail)
    with pytest.raises(RuntimeError):
        await service.submit_array(request(spec))
    async with database.session() as session:
        assert await session.scalar(select(func.count()).select_from(JobRow)) == 0
        assert await session.scalar(select(func.count()).select_from(JobArrayRow)) == 0


async def test_array_limit_hold_reduction_and_finalizer_release(array_service, database):
    service, spec, messages = array_service
    result = await service.submit_array(request(spec))
    array_id = result["array_id"]
    assert await service.dispatch_once()
    assert await service.dispatch_once()
    assert not await service.dispatch_once()
    assert (await array_detail(service, array_id))["active_compute"] == 2
    await update_array(service, array_id, ArrayUpdate(max_parallel=1))
    assert not await service.dispatch_once()
    assert len(messages) == 2  # No preemption on ceiling reduction.
    await update_array(service, array_id, ArrayUpdate(max_parallel=4, held=True))
    assert not await service.dispatch_once()
    assert len(messages) == 2
    await update_array(service, array_id, ArrayUpdate(max_parallel=2, held=False))
    async with database.transaction() as session:
        attempt = await session.get(AttemptRow, messages[0]["payload"]["attempt_id"])
        lease = await session.scalar(select(LeaseRow).where(LeaseRow.attempt_id == attempt.id))
        lease.state = LeaseState.RELEASING
        attempt.state = AttemptState.COMMITTING
        for claim in await session.scalars(
            select(LeaseGPUClaimRow).where(LeaseGPUClaimRow.lease_id == lease.id)
        ):
            claim.active = False
    assert await service.dispatch_once()
    assert len(messages) == 3
    assert (await array_detail(service, array_id))["active_compute"] == 2


async def test_cancel_array_persists_and_preserves_success(array_service, database):
    service, spec, _ = array_service
    result = await service.submit_array(request(spec, count=3))
    detail = await array_detail(service, result["array_id"])
    successful = detail["members"][0]["job_id"]
    async with database.transaction() as session:
        job = await session.get(JobRow, successful)
        job.state = JobState.SUCCEEDED
    await cancel_array(service, result["array_id"])
    await cancel_array(service, result["array_id"])
    final = await array_detail(service, result["array_id"])
    assert final["counts"] == {"succeeded": 1, "canceled": 2}
    assert final["terminal"] == 3
    assert not await service.dispatch_once()
    with pytest.raises(ConflictError):
        await update_array(service, result["array_id"], ArrayUpdate(held=False))


async def test_cancellation_recovers_after_scheduler_restart(array_service, database):
    from topodrome.scheduler.arrays import reconcile_cancellations

    service, spec, _ = array_service
    result = await service.submit_array(request(spec, count=3))
    async with database.transaction() as session:
        parent = await session.get(JobArrayRow, result["array_id"])
        parent.cancel_requested = True  # Crash after durable intent, before child cancellation.
    restarted = SchedulerService(database=database, settings=service.settings)
    await reconcile_cancellations(restarted)
    assert (await array_detail(restarted, result["array_id"]))["counts"] == {"canceled": 3}


async def test_replay_after_hold_does_not_reset_array_or_need_source(array_service, monkeypatch):
    service, spec, _ = array_service
    result = await service.submit_array(request(spec), idempotency_key="replay")
    await update_array(service, result["array_id"], ArrayUpdate(held=True))

    async def unavailable(*args, **kwargs):
        raise AssertionError("accepted replay must not re-resolve expired input")

    monkeypatch.setattr(service, "prepare_submission", unavailable)
    repeated = await service.submit_array(request(spec), idempotency_key="replay")
    assert repeated["array_id"] == result["array_id"] and repeated["held"]


async def test_aged_jobs_override_new_priority_with_stable_pagination(database):
    repo = SchedulerRepository()
    now = utc_now()
    async with database.transaction() as session:
        for name, priority, minutes, runtime in [
            ("old", -1000, 60, None),
            ("young", 1000, 0, 1),
            ("old-short", 5, 40, 1),
            ("normal", 0, 0, 1),
        ]:
            job = await repo.submit_job(
                session,
                JobSpec(
                    entrypoint="run.py",
                    model="model",
                    name=name,
                    priority=priority,
                    estimated_runtime_seconds=runtime,
                ),
            )
            job.eligible_since = now - timedelta(minutes=minutes)
    async with database.transaction() as session:
        names, cursor = [], None
        while True:
            rows = await repo.lock_job_candidates(session, limit=1, now=now, after=cursor)
            if not rows:
                break
            names.extend(j.spec["name"] for j in rows)
            cursor = queue_cursor(rows[-1], now)
    assert names == ["old", "old-short", "young", "normal"]


def test_nonoverlapping_timing_totals_and_terminal_freeze():
    now = utc_now()
    job = SimpleNamespace(
        state=JobState.QUEUED,
        created_at=now,
        updated_at=now,
        phase="queued",
        phase_started_at=now,
        phase_seconds={},
    )
    set_phase(job, "inputs", now=now + timedelta(seconds=10))
    set_phase(job, "compute", now=now + timedelta(seconds=30))
    set_phase(job, "finalizing", now=now + timedelta(seconds=80))
    job.state = JobState.SUCCEEDED
    job.updated_at = now + timedelta(seconds=100)
    set_phase(job, "succeeded", now=job.updated_at)
    result = job_timing(job, now=now + timedelta(hours=1))
    assert result["seconds"] == {"queued": 10, "inputs": 20, "compute": 50, "finalizing": 20}
    assert result["total_seconds"] == 100
    assert result["complete"]


def test_array_validation_and_index_parsing():
    from topodrome.client.arrays import parse_indices

    assert [v.index for v in parse_indices("0-4:2,9")] == [0, 2, 4, 9]
    for invalid in ["1,1", "0-10000000", "4-0", "0-9:0", "-1", "$(whoami)"]:
        with pytest.raises(ValueError):
            parse_indices(invalid)
    spec = JobSpec(entrypoint="run.py", model="model")
    with pytest.raises(ValueError):
        JobArraySpec(job=spec, elements=[{"index": 0}, {"index": 0}])
    with pytest.raises(ValueError):
        JobArraySpec(job=spec, elements=[{"index": 0, "kwargs": {"array_index": 2}}])
    with pytest.raises(ValueError):
        JobArraySpec(job=spec, elements=[{"index": 0, "kwargs": {"api_token": "secret"}}])


async def test_aged_gang_stops_backfill_without_preemption(array_service, database):
    service, spec, messages = array_service
    for _ in range(3):
        await service.submit(spec)
        assert await service.dispatch_once()
    # One slot is free, but the aged two-GPU job cannot yet acquire a gang.
    raw = spec.model_dump()
    raw["resources"]["gpu"] = 2
    raw["model"]["parallelism"] = {
        "mode": "model_sharded",
        "degree": 2,
        "engine": "transformers_device_map",
        "max_memory_gb_per_gpu": 15,
    }
    big = await service.submit(JobSpec.model_validate(raw))
    async with database.transaction() as session:
        job = await session.get(JobRow, big.job_id)
        job.eligible_since = utc_now() - timedelta(hours=1)
    small = await service.submit(spec)
    assert not await service.dispatch_once()
    assert len(messages) == 3
    assert (await service.status(big.job_id)).blocked_reason == "waiting_for_aged_gang"
    assert (await service.status(small.job_id)).state is JobState.QUEUED
    # This is only admission protection. It never sends an eviction command.
    assert all(m["kind"] == "offer_assignment" for m in messages)
    async with database.transaction() as session:
        for claim in await session.scalars(select(LeaseGPUClaimRow)):
            claim.active = False
        for lease in await session.scalars(select(LeaseRow)):
            lease.state = LeaseState.RELEASED
    assert await service.dispatch_once()
    assert messages[-1]["payload"]["job_id"] == big.job_id
    assert len(messages[-1]["payload"]["claims"]) == 2


async def test_array_summary_is_visible_in_snapshot(array_service, database):
    from topodrome.scheduler.status import system_status_snapshot

    service, spec, _ = array_service
    result = await service.submit_array(request(spec), owner_username="alice")
    async with database.session() as session:
        own = await system_status_snapshot(session, owner_username="alice")
        other = await system_status_snapshot(session, owner_username="bob")
    assert own["arrays"][0]["array_id"] == result["array_id"]
    assert own["arrays"][0]["counts"] == {"queued": 5}
    assert other["arrays"] == []
    assert other["jobs"] == []


def test_array_notification_only_observes_terminal_failures():
    from topodrome.client.codex_bridge import array_notification_summary

    result = {"state": "running", "counts": {"retry_wait": 1, "succeeded": 1}}
    client = SimpleNamespace(array=lambda *_args, **_kwargs: result)
    assert array_notification_summary(client, "array_a").state.value == "pending"
    result["counts"] = {"failed": 1, "running": 1}
    assert array_notification_summary(client, "array_a").state.value == "failed"
    result["counts"] = {"succeeded": 2}
    result["state"] = "succeeded"
    assert array_notification_summary(client, "array_a").state.value == "succeeded"
