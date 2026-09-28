from datetime import timedelta

import pytest

from topodrome.db.repositories import SchedulerRepository
from topodrome.domain.enums import JobState
from topodrome.domain.specs import JobSpec, utc_now
from topodrome.scheduler.status import system_status_snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("page_size", [1, 2, 3, 20])
async def test_shortest_first_paging_priority_ties_unknown_and_retry(
    database, page_size, monkeypatch
):
    repo = SchedulerRepository()
    counter = iter(range(1000))
    monkeypatch.setattr(
        "topodrome.db.repositories.new_id", lambda prefix: f"{prefix}_{next(counter):04}"
    )
    now = utc_now()
    specs = [
        ("unknown-first", 0, None),
        ("long", 0, 600),
        ("short", 0, 10),
        ("high-unknown", 1, None),
        ("short-tie", 0, 10),
        ("unknown-last", 0, None),
        ("low-fast", -1, 1),
        ("retry", 0, 5),
        ("backoff", 0, 0.1),
        ("canceled", 0, 0.1),
        ("active", 0, 0.1),
    ]
    async with database.transaction() as session:
        for name, priority, estimate in specs:
            row = await repo.submit_job(
                session,
                JobSpec(
                    entrypoint="run.py",
                    model="llama",
                    name=name,
                    priority=priority,
                    estimated_runtime_seconds=estimate,
                ),
            )
            # Equal timestamps exercise ID tie-breaking across page boundaries.
            row.created_at = now
            if estimate is None and name == "unknown-first":
                row.spec = {k: v for k, v in row.spec.items() if k != "estimated_runtime_seconds"}
            if name in {"retry", "backoff"}:
                row.state = JobState.RETRY_WAIT
                row.retry_not_before = now + timedelta(seconds=60 if name == "backoff" else -60)
            if name == "canceled":
                row.cancel_requested = True
            if name == "active":
                row.state = JobState.RUNNING
    names, cursor = [], None
    async with database.transaction() as session:
        for _ in range(20):
            rows = await repo.lock_job_candidates(session, limit=page_size, now=now, after=cursor)
            if not rows:
                break
            names.extend(row.spec["name"] for row in rows)
            last = rows[-1]
            cursor = (
                last.priority,
                last.spec.get("estimated_runtime_seconds"),
                last.created_at,
                last.id,
            )
        else:
            pytest.fail("candidate pagination did not terminate")
    assert names == [
        "high-unknown",
        "retry",
        "short",
        "short-tie",
        "long",
        "unknown-first",
        "unknown-last",
        "low-fast",
    ]


@pytest.mark.asyncio
async def test_dashboard_queue_uses_same_shortest_first_order(database):
    repo = SchedulerRepository()
    async with database.transaction() as session:
        for name, estimate in [("unknown", None), ("long", 300), ("short", 10)]:
            await repo.submit_job(
                session,
                JobSpec(
                    entrypoint="run.py",
                    model="llama",
                    name=name,
                    estimated_runtime_seconds=estimate,
                ),
            )
    async with database.session() as session:
        snapshot = await system_status_snapshot(session)
    assert [j["name"] for j in snapshot["jobs"]] == ["short", "long", "unknown"]
    assert snapshot["jobs"][0]["estimated_runtime_seconds"] == 10


@pytest.mark.asyncio
async def test_dispatch_selects_shorter_job_and_keeps_hint_off_worker_wire(
    database, tmp_path, pre_staged_inputs
):
    from tests.unit.test_scheduler_dispatch import _heartbeat, _source_ref
    from topodrome.config import SchedulerSettings
    from topodrome.domain.enums import SiteKind
    from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
    from topodrome.scheduler.service import SchedulerService

    repo = SchedulerRepository()
    connections = AgentConnectionRegistry()
    messages = []

    async def send(message):
        messages.append(message)

    connection = AgentConnection("lab1", "lab1-session", send)
    await connections.register(connection)
    async with database.transaction() as session:
        await repo.upsert_heartbeat(
            session, _heartbeat("lab1", SiteKind.LAB), connection_generation=connection.generation
        )
    service = SchedulerService(
        database=database,
        connections=connections,
        repository=repo,
        settings=SchedulerSettings(global_root=tmp_path, fence_signing_key="f" * 64),
    )
    source = _source_ref(service)
    jobs = []
    for runtime in (600, 10):
        jobs.append(
            await service.submit(
                JobSpec(
                    entrypoint="experiment.py",
                    source_ref=source,
                    model={"alias": "model", "revision": "rev"},
                    resources={"gpu": 1, "min_vram_gb": 15},
                    estimated_runtime_seconds=runtime,
                )
            )
        )
    assert await service.dispatch_once()
    assert messages[0]["payload"]["job_id"] == jobs[1].job_id
    assert "estimated_runtime_seconds" not in messages[0]["payload"]["spec"]
    assert (await service.status(jobs[0].job_id)).state is JobState.QUEUED
