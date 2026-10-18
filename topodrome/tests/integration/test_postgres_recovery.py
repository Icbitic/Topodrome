from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, func, inspect, select, text
from sqlalchemy.ext.asyncio import AsyncSession

pgserver = pytest.importorskip(
    "pgserver",
    reason="install the postgres dependency group with `uv sync --group postgres`",
)

from tests.schema_assertions import assert_schema_matches_models  # noqa: E402
from topodrome.db.models import (  # noqa: E402
    AttemptRow,
    Base,
    GPUFenceCounterRow,
    GPUReservationRow,
    JobArrayRow,
    JobRow,
    JobStateEventRow,
    LeaseGPUClaimRow,
    SiteRow,
)
from topodrome.db.repositories import SchedulerRepository  # noqa: E402
from topodrome.db.session import CURRENT_SCHEMA_REVISION, Database  # noqa: E402
from topodrome.domain.enums import (  # noqa: E402
    AttemptState,
    JobState,
    LeaseState,
    ModelAvailability,
    SiteKind,
)
from topodrome.domain.errors import ConflictError  # noqa: E402
from topodrome.domain.specs import (  # noqa: E402
    GPUInventory,
    GPUReservationRequest,
    JobSpec,
    ModelCapability,
    SiteHeartbeat,
    new_id,
    utc_now,
)
from topodrome.protocol.fencing import FencingAuthority  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3


@pytest.mark.postgres
@pytest.mark.parametrize("operation", ["offer", "reservation"])
async def test_heartbeat_and_inventory_claim_do_not_deadlock(
    migrated_postgres_dsn: str, operation: str,
) -> None:
    """Reproduce the actual parent-site/GPU-child lock cycle on isolated PG."""
    database = Database(migrated_postgres_dsn)
    repository = SchedulerRepository()
    site_id = "heartbeat-claim-" + operation
    sample = _heartbeat().model_copy(update={"site_id": site_id})
    job_id = None
    task = None
    worker_started = asyncio.Event()
    worker_pid = None

    async def claim():
        nonlocal worker_pid
        async with database.transaction() as session:
            worker_pid = await session.scalar(text("SELECT pg_backend_pid()"))
            worker_started.set()
            if operation == "reservation":
                row = await repository.create_gpu_reservation(
                    session, GPUReservationRequest(
                        site_id=site_id, gpu_uuid="gpu-0", duration_seconds=600,
                        reason="isolated PostgreSQL lock-order regression",
                    ),
                )
                return row.id
            job = await repository.get_job(session, job_id, lock=True)
            await repository.claim_locked_job(session, job)
            offer = await repository.create_offer(
                session, job=job, site_id=site_id, gpu_uuids=("gpu-0",),
                authority=FencingAuthority(b"postgres-lock-order-test-key-32b"),
                offer_ttl=timedelta(seconds=180),
            )
            return offer.attempt_id

    async def wait_for_parent_lock(parent_pid):
        # No timing guess: wait until PG proves the claim is blocked by the
        # heartbeat. On the broken implementation it has already locked GPUs
        # and is waiting for its site foreign-key check instead.
        async with database.session() as observer:
            while True:
                blockers = await observer.scalar(
                    text("SELECT pg_blocking_pids(:pid)"), {"pid": worker_pid},
                )
                if parent_pid in blockers:
                    return
                await asyncio.sleep(0.01)

    try:
        async with database.transaction() as session:
            await repository.upsert_heartbeat(session, sample)
            job = await repository.submit_job(session, _job_spec(priority=0))
            job_id = job.id
        async with database.transaction() as session:
            parent_pid = await session.scalar(text("SELECT pg_backend_pid()"))
            await session.get(SiteRow, site_id, with_for_update=True)
            task = asyncio.create_task(claim())
            await asyncio.wait_for(worker_started.wait(), timeout=10)
            await asyncio.wait_for(wait_for_parent_lock(parent_pid), timeout=10)
            # This used to deadlock while deleting GPUs. It must now complete
            # and commit before the blocked claim obtains site/GPU locks.
            assert await asyncio.wait_for(repository.upsert_heartbeat(
                session, sample.model_copy(update={"sequence": 2}),
            ), timeout=10)
        assert await asyncio.wait_for(task, timeout=10)
    finally:
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        async with database.transaction() as session:
            if job_id is not None:
                await session.execute(delete(JobRow).where(JobRow.id == job_id))
            await session.execute(delete(GPUReservationRow).where(
                GPUReservationRow.site_id == site_id,
            ))
            await session.execute(delete(SiteRow).where(SiteRow.id == site_id))
        await database.close()


def _asyncpg_dsn(uri: str) -> str:
    if not uri.startswith("postgresql://"):
        raise ValueError(f"unexpected pgserver URI: {uri}")
    return uri.replace("postgresql://", "postgresql+asyncpg://", 1)


def _run_alembic(dsn: str, operation: str, revision: str) -> None:
    """Run the production Alembic path without nesting an event loop."""
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    variable = "TOPODROME_DATABASE_DSN"
    previous = os.environ.get(variable)
    os.environ[variable] = dsn
    try:
        getattr(command, operation)(config, revision)
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def _upgrade(dsn: str, revision: str = "head") -> None:
    _run_alembic(dsn, "upgrade", revision)


def _stamp(dsn: str, revision: str) -> None:
    _run_alembic(dsn, "stamp", revision)


@pytest.fixture(scope="module")
def migrated_postgres_dsn(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A real, unprivileged PostgreSQL 16 process with an isolated data directory."""
    data_dir = tmp_path_factory.mktemp("topodrome-postgres") / "data"
    server = pgserver.get_server(data_dir, cleanup_mode="stop")
    dsn = _asyncpg_dsn(server.get_uri())
    try:
        _upgrade(dsn)
        yield dsn
    finally:
        server.cleanup()


def _job_spec(*, priority: int) -> JobSpec:
    return JobSpec(
        entrypoint="experiment.py",
        model={"repo_id": "test/model", "revision": "rev"},
        resources={"gpu": 1, "min_vram_gb": 15},
        priority=priority,
    )


def _heartbeat() -> SiteHeartbeat:
    return SiteHeartbeat(
        site_id="lab1",
        site_kind=SiteKind.LAB,
        agent_session_id="postgres-recovery-test",
        sequence=1,
        gpus=(
            GPUInventory(
                uuid="gpu-0",
                index=0,
                name="test-gpu",
                total_memory_bytes=80 * GIB,
                free_memory_bytes=70 * GIB,
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


class _ConcurrentSubmitRepository(SchedulerRepository):
    """Force two transactions past the absent-row idempotency precheck."""

    def __init__(self) -> None:
        self._arrived = 0
        self._release = asyncio.Event()
        self._first_reads: set[int] = set()

    async def _find_idempotent_job(
        self,
        session: AsyncSession,
        *,
        stored_idempotency_key: str,
    ) -> JobRow | None:
        row = await super()._find_idempotent_job(
            session,
            stored_idempotency_key=stored_idempotency_key,
        )
        task = asyncio.current_task()
        assert task is not None
        task_id = id(task)
        if row is None and task_id not in self._first_reads:
            self._first_reads.add(task_id)
            self._arrived += 1
            if self._arrived == 2:
                self._release.set()
            await asyncio.wait_for(self._release.wait(), timeout=5)
        return row


class _ConcurrentReservationRepository(SchedulerRepository):
    """Force two reservation requests past their absent-key precheck."""

    def __init__(self) -> None:
        self._arrived = 0
        self._release = asyncio.Event()

    async def expire_gpu_reservations(
        self,
        session: AsyncSession,
        *,
        now: datetime | None = None,
    ) -> int:
        result = await super().expire_gpu_reservations(
            session,
            now=now,
        )
        self._arrived += 1
        if self._arrived == 2:
            self._release.set()
        await asyncio.wait_for(self._release.wait(), timeout=5)
        return result


@pytest.mark.postgres
def test_schema_parity_repairs_a_falsely_stamped_postgres_database(
    tmp_path: Path,
) -> None:
    server = pgserver.get_server(tmp_path / "repair-postgres", cleanup_mode="stop")
    dsn = _asyncpg_dsn(server.get_uri())
    try:
        _upgrade(dsn, "0001_initial")
        _stamp(dsn, "0008_model_replicas")
        _upgrade(dsn)

        async def assert_repaired() -> str | None:
            database = Database(dsn)
            try:
                async with database.engine.connect() as connection:
                    await connection.run_sync(assert_schema_matches_models)
                    value = await connection.scalar(text("SELECT version_num FROM alembic_version"))
                    return None if value is None else str(value)
            finally:
                await database.close()

        revision = asyncio.run(assert_repaired())
        assert revision == CURRENT_SCHEMA_REVISION
    finally:
        server.cleanup()


@pytest.mark.postgres
def test_database_cleanup_guards_and_round_trip_on_postgres(tmp_path: Path) -> None:
    server = pgserver.get_server(tmp_path / "cleanup-data", cleanup_mode="stop")
    dsn = _asyncpg_dsn(server.get_uri())

    async def sql(*statements: str) -> list[object]:
        database = Database(dsn)
        try:
            async with database.transaction() as session:
                results = []
                for statement in statements:
                    result = await session.execute(text(statement))
                    results.append(result.all() if result.returns_rows else None)
                return results
        finally:
            await database.close()

    try:
        _upgrade(dsn, "0018_job_arrays")
        asyncio.run(sql(
            "INSERT INTO jobs (id, state, spec, priority, cancel_requested, "
            "created_at, updated_at) VALUES "
            "('cleanup-job', 'succeeded', '{}', 7, false, now(), now())",
            "INSERT INTO kernel_sessions (id, job_id, state, mode, request, "
            "access_generation, connect_url, created_at, last_activity_at) VALUES "
            "('cleanup-kernel', 'cleanup-job', 'stopped', 'service', '{}', 3, "
            "'https://example.invalid/kernel', now(), now())",
        ))
        with pytest.raises(RuntimeError, match="stored kernel URLs"):
            _upgrade(dsn)
        assert asyncio.run(sql(
            "SELECT version_num FROM alembic_version",
            "SELECT connect_url FROM kernel_sessions",
            "SELECT count(*) FROM log_indexes",
        )) == [[("0018_job_arrays",)], [("https://example.invalid/kernel",)], [(0,)]]

        asyncio.run(sql("UPDATE kernel_sessions SET connect_url = NULL"))
        queries = (
            "SELECT id, state, priority, spec FROM jobs",
            "SELECT id, job_id, state, access_generation FROM kernel_sessions",
        )
        before = asyncio.run(sql(*queries))
        _upgrade(dsn, "0019_remove_db_legacies")
        assert asyncio.run(sql(*queries)) == before
        _run_alembic(dsn, "downgrade", "0018_job_arrays")
        assert asyncio.run(sql(
            "SELECT connect_url FROM kernel_sessions",
            "SELECT count(*) FROM log_indexes",
        )) == [[(None,)], [(0,)]]
        _upgrade(dsn)
        assert asyncio.run(sql(*queries)) == before

        async def assert_schema() -> None:
            database = Database(dsn)
            try:
                async with database.engine.connect() as connection:
                    await connection.run_sync(assert_schema_matches_models)
            finally:
                await database.close()

        asyncio.run(assert_schema())
    finally:
        server.cleanup()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_concurrent_idempotent_creates_are_race_safe_on_postgres(
    migrated_postgres_dsn: str,
) -> None:
    database = Database(migrated_postgres_dsn)
    reservation_site_id = "postgres-race-reservation-site"
    try:
        same_repository = _ConcurrentSubmitRepository()
        same_spec = _job_spec(priority=31)

        async def submit_same() -> str:
            async with database.transaction() as session:
                row = await same_repository.submit_job(
                    session,
                    same_spec,
                    idempotency_key="postgres-concurrent-identical-job",
                    owner_username="postgres-race-same",
                )
                return row.id

        same_ids = await asyncio.gather(submit_same(), submit_same())
        assert same_ids[0] == same_ids[1]
        async with database.session() as session:
            same_rows = list(
                await session.scalars(
                    select(JobRow).where(JobRow.owner_username == "postgres-race-same")
                )
            )
            assert [row.id for row in same_rows] == [same_ids[0]]
            assert (
                await session.scalar(
                    select(func.count(JobStateEventRow.id)).where(
                        JobStateEventRow.job_id == same_ids[0]
                    )
                )
                == 1
            )

        different_repository = _ConcurrentSubmitRepository()

        async def submit_different(priority: int) -> tuple[str, str | None]:
            try:
                async with database.transaction() as session:
                    row = await different_repository.submit_job(
                        session,
                        _job_spec(priority=priority),
                        idempotency_key="postgres-concurrent-conflicting-job",
                        owner_username="postgres-race-different",
                    )
                    return "created", row.id
            except ConflictError:
                return "conflict", None

        different_results = await asyncio.gather(submit_different(41), submit_different(42))
        assert sorted(state for state, _ in different_results) == ["conflict", "created"]
        async with database.session() as session:
            assert (
                await session.scalar(
                    select(func.count(JobRow.id)).where(
                        JobRow.owner_username == "postgres-race-different"
                    )
                )
                == 1
            )

        # A shared idempotency key on different physical targets is likewise
        # resolved as one semantic reservation conflict, never an unhandled
        # unique-constraint failure.
        setup_repository = SchedulerRepository()
        two_gpu_heartbeat = _heartbeat().model_copy(
            update={
                "site_id": reservation_site_id,
                "sequence": 20,
                "gpus": (
                    _heartbeat().gpus[0],
                    _heartbeat().gpus[0].model_copy(update={"uuid": "gpu-1", "index": 1}),
                ),
            }
        )
        async with database.transaction() as session:
            await setup_repository.upsert_heartbeat(session, two_gpu_heartbeat)
        reservation_repository = _ConcurrentReservationRepository()

        async def reserve(gpu_uuid: str) -> tuple[str, str | None]:
            try:
                async with database.transaction() as session:
                    row = await reservation_repository.create_gpu_reservation(
                        session,
                        GPUReservationRequest(
                            site_id=reservation_site_id,
                            gpu_uuid=gpu_uuid,
                            duration_seconds=600,
                            reason="concurrent reservation race",
                        ),
                        idempotency_key="postgres-concurrent-reservation",
                    )
                    return "created", row.id
            except ConflictError:
                return "conflict", None

        reservation_results = await asyncio.gather(reserve("gpu-0"), reserve("gpu-1"))
        assert sorted(state for state, _ in reservation_results) == ["conflict", "created"]
        async with database.session() as session:
            assert (
                await session.scalar(
                    select(func.count(GPUReservationRow.id)).where(
                        GPUReservationRow.idempotency_key == "postgres-concurrent-reservation"
                    )
                )
                == 1
            )
    finally:
        async with database.transaction() as session:
            await session.execute(
                delete(JobRow).where(
                    JobRow.owner_username.in_(["postgres-race-same", "postgres-race-different"])
                )
            )
            await session.execute(
                delete(GPUReservationRow).where(
                    GPUReservationRow.idempotency_key == "postgres-concurrent-reservation"
                )
            )
            await session.execute(delete(SiteRow).where(SiteRow.id == reservation_site_id))
        await database.close()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_migrations_and_scheduler_recovery_use_real_postgres(
    migrated_postgres_dsn: str,
) -> None:
    """Durable queue, lease, and fence state survives fresh scheduler DB pools."""
    signing_key = b"postgres-recovery-signing-key-32b"
    first_repository = SchedulerRepository()
    first_authority = FencingAuthority(signing_key)
    start = utc_now()

    first_database = Database(migrated_postgres_dsn)
    try:
        async with first_database.engine.connect() as connection:
            assert connection.dialect.name == "postgresql"
            tables = await connection.run_sync(
                lambda sync_connection: set(inspect(sync_connection).get_table_names())
            )
            assert set(Base.metadata.tables).issubset(tables)
            await connection.run_sync(assert_schema_matches_models)
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
        assert revision == CURRENT_SCHEMA_REVISION

        async with first_database.transaction() as session:
            assert await first_repository.upsert_heartbeat(session, _heartbeat())
            leased_job = await first_repository.submit_job(
                session,
                _job_spec(priority=10),
                idempotency_key="postgres-leased-job",
            )
            queued_job = await first_repository.submit_job(
                session,
                _job_spec(priority=0),
                idempotency_key="postgres-queued-job",
            )
            leased_job_id = leased_job.id
            queued_job_id = queued_job.id

        async with first_database.transaction() as session:
            claimed = await first_repository.claim_next_job(session)
            assert claimed is not None and claimed.id == leased_job_id
            first_offer = await first_repository.create_offer(
                session,
                job=claimed,
                site_id="lab1",
                gpu_uuids=("gpu-0",),
                authority=first_authority,
                offer_ttl=timedelta(seconds=30),
                token_ttl=timedelta(hours=1),
                now=start,
            )

        async with first_database.transaction() as session:
            await first_repository.accept_offer(
                session,
                attempt_id=first_offer.attempt_id,
                site_id="lab1",
                token=first_offer.fencing_token,
                authority=first_authority,
                lease_ttl=timedelta(seconds=10),
                now=start,
            )
    finally:
        await first_database.close()

    # A fresh Database and repository stand in for a restarted scheduler process.
    second_repository = SchedulerRepository()
    second_authority = FencingAuthority(signing_key)
    second_database = Database(migrated_postgres_dsn)
    try:
        assert await second_database.ready()
        async with second_database.session() as session:
            recovered_queued = await second_repository.get_job(session, queued_job_id)
            recovered_running = await second_repository.get_job(session, leased_job_id)
            counter = await session.get(
                GPUFenceCounterRow,
                {"site_id": "lab1", "gpu_uuid": "gpu-0"},
            )
            assert recovered_queued.state is JobState.QUEUED
            assert recovered_running.state is JobState.RUNNING
            assert recovered_running.attempts[0].state is AttemptState.ACCEPTED
            assert recovered_running.attempts[0].lease is not None
            assert recovered_running.attempts[0].lease.state is LeaseState.ACCEPTED
            assert recovered_running.attempts[0].lease.claims[0].active
            assert counter is not None and counter.epoch == 1

        async with second_database.transaction() as session:
            reaped = await second_repository.reap_expired_leases(
                session,
                now=start + timedelta(seconds=11),
            )
            assert reaped == 1

        async with second_database.transaction() as session:
            recovered = await second_repository.get_job(session, leased_job_id)
            assert recovered.state is JobState.RETRY_WAIT
            assert recovered.retry_not_before is not None
            assert recovered.attempts[0].state is AttemptState.LOST
            assert recovered.attempts[0].lease is not None
            assert recovered.attempts[0].lease.state is LeaseState.LOST
            assert not recovered.attempts[0].lease.claims[0].active

            claimed = await second_repository.claim_next_job(
                session,
                now=start + timedelta(seconds=12),
            )
            assert claimed is not None and claimed.id == leased_job_id
            retry_offer = await second_repository.create_offer(
                session,
                job=claimed,
                site_id="lab1",
                gpu_uuids=("gpu-0",),
                authority=second_authority,
                offer_ttl=timedelta(seconds=10),
                token_ttl=timedelta(hours=1),
                now=start + timedelta(seconds=12),
            )
            assert retry_offer.job_id == leased_job_id
            assert retry_offer.attempt_id != first_offer.attempt_id
            assert retry_offer.attempt_number == 2
            assert retry_offer.claims[0].epoch == 2
    finally:
        await second_database.close()

    # Restart once more: the new epoch and retry offer must also be durable. Reaping
    # the expired offer must free its claim without rolling the fence epoch back.
    third_repository = SchedulerRepository()
    third_database = Database(migrated_postgres_dsn)
    try:
        async with third_database.session() as session:
            attempts = list(
                await session.scalars(
                    select(AttemptRow)
                    .where(AttemptRow.job_id == leased_job_id)
                    .order_by(AttemptRow.number)
                )
            )
            active_claims = list(
                await session.scalars(
                    select(LeaseGPUClaimRow).where(LeaseGPUClaimRow.active.is_(True))
                )
            )
            counter = await session.get(
                GPUFenceCounterRow,
                {"site_id": "lab1", "gpu_uuid": "gpu-0"},
            )
            assert [attempt.state for attempt in attempts] == [
                AttemptState.LOST,
                AttemptState.OFFERED,
            ]
            assert len(active_claims) == 1
            assert active_claims[0].epoch == 2
            assert counter is not None and counter.epoch == 2

        async with third_database.transaction() as session:
            assert (
                await third_repository.reap_expired_offers(
                    session,
                    now=start + timedelta(seconds=23),
                )
                == 1
            )

        async with third_database.session() as session:
            recovered = await third_repository.get_job(session, leased_job_id)
            still_queued = await third_repository.get_job(session, queued_job_id)
            counter = await session.get(
                GPUFenceCounterRow,
                {"site_id": "lab1", "gpu_uuid": "gpu-0"},
            )
            assert recovered.state is JobState.QUEUED
            assert still_queued.state is JobState.QUEUED
            assert recovered.attempts[1].state is AttemptState.REJECTED
            assert recovered.attempts[1].lease is not None
            assert recovered.attempts[1].lease.state is LeaseState.EXPIRED
            assert not recovered.attempts[1].lease.claims[0].active
            assert counter is not None and counter.epoch == 2
    finally:
        await third_database.close()


@pytest.mark.postgres
async def test_array_limit_serializes_distinct_child_admissions(migrated_postgres_dsn):
    """Two dispatchers hold different child locks, then contend on one parent."""
    from topodrome.scheduler.arrays import active_array_jobs

    database = Database(migrated_postgres_dsn)
    repository = SchedulerRepository()
    parent_id = new_id("array")
    site_id = "array-concurrency-test"
    authority = FencingAuthority("a" * 64)
    try:
        async with database.transaction() as session:
            heartbeat = _heartbeat().model_copy(update={"site_id": site_id})
            await repository.upsert_heartbeat(session, heartbeat)
            session.add(
                JobArrayRow(
                    id=parent_id,
                    owner_username="alice",
                    name="concurrent array",
                    max_parallel=1,
                    held=False,
                    cancel_requested=False,
                    request_fingerprint="a" * 64,
                )
            )
            await session.flush()
            ids = []
            for index in range(2):
                job = await repository.submit_job(session, _job_spec(priority=0))
                job.array_id, job.array_index = parent_id, index
                ids.append(job.id)
        arrived = 0
        barrier = asyncio.Event()

        async def admit(job_id):
            nonlocal arrived
            async with database.transaction() as session:
                job = await repository.get_job(session, job_id, lock=True)
                arrived += 1
                if arrived == 2:
                    barrier.set()
                await asyncio.wait_for(barrier.wait(), timeout=5)
                await repository.claim_locked_job(session, job)
                await repository.create_offer(
                    session,
                    job=job,
                    site_id=site_id,
                    gpu_uuids=("gpu-0",),
                    authority=authority,
                    offer_ttl=timedelta(seconds=20),
                    token_ttl=timedelta(minutes=5),
                )
            return "admitted"

        results = await asyncio.wait_for(
            asyncio.gather(*(admit(job_id) for job_id in ids), return_exceptions=True), timeout=10
        )
        assert results.count("admitted") == 1
        assert sum(isinstance(result, ConflictError) for result in results) == 1
        async with database.session() as session:
            assert await session.scalar(active_array_jobs(parent_id)) == 1
            states = list(
                await session.scalars(select(JobRow.state).where(JobRow.array_id == parent_id))
            )
            assert set(states) == {JobState.ASSIGNING, JobState.QUEUED}
    finally:
        await database.close()


@pytest.mark.postgres
async def test_opposite_array_scan_order_skips_contended_parent_locks(migrated_postgres_dsn):
    from topodrome.scheduler.arrays import array_gate

    database = Database(migrated_postgres_dsn)
    parents = [new_id("array"), new_id("array")]
    try:
        async with database.transaction() as session:
            for parent in parents:
                session.add(
                    JobArrayRow(
                        id=parent,
                        owner_username="alice",
                        max_parallel=1,
                        request_fingerprint="a" * 64,
                    )
                )
        arrivals = [0, 0]
        barriers = [asyncio.Event(), asyncio.Event()]

        async def rendezvous(index):
            arrivals[index] += 1
            if arrivals[index] == 2:
                barriers[index].set()
            await asyncio.wait_for(barriers[index].wait(), timeout=3)

        async def scan(first, second):
            async with database.transaction() as session:
                assert await array_gate(session, SimpleNamespace(array_id=first)) is None
                await rendezvous(0)
                result = await array_gate(session, SimpleNamespace(array_id=second))
                await rendezvous(1)  # Keep both first locks held until both second checks return.
                return result

        results = await asyncio.wait_for(
            asyncio.gather(scan(*parents), scan(*reversed(parents))), timeout=8
        )
        assert results == ["array_admission_busy", "array_admission_busy"]
    finally:
        await database.close()
