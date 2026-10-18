from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from topodrome.config import SchedulerSettings
from topodrome.db.models import AttemptRow, GPUReservationRow, GPURow, LeaseGPUClaimRow
from topodrome.db.repositories import SchedulerRepository
from topodrome.db.session import Database
from topodrome.domain.enums import (
    AttemptState,
    GPUReservationState,
    JobState,
    ModelAvailability,
    SiteKind,
)
from topodrome.domain.errors import ConflictError
from topodrome.domain.specs import (
    GPUInventory,
    GPUReservationExtensionRequest,
    GPUReservationRequest,
    JobSpec,
    ModelCapability,
    SiteHeartbeat,
    utc_now,
)
from topodrome.protocol.fencing import FencingAuthority
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
from topodrome.scheduler.reservations import GPUReservationCoordinator
from topodrome.scheduler.service import SchedulerService

pytestmark = pytest.mark.unit

GIB = 1024**3
REPLICA_KEY_HASH = "a" * 64


class _Provider:
    def __init__(self) -> None:
        self.demands: list[tuple[str, JobSpec]] = []

    def supports_demand(self, _spec: JobSpec) -> bool:
        return True

    async def ensure_for_demand(self, job_id: str, spec: JobSpec) -> bool:
        self.demands.append((job_id, spec))
        return True

    async def capacity_pending_for_demand(self, _spec: JobSpec) -> bool:
        return False


class _RecheckBarrierRepository(SchedulerRepository):
    """Pause only the coordinator's final locked intent read."""

    def __init__(self) -> None:
        self.recheck_started = asyncio.Event()
        self.continue_recheck = asyncio.Event()
        self.pause_next_locked_get = True

    async def get_gpu_reservation(
        self,
        session: AsyncSession,
        reservation_id: str,
        *,
        lock: bool = False,
    ) -> GPUReservationRow:
        if lock and self.pause_next_locked_get:
            self.pause_next_locked_get = False
            self.recheck_started.set()
            await self.continue_recheck.wait()
        return await super().get_gpu_reservation(session, reservation_id, lock=lock)


def _request(
    *,
    gpu_uuid: str | None = None,
    gpu_index: int | None = 0,
    duration_seconds: int = 600,
    force: bool = False,
) -> GPUReservationRequest:
    return GPUReservationRequest(
        site_id="lab1",
        gpu_uuid=gpu_uuid,
        gpu_index=gpu_index,
        duration_seconds=duration_seconds,
        reason="operator maintenance",
        force=force,
    )


async def test_extension_is_in_place_replay_safe_and_preserves_eviction_intent(database):
    repo = SchedulerRepository()
    now = utc_now()
    async with database.transaction() as session:
        await repo.upsert_heartbeat(session, _heartbeat(sequence=1))
        row = await repo.create_gpu_reservation(session, _request(force=True), now=now)
        row.eviction_requested_at = now
        reservation_id, state = row.id, row.state
    extension = GPUReservationExtensionRequest(
        expected_expires_at=now + timedelta(seconds=600), additional_seconds=1800
    )
    for attempt in range(2):
        # A fresh transaction / repository also models retry after scheduler restart.
        async with database.transaction() as session:
            row = await SchedulerRepository().extend_gpu_reservation(
                session, reservation_id, extension, now=now + timedelta(seconds=attempt)
            )
            result = repo.gpu_reservation_summary(row)
            assert result.expires_at == now + timedelta(seconds=2400)
            assert result.duration_seconds == 2400
            assert result.reservation_id == reservation_id and result.state == state
            assert result.force and result.eviction_requested_at == now
            assert result.reason == "operator maintenance" and result.gpu_uuid == "GPU-A"
            with pytest.raises(ConflictError, match="already operator-reserved"):
                await repo.create_gpu_reservation(session, _request(), now=now)


@pytest.mark.parametrize("condition", ["changed", "canceled", "expired", "too_long"])
async def test_extension_rejects_stale_terminal_or_overlong_reservations(database, condition):
    repo = SchedulerRepository()
    now = utc_now()
    async with database.transaction() as session:
        await repo.upsert_heartbeat(session, _heartbeat(sequence=1))
        row = await repo.create_gpu_reservation(session, _request(), now=now)
        reservation_id = row.id
        if condition == "canceled":
            await repo.cancel_gpu_reservation(session, reservation_id, now=now)
    extension = GPUReservationExtensionRequest(
        expected_expires_at=now + timedelta(seconds=601 if condition == "changed" else 600),
        additional_seconds=31_536_000 if condition == "too_long" else 1800,
    )
    with pytest.raises(ConflictError):
        async with database.transaction() as session:
            await repo.extend_gpu_reservation(
                session,
                reservation_id,
                extension,
                now=now + timedelta(seconds=600 if condition == "expired" else 0),
            )
    async with database.session() as session:
        row = await repo.get_gpu_reservation(session, reservation_id)
        assert repo.gpu_reservation_summary(row).expires_at == now + timedelta(seconds=600)


async def test_simultaneous_extensions_do_not_overwrite_one_another(database, tmp_path):
    repo = SchedulerRepository()
    async with database.transaction() as session:
        await repo.upsert_heartbeat(session, _heartbeat(sequence=1))
        row = await repo.create_gpu_reservation(session, _request())
        original = repo.gpu_reservation_summary(row)
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 32,
        ),
    )
    results = await asyncio.gather(
        *(
            service.extend_gpu_reservation(
                original.reservation_id,
                GPUReservationExtensionRequest(
                    expected_expires_at=original.expires_at,
                    additional_seconds=seconds,
                ),
            )
            for seconds in (1800, 3600)
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(r, ConflictError) for r in results) == 1
    assert sum(not isinstance(r, Exception) for r in results) == 1


@pytest.mark.parametrize("seconds", [0, -1, 31_536_001])
def test_extension_duration_validation(seconds):
    with pytest.raises(ValidationError):
        GPUReservationExtensionRequest(expected_expires_at=utc_now(), additional_seconds=seconds)


def test_extension_requires_timezone():
    with pytest.raises(ValidationError):
        GPUReservationExtensionRequest(
            expected_expires_at="2026-09-28T12:00:00", additional_seconds=60
        )


def _heartbeat(
    *,
    sequence: int,
    warm_members: tuple[str, ...] = (),
) -> SiteHeartbeat:
    gpus = tuple(
        GPUInventory(
            uuid=gpu_uuid,
            index=index,
            name="NVIDIA A800",
            total_memory_bytes=80 * GIB,
            free_memory_bytes=60 * GIB,
            utilization_percent=0,
        )
        for index, gpu_uuid in enumerate(("GPU-A", "GPU-B"))
    )
    models = tuple(
        ModelCapability(
            repo_id="test/model-a",
            revision="rev",
            backend="transformers",
            backend_version="4.44",
            dtype="bfloat16",
            availability=ModelAvailability.WARM,
            gpu_uuid=gpu_uuid,
            runtime_fingerprint="b" * 64,
            replica_id="ray-replica-1",
            replica_gpu_uuids=warm_members,
            replica_key_hash=REPLICA_KEY_HASH,
        )
        for gpu_uuid in warm_members
    )
    return SiteHeartbeat(
        site_id="lab1",
        site_kind=SiteKind.LAB,
        agent_session_id="agent-session",
        sequence=sequence,
        gpus=gpus,
        models=models,
    )


async def _connect_and_publish(
    database: Database,
    repository: SchedulerRepository,
    connections: AgentConnectionRegistry,
    sent: list[dict[str, Any]],
    *,
    now: Any,
    warm_members: tuple[str, ...],
) -> str:
    async def sender(message: dict[str, object]) -> None:
        sent.append(dict(message))

    await connections.register(
        AgentConnection(
            site_id="lab1",
            session_id="agent-session",
            send=sender,
        )
    )
    identity = await connections.identity("lab1")
    assert identity is not None
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            _heartbeat(sequence=1, warm_members=warm_members),
            connection_generation=identity[1],
            received_at=now,
        )
    return identity[1]


async def _running_offer(database, repository, authority, members, *, interactive=False):
    model = {"repo_id": "test/model-a"}
    if len(members) > 1:
        model["parallelism"] = {
            "mode": "model_sharded",
            "degree": len(members),
            "max_memory_gb_per_gpu": 15,
        }
    async with database.transaction() as session:
        job = await repository.submit_job(
            session,
            JobSpec(
                entrypoint="run.py",
                workload_kind="interactive" if interactive else "batch",
                model=model,
                resources={"gpu": len(members), "min_vram_gb": 15},
                retry={"max_attempts": 1, "retry_on": [], "backoff_seconds": 0},
            ),
        )
        await repository.claim_locked_job(session, job)
        offer = await repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=members,
            authority=authority,
            offer_ttl=timedelta(minutes=5),
            token_ttl=timedelta(hours=1),
        )
    async with database.transaction() as session:
        await repository.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(minutes=5),
        )
        for state in (AttemptState.STARTING, AttemptState.RUNNING):
            await repository.update_attempt(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=authority,
                target=state,
            )
    return offer


@pytest.mark.parametrize("gang", [False, True])
@pytest.mark.parametrize("user_cancel", [False, True])
async def test_forced_reservation_preempts_exact_attempt_and_retries_despite_budget(
    database: Database, gang: bool, user_cancel: bool
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    sent = []
    await _connect_and_publish(
        database, repository, connections, sent, now=utc_now(), warm_members=()
    )
    authority = FencingAuthority(b"s" * 32)
    offer = await _running_offer(
        database, repository, authority, ("GPU-A", "GPU-B") if gang else ("GPU-A",)
    )
    other = None if gang else await _running_offer(database, repository, authority, ("GPU-B",))
    async with database.transaction() as session:
        reservation = await repository.create_gpu_reservation(session, _request(force=True))
    coordinator = GPUReservationCoordinator(
        database=database, repository=repository, connections=connections, heartbeat_ttl_seconds=30
    )
    assert await coordinator.reconcile_once() == 1
    assert sent[-1]["kind"] == "request_checkpoint"
    assert sent[-1]["payload"]["attempt_id"] == offer.attempt_id
    assert sent[-1]["payload"]["cancel_requested"] is False
    async with database.transaction() as session:
        attempt = await session.get(AttemptRow, offer.attempt_id)
        assert attempt.forced_reservation_id == reservation.id
        if user_cancel:
            await repository.cancel_job(session, offer.job_id)
        for state in (AttemptState.PREEMPT_REQUESTED, AttemptState.PREEMPTED):
            await repository.update_attempt(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=authority,
                target=state,
                reason_code="preempted",
            )
        job = await repository.get_job(session, offer.job_id)
        assert job.state is (JobState.CANCELED if user_cancel else JobState.RETRY_WAIT)
        assert job.cancel_requested is user_cancel
        if other:
            untouched = await session.get(AttemptRow, other.attempt_id)
            assert untouched.state is AttemptState.RUNNING
            assert untouched.forced_reservation_id is None
    assert await coordinator.reconcile_once() == 0
    async with database.session() as session:
        row = await repository.get_gpu_reservation(session, reservation.id)
        assert row.state is GPUReservationState.READY


@pytest.mark.parametrize("transition", ["cancel", "expire"])
async def test_forced_reservation_rechecks_intent_before_preemption(
    database: Database, transition: str
) -> None:
    setup = SchedulerRepository()
    barrier = _RecheckBarrierRepository()
    connections = AgentConnectionRegistry()
    sent = []
    await _connect_and_publish(database, setup, connections, sent, now=utc_now(), warm_members=())
    await _running_offer(database, setup, FencingAuthority(b"s" * 32), ("GPU-A",))
    async with database.transaction() as session:
        row = await setup.create_gpu_reservation(session, _request(force=True))
    coordinator = GPUReservationCoordinator(
        database=database, repository=barrier, connections=connections, heartbeat_ttl_seconds=30
    )
    task = asyncio.create_task(coordinator.reconcile_once())
    await asyncio.wait_for(barrier.recheck_started.wait(), timeout=2)
    async with database.transaction() as session:
        if transition == "cancel":
            await setup.cancel_gpu_reservation(session, row.id)
        else:
            current = await setup.get_gpu_reservation(session, row.id, lock=True)
            current.expires_at = utc_now() - timedelta(seconds=1)
    barrier.continue_recheck.set()
    assert await task == 0
    assert sent == []


async def test_forced_reservation_cancels_interactive_host_without_retry(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    sent = []
    await _connect_and_publish(
        database, repository, connections, sent, now=utc_now(), warm_members=()
    )
    authority = FencingAuthority(b"s" * 32)
    offer = await _running_offer(database, repository, authority, ("GPU-A",), interactive=True)
    async with database.transaction() as session:
        await repository.create_gpu_reservation(session, _request(force=True))
    coordinator = GPUReservationCoordinator(
        database=database, repository=repository, connections=connections, heartbeat_ttl_seconds=30
    )
    assert await coordinator.reconcile_once() == 1
    assert sent[-1]["payload"]["cancel_requested"] is True
    async with database.session() as session:
        job = await repository.get_job(session, offer.job_id)
        assert job.cancel_requested
        assert job.state is JobState.CANCELING
        attempt = await session.get(AttemptRow, offer.attempt_id)
        assert attempt.forced_reservation_id is None


def test_request_requires_exactly_one_gpu_selector() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        _request(gpu_uuid=None, gpu_index=None)
    with pytest.raises(ValidationError, match="exactly one"):
        _request(gpu_uuid="GPU-A", gpu_index=0)


@pytest.mark.asyncio
async def test_repository_resolves_index_excludes_placement_and_is_idempotent(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    now = utc_now()
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, _heartbeat(sequence=1))
        reservation = await repository.create_gpu_reservation(
            session,
            _request(),
            requested_by="alice",
            idempotency_key="maintenance-1",
            now=now,
        )
        replay = await repository.create_gpu_reservation(
            session,
            _request(),
            requested_by="alice",
            idempotency_key="maintenance-1",
            now=now + timedelta(seconds=1),
        )
        inventory = await repository.placement_inventory(session)

    assert reservation.id == replay.id
    assert reservation.gpu_uuid == "GPU-A"
    assert reservation.gpu_index == 0
    assert reservation.requested_by == "alice"
    assert reservation.state is GPUReservationState.READY
    assert inventory[0].gpus[0].externally_reserved is True
    assert inventory[0].gpus[1].externally_reserved is False

    async with database.transaction() as session:
        with pytest.raises(ConflictError, match="different GPU reservation"):
            await repository.create_gpu_reservation(
                session,
                _request(duration_seconds=601),
                requested_by="alice",
                idempotency_key="maintenance-1",
                now=now,
            )


@pytest.mark.asyncio
async def test_offer_rechecks_reservation_after_a_stale_placement_read(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    now = utc_now()
    spec = JobSpec(
        entrypoint="experiment.py",
        source_ref=f"sha256:{'d' * 64}",
        model="test/model-a",
        resources={"gpu": 1, "min_vram_gb": 1},
    )
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, _heartbeat(sequence=1))
        job = await repository.submit_job(session, spec)
        claimed = await repository.claim_next_job(session)
        assert claimed is job
        # This represents a different scheduler generation committing an
        # operator reservation after the placement snapshot was selected.
        await repository.create_gpu_reservation(session, _request(), now=now)
        with pytest.raises(ConflictError, match="became operator-reserved"):
            await repository.create_offer(
                session,
                job=claimed,
                site_id="lab1",
                gpu_uuids=("GPU-A",),
                authority=FencingAuthority(b"f" * 32),
                offer_ttl=timedelta(seconds=30),
                now=now,
            )
        await repository.release_claimed_job(session, claimed)


@pytest.mark.asyncio
async def test_cancel_and_expiry_are_idempotent_and_restore_eligibility(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    now = utc_now()
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, _heartbeat(sequence=1))
        canceled = await repository.create_gpu_reservation(
            session,
            _request(gpu_uuid="GPU-A", gpu_index=None),
            idempotency_key="cancel-me",
            now=now,
        )
        first = await repository.cancel_gpu_reservation(
            session,
            canceled.id,
            now=now + timedelta(seconds=1),
        )
        second = await repository.cancel_gpu_reservation(
            session,
            canceled.id,
            now=now + timedelta(seconds=2),
        )
        inventory = await repository.placement_inventory(session)
    assert first.state is GPUReservationState.CANCELED
    assert second.canceled_at == first.canceled_at
    assert inventory[0].gpus[0].externally_reserved is False

    async with database.transaction() as session:
        expiring = await repository.create_gpu_reservation(
            session,
            _request(gpu_uuid="GPU-B", gpu_index=None, duration_seconds=1),
            now=now,
        )
        assert (
            await repository.active_gpu_reservation_keys(
                session,
                now=now + timedelta(seconds=2),
            )
            == set()
        )
        assert (
            await repository.expire_gpu_reservations(
                session,
                now=now + timedelta(seconds=2),
            )
            == 1
        )
        assert expiring.state is GPUReservationState.EXPIRED


@pytest.mark.asyncio
async def test_active_claim_delays_last_actor_eviction_until_claim_release(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    sent: list[dict[str, Any]] = []
    now = utc_now()
    await _connect_and_publish(
        database,
        repository,
        connections,
        sent,
        now=now,
        warm_members=("GPU-A",),
    )
    async with database.transaction() as session:
        session.add(
            LeaseGPUClaimRow(
                lease_id="lease-active",
                site_id="lab1",
                gpu_uuid="GPU-A",
                epoch=1,
                active=True,
            )
        )
        reservation = await repository.create_gpu_reservation(
            session,
            _request(),
            now=now,
        )

    coordinator = GPUReservationCoordinator(
        database=database,
        repository=repository,
        connections=connections,
        heartbeat_ttl_seconds=30,
    )
    assert await coordinator.reconcile_once(now=now) == 0
    assert sent == []

    async with database.transaction() as session:
        claim = await session.scalar(select(LeaseGPUClaimRow))
        assert claim is not None
        claim.active = False
    assert await coordinator.reconcile_once(now=now + timedelta(seconds=1)) == 1
    command = sent[0]
    assert command["kind"] == "evict_actor"
    assert command["payload"]["actor_id"] == "ray-replica-1"
    assert command["payload"]["gpu_uuids"] == ["GPU-A"]
    assert command["payload"]["replica_key_hash"] == REPLICA_KEY_HASH
    assert str(command["payload"]["eviction_id"]).startswith(f"reservation:{reservation.id}:")

    # Delivery is not clearance: readiness waits for a new heartbeat proving
    # that the exact warm actor is gone.
    async with database.session() as session:
        pending = await repository.get_gpu_reservation(session, reservation.id)
        assert pending.state is GPUReservationState.PENDING

    identity = await connections.identity("lab1")
    assert identity is not None
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            _heartbeat(sequence=2),
            connection_generation=identity[1],
            received_at=now + timedelta(seconds=2),
        )
    assert await coordinator.reconcile_once(now=now + timedelta(seconds=2)) == 0
    async with database.session() as session:
        ready = await repository.get_gpu_reservation(session, reservation.id)
        assert ready.state is GPUReservationState.READY


@pytest.mark.asyncio
async def test_multigpu_actor_is_never_partially_evicted_and_retry_is_restart_safe(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    sent: list[dict[str, Any]] = []
    now = utc_now()
    await _connect_and_publish(
        database,
        repository,
        connections,
        sent,
        now=now,
        warm_members=("GPU-A", "GPU-B"),
    )
    async with database.transaction() as session:
        session.add(
            LeaseGPUClaimRow(
                lease_id="peer-lease",
                site_id="lab1",
                gpu_uuid="GPU-B",
                epoch=1,
                active=True,
            )
        )
        reservation = await repository.create_gpu_reservation(
            session,
            _request(),
            now=now,
        )
    coordinator = GPUReservationCoordinator(
        database=database,
        repository=repository,
        connections=connections,
        heartbeat_ttl_seconds=30,
    )
    assert await coordinator.reconcile_once(now=now) == 0

    async with database.transaction() as session:
        claim = await session.scalar(select(LeaseGPUClaimRow))
        assert claim is not None
        claim.active = False
    assert await coordinator.reconcile_once(now=now + timedelta(seconds=1)) == 1
    first_eviction_id = sent[-1]["payload"]["eviction_id"]
    assert sent[-1]["payload"]["gpu_uuids"] == ["GPU-A", "GPU-B"]

    # A fresh coordinator after a scheduler restart retries the same durable
    # site intent, rather than creating an ambiguous second eviction.
    restarted = GPUReservationCoordinator(
        database=database,
        repository=repository,
        connections=connections,
        heartbeat_ttl_seconds=30,
    )
    assert await restarted.reconcile_once(now=now + timedelta(seconds=7)) == 1
    assert sent[-1]["payload"]["eviction_id"] == first_eviction_id
    async with database.session() as session:
        row = await repository.get_gpu_reservation(session, reservation.id)
        assert row.eviction_requested_at is not None


@pytest.mark.asyncio
async def test_cancel_or_expire_while_claimed_never_later_evicts_actor(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    sent: list[dict[str, Any]] = []
    now = utc_now()
    await _connect_and_publish(
        database,
        repository,
        connections,
        sent,
        now=now,
        warm_members=("GPU-A",),
    )
    async with database.transaction() as session:
        session.add(
            LeaseGPUClaimRow(
                lease_id="lease-active",
                site_id="lab1",
                gpu_uuid="GPU-A",
                epoch=1,
                active=True,
            )
        )
        canceled = await repository.create_gpu_reservation(
            session,
            _request(),
            now=now,
        )
        await repository.cancel_gpu_reservation(session, canceled.id, now=now)
        claim = await session.scalar(select(LeaseGPUClaimRow))
        assert claim is not None
        claim.active = False
    coordinator = GPUReservationCoordinator(
        database=database,
        repository=repository,
        connections=connections,
        heartbeat_ttl_seconds=30,
    )
    assert await coordinator.reconcile_once(now=now + timedelta(seconds=1)) == 0
    assert sent == []

    async with database.transaction() as session:
        expiring = await repository.create_gpu_reservation(
            session,
            _request(duration_seconds=1),
            now=now,
        )
    assert await coordinator.reconcile_once(now=now + timedelta(seconds=2)) == 0
    assert sent == []
    async with database.session() as session:
        row = await repository.get_gpu_reservation(session, expiring.id)
        assert row.state is GPUReservationState.EXPIRED


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["cancel", "expire"])
async def test_cancel_or_expiry_between_selection_and_delivery_fences_eviction(
    database: Database,
    transition: str,
) -> None:
    setup_repository = SchedulerRepository()
    barrier_repository = _RecheckBarrierRepository()
    connections = AgentConnectionRegistry()
    sent: list[dict[str, Any]] = []
    now = utc_now()
    await _connect_and_publish(
        database,
        setup_repository,
        connections,
        sent,
        now=now,
        warm_members=("GPU-A",),
    )
    async with database.transaction() as session:
        reservation = await setup_repository.create_gpu_reservation(
            session,
            _request(),
            now=now,
        )

    coordinator = GPUReservationCoordinator(
        database=database,
        repository=barrier_repository,
        connections=connections,
        heartbeat_ttl_seconds=30,
    )
    reconciliation = asyncio.create_task(coordinator.reconcile_once(now=now))
    await asyncio.wait_for(barrier_repository.recheck_started.wait(), timeout=2)

    # The command was selected and the selection transaction has ended, but
    # the final locked intent read has not yet happened.
    async with database.transaction() as session:
        if transition == "cancel":
            await setup_repository.cancel_gpu_reservation(
                session,
                reservation.id,
                now=now + timedelta(seconds=1),
            )
        else:
            assert (
                await setup_repository.expire_gpu_reservations(
                    session,
                    now=now + timedelta(seconds=reservation.duration_seconds + 1),
                )
                == 1
            )
    barrier_repository.continue_recheck.set()

    assert await asyncio.wait_for(reconciliation, timeout=2) == 0
    assert sent == []
    async with database.session() as session:
        row = await setup_repository.get_gpu_reservation(session, reservation.id)
        assert row.state is (
            GPUReservationState.CANCELED if transition == "cancel" else GPUReservationState.EXPIRED
        )


@pytest.mark.asyncio
async def test_active_claim_between_selection_and_delivery_fences_eviction(
    database: Database,
) -> None:
    setup_repository = SchedulerRepository()
    barrier_repository = _RecheckBarrierRepository()
    connections = AgentConnectionRegistry()
    sent: list[dict[str, Any]] = []
    now = utc_now()
    await _connect_and_publish(
        database,
        setup_repository,
        connections,
        sent,
        now=now,
        warm_members=("GPU-A",),
    )
    async with database.transaction() as session:
        await setup_repository.create_gpu_reservation(session, _request(), now=now)

    coordinator = GPUReservationCoordinator(
        database=database,
        repository=barrier_repository,
        connections=connections,
        heartbeat_ttl_seconds=30,
    )
    reconciliation = asyncio.create_task(coordinator.reconcile_once(now=now))
    await asyncio.wait_for(barrier_repository.recheck_started.wait(), timeout=2)
    async with database.transaction() as session:
        session.add(
            LeaseGPUClaimRow(
                lease_id="late-active-lease",
                site_id="lab1",
                gpu_uuid="GPU-A",
                epoch=2,
                active=True,
            )
        )
    barrier_repository.continue_recheck.set()

    assert await asyncio.wait_for(reconciliation, timeout=2) == 0
    assert sent == []


@pytest.mark.asyncio
async def test_last_actor_reservation_overrides_floor_and_subsequent_demand_queues(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    sent: list[dict[str, Any]] = []
    now = utc_now()
    await _connect_and_publish(
        database,
        repository,
        connections,
        sent,
        now=now,
        warm_members=("GPU-A",),
    )
    async with database.transaction() as session:
        other_gpu = await session.get(
            GPURow,
            {"site_id": "lab1", "uuid": "GPU-B"},
        )
        assert other_gpu is not None
        other_gpu.healthy = False
    provider = _Provider()
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 32,
            scheduling={
                "institutional_actor_floor_gpu_count": 1,
                "contended_actor_shedding_threshold_gpu_count": 5,
            },
        ),
        connections=connections,
        repository=repository,
        provider_controller=provider,  # type: ignore[arg-type]
    )

    reservation = await service.reserve_gpu(
        _request(),
        requested_by="operator",
        idempotency_key="last-actor-reservation",
    )
    assert reservation.state is GPUReservationState.PENDING
    assert len(sent) == 1
    assert sent[0]["kind"] == "evict_actor"
    async with database.session() as session:
        inventory = await repository.placement_inventory(session)
        assert inventory[0].gpus[0].externally_reserved is True

    spec = JobSpec(
        entrypoint="experiment.py",
        source_ref=f"sha256:{'c' * 64}",
        model={
            "repo_id": "test/model-a",
            "revision": "rev",
            "backend": "transformers",
            "backend_version": "4.44",
            "dtype": "bfloat16",
        },
        resources={"gpu": 1, "min_vram_gb": 1},
    )
    async with database.transaction() as session:
        job = await repository.submit_job(session, spec)
        job_id = job.id
    assert await service.dispatch_once() is False
    assert provider.demands == []
    assert (await service.status(job_id)).state.value == "queued"
    # With no registered capacity, the reservation leaves demand queued.
    assert [message["kind"] for message in sent] == ["evict_actor"]
