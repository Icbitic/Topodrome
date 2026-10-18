from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from topodrome.db.models import ModelActorRow
from topodrome.db.repositories import SchedulerRepository
from topodrome.db.session import Database
from topodrome.domain.enums import AttemptState, JobState, ModelAvailability, SiteKind
from topodrome.domain.errors import StaleFenceError
from topodrome.domain.specs import GPUInventory, JobSpec, ModelCapability, SiteHeartbeat, utc_now
from topodrome.protocol.fencing import FencingAuthority


def job_spec(
    *,
    required_outputs: tuple[str, ...] = (),
    cpu: float = 1,
    memory_gb: float = 1,
) -> JobSpec:
    return JobSpec(
        entrypoint="experiment.py",
        model={"repo_id": "test/model", "revision": "rev"},
        resources={
            "gpu": 1,
            "min_vram_gb": 15,
            "cpu": cpu,
            "memory_gb": memory_gb,
        },
        required_outputs=required_outputs,
    )


def heartbeat(
    sequence: int = 1,
    *,
    safety_margin_bytes: int = 0,
    host_capacity: bool = False,
    externally_reserved: bool = False,
) -> SiteHeartbeat:
    return SiteHeartbeat(
        site_id="lab1",
        site_kind=SiteKind.LAB,
        agent_session_id="session",
        sequence=sequence,
        gpus=(
            GPUInventory(
                uuid="gpu-0",
                index=0,
                name="A800",
                total_memory_bytes=80 * 1024**3,
                free_memory_bytes=25 * 1024**3,
                utilization_percent=50,
                externally_reserved=externally_reserved,
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
        gpu_safety_margin_bytes=safety_margin_bytes,
        **(
            {
                "total_cpu": 8,
                "available_cpu": 7,
                "total_memory_bytes": 32 * 1024**3,
                "available_memory_bytes": 28 * 1024**3,
            }
            if host_capacity
            else {}
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_username", ["@operator", "alice"])
async def test_idempotent_submit_and_heartbeat_sequence(
    database: Database, owner_username: str,
) -> None:
    repository = SchedulerRepository()
    async with database.transaction() as session:
        first = await repository.submit_job(
            session, job_spec(), idempotency_key="submission-1", owner_username=owner_username,
        )
        second = await repository.submit_job(
            session, job_spec(), idempotency_key="submission-1", owner_username=owner_username,
        )
        assert first.id == second.id
        assert first.idempotency_key.startswith("owner-v1:")
        assert first.owner_username == owner_username
        assert await repository.upsert_heartbeat(session, heartbeat())
        assert not await repository.upsert_heartbeat(session, heartbeat())


@pytest.mark.asyncio
async def test_heartbeat_liveness_uses_scheduler_receipt_time(database: Database) -> None:
    repository = SchedulerRepository()
    received_at = datetime(2026, 9, 9, 0, 0, tzinfo=UTC)
    skewed = heartbeat().model_copy(update={"sent_at": received_at + timedelta(minutes=5)})

    async with database.transaction() as session:
        assert await repository.upsert_heartbeat(
            session,
            skewed,
            received_at=received_at,
        )
    async with database.session() as session:
        inventory = await repository.placement_inventory(session)

    assert inventory[0].heartbeat_at == received_at


@pytest.mark.asyncio
async def test_heartbeat_margin_is_persisted_into_placement_inventory(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    margin = 5 * 1024**3
    async with database.transaction() as session:
        assert await repository.upsert_heartbeat(session, heartbeat(safety_margin_bytes=margin))
    async with database.session() as session:
        inventory = await repository.placement_inventory(session)

    assert len(inventory) == 1
    assert inventory[0].safety_margin_bytes == margin
    assert inventory[0].gpus[0].name == "A800"


@pytest.mark.asyncio
async def test_external_reservation_is_persisted_into_placement_inventory(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    async with database.transaction() as session:
        assert await repository.upsert_heartbeat(
            session,
            heartbeat(externally_reserved=True),
        )
    async with database.session() as session:
        inventory = await repository.placement_inventory(session)

    assert inventory[0].gpus[0].externally_reserved is True


@pytest.mark.asyncio
async def test_warm_runtime_fingerprint_is_preserved_for_actor_reuse(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    runtime_fingerprint = "a" * 64
    reported = heartbeat().model_copy(
        update={
            "models": (
                heartbeat()
                .models[0]
                .model_copy(
                    update={
                        "availability": ModelAvailability.WARM,
                        "gpu_uuid": "gpu-0",
                        "runtime_fingerprint": runtime_fingerprint,
                    }
                ),
            )
        }
    )
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, reported)
    async with database.session() as session:
        inventory = await repository.placement_inventory(session)

    assert inventory[0].models[0].runtime_fingerprint == runtime_fingerprint
    assert inventory[0].gpus[0].actor_key_data is not None
    assert inventory[0].gpus[0].actor_key_data["runtime_fingerprint"] == runtime_fingerprint


@pytest.mark.asyncio
async def test_heartbeat_persists_site_authoritative_actor_idle_since(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    first_idle = datetime(2026, 9, 12, 1, 0, tzinfo=UTC)
    second_idle = first_idle + timedelta(minutes=5)

    def reported(sequence: int, idle_since: datetime) -> SiteHeartbeat:
        base = heartbeat(sequence=sequence)
        return base.model_copy(
            update={
                "models": (
                    ModelCapability(
                        repo_id="test/model",
                        revision="rev",
                        backend="transformers",
                        backend_version="configured",
                        dtype="bfloat16",
                        availability=ModelAvailability.WARM,
                        gpu_uuid="gpu-0",
                        runtime_fingerprint="a" * 64,
                        replica_id="site-actor",
                        replica_gpu_uuids=("gpu-0",),
                        replica_key_hash="b" * 64,
                        idle_since=idle_since,
                    ),
                )
            }
        )

    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, reported(1, first_idle))
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, reported(2, second_idle))
    async with database.session() as session:
        actor = await session.scalar(
            select(ModelActorRow).where(ModelActorRow.active_allocation.is_(True))
        )
        inventory = await repository.placement_inventory(session)

    assert actor is not None
    assert actor.last_used_at.replace(tzinfo=UTC) == second_idle
    assert inventory[0].gpus[0].actor_key_data is not None
    assert inventory[0].gpus[0].actor_key_data["idle_since"] == second_idle.isoformat()


@pytest.mark.asyncio
async def test_heartbeat_persists_shared_sharded_replica_identity(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    runtime_fingerprint = "b" * 64
    base = heartbeat()
    members = ("gpu-1", "gpu-0")
    gpus = (
        base.gpus[0].model_copy(update={"uuid": "gpu-0", "index": 0}),
        base.gpus[0].model_copy(update={"uuid": "gpu-1", "index": 1}),
    )
    models = tuple(
        ModelCapability(
            repo_id="test/model",
            revision="rev",
            backend="transformers",
            backend_version="configured",
            dtype="bfloat16",
            availability=ModelAvailability.WARM,
            gpu_uuid=gpu_uuid,
            runtime_fingerprint=runtime_fingerprint,
            replica_id="replica-shared",
            replica_gpu_uuids=members,
        )
        for gpu_uuid in members
    )
    reported = base.model_copy(update={"gpus": gpus, "models": models})

    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, reported)
    async with database.session() as session:
        inventory = await repository.placement_inventory(session)

    by_uuid = {gpu.uuid: gpu for gpu in inventory[0].gpus}
    assert {
        by_uuid[uuid].actor_key_data["replica_id"]  # type: ignore[index]
        for uuid in members
    } == {"replica-shared"}
    assert all(
        by_uuid[uuid].actor_key_data["replica_gpu_uuids"]  # type: ignore[index]
        == list(members)
        for uuid in members
    )
    assert {model.replica_id for model in inventory[0].models} == {"replica-shared"}


@pytest.mark.asyncio
async def test_placement_inventory_accounts_active_host_resource_reservations(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    authority = FencingAuthority(b"h" * 32)
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, heartbeat(host_capacity=True))
        await repository.submit_job(session, job_spec(cpu=2.5, memory_gb=6))
    async with database.transaction() as session:
        job = await repository.claim_next_job(session)
        assert job is not None
        await repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("gpu-0",),
            authority=authority,
            offer_ttl=timedelta(seconds=30),
        )
    async with database.session() as session:
        inventory = await repository.placement_inventory(session)

    assert inventory[0].total_cpu == 8
    assert inventory[0].available_cpu == 5.5
    assert inventory[0].total_memory_bytes == 32 * 1024**3
    assert inventory[0].available_memory_bytes == 26 * 1024**3


@pytest.mark.asyncio
async def test_full_fenced_attempt_and_recovery(database: Database) -> None:
    repository = SchedulerRepository()
    authority = FencingAuthority(b"s" * 32)
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, heartbeat())
        await repository.submit_job(session, job_spec(), idempotency_key="stable")
    async with database.transaction() as session:
        job = await repository.claim_next_job(session)
        assert job is not None
        offer = await repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("gpu-0",),
            authority=authority,
            offer_ttl=timedelta(seconds=30),
            token_ttl=timedelta(hours=1),
        )
    async with database.transaction() as session:
        await repository.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(seconds=90),
        )
        await repository.update_attempt(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            target=AttemptState.STARTING,
        )
        await repository.update_attempt(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            target=AttemptState.RUNNING,
        )
        await repository.update_attempt(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            target=AttemptState.COMMITTING,
        )
        await repository.update_attempt(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            target=AttemptState.SUCCEEDED,
        )
    async with database.session() as session:
        recovered = await repository.get_job(session, offer.job_id)
        assert recovered.state is JobState.SUCCEEDED
        assert recovered.attempts[0].id == offer.attempt_id
        assert recovered.attempts[0].lease is not None
        assert not recovered.attempts[0].lease.claims[0].active


@pytest.mark.asyncio
async def test_admission_rejection_requeues_without_consuming_identity(database: Database) -> None:
    repository = SchedulerRepository()
    authority = FencingAuthority(b"s" * 32)
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, heartbeat())
        job = await repository.submit_job(session, job_spec())
        stable_job_id = job.id
    async with database.transaction() as session:
        job = await repository.claim_next_job(session)
        assert job is not None
        first = await repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("gpu-0",),
            authority=authority,
            offer_ttl=timedelta(seconds=30),
        )
    async with database.transaction() as session:
        await repository.reject_offer(
            session,
            attempt_id=first.attempt_id,
            site_id="lab1",
            token=first.fencing_token,
            authority=authority,
            reason_code="stale_inventory",
        )
    async with database.transaction() as session:
        job = await repository.claim_next_job(session)
        assert job is not None and job.id == stable_job_id
        second = await repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("gpu-0",),
            authority=authority,
            offer_ttl=timedelta(seconds=30),
        )
        assert second.attempt_id != first.attempt_id
        assert second.attempt_number == 2


@pytest.mark.asyncio
async def test_acceptance_after_offer_deadline_is_rejected_even_with_live_token(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    authority = FencingAuthority(b"s" * 32)
    now = utc_now()
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, heartbeat())
        await repository.submit_job(session, job_spec())
    async with database.transaction() as session:
        job = await repository.claim_next_job(session)
        assert job is not None
        offer = await repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("gpu-0",),
            authority=authority,
            offer_ttl=timedelta(seconds=10),
            token_ttl=timedelta(hours=1),
            now=now,
        )

    with pytest.raises(StaleFenceError, match="offer expired"):
        async with database.transaction() as session:
            await repository.accept_offer(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=authority,
                lease_ttl=timedelta(seconds=90),
                now=now + timedelta(seconds=11),
            )

    async with database.transaction() as session:
        assert await repository.reap_expired_offers(session, now=now + timedelta(seconds=11)) == 1
    async with database.session() as session:
        recovered = await repository.get_job(session, offer.job_id)
        assert recovered.state is JobState.QUEUED
        assert recovered.attempts[0].state is AttemptState.REJECTED


@pytest.mark.asyncio
async def test_expired_active_lease_requeues_job(database: Database) -> None:
    repository = SchedulerRepository()
    authority = FencingAuthority(b"s" * 32)
    now = utc_now()
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, heartbeat())
        await repository.submit_job(session, job_spec())
    async with database.transaction() as session:
        job = await repository.claim_next_job(session)
        assert job is not None
        offer = await repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("gpu-0",),
            authority=authority,
            offer_ttl=timedelta(seconds=30),
            token_ttl=timedelta(hours=1),
            now=now,
        )
    async with database.transaction() as session:
        await repository.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(seconds=10),
            now=now,
        )
    async with database.transaction() as session:
        assert await repository.reap_expired_leases(session, now=now + timedelta(seconds=11)) == 1
    async with database.session() as session:
        job = await repository.get_job(session, offer.job_id)
        assert job.state is JobState.RETRY_WAIT
        assert job.retry_not_before is not None
        assert job.attempts[0].state is AttemptState.LOST


@pytest.mark.asyncio
async def test_expired_lease_cannot_be_revived_by_late_renewal(database: Database) -> None:
    repository = SchedulerRepository()
    authority = FencingAuthority(b"s" * 32)
    now = utc_now()
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, heartbeat())
        await repository.submit_job(session, job_spec())
    async with database.transaction() as session:
        job = await repository.claim_next_job(session)
        assert job is not None
        offer = await repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("gpu-0",),
            authority=authority,
            offer_ttl=timedelta(seconds=30),
            token_ttl=timedelta(hours=1),
            now=now,
        )
    async with database.transaction() as session:
        await repository.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(seconds=10),
            now=now,
        )

    with pytest.raises(StaleFenceError, match="expired before renewal"):
        async with database.transaction() as session:
            await repository.renew_lease(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=authority,
                lease_ttl=timedelta(seconds=10),
                token_ttl=timedelta(hours=1),
                renewal_id="late-renewal",
                renewal_issued_at=now + timedelta(seconds=11),
                now=now + timedelta(seconds=11),
            )

    async with database.transaction() as session:
        assert await repository.reap_expired_leases(session, now=now + timedelta(seconds=11)) == 1


@pytest.mark.asyncio
async def test_gang_renewal_rotates_atomically_and_runs_past_original_token_ttl(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    authority = FencingAuthority(b"r" * 32)
    started_at = utc_now()
    gang_spec = JobSpec(
        entrypoint="experiment.py",
        model={
            "repo_id": "test/model",
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
    second_gpu = GPUInventory(
        uuid="gpu-1",
        index=1,
        name="A800",
        total_memory_bytes=80 * 1024**3,
        free_memory_bytes=25 * 1024**3,
        utilization_percent=50,
    )
    gang_heartbeat = heartbeat().model_copy(update={"gpus": (*heartbeat().gpus, second_gpu)})

    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, gang_heartbeat)
        await repository.submit_job(session, gang_spec)
    async with database.transaction() as session:
        job = await repository.claim_next_job(session, now=started_at)
        assert job is not None
        offer = await repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("gpu-0", "gpu-1"),
            authority=authority,
            offer_ttl=timedelta(seconds=30),
            token_ttl=timedelta(hours=3),
            lease_ttl=timedelta(seconds=90),
            now=started_at,
        )
    async with database.transaction() as session:
        await repository.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(seconds=90),
            now=started_at,
        )

    first_time = started_at + timedelta(seconds=60)
    async with database.transaction() as session:
        first = await repository.renew_lease(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(seconds=90),
            token_ttl=timedelta(hours=3),
            renewal_id="renewal-1",
            renewal_issued_at=first_time,
            now=first_time,
        )
        attempt = await repository._get_attempt(session, offer.attempt_id, lock=False)
        rotated_digest = authority.digest(first.fencing_token)
        assert attempt.fencing_token_digest == rotated_digest
        assert attempt.lease is not None
        assert attempt.lease.fencing_token_digest == rotated_digest
        assert {(claim.gpu_uuid, claim.epoch, claim.active) for claim in attempt.lease.claims} == {
            ("gpu-0", 1, True),
            ("gpu-1", 1, True),
        }

    # An ACK-lost retry reconstructs the exact grant and does not extend the
    # lease a second time.
    async with database.transaction() as session:
        replay = await repository.renew_lease(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(seconds=90),
            token_ttl=timedelta(hours=3),
            renewal_id="renewal-1",
            renewal_issued_at=first_time,
            replayed=True,
            now=first_time + timedelta(seconds=1),
        )
    assert replay == first

    with pytest.raises(StaleFenceError, match="no longer current"):
        async with database.transaction() as session:
            await repository.renew_lease(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=authority,
                lease_ttl=timedelta(seconds=90),
                token_ttl=timedelta(hours=3),
                renewal_id="stale-renewal",
                renewal_issued_at=first_time + timedelta(seconds=1),
                now=first_time + timedelta(seconds=1),
            )

    current_token = first.fencing_token
    for sequence in range(2, 182):
        current = started_at + timedelta(seconds=60 * sequence)
        async with database.transaction() as session:
            grant = await repository.renew_lease(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=current_token,
                authority=authority,
                lease_ttl=timedelta(seconds=90),
                token_ttl=timedelta(hours=3),
                renewal_id=f"renewal-{sequence}",
                renewal_issued_at=current,
                now=current,
            )
        current_token = grant.fencing_token

    beyond_original_expiry = started_at + timedelta(seconds=60 * 181 + 1)
    verified = authority.verify(
        current_token,
        attempt_id=offer.attempt_id,
        site_id="lab1",
        now=beyond_original_expiry,
    )
    assert tuple(claim.gpu_uuid for claim in verified.claims) == ("gpu-0", "gpu-1")
    async with database.transaction() as session:
        await repository.update_attempt(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=current_token,
            authority=authority,
            target=AttemptState.STARTING,
            now=beyond_original_expiry,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retry",
    [
        {"max_attempts": 1, "backoff_seconds": 0, "retry_on": {"runtime_error"}},
        {"max_attempts": 3, "backoff_seconds": 0, "retry_on": {"lost"}},
    ],
)
async def test_retry_policy_honors_max_attempts_and_retry_on(
    database: Database,
    retry: dict[str, object],
) -> None:
    repository = SchedulerRepository()
    authority = FencingAuthority(b"s" * 32)
    now = utc_now()
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, heartbeat())
        submitted = await repository.submit_job(
            session,
            JobSpec(
                entrypoint="experiment.py",
                model={"repo_id": "test/model", "revision": "rev"},
                retry=retry,
            ),
        )
        job_id = submitted.id
    async with database.transaction() as session:
        claimed = await repository.claim_next_job(session, now=now)
        assert claimed is not None
        offer = await repository.create_offer(
            session,
            job=claimed,
            site_id="lab1",
            gpu_uuids=("gpu-0",),
            authority=authority,
            offer_ttl=timedelta(minutes=5),
            token_ttl=timedelta(hours=1),
            now=now,
        )
    async with database.transaction() as session:
        await repository.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(minutes=5),
            now=now,
        )
        for state in (AttemptState.STARTING, AttemptState.RUNNING):
            await repository.update_attempt(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=authority,
                target=state,
                now=now,
            )
        await repository.update_attempt(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            target=AttemptState.FAILED,
            reason_code="runtime_error",
            now=now,
        )

    async with database.session() as session:
        recovered = await repository.get_job(session, job_id)
        assert recovered.state is JobState.FAILED
        assert recovered.retry_not_before is None
