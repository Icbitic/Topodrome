from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from topodrome.db.models import GPUReservationRow, JobRow, KernelSessionRow, ProviderInstanceRow
from topodrome.db.repositories import SchedulerRepository
from topodrome.db.session import Database
from topodrome.domain.enums import (
    GPUReservationState,
    JobState,
    KernelMode,
    KernelState,
    ModelAvailability,
    ProviderInstanceState,
    SiteKind,
)
from topodrome.domain.specs import (
    GPUInventory,
    GPUProcess,
    JobSpec,
    ModelCapability,
    SiteHeartbeat,
)
from topodrome.protocol.fencing import FencingAuthority
from topodrome.scheduler.status import system_status_snapshot


@pytest.mark.asyncio
async def test_status_limits_do_not_hide_active_work(database: Database) -> None:
    now = datetime.now(UTC)
    spec = JobSpec(entrypoint="run.py", model="test/model").model_dump(mode="json")
    async with database.transaction() as session:
        for i, state in enumerate(
            (
                JobState.RUNNING,
                JobState.QUEUED,
                JobState.RETRY_WAIT,
                JobState.SUCCEEDED,
                JobState.FAILED,
            )
        ):
            session.add(
                JobRow(id=f"job-{i}", state=state, spec=spec, created_at=now + timedelta(seconds=i))
            )
        for i, state in enumerate((KernelState.IDLE, KernelState.STOPPED, KernelState.LOST)):
            session.add(
                KernelSessionRow(
                    id=f"kernel-{i}",
                    state=state,
                    mode=KernelMode.RAW_MODEL,
                    request={},
                    created_at=now + timedelta(seconds=i),
                )
            )
    async with database.session() as session:
        snapshot = await system_status_snapshot(session, job_limit=3, kernel_limit=1)
    assert [row["state"] for row in snapshot["jobs"]] == ["running", "queued", "retry_wait"]
    assert [row["session_id"] for row in snapshot["kernels"]] == ["kernel-0"]


@pytest.mark.asyncio
async def test_status_snapshot_reports_owners_and_sanitizes_provider_metadata(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    now = datetime(2026, 9, 8, 1, 2, 3, tzinfo=UTC)
    spec = JobSpec(
        entrypoint="experiment.py",
        source_ref=f"sha256:{'a' * 64}",
        model={"repo_id": "test/model", "revision": "rev"},
        resources={"gpu": 1, "min_vram_gb": 15},
    )
    heartbeat = SiteHeartbeat(
        site_id="lab1",
        site_kind=SiteKind.LAB,
        agent_session_id="agent-session",
        sequence=1,
        gpus=(
            GPUInventory(
                uuid="gpu-0",
                index=0,
                name="A800",
                total_memory_bytes=80 * 1024**3,
                free_memory_bytes=60 * 1024**3,
                utilization_percent=25,
                externally_reserved=True,
                processes=(GPUProcess(pid=123, used_memory_bytes=1024),),
            ),
        ),
        models=(
            ModelCapability(
                repo_id="test/model",
                revision="rev",
                backend="transformers",
                backend_version="configured",
                dtype="bfloat16",
                availability=ModelAvailability.WARM,
                gpu_uuid="gpu-0",
            ),
        ),
    )
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, heartbeat)
        await repository.submit_job(session, spec)
    async with database.transaction() as session:
        job = await repository.claim_next_job(session)
        assert job is not None
        offer = await repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("gpu-0",),
            authority=FencingAuthority(b"f" * 32),
            offer_ttl=timedelta(minutes=1),
        )
        session.add(
            KernelSessionRow(
                id="kernel-1",
                state=KernelState.RUNNING,
                mode=KernelMode.RAW_MODEL,
                request={
                    "model": spec.model.model_dump(mode="json"),
                    "resources": spec.resources.model_dump(mode="json"),
                    "mode": "raw-model",
                    "idle_timeout_seconds": 600,
                },
                job_id=job.id,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                relay_id="relay-1",
                runtime_id="runtime-1",
                actor_id="actor-1",
                last_activity_at=now,
                expires_at=now + timedelta(minutes=10),
            )
        )
        session.add(
            ProviderInstanceRow(
                id="provider-1",
                provider="autodl",
                external_id="external-1",
                site_id=None,
                state=ProviderInstanceState.RUNNING,
                region="bj-B2",
                gpu_spec="5090-p",
                image_uuid="image-1",
                idle_since=now - timedelta(minutes=5),
                metadata_json={
                    "requested_gpu_count": 2,
                    "observed_gpu_count": 1,
                    "shape_mismatch": True,
                    "access_token": "must-not-leak",
                },
            )
        )
        session.add(
            GPUReservationRow(
                id="reservation-1",
                site_id="lab1",
                gpu_uuid="gpu-0",
                gpu_index=0,
                requested_by="operator",
                reason="maintenance",
                duration_seconds=600,
                request_fingerprint="a" * 64,
                state=GPUReservationState.PENDING,
                created_at=now,
                expires_at=now + timedelta(minutes=10),
                updated_at=now,
            )
        )

    async with database.session() as session:
        snapshot = await system_status_snapshot(session, now=now)

    assert snapshot["schema_version"] == 1
    assert snapshot["generated_at"] == now
    assert snapshot["sites"][0]["site_id"] == "lab1"
    assert snapshot["gpus"][0]["active_lease"] is True
    assert snapshot["gpus"][0]["warm_replica"] is True
    assert snapshot["gpus"][0]["externally_reserved"] is True
    assert snapshot["gpus"][0]["operator_reserved"] is True
    assert "reservations" not in snapshot
    assert snapshot["jobs"][0] == {
        "array_id": None,
        "array_index": None,
        "blocked_reason": None,
        "timing": snapshot["jobs"][0]["timing"],
        "job_id": offer.job_id,
        "attempt_id": offer.attempt_id,
        "name": None,
        "state": "assigning",
        "site_id": "lab1",
        "gpu_count": 1,
        "model": "test/model",
        "workload_kind": "batch",
        "estimated_runtime_seconds": None,
        "created_at": snapshot["jobs"][0]["created_at"],
        "updated_at": snapshot["jobs"][0]["updated_at"],
    }
    assert snapshot["kernels"][0]["session_id"] == "kernel-1"
    assert snapshot["autodl_instances"][0]["gpu_count"] == 2
    assert snapshot["autodl_instances"][0]["observed_gpu_count"] == 1
    assert snapshot["autodl_instances"][0]["shape_mismatch"] is True
    assert snapshot["autodl_instances"][0]["idle_seconds"] == 300
    assert "must-not-leak" not in repr(snapshot)


@pytest.mark.asyncio
async def test_status_snapshot_rejects_unbounded_limits(database: Database) -> None:
    async with database.session() as session:
        with pytest.raises(ValueError, match="positive"):
            await system_status_snapshot(session, job_limit=0)
        with pytest.raises(ValueError, match="heartbeat TTL"):
            await system_status_snapshot(session, heartbeat_ttl_seconds=0)


@pytest.mark.asyncio
async def test_status_snapshot_marks_stale_ready_site_offline(database: Database) -> None:
    repository = SchedulerRepository()
    now = datetime(2026, 9, 9, 0, 0, tzinfo=UTC)
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            SiteHeartbeat(
                site_id="stale-jupyter",
                site_kind=SiteKind.JUPYTER,
                agent_session_id="old-agent",
                sequence=1,
                sent_at=now - timedelta(seconds=31),
            ),
        )

    async with database.session() as session:
        snapshot = await system_status_snapshot(
            session,
            heartbeat_ttl_seconds=30,
            now=now,
        )

    assert snapshot["sites"][0]["state"] == "offline"


@pytest.mark.asyncio
async def test_status_snapshot_bounds_site_and_gpu_inventory(database: Database) -> None:
    repository = SchedulerRepository()
    async with database.transaction() as session:
        for index in range(3):
            await repository.upsert_heartbeat(
                session,
                SiteHeartbeat(
                    site_id=f"site-{index}",
                    site_kind=SiteKind.LAB,
                    agent_session_id=f"agent-{index}",
                    sequence=1,
                    gpus=(
                        GPUInventory(
                            uuid=f"gpu-{index}",
                            index=0,
                            name="A800",
                            total_memory_bytes=80 * 1024**3,
                            free_memory_bytes=79 * 1024**3,
                            utilization_percent=0,
                        ),
                    ),
                ),
            )

    async with database.session() as session:
        snapshot = await system_status_snapshot(
            session,
            site_limit=2,
            gpu_limit=1,
        )

    assert [row["site_id"] for row in snapshot["sites"]] == ["site-0", "site-1"]
    assert len(snapshot["gpus"]) == 1
