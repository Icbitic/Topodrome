from __future__ import annotations

import base64
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select, text

from topodrome.client.async_client import AsyncClient
from topodrome.config import SchedulerSettings
from topodrome.db.models import JobRow
from topodrome.db.repositories import SchedulerRepository
from topodrome.db.session import CURRENT_SCHEMA_REVISION, Database
from topodrome.domain.enums import GPUReservationState, JobState, SiteKind
from topodrome.domain.specs import GPUInventory, GPUReservationExtensionRequest, SiteHeartbeat
from topodrome.scheduler.app import create_app


@pytest.mark.asyncio
async def test_submission_source_and_status_through_sdk(database: Database, tmp_path: Path) -> None:
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        fence_signing_key="f" * 32,
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient("http://test", transport=transport) as client:
        user_manifest = b'{"project":"api-regression"}\n'
        source_ref = await client._http.post(
            "/api/v1/sources",
            json={
                "files": {
                    "experiment.py": base64.b64encode(b"def run(context): pass").decode(),
                    "manifest.json": base64.b64encode(user_manifest).decode(),
                }
            },
        )
        assert source_ref.status_code == 201
        digest = source_ref.json()["source_ref"].removeprefix("sha256:")
        published = settings.global_root / "projects" / ".topodrome-sources" / digest
        assert published.joinpath("manifest.json").read_bytes() == user_manifest
        assert published.joinpath(".topodrome", "manifest.json").is_file()
        summary = await client.submit(
            entrypoint="experiment.py",
            name="API regression",
            description="Persisted job metadata\nNo GPU execution needed.",
            estimated_runtime_seconds=120,
            source_ref=source_ref.json()["source_ref"],
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            resources={"gpu": 1, "min_vram_gb": 15},
            idempotency_key="sdk-submit-1",
        )
        assert summary.state is JobState.QUEUED
        recovered = await client.status(summary.job_id)
        assert recovered.job_id == summary.job_id
        assert recovered.spec.name == "API regression"
        assert recovered.spec.estimated_runtime_seconds == 120
        assert recovered.spec.description == "Persisted job metadata\nNo GPU execution needed."
        frame = (await client._http.get("/api/v1/status")).json()
        assert (
            next(j for j in frame["jobs"] if j["job_id"] == summary.job_id)["name"]
            == "API regression"
        )
        log_path = (
            settings.global_root
            / "jobs"
            / summary.job_id
            / "attempts"
            / "attempt-1"
            / "logs"
            / "attempt.log"
        )
        log_path.parent.mkdir(parents=True)
        log_path.write_text("captured batch output\n", encoding="utf-8")
        logs = await client._http.get(f"/api/v1/jobs/{summary.job_id}/logs")
        assert logs.status_code == 200
        # A guessed filesystem attempt absent from the job's database history
        # is not an authorized/verified log source.
        assert logs.text == ""
        status_response = await client._http.get("/api/v1/status")
        assert status_response.headers["cache-control"] == "no-store"
        snapshot = status_response.json()
        assert snapshot["schema_version"] == 1
        assert snapshot["jobs"][0]["job_id"] == summary.job_id
        assert "spec" not in snapshot["jobs"][0]
        assert await client.autodl_instances() == []
        assert await client.autodl_operations() == []


@pytest.mark.asyncio
async def test_operator_gpu_reservation_api_and_async_sdk_are_idempotent(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            SiteHeartbeat(
                site_id="lab1",
                site_kind=SiteKind.LAB,
                agent_session_id="agent-1",
                sequence=1,
                gpus=(
                    GPUInventory(
                        uuid="GPU-A",
                        index=3,
                        name="A800",
                        total_memory_bytes=80 * 1024**3,
                        free_memory_bytes=79 * 1024**3,
                        utilization_percent=0,
                    ),
                ),
            ),
        )
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        fence_signing_key="f" * 32,
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient("http://test", transport=transport) as client:
        created = await client.reserve_gpu(
            site_id="lab1",
            gpu_index=3,
            duration_seconds=600,
            reason="driver maintenance",
            idempotency_key="reserve-api-1",
        )
        replayed = await client.reserve_gpu(
            site_id="lab1",
            gpu_index=3,
            duration_seconds=600,
            reason="driver maintenance",
            idempotency_key="reserve-api-1",
        )
        assert replayed.reservation_id == created.reservation_id
        assert created.gpu_uuid == "GPU-A"
        assert created.gpu_index == 3
        assert created.requested_by == "@operator"
        # The disconnected site makes physical clearance pending, while the
        # durable admission exclusion is already visible to every user.
        assert created.state is GPUReservationState.PENDING
        gpus = await client.gpus()
        assert gpus[0]["operator_reserved"] is True

        listed = await client.gpu_reservations(active_only=True)
        assert [row.reservation_id for row in listed] == [created.reservation_id]
        assert listed[0].reason == "driver maintenance"
        detail = await client.gpu_reservation(created.reservation_id)
        assert detail == listed[0]

        extension = GPUReservationExtensionRequest(
            expected_expires_at=detail.expires_at, additional_seconds=1800
        )
        extended = await client.extend_gpu_reservation(detail.reservation_id, extension)
        replay = await client.extend_gpu_reservation(detail.reservation_id, extension)
        assert extended == replay
        assert extended.expires_at == detail.expires_at + timedelta(seconds=1800)
        assert extended.state == detail.state and extended.reservation_id == detail.reservation_id
        assert (await client.gpus())[0]["operator_reserved"] is True
        stale = await client._http.post(
            f"/api/v1/gpu-reservations/{detail.reservation_id}/extend",
            json={**extension.model_dump(mode="json"), "additional_seconds": 3600},
        )
        assert stale.status_code == 409

        canceled = await client.cancel_gpu_reservation(created.reservation_id)
        replayed_cancel = await client.cancel_gpu_reservation(created.reservation_id)
        assert canceled.state is GPUReservationState.CANCELED
        assert replayed_cancel.canceled_at == canceled.canceled_at
        assert (await client.gpus())[0]["operator_reserved"] is False
        dead = await client._http.post(
            f"/api/v1/gpu-reservations/{detail.reservation_id}/extend",
            json=extension.model_dump(mode="json"),
        )
        assert dead.status_code == 409

        invalid = await client._http.post(
            "/api/v1/gpu-reservations",
            json={
                "site_id": "lab1",
                "gpu_uuid": "GPU-A",
                "gpu_index": 3,
                "duration_seconds": 60,
                "reason": "invalid ambiguous target",
            },
        )
        assert invalid.status_code == 422


@pytest.mark.asyncio
async def test_ready_reflects_database_and_unknown_job_is_404(
    database: Database, tmp_path: Path
) -> None:
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path,
        fence_signing_key="f" * 32,
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/health/ready")).status_code == 200
        response = await client.get("/api/v1/jobs/missing")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"


@pytest.mark.asyncio
async def test_production_api_requires_bearer_auth_while_health_remains_probeable(
    database: Database,
    tmp_path: Path,
) -> None:
    credential = "production-api-test-secret"
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path,
        public_url="https://scheduler.example",
        insecure_development_mode=False,
        api_token=credential,
        fence_signing_key="f" * 32,
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://scheduler.example",
    ) as client:
        # Health probes remain unauthenticated, but production readiness is
        # false until the exact application migration is recorded.
        assert (await client.get("/health/ready")).status_code == 503
        async with database.engine.begin() as connection:
            await connection.execute(
                text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)")
            )
            await connection.execute(
                text("INSERT INTO alembic_version(version_num) VALUES (:revision)"),
                {"revision": CURRENT_SCHEMA_REVISION},
            )
        assert (await client.get("/health/ready")).status_code == 200
        assert (await client.get("/api/v1/jobs/missing")).status_code == 401
        assert (
            await client.get(
                "/api/v1/jobs/missing",
                headers={"Authorization": "Token malformed"},
            )
        ).status_code == 401
        response = await client.get(
            "/api/v1/jobs/missing",
            headers={"Authorization": f"Bearer {credential}"},
        )
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_direct_job_api_rejects_missing_malformed_unknown_and_corrupt_sources(
    database: Database,
    tmp_path: Path,
) -> None:
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        fence_signing_key="f" * 32,
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    base_spec = {
        "entrypoint": "experiment.py",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "resources": {"gpu": 1, "min_vram_gb": 15},
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for source_ref in (None, "sha256:not-a-digest", f"sha256:{'a' * 64}"):
            payload = dict(base_spec)
            if source_ref is not None:
                payload["source_ref"] = source_ref
            response = await client.post("/api/v1/jobs", json=payload)
            assert response.status_code == 400
            assert response.json()["code"] == "admission_rejected"

        uploaded = await client.post(
            "/api/v1/sources",
            json={"files": {"experiment.py": base64.b64encode(b"def run(context): pass").decode()}},
        )
        source_ref = uploaded.json()["source_ref"]
        digest = source_ref.removeprefix("sha256:")
        source_path = (
            settings.global_root / "projects" / ".topodrome-sources" / digest / "experiment.py"
        )
        source_path.write_bytes(b"tampered")
        response = await client.post(
            "/api/v1/jobs",
            json={**base_spec, "source_ref": source_ref},
        )
        assert response.status_code == 400

        missing_entrypoint = await client.post(
            "/api/v1/sources",
            json={"files": {"helper.py": base64.b64encode(b"value = 1").decode()}},
        )
        response = await client.post(
            "/api/v1/jobs",
            json={**base_spec, "source_ref": missing_entrypoint.json()["source_ref"]},
        )
        assert response.status_code == 400

    async with database.session() as session:
        assert await session.scalar(select(func.count()).select_from(JobRow)) == 0
