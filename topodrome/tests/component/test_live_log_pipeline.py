from __future__ import annotations

import base64
import hashlib
import os
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from topodrome.config import SchedulerSettings
from topodrome.db.models import AgentEventAckRow, LeaseRow
from topodrome.db.session import Database
from topodrome.domain.enums import AttemptState, ModelAvailability, SiteKind
from topodrome.domain.errors import DurabilityError, StaleFenceError
from topodrome.domain.events import AgentEnvelope
from topodrome.domain.specs import GPUInventory, JobSpec, ModelCapability, SiteHeartbeat, utc_now
from topodrome.scheduler.app import create_app
from topodrome.scheduler.service import SchedulerService

pytestmark = pytest.mark.component
GIB = 1024**3


async def _running_attempt(
    database: Database,
    tmp_path: Path,
) -> tuple[SchedulerService, SchedulerSettings, str, str, str]:
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        fence_signing_key="live-log-fencing-key-with-32-bytes",
    )
    service = SchedulerService(database=database, settings=settings)
    heartbeat = SiteHeartbeat(
        site_id="remote-1",
        site_kind=SiteKind.JUPYTER,
        agent_session_id="session-1",
        sequence=1,
        gpus=(
            GPUInventory(
                uuid="GPU-A",
                index=0,
                name="A800",
                total_memory_bytes=80 * GIB,
                free_memory_bytes=70 * GIB,
                utilization_percent=0,
            ),
        ),
        models=(
            ModelCapability(
                repo_id="test/model",
                revision="revision",
                backend="transformers",
                backend_version="configured",
                dtype="bfloat16",
                availability=ModelAvailability.CACHED,
            ),
        ),
    )
    spec = JobSpec(
        entrypoint="experiment.py",
        model={"repo_id": "test/model", "revision": "revision"},
        resources={"gpu": 1, "min_vram_gb": 1},
        retry={"max_attempts": 1},
    )
    now = utc_now()
    async with database.transaction() as session:
        await service.repository.upsert_heartbeat(session, heartbeat)
        submitted = await service.repository.submit_job(session, spec)
    async with database.transaction() as session:
        job = await service.repository.claim_next_job(session, now=now)
        assert job is not None
        offer = await service.repository.create_offer(
            session,
            job=job,
            site_id="remote-1",
            gpu_uuids=("GPU-A",),
            authority=service.fencing,
            offer_ttl=timedelta(minutes=2),
            lease_ttl=timedelta(minutes=2),
            token_ttl=timedelta(minutes=2),
            now=now,
        )
    async with database.transaction() as session:
        await service.repository.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id="remote-1",
            token=offer.fencing_token,
            authority=service.fencing,
            lease_ttl=timedelta(minutes=2),
            now=now,
        )
        for state in (AttemptState.STARTING, AttemptState.RUNNING):
            await service.repository.update_attempt(
                session,
                attempt_id=offer.attempt_id,
                site_id="remote-1",
                token=offer.fencing_token,
                authority=service.fencing,
                target=state,
                now=now,
            )
    return service, settings, submitted.id, offer.attempt_id, offer.fencing_token


def _chunk_event(job_id: str, attempt_id: str, token: str, data: bytes) -> AgentEnvelope:
    return AgentEnvelope(
        site_id="remote-1",
        agent_session_id="session-1",
        sequence=0,
        kind="attempt_log_chunk",
        payload={
            "site_id": "remote-1",
            "job_id": job_id,
            "attempt_id": attempt_id,
            "fencing_token": token,
            "generation": 0,
            "offset": 0,
            "data_base64": base64.b64encode(data).decode("ascii"),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    )


@pytest.mark.asyncio
async def test_fenced_progress_reaches_dashboard_without_completing_job(database, tmp_path):
    service, settings, job_id, attempt_id, token = await _running_attempt(database, tmp_path)
    data = b'TOPODROME_PROGRESS_V1 {"current":100,"total":100,"stage":"training"}\n'
    await service.handle_agent_envelope(_chunk_event(job_id, attempt_id, token, data))
    app = create_app(settings, database=database, service=service, run_dispatcher=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.get("/api/v1/status")
        assert response.status_code == 200
        job = next(j for j in response.json()["jobs"] if j["job_id"] == job_id)
        assert job["progress"]["current"] == 100
        assert job["state"] == "running"
        assert job["attempt_id"] == attempt_id


@pytest.mark.asyncio
async def test_failed_job_keeps_predeath_logs_after_advisory_retention(
    database: Database,
    tmp_path: Path,
) -> None:
    service, settings, job_id, attempt_id, token = await _running_attempt(database, tmp_path)
    data = b"last output before worker death\n"
    await service.handle_agent_envelope(_chunk_event(job_id, attempt_id, token, data))
    await service.handle_agent_envelope(
        AgentEnvelope(
            site_id="remote-1",
            agent_session_id="session-1",
            sequence=1,
            kind="attempt_event",
            payload={
                "site_id": "remote-1",
                "job_id": job_id,
                "attempt_id": attempt_id,
                "fencing_token": token,
                "state": "failed",
                "log_complete": False,
            },
        )
    )
    path = service.live_logs.latest_path(job_id, attempt_id)
    assert path is not None
    os.utime(path, (1, 1))
    assert await service.prune_live_logs_once() == 0
    app = create_app(settings, database=database, service=service, run_dispatcher=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(f"/api/v1/jobs/{job_id}/logs")
        assert response.status_code == 200
        assert response.content == data


@pytest.mark.asyncio
async def test_remote_live_chunks_are_fenced_idempotent_and_terminally_authoritative(
    database: Database,
    tmp_path: Path,
) -> None:
    service, settings, job_id, attempt_id, token = await _running_attempt(database, tmp_path)
    event = _chunk_event(job_id, attempt_id, token, b"hello ")

    accepted = await service.handle_agent_envelope(event)
    replay = await service.handle_agent_envelope(event)
    assert accepted.payload["attempt_log_chunk"] == {
        "accepted": True,
        "generation": 0,
        "next_offset": 6,
        "duplicate": False,
        "reset": False,
        "stop": False,
    }
    assert replay.payload["attempt_log_chunk"]["duplicate"] is True
    async with database.session() as session:
        assert int(await session.scalar(select(func.count(AgentEventAckRow.message_id))) or 0) == 0

    app = create_app(
        settings,
        database=database,
        service=service,
        run_dispatcher=False,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        live = await client.get(f"/api/v1/jobs/{job_id}/attempts/{attempt_id}/logs")
        assert live.content == b"hello "
        assert live.headers["x-topodrome-log-complete"] == "false"
        assert not (settings.global_root / "jobs" / job_id).exists()
        tail = await client.get(f"/api/v1/jobs/{job_id}/logs")
        assert tail.content == b"hello "
        assert tail.headers["cache-control"] == "no-store"
        bounded = await client.get(f"/api/v1/jobs/{job_id}/logs?tail_bytes=3")
        assert bounded.content == b"lo "
        offset = int(live.headers["x-topodrome-log-next-offset"])
        anchor = live.headers["x-topodrome-log-anchor-sha256"]

        authoritative = (
            settings.global_root
            / "jobs"
            / job_id
            / "attempts"
            / attempt_id
            / "logs"
            / "attempt.log"
        )
        authoritative.parent.mkdir(parents=True, exist_ok=True)
        authoritative.write_bytes(b"hello world\n")
        # An unverified upload must not replace the live view early.
        assert (await client.get(f"/api/v1/jobs/{job_id}/logs")).content == b"hello "
        committing = AgentEnvelope(
            site_id="remote-1",
            agent_session_id="session-1",
            sequence=1,
            kind="attempt_event",
            payload={
                "site_id": "remote-1",
                "job_id": job_id,
                "attempt_id": attempt_id,
                "fencing_token": token,
                "state": "committing",
            },
        )
        await service.handle_agent_envelope(committing)
        terminal = AgentEnvelope(
            site_id="remote-1",
            agent_session_id="session-1",
            sequence=2,
            kind="attempt_event",
            payload={
                "site_id": "remote-1",
                "job_id": job_id,
                "attempt_id": attempt_id,
                "fencing_token": token,
                "state": "succeeded",
                "log": {
                    "logical_path": (f"jobs/{job_id}/attempts/{attempt_id}/logs/attempt.log"),
                    "start_offset": 0,
                    "end_offset": len(b"hello world\n"),
                    "sha256": hashlib.sha256(b"hello world\n").hexdigest(),
                },
                "log_complete": True,
            },
        )
        await service.handle_agent_envelope(terminal)
        assert service.live_logs.latest_path(job_id, attempt_id) is None

        final = await client.get(
            f"/api/v1/jobs/{job_id}/attempts/{attempt_id}/logs",
            params={"offset": offset, "anchor_sha256": anchor},
        )
        assert final.headers["x-topodrome-log-reset"] == "false"
        assert final.headers["x-topodrome-log-complete"] == "true"
        assert final.content == b"world\n"
        assert (await client.get(f"/api/v1/jobs/{job_id}/logs")).content == b"hello world\n"

    stale = _chunk_event(job_id, attempt_id, token, b"replace")
    stale_reply = await service.handle_agent_envelope(stale)
    assert stale_reply.payload["attempt_log_chunk"]["stale"] is True
    assert authoritative.read_bytes() == b"hello world\n"


@pytest.mark.asyncio
async def test_invalid_live_chunk_error_never_echoes_content_or_fence(
    database: Database,
    tmp_path: Path,
) -> None:
    service, _, job_id, attempt_id, token = await _running_attempt(database, tmp_path)
    captured_output = "captured-secret-output"
    event = _chunk_event(job_id, attempt_id, token, captured_output.encode())
    event.payload["sha256"] = "0" * 64

    with pytest.raises(StaleFenceError) as captured:
        await service.handle_agent_envelope(event)

    assert str(captured.value) == "invalid live attempt log chunk"
    assert captured_output not in str(captured.value)
    assert token not in str(captured.value)


@pytest.mark.asyncio
async def test_terminal_event_replay_retries_authoritative_registration_after_commit(
    database: Database,
    tmp_path: Path,
) -> None:
    service, settings, job_id, attempt_id, token = await _running_attempt(database, tmp_path)
    await service.handle_agent_envelope(
        _chunk_event(job_id, attempt_id, token, b"terminal output\n")
    )
    await service.handle_agent_envelope(
        AgentEnvelope(
            site_id="remote-1",
            agent_session_id="session-1",
            sequence=1,
            kind="attempt_event",
            payload={
                "site_id": "remote-1",
                "job_id": job_id,
                "attempt_id": attempt_id,
                "fencing_token": token,
                "state": "committing",
            },
        )
    )
    final_bytes = b"terminal output\n"
    terminal = AgentEnvelope(
        message_id="msg-terminal-log-replay",
        site_id="remote-1",
        agent_session_id="session-1",
        sequence=2,
        kind="attempt_event",
        payload={
            "site_id": "remote-1",
            "job_id": job_id,
            "attempt_id": attempt_id,
            "fencing_token": token,
            "state": "succeeded",
            "log": {
                "logical_path": (f"jobs/{job_id}/attempts/{attempt_id}/logs/attempt.log"),
                "start_offset": 0,
                "end_offset": len(final_bytes),
                "sha256": hashlib.sha256(final_bytes).hexdigest(),
            },
            "log_complete": True,
        },
    )

    with pytest.raises(DurabilityError):
        await service.handle_agent_envelope(terminal)
    assert (await service.status(job_id)).state.value == "succeeded"
    assert service.live_logs.latest_path(job_id, attempt_id) is not None

    authoritative = (
        settings.global_root / "jobs" / job_id / "attempts" / attempt_id / "logs" / "attempt.log"
    )
    authoritative.parent.mkdir(parents=True, exist_ok=True)
    authoritative.write_bytes(final_bytes)
    replay = await service.handle_agent_envelope(terminal)

    assert replay.payload["duplicate"] is True
    assert service.live_logs.latest_path(job_id, attempt_id) is None
    assert service.live_logs.is_complete(job_id, attempt_id)
    assert authoritative.read_bytes() == final_bytes


@pytest.mark.asyncio
async def test_expired_active_lease_live_chunk_is_acknowledged_without_ingestion(
    database: Database,
    tmp_path: Path,
) -> None:
    service, _, job_id, attempt_id, token = await _running_attempt(database, tmp_path)
    async with database.transaction() as session:
        lease = await session.scalar(select(LeaseRow).where(LeaseRow.attempt_id == attempt_id))
        assert lease is not None
        lease.expires_at = utc_now() - timedelta(seconds=1)

    reply = await service.handle_agent_envelope(
        _chunk_event(job_id, attempt_id, token, b"must-not-land")
    )

    assert reply.payload["attempt_log_chunk"]["stale"] is True
    assert service.live_logs.latest_path(job_id, attempt_id) is None


@pytest.mark.asyncio
async def test_rotated_live_log_fence_rejects_old_token_and_accepts_current_token(
    database: Database,
    tmp_path: Path,
) -> None:
    service, _, job_id, attempt_id, token = await _running_attempt(database, tmp_path)
    now = utc_now()
    async with database.transaction() as session:
        grant = await service.repository.renew_lease(
            session,
            attempt_id=attempt_id,
            site_id="remote-1",
            token=token,
            authority=service.fencing,
            lease_ttl=timedelta(minutes=2),
            token_ttl=timedelta(minutes=2),
            renewal_id="renew-live-log",
            renewal_issued_at=now,
            now=now,
        )

    stale = await service.handle_agent_envelope(
        _chunk_event(job_id, attempt_id, token, b"old-token")
    )
    accepted = await service.handle_agent_envelope(
        _chunk_event(
            job_id,
            attempt_id,
            grant.fencing_token,
            b"current-token",
        )
    )

    assert stale.payload["attempt_log_chunk"]["stale"] is True
    assert accepted.payload["attempt_log_chunk"]["accepted"] is True
    path = service.live_logs.latest_path(job_id, attempt_id)
    assert path is not None and path.read_bytes() == b"current-token"
