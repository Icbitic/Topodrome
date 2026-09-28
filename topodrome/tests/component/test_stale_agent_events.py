from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from topodrome.config import SchedulerSettings
from topodrome.db.models import AgentEventAckRow, KernelSessionRow
from topodrome.db.session import Database
from topodrome.domain.enums import (
    AttemptState,
    JobState,
    KernelMode,
    KernelState,
    ModelAvailability,
    SiteKind,
)
from topodrome.domain.events import AgentEnvelope
from topodrome.domain.specs import GPUInventory, JobSpec, ModelCapability, SiteHeartbeat, utc_now
from topodrome.scheduler.app import create_app
from topodrome.scheduler.service import SchedulerService

pytestmark = pytest.mark.component

GIB = 1024**3


@dataclass(frozen=True)
class _ExpiredAttempt:
    job_id: str
    attempt_id: str
    fencing_token: str
    retry_attempt_id: str


async def _service_with_expired_attempt(
    database: Database,
    tmp_path: Path,
) -> tuple[SchedulerService, SchedulerSettings, _ExpiredAttempt]:
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        fence_signing_key="late-agent-envelope-fence-key-000000",
        agent_tokens={"lab1": "agent-control-test-token"},
    )
    service = SchedulerService(database=database, settings=settings)
    issued_at = utc_now() - timedelta(hours=2)
    heartbeat = SiteHeartbeat(
        site_id="lab1",
        site_kind=SiteKind.LAB,
        agent_session_id="agent-session",
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
                alias="model",
                revision="revision",
                backend="transformers",
                backend_version="test",
                dtype="bfloat16",
                availability=ModelAvailability.CACHED,
            ),
        ),
    )
    spec = JobSpec(
        entrypoint="experiment.py",
        model={"alias": "model", "revision": "revision"},
        resources={"gpu": 1, "min_vram_gb": 15},
        retry={"max_attempts": 2, "backoff_seconds": 0, "retry_on": {"lost"}},
    )
    async with database.transaction() as session:
        await service.repository.upsert_heartbeat(session, heartbeat)
        submitted = await service.repository.submit_job(session, spec)
    async with database.transaction() as session:
        job = await service.repository.claim_next_job(session, now=issued_at)
        assert job is not None
        offer = await service.repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("GPU-A",),
            authority=service.fencing,
            offer_ttl=timedelta(minutes=5),
            token_ttl=timedelta(minutes=5),
            lease_ttl=timedelta(seconds=30),
            now=issued_at,
        )
    async with database.transaction() as session:
        await service.repository.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=service.fencing,
            lease_ttl=timedelta(seconds=30),
            now=issued_at,
        )
        for state in (AttemptState.STARTING, AttemptState.RUNNING):
            await service.repository.update_attempt(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=service.fencing,
                target=state,
                now=issued_at,
            )
    async with database.transaction() as session:
        assert (
            await service.repository.reap_expired_leases(
                session,
                now=issued_at + timedelta(seconds=31),
            )
            == 1
        )
    async with database.transaction() as session:
        retry_job = await service.repository.claim_next_job(
            session,
            now=issued_at + timedelta(seconds=32),
        )
        assert retry_job is not None
        retry = await service.repository.create_offer(
            session,
            job=retry_job,
            site_id="lab1",
            gpu_uuids=("GPU-A",),
            authority=service.fencing,
            offer_ttl=timedelta(minutes=5),
            token_ttl=timedelta(minutes=5),
            now=issued_at + timedelta(seconds=32),
        )
    return (
        service,
        settings,
        _ExpiredAttempt(
            job_id=submitted.id,
            attempt_id=offer.attempt_id,
            fencing_token=offer.fencing_token,
            retry_attempt_id=retry.attempt_id,
        ),
    )


def _late_event(
    expired: _ExpiredAttempt,
    target: AttemptState,
    *,
    message_id: str,
    sequence: int,
    token: str | None = None,
    job_id: str | None = None,
) -> AgentEnvelope:
    return AgentEnvelope(
        message_id=message_id,
        site_id="lab1",
        agent_session_id="agent-session",
        sequence=sequence,
        kind="attempt_event",
        payload={
            "attempt_id": expired.attempt_id,
            "job_id": job_id or expired.job_id,
            "site_id": "lab1",
            "fencing_token": token or expired.fencing_token,
            "state": target.value,
            "reason_code": "late_agent_report",
        },
    )


@pytest.mark.parametrize(
    "target",
    [
        AttemptState.STARTING,
        AttemptState.PREEMPT_REQUESTED,
        AttemptState.SUCCEEDED,
        AttemptState.FAILED,
    ],
)
@pytest.mark.asyncio
async def test_late_attempt_reports_are_durably_acked_without_regressing_terminal_state(
    database: Database,
    tmp_path: Path,
    target: AttemptState,
) -> None:
    service, _, expired = await _service_with_expired_attempt(database, tmp_path)
    event = _late_event(expired, target, message_id=f"late-{target.value}", sequence=10)

    reply = await service.handle_agent_envelope(event)
    assert reply.payload == {"message_id": event.message_id, "stale": True}
    duplicate = await service.handle_agent_envelope(event)
    assert duplicate.payload == {"message_id": event.message_id, "duplicate": True}

    async with database.session() as session:
        attempts = await service.repository.list_attempts(session, expired.job_id)
        assert [(row.id, row.state) for row in attempts] == [
            (expired.attempt_id, AttemptState.LOST),
            (expired.retry_attempt_id, AttemptState.OFFERED),
        ]
        job = await service.repository.get_job(session, expired.job_id)
        assert job.state is JobState.ASSIGNING
        receipts = int(
            await session.scalar(select(func.count(AgentEventAckRow.message_id))) or 0
        )
        assert receipts == 1


@pytest.mark.asyncio
async def test_ordered_late_preemption_then_renewal_are_both_acked_after_lease_expiry(
    database: Database,
    tmp_path: Path,
) -> None:
    service, _, expired = await _service_with_expired_attempt(database, tmp_path)
    preemption = _late_event(
        expired,
        AttemptState.PREEMPT_REQUESTED,
        message_id="ordered-late-preemption",
        sequence=30,
    )
    renewal = AgentEnvelope(
        message_id="ordered-late-renewal",
        site_id="lab1",
        agent_session_id="agent-session",
        sequence=31,
        kind="lease_renewal",
        payload={
            "attempt_id": expired.attempt_id,
            "fencing_token": expired.fencing_token,
        },
    )

    preemption_reply = await service.handle_agent_envelope(preemption)
    renewal_reply = await service.handle_agent_envelope(renewal)
    renewal_replay = await service.handle_agent_envelope(renewal)

    assert preemption_reply.payload == {
        "message_id": preemption.message_id,
        "stale": True,
    }
    assert renewal_reply.payload == {
        "message_id": renewal.message_id,
        "stale": True,
        "lease_renewal_terminal": {
            "attempt_id": expired.attempt_id,
            "state": AttemptState.LOST.value,
        },
    }
    assert renewal_replay.payload == {
        **renewal_reply.payload,
        "duplicate": True,
    }
    async with database.session() as session:
        attempts = await service.repository.list_attempts(session, expired.job_id)
        assert [(row.id, row.state) for row in attempts] == [
            (expired.attempt_id, AttemptState.LOST),
            (expired.retry_attempt_id, AttemptState.OFFERED),
        ]
        assert (
            int(await session.scalar(select(func.count(AgentEventAckRow.message_id))) or 0)
            == 2
        )


@pytest.mark.parametrize("kind", ["kernel_event", "kernel_result"])
@pytest.mark.asyncio
async def test_late_terminal_kernel_messages_are_acked_without_reviving_session(
    database: Database,
    tmp_path: Path,
    kind: str,
) -> None:
    service, _, expired = await _service_with_expired_attempt(database, tmp_path)
    session_id = "kernel-expired"
    async with database.transaction() as session:
        session.add(
            KernelSessionRow(
                id=session_id,
                state=KernelState.LOST,
                mode=KernelMode.RAW_MODEL,
                request={},
                job_id=expired.job_id,
                attempt_id=expired.attempt_id,
                site_id="lab1",
                relay_id="relay-expired",
                runtime_id=None,
                actor_id=None,
                expires_at=None,
            )
        )

    payload: dict[str, Any] = {
        "session_id": session_id,
        "attempt_id": expired.attempt_id,
        "fencing_token": expired.fencing_token,
    }
    if kind == "kernel_event":
        payload.update(
            {
                "state": "running",
                "relay_id": "relay-expired",
                "runtime_id": "runtime-expired",
                "actor_id": "actor-expired",
            }
        )
    else:
        payload.update(
            {
                "operation_id": "operation-expired",
                "succeeded": True,
                "result": {"must_not_commit": True},
            }
        )
    event = AgentEnvelope(
        message_id=f"late-{kind}",
        site_id="lab1",
        agent_session_id="agent-session",
        sequence=35,
        kind=kind,  # type: ignore[arg-type]
        payload=payload,
    )

    reply = await service.handle_agent_envelope(event)

    assert reply.payload == {"message_id": event.message_id, "stale": True}
    async with database.session() as session:
        kernel = await service.repository.get_kernel(session, session_id)
        assert kernel.state is KernelState.LOST
        assert kernel.runtime_id is None
        assert kernel.actor_id is None


@pytest.mark.parametrize("state_before_preemption", [AttemptState.ACCEPTED, AttemptState.STARTING])
@pytest.mark.asyncio
async def test_preemption_is_protocol_valid_before_running(
    database: Database,
    tmp_path: Path,
    state_before_preemption: AttemptState,
) -> None:
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        fence_signing_key="early-preemption-fence-key-0000000000",
    )
    service = SchedulerService(database=database, settings=settings)
    heartbeat = SiteHeartbeat(
        site_id="lab1",
        site_kind=SiteKind.LAB,
        agent_session_id="agent-session",
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
        models=(),
    )
    async with database.transaction() as session:
        await service.repository.upsert_heartbeat(session, heartbeat)
        submitted = await service.repository.submit_job(
            session,
            JobSpec(
                entrypoint="experiment.py",
                model={"alias": "model", "revision": "revision"},
                resources={"gpu": 1, "min_vram_gb": 15},
            ),
        )
    async with database.transaction() as session:
        job = await service.repository.claim_next_job(session)
        assert job is not None
        offer = await service.repository.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("GPU-A",),
            authority=service.fencing,
            offer_ttl=timedelta(minutes=5),
            token_ttl=timedelta(minutes=5),
        )
    async with database.transaction() as session:
        await service.repository.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=service.fencing,
            lease_ttl=timedelta(minutes=2),
        )
        if state_before_preemption is AttemptState.STARTING:
            await service.repository.update_attempt(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=service.fencing,
                target=AttemptState.STARTING,
            )

    report = AgentEnvelope(
        message_id=f"early-preemption-{state_before_preemption.value}",
        site_id="lab1",
        agent_session_id="agent-session",
        sequence=40,
        kind="attempt_event",
        payload={
            "attempt_id": offer.attempt_id,
            "job_id": submitted.id,
            "site_id": "lab1",
            "fencing_token": offer.fencing_token,
            "state": AttemptState.PREEMPT_REQUESTED.value,
        },
    )
    reply = await service.handle_agent_envelope(report)
    assert reply.payload == {"message_id": report.message_id}
    async with database.session() as session:
        attempts = await service.repository.list_attempts(session, submitted.id)
        assert attempts[0].state is AttemptState.PREEMPT_REQUESTED
        job = await service.repository.get_job(session, submitted.id)
        assert job.state is JobState.RUNNING


class _AgentSocket:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("test", 80),
            "client": ("127.0.0.1", 12345),
            "root_path": "",
            "path": "/api/v1/agents/control",
            "raw_path": b"/api/v1/agents/control",
            "query_string": b"site_id=lab1&session_id=agent-session",
            "headers": [
                (b"host", b"test"),
                (b"authorization", b"Bearer agent-control-test-token"),
            ],
            "subprotocols": [],
            "state": {},
        }
        await self.incoming.put({"type": "websocket.connect"})
        self.task = asyncio.create_task(
            self.app(scope, self.incoming.get, self.outgoing.put)
        )
        accepted = await asyncio.wait_for(self.outgoing.get(), timeout=2)
        assert accepted["type"] == "websocket.accept"

    async def send(self, envelope: AgentEnvelope) -> dict[str, Any]:
        await self.incoming.put(
            {
                "type": "websocket.receive",
                "text": json.dumps(envelope.model_dump(mode="json")),
            }
        )
        return await asyncio.wait_for(self.outgoing.get(), timeout=2)

    async def disconnect(self) -> None:
        if self.task is None:
            return
        if not self.task.done():
            await self.incoming.put({"type": "websocket.disconnect", "code": 1000})
        results = await asyncio.wait_for(
            asyncio.gather(self.task, return_exceptions=True),
            timeout=2,
        )
        assert results == [None]


@pytest.mark.asyncio
async def test_agent_websocket_acks_stale_report_and_closes_invalid_fence_in_protocol_space(
    database: Database,
    tmp_path: Path,
) -> None:
    service, settings, expired = await _service_with_expired_attempt(database, tmp_path)
    app = create_app(settings, database=database, service=service, run_dispatcher=False)
    socket = _AgentSocket(app)
    await socket.connect()
    try:
        stale = _late_event(
            expired,
            AttemptState.PREEMPT_REQUESTED,
            message_id="late-over-websocket",
            sequence=20,
        )
        response = await socket.send(stale)
        assert response["type"] == "websocket.send"
        assert json.loads(response["text"])["payload"] == {
            "message_id": stale.message_id,
            "stale": True,
        }
        assert socket.task is not None and not socket.task.done()

        invalid = _late_event(
            expired,
            AttemptState.STARTING,
            message_id="bad-fence-over-websocket",
            sequence=21,
            token="not-a-fencing-token",  # noqa: S106 - deliberately malformed canary
        )
        response = await socket.send(invalid)
        assert response == {
            "type": "websocket.close",
            "code": 4400,
            "reason": "stale_fence",
        }
    finally:
        await socket.disconnect()
