from pathlib import Path

import httpx
import pytest

from tests.unit.test_process_telemetry import heartbeat
from topodrome.config import SchedulerSettings
from topodrome.domain.events import AgentEnvelope
from topodrome.scheduler.app import create_app
from topodrome.scheduler.control import AgentConnection

pytestmark = pytest.mark.component


async def test_status_processes_follow_only_committed_fresh_agent_samples(database, tmp_path: Path):
    settings = SchedulerSettings(global_root=tmp_path, fence_signing_key="f" * 32)
    app = create_app(settings, database=database, run_dispatcher=False)
    service = app.state.service

    async def send(message):
        pass

    connection = AgentConnection("lab1", "session", send)
    await service.connections.register(connection)

    async def deliver(sample):
        envelope = AgentEnvelope(
            kind="heartbeat",
            site_id="lab1",
            agent_session_id="session",
            sequence=sample.sequence,
            payload=sample.model_dump(mode="json"),
        )
        await service.handle_agent_envelope(envelope, connection_generation=connection.generation)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await deliver(heartbeat(sequence=2))
        response = await client.get("/api/v1/status")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        row = response.json()["gpus"][0]
        assert row["processes_status"] == "current"
        assert [p["pid"] for p in row["processes"]] == [20, 10]
        assert row["foreign_process_count"] == 1  # Idle workers remain excluded from admission.
        received = row["processes_received_at"]
        await deliver(heartbeat(sequence=1))
        row = (await client.get("/api/v1/status")).json()["gpus"][0]
        assert row["processes_received_at"] == received
        replacement = AgentConnection("lab1", "session", send)
        await service.connections.register(replacement)
        row = (await client.get("/api/v1/status")).json()["gpus"][0]
        assert row["processes"] is None
        assert row["processes_status"] == "unavailable"
