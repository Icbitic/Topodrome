
import pytest

from topodrome.config import SchedulerSettings
from topodrome.domain.enums import SiteKind
from topodrome.domain.events import AgentEnvelope
from topodrome.domain.specs import GPUInventory, GPUReservationRequest, SiteHeartbeat
from topodrome.scheduler.service import SchedulerService

pytestmark = pytest.mark.component


@pytest.mark.asyncio
async def test_idle_permission_is_fresh_site_scoped_and_reservation_aware(database, tmp_path):
    service = SchedulerService(database=database, settings=SchedulerSettings(
        database_dsn="sqlite+aiosqlite://", global_root=tmp_path / "global",
        fence_signing_key="idle-compute-permission-fence-key-0000",
    ))

    def heartbeat(site, sequence, *, requested=True, state="ready"):
        model = SiteHeartbeat(
            site_id=site, site_kind=SiteKind.LAB, agent_session_id="session",
            sequence=sequence, state=state, idle_compute_requested=requested,
            gpus=(GPUInventory(
                uuid="GPU-A", index=0, name="A800", total_memory_bytes=80 * 1024**3,
                free_memory_bytes=80 * 1024**3, utilization_percent=0,
            ),),
        )
        return AgentEnvelope(site_id=site, agent_session_id="session", sequence=sequence,
                             kind="heartbeat", payload=model.model_dump(mode="json"))

    ordinary = await service.handle_agent_envelope(heartbeat("ordinary", 1, requested=False))
    assert "idle_compute_permission" not in ordinary.payload
    initial = heartbeat("sqz2", 1)
    ack = await service.handle_agent_envelope(initial)
    assert ack.payload["idle_compute_permission"]["allowed"] is True
    assert ack.payload["idle_compute_permission"]["blocked_gpu_uuids"] == []
    async with database.transaction() as session:
        await service.repository.create_gpu_reservation(session, GPUReservationRequest(
            site_id="sqz2", gpu_uuid="GPU-A", duration_seconds=1800, reason="test",
        ), requested_by="operator")
    ack = await service.handle_agent_envelope(heartbeat("sqz2", 2))
    assert ack.payload["idle_compute_permission"]["blocked_gpu_uuids"] == ["GPU-A"]
    ack = await service.handle_agent_envelope(heartbeat("ordinary", 2))
    assert ack.payload["idle_compute_permission"]["blocked_gpu_uuids"] == []
    replay = await service.handle_agent_envelope(initial)
    assert "idle_compute_permission" not in replay.payload
    drained = await service.handle_agent_envelope(heartbeat("sqz2", 3, state="draining"))
    assert drained.payload["idle_compute_permission"]["allowed"] is False
