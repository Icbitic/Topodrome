from datetime import timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from tests.unit.test_scheduler_dispatch import _heartbeat, _source_ref
from tests.unit.test_scheduler_repository import heartbeat, job_spec
from topodrome.config import InputPlacementSettings, SchedulerSettings
from topodrome.db.models import InputStageRow, LeaseGPUClaimRow, SchedulingMeasurementRow
from topodrome.db.repositories import SchedulerRepository
from topodrome.domain.enums import JobState, SiteKind
from topodrome.domain.events import AgentEnvelope
from topodrome.domain.specs import AssetMember, AssetRef, InputStageMetrics, utc_now
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
from topodrome.scheduler.estimates import (
    history,
    model_context,
    observe,
    remaining_seconds,
    startup_estimates,
    worthwhile_switch,
)
from topodrome.scheduler.input_staging import prepare_inputs, receive_stage, retarget_inputs
from topodrome.scheduler.service import SchedulerService
from topodrome.site_agent.input_staging import InputStager
from topodrome.storage.assets import asset_manifest_digest
from topodrome.storage.progress import transfer_progress

pytestmark = pytest.mark.unit
MIB = 1024**2


def asset(size=1024 * MIB):
    member = AssetMember(
        logical_path="artifacts/uploads/test/files/a",
        relative_path="a",
        size_bytes=size,
        sha256="a" * 64,
    )
    return AssetRef(
        source_ref="asset://shared/input@v1",
        kind="file",
        members=(member,),
        digest=asset_manifest_digest((member,)),
        total_bytes=size,
    )


@pytest.mark.asyncio
async def test_history_persists_clips_outliers_expires_and_is_model_specific(database):
    repo = SchedulerRepository()
    spec = job_spec()
    now = utc_now()
    key = model_context(spec)
    async with database.transaction() as session:
        await repo.upsert_heartbeat(session, heartbeat())
        await observe(session, "lab1", "startup", key, 10, now=now)
        await observe(session, "lab1", "startup", key, 10000, now=now)
        await observe(session, "lab1", "startup", key, float("nan"), now=now)
    async with database.session() as session:
        values = await history(session, spec, ["lab1"], InputPlacementSettings(), now=now)
        assert values[("lab1", "startup", key)] == pytest.approx(19)
        row = await session.get(SchedulingMeasurementRow, ("lab1", "startup", key))
        assert row.samples == 2
        sites = [SimpleNamespace(site_id="lab1")]
        assert startup_estimates(spec, sites, {"lab1": 120}, values) == {"lab1": 19}
        other = spec.model_copy(
            update={"model": spec.model.model_copy(update={"dtype": "float32"})}
        )
        assert not await history(session, other, ["lab1"], InputPlacementSettings(), now=now)
        assert not await history(
            session, spec, ["lab1"], InputPlacementSettings(), now=now + timedelta(days=8)
        )


def test_remaining_cost_uses_live_speed_remaining_bytes_and_local_authority():
    now = utc_now()
    spec = job_spec().model_copy(update={"assets": {"input": asset()}})
    policy = InputPlacementSettings()
    site = SimpleNamespace(site_id="sqz", cached_assets=(), storage_authoritative=False)
    placement = SimpleNamespace(estimated_ready_seconds=2)
    stage = SimpleNamespace(
        site_id="sqz",
        state="downloading",
        placement_data={
            "metrics_at": now.timestamp(),
            "metrics": {
                "remaining_bytes": 10 * MIB,
                "downloaded_bytes": 20 * MIB,
                "download_seconds": 10,
            },
        },
    )
    old = {("sqz", "download_bps", ""): MIB / 100}
    assert remaining_seconds(spec, site, placement, old, policy, stage, now=now) < 13
    site.storage_authoritative = True
    assert remaining_seconds(spec, site, placement, {}, policy, now=now) < 10
    stage.state = "ready"
    assert remaining_seconds(spec, site, placement, {}, policy, stage, now=now) == 2
    site.storage_authoritative = False
    stage.state = "downloading"
    stage.placement_data["metrics_at"] -= 40
    assert remaining_seconds(spec, site, placement, old, policy, stage, now=now) > 100000


def test_switch_thresholds_and_durable_cooldown():
    settings = InputPlacementSettings()
    now = utc_now()
    row = SimpleNamespace(placement_data={})
    assert worthwhile_switch(120, 10, row, settings, now=now)
    assert not worthwhile_switch(35, 10, row, settings, now=now)
    assert not worthwhile_switch(1000, 900, row, settings, now=now)
    assert worthwhile_switch(None, 10, row, settings, now=now)
    row.placement_data = {"last_switch_at": now.timestamp()}
    assert not worthwhile_switch(1000, 10, row, settings, now=now)
    assert not worthwhile_switch(None, 10, row, settings, now=now)
    assert worthwhile_switch(1000, 10, row, settings, now=now + timedelta(seconds=61))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_metrics_rejected(value):
    with pytest.raises(ValidationError):
        InputStageMetrics(download_seconds=value)


async def setup_scheduler(database, tmp_path, *, affine=False, enabled=True):
    registry = AgentConnectionRegistry()
    messages = {"lab1": [], "sqz": []}
    connections = {}
    service = SchedulerService(
        database=database,
        connections=registry,
        settings=SchedulerSettings(
            global_root=tmp_path,
            fence_signing_key="f" * 64,
            scheduling={"input_placement": {"enabled": enabled}},
        ),
    )
    for site in messages:

        async def send(message, site=site):
            messages[site].append(message)

        connections[site] = AgentConnection(site, site + "-session", send)
        await registry.register(connections[site])
    async with database.transaction() as session:
        for site in messages:
            hb = _heartbeat(site, SiteKind.LAB, blocking_foreign_process=site == "lab1")
            hb = hb.model_copy(update={"storage_authoritative": site == "lab1"})
            await service.repository.upsert_heartbeat(
                session, hb, connection_generation=connections[site].generation
            )
        spec = job_spec().model_copy(
            update={
                "assets": {"input": asset()},
                "source_ref": _source_ref(service),
                "affinity_site_id": "sqz" if affine else None,
            }
        )
        job = await service.repository.submit_job(session, spec)
    return service, messages, connections, job


async def free_lab1(database, service, connections):
    async with database.transaction() as session:
        hb = _heartbeat("lab1", SiteKind.LAB).model_copy(
            update={"sequence": 2, "storage_authoritative": True}
        )
        await service.repository.upsert_heartbeat(
            session, hb, connection_generation=connections["lab1"].generation
        )


@pytest.mark.asyncio
async def test_lab1_frees_mid_transfer_retargets_without_duplicate_attempt(database, tmp_path):
    service, messages, connections, job = await setup_scheduler(database, tmp_path)
    assert not await service.dispatch_once()
    old = messages["sqz"][-1]["payload"]
    await free_lab1(database, service, connections)
    assert not await service.dispatch_once()
    assert messages["sqz"][-1]["kind"] == "cancel_input_stage"
    new = messages["lab1"][-1]["payload"]
    assert new["stage_id"] != old["stage_id"]
    async with database.transaction() as session:
        assert not await session.scalar(select(func.count()).select_from(LeaseGPUClaimRow))
        row = await session.get(InputStageRow, job.id)
        assert row.site_id == "lab1" and row.placement_data["last_switch_at"]
        # Delayed old-site reports can neither publish readiness nor train history.
        await receive_stage(
            session,
            "sqz",
            {
                **old,
                "state": "ready",
                "progress": {"current": 0},
                "metrics": {"downloaded_bytes": MIB, "download_seconds": 2},
            },
        )
        assert row.state == "downloading"
        assert not await session.scalar(select(func.count()).select_from(SchedulingMeasurementRow))
        await receive_stage(session, "lab1", {**new, "state": "ready", "progress": {"current": 0}})
    assert await service.dispatch_once()
    assert messages["lab1"][-1]["kind"] == "offer_assignment"
    async with database.session() as session:
        row = await service.repository.get_job(session, job.id)
        assert row.state is not JobState.QUEUED
        assert await session.scalar(select(func.count()).select_from(LeaseGPUClaimRow)) == 1
    counts = {s: len(m) for s, m in messages.items()}
    await service.dispatch_once()
    assert counts == {s: len(m) for s, m in messages.items()}


@pytest.mark.asyncio
@pytest.mark.parametrize("affine,enabled", [(True, True), (False, False)])
async def test_no_switch_when_affinity_or_disabled(database, tmp_path, affine, enabled):
    service, messages, connections, _ = await setup_scheduler(
        database, tmp_path, affine=affine, enabled=enabled
    )
    await service.dispatch_once()
    await free_lab1(database, service, connections)
    await service.dispatch_once()
    assert not messages["lab1"]
    assert not any(m["kind"] == "cancel_input_stage" for m in messages["sqz"])


@pytest.mark.asyncio
async def test_full_destination_keeps_old_stage_and_does_not_cancel(database, tmp_path):
    service, _messages, _connections, job = await setup_scheduler(database, tmp_path)
    await service.dispatch_once()
    async with database.transaction() as session:
        row = await session.get(InputStageRow, job.id)
        generation = row.stage_id
        for _ in range(2):
            other = await service.repository.submit_job(session, job_spec())
            await prepare_inputs(session, other, "lab1")
        assert await retarget_inputs(session, job, row, "lab1") is None
        assert row.site_id == "sqz" and row.stage_id == generation


@pytest.mark.asyncio
async def test_ready_stage_near_completion_not_needlessly_moved(database, tmp_path):
    service, messages, connections, _job = await setup_scheduler(database, tmp_path)
    await service.dispatch_once()
    old = messages["sqz"][-1]["payload"]
    async with database.transaction() as session:
        await receive_stage(session, "sqz", {**old, "state": "ready", "progress": {"current": 0}})
    await free_lab1(database, service, connections)
    assert await service.dispatch_once()
    assert messages["sqz"][-1]["kind"] == "offer_assignment"
    assert not messages["lab1"]


@pytest.mark.asyncio
async def test_stage_network_metrics_exclude_resumed_bytes_and_survive_clock_skew(monkeypatch):
    import topodrome.site_agent.input_staging as module

    clock = [100.0]
    # Replace only this module's clock namespace; never patch asyncio's global clock.
    monkeypatch.setattr(
        module, "time", SimpleNamespace(time=lambda: 10000, monotonic=lambda: clock[0])
    )
    recorded = []

    def resolve(ref):
        transfer_progress("downloading", 4 * MIB, 10 * MIB)
        clock[0] += 3
        transfer_progress("downloading", 10 * MIB, 10 * MIB)
        transfer_progress("download-complete", 10 * MIB, 10 * MIB)
        transfer_progress("input-member-ready", 10 * MIB, 10 * MIB)

    async def publish(kind, payload):
        recorded.append(payload)

    stager = InputStager(
        SimpleNamespace(pin=lambda *a: None, unpin=lambda *a: None, resolve=resolve),
        SimpleNamespace(publish=publish),
    )
    await stager.accept(
        {
            "job_id": "j",
            "stage_id": "s",
            "assets": {"a": asset(10 * MIB).model_dump()},
            "expires_at": "2000-01-01T00:00:00Z",
            "ttl_seconds": 120,
        }
    )
    await stager.stages["j"].task
    await stager.tick()
    assert recorded[-1]["metrics"] == {
        "remaining_bytes": 0,
        "downloaded_bytes": 6 * MIB,
        "download_seconds": 3,
    }
    await stager.close()


@pytest.mark.asyncio
async def test_fenced_startup_measurements_deduplicate_replayed_and_repeated_states(
    database, tmp_path
):
    service, messages, connections, _job = await setup_scheduler(database, tmp_path, affine=True)
    await service.dispatch_once()
    old = messages["sqz"][-1]["payload"]
    async with database.transaction() as session:
        await receive_stage(session, "sqz", {**old, "state": "ready", "progress": {"current": 0}})
    assert await service.dispatch_once()
    offer = messages["sqz"][-1]["payload"]
    identity = {k: offer[k] for k in ("job_id", "attempt_id", "fencing_token")}

    async def send(seq, kind, payload):
        envelope = AgentEnvelope(
            site_id="sqz", agent_session_id="sqz-session", sequence=seq, kind=kind, payload=payload
        )
        await service.handle_agent_envelope(
            envelope, connection_generation=connections["sqz"].generation
        )
        return envelope

    await send(10, "assignment_accepted", identity)
    await send(11, "attempt_event", {**identity, "site_id": "sqz", "state": "starting"})
    running = await send(
        12,
        "attempt_event",
        {**identity, "site_id": "sqz", "state": "running", "timings": {"startup_seconds": 2}},
    )
    await service.handle_agent_envelope(
        running, connection_generation=connections["sqz"].generation
    )
    await send(
        13,
        "attempt_event",
        {**identity, "site_id": "sqz", "state": "running", "timings": {"startup_seconds": 200}},
    )
    async with database.session() as session:
        rows = list(await session.scalars(select(SchedulingMeasurementRow)))
        assert len(rows) == 1 and rows[0].value == 2 and rows[0].samples == 1
        assert rows[0].metric == "startup"


@pytest.mark.asyncio
async def test_ready_bandwidth_measurements_only_once(database, tmp_path):
    service, messages, _connections, _job = await setup_scheduler(database, tmp_path, affine=True)
    await service.dispatch_once()
    old = messages["sqz"][-1]["payload"]
    payload = {
        **old,
        "state": "ready",
        "progress": {"current": 0},
        "metrics": {"downloaded_bytes": 4 * MIB, "download_seconds": 2},
    }
    async with database.transaction() as session:
        await receive_stage(session, "sqz", payload)
        await receive_stage(session, "sqz", payload)
    async with database.session() as session:
        rows = list(await session.scalars(select(SchedulingMeasurementRow)))
        assert len(rows) == 1 and rows[0].value == 2 * MIB and rows[0].samples == 1
