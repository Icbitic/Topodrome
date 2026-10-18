from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from topodrome.config import SchedulerSettings
from topodrome.db.models import ProviderInstanceRow, SiteRow
from topodrome.db.repositories import SchedulerRepository
from topodrome.db.session import Database
from topodrome.domain.enums import (
    ModelAvailability,
    ProviderInstanceState,
    SiteKind,
    SiteState,
)
from topodrome.domain.events import AgentEnvelope
from topodrome.domain.specs import GPUInventory, ModelCapability, SiteHeartbeat
from topodrome.providers.autodl.lifecycle import (
    OperationKind,
    OperationState,
    ProviderOperationIntent,
)
from topodrome.scheduler.autodl import (
    SchedulerAutoDLBridge,
    build_autodl_orchestrator,
)
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
from topodrome.scheduler.provider_controller import SQLProviderOperationStore

pytestmark = pytest.mark.unit
GIB = 1024**3


def _heartbeat(
    *,
    sent_at: datetime | None = None,
    gpu_count: int = 1,
    **activity: bool,
) -> SiteHeartbeat:
    return SiteHeartbeat(
        site_id="autodl",
        site_kind=SiteKind.AUTODL,
        provider_instance_id="pro-1",
        region="westDC3",
        agent_session_id="agent-session",
        sequence=1,
        gpus=tuple(
            GPUInventory(
                uuid=f"gpu-{index + 1}",
                index=index,
                name="RTX 5090",
                total_memory_bytes=32 * GIB,
                free_memory_bytes=31 * GIB,
                utilization_percent=0,
            )
            for index in range(gpu_count)
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
        sent_at=sent_at or datetime.now(UTC),
        **activity,
    )


async def _seed_site(
    database: Database,
    heartbeat: SiteHeartbeat | None = None,
    *,
    connection_generation: str | None = None,
) -> None:
    async with database.transaction() as session:
        await SchedulerRepository().upsert_heartbeat(
            session,
            heartbeat or _heartbeat(),
            connection_generation=connection_generation,
        )


async def test_provider_fallback_repairs_missing_heartbeat_identity_and_region(
    database: Database,
) -> None:
    heartbeat = _heartbeat().model_copy(update={"provider_instance_id": None, "region": None})
    async with database.transaction() as session:
        session.add(
            ProviderInstanceRow(
                id="provider-row-identity",
                provider="autodl",
                external_id="pro-provider-truth",
                site_id="autodl",
                state=ProviderInstanceState.RUNNING,
                region="beijingDC2",
                gpu_spec="5090-p",
                image_uuid="image-cu128",
            )
        )
        await SchedulerRepository().upsert_heartbeat(session, heartbeat)
    bridge = SchedulerAutoDLBridge(
        database,
        AgentConnectionRegistry(),
        heartbeat_ttl_seconds=30,
        acknowledgement_timeout_seconds=1,
    )

    await bridge.associate_heartbeat(heartbeat)

    async with database.session() as session:
        site = await session.get(SiteRow, "autodl")
        provider = await session.get(ProviderInstanceRow, "provider-row-identity")
    assert site is not None
    assert site.provider_instance_id == "pro-provider-truth"
    assert site.region == "beijingDC2"
    assert provider is not None and provider.region == "beijingDC2"


async def test_provider_site_identity_requires_the_full_atomic_gpu_inventory(
    database: Database,
) -> None:
    heartbeat = _heartbeat(gpu_count=1)
    async with database.transaction() as session:
        session.add(
            ProviderInstanceRow(
                id="provider-row-two-gpu",
                provider="autodl",
                external_id="pro-1",
                site_id="autodl",
                state=ProviderInstanceState.STARTING,
                region="westDC3",
                gpu_spec="5090-p",
                image_uuid="image-cu128",
                metadata_json={"requested_gpu_count": 2},
            )
        )
        await SchedulerRepository().upsert_heartbeat(session, heartbeat)
    bridge = SchedulerAutoDLBridge(
        database,
        AgentConnectionRegistry(),
        heartbeat_ttl_seconds=30,
        acknowledgement_timeout_seconds=1,
    )

    await bridge.associate_heartbeat(heartbeat)

    async with database.session() as session:
        site = await session.get(SiteRow, "autodl")
        provider = await session.get(ProviderInstanceRow, "provider-row-two-gpu")
    assert site is not None and site.state is SiteState.ERROR
    assert provider is not None and provider.state is ProviderInstanceState.STARTING


async def test_provider_site_identity_accepts_exact_two_gpu_inventory(
    database: Database,
) -> None:
    heartbeat = _heartbeat(gpu_count=2)
    async with database.transaction() as session:
        session.add(
            ProviderInstanceRow(
                id="provider-row-two-gpu-ready",
                provider="autodl",
                external_id="pro-1",
                site_id="autodl",
                state=ProviderInstanceState.STARTING,
                region="westDC3",
                gpu_spec="5090-p",
                image_uuid="image-cu128",
                metadata_json={"requested_gpu_count": 2},
            )
        )
        await SchedulerRepository().upsert_heartbeat(session, heartbeat)
    bridge = SchedulerAutoDLBridge(
        database,
        AgentConnectionRegistry(),
        heartbeat_ttl_seconds=30,
        acknowledgement_timeout_seconds=1,
    )

    await bridge.associate_heartbeat(heartbeat)

    async with database.session() as session:
        site = await session.get(SiteRow, "autodl")
        provider = await session.get(
            ProviderInstanceRow,
            "provider-row-two-gpu-ready",
        )
    assert site is not None and site.state is SiteState.READY
    assert site.provider_instance_id == "pro-1"
    assert provider is not None and provider.state is ProviderInstanceState.RUNNING
    assert provider.metadata_json["startup_heartbeat_seen"] is True
    assert provider.metadata_json["startup_deadline"] is None


async def test_sql_operation_store_is_durable_before_mutation(database: Database) -> None:
    store = SQLProviderOperationStore(database)
    intent = ProviderOperationIntent(
        operation_id="operation-1",
        idempotency_key="job:one:power-on",
        kind=OperationKind.POWER_ON,
        request={"instance_uuid": "pro-1", "payload": "gpu"},
        instance_uuid="pro-1",
    )

    claim = await store.claim(intent)

    assert claim.created
    assert [record.intent.operation_id for record in await store.unresolved()] == ["operation-1"]
    await store.mark_succeeded(
        "operation-1",
        provider_request_id="request-1",
        result_instance_uuid="pro-1",
    )
    assert await store.unresolved() == ()
    history = await store.records()
    assert len(history) == 1
    assert history[0].state is OperationState.SUCCEEDED
    assert history[0].result_instance_uuid == "pro-1"


async def test_sql_operation_store_allows_only_one_pre_call_intent_owner(
    database: Database,
) -> None:
    store = SQLProviderOperationStore(database)
    intent = ProviderOperationIntent(
        operation_id="operation-race",
        idempotency_key="job:race:power-on",
        kind=OperationKind.POWER_ON,
        request={"instance_uuid": "pro-1", "payload": "gpu"},
        instance_uuid="pro-1",
    )
    await store.claim(intent)

    owners = await asyncio.gather(
        store.claim_intent_for_call(intent.operation_id),
        store.claim_intent_for_call(intent.operation_id),
    )

    assert sorted(owners) == [False, True]


@pytest.mark.parametrize(
    "signal",
    [
        "job_running",
        "assignment_queued_locally",
        "interactive_kernel_active",
        "artifact_flush_pending",
        "staging_operation_active",
    ],
)
async def test_bridge_maps_each_reported_activity_signal(
    database: Database,
    signal: str,
) -> None:
    heartbeat = _heartbeat().model_copy(update={signal: True})
    connections = AgentConnectionRegistry()

    async def discard(_: dict[str, object]) -> None:
        return None

    connection = AgentConnection("autodl", "agent-session", discard)
    await connections.register(connection)
    await _seed_site(
        database,
        heartbeat,
        connection_generation=connection.generation,
    )
    async with database.transaction() as session:
        session.add(
            ProviderInstanceRow(
                id="provider-1",
                provider="autodl",
                external_id="pro-1",
                site_id="autodl",
                state=ProviderInstanceState.RUNNING,
                gpu_spec="5090-p",
                image_uuid="image-cu128",
            )
        )
    bridge = SchedulerAutoDLBridge(
        database,
        connections,
        heartbeat_ttl_seconds=30,
        acknowledgement_timeout_seconds=1,
    )

    activity = await bridge.current_activity("pro-1")

    assert not activity.is_idle
    assert getattr(activity, signal)


async def test_stale_telemetry_never_counts_as_idle(database: Database) -> None:
    await _seed_site(database, _heartbeat(sent_at=datetime.now(UTC) - timedelta(minutes=2)))
    async with database.transaction() as session:
        session.add(
            ProviderInstanceRow(
                id="provider-1",
                provider="autodl",
                external_id="pro-1",
                site_id="autodl",
                state=ProviderInstanceState.RUNNING,
                gpu_spec="5090-p",
                image_uuid="image-cu128",
            )
        )
    bridge = SchedulerAutoDLBridge(
        database,
        AgentConnectionRegistry(),
        heartbeat_ttl_seconds=30,
        acknowledgement_timeout_seconds=1,
    )

    activity = await bridge.current_activity("pro-1")

    assert activity.staging_operation_active
    assert not activity.is_idle


async def test_power_off_retry_requires_exact_durable_stopped_agent_generation(
    database: Database,
) -> None:
    connections = AgentConnectionRegistry()

    async def discard(_: dict[str, object]) -> None:
        return None

    original = AgentConnection("autodl", "agent-session", discard)
    await connections.register(original)
    await _seed_site(database, connection_generation=original.generation)
    async with database.transaction() as session:
        site = await session.get(SiteRow, "autodl", with_for_update=True)
        assert site is not None
        site.state = SiteState.DRAINING
        session.add(
            ProviderInstanceRow(
                id="provider-power-off-retry",
                provider="autodl",
                external_id="pro-retry",
                site_id="autodl",
                state=ProviderInstanceState.STOPPING,
                gpu_spec="5090-p",
                image_uuid="image-cu128",
                durable_state_confirmed=True,
                shutdown_generation=4,
                shutdown_agent_session_id="agent-session",
                shutdown_connection_generation=original.generation,
                drain_command_nonce="drain-nonce-that-is-long-enough",
                drain_ack_nonce="drain-nonce-that-is-long-enough",
                durability_command_nonce="flush-nonce-that-is-long-enough",
                durability_ack_nonce="flush-nonce-that-is-long-enough",
            )
        )
    bridge = SchedulerAutoDLBridge(
        database,
        connections,
        heartbeat_ttl_seconds=30,
        acknowledgement_timeout_seconds=1,
    )

    assert not await bridge.power_off_retry_authorized("pro-retry")
    await connections.unregister(
        "autodl",
        "agent-session",
        generation=original.generation,
    )
    assert await bridge.power_off_retry_authorized("pro-retry")

    replacement = AgentConnection("autodl", "replacement-session", discard)
    await connections.register(replacement)
    assert not await bridge.power_off_retry_authorized("pro-retry")


async def test_shutdown_ack_is_bound_to_generation_nonce_and_agent_connection(
    database: Database,
) -> None:
    connections = AgentConnectionRegistry()
    commands: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def capture(message: dict[str, object]) -> None:
        await commands.put(dict(message))

    connection = AgentConnection("autodl", "agent-session", capture)
    await connections.register(connection)
    await _seed_site(database, connection_generation=connection.generation)
    async with database.transaction() as session:
        session.add(
            ProviderInstanceRow(
                id="provider-replay",
                provider="autodl",
                external_id="pro-replay",
                site_id="autodl",
                state=ProviderInstanceState.RUNNING,
                gpu_spec="5090-p",
                image_uuid="image-cu128",
            )
        )
    bridge = SchedulerAutoDLBridge(
        database,
        connections,
        heartbeat_ttl_seconds=30,
        acknowledgement_timeout_seconds=0.5,
        poll_interval_seconds=0,
    )

    first_drain = asyncio.create_task(bridge.mark_draining("pro-replay"))
    first_command = await commands.get()
    assert first_command["kind"] == "begin_drain"
    first_coordinates = dict(first_command["payload"])

    async def readiness(
        *,
        kind: str,
        payload: dict[str, object],
        session_id: str = "agent-session",
        connection_generation: str = connection.generation,
        site_id: str = "autodl",
    ) -> None:
        await bridge.note_agent_event(
            AgentEnvelope(
                site_id=site_id,
                agent_session_id=session_id,
                sequence=1,
                kind=kind,
                payload=payload,
            ),
            connection_generation=connection_generation,
        )

    tampered = {
        **first_coordinates,
        "command_nonce": "tampered-nonce-that-is-long-enough",
    }
    await readiness(kind="drain_ready", payload=tampered)
    await readiness(
        kind="drain_ready",
        payload=first_coordinates,
        session_id="displaced-session",
    )
    await readiness(
        kind="drain_ready",
        payload=first_coordinates,
        connection_generation="displaced-connection-generation",
    )
    await readiness(kind="drain_ready", payload=first_coordinates, site_id="other-site")
    await asyncio.sleep(0)
    assert not first_drain.done()

    await readiness(kind="drain_ready", payload=first_coordinates)
    await first_drain

    first_flush = asyncio.create_task(bridge.flush_and_verify("pro-replay"))
    durability_command = await commands.get()
    assert durability_command["kind"] == "verify_durability"
    durability_coordinates = dict(durability_command["payload"])
    assert durability_coordinates["shutdown_generation"] == first_coordinates["shutdown_generation"]
    assert durability_coordinates["command_nonce"] != first_coordinates["command_nonce"]

    # A drain ACK and a wrong stage nonce cannot satisfy the durability gate.
    await readiness(kind="durability_ready", payload=first_coordinates)
    await readiness(kind="durability_ready", payload=tampered)
    await asyncio.sleep(0)
    assert not first_flush.done()
    await readiness(kind="durability_ready", payload=durability_coordinates)
    assert await first_flush
    assert await bridge.persist_regional_state("pro-replay")
    assert await bridge.is_durable("pro-replay")

    second_drain = asyncio.create_task(bridge.mark_draining("pro-replay"))
    second_command = await commands.get()
    second_coordinates = dict(second_command["payload"])
    assert int(second_coordinates["shutdown_generation"]) == (
        int(first_coordinates["shutdown_generation"]) + 1
    )

    # Delayed valid events from generation one cannot authorize generation two.
    await readiness(kind="drain_ready", payload=first_coordinates)
    await readiness(kind="durability_ready", payload=durability_coordinates)
    await asyncio.sleep(0)
    assert not second_drain.done()
    assert not await bridge.persist_regional_state("pro-replay")
    async with database.session() as session:
        row = await session.scalar(
            select(ProviderInstanceRow).where(ProviderInstanceRow.external_id == "pro-replay")
        )
        site = await session.get(SiteRow, "autodl")
        assert row is not None and not row.durable_state_confirmed
        assert row.drain_ack_nonce is None
        assert site is not None and not site.drain_ready
        assert not site.durability_ready

    await readiness(kind="drain_ready", payload=second_coordinates)
    await second_drain


def test_disabled_provider_does_not_read_secret_file(tmp_path: Path) -> None:
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path,
        fence_signing_key="f" * 64,
        secret_file=tmp_path / "missing.env",
    )
    database = Database(settings.database_dsn)

    result = build_autodl_orchestrator(settings, database, AgentConnectionRegistry())

    assert result is None
