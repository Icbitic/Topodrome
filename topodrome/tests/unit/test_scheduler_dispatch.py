from __future__ import annotations

import base64
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from topodrome.config import ModelCatalog, SchedulerSettings
from topodrome.db.repositories import SchedulerRepository
from topodrome.db.session import Database
from topodrome.domain.enums import ModelAvailability, SiteKind
from topodrome.domain.specs import (
    AssignmentOffer,
    GPUInventory,
    GPUProcess,
    JobSpec,
    ModelCapability,
    SiteHeartbeat,
    utc_now,
)
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
from topodrome.scheduler.service import SchedulerService
from topodrome.scheduler.source_store import SourceBundle

GIB = 1024**3
pytestmark = pytest.mark.usefixtures("pre_staged_inputs")


def _source_ref(service: SchedulerService) -> str:
    return service.sources.put(
        SourceBundle(
            files={"experiment.py": base64.b64encode(b"def run(context): return None\n").decode()}
        )
    ).source_ref


class FakeProviderController:
    def __init__(
        self,
        *,
        ensure_result: bool = True,
        capacity_pending: bool = False,
    ) -> None:
        self.demands: list[tuple[str, JobSpec]] = []
        self.ensure_result = ensure_result
        self.capacity_pending = capacity_pending

    def supports_demand(self, spec: JobSpec) -> bool:
        return spec.affinity_site_id in {None, "autodl"}

    async def ensure_for_demand(self, job_id: str, spec: JobSpec) -> bool:
        self.demands.append((job_id, spec))
        return self.ensure_result

    async def capacity_pending_for_demand(self, spec: JobSpec) -> bool:
        return self.capacity_pending

    async def reconcile_once(self) -> Any:
        return None

    async def observe_agent_event(
        self,
        envelope: Any,
        *,
        connection_generation: str | None = None,
    ) -> None:
        return None

    async def instance_summaries(self) -> list[dict[str, Any]]:
        return []

    async def operation_summaries(self) -> list[dict[str, Any]]:
        return []

    async def close(self) -> None:
        return None


def _heartbeat(
    site_id: str,
    kind: SiteKind,
    *,
    gpu_name: str = "NVIDIA A800-SXM4-80GB",
    free_gib: int = 25,
    blocking_foreign_process: bool = False,
) -> SiteHeartbeat:
    return SiteHeartbeat(
        site_id=site_id,
        site_kind=kind,
        agent_session_id=f"{site_id}-session",
        sequence=1,
        gpus=(
            GPUInventory(
                uuid=f"{site_id}-gpu",
                index=0,
                name=gpu_name,
                total_memory_bytes=32 * GIB,
                free_memory_bytes=free_gib * GIB,
                utilization_percent=0,
                processes=(
                    (GPUProcess(pid=1234, used_memory_bytes=GIB),)
                    if blocking_foreign_process
                    else ()
                ),
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
    )


def _burst_heartbeat(
    site_id: str,
    kind: SiteKind,
    *,
    free_gib: tuple[int, ...],
    foreign_gib: tuple[int, ...],
    warm_gpu_indices: frozenset[int],
    runtime_fingerprint: str,
) -> SiteHeartbeat:
    assert len(free_gib) == len(foreign_gib)
    gpus = tuple(
        GPUInventory(
            uuid=f"{site_id}-gpu-{index}",
            index=index,
            name="NVIDIA A800-SXM4-80GB",
            total_memory_bytes=80 * GIB,
            free_memory_bytes=free * GIB,
            utilization_percent=0,
            processes=(
                (
                    GPUProcess(
                        pid=10_000 + index,
                        used_memory_bytes=foreign * GIB,
                    ),
                )
                if foreign
                else ()
            ),
        )
        for index, (free, foreign) in enumerate(zip(free_gib, foreign_gib, strict=True))
    )
    if warm_gpu_indices:
        models = tuple(
            ModelCapability(
                repo_id="test/model",
                revision="rev",
                backend="transformers",
                backend_version="configured",
                dtype="bfloat16",
                availability=ModelAvailability.WARM,
                gpu_uuid=f"{site_id}-gpu-{index}",
                runtime_fingerprint=runtime_fingerprint,
            )
            for index in sorted(warm_gpu_indices)
        )
    else:
        models = (
            ModelCapability(
                repo_id="test/model",
                revision="rev",
                backend="transformers",
                backend_version="configured",
                dtype="bfloat16",
                availability=ModelAvailability.CACHED,
            ),
        )
    return SiteHeartbeat(
        site_id=site_id,
        site_kind=kind,
        agent_session_id=f"{site_id}-session",
        sequence=1,
        gpus=gpus,
        models=models,
        gpu_safety_margin_bytes=GIB,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connect_autodl", "expected_site"),
    [(True, "lab1"), (False, "lab1")],
)
async def test_dispatch_prefers_connected_institutional_capacity_over_paid(
    database: Database,
    tmp_path: Path,
    connect_autodl: bool,
    expected_site: str,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, Any]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(dict(message))

    lab_connection = AgentConnection("lab1", "lab1-session", capture)
    await connections.register(lab_connection)
    autodl_connection: AgentConnection | None = None
    if connect_autodl:
        autodl_connection = AgentConnection("autodl", "autodl-session", capture)
        await connections.register(autodl_connection)
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            _heartbeat("lab1", SiteKind.LAB),
            connection_generation=lab_connection.generation,
        )
        await repository.upsert_heartbeat(
            session,
            _heartbeat("autodl", SiteKind.AUTODL),
            connection_generation=(
                autodl_connection.generation if autodl_connection is not None else None
            ),
        )

    provider = FakeProviderController()
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
        ),
        connections=connections,
        repository=repository,
        provider_controller=provider,
    )
    await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        )
    )

    assert await service.dispatch_once()
    assert len(deliveries) == 1
    assert deliveries[0]["site_id"] == expected_site
    assert provider.demands == []


async def test_cancel_uses_rolling_compatible_bounded_stop_command(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, Any]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(dict(message))

    connection = AgentConnection("lab1", "lab1-session", capture)
    await connections.register(connection)
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            _heartbeat("lab1", SiteKind.LAB),
            connection_generation=connection.generation,
        )
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
        ),
        connections=connections,
        repository=repository,
    )
    submitted = await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        )
    )

    assert await service.dispatch_once()
    offer = AssignmentOffer.model_validate(deliveries[-1]["payload"])
    await service.cancel(submitted.job_id)

    assert deliveries[-1]["kind"] == "request_checkpoint"
    assert deliveries[-1]["payload"]["attempt_id"] == offer.attempt_id
    assert deliveries[-1]["payload"]["cancel_requested"] is True


async def test_burst_spills_from_crowded_warm_lab_slots_to_roomy_jupyter(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, Any]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(dict(message))

    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
            scheduling={
                "fast_gpu_name_patterns": [r"^NVIDIA A800-SXM4-80GB$"],
                "foreign_capacity_sharing_site_ids": ["lab1", "remote-jupyter-1"],
            },
        ),
        connections=connections,
        repository=repository,
    )
    job_spec = JobSpec(
        entrypoint="experiment.py",
        source_ref=_source_ref(service),
        model={"repo_id": "test/model", "revision": "rev"},
        resources={"gpu": 1, "min_vram_gb": 20},
    )
    for site_id, heartbeat in (
        (
            "lab1",
            _burst_heartbeat(
                "lab1",
                SiteKind.LAB,
                # Two roomy clean warm actors, followed by two warm actors
                # sharing GPUs with large foreign allocations.
                free_gib=(64, 64, 38, 38),
                foreign_gib=(0, 0, 24, 24),
                warm_gpu_indices=frozenset(range(4)),
                runtime_fingerprint=job_spec.runtime_fingerprint,
            ),
        ),
        (
            "remote-jupyter-1",
            _burst_heartbeat(
                "remote-jupyter-1",
                SiteKind.JUPYTER,
                # These remain usable under explicit capacity sharing and have
                # room for at least two requested footprints apiece.
                free_gib=(64, 64, 64, 64),
                foreign_gib=(15, 15, 15, 15),
                warm_gpu_indices=frozenset(),
                runtime_fingerprint=job_spec.runtime_fingerprint,
            ),
        ),
    ):
        connection = AgentConnection(site_id, f"{site_id}-session", capture)
        await connections.register(connection)
        async with database.transaction() as session:
            await repository.upsert_heartbeat(
                session,
                heartbeat,
                connection_generation=connection.generation,
            )

    for _ in range(4):
        await service.submit(job_spec)
    for _ in range(4):
        assert await service.dispatch_once()

    assert [delivery["site_id"] for delivery in deliveries] == [
        "lab1",
        "lab1",
        "remote-jupyter-1",
        "remote-jupyter-1",
    ]


async def test_dispatch_uses_catalog_site_model_startup_estimates(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, Any]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(dict(message))

    for site_id, kind, free_gib in (
        ("lab1", SiteKind.LAB, 24),
        ("remote-jupyter-1", SiteKind.JUPYTER, 64),
    ):
        connection = AgentConnection(site_id, f"{site_id}-session", capture)
        await connections.register(connection)
        async with database.transaction() as session:
            await repository.upsert_heartbeat(
                session,
                _heartbeat(site_id, kind, free_gib=free_gib),
                connection_generation=connection.generation,
            )

    catalog = ModelCatalog.model_validate(
        {
            "models": {
                "test/model": {
                    "repo_id": "test/model",
                    "revision": "rev",
                    "sites": {
                        "lab1": {"path": "/models/model", "estimated_startup_seconds": 2},
                        "remote-jupyter-1": {
                            "path": "/models/model",
                            "estimated_startup_seconds": 120,
                        },
                    },
                }
            }
        }
    )
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
        ),
        connections=connections,
        repository=repository,
        model_catalog=catalog,
    )
    await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        )
    )

    assert await service.dispatch_once()
    assert deliveries[-1]["site_id"] == "lab1"


async def test_local_rejection_quarantines_exact_gpu_until_next_heartbeat(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, Any]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(dict(message))

    site_connections: dict[str, AgentConnection] = {}
    for site_id, kind, free_gib in (
        ("lab1", SiteKind.LAB, 30),
        ("remote-jupyter-1", SiteKind.JUPYTER, 25),
    ):
        connection = AgentConnection(site_id, f"{site_id}-session", capture)
        site_connections[site_id] = connection
        await connections.register(connection)
        async with database.transaction() as session:
            await repository.upsert_heartbeat(
                session,
                _heartbeat(site_id, kind, free_gib=free_gib),
                connection_generation=connection.generation,
            )

    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
        ),
        connections=connections,
        repository=repository,
    )
    job = await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        )
    )

    assert await service.dispatch_once()
    assert deliveries[-1]["site_id"] == "lab1"
    offer = AssignmentOffer.model_validate(deliveries[-1]["payload"])
    async with database.transaction() as session:
        await repository.reject_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id=offer.site_id,
            token=offer.fencing_token,
            authority=service.fencing,
            reason_code="local_admission_rejected",
            message="physical capacity changed after the heartbeat",
        )
    async with database.session() as session:
        inventory = {site.site_id: site for site in await repository.placement_inventory(session)}
    assert inventory["lab1"].gpus[0].admission_blocked
    assert not inventory["remote-jupyter-1"].gpus[0].admission_blocked

    # The same portable job immediately spills instead of selecting the stale
    # Lab1 snapshot in a tight rejection loop.
    assert await service.dispatch_once()
    assert deliveries[-1]["site_id"] == "remote-jupyter-1"
    assert (await service.status(job.job_id)).state.value == "assigning"

    # A fresh Lab1 receipt bounds the quarantine. It is not an external
    # reservation and cannot stick after new physical telemetry arrives.
    refreshed = _heartbeat("lab1", SiteKind.LAB, free_gib=30).model_copy(update={"sequence": 2})
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            refreshed,
            connection_generation=site_connections["lab1"].generation,
            received_at=utc_now() + timedelta(seconds=1),
        )
    async with database.session() as session:
        inventory = {site.site_id: site for site in await repository.placement_inventory(session)}
    assert not inventory["lab1"].gpus[0].admission_blocked

    await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        )
    )
    assert await service.dispatch_once()
    assert deliveries[-1]["site_id"] == "lab1"


async def test_dispatch_uses_active_autodl_only_when_institutional_capacity_is_ineligible(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, Any]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(dict(message))

    for site_id in ("lab1", "autodl"):
        connection = AgentConnection(site_id, f"{site_id}-session", capture)
        await connections.register(connection)
        heartbeat = _heartbeat(
            site_id,
            SiteKind.AUTODL if site_id == "autodl" else SiteKind.LAB,
            free_gib=25 if site_id == "autodl" else 14,
        )
        async with database.transaction() as session:
            await repository.upsert_heartbeat(
                session,
                heartbeat,
                connection_generation=connection.generation,
            )

    provider = FakeProviderController()
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
            scheduling={"fast_gpu_name_patterns": [r"A800"]},
        ),
        connections=connections,
        repository=repository,
        provider_controller=provider,
    )
    await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        )
    )

    assert await service.dispatch_once()
    assert len(deliveries) == 1
    assert deliveries[0]["site_id"] == "autodl"
    assert provider.demands == []


async def test_dispatch_foreign_blocker_on_jupyter_cannot_mask_active_autodl(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, Any]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(dict(message))

    for site_id, kind, blocking_foreign_process in (
        ("remote-jupyter-1", SiteKind.JUPYTER, True),
        ("autodl", SiteKind.AUTODL, False),
    ):
        connection = AgentConnection(site_id, f"{site_id}-session", capture)
        await connections.register(connection)
        async with database.transaction() as session:
            await repository.upsert_heartbeat(
                session,
                _heartbeat(
                    site_id,
                    kind,
                    blocking_foreign_process=blocking_foreign_process,
                ),
                connection_generation=connection.generation,
            )

    provider = FakeProviderController()
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
            scheduling={"foreign_capacity_sharing_site_ids": ["lab1"]},
        ),
        connections=connections,
        repository=repository,
        provider_controller=provider,
    )
    await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
            workload_kind="interactive",
            kwargs={"mode": "raw-model"},
        )
    )

    assert await service.dispatch_once()
    assert [delivery["site_id"] for delivery in deliveries] == ["autodl"]
    assert provider.demands == []


async def test_dispatch_lab1_foreign_capacity_sharing_is_explicitly_opted_in(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, Any]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(dict(message))

    for site_id, kind, blocking_foreign_process in (("lab1", SiteKind.LAB, True),):
        connection = AgentConnection(site_id, f"{site_id}-session", capture)
        await connections.register(connection)
        async with database.transaction() as session:
            await repository.upsert_heartbeat(
                session,
                _heartbeat(
                    site_id,
                    kind,
                    blocking_foreign_process=blocking_foreign_process,
                ),
                connection_generation=connection.generation,
            )

    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
            scheduling={"foreign_capacity_sharing_site_ids": ["lab1"]},
        ),
        connections=connections,
        repository=repository,
    )
    await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        )
    )

    assert await service.dispatch_once()
    assert [delivery["site_id"] for delivery in deliveries] == ["lab1"]


async def test_dispatch_applies_configured_fast_gpu_names_before_normal_scoring(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, Any]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(dict(message))

    site_data = (
        ("aa-fast-jupyter", SiteKind.JUPYTER, "NVIDIA A800-SXM4-80GB"),
        ("zz-slow-lab", SiteKind.LAB, "NVIDIA Tesla T4"),
    )
    for site_id, kind, gpu_name in site_data:
        connection = AgentConnection(site_id, f"{site_id}-session", capture)
        await connections.register(connection)
        async with database.transaction() as session:
            await repository.upsert_heartbeat(
                session,
                _heartbeat(site_id, kind, gpu_name=gpu_name),
                connection_generation=connection.generation,
            )

    provider = FakeProviderController()
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
            scheduling={"fast_gpu_name_patterns": [r"^NVIDIA A800-SXM4-80GB$"]},
        ),
        connections=connections,
        repository=repository,
        provider_controller=provider,
    )
    await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        )
    )

    assert await service.dispatch_once()
    assert len(deliveries) == 1
    assert deliveries[0]["site_id"] == "aa-fast-jupyter"
    assert provider.demands == []


async def test_unplaced_demand_stays_queued_without_renting(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()

    async def discard(_message: dict[str, object]) -> None:
        return None

    connection = AgentConnection("lab1", "lab1-session", discard)
    await connections.register(connection)
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            _heartbeat("lab1", SiteKind.LAB, free_gib=14),
            connection_generation=connection.generation,
        )

    provider = FakeProviderController()
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
        ),
        connections=connections,
        repository=repository,
        provider_controller=provider,
    )
    submitted = await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        )
    )

    assert not await service.dispatch_once()
    assert provider.demands == []
    assert (await service.status(submitted.job_id)).state.value == "queued"


async def test_unplaced_job_never_requests_provider_capacity(
    database: Database,
    tmp_path: Path,
) -> None:
    provider = FakeProviderController()
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
        ),
        provider_controller=provider,
    )

    assert not await service.dispatch_once()
    assert provider.demands == []

    submitted = await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        )
    )
    assert not await service.dispatch_once()

    assert provider.demands == []
    assert (await service.status(submitted.job_id)).state.value == "queued"


async def test_unplaceable_autodl_job_does_not_starve_institutional_work(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, Any]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(dict(message))

    connection = AgentConnection("lab1", "lab1-session", capture)
    await connections.register(connection)
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            _heartbeat("lab1", SiteKind.LAB),
            connection_generation=connection.generation,
        )

    provider = FakeProviderController()
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
        ),
        connections=connections,
        repository=repository,
        provider_controller=provider,
    )
    blocked = await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
            affinity_site_id="autodl",
            priority=20,
        )
    )
    selected = await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
            priority=10,
        )
    )
    lower_priority = await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
            priority=0,
        )
    )

    assert await service.dispatch_once()
    assert len(deliveries) == 1
    assert deliveries[0]["payload"]["job_id"] == selected.job_id
    assert provider.demands == []
    assert (await service.status(blocked.job_id)).state.value == "queued"
    assert (await service.status(lower_priority.job_id)).state.value == "queued"


@pytest.mark.parametrize("estimates", [False, True])
async def test_dispatch_pages_past_more_than_one_batch_of_unplaceable_jobs(
    database: Database,
    tmp_path: Path,
    estimates: bool,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, Any]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(dict(message))

    connection = AgentConnection("lab1", "lab1-session", capture)
    await connections.register(connection)
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            _heartbeat("lab1", SiteKind.LAB),
            connection_generation=connection.generation,
        )

    provider = FakeProviderController()
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
        ),
        connections=connections,
        repository=repository,
        provider_controller=provider,
    )
    source_ref = _source_ref(service)
    blocked = [
        await service.submit(
            JobSpec(
                entrypoint="experiment.py",
                source_ref=source_ref,
                model={"repo_id": "test/model", "revision": "rev"},
                resources={"gpu": 1, "min_vram_gb": 15},
                affinity_site_id="autodl",
                priority=0 if estimates else 10,
                estimated_runtime_seconds=index + 1 if estimates else None,
            )
        )
        for index in range(33)
    ]
    selected = await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=source_ref,
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
            priority=0,
            estimated_runtime_seconds=200 if estimates else None,
        )
    )

    assert await service.dispatch_once()
    assert len(deliveries) == 1
    assert deliveries[0]["payload"]["job_id"] == selected.job_id
    assert provider.demands == []
    blocked_states = [(await service.status(job.job_id)).state.value for job in blocked]
    assert set(blocked_states) == {"queued"}


async def test_replaced_agent_inventory_is_not_combined_with_new_command_lane(
    database: Database,
    tmp_path: Path,
) -> None:
    repository = SchedulerRepository()
    connections = AgentConnectionRegistry()
    old_deliveries: list[dict[str, Any]] = []
    new_deliveries: list[dict[str, Any]] = []

    async def old_send(message: dict[str, object]) -> None:
        old_deliveries.append(dict(message))

    async def new_send(message: dict[str, object]) -> None:
        new_deliveries.append(dict(message))

    old = AgentConnection("lab1", "lab1-session", old_send)
    await connections.register(old)
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            _heartbeat("lab1", SiteKind.LAB),
            connection_generation=old.generation,
        )

    replacement = AgentConnection("lab1", "replacement-session", new_send)
    await connections.register(replacement)
    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
        ),
        connections=connections,
        repository=repository,
    )
    await service.submit(
        JobSpec(
            entrypoint="experiment.py",
            source_ref=_source_ref(service),
            model={"repo_id": "test/model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
        )
    )

    # The DB inventory belongs to the displaced connection.  It cannot be
    # selected merely because another authenticated socket now serves lab1.
    assert not await service.dispatch_once()
    assert old_deliveries == []
    assert new_deliveries == []

    heartbeat = _heartbeat("lab1", SiteKind.LAB).model_copy(
        update={"agent_session_id": "replacement-session"}
    )
    async with database.transaction() as session:
        await repository.upsert_heartbeat(
            session,
            heartbeat,
            connection_generation=replacement.generation,
        )

    assert await service.dispatch_once()
    assert old_deliveries == []
    assert len(new_deliveries) == 1


async def test_heartbeat_asset_locality_is_persisted_for_placement(
    database: Database,
) -> None:
    repository = SchedulerRepository()
    heartbeat = _heartbeat("lab1", SiteKind.LAB).model_copy(
        update={"cached_assets": {"a" * 64, "b" * 64}}
    )
    async with database.transaction() as session:
        await repository.upsert_heartbeat(session, heartbeat)
    async with database.session() as session:
        sites = await repository.placement_inventory(session)

    assert len(sites) == 1
    assert sites[0].cached_assets == frozenset({"a" * 64, "b" * 64})
