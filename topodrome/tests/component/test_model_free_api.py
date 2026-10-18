from __future__ import annotations

import base64
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest

from topodrome.client.async_client import AsyncClient
from topodrome.config import SchedulerSettings
from topodrome.db.repositories import SchedulerRepository
from topodrome.domain.specs import GPUInventory, JobSpec, SiteHeartbeat, utc_now
from topodrome.scheduler.app import create_app
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
from topodrome.scheduler.placement import choose_placement
from topodrome.scheduler.service import SchedulerService
from topodrome.scheduler.source_store import SourceBundle

pytestmark = pytest.mark.component


async def test_null_model_sdk_arrays_status_and_catalog_bypass(database, tmp_path):
    app = create_app(
        SchedulerSettings(global_root=tmp_path / "global", fence_signing_key="f" * 32),
        database=database,
        run_dispatcher=False,
    )

    def unexpected_catalog_lookup(*_):
        pytest.fail("model-free submission must not resolve a model catalog")

    app.state.service.model_catalog = SimpleNamespace(resolve=unexpected_catalog_lookup)
    async with AsyncClient("http://test", transport=httpx.ASGITransport(app=app)) as client:
        source = await client._http.post(
            "/api/v1/sources",
            json={
                "files": {
                    "run.py": base64.b64encode(b"def run(context): pass\n").decode(),
                }
            },
        )
        assert source.status_code == 201
        manifest = {
            "entrypoint": "run.py",
            "model": None,
            "execution": "distributed",
            "source_ref": source.json()["source_ref"],
            "resources": {"gpu": 2, "min_vram_gb": 2},
        }
        submitted = await client.submit(**manifest, idempotency_key="null-job")
        replayed = await client.submit(**manifest, idempotency_key="null-job")
        assert submitted.job_id == replayed.job_id
        detail = await client.status(submitted.job_id)
        assert detail.spec.model is None and detail.spec.execution_mode == "distributed"
        array = await client.submit_array(
            job=manifest,
            elements=[{"index": 0}, {"index": 1}],
            max_parallel=2,
        )
        for member in (await client.array(array["array_id"]))["members"]:
            child = await client.status(member["job_id"])
            assert child.spec.model is None and child.spec.resources.gpu_count == 2
        frame = (await client._http.get("/api/v1/status")).json()
        row = next(row for row in frame["jobs"] if row["job_id"] == submitted.job_id)
        assert row["model"] is None and row["execution"] == "distributed"


@pytest.mark.parametrize("capable", [False, True])
async def test_model_free_agent_capability_is_durable_and_fail_closed(database, capable):
    repo = SchedulerRepository()
    heartbeat = SiteHeartbeat(
        site_id="test",
        site_kind="lab",
        agent_session_id="s",
        sequence=1,
        storage_available=True,
        supports_model_free=capable,
        gpus=tuple(
            GPUInventory(
                uuid=f"GPU-{index}",
                index=index,
                name="GPU",
                utilization_percent=0,
                total_memory_bytes=32 * 1024**3,
                free_memory_bytes=31 * 1024**3,
            )
            for index in range(2)
        ),
    )
    async with database.transaction() as session:
        await repo.upsert_heartbeat(session, heartbeat)
    async with database.session() as session:
        candidates = await repo.placement_inventory(session)
    wanted = JobSpec(entrypoint="run.py", model=None, resources={"gpu": 2, "min_vram_gb": 2})
    result = choose_placement(
        wanted, candidates, heartbeat_cutoff=utc_now() - timedelta(seconds=30)
    )
    assert (result is not None) is capable
    # An older replacement agent heartbeat must revoke the capability too.
    if capable:
        async with database.transaction() as session:
            await repo.upsert_heartbeat(
                session,
                heartbeat.model_copy(
                    update={
                        "sequence": 2,
                        "supports_model_free": False,
                    }
                ),
            )
        async with database.session() as session:
            candidates = await repo.placement_inventory(session)
        assert "model_free" not in candidates[0].supported_workloads


@pytest.mark.usefixtures("pre_staged_inputs")
@pytest.mark.parametrize("gpus,execution", [(1, None), (2, "single"), (2, "distributed")])
async def test_model_free_dispatch_with_adaptive_costs_and_no_model_catalog(
    database,
    tmp_path,
    gpus,
    execution,
):
    deliveries = []

    async def capture(message):
        deliveries.append(message)

    connections = AgentConnectionRegistry()
    connection = AgentConnection("test", "s", capture)
    await connections.register(connection)
    repo = SchedulerRepository()
    heartbeat = SiteHeartbeat(
        site_id="test",
        site_kind="lab",
        agent_session_id="s",
        sequence=1,
        storage_available=True,
        supports_model_free=True,
        gpus=tuple(
            GPUInventory(
                uuid=f"GPU-{index}",
                index=index,
                name="GPU",
                utilization_percent=0,
                total_memory_bytes=32 * 1024**3,
                free_memory_bytes=31 * 1024**3,
            )
            for index in range(gpus)
        ),
    )
    async with database.transaction() as session:
        await repo.upsert_heartbeat(session, heartbeat, connection_generation=connection.generation)
    service = SchedulerService(
        database=database,
        repository=repo,
        connections=connections,
        settings=SchedulerSettings(global_root=tmp_path / "global", fence_signing_key="f" * 32),
    )
    source_ref = service.sources.put(
        SourceBundle(
            files={
                "run.py": base64.b64encode(b"def run(context): pass\n").decode(),
            }
        )
    ).source_ref
    await service.submit(
        JobSpec(
            entrypoint="run.py",
            model=None,
            execution=execution,
            source_ref=source_ref,
            resources={"gpu": gpus, "min_vram_gb": 2},
        )
    )
    assert await service.dispatch_once()
    assert len(deliveries) == 1
    assignment = deliveries[0]["payload"]
    assert assignment["spec"]["model"] is None
    assert len(assignment["claims"]) == gpus
