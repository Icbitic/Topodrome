from dataclasses import replace

import pytest
from sqlalchemy import func, select

from tests.unit.test_job_arrays import array_service as _array_service
from tests.unit.test_site_agent_admission import StaticInventory, gpu
from topodrome.config import ModelCatalog
from topodrome.db.models import JobArrayRow, JobRow
from topodrome.domain.arrays import JobArraySpec
from topodrome.domain.errors import AdmissionRejectedError
from topodrome.domain.specs import JobSpec
from topodrome.site_agent.admission import GIB, AdmissionController, LeaseFenceStore, LeaseRequest
from topodrome.site_agent.inventory import GPUProcess, ProcessOwner

array_service = _array_service


def test_child_inheritance_null_resources_and_sdk_roundtrip():
    request = JobArraySpec.model_validate(
        {
            "job": {
                "entrypoint": "run.py",
                "model": "test/base",
                "resources": {"gpu": 1, "min_vram_gb": 16, "cpu": 3, "memory_gb": 6},
            },
            "elements": [
                {"index": 0},
                {"index": 1, "model": {"repo_id": "test/other", "revision": "step1000"}},
                {"index": 2, "model": None, "resources": {"gpu": 2, "min_vram_gb": 4}},
            ],
        }
    )
    for _ in range(3):
        encoded = request.model_dump(mode="json")
        assert "model" not in encoded["elements"][0]
        assert encoded["elements"][2]["model"] is None
        assert encoded["elements"][2]["resources"] == {"gpu": 2, "min_vram_gb": 4}
        request = JobArraySpec.model_validate(encoded)
    first, second, third = [request.element_spec(element) for element in request.elements]
    assert first.model.repo_id == "test/base"
    assert second.model.repo_id == "test/other" and second.model.revision == "step1000"
    assert third.model is None and third.resources.gpu == 2
    assert third.resources.cpu == 3 and third.resources.memory_gb == 6
    assert third.kwargs == {"array_index": 2}


@pytest.mark.parametrize(
    "element",
    [
        {"index": 1, "resources": None},
        {"index": 1, "model": "test/other", "resources": {"gpu": 2}},
        {"index": 1, "model": None, "resources": {"min_vram_gb": 0}},
        {"index": 1, "model": {"repo_id": "test/other", "revision": "../bad"}},
    ],
)
def test_invalid_override_rejected_before_submission(element):
    with pytest.raises(ValueError):
        JobArraySpec(job={"entrypoint": "run.py", "model": "test/base"}, elements=[element])


def catalogue():
    return ModelCatalog.model_validate(
        {
            "models": {
                "test/model": {
                    "repo_id": "test/model",
                    "revision": "a" * 40,
                    "refs": ["main"],
                    "min_vram_gb": 16,
                    "sites": {},
                    "snapshots": [{"revision": "b" * 40, "refs": ["step1000"], "sites": {}}],
                },
                "other/model": {
                    "repo_id": "other/model",
                    "revision": "c" * 40,
                    "min_vram_gb": 24,
                    "sites": {},
                },
            }
        }
    )


async def test_mixed_array_pins_each_child_once_and_replays_atomically(
    array_service, database, monkeypatch
):
    service, base, _ = array_service
    service.model_catalog = catalogue()
    base = JobSpec.model_validate(
        {**base.model_dump(), "model": "test/model", "resources": {"gpu": 1}}
    )
    request = JobArraySpec(
        job=base,
        elements=[
            {"index": 0},
            {"index": 1, "model": {"repo_id": "test/model", "revision": "step1000"}},
            {"index": 2, "model": "other/model"},
            {"index": 3, "model": None, "resources": {"min_vram_gb": 3}},
        ],
    )
    pins = 0
    original = service._pin_assets

    async def pin(*args, **kwargs):
        nonlocal pins
        pins += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(service, "_pin_assets", pin)
    result = await service.submit_array(request, owner_username="alice", idempotency_key="mixed")
    assert pins == 1
    async with database.session() as session:
        rows = list(
            await session.scalars(
                select(JobRow)
                .where(JobRow.array_id == result["array_id"])
                .order_by(JobRow.array_index)
            )
        )
    specs = [JobSpec.model_validate(row.spec) for row in rows]
    assert [spec.model.revision if spec.model else None for spec in specs] == [
        "a" * 40,
        "b" * 40,
        "c" * 40,
        None,
    ]
    assert [spec.resources.min_vram_gb for spec in specs] == [16, 16, 24, 3]
    assert len({spec.source_ref for spec in specs}) == 1
    service.model_catalog = None  # Replays must not re-resolve defaults or inputs.
    repeated = await service.submit_array(request, owner_username="alice", idempotency_key="mixed")
    assert repeated["array_id"] == result["array_id"] and pins == 1


async def test_unknown_last_child_does_not_create_partial_array(array_service, database):
    service, base, _ = array_service
    service.model_catalog = catalogue()
    base = base.model_copy(update={"model": JobSpec(entrypoint="run.py", model="test/model").model})
    request = JobArraySpec(
        job=base, elements=[{"index": 0}, {"index": 1, "model": "missing/model"}]
    )
    with pytest.raises(AdmissionRejectedError):
        await service.submit_array(request)
    async with database.session() as session:
        assert await session.scalar(select(func.count()).select_from(JobRow)) == 0
        assert await session.scalar(select(func.count()).select_from(JobArrayRow)) == 0


async def test_prefetch_hint_does_not_admit_or_reserve_a_sibling(array_service, database):
    service, base, messages = array_service
    request = JobArraySpec(
        job=base,
        max_parallel=1,
        elements=[
            {"index": 0},
            {"index": 1, "model": {"repo_id": "test/model", "revision": "next"}},
        ],
    )
    result = await service.submit_array(request)
    assert await service.dispatch_once()
    payload = messages[-1]["payload"]
    assert payload["next_model"]["revision"] == "next"
    assert not await service.dispatch_once()
    from topodrome.scheduler.arrays import array_detail

    detail = await array_detail(service, result["array_id"])
    assert detail["active_compute"] == 1 and detail["counts"]["queued"] == 1


def switch_request():
    return LeaseRequest(
        "next",
        "attempt-next",
        ("GPU-A",),
        {"GPU-A": 1},
        20 * GIB,
        4_000_000_000,
        model_key="new",
        replace_idle_actors=True,
        runtime_switch_actor_id="old-actor",
        runtime_switch_model_key="old",
        runtime_switch_process_ids=(7,),
    )


def test_switch_claim_keeps_actor_identity_and_updates_model_only_after_prepare(tmp_path):
    store = LeaseFenceStore(tmp_path / "lease.sqlite3")
    store.reserve_actor(gpu_uuid="GPU-A", actor_id="old-actor", model_key="old", state="idle")
    snapshot = gpu(
        "GPU-A", free_gib=16, processes=(GPUProcess(7, 16 * GIB, ProcessOwner.TOPODROME),)
    )
    admission = AdmissionController(
        StaticInventory([snapshot]), store, warm_actor_validator=lambda _: True
    )
    decision = admission.admit(switch_request())
    assert decision.accepted and decision.reused_actor_id is None
    assert store.actor_allocation("GPU-A").model_key == "old"
    assert store.actor_allocation("GPU-A").state == "busy"
    store.reserve_actor(gpu_uuid="GPU-A", actor_id="old-actor", model_key="new", state="busy")
    assert admission.admit(switch_request()).accepted  # Replay after identity publication.
    admission.release("next")
    assert store.actor_allocation("GPU-A").model_key == "new"
    assert store.actor_allocation("GPU-A").state == "idle"
    store.close()


@pytest.mark.parametrize(
    "change", ["partial", "foreign", "unknown-memory", "wrong-pid", "oversized", "stale"]
)
def test_switch_admission_never_bypasses_physical_or_identity_safety(tmp_path, change):
    store = LeaseFenceStore(tmp_path / "lease.sqlite3")
    uuids = ("GPU-A", "GPU-B") if change == "partial" else ("GPU-A",)
    store.reserve_actor_group(gpu_uuids=uuids, actor_id="old-actor", model_key="old", state="idle")
    owner = ProcessOwner.FOREIGN if change == "foreign" else ProcessOwner.TOPODROME
    memory = None if change == "unknown-memory" else 16 * GIB
    pid = 99 if change == "wrong-pid" else 7
    snapshot = gpu("GPU-A", free_gib=16, processes=(GPUProcess(pid, memory, owner),))
    request = switch_request()
    if change == "oversized":
        request = replace(request, min_free_vram_bytes=40 * GIB)
    if change == "stale":
        request = replace(request, runtime_switch_model_key="other")
    admission = AdmissionController(
        StaticInventory([snapshot]), store, warm_actor_validator=lambda _: True
    )
    assert not admission.admit(request).accepted
    assert not store.active_claims()
    assert store.actor_allocation("GPU-A").state == "idle"
    store.close()
