from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from topodrome.domain.errors import StaleFenceError
from topodrome.domain.specs import AssignmentOffer, FenceClaim, JobSpec
from topodrome.protocol.fencing import FencingAuthority
from topodrome.site_agent.admission import (
    GIB,
    ActorAllocation,
    AdmissionController,
    LeaseFenceStore,
    LeaseRequest,
)
from topodrome.site_agent.agent import (
    Assignment,
    PreparedRuntime,
    SiteAgent,
    _finish_input_staging,
)
from topodrome.site_agent.inventory import (
    GPUProcess,
    GPUReservation,
    GPUSnapshot,
    HostSnapshot,
    ProcessOwner,
    ProcessOwnershipRegistry,
)

pytestmark = pytest.mark.unit


def fenced_assignment(token: str) -> Assignment:
    return Assignment(
        job_id="job-fenced",
        attempt_id="attempt-fenced",
        lease=LeaseRequest(
            lease_id="lease-fenced",
            attempt_id="attempt-fenced",
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 7},
            min_free_vram_bytes=1,
            expires_at=4_000_000_000,
            model_key="model-key",
        ),
        model_key="model-key",
        entrypoint="experiment.py",
        fencing_token=token,
    )


@pytest.mark.asyncio
async def test_input_staging_finishes_before_attempt_cancellation_propagates() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def stage() -> None:
        started.set()
        await release.wait()
        finished.set()

    attempt = asyncio.create_task(_finish_input_staging(stage()))
    await started.wait()
    attempt.cancel()
    await asyncio.sleep(0)

    assert not attempt.done()
    assert not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    assert finished.is_set()


def test_assignment_accepts_shared_domain_offer_shape_without_exposing_token() -> None:
    offer = AssignmentOffer(
        message_id="offer-1",
        job_id="job-1",
        attempt_id="attempt-1",
        attempt_number=1,
        site_id="lab1",
        spec=JobSpec(
            entrypoint="experiments/probe.py",
            model={
                "alias": "meta-llama-3-8b-instruct",
                "revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
                "backend": "transformers",
                "backend_version": "4.48",
                "dtype": "bfloat16",
            },
            resources={"gpu": 1, "min_vram_gb": 15, "cpu": 3.5, "memory_gb": 6},
        ),
        claims=(FenceClaim(gpu_uuid="GPU-A", epoch=9),),
        fencing_token=secrets.token_hex(16),
        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
        replace_idle_actors=True,
    )

    assignment = Assignment.from_mapping(offer.model_dump(mode="python"))

    assert assignment.lease.gpu_uuids == ("GPU-A",)
    assert assignment.lease.fence_epochs == {"GPU-A": 9}
    assert assignment.lease.min_free_vram_bytes == 15 * 1024**3
    assert assignment.lease.cpu_cores == 3.5
    assert assignment.lease.memory_bytes == 6 * 1024**3
    assert assignment.lease.replace_idle_actors
    assert assignment.entrypoint == "experiments/probe.py"
    assert "meta-llama-3-8b-instruct" in assignment.model_key
    assert json.loads(assignment.model_key)["runtime_fingerprint"] == (
        offer.spec.runtime_fingerprint
    )
    assert assignment.lease.lease_id.startswith("lease_")
    assert offer.fencing_token not in repr(assignment)


@pytest.mark.asyncio
async def test_shutdown_disconnects_driver_before_exact_head_and_is_single_flight() -> None:
    events: list[str] = []

    class Backend:
        async def shutdown_driver(self) -> bool:
            events.append("driver-disconnected")
            await asyncio.sleep(0)
            return True

    class AdoptedRayLifecycle:
        async def stop(self) -> bool:
            # False is the lifecycle's proof that an adopted head was left
            # running. The site agent must still finish its own shutdown.
            events.append("adopted-head-preserved")
            return False

    async def publish_drain() -> None:
        events.append("drain-published")

    agent = object.__new__(SiteAgent)
    agent.backend = cast(Any, Backend())
    agent.ray = cast(Any, AdoptedRayLifecycle())
    agent._active = {}
    agent._active_lock = asyncio.Lock()
    agent._shutdown_lock = asyncio.Lock()
    agent._stop = asyncio.Event()
    agent._draining = False
    agent._publish_drain_if_ready = publish_drain  # type: ignore[method-assign]

    await asyncio.gather(
        agent.shutdown(stop_ray=True),
        agent.shutdown(stop_ray=True),
    )

    assert events == [
        "drain-published",
        "driver-disconnected",
        "adopted-head-preserved",
    ]
    assert agent._stop.is_set()


@pytest.mark.asyncio
async def test_shutdown_never_stops_head_when_driver_disconnect_is_unconfirmed() -> None:
    events: list[str] = []

    class Backend:
        async def shutdown_driver(self) -> bool:
            events.append("driver-disconnect-failed")
            raise RuntimeError("disconnect unconfirmed")

    class RayLifecycle:
        async def stop(self) -> bool:
            events.append("unsafe-head-stop")
            return True

    async def publish_drain() -> None:
        events.append("drain-published")

    agent = object.__new__(SiteAgent)
    agent.backend = cast(Any, Backend())
    agent.ray = cast(Any, RayLifecycle())
    agent._active = {}
    agent._active_lock = asyncio.Lock()
    agent._shutdown_lock = asyncio.Lock()
    agent._stop = asyncio.Event()
    agent._draining = False
    agent._publish_drain_if_ready = publish_drain  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="disconnect unconfirmed"):
        await agent.shutdown(stop_ray=True)

    assert events == ["drain-published", "driver-disconnect-failed"]
    assert not agent._stop.is_set()


def test_adapter_training_assignment_disables_warm_actor_reuse() -> None:
    offer = AssignmentOffer(
        message_id="offer-adapter",
        job_id="job-adapter",
        attempt_id="attempt-adapter",
        attempt_number=1,
        site_id="lab1",
        spec=JobSpec(
            entrypoint="experiments/train_adapter.py",
            model={"alias": "model", "revision": "rev"},
            resources={"gpu": 1, "min_vram_gb": 15},
            workload_kind="adapter_training",
        ),
        claims=(FenceClaim(gpu_uuid="GPU-A", epoch=9),),
        fencing_token=secrets.token_hex(16),
        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
    )

    assignment = Assignment.from_mapping(offer.model_dump(mode="python"))

    assert not assignment.lease.allow_actor_reuse


def test_multi_gpu_assignment_disables_single_actor_reuse() -> None:
    offer = AssignmentOffer(
        message_id="offer-multi",
        job_id="job-multi",
        attempt_id="attempt-multi",
        attempt_number=1,
        site_id="lab1",
        spec=JobSpec(
            entrypoint="experiments/distributed.py",
            model={
                "alias": "model",
                "revision": "rev",
                "parallelism": {
                    "mode": "distributed",
                    "degree": 2,
                    "engine": "ray_worker_group",
                },
            },
            resources={"gpu": 2, "min_vram_gb": 15},
        ),
        claims=(
            FenceClaim(gpu_uuid="GPU-A", epoch=9),
            FenceClaim(gpu_uuid="GPU-B", epoch=10),
        ),
        fencing_token=secrets.token_hex(16),
        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
    )

    assignment = Assignment.from_mapping(offer.model_dump(mode="python"))

    assert not assignment.lease.allow_actor_reuse


def test_model_sharded_assignment_allows_only_complete_group_reuse() -> None:
    offer = AssignmentOffer(
        message_id="offer-sharded",
        job_id="job-sharded",
        attempt_id="attempt-sharded",
        attempt_number=1,
        site_id="lab1",
        spec=JobSpec(
            entrypoint="experiments/inference.py",
            model={
                "alias": "model",
                "revision": "rev",
                "parallelism": {
                    "mode": "model_sharded",
                    "degree": 2,
                    "max_memory_gb_per_gpu": 10,
                },
            },
            resources={"gpu": 2, "min_vram_gb": 10},
        ),
        claims=(
            FenceClaim(gpu_uuid="GPU-A", epoch=9),
            FenceClaim(gpu_uuid="GPU-B", epoch=10),
        ),
        fencing_token=secrets.token_hex(16),
        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
    )

    assignment = Assignment.from_mapping(offer.model_dump(mode="python"))

    assert assignment.lease.allow_actor_reuse
    assert assignment.lease.gpu_uuids == ("GPU-A", "GPU-B")


@pytest.mark.parametrize(
    "claims",
    [
        ({"gpu_uuid": "GPU-A", "epoch": 9},),
        (
            {"gpu_uuid": "GPU-A", "epoch": 9},
            {"gpu_uuid": "GPU-B", "epoch": 10},
            {"gpu_uuid": "GPU-C", "epoch": 11},
        ),
    ],
)
def test_model_sharded_assignment_rejects_incomplete_or_oversized_claims_before_local_claim(
    tmp_path: Path,
    claims: tuple[dict[str, object], ...],
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    payload = {
        "message_id": "offer-malformed-gang",
        "job_id": "job-sharded",
        "attempt_id": "attempt-sharded",
        "attempt_number": 1,
        "site_id": "lab1",
        "spec": {
            "entrypoint": "experiments/inference.py",
            "model": {
                "alias": "model",
                "revision": "rev",
                "parallelism": {
                    "mode": "model_sharded",
                    "degree": 2,
                    "max_memory_gb_per_gpu": 10,
                },
            },
            "resources": {"gpu": 2, "min_vram_gb": 10},
        },
        "claims": claims,
        "fencing_token": secrets.token_hex(16),
        "expires_at": datetime(2099, 1, 1, tzinfo=UTC),
    }

    with pytest.raises(ValueError, match="GPU claim count"):
        Assignment.from_mapping(payload)

    assert store.active_claims() == {}
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_spec", "kwargs"),
    [
        ({"workload_kind": "adapter_training"}, {}),
        ({"workload_kind": "interactive"}, {"mode": "raw-model"}),
    ],
)
async def test_private_workload_exactly_evicts_idle_batch_actor_before_admission(
    tmp_path: Path,
    raw_spec: dict[str, str],
    kwargs: dict[str, str],
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="batch-actor",
        model_key="batch-runtime",
        state="idle",
    )
    allocation = store.actor_allocation("GPU-A")
    assert allocation is not None

    class Backend:
        def __init__(self) -> None:
            self.evicted: list[str] = []

        def is_warm_allocation(self, candidate: ActorAllocation) -> bool:
            return candidate == allocation

        async def evict_warm_allocation(
            self,
            candidate: ActorAllocation,
        ) -> tuple[int, ...]:
            self.evicted.append(candidate.actor_id)
            return (8123,)

    backend = Backend()
    ownership = ProcessOwnershipRegistry(proc_root=tmp_path / "proc")
    ownership.register(8123, actor_id="batch-actor")

    class Inventory:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self) -> tuple[GPUSnapshot, ...]:
            self.calls += 1
            lingering = self.calls == 1
            if lingering:
                # Exact ownership must remain registered until the physical
                # process/VRAM accounting has disappeared from NVML.
                assert ownership.exact_owner(8123) is not None
            return (
                GPUSnapshot(
                    uuid="GPU-A",
                    index=0,
                    name="GPU",
                    total_memory_bytes=32 * GIB,
                    free_memory_bytes=(20 if lingering else 32) * GIB,
                    used_memory_bytes=(12 if lingering else 0) * GIB,
                    utilization_percent=0,
                    processes=(
                        (GPUProcess(8123, 12 * GIB, ProcessOwner.TOPODROME),) if lingering else ()
                    ),
                ),
            )

    inventory = Inventory()
    agent = object.__new__(SiteAgent)
    agent.backend = cast(Any, backend)
    agent.fences = store
    agent.process_ownership = ownership
    agent.inventory = inventory  # type: ignore[assignment]
    agent.admission = cast(
        Any,
        SimpleNamespace(
            safety_margin_bytes=0,
            max_unattributed_utilization_percent=10,
            allow_foreign_processes_when_capacity_sufficient=False,
        ),
    )
    assignment = Assignment(
        job_id="private-job",
        attempt_id="private-attempt",
        lease=LeaseRequest(
            lease_id="private-lease",
            attempt_id="private-attempt",
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 2},
            min_free_vram_bytes=1,
            expires_at=4_000_000_000,
            allow_actor_reuse=False,
        ),
        model_key="private-runtime",
        entrypoint="experiment.py",
        kwargs=kwargs,
        raw_spec=raw_spec,
    )

    evicted_gpu_uuids = await agent._evict_private_slot_actors(assignment)

    assert backend.evicted == ["batch-actor"]
    assert evicted_gpu_uuids == ("GPU-A",)
    assert inventory.calls == 2
    assert store.actor_allocation("GPU-A") is None
    assert ownership.exact_owner(8123) is None
    store.close()


@pytest.mark.asyncio
async def test_post_eviction_wait_absorbs_nvml_process_teardown_lag(
    tmp_path: Path,
) -> None:
    total = 32 * GIB
    lingering = GPUSnapshot(
        uuid="GPU-A",
        index=0,
        name="GPU",
        total_memory_bytes=total,
        free_memory_bytes=20 * GIB,
        used_memory_bytes=12 * GIB,
        utilization_percent=0,
        processes=(GPUProcess(8123, 12 * GIB, ProcessOwner.TOPODROME),),
    )
    released = replace(
        lingering,
        free_memory_bytes=total,
        used_memory_bytes=0,
        processes=(),
    )

    class Inventory:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self) -> tuple[GPUSnapshot, ...]:
            self.calls += 1
            return (lingering if self.calls == 1 else released,)

    inventory = Inventory()
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    admission = AdmissionController(
        inventory,  # type: ignore[arg-type]
        store,
        safety_margin_bytes=GIB,
    )
    agent = object.__new__(SiteAgent)
    agent.inventory = inventory  # type: ignore[assignment]
    agent.admission = admission
    assignment = Assignment(
        job_id="gang-job",
        attempt_id="gang-attempt",
        lease=LeaseRequest(
            lease_id="gang-lease",
            attempt_id="gang-attempt",
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 1},
            min_free_vram_bytes=10 * GIB,
            expires_at=4_000_000_000,
        ),
        model_key="sharded-runtime",
        entrypoint="experiment.py",
    )

    await agent._wait_for_evicted_slot_release(assignment, ("GPU-A",))
    decision = admission.admit(assignment.lease)

    assert decision.accepted
    assert inventory.calls == 3
    store.close()


@pytest.mark.asyncio
async def test_sharded_offer_preserves_exact_replica_and_evicts_incompatible_group(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    model_key = "exact-sharded-runtime"
    store.reserve_actor_group(
        gpu_uuids=("GPU-A", "GPU-B"),
        actor_id="sharded-actor",
        model_key=model_key,
        state="idle",
    )

    class Backend:
        def __init__(self) -> None:
            self.evicted: list[str] = []

        def is_warm_allocation(self, candidate: ActorAllocation) -> bool:
            return candidate.actor_id == "sharded-actor"

        async def evict_warm_allocation(self, candidate: ActorAllocation) -> tuple[int, ...]:
            self.evicted.append(candidate.actor_id)
            return (1001,)

    backend = Backend()
    agent = object.__new__(SiteAgent)
    agent.backend = cast(Any, backend)
    agent.fences = store
    agent.process_ownership = None
    agent.inventory = cast(
        Any,
        SimpleNamespace(
            sample=lambda: tuple(
                GPUSnapshot(
                    uuid=gpu_uuid,
                    index=index,
                    name="GPU",
                    total_memory_bytes=32 * GIB,
                    free_memory_bytes=32 * GIB,
                    used_memory_bytes=0,
                    utilization_percent=0,
                )
                for index, gpu_uuid in enumerate(("GPU-A", "GPU-B"))
            )
        ),
    )
    agent.admission = cast(
        Any,
        SimpleNamespace(
            safety_margin_bytes=0,
            max_unattributed_utilization_percent=10,
            allow_foreign_processes_when_capacity_sufficient=False,
        ),
    )
    assignment = Assignment(
        job_id="job",
        attempt_id="attempt",
        lease=LeaseRequest(
            lease_id="lease",
            attempt_id="attempt",
            gpu_uuids=("GPU-A", "GPU-B"),
            fence_epochs={"GPU-A": 2, "GPU-B": 2},
            min_free_vram_bytes=1,
            expires_at=4_000_000_000,
            model_key=model_key,
            allow_actor_reuse=True,
        ),
        model_key=model_key,
        entrypoint="experiment.py",
        raw_spec={"model": {"parallelism": {"mode": "model_sharded", "degree": 2}}},
    )

    await agent._evict_private_slot_actors(assignment)
    assert backend.evicted == []
    assert len(store.actor_allocations()) == 2

    incompatible = replace(assignment, model_key="another-runtime")
    await agent._evict_private_slot_actors(incompatible)
    assert backend.evicted == ["sharded-actor"]
    assert store.actor_allocations() == ()
    store.close()


def test_site_fencing_rejects_tampered_expired_and_wrong_site_tokens() -> None:
    authority = FencingAuthority(b"stable-fencing-key" + b"x" * 32)
    claims = (FenceClaim(gpu_uuid="GPU-A", epoch=7),)
    valid, _ = authority.issue(
        attempt_id="attempt-fenced",
        site_id="lab1",
        claims=claims,
        ttl=timedelta(minutes=5),
    )
    agent = object.__new__(SiteAgent)
    agent.site_id = "lab1"
    agent.fence_verifier = authority
    assignment = fenced_assignment(valid)

    agent._verify_fencing_token(assignment)

    encoded, signature = valid.split(".", 1)
    changed = ("A" if signature[0] != "A" else "B") + signature[1:]
    with pytest.raises(StaleFenceError, match="signature"):
        agent._verify_fencing_token(replace(assignment, fencing_token=f"{encoded}.{changed}"))

    expired, _ = authority.issue(
        attempt_id="attempt-fenced",
        site_id="lab1",
        claims=claims,
        ttl=timedelta(seconds=1),
        now=datetime(2000, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(StaleFenceError, match="expired"):
        agent._verify_fencing_token(replace(assignment, fencing_token=expired))

    wrong_site, _ = authority.issue(
        attempt_id="attempt-fenced",
        site_id="remote-jupyter-1",
        claims=claims,
        ttl=timedelta(minutes=5),
    )
    with pytest.raises(StaleFenceError, match="another site"):
        agent._verify_fencing_token(replace(assignment, fencing_token=wrong_site))


@pytest.mark.asyncio
async def test_startup_stops_runtime_then_clears_stale_claim_for_higher_epoch(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    stale = LeaseRequest(
        lease_id="lease-before-crash",
        attempt_id="attempt-before-crash",
        gpu_uuids=("GPU-A",),
        fence_epochs={"GPU-A": 8},
        min_free_vram_bytes=1,
        expires_at=101,
        model_key="model-key",
    )
    store.claim(
        stale,
        fresh_eligible=frozenset({"GPU-A"}),
        pressure_safe=frozenset({"GPU-A"}),
        now=100,
    )
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="actor-before-crash",
        model_key="model-key",
        state="busy",
        now=100,
    )

    class Ray:
        stopped = False

        async def stop(self) -> bool:
            self.stopped = True
            return True

    ray = Ray()

    class GuardedFences:
        def active_claims(self) -> dict[str, str]:
            return store.active_claims()

        def reconcile_after_runtime_stop(self) -> tuple[str, ...]:
            assert ray.stopped
            return store.reconcile_after_runtime_stop()

    agent = object.__new__(SiteAgent)
    agent.ray = cast(Any, ray)
    agent.fences = cast(Any, GuardedFences())

    assert await agent._reconcile_startup_claims() == ("lease-before-crash",)
    assert ray.stopped
    assert store.active_claims() == {}
    assert store.actor_allocation("GPU-A") is None
    assert store.highest_fence("GPU-A") == 8

    replacement = replace(
        stale,
        lease_id="lease-after-crash",
        attempt_id="attempt-after-crash",
        fence_epochs={"GPU-A": 9},
        expires_at=300,
    )
    store.claim(
        replacement,
        fresh_eligible=frozenset({"GPU-A"}),
        pressure_safe=frozenset({"GPU-A"}),
        now=200,
    )
    assert store.active_claims() == {"GPU-A": "lease-after-crash"}
    assert store.highest_fence("GPU-A") == 9
    store.close()


@pytest.mark.asyncio
async def test_startup_never_clears_claims_when_ray_stop_is_unowned(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "foreign-ray-leases.sqlite3")
    stale = LeaseRequest(
        lease_id="lease-before-crash",
        attempt_id="attempt-before-crash",
        gpu_uuids=("GPU-A",),
        fence_epochs={"GPU-A": 8},
        min_free_vram_bytes=1,
        expires_at=101,
        model_key="model-key",
    )
    store.claim(
        stale,
        fresh_eligible=frozenset({"GPU-A"}),
        pressure_safe=frozenset({"GPU-A"}),
        now=100,
    )

    class AdoptedRay:
        async def stop(self) -> bool:
            return False

    agent = object.__new__(SiteAgent)
    agent.ray = cast(Any, AdoptedRay())
    agent.fences = store

    with pytest.raises(RuntimeError, match="without proof"):
        await agent._reconcile_startup_claims()

    assert store.active_claims() == {"GPU-A": "lease-before-crash"}
    store.close()


@pytest.mark.asyncio
async def test_startup_reconnects_persisted_warm_actor_and_drops_only_stale_record(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="actor-warm",
        model_key="model-key",
        state="idle",
    )
    store.reserve_actor(
        gpu_uuid="GPU-MISSING",
        actor_id="actor-stale",
        model_key="other-key",
        state="idle",
    )

    class Inventory:
        def sample(self) -> tuple[GPUSnapshot, ...]:
            return (
                GPUSnapshot(
                    uuid="GPU-A",
                    index=3,
                    name="GPU",
                    total_memory_bytes=32 * 1024**3,
                    free_memory_bytes=16 * 1024**3,
                    used_memory_bytes=16 * 1024**3,
                    utilization_percent=0,
                ),
            )

    class Backend:
        async def reconnect_warm(
            self, allocation: ActorAllocation, gpu_index: int
        ) -> PreparedRuntime:
            assert allocation.actor_id == "actor-warm"
            assert gpu_index == 3
            return PreparedRuntime(
                "actor-warm",
                "model-key",
                True,
                process_ids=(8123,),
            )

        def stale_warm_cleanup_confirmed(self, actor_id: str) -> bool:
            return actor_id == "actor-stale"

    class ExistingRay:
        async def ensure_running(self) -> bool:
            return False

    ownership = ProcessOwnershipRegistry(proc_root=tmp_path / "proc")
    agent = object.__new__(SiteAgent)
    agent.backend = cast(Any, Backend())
    agent.ray = cast(Any, ExistingRay())
    agent.fences = store
    agent.inventory = cast(Any, Inventory())
    agent.process_ownership = ownership

    assert await agent._ensure_ray_and_reconcile_warm_actors() == ("actor-warm",)
    assert store.actor_allocation("GPU-A") is not None
    assert store.actor_allocation("GPU-MISSING") is None
    assert ownership.exact_owner(8123) is not None
    assert ownership.exact_owner(8123).actor_id == "actor-warm"  # type: ignore[union-attr]
    store.close()


@pytest.mark.asyncio
async def test_startup_reconnects_sharded_replica_once_with_rank_order(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor_group(
        gpu_uuids=("GPU-B", "GPU-A"),
        actor_id="replica-warm",
        model_key="model-key",
        state="idle",
    )

    class Inventory:
        def sample(self) -> tuple[GPUSnapshot, ...]:
            return tuple(
                GPUSnapshot(
                    uuid=uuid,
                    index=index,
                    name="GPU",
                    total_memory_bytes=32 * 1024**3,
                    free_memory_bytes=8 * 1024**3,
                    used_memory_bytes=24 * 1024**3,
                    utilization_percent=0,
                )
                for uuid, index in (("GPU-A", 3), ("GPU-B", 7))
            )

    class Backend:
        calls = 0

        async def reconnect_warm(self, _allocation: ActorAllocation, _gpu_index: int) -> None:
            raise AssertionError("group recovery must not reconnect one member")

        async def reconnect_warm_group(
            self,
            allocations: tuple[ActorAllocation, ...],
            gpu_indices: Mapping[str, int],
        ) -> PreparedRuntime:
            self.calls += 1
            assert [item.gpu_uuid for item in allocations] == ["GPU-B", "GPU-A"]
            assert gpu_indices == {"GPU-B": 7, "GPU-A": 3}
            return PreparedRuntime(
                "replica-warm",
                "model-key",
                True,
                process_ids=(9001,),
            )

    backend = Backend()
    ownership = ProcessOwnershipRegistry(proc_root=tmp_path / "proc")
    agent = object.__new__(SiteAgent)
    agent.backend = cast(Any, backend)
    agent.fences = store
    agent.inventory = cast(Any, Inventory())
    agent.process_ownership = ownership

    assert await agent._reconcile_warm_actors() == ("replica-warm",)
    assert backend.calls == 1
    assert len(store.actor_allocations()) == 2
    assert ownership.exact_owner(9001).actor_id == "replica-warm"  # type: ignore[union-attr]
    store.close()


@pytest.mark.asyncio
async def test_startup_completes_crash_left_stopping_replica_disposal(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor_group(
        gpu_uuids=("GPU-B", "GPU-A"),
        actor_id="replica-stopping",
        model_key="model-key",
        state="stopping",
    )

    class Inventory:
        def sample(self) -> tuple[GPUSnapshot, ...]:
            return tuple(
                GPUSnapshot(
                    uuid=uuid,
                    index=index,
                    name="GPU",
                    total_memory_bytes=32 * 1024**3,
                    free_memory_bytes=8 * 1024**3,
                    used_memory_bytes=24 * 1024**3,
                    utilization_percent=0,
                )
                for uuid, index in (("GPU-A", 3), ("GPU-B", 7))
            )

    class Backend:
        async def reconnect_warm(self, _allocation: ActorAllocation, _gpu_index: int) -> None:
            raise AssertionError("group recovery must reconnect the complete replica")

        async def reconnect_warm_group(
            self,
            allocations: tuple[ActorAllocation, ...],
            _gpu_indices: Mapping[str, int],
        ) -> PreparedRuntime:
            assert [item.gpu_uuid for item in allocations] == ["GPU-B", "GPU-A"]
            return PreparedRuntime(
                "replica-stopping",
                "model-key",
                True,
                process_ids=(9001,),
            )

        async def evict_warm_allocation(self, allocation: ActorAllocation) -> tuple[int, ...]:
            assert allocation.actor_id == "replica-stopping"
            return (9001,)

    ownership = ProcessOwnershipRegistry(proc_root=tmp_path / "proc")
    agent = object.__new__(SiteAgent)
    agent.backend = cast(Any, Backend())
    agent.fences = store
    agent.inventory = cast(Any, Inventory())
    agent.process_ownership = ownership

    assert await agent._reconcile_warm_actors() == ()
    assert store.actor_allocations() == ()
    assert ownership.exact_owner(9001) is None
    store.close()


@pytest.mark.asyncio
async def test_fresh_ray_discards_single_and_sharded_allocations_without_reconnect(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "fresh-ray-leases.sqlite3")
    prior = LeaseRequest(
        lease_id="lease-prior",
        attempt_id="attempt-prior",
        gpu_uuids=("GPU-A",),
        fence_epochs={"GPU-A": 11},
        min_free_vram_bytes=1,
        expires_at=200,
        model_key="single-model",
    )
    store.claim(
        prior,
        fresh_eligible=frozenset({"GPU-A"}),
        pressure_safe=frozenset({"GPU-A"}),
        now=100,
    )
    store.release_lease("lease-prior")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="actor-single",
        model_key="single-model",
        state="idle",
    )
    store.reserve_actor_group(
        gpu_uuids=("GPU-C", "GPU-B"),
        actor_id="actor-sharded",
        model_key="sharded-model",
        state="idle",
    )

    class FreshRay:
        calls = 0

        async def ensure_running(self) -> bool:
            self.calls += 1
            return True

    class Backend:
        async def reconnect_warm(self, _allocation: ActorAllocation, _gpu_index: int) -> None:
            raise AssertionError("fresh Ray must not reconnect a persisted actor")

        async def reconnect_warm_group(
            self,
            _allocations: tuple[ActorAllocation, ...],
            _gpu_indices: Mapping[str, int],
        ) -> None:
            raise AssertionError("fresh Ray must not reconnect a persisted actor group")

    class Inventory:
        def sample(self) -> tuple[GPUSnapshot, ...]:
            raise AssertionError("fresh Ray cleanup must not depend on an NVML sample")

    class TrackingFences:
        def __init__(self) -> None:
            self.evicted_actor_ids: list[str] = []

        def actor_allocations(self) -> tuple[ActorAllocation, ...]:
            return store.actor_allocations()

        def evict_actor(self, actor_id: str) -> int:
            self.evicted_actor_ids.append(actor_id)
            return store.evict_actor(actor_id)

    ray = FreshRay()
    fences = TrackingFences()
    agent = object.__new__(SiteAgent)
    agent.backend = cast(Any, Backend())
    agent.ray = cast(Any, ray)
    agent.fences = cast(Any, fences)
    agent.inventory = cast(Any, Inventory())
    agent.process_ownership = None

    assert await agent._ensure_ray_and_reconcile_warm_actors() == ()
    assert ray.calls == 1
    assert sorted(fences.evicted_actor_ids) == ["actor-sharded", "actor-single"]
    assert store.actor_allocations() == ()
    assert store.active_claims() == {}
    assert store.highest_fence("GPU-A") == 11
    store.close()


@pytest.mark.asyncio
async def test_expired_local_claim_requests_runtime_cancellation_without_early_release(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    assignment = fenced_assignment(secrets.token_urlsafe())
    expiring = replace(assignment.lease, expires_at=101)
    assignment = replace(assignment, lease=expiring)
    store.claim(
        expiring,
        fresh_eligible=frozenset({"GPU-A"}),
        pressure_safe=frozenset({"GPU-A"}),
        now=100,
    )

    class Backend:
        def __init__(self) -> None:
            self.canceled: list[str] = []

        async def cancel(self, attempt_id: str) -> None:
            self.canceled.append(attempt_id)

    backend = Backend()
    active = SimpleNamespace(
        assignment=assignment,
        lease_expired=False,
        cancel_requested=False,
    )
    agent = object.__new__(SiteAgent)
    agent.fences = store
    agent.backend = cast(Any, backend)
    agent._active_lock = asyncio.Lock()
    agent._active = cast(Any, {assignment.attempt_id: active})

    await agent._enforce_local_lease_expiry(now=101)

    assert active.lease_expired
    assert active.cancel_requested
    assert backend.canceled == [assignment.attempt_id]
    # Releasing before runtime cancellation completes would permit split-brain
    # execution under a higher epoch, so cleanup still owns this claim.
    assert store.active_claims() == {"GPU-A": expiring.lease_id}
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_released", [False, True])
async def test_active_assignment_adopts_rotated_gang_token_before_reporting(gpu_released) -> None:
    authority = FencingAuthority(b"rotating-fencing-key" + b"x" * 32)
    claims = (
        FenceClaim(gpu_uuid="GPU-A", epoch=1),
        FenceClaim(gpu_uuid="GPU-B", epoch=4),
    )
    now = datetime.now(UTC)
    scoped_value, _ = authority.issue(
        attempt_id="attempt-1",
        site_id="lab1",
        claims=claims,
        ttl=timedelta(hours=3),
        now=now,
    )
    rotated_value, rotated_payload = authority.issue(
        attempt_id="attempt-1",
        site_id="lab1",
        claims=claims,
        ttl=timedelta(hours=3),
        now=now + timedelta(seconds=1),
    )
    assignment = Assignment(
        job_id="job-1",
        attempt_id="attempt-1",
        lease=LeaseRequest(
            lease_id="lease-1",
            attempt_id="attempt-1",
            gpu_uuids=("GPU-A", "GPU-B"),
            fence_epochs={"GPU-A": 1, "GPU-B": 4},
            min_free_vram_bytes=1,
            expires_at=(now - timedelta(seconds=120)).timestamp(),
        ),
        model_key="model-key",
        entrypoint="experiment.py",
        fencing_token=scoped_value,
    )

    class Sink:
        def __init__(self) -> None:
            self.events: list[tuple[str, Mapping[str, Any]]] = []
            self.live_events: list[tuple[str, Mapping[str, Any]]] = []

        async def publish(self, kind: str, payload: Mapping[str, Any]) -> None:
            self.events.append((kind, payload))

        async def publish_request(
            self,
            request_key: str,
            kind: str,
            payload: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            assert request_key == "lease-renewal:attempt-1"
            self.events.append((kind, payload))
            return {
                "message_id": "renewal-1",
                "lease_renewal": {
                    "attempt_id": "attempt-1",
                    "fencing_token": rotated_value,
                    "fencing_token_expires_at": rotated_payload.expires_at,
                    "lease_expires_at": now + timedelta(seconds=91),
                },
            }

        async def publish_live(self, kind: str, payload: Mapping[str, Any]) -> bool:
            self.live_events.append((kind, payload))
            return True

    class Admission:
        def __init__(self) -> None:
            self.renewed: list[LeaseRequest] = []

        def renew(self, lease: LeaseRequest) -> None:
            self.renewed.append(lease)

    sink = Sink()
    agent = object.__new__(SiteAgent)
    agent._active_lock = asyncio.Lock()
    agent._active = cast(
        Any,
        {
            assignment.attempt_id: SimpleNamespace(
                assignment=assignment,
                fencing_lock=asyncio.Lock(),
                renewal_pending=False,
                scheduler_terminal=False,
                gpu_released=gpu_released,
                lease_expired=False,
            )
        },
    )
    agent.site_id = "lab1"
    agent.fence_verifier = authority
    agent.outbound = cast(Any, sink)
    admission = Admission()
    agent.admission = cast(Any, admission)

    await agent._publish_lease_renewals()
    await agent._publish_attempt_state(assignment, "running")
    await agent._publish_kernel_result(
        {
            "operation_id": "operation-1",
            "session_id": "session-1",
            "attempt_id": "attempt-1",
            "succeeded": True,
            "result": {"ok": True},
            "message": None,
        }
    )

    assert sink.events[0] == (
        "lease_renewal",
        {"attempt_id": "attempt-1", "fencing_token": scoped_value},
    )
    assert sink.events[1][0] == "attempt_event"
    assert sink.events[1][1]["fencing_token"] == rotated_value
    assert sink.live_events[0][1]["fencing_token"] == rotated_value
    assert agent._active[assignment.attempt_id].assignment.fencing_token == rotated_value
    renewed_lease = replace(
        assignment.lease, expires_at=(now + timedelta(seconds=91)).timestamp()
    )
    assert agent._active[assignment.attempt_id].assignment.lease == renewed_lease
    assert admission.renewed == ([] if gpu_released else [renewed_lease])

    # A long computation has outlived its original offer. Releasing its GPU
    # must not cancel uploads while the renewed finalization authority lives.
    active = agent._active[assignment.attempt_id]
    active.gpu_released = True
    agent.fences = cast(Any, SimpleNamespace(expired_lease_ids=lambda **_: ()))
    canceled = []
    agent.backend = cast(Any, SimpleNamespace(cancel_finalization=canceled.append))
    await agent._enforce_local_lease_expiry(now=now.timestamp())
    assert not active.lease_expired
    assert canceled == []
    # Genuine expiry still fences uploads; no GPU actor cancellation is used.
    await agent._enforce_local_lease_expiry(now=renewed_lease.expires_at)
    assert active.lease_expired
    assert canceled == [assignment.attempt_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_released", [False, True])
async def test_terminal_renewal_response_fences_local_attempt_without_retrying(
    gpu_released,
) -> None:
    assignment = Assignment(
        job_id="job-1",
        attempt_id="attempt-1",
        lease=LeaseRequest(
            lease_id="lease-1",
            attempt_id="attempt-1",
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 1},
            min_free_vram_bytes=1,
            expires_at=4_000_000_000,
        ),
        model_key="model-key",
        entrypoint="experiment.py",
        fencing_token="historical-token",  # noqa: S106 - protocol canary
    )

    class Sink:
        def __init__(self) -> None:
            self.requests = 0

        async def publish_request(
            self,
            request_key: str,
            kind: str,
            payload: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            assert request_key == "lease-renewal:attempt-1"
            assert kind == "lease_renewal"
            assert payload["attempt_id"] == "attempt-1"
            self.requests += 1
            return {
                "stale": True,
                "lease_renewal_terminal": {
                    "attempt_id": "attempt-1",
                    "state": "lost",
                },
            }

    class Backend:
        def __init__(self) -> None:
            self.canceled: list[str] = []
            self.finalization_canceled: list[str] = []

        async def cancel(self, attempt_id: str) -> None:
            self.canceled.append(attempt_id)

        def cancel_finalization(self, attempt_id: str) -> None:
            self.finalization_canceled.append(attempt_id)

    active = SimpleNamespace(
        assignment=assignment,
        fencing_lock=asyncio.Lock(),
        renewal_pending=False,
        scheduler_terminal=False,
        lease_expired=False,
        cancel_requested=False,
        gpu_released=gpu_released,
    )
    sink = Sink()
    backend = Backend()
    agent = object.__new__(SiteAgent)
    agent._active_lock = asyncio.Lock()
    agent._active = cast(Any, {assignment.attempt_id: active})
    agent._kernel_hosts = {}
    agent.outbound = cast(Any, sink)
    agent.backend = cast(Any, backend)

    await agent._publish_lease_renewals()
    await asyncio.sleep(0)
    await agent._publish_lease_renewals()

    assert sink.requests == 1
    assert backend.canceled == ([] if gpu_released else [assignment.attempt_id])
    assert backend.finalization_canceled == ([assignment.attempt_id] if gpu_released else [])
    assert active.scheduler_terminal
    assert active.lease_expired
    assert active.cancel_requested
    assert not active.renewal_pending


@pytest.mark.asyncio
async def test_provider_idle_activity_reports_local_execution_phases() -> None:
    queued = Assignment(
        job_id="job-queued",
        attempt_id="attempt-queued",
        lease=LeaseRequest(
            lease_id="lease-queued",
            attempt_id="attempt-queued",
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 1},
            min_free_vram_bytes=1,
            expires_at=4_000_000_000,
        ),
        model_key="model-key",
        entrypoint="experiment.py",
    )
    running = Assignment(
        job_id="job-running",
        attempt_id="attempt-running",
        lease=LeaseRequest(
            lease_id="lease-running",
            attempt_id="attempt-running",
            gpu_uuids=("GPU-B",),
            fence_epochs={"GPU-B": 1},
            min_free_vram_bytes=1,
            expires_at=4_000_000_000,
        ),
        model_key="model-key",
        entrypoint="experiment.py",
        raw_spec={"workload_kind": "interactive", "required_outputs": ["result.json"]},
    )
    agent = object.__new__(SiteAgent)
    agent._active_lock = asyncio.Lock()
    agent._active = cast(
        Any,
        {
            queued.attempt_id: SimpleNamespace(assignment=queued, prepared=None),
            running.attempt_id: SimpleNamespace(assignment=running, prepared=object()),
        },
    )

    activity = await agent._activity_payload()

    assert activity == {
        "job_running": True,
        "assignment_queued_locally": True,
        "interactive_kernel_active": True,
        "artifact_flush_pending": True,
        "staging_operation_active": True,
    }


@pytest.mark.asyncio
async def test_heartbeat_reports_host_capacity_and_verified_asset_cache() -> None:
    class Inventory:
        def sample(self) -> tuple[GPUSnapshot, ...]:
            return (
                GPUSnapshot(
                    uuid="GPU-A",
                    index=0,
                    name="GPU",
                    total_memory_bytes=32 * 1024**3,
                    free_memory_bytes=24 * 1024**3,
                    used_memory_bytes=8 * 1024**3,
                    utilization_percent=3,
                    external_reservation=GPUReservation(
                        physical_gpu=0,
                        supervisor_pid=4321,
                        owner="foreign-job",
                        world_size=1,
                        acquired_at=100,
                        supervisor_start_ticks=12345,
                    ),
                ),
            )

    class Host:
        def sample(self) -> HostSnapshot:
            return HostSnapshot(8, 6.5, 64 * 1024**3, 48 * 1024**3)

    class Sink:
        payload: dict[str, Any] | None = None

        async def publish_heartbeat(self, payload: Mapping[str, Any]) -> None:
            self.payload = dict(payload)
            agent._stop.set()

    class Admission:
        safety_margin_bytes = 1024

    sink = Sink()
    agent = object.__new__(SiteAgent)
    agent.site_id = "lab1"
    agent.site_kind = "lab"
    agent.inventory = cast(Any, Inventory())
    agent.host_inventory = cast(Any, Host())
    agent.outbound = cast(Any, sink)
    agent.admission = cast(Any, Admission())
    agent.model_capabilities = tuple
    agent.cached_assets = lambda: ("a" * 64,)
    agent.supported_workloads = ("batch",)
    agent.storage_available = lambda: True
    agent.provider_instance_id = None
    agent.region = None
    agent._draining = False
    agent._active = {}
    agent._active_lock = asyncio.Lock()
    agent._kernel_hosts = {}
    agent._stop = asyncio.Event()
    agent.heartbeat_interval_seconds = 1

    await agent._heartbeat_loop()

    assert sink.payload is not None
    assert sink.payload["cached_assets"] == ["a" * 64]
    assert sink.payload["total_cpu"] == 8
    assert sink.payload["available_cpu"] == 6.5
    assert sink.payload["total_memory_bytes"] == 64 * 1024**3
    assert sink.payload["available_memory_bytes"] == 48 * 1024**3
    assert sink.payload["gpus"][0]["externally_reserved"] is True


@pytest.mark.asyncio
async def test_shutdown_commands_reject_stale_generations_and_conflicting_nonces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Sink:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        async def publish(self, kind: str, payload: Mapping[str, Any]) -> None:
            self.events.append((kind, dict(payload)))

    class Regional:
        def __init__(self) -> None:
            self.calls: list[tuple[int, str]] = []

        def durability_barrier(
            self,
            *,
            shutdown_generation: int,
            command_nonce: str,
        ) -> None:
            self.calls.append((shutdown_generation, command_nonce))

    sink = Sink()
    regional = Regional()
    agent = object.__new__(SiteAgent)
    agent.site_id = "autodl"
    agent.site_kind = "autodl"
    agent.outbound = cast(Any, sink)
    agent._active = {}
    agent._active_lock = asyncio.Lock()
    agent._draining = False
    agent._drain_ready_emitted = False
    agent._durability_ready_emitted = False
    agent._shutdown_generation = 0
    agent._drain_command_nonce = None
    agent._durability_command_nonce = None
    agent.storage_available = lambda: True
    agent.regional_durability = regional
    monkeypatch.setattr("topodrome.site_agent.agent.os.sync", lambda: None)

    drain_nonce = "drain-current-nonce-000001"
    await agent.handle_command(
        {
            "message_id": "drain-2",
            "kind": "begin_drain",
            "payload": {
                "shutdown_generation": 2,
                "command_nonce": drain_nonce,
            },
        }
    )
    assert sink.events[0] == (
        "drain_ready",
        {
            "site_id": "autodl",
            "shutdown_generation": 2,
            "command_nonce": drain_nonce,
        },
    )
    assert sink.events[1][0] == "command_ack"
    assert sink.events[1][1]["accepted"] is True

    for message_id, generation, nonce in (
        ("old-generation", 1, "drain-old-generation-0001"),
        ("conflicting-nonce", 2, "drain-conflicting-nonce-1"),
    ):
        await agent.handle_command(
            {
                "message_id": message_id,
                "kind": "begin_drain",
                "payload": {
                    "shutdown_generation": generation,
                    "command_nonce": nonce,
                },
            }
        )
    assert [kind for kind, _ in sink.events].count("drain_ready") == 1
    rejected = [
        payload
        for kind, payload in sink.events
        if kind == "command_ack" and payload["accepted"] is False
    ]
    assert {payload["command_id"] for payload in rejected} == {
        "old-generation",
        "conflicting-nonce",
    }

    durability_nonce = "durability-current-nonce-01"
    await agent.handle_command(
        {
            "message_id": "durability-2",
            "kind": "verify_durability",
            "payload": {
                "shutdown_generation": 2,
                "command_nonce": durability_nonce,
            },
        }
    )
    durability_events = [payload for kind, payload in sink.events if kind == "durability_ready"]
    assert durability_events == [
        {
            "site_id": "autodl",
            "shutdown_generation": 2,
            "command_nonce": durability_nonce,
        }
    ]
    assert regional.calls == [(2, durability_nonce)]

    # Replaying an earlier generation or changing the nonce cannot replace the
    # current durability coordinate used by the later stop command.
    await agent.handle_command(
        {
            "message_id": "stale-durability",
            "kind": "verify_durability",
            "payload": {
                "shutdown_generation": 1,
                "command_nonce": "durability-old-nonce-0001",
            },
        }
    )
    assert [kind for kind, _ in sink.events].count("durability_ready") == 1
    assert agent._shutdown_generation == 2
    assert agent._drain_command_nonce == drain_nonce
    assert agent._durability_command_nonce == durability_nonce
