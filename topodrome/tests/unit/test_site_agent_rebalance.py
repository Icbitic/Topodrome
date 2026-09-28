from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from topodrome.site_agent.admission import (
    GIB,
    ActorAllocation,
    GPUUnavailableError,
    LeaseFenceStore,
    LeaseRequest,
)
from topodrome.site_agent.agent import SiteAgent
from topodrome.site_agent.inventory import GPUProcess, GPUSnapshot, ProcessOwner
from topodrome.site_agent.rebalance import (
    IdleActorRebalancePolicy,
    IdleActorRebalancer,
)

pytestmark = pytest.mark.unit


def _gpu(
    uuid: str,
    index: int,
    *,
    free_gib: int,
    utilization: int = 0,
    topodrome_gib: int = 0,
    foreign_gib: int = 0,
    foreign_pid: int | None = None,
    cooperative_gib: int = 0,
    cooperative_pid: int | None = None,
) -> GPUSnapshot:
    processes: list[GPUProcess] = []
    if topodrome_gib:
        processes.append(GPUProcess(1000 + index, topodrome_gib * GIB, ProcessOwner.TOPODROME))
    if foreign_gib or foreign_pid is not None:
        processes.append(
            GPUProcess(
                foreign_pid or 9000 + index,
                foreign_gib * GIB,
                ProcessOwner.FOREIGN,
            )
        )
    if cooperative_gib or cooperative_pid is not None:
        processes.append(
            GPUProcess(
                cooperative_pid or 8000 + index,
                cooperative_gib * GIB,
                ProcessOwner.COOPERATIVE,
            )
        )
    return GPUSnapshot(
        uuid=uuid,
        index=index,
        name="shared GPU",
        total_memory_bytes=80 * GIB,
        free_memory_bytes=free_gib * GIB,
        used_memory_bytes=(80 - free_gib) * GIB,
        utilization_percent=utilization,
        processes=tuple(processes),
    )


def _rebalancer(
    *,
    sustained_seconds: float = 10,
    cooldown_seconds: float = 300,
    allow_foreign: bool = True,
) -> IdleActorRebalancer:
    return IdleActorRebalancer(
        IdleActorRebalancePolicy(
            sustained_seconds=sustained_seconds,
            cooldown_seconds=cooldown_seconds,
            allow_foreign_processes_when_capacity_sufficient=allow_foreign,
        )
    )


def test_rebalance_requires_sustained_heavy_pressure_and_better_band() -> None:
    rebalancer = _rebalancer()
    heavy = _gpu(
        "GPU-A",
        0,
        free_gib=55,
        utilization=70,
        topodrome_gib=16,
        foreign_gib=8,
        foreign_pid=9123,
    )
    clean = _gpu("GPU-B", 1, free_gib=80)
    allocation = ActorAllocation("GPU-A", "actor-a", "llama", "idle")

    assert rebalancer.observe((heavy, clean), now=0) == ()
    assert rebalancer.observe((heavy, clean), now=9.9) == ()
    events = rebalancer.observe((heavy, clean), now=10)
    assert events[0].foreign_pids == (9123,)

    plans = rebalancer.plans(
        events,
        (heavy, clean),
        (allocation,),
        {},
        safety_margin_bytes=GIB,
    )

    assert len(plans) == 1
    assert plans[0].actor_id == "actor-a"
    assert plans[0].gpu_uuids == ("GPU-A",)
    assert plans[0].alternative_gpu_uuids == ("GPU-B",)
    assert plans[0].alternative_band == "clean"

    light_source = _gpu(
        "GPU-A",
        0,
        free_gib=60,
        utilization=5,
        topodrome_gib=16,
        foreign_gib=1,
    )
    assert (
        rebalancer.plans(
            events,
            (light_source, clean),
            (allocation,),
            {},
            safety_margin_bytes=GIB,
        )
        == ()
    )


def test_rebalance_never_uses_heavy_or_policy_ineligible_alternative() -> None:
    rebalancer = _rebalancer(allow_foreign=False)
    source = _gpu(
        "GPU-A",
        0,
        free_gib=20,
        utilization=80,
        topodrome_gib=16,
        foreign_gib=20,
    )
    contended_target = _gpu(
        "GPU-B",
        1,
        free_gib=60,
        utilization=5,
        foreign_gib=1,
    )
    allocation = ActorAllocation("GPU-A", "actor-a", "llama", "idle")
    assert rebalancer.observe((source, contended_target), now=0) == ()
    events = rebalancer.observe((source, contended_target), now=10)

    assert (
        rebalancer.plans(
            events,
            (source, contended_target),
            (allocation,),
            {},
            safety_margin_bytes=GIB,
        )
        == ()
    )


def test_sustained_pressure_replans_when_a_better_target_becomes_free() -> None:
    rebalancer = _rebalancer()
    source = _gpu(
        "GPU-A",
        0,
        free_gib=20,
        utilization=80,
        topodrome_gib=16,
        foreign_gib=20,
    )
    unavailable_target = _gpu(
        "GPU-B",
        1,
        free_gib=20,
        utilization=80,
        foreign_gib=20,
    )
    allocation = ActorAllocation("GPU-A", "actor-a", "llama", "idle")

    assert rebalancer.observe((source, unavailable_target), now=0) == ()
    first_events = rebalancer.observe((source, unavailable_target), now=10)
    assert first_events
    assert (
        rebalancer.plans(
            first_events,
            (source, unavailable_target),
            (allocation,),
            {},
            safety_margin_bytes=GIB,
        )
        == ()
    )

    free_target = _gpu("GPU-B", 1, free_gib=80)
    retry_events = rebalancer.observe((source, free_target), now=11)
    plans = rebalancer.plans(
        retry_events,
        (source, free_target),
        (allocation,),
        {},
        safety_margin_bytes=GIB,
    )
    assert len(plans) == 1
    assert plans[0].alternative_gpu_uuids == ("GPU-B",)

    # Clearing pressure resets the sustained-duration hysteresis for a later
    # episode rather than immediately reusing the old observation window.
    assert rebalancer.observe((free_target,), now=12) == ()
    assert rebalancer.observe((source, free_target), now=13) == ()
    assert rebalancer.observe((source, free_target), now=22.9) == ()
    assert len(rebalancer.observe((source, free_target), now=23)) == 1


def test_cooperative_scavenger_is_a_nonblocking_rebalance_target() -> None:
    rebalancer = _rebalancer(sustained_seconds=0, allow_foreign=False)
    source = _gpu(
        "GPU-A",
        0,
        free_gib=20,
        utilization=80,
        topodrome_gib=16,
        foreign_gib=20,
    )
    cooperative_target = _gpu(
        "GPU-B",
        1,
        free_gib=60,
        cooperative_gib=4,
        cooperative_pid=8123,
    )
    allocation = ActorAllocation("GPU-A", "actor-a", "llama", "idle")

    events = rebalancer.observe((source, cooperative_target), now=0)
    plans = rebalancer.plans(
        events,
        (source, cooperative_target),
        (allocation,),
        {},
        safety_margin_bytes=GIB,
    )

    assert len(plans) == 1
    assert plans[0].alternative_gpu_uuids == ("GPU-B",)
    assert plans[0].alternative_band == "clean"


def test_pressure_on_one_replica_member_requires_complete_better_gang() -> None:
    rebalancer = _rebalancer()
    snapshots = (
        _gpu(
            "GPU-A",
            0,
            free_gib=25,
            utilization=80,
            topodrome_gib=12,
            foreign_gib=20,
        ),
        _gpu("GPU-B", 1, free_gib=68, topodrome_gib=12),
        _gpu("GPU-C", 2, free_gib=80),
        _gpu("GPU-D", 3, free_gib=80),
    )
    allocations = (
        ActorAllocation("GPU-A", "replica", "llama-sharded", "idle", 0),
        ActorAllocation("GPU-B", "replica", "llama-sharded", "idle", 1),
    )
    assert rebalancer.observe(snapshots, now=0) == ()
    events = rebalancer.observe(snapshots, now=10)

    plans = rebalancer.plans(
        events,
        snapshots,
        allocations,
        {},
        safety_margin_bytes=GIB,
    )
    assert len(plans) == 1
    assert plans[0].gpu_uuids == ("GPU-A", "GPU-B")
    assert plans[0].alternative_gpu_uuids == ("GPU-C", "GPU-D")

    assert (
        rebalancer.plans(
            events,
            snapshots[:3],
            allocations,
            {},
            safety_margin_bytes=GIB,
        )
        == ()
    )


def test_busy_or_locally_active_actor_is_never_planned() -> None:
    rebalancer = _rebalancer(sustained_seconds=0)
    snapshots = (
        _gpu(
            "GPU-A",
            0,
            free_gib=20,
            utilization=80,
            topodrome_gib=16,
            foreign_gib=20,
        ),
        _gpu("GPU-B", 1, free_gib=80),
    )
    events = rebalancer.observe(snapshots, now=0)

    busy = ActorAllocation("GPU-A", "actor", "llama", "busy")
    assert rebalancer.plans(events, snapshots, (busy,), {}, safety_margin_bytes=GIB) == ()
    idle = ActorAllocation("GPU-A", "actor", "llama", "idle")
    assert (
        rebalancer.plans(
            events,
            snapshots,
            (idle,),
            {"GPU-A": "lease-active"},
            safety_margin_bytes=GIB,
        )
        == ()
    )


def test_idle_eviction_fence_is_replica_atomic_and_cooldown_is_durable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "leases.sqlite3"
    store = LeaseFenceStore(path)
    store.reserve_actor_group(
        gpu_uuids=("GPU-A", "GPU-B"),
        actor_id="replica",
        model_key="llama-sharded",
        state="idle",
    )

    assert store.begin_idle_actor_eviction(
        "replica",
        expected_gpu_uuids=("GPU-A", "GPU-B"),
        alternative_gpu_uuids=("GPU-C", "GPU-D"),
        cooldown_seconds=100,
        now=10,
    )
    assert {allocation.state for allocation in store.actor_allocations()} == {"stopping"}
    assert store.complete_idle_actor_eviction("replica", expected_gpu_uuids=("GPU-B", "GPU-A")) == 0
    assert len(store.actor_allocations()) == 2
    assert store.complete_idle_actor_eviction("replica", expected_gpu_uuids=("GPU-A", "GPU-B")) == 2
    assert store.idle_actor_rebalance_cooldown_until("GPU-A") == 110
    assert store.idle_actor_rebalance_cooldown_until("GPU-D") == 110
    store.close()

    reopened = LeaseFenceStore(path)
    reopened.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="replacement",
        model_key="llama",
        state="idle",
    )
    assert not reopened.begin_idle_actor_eviction(
        "replacement",
        expected_gpu_uuids=("GPU-A",),
        alternative_gpu_uuids=("GPU-C",),
        cooldown_seconds=100,
        now=109,
    )
    assert reopened.begin_idle_actor_eviction(
        "replacement",
        expected_gpu_uuids=("GPU-A",),
        alternative_gpu_uuids=("GPU-C",),
        cooldown_seconds=100,
        now=111,
    )
    reopened.close()


def test_idle_eviction_fence_loses_safely_to_a_concurrent_claim(tmp_path: Path) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="actor",
        model_key="llama",
        state="idle",
    )
    target_claim = LeaseRequest(
        lease_id="target-lease",
        attempt_id="target-attempt",
        gpu_uuids=("GPU-B",),
        fence_epochs={"GPU-B": 1},
        min_free_vram_bytes=1,
        expires_at=100,
    )
    store.claim(
        target_claim,
        fresh_eligible=frozenset({"GPU-B"}),
        pressure_safe=frozenset({"GPU-B"}),
        now=1,
    )

    assert not store.begin_idle_actor_eviction(
        "actor",
        expected_gpu_uuids=("GPU-A",),
        alternative_gpu_uuids=("GPU-B",),
        cooldown_seconds=100,
        now=2,
    )
    assert store.actor_allocation("GPU-A") == ActorAllocation("GPU-A", "actor", "llama", "idle")
    with pytest.raises(GPUUnavailableError, match="allocated"):
        store.claim(
            LeaseRequest(
                lease_id="source-lease",
                attempt_id="source-attempt",
                gpu_uuids=("GPU-A",),
                fence_epochs={"GPU-A": 1},
                min_free_vram_bytes=1,
                expires_at=100,
            ),
            fresh_eligible=frozenset({"GPU-A"}),
            pressure_safe=frozenset({"GPU-A"}),
            now=3,
        )

    assert store.release_lease("target-lease") == 1
    assert store.begin_idle_actor_eviction(
        "actor",
        expected_gpu_uuids=("GPU-A",),
        alternative_gpu_uuids=("GPU-B",),
        cooldown_seconds=100,
        now=4,
    )
    with pytest.raises(GPUUnavailableError, match="stopping"):
        store.claim(
            LeaseRequest(
                lease_id="late-source-lease",
                attempt_id="late-source-attempt",
                gpu_uuids=("GPU-A",),
                fence_epochs={"GPU-A": 2},
                min_free_vram_bytes=1,
                expires_at=100,
                reuse_actor_id="actor",
            ),
            fresh_eligible=frozenset(),
            pressure_safe=frozenset({"GPU-A"}),
            now=5,
        )
    store.close()


@pytest.mark.asyncio
async def test_agent_evicts_exact_whole_idle_replica_without_foreign_signal(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor_group(
        gpu_uuids=("GPU-A", "GPU-B"),
        actor_id="replica",
        model_key="llama-sharded",
        state="idle",
    )
    snapshots = (
        _gpu(
            "GPU-A",
            0,
            free_gib=25,
            utilization=80,
            topodrome_gib=12,
            foreign_gib=20,
            foreign_pid=9123,
        ),
        _gpu("GPU-B", 1, free_gib=68, topodrome_gib=12),
        _gpu("GPU-C", 2, free_gib=80),
        _gpu("GPU-D", 3, free_gib=80),
    )
    rebalancer = _rebalancer(sustained_seconds=0)
    events = rebalancer.observe(snapshots, now=0)

    class Backend:
        def __init__(self) -> None:
            self.evicted: list[ActorAllocation] = []

        def is_warm_allocation(self, allocation: ActorAllocation) -> bool:
            return allocation.actor_id == "replica" and allocation.state == "idle"

        async def evict_warm_allocation(self, allocation: ActorAllocation) -> tuple[int, ...]:
            # The foreign PID from the pressure event is never an eviction input.
            assert allocation.actor_id == "replica"
            assert allocation.gpu_uuid == "GPU-A"
            self.evicted.append(allocation)
            await asyncio.sleep(0)
            return (1000, 1001)

    backend = Backend()
    agent = object.__new__(SiteAgent)
    agent.idle_actor_rebalancer = rebalancer
    agent.backend = cast(Any, backend)
    agent.fences = store
    agent.inventory = cast(Any, SimpleNamespace(sample=lambda: snapshots))
    agent.admission = cast(Any, SimpleNamespace(safety_margin_bytes=GIB))
    agent.process_ownership = None
    agent._active = {}
    agent._active_lock = asyncio.Lock()

    assert await agent._rebalance_idle_warm_actors(events) == ("replica",)
    assert len(backend.evicted) == 1
    assert store.actor_allocations() == ()
    store.close()


@pytest.mark.asyncio
async def test_failed_exact_stop_restores_only_a_reproven_idle_actor(tmp_path: Path) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="actor",
        model_key="llama",
        state="idle",
    )
    snapshots = (
        _gpu(
            "GPU-A",
            0,
            free_gib=25,
            utilization=80,
            topodrome_gib=16,
            foreign_gib=20,
        ),
        _gpu("GPU-B", 1, free_gib=80),
    )
    rebalancer = _rebalancer(sustained_seconds=0)

    class Backend:
        def is_warm_allocation(self, allocation: ActorAllocation) -> bool:
            return allocation.actor_id == "actor"

        async def evict_warm_allocation(self, _allocation: ActorAllocation) -> tuple[int, ...]:
            raise RuntimeError("exact actor remained live")

    agent = object.__new__(SiteAgent)
    agent.idle_actor_rebalancer = rebalancer
    agent.backend = cast(Any, Backend())
    agent.fences = store
    agent.inventory = cast(Any, SimpleNamespace(sample=lambda: snapshots))
    agent.admission = cast(Any, SimpleNamespace(safety_margin_bytes=GIB))
    agent.process_ownership = None
    agent._active = {}
    agent._active_lock = asyncio.Lock()

    events = rebalancer.observe(snapshots, now=0)
    assert await agent._rebalance_idle_warm_actors(events) == ()
    allocation = store.actor_allocation("GPU-A")
    assert allocation is not None and allocation.state == "idle"
    store.close()


@pytest.mark.asyncio
async def test_idle_timeout_evicts_exact_sharded_replica_without_alternative_gpu(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor_group(
        gpu_uuids=("GPU-A", "GPU-B"),
        actor_id="replica",
        model_key="llama-sharded",
        state="idle",
        now=0,
    )

    class Backend:
        def __init__(self) -> None:
            self.evicted: list[ActorAllocation] = []

        def is_warm_allocation(self, allocation: ActorAllocation) -> bool:
            return allocation.actor_id == "replica"

        async def evict_warm_allocation(
            self, allocation: ActorAllocation
        ) -> tuple[int, ...]:
            self.evicted.append(allocation)
            return (1000,)

    backend = Backend()
    agent = object.__new__(SiteAgent)
    agent.warm_actor_idle_timeout_seconds = 7200
    agent.backend = cast(Any, backend)
    agent.fences = store
    agent.process_ownership = None
    agent._active = {}
    agent._active_lock = asyncio.Lock()

    assert await agent._release_expired_idle_warm_actor(now=7199.999) == ()
    assert await agent._release_expired_idle_warm_actor(now=7200) == ("replica",)
    assert len(backend.evicted) == 1
    assert backend.evicted[0].gpu_uuid == "GPU-A"
    assert store.actor_allocations() == ()
    store.close()


@pytest.mark.asyncio
async def test_idle_timeout_skips_actor_still_referenced_by_attempt_cleanup(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="actor",
        model_key="llama",
        state="idle",
        now=0,
    )

    class Backend:
        def is_warm_allocation(self, _allocation: ActorAllocation) -> bool:
            return True

        async def evict_warm_allocation(
            self, _allocation: ActorAllocation
        ) -> tuple[int, ...]:
            raise AssertionError("active-attempt actor must not be evicted")

    agent = object.__new__(SiteAgent)
    agent.warm_actor_idle_timeout_seconds = 7200
    agent.backend = cast(Any, Backend())
    agent.fences = store
    agent.process_ownership = None
    agent._active = {
        "attempt": SimpleNamespace(
            assignment=SimpleNamespace(lease=SimpleNamespace(gpu_uuids=("GPU-A",)))
        )
    }
    agent._active_lock = asyncio.Lock()

    assert await agent._release_expired_idle_warm_actor(now=8000) == ()
    allocation = store.actor_allocation("GPU-A")
    assert allocation is not None and allocation.state == "idle"
    store.close()


@pytest.mark.asyncio
async def test_failed_idle_timeout_restores_live_actor_and_restarts_timer(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="actor",
        model_key="llama",
        state="idle",
        now=0,
    )

    class Backend:
        def is_warm_allocation(self, allocation: ActorAllocation) -> bool:
            return allocation.actor_id == "actor"

        async def evict_warm_allocation(
            self, _allocation: ActorAllocation
        ) -> tuple[int, ...]:
            raise RuntimeError("exact actor remained live")

    agent = object.__new__(SiteAgent)
    agent.warm_actor_idle_timeout_seconds = 7200
    agent.backend = cast(Any, Backend())
    agent.fences = store
    agent.process_ownership = None
    agent._active = {}
    agent._active_lock = asyncio.Lock()

    assert await agent._release_expired_idle_warm_actor(now=7200) == ()
    allocation = store.actor_allocation("GPU-A")
    assert allocation is not None and allocation.state == "idle"
    assert store.expired_idle_actor_groups(idle_before=7199.999) == ()
    assert len(store.expired_idle_actor_groups(idle_before=7200)) == 1
    store.close()
