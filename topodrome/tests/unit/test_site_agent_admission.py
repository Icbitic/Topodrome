from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from topodrome.site_agent.admission import (
    GIB,
    AdmissionController,
    GPUUnavailableError,
    LeaseFenceStore,
    LeaseRequest,
)
from topodrome.site_agent.inventory import (
    GPUProcess,
    GPUReservation,
    GPUSnapshot,
    HostSnapshot,
    ProcessOwner,
    SystemHostInventory,
)

pytestmark = pytest.mark.unit


class StaticInventory:
    def __init__(self, snapshots: Sequence[GPUSnapshot]) -> None:
        self.snapshots = tuple(snapshots)

    def sample(self) -> tuple[GPUSnapshot, ...]:
        return self.snapshots


class StaticHostInventory:
    def __init__(self, snapshot: HostSnapshot) -> None:
        self.snapshot = snapshot

    def sample(self) -> HostSnapshot:
        return self.snapshot


def gpu(
    uuid: str,
    *,
    free_gib: float = 24,
    utilization: int = 0,
    processes: tuple[GPUProcess, ...] = (),
    external_reservation: GPUReservation | None = None,
) -> GPUSnapshot:
    total = 32 * GIB
    free = int(free_gib * GIB)
    return GPUSnapshot(
        uuid=uuid,
        index=0,
        name="RTX 5090",
        total_memory_bytes=total,
        free_memory_bytes=free,
        used_memory_bytes=total - free,
        utilization_percent=utilization,
        external_reservation=external_reservation,
        processes=processes,
        sampled_at=1.0,
    )


def request(
    lease_id: str,
    uuids: tuple[str, ...],
    epoch: int,
    *,
    reuse_actor_id: str | None = None,
    min_vram_gib: int = 15,
    cpu_cores: float = 1,
    memory_gib: int = 1,
) -> LeaseRequest:
    return LeaseRequest(
        lease_id=lease_id,
        attempt_id=f"attempt-{lease_id}",
        gpu_uuids=uuids,
        fence_epochs={uuid: epoch for uuid in uuids},
        min_free_vram_bytes=min_vram_gib * GIB,
        expires_at=4_000_000_000,
        cpu_cores=cpu_cores,
        memory_bytes=memory_gib * GIB,
        reuse_actor_id=reuse_actor_id,
        model_key="llama-key",
    )


def test_host_cpu_and_memory_are_atomically_reserved_per_lease(tmp_path: Path) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    host = StaticHostInventory(
        HostSnapshot(
            total_cpu=4,
            available_cpu=4,
            total_memory_bytes=8 * GIB,
            available_memory_bytes=8 * GIB,
        )
    )
    controller = AdmissionController(
        StaticInventory([gpu("GPU-A"), gpu("GPU-B")]),
        store,
        host_inventory=host,
    )

    first = request(
        "first",
        ("GPU-A",),
        1,
        cpu_cores=3,
        memory_gib=6,
    )
    assert controller.admit(first).accepted
    assert controller.admit(first).accepted
    assert store.active_resource_totals() == (3, 6 * GIB)

    rejected = controller.admit(request("second", ("GPU-B",), 1, cpu_cores=2, memory_gib=3))
    assert not rejected.accepted
    assert "CPU" in (rejected.reason or "")
    assert store.active_claims() == {"GPU-A": "first"}

    assert controller.release("first") == 1
    assert store.active_resource_totals() == (0, 0)
    assert controller.admit(request("second", ("GPU-B",), 1, cpu_cores=2, memory_gib=3)).accepted
    store.close()


def test_system_host_inventory_ignores_transient_load_for_hard_cpu_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       16777216 kB\nMemAvailable:   12582912 kB\n",
        encoding="utf-8",
    )
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
    (cgroup / "memory.max").write_text(str(10 * GIB), encoding="utf-8")
    (cgroup / "memory.current").write_text(str(3 * GIB), encoding="utf-8")
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr("os.getloadavg", lambda: (999.0, 999.0, 999.0))

    snapshot = SystemHostInventory(
        proc_meminfo=meminfo,
        cgroup_root=cgroup,
        clock=lambda: 123,
    ).sample()

    assert snapshot.total_cpu == 2
    assert snapshot.available_cpu == 2
    assert snapshot.total_memory_bytes == 10 * GIB
    assert snapshot.available_memory_bytes == 7 * GIB
    assert snapshot.sampled_at == 123


def test_fifteen_gib_request_includes_safety_margin(tmp_path: Path) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    exact = AdmissionController(
        StaticInventory([gpu("GPU-A", free_gib=16)]),
        store,
        safety_margin_bytes=GIB,
    )
    assert exact.admit(request("lease-1", ("GPU-A",), 1)).accepted
    exact.release("lease-1")

    too_tight = AdmissionController(
        StaticInventory([gpu("GPU-A", free_gib=15.9)]),
        store,
        safety_margin_bytes=GIB,
    )
    decision = too_tight.admit(request("lease-2", ("GPU-A",), 2))
    assert not decision.accepted
    assert "capacity" in (decision.reason or "")
    store.close()


def test_external_reservation_blocks_even_when_foreign_colocation_is_allowed(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    reservation = GPUReservation(
        physical_gpu=0,
        supervisor_pid=4321,
        owner="foreign-job",
        world_size=1,
        acquired_at=100,
        supervisor_start_ticks=12345,
    )
    controller = AdmissionController(
        StaticInventory([gpu("GPU-A", external_reservation=reservation)]),
        store,
        allow_foreign_processes_when_capacity_sufficient=True,
    )

    decision = controller.admit(request("lease-reserved", ("GPU-A",), 1))

    assert not decision.accepted
    assert "foreign pressure" in (decision.reason or "")
    assert store.active_claims() == {}
    store.close()


def test_stale_fence_remains_rejected_after_release_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "leases.sqlite3"
    store = LeaseFenceStore(path)
    controller = AdmissionController(StaticInventory([gpu("GPU-A")]), store)
    assert controller.admit(request("new", ("GPU-A",), 8)).accepted
    assert controller.release("new") == 1
    store.close()

    reopened = LeaseFenceStore(path)
    controller = AdmissionController(StaticInventory([gpu("GPU-A")]), reopened)
    stale = controller.admit(request("old", ("GPU-A",), 7))
    assert not stale.accepted
    assert "not newer" in (stale.reason or "")
    assert reopened.active_claims() == {}
    reopened.close()


def test_expired_offer_is_rejected_without_advancing_fence(tmp_path: Path) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    controller = AdmissionController(StaticInventory([gpu("GPU-A")]), store)
    expired = LeaseRequest(
        lease_id="expired",
        attempt_id="attempt-expired",
        gpu_uuids=("GPU-A",),
        fence_epochs={"GPU-A": 4},
        min_free_vram_bytes=15 * GIB,
        expires_at=99,
        model_key="llama-key",
    )

    decision = controller.admit(expired, now=100)
    assert not decision.accepted
    assert "expired" in (decision.reason or "")
    assert store.highest_fence("GPU-A") is None
    store.close()


def test_local_lease_expiry_advances_only_after_acknowledged_renewal(tmp_path: Path) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    controller = AdmissionController(StaticInventory([gpu("GPU-A")]), store)
    lease = LeaseRequest(
        lease_id="renewed",
        attempt_id="attempt-renewed",
        gpu_uuids=("GPU-A",),
        fence_epochs={"GPU-A": 4},
        min_free_vram_bytes=15 * GIB,
        expires_at=110,
        model_key="llama-key",
        renewal_ttl_seconds=30,
    )

    assert controller.admit(lease, now=100).accepted
    assert store.lease_expiry(lease.lease_id) == 110
    assert store.expired_lease_ids(now=109) == ()

    assert controller.renew(lease, now=105) == 1
    assert store.lease_expiry(lease.lease_id) == 135
    assert store.expired_lease_ids(now=134) == ()
    assert store.expired_lease_ids(now=135) == (lease.lease_id,)
    store.close()


def test_multi_gpu_admission_is_all_or_none(tmp_path: Path) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    controller = AdmissionController(StaticInventory([gpu("GPU-A"), gpu("GPU-B")]), store)
    assert controller.admit(request("occupier", ("GPU-B",), 1)).accepted

    group = controller.admit(request("group", ("GPU-A", "GPU-B"), 2))
    assert not group.accepted
    assert store.active_claims() == {"GPU-B": "occupier"}
    assert store.highest_fence("GPU-A") is None

    # Failure of the group did not leave a partial claim on GPU-A.
    assert controller.admit(request("single", ("GPU-A",), 3)).accepted
    assert store.active_claims() == {
        "GPU-A": "single",
        "GPU-B": "occupier",
    }
    store.close()


def test_replica_allocation_publication_is_atomic_and_ranked(tmp_path: Path) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-B",
        actor_id="existing",
        model_key="other",
        state="idle",
    )

    with pytest.raises(GPUUnavailableError, match="allocated to actor"):
        store.reserve_actor_group(
            gpu_uuids=("GPU-A", "GPU-B"),
            actor_id="sharded",
            model_key="llama-key",
            state="idle",
        )
    assert store.actor_allocation("GPU-A") is None

    assert store.evict_actor("existing") == 1
    store.reserve_actor_group(
        gpu_uuids=("GPU-B", "GPU-A"),
        actor_id="sharded",
        model_key="llama-key",
        state="idle",
    )
    group = store.actor_allocations()
    assert [(item.gpu_uuid, item.actor_id, item.replica_rank) for item in group] == [
        ("GPU-B", "sharded", 0),
        ("GPU-A", "sharded", 1),
    ]
    assert store.evict_actor("sharded") == 2
    assert store.actor_allocations() == ()
    store.close()


def test_idle_actor_expiry_is_durable_and_replica_atomic(tmp_path: Path) -> None:
    path = tmp_path / "leases.sqlite3"
    store = LeaseFenceStore(path)
    store.reserve_actor_group(
        gpu_uuids=("GPU-B", "GPU-A"),
        actor_id="sharded",
        model_key="llama-key",
        state="idle",
        now=100,
    )

    assert store.expired_idle_actor_groups(idle_before=99.999) == ()
    candidates = store.expired_idle_actor_groups(idle_before=100)
    assert len(candidates) == 1
    assert tuple(item.gpu_uuid for item in candidates[0]) == ("GPU-B", "GPU-A")
    store.close()

    # The idle deadline survives a process restart. A mismatched gang cannot
    # partially advance the durable state.
    reopened = LeaseFenceStore(path)
    assert not reopened.begin_expired_idle_actor_eviction(
        "sharded",
        expected_gpu_uuids=("GPU-A", "GPU-B"),
        idle_before=100,
        now=101,
    )
    assert {item.state for item in reopened.actor_allocations()} == {"idle"}
    assert reopened.begin_expired_idle_actor_eviction(
        "sharded",
        expected_gpu_uuids=("GPU-B", "GPU-A"),
        idle_before=100,
        now=101,
    )
    assert {item.state for item in reopened.actor_allocations()} == {"stopping"}
    assert (
        reopened.complete_idle_actor_eviction(
            "sharded", expected_gpu_uuids=("GPU-B", "GPU-A")
        )
        == 2
    )
    reopened.close()


def test_idle_actor_expiry_loses_safely_to_reuse_and_restarts_after_release(
    tmp_path: Path,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="actor",
        model_key="llama-key",
        state="idle",
        now=10,
    )
    candidates = store.expired_idle_actor_groups(idle_before=10)
    assert len(candidates) == 1

    # A lease can win after the candidate scan. The atomic begin step must then
    # refuse expiration rather than interrupt that work.
    store.claim(
        request("reuse", ("GPU-A",), 1, reuse_actor_id="actor"),
        fresh_eligible=frozenset(),
        pressure_safe=frozenset({"GPU-A"}),
        now=11,
    )
    assert not store.begin_expired_idle_actor_eviction(
        "actor",
        expected_gpu_uuids=("GPU-A",),
        idle_before=10,
        now=12,
    )
    assert store.expired_idle_actor_groups(idle_before=1_000) == ()

    assert store.release_lease("reuse", now=20) == 1
    assert store.expired_idle_actor_groups(idle_before=19.999) == ()
    assert len(store.expired_idle_actor_groups(idle_before=20)) == 1
    store.close()


def test_complete_sharded_replica_is_reused_as_one_atomic_lease(tmp_path: Path) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor_group(
        gpu_uuids=("GPU-A", "GPU-B"),
        actor_id="sharded-actor",
        model_key="llama-key",
        state="idle",
    )
    process = GPUProcess(123, 10 * GIB, ProcessOwner.TOPODROME)
    controller = AdmissionController(
        StaticInventory(
            [
                gpu("GPU-A", free_gib=4, processes=(process,)),
                gpu("GPU-B", free_gib=4, processes=(process,)),
            ]
        ),
        store,
        warm_actor_validator=lambda allocation: allocation.actor_id == "sharded-actor",
    )

    decision = controller.admit(request("reuse", ("GPU-A", "GPU-B"), 1))

    assert decision.accepted
    assert decision.reused_actor_id == "sharded-actor"
    assert {item.state for item in store.actor_allocations()} == {"busy"}
    assert controller.release("reuse") == 2
    assert {item.state for item in store.actor_allocations()} == {"idle"}
    store.close()


def test_warm_actor_allocation_is_distinct_from_workload_lease(tmp_path: Path) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="actor-1",
        model_key="llama-key",
        state="idle",
    )
    model_process = GPUProcess(123, 25 * GIB, ProcessOwner.TOPODROME)
    controller = AdmissionController(
        StaticInventory([gpu("GPU-A", free_gib=4, processes=(model_process,))]),
        store,
    )

    reuse = controller.admit(request("reuse", ("GPU-A",), 1, reuse_actor_id="actor-1"))
    assert reuse.accepted
    assert store.actor_allocation("GPU-A").state == "busy"  # type: ignore[union-attr]
    controller.release("reuse")
    assert store.actor_allocation("GPU-A").state == "idle"  # type: ignore[union-attr]

    incompatible = controller.admit(request("other", ("GPU-A",), 2))
    assert not incompatible.accepted
    assert "allocated to actor" in (incompatible.reason or "")
    store.close()


def test_live_compatible_warm_actor_can_be_selected_locally(tmp_path: Path) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="actor-1",
        model_key="llama-key",
        state="idle",
    )
    model_process = GPUProcess(123, 25 * GIB, ProcessOwner.TOPODROME)
    controller = AdmissionController(
        StaticInventory([gpu("GPU-A", free_gib=4, processes=(model_process,))]),
        store,
        warm_actor_validator=lambda allocation: allocation.actor_id == "actor-1",
    )

    decision = controller.admit(request("reuse", ("GPU-A",), 1))

    assert decision.accepted
    assert decision.reused_actor_id == "actor-1"
    controller.release("reuse")
    store.close()


def test_only_explicitly_cooperative_load_is_reclaimable(tmp_path: Path) -> None:
    cooperative = GPUProcess(700, 512 * 1024**2, ProcessOwner.COOPERATIVE)
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    controller = AdmissionController(
        StaticInventory([gpu("GPU-A", free_gib=20, utilization=99, processes=(cooperative,))]),
        store,
    )

    decision = controller.admit(request("cooperative", ("GPU-A",), 1))
    assert decision.accepted
    assert decision.cooperative_pids == {"GPU-A": (700,)}
    controller.release("cooperative")

    ordinary = GPUProcess(701, 512 * 1024**2, ProcessOwner.FOREIGN)
    controller = AdmissionController(
        StaticInventory([gpu("GPU-A", free_gib=20, utilization=99, processes=(ordinary,))]),
        store,
    )
    rejected = controller.admit(request("foreign", ("GPU-A",), 2))
    assert not rejected.accepted
    assert "foreign pressure" in (rejected.reason or "")
    store.close()


def test_cooperative_idle_worker_does_not_hide_a_genuine_foreign_job(tmp_path: Path) -> None:
    cooperative = GPUProcess(700, 512 * 1024**2, ProcessOwner.COOPERATIVE)
    ordinary = GPUProcess(701, 512 * 1024**2, ProcessOwner.FOREIGN)
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    controller = AdmissionController(
        StaticInventory(
            [
                gpu(
                    "GPU-A",
                    free_gib=20,
                    utilization=99,
                    processes=(cooperative, ordinary),
                )
            ]
        ),
        store,
    )

    decision = controller.admit(request("mixed", ("GPU-A",), 1))

    assert not decision.accepted
    assert "foreign pressure" in (decision.reason or "")
    assert store.active_claims() == {}
    store.close()


def test_configured_foreign_colocation_uses_free_vram_not_utilization(
    tmp_path: Path,
) -> None:
    ordinary = GPUProcess(800, 3 * GIB, ProcessOwner.FOREIGN)
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    controller = AdmissionController(
        StaticInventory([gpu("GPU-A", free_gib=20, utilization=100, processes=(ordinary,))]),
        store,
        safety_margin_bytes=5 * GIB,
        allow_foreign_processes_when_capacity_sufficient=True,
    )

    accepted = controller.admit(request("shared", ("GPU-A",), 1))
    assert accepted.accepted
    controller.release("shared")

    controller = AdmissionController(
        StaticInventory([gpu("GPU-A", free_gib=19.9, utilization=100, processes=(ordinary,))]),
        store,
        safety_margin_bytes=5 * GIB,
        allow_foreign_processes_when_capacity_sufficient=True,
    )
    rejected = controller.admit(request("too-small", ("GPU-A",), 2))
    assert not rejected.accepted
    assert "capacity" in (rejected.reason or "")
    store.close()
