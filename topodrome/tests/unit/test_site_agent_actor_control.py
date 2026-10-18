from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from topodrome.site_agent.admission import (
    ActorAllocation,
    GPUUnavailableError,
    LeaseFenceStore,
    LeaseRequest,
)
from topodrome.site_agent.agent import Assignment, PreparedRuntime, SiteAgent
from topodrome.site_agent.inventory import GPUSnapshot, ProcessOwnershipRegistry

pytestmark = pytest.mark.unit


def _replica_key_hash(model_key: str) -> str:
    return hashlib.sha256(model_key.encode("utf-8")).hexdigest()


def _group(store: LeaseFenceStore, actor_id: str) -> tuple[ActorAllocation, ...]:
    return tuple(
        allocation for allocation in store.actor_allocations() if allocation.actor_id == actor_id
    )


def test_scheduler_eviction_intent_is_exact_durable_and_idempotent(tmp_path: Any) -> None:
    path = tmp_path / "leases.sqlite3"
    store = LeaseFenceStore(path)
    store.reserve_actor_group(
        gpu_uuids=("GPU-B", "GPU-A"),
        actor_id="ray-replica-id",
        model_key="llama-runtime-fingerprint",
        state="idle",
    )

    intent = store.begin_scheduler_actor_eviction(
        "eviction-1",
        actor_id="ray-replica-id",
        expected_replica_key_hash=_replica_key_hash("llama-runtime-fingerprint"),
        expected_gpu_uuids=("GPU-B", "GPU-A"),
        now=10,
    )
    assert intent.state == "stopping"
    assert {item.state for item in store.actor_allocations()} == {"stopping"}

    replay = store.begin_scheduler_actor_eviction(
        "eviction-1",
        actor_id="ray-replica-id",
        expected_replica_key_hash=_replica_key_hash("llama-runtime-fingerprint"),
        expected_gpu_uuids=("GPU-B", "GPU-A"),
        now=11,
    )
    assert replay == intent
    with pytest.raises(GPUUnavailableError, match="different actor coordinates"):
        store.begin_scheduler_actor_eviction(
            "eviction-1",
            actor_id="scheduler-synthetic-row-id",
            expected_replica_key_hash=_replica_key_hash("llama-runtime-fingerprint"),
            expected_gpu_uuids=("GPU-B", "GPU-A"),
        )

    assert store.complete_scheduler_actor_eviction("eviction-1") == 2
    assert store.actor_allocations() == ()
    assert store.complete_scheduler_actor_eviction("eviction-1") == 0
    store.close()

    reopened = LeaseFenceStore(path)
    completed = reopened.begin_scheduler_actor_eviction(
        "eviction-1",
        actor_id="ray-replica-id",
        expected_replica_key_hash=_replica_key_hash("llama-runtime-fingerprint"),
        expected_gpu_uuids=("GPU-B", "GPU-A"),
    )
    assert completed.state == "completed"
    reopened.close()


def test_heartbeat_capability_uses_durable_local_actor_idle_clock(tmp_path: Any) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="ray-replica-id",
        model_key="model",
        state="idle",
        now=100,
    )
    agent = object.__new__(SiteAgent)
    agent.fences = store
    capability = {
        "availability": "warm",
        "gpu_uuid": "GPU-A",
        "replica_id": "ray-replica-id",
        "replica_gpu_uuids": ["GPU-A"],
    }

    published = agent._publishable_model_capabilities((capability,))
    assert datetime.fromisoformat(published[0]["idle_since"]) == datetime.fromtimestamp(
        100,
        tz=UTC,
    )

    store.set_actor_state("ray-replica-id", "busy", now=200)
    assert "idle_since" not in agent._publishable_model_capabilities((capability,))[0]
    store.close()


def test_scheduler_eviction_rejects_busy_claim_and_wrong_replica_shape(tmp_path: Any) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor_group(
        gpu_uuids=("GPU-A", "GPU-B"),
        actor_id="replica",
        model_key="model",
        state="idle",
    )
    with pytest.raises(GPUUnavailableError, match="complete idle replica"):
        store.begin_scheduler_actor_eviction(
            "wrong-order",
            actor_id="replica",
            expected_replica_key_hash=_replica_key_hash("model"),
            expected_gpu_uuids=("GPU-B", "GPU-A"),
        )
    with pytest.raises(GPUUnavailableError, match="replica key hash"):
        store.begin_scheduler_actor_eviction(
            "wrong-model",
            actor_id="replica",
            expected_replica_key_hash=_replica_key_hash("other-model"),
            expected_gpu_uuids=("GPU-A", "GPU-B"),
        )
    assert {item.state for item in store.actor_allocations()} == {"idle"}

    lease = LeaseRequest(
        lease_id="lease",
        attempt_id="attempt",
        gpu_uuids=("GPU-A", "GPU-B"),
        fence_epochs={"GPU-A": 1, "GPU-B": 1},
        min_free_vram_bytes=1,
        expires_at=100,
        reuse_actor_id="replica",
        model_key="model",
    )
    store.claim(
        lease,
        fresh_eligible=frozenset(),
        pressure_safe=frozenset({"GPU-A", "GPU-B"}),
        now=1,
    )
    with pytest.raises(GPUUnavailableError, match="complete idle replica"):
        store.begin_scheduler_actor_eviction(
            "busy",
            actor_id="replica",
            expected_replica_key_hash=_replica_key_hash("model"),
            expected_gpu_uuids=("GPU-A", "GPU-B"),
        )
    assert store.actor_eviction_intent("busy") is None
    assert {item.state for item in store.actor_allocations()} == {"busy"}
    store.close()


def test_scheduler_eviction_rechecks_minimum_idle_age_on_site(tmp_path: Any) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="replica",
        model_key="model",
        state="idle",
        now=100,
    )

    with pytest.raises(GPUUnavailableError, match="required interval"):
        store.begin_scheduler_actor_eviction(
            "too-early",
            actor_id="replica",
            expected_replica_key_hash=_replica_key_hash("model"),
            expected_gpu_uuids=("GPU-A",),
            minimum_idle_seconds=1200,
            now=1299.999,
        )
    assert store.actor_eviction_intent("too-early") is None
    assert (
        store.begin_scheduler_actor_eviction(
            "expired",
            actor_id="replica",
            expected_replica_key_hash=_replica_key_hash("model"),
            expected_gpu_uuids=("GPU-A",),
            minimum_idle_seconds=1200,
            now=1300,
        ).state
        == "stopping"
    )
    store.close()


def test_batch_eviction_fence_is_all_or_none_across_source_actors(tmp_path: Any) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(gpu_uuid="GPU-A", actor_id="llama-a", model_key="llama", state="idle")
    store.reserve_actor(gpu_uuid="GPU-B", actor_id="llama-b", model_key="llama", state="idle")
    first, second = (_group(store, actor_id) for actor_id in ("llama-a", "llama-b"))

    store.set_actor_state("llama-b", "busy")
    assert not store.begin_idle_actor_eviction_batch((first, second), now=10)
    assert store.actor_allocation("GPU-A").state == "idle"  # type: ignore[union-attr]
    assert store.actor_allocation("GPU-B").state == "busy"  # type: ignore[union-attr]

    store.set_actor_state("llama-b", "idle")
    refreshed = (_group(store, actor_id) for actor_id in ("llama-a", "llama-b"))
    assert store.begin_idle_actor_eviction_batch(tuple(refreshed), now=11)
    assert {item.state for item in store.actor_allocations()} == {"stopping"}
    store.close()


class _Sink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    async def publish(self, kind: str, payload: Any) -> None:
        self.events.append((kind, dict(payload)))


class _ExactBackend:
    def __init__(self, live: set[str], *, fail: bool = False) -> None:
        self.live = live
        self.fail = fail
        self.calls: list[tuple[ActorAllocation, ...]] = []

    def is_warm_allocation(self, allocation: ActorAllocation) -> bool:
        return allocation.actor_id in self.live

    async def evict_warm_replica(self, allocations: tuple[ActorAllocation, ...]) -> tuple[int, ...]:
        self.calls.append(tuple(allocations))
        if self.fail:
            raise RuntimeError("ray control failed")
        self.live.remove(allocations[0].actor_id)
        return (4101,)


def _agent(
    store: LeaseFenceStore,
    backend: _ExactBackend,
    sink: _Sink,
    ownership: ProcessOwnershipRegistry | None = None,
) -> SiteAgent:
    agent = object.__new__(SiteAgent)
    agent.site_id = "lab1"
    agent.fences = store
    agent.backend = cast(Any, backend)
    agent.outbound = cast(Any, sink)
    agent.process_ownership = ownership
    agent._active = {}
    agent._active_lock = asyncio.Lock()
    agent._warm_maintenance_lock = asyncio.Lock()
    agent._kernel_hosts = {}
    return agent


def _command(*, message_id: str = "command-1") -> dict[str, object]:
    return {
        "message_id": message_id,
        "kind": "evict_actor",
        "payload": {
            "eviction_id": "eviction-1",
            "actor_id": "ray-actor-id",
            "replica_key_hash": _replica_key_hash("exact-model-runtime"),
            "gpu_uuids": ["GPU-A", "GPU-B"],
        },
    }


@pytest.mark.asyncio
async def test_evict_actor_command_releases_exact_idle_replica_and_replays(
    tmp_path: Any,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor_group(
        gpu_uuids=("GPU-A", "GPU-B"),
        actor_id="ray-actor-id",
        model_key="exact-model-runtime",
        state="idle",
    )
    ownership = ProcessOwnershipRegistry(proc_root=tmp_path / "proc")
    ownership.register(4101, actor_id="ray-actor-id")
    ownership.register(9999, actor_id="unrelated-owned-actor")
    sink = _Sink()
    backend = _ExactBackend({"ray-actor-id"})
    agent = _agent(store, backend, sink, ownership)

    await agent.handle_command(_command())
    await agent.handle_command(_command(message_id="command-replay"))

    acknowledgements = [payload for kind, payload in sink.events if kind == "command_ack"]
    assert [item["accepted"] for item in acknowledgements] == [True, True]
    assert all(item["eviction_state"] == "completed" for item in acknowledgements)
    assert len(backend.calls) == 1
    assert tuple(item.gpu_uuid for item in backend.calls[0]) == ("GPU-A", "GPU-B")
    assert store.actor_allocations() == ()
    assert ownership.exact_owner(4101) is None
    assert ownership.exact_owner(9999) is not None
    store.close()


@pytest.mark.asyncio
async def test_evict_actor_command_never_accepts_pid_targets(tmp_path: Any) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="ray-actor-id",
        model_key="exact-model-runtime",
        state="idle",
    )
    sink = _Sink()
    backend = _ExactBackend({"ray-actor-id"})
    agent = _agent(store, backend, sink)
    command = _command()
    cast(dict[str, object], command["payload"])["worker_pids"] = [9999]

    await agent.handle_command(command)

    acknowledgement = sink.events[-1][1]
    assert acknowledgement["accepted"] is False
    assert "must not contain process identities" in str(acknowledgement["error"])
    assert backend.calls == []
    assert {item.state for item in store.actor_allocations()} == {"idle"}
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["attempt", "kernel"])
async def test_evict_actor_command_refuses_active_attempt_or_kernel(
    tmp_path: Any,
    reference: str,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor_group(
        gpu_uuids=("GPU-A", "GPU-B"),
        actor_id="ray-actor-id",
        model_key="exact-model-runtime",
        state="idle",
    )
    sink = _Sink()
    backend = _ExactBackend({"ray-actor-id"})
    agent = _agent(store, backend, sink)
    assignment = SimpleNamespace(lease=SimpleNamespace(gpu_uuids=("GPU-A", "GPU-B")))
    prepared = PreparedRuntime("ray-actor-id", "exact-model-runtime", True)
    if reference == "attempt":
        agent._active = cast(
            Any,
            {"attempt": SimpleNamespace(assignment=assignment, prepared=prepared)},
        )
    else:
        agent._kernel_hosts = cast(
            Any,
            {"attempt": SimpleNamespace(prepared=prepared)},
        )

    await agent.handle_command(_command())

    acknowledgement = sink.events[-1][1]
    assert acknowledgement["accepted"] is False
    assert "attempt or kernel references it" in str(acknowledgement["error"])
    assert backend.calls == []
    assert store.actor_eviction_intent("eviction-1") is None
    assert {item.state for item in store.actor_allocations()} == {"idle"}
    store.close()


@pytest.mark.asyncio
async def test_failed_exact_teardown_restores_retryable_intent(tmp_path: Any) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor_group(
        gpu_uuids=("GPU-A", "GPU-B"),
        actor_id="ray-actor-id",
        model_key="exact-model-runtime",
        state="idle",
    )
    sink = _Sink()
    backend = _ExactBackend({"ray-actor-id"}, fail=True)
    agent = _agent(store, backend, sink)

    await agent.handle_command(_command())
    assert sink.events[-1][1]["accepted"] is False
    assert store.actor_eviction_intent("eviction-1").state == "retryable"  # type: ignore[union-attr]
    assert {item.state for item in store.actor_allocations()} == {"idle"}

    backend.fail = False
    await agent.handle_command(_command(message_id="retry"))
    assert sink.events[-1][1]["accepted"] is True
    assert store.actor_eviction_intent("eviction-1").state == "completed"
    store.close()


@pytest.mark.asyncio
async def test_assignment_replacement_fences_all_sources_before_first_teardown(
    tmp_path: Any,
) -> None:
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    for gpu_uuid, actor_id in (("GPU-A", "llama-a"), ("GPU-B", "llama-b")):
        store.reserve_actor(
            gpu_uuid=gpu_uuid,
            actor_id=actor_id,
            model_key="llama",
            state="idle",
        )

    class Backend(_ExactBackend):
        async def evict_warm_replica(
            self, allocations: tuple[ActorAllocation, ...]
        ) -> tuple[int, ...]:
            # Even before the first Ray teardown, both source actors are
            # durably unavailable to a racing lease.
            assert {item.state for item in store.actor_allocations()} == {"stopping"}
            self.calls.append(tuple(allocations))
            self.live.remove(allocations[0].actor_id)
            return ()

    backend = Backend({"llama-a", "llama-b"})
    agent = _agent(store, backend, _Sink())
    agent.inventory = cast(
        Any,
        SimpleNamespace(
            sample=lambda: tuple(
                GPUSnapshot(
                    uuid=gpu_uuid,
                    index=index,
                    name="GPU",
                    total_memory_bytes=80 * 1024**3,
                    free_memory_bytes=80 * 1024**3,
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
        job_id="gemma-job",
        attempt_id="gemma-attempt",
        lease=LeaseRequest(
            lease_id="gemma-lease",
            attempt_id="gemma-attempt",
            gpu_uuids=("GPU-A", "GPU-B"),
            fence_epochs={"GPU-A": 2, "GPU-B": 2},
            min_free_vram_bytes=1,
            expires_at=4_000_000_000,
            model_key="gemma-sharded",
        ),
        model_key="gemma-sharded",
        entrypoint="experiment.py",
    )

    assert await agent._evict_private_slot_actors(assignment) == ("GPU-A", "GPU-B")
    assert [group[0].actor_id for group in backend.calls] == ["llama-a", "llama-b"]
    assert store.actor_allocations() == ()
    store.close()


@pytest.mark.asyncio
async def test_stale_single_gpu_replacement_flag_preserves_an_exact_warm_actor(tmp_path):
    store = LeaseFenceStore(tmp_path / "leases.sqlite3")
    store.reserve_actor(gpu_uuid="GPU-A", actor_id="resident", model_key="target", state="idle")
    backend = _ExactBackend({"resident"})
    agent = _agent(store, backend, _Sink())
    assignment = Assignment(
        "job",
        "attempt",
        LeaseRequest(
            "lease",
            "attempt",
            ("GPU-A",),
            {"GPU-A": 2},
            1,
            4_000_000_000,
            model_key="target",
            replace_idle_actors=True,
        ),
        "target",
        "run.py",
        raw_spec={"model": {"parallelism": {"mode": "single"}}},
    )
    assert await agent._evict_private_slot_actors(assignment) == ()
    assert not backend.calls and store.actor_allocation("GPU-A").state == "idle"
    store.close()


def test_crash_left_stopping_intent_completes_during_exact_reconciliation(
    tmp_path: Any,
) -> None:
    path = tmp_path / "leases.sqlite3"
    store = LeaseFenceStore(path)
    store.reserve_actor(
        gpu_uuid="GPU-A",
        actor_id="ray-actor-id",
        model_key="exact-model-runtime",
        state="idle",
    )
    store.begin_scheduler_actor_eviction(
        "eviction-1",
        actor_id="ray-actor-id",
        expected_replica_key_hash=_replica_key_hash("exact-model-runtime"),
        expected_gpu_uuids=("GPU-A",),
    )
    assert store.restore_idle_actor_eviction("ray-actor-id", expected_gpu_uuids=("GPU-A",))
    assert store.actor_eviction_intent("eviction-1").state == "retryable"  # type: ignore[union-attr]
    store.begin_scheduler_actor_eviction(
        "eviction-1",
        actor_id="ray-actor-id",
        expected_replica_key_hash=_replica_key_hash("exact-model-runtime"),
        expected_gpu_uuids=("GPU-A",),
    )
    store.close()

    recovered = LeaseFenceStore(path)
    assert (
        recovered.complete_idle_actor_eviction("ray-actor-id", expected_gpu_uuids=("GPU-A",)) == 1
    )
    assert recovered.actor_eviction_intent("eviction-1").state == "completed"  # type: ignore[union-attr]
    recovered.close()
