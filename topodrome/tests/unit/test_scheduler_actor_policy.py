from __future__ import annotations

from dataclasses import replace

import pytest

from topodrome.config import SchedulerSettings
from topodrome.domain.enums import ActorState, SiteKind, SiteState
from topodrome.domain.specs import JobSpec, utc_now
from topodrome.scheduler.actor_policy import (
    InstitutionalActorPolicy,
    occupied_institutional_gpu_count,
)
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
from topodrome.scheduler.placement import GPUCandidate, SiteCandidate
from topodrome.scheduler.service import SchedulerService

pytestmark = pytest.mark.unit
GIB = 1024**3


def _gpu(
    site_id: str,
    index: int,
    *,
    actor_id: str | None = None,
    repo_id: str = "test/llama",
    members: tuple[str, ...] | None = None,
    foreign: bool = False,
    active_lease: bool = False,
    idle_since: float = 0,
) -> GPUCandidate:
    uuid = f"{site_id}-gpu-{index}"
    actor_members = members or (uuid,)
    return GPUCandidate(
        uuid=uuid,
        index=index,
        name="NVIDIA A800-SXM4-80GB",
        total_memory_bytes=80 * GIB,
        free_memory_bytes=(60 if actor_id else 80) * GIB,
        utilization_percent=3 if foreign else 0,
        foreign_process_count=1 if foreign else 0,
        foreign_memory_bytes=GIB if foreign else 0,
        externally_reserved=False,
        healthy=True,
        active_lease=active_lease,
        actor_key_hash=f"hash-{actor_id}-{index}" if actor_id else None,
        actor_state=ActorState.IDLE if actor_id else None,
        actor_key_data=(
            {
                "repo_id": repo_id,
                "revision": "rev",
                "backend": "transformers",
                "backend_version": "4.57.6",
                "dtype": "bfloat16",
                "quantization": None,
                "adapter": None,
                "runtime_fingerprint": "a" * 64,
                "gpu_uuid": uuid,
                "replica_id": actor_id,
                "replica_gpu_uuids": list(actor_members),
                "replica_key_hash": "b" * 64,
                "idle_since": idle_since,
            }
            if actor_id
            else None
        ),
    )


def _site(site_id: str, gpus: tuple[GPUCandidate, ...]) -> SiteCandidate:
    return SiteCandidate(
        site_id=site_id,
        kind=SiteKind.LAB if site_id == "lab1" else SiteKind.JUPYTER,
        state=SiteState.READY,
        heartbeat_at=utc_now(),
        storage_available=True,
        supported_workloads=frozenset({"batch"}),
        queue_depth=0,
        reliability=1,
        paid=False,
        region=None,
        gpus=gpus,
        models=(),
        agent_connection_generation=f"generation-{site_id}",
    )


def _job(repo_id: str) -> JobSpec:
    return JobSpec(entrypoint="experiment.py", model={"repo_id": repo_id})


def _choose(
    policy: InstitutionalActorPolicy,
    sites: list[SiteCandidate],
    queued: tuple[JobSpec, ...] = (),
    *,
    now: float,
    threshold: int = 5,
    stability: float = 30,
    idle_timeout: float | None = 1200,
    overrides: dict[str, float | None] | None = None,
):
    return policy.choose_eviction(
        sites,
        queued,
        actor_floor_gpu_count=1,
        contention_threshold_gpu_count=threshold,
        stability_seconds=stability,
        idle_timeout_seconds=idle_timeout,
        idle_timeout_seconds_by_site=overrides,
        now=now,
    )


def test_occupied_count_is_union_of_leases_and_resident_actors() -> None:
    sites = [
        _site(
            "lab1",
            (
                _gpu("lab1", 0, actor_id="actor-a", active_lease=True),
                _gpu("lab1", 1, actor_id="actor-b"),
                _gpu("lab1", 2, active_lease=True),
                _gpu("lab1", 3),
            ),
        )
    ]

    assert occupied_institutional_gpu_count(sites) == 3


def test_contention_shedding_requires_threshold_stability_and_preserves_one_actor_for_queue() -> (
    None
):
    policy = InstitutionalActorPolicy()
    sites = [
        _site(
            "lab1",
            (
                _gpu("lab1", 0, actor_id="contended", foreign=True),
                _gpu("lab1", 1, actor_id="protected"),
                _gpu("lab1", 2, active_lease=True),
                _gpu("lab1", 3, active_lease=True),
                _gpu("lab1", 4, active_lease=True),
            ),
        )
    ]

    queued = (_job("test/queued-model"),)
    assert _choose(policy, sites, queued, now=0) is None
    assert _choose(policy, sites, queued, now=29.999) is None
    plan = _choose(policy, sites, queued, now=30)

    assert plan is not None
    assert plan.actor_id == "contended"
    assert plan.gpu_uuids == ("lab1-gpu-0",)
    assert plan.connection_generation == "generation-lab1"
    assert plan.reason_code == "foreign_contention_surplus"
    assert plan.occupied_institutional_gpu_count == 5

    # Once only the protected warm membership remains, no automatic policy may
    # evict it even though active/private lease occupancy keeps the threshold up.
    one_actor = [replace(sites[0], gpus=sites[0].gpus[1:])]
    assert _choose(policy, one_actor, queued, now=2000, stability=0) is None


def test_contention_shedding_does_not_activate_below_five_occupied_gpus() -> None:
    policy = InstitutionalActorPolicy()
    sites = [
        _site(
            "lab1",
            (
                _gpu("lab1", 0, actor_id="contended", foreign=True),
                _gpu("lab1", 1, actor_id="other"),
                _gpu("lab1", 2, active_lease=True),
                _gpu("lab1", 3, active_lease=True),
            ),
        )
    ]

    assert _choose(policy, sites, now=0, stability=0, idle_timeout=None) is None


def test_excess_idle_timeout_uses_durable_age_and_preserves_floor_for_queue() -> None:
    policy = InstitutionalActorPolicy()
    sites = [
        _site(
            "lab1",
            (
                _gpu("lab1", 0, actor_id="older"),
                _gpu("lab1", 1, actor_id="floor"),
            ),
        )
    ]

    queued = (_job("test/queued-model"),)
    assert _choose(policy, sites, queued, now=1199.999, threshold=5) is None
    restarted_policy = InstitutionalActorPolicy()
    plan = _choose(restarted_policy, sites, queued, now=1200, threshold=5)
    assert plan is not None
    assert plan.reason_code == "excess_idle_warm_actor"

    remaining = [replace(sites[0], gpus=(sites[0].gpus[1],))]
    assert _choose(restarted_policy, remaining, queued, now=3000, threshold=5) is None


def test_per_site_idle_timeouts_preserve_default_floor_and_active_work() -> None:
    policy = InstitutionalActorPolicy()
    overrides = {"lab1": 300, "remote-jupyter-1": 300}
    for site_id, expiry in (("lab1", 300), ("remote-jupyter-1", 300), ("sqz", 1200)):
        gpu = _gpu(site_id, 0, actor_id="warm")
        sites = [_site(site_id, (gpu,))]
        assert _choose(policy, sites, now=expiry - 0.001, overrides=overrides) is None
        plan = _choose(policy, sites, now=expiry, overrides=overrides)
        assert plan is not None and plan.minimum_idle_seconds == expiry
        assert _choose(policy, sites, (_job("test/model"),), now=5000, overrides=overrides) is None
        busy = [_site(site_id, (replace(gpu, active_lease=True),))]
        assert _choose(policy, busy, now=5000, overrides=overrides) is None
    sites = [_site("lab1", (_gpu("lab1", 0, actor_id="warm"),))]
    assert _choose(policy, sites, now=5000, overrides={"lab1": None}) is None
    plan = _choose(policy, sites, now=300, idle_timeout=None, overrides=overrides)
    assert plan is not None and plan.minimum_idle_seconds == 300


def test_idle_timeout_may_release_final_actor_when_queue_is_empty() -> None:
    policy = InstitutionalActorPolicy()
    sites = [_site("lab1", (_gpu("lab1", 0, actor_id="last-warm"),))]

    plan = _choose(policy, sites, now=1200, threshold=5)

    assert plan is not None
    assert plan.actor_id == "last-warm"
    assert plan.minimum_idle_seconds == 1200


def test_queue_demand_prefers_shedding_an_unrequested_model() -> None:
    policy = InstitutionalActorPolicy()
    sites = [
        _site(
            "lab1",
            (
                _gpu("lab1", 0, actor_id="llama", repo_id="test/llama", foreign=True),
                _gpu("lab1", 1, actor_id="gemma", repo_id="test/gemma", foreign=True),
                _gpu("lab1", 2, active_lease=True),
                _gpu("lab1", 3, active_lease=True),
                _gpu("lab1", 4, active_lease=True),
            ),
        )
    ]

    plan = _choose(policy, sites, (_job("test/gemma"),), now=0, stability=0)
    assert plan is not None
    assert plan.actor_id == "llama"


def test_multi_gpu_actor_is_selected_and_reported_as_one_complete_replica() -> None:
    members = ("lab1-gpu-0", "lab1-gpu-1")
    policy = InstitutionalActorPolicy()
    sites = [
        _site(
            "lab1",
            (
                _gpu(
                    "lab1",
                    0,
                    actor_id="sharded-actor",
                    members=members,
                    foreign=True,
                ),
                _gpu(
                    "lab1",
                    1,
                    actor_id="sharded-actor",
                    members=members,
                    foreign=True,
                ),
                _gpu("lab1", 2, actor_id="floor"),
                _gpu("lab1", 3, active_lease=True),
                _gpu("lab1", 4, active_lease=True),
            ),
        )
    ]

    plan = _choose(policy, sites, now=0, stability=0)
    assert plan is not None
    assert plan.actor_id == "sharded-actor"
    assert plan.gpu_uuids == members


def test_dispatched_plan_is_cooled_down_until_heartbeat_removes_it() -> None:
    policy = InstitutionalActorPolicy()
    sites = [
        _site(
            "lab1",
            (
                _gpu("lab1", 0, actor_id="contended", foreign=True),
                _gpu("lab1", 1, actor_id="floor"),
                _gpu("lab1", 2, active_lease=True),
                _gpu("lab1", 3, active_lease=True),
                _gpu("lab1", 4, active_lease=True),
            ),
        )
    ]
    plan = _choose(policy, sites, now=0, stability=0)
    assert plan is not None
    policy.mark_dispatched(plan, cooldown_seconds=300, now=0)

    assert _choose(policy, sites, now=299, stability=0) is None
    assert _choose(policy, sites, now=300, stability=0) is not None


def test_eviction_id_is_stable_for_retries_and_changes_for_later_residency() -> None:
    policy = InstitutionalActorPolicy()
    sites = [
        _site(
            "lab1",
            (
                _gpu("lab1", 0, actor_id="contended", foreign=True),
                _gpu("lab1", 1, actor_id="floor"),
                _gpu("lab1", 2, active_lease=True),
                _gpu("lab1", 3, active_lease=True),
                _gpu("lab1", 4, active_lease=True),
            ),
        )
    ]

    first = _choose(policy, sites, now=0, stability=0)
    retry = _choose(policy, sites, now=1, stability=0)
    assert first is not None
    assert retry is not None
    assert first.eviction_id(policy_generation=7) == retry.eviction_id(policy_generation=7)

    without_target = [replace(sites[0], gpus=sites[0].gpus[1:])]
    assert _choose(policy, without_target, now=2, stability=0) is None
    later = _choose(policy, sites, now=3, stability=0)
    assert later is not None
    assert later.eviction_id(policy_generation=7) != first.eviction_id(policy_generation=7)


@pytest.mark.asyncio
async def test_scheduler_delivers_actual_replica_identity_and_opaque_key_hash(
    database,
    tmp_path,
) -> None:
    connections = AgentConnectionRegistry()
    deliveries: list[dict[str, object]] = []

    async def capture(message: dict[str, object]) -> None:
        deliveries.append(message)

    connection = AgentConnection("lab1", "lab1-session", capture)
    await connections.register(connection)
    gpus = tuple(
        _gpu(
            "lab1",
            index,
            actor_id=f"site-ray-actor-{index}",
            foreign=index == 0,
        )
        for index in range(5)
    )
    candidate = replace(
        _site("lab1", gpus),
        agent_session_id="lab1-session",
        agent_connection_generation=connection.generation,
    )

    class InventoryRepository:
        async def placement_inventory(self, _session):
            return [candidate]

    service = SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path,
            fence_signing_key="f" * 64,
            scheduling={
                "institutional_actor_floor_gpu_count": 1,
                "contended_actor_shedding_threshold_gpu_count": 5,
                "actor_policy_stability_seconds": 0,
                "institutional_warm_actor_idle_timeout_seconds": None,
            },
        ),
        connections=connections,
        repository=InventoryRepository(),  # type: ignore[arg-type]
    )

    assert await service.reconcile_institutional_actors_once() == 1
    assert len(deliveries) == 1
    command = deliveries[0]
    assert command["kind"] == "evict_actor"
    payload = command["payload"]
    assert isinstance(payload, dict)
    assert str(payload["eviction_id"]).startswith("actor-policy:1:")
    assert payload["actor_id"] == "site-ray-actor-0"
    assert payload["gpu_uuids"] == ["lab1-gpu-0"]
    assert payload["replica_key_hash"] == "b" * 64
    assert payload["policy_generation"] == 1
    assert payload["minimum_idle_seconds"] is None
