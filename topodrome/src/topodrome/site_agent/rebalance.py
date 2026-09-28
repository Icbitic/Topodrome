"""Plan conservative eviction of idle warm actors from contended GPUs."""

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .admission import ActorAllocation
from .inventory import GPUSnapshot
from .pressure import MIB, PressureEvent, PressurePolicy


@dataclass(frozen=True, slots=True)
class IdleActorRebalancePolicy:
    """Hysteretic policy for yielding only idle, exactly owned model actors."""

    enabled: bool = True
    sustained_seconds: float = 30.0
    cooldown_seconds: float = 300.0
    foreign_memory_threshold_bytes: int = 256 * MIB
    pressure_utilization_threshold_percent: int = 20
    light_contention_utilization_percent: float = 10.0
    allow_foreign_processes_when_capacity_sufficient: bool = False

    def __post_init__(self) -> None:
        if self.sustained_seconds < 0:
            raise ValueError("sustained contention duration cannot be negative")
        if self.cooldown_seconds < 0:
            raise ValueError("idle actor rebalance cooldown cannot be negative")
        if self.foreign_memory_threshold_bytes < 0:
            raise ValueError("foreign memory threshold cannot be negative")
        if not 0 <= self.pressure_utilization_threshold_percent <= 100:
            raise ValueError("pressure utilization threshold must be between 0 and 100")
        if not 0 <= self.light_contention_utilization_percent <= 100:
            raise ValueError("light contention utilization threshold must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class IdleActorEvictionPlan:
    """One exact warm replica and the better same-site capacity justifying eviction."""

    actor_id: str
    allocations: tuple[ActorAllocation, ...]
    alternative_gpu_uuids: tuple[str, ...]
    required_memory_bytes: int
    alternative_band: Literal["clean", "light"]

    @property
    def gpu_uuids(self) -> tuple[str, ...]:
        return tuple(allocation.gpu_uuid for allocation in self.allocations)


class IdleActorRebalancer:
    """Detect new contention and plan idle-only, replica-atomic evictions.

    This object deliberately does not move or start an actor.  Removing the
    contended idle copy lets the normal scheduler recreate it lazily when real
    demand arrives, using the scheduler's clean/light/heavy placement bands.
    """

    def __init__(
        self,
        policy: IdleActorRebalancePolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self._pressure_policy = PressurePolicy(
            enabled=policy.enabled,
            sustained_seconds=policy.sustained_seconds,
            foreign_memory_threshold_bytes=policy.foreign_memory_threshold_bytes,
            utilization_threshold_percent=(policy.pressure_utilization_threshold_percent),
        )
        self._clock = clock
        self._pressure_started_at: dict[str, float] = {}

    def observe(
        self,
        snapshots: Sequence[GPUSnapshot],
        *,
        now: float | None = None,
    ) -> tuple[PressureEvent, ...]:
        """Return every currently sustained contention episode for replanning.

        Unlike active-work preemption, an idle-actor rebalance can harmlessly
        miss because no better target exists yet or because a target wins a
        concurrent lease claim. Keep reporting pressure after the hysteresis
        interval so a later-free target can be considered. The durable
        source/target cooldown and atomic eviction fence remain the authority
        that prevents repeated movement.
        """

        observed_at = self._clock() if now is None else now
        present = {snapshot.uuid for snapshot in snapshots}
        for missing in set(self._pressure_started_at) - present:
            self._pressure_started_at.pop(missing, None)

        events: list[PressureEvent] = []
        for snapshot in snapshots:
            if not self._pressure_policy.is_pressure(snapshot):
                self._pressure_started_at.pop(snapshot.uuid, None)
                continue
            started_at = self._pressure_started_at.setdefault(snapshot.uuid, observed_at)
            if observed_at - started_at < self.policy.sustained_seconds:
                continue
            events.append(
                PressureEvent(
                    gpu_uuid=snapshot.uuid,
                    first_observed_at=started_at,
                    observed_at=observed_at,
                    foreign_pids=tuple(process.pid for process in snapshot.foreign_processes),
                )
            )
        return tuple(events)

    def plans(
        self,
        events: Sequence[PressureEvent],
        snapshots: Sequence[GPUSnapshot],
        allocations: Sequence[ActorAllocation],
        active_claims: Mapping[str, str],
        *,
        safety_margin_bytes: int,
        protected_gpu_uuids: frozenset[str] = frozenset(),
    ) -> tuple[IdleActorEvictionPlan, ...]:
        """Choose disjoint better gangs for complete idle replicas.

        The returned alternatives are evidence for yielding, not reservations.
        The caller must recheck local durable state atomically immediately before
        stopping an actor.
        """

        if not self.policy.enabled or not events:
            return ()
        if safety_margin_bytes < 0:
            raise ValueError("GPU safety margin cannot be negative")

        by_uuid = {snapshot.uuid: snapshot for snapshot in snapshots}
        pressured = frozenset(event.gpu_uuid for event in events)
        allocated_gpu_uuids = frozenset(allocation.gpu_uuid for allocation in allocations)
        unavailable = allocated_gpu_uuids | frozenset(active_claims) | protected_gpu_uuids
        grouped: dict[str, list[ActorAllocation]] = defaultdict(list)
        for allocation in allocations:
            grouped[allocation.actor_id].append(allocation)

        planned_targets: set[str] = set()
        plans: list[IdleActorEvictionPlan] = []
        for actor_id in sorted(grouped):
            ordered = tuple(
                sorted(grouped[actor_id], key=lambda allocation: allocation.replica_rank)
            )
            source_gpu_uuids = tuple(allocation.gpu_uuid for allocation in ordered)
            if not set(source_gpu_uuids) & pressured:
                continue
            if not _complete_idle_replica(ordered):
                continue
            if set(source_gpu_uuids) & (frozenset(active_claims) | protected_gpu_uuids):
                continue
            source_snapshots = tuple(by_uuid.get(gpu_uuid) for gpu_uuid in source_gpu_uuids)
            if any(snapshot is None for snapshot in source_snapshots):
                continue
            sources = tuple(snapshot for snapshot in source_snapshots if snapshot is not None)
            # An event can race the foreign process exiting.  It is never proof
            # enough on its own to dispose a resident model.
            if not any(
                snapshot.uuid in pressured and self._pressure_policy.is_pressure(snapshot)
                for snapshot in sources
            ):
                continue

            required_memory = max(
                _resident_footprint_bytes(snapshot) + safety_margin_bytes for snapshot in sources
            )
            required_memory = max(1, required_memory)
            if (
                _contention_band(
                    sources,
                    required_memory,
                    self.policy.light_contention_utilization_percent,
                )
                != "heavy"
            ):
                # Moving between equivalent bands would trade model reload cost
                # for negligible physical improvement.
                continue

            candidates = tuple(
                snapshot
                for snapshot in snapshots
                if snapshot.uuid not in unavailable
                and snapshot.uuid not in planned_targets
                and not snapshot.externally_reserved
                and not snapshot.topodrome_processes
                and snapshot.total_memory_bytes >= required_memory
                and snapshot.free_memory_bytes >= required_memory
                and (
                    self.policy.allow_foreign_processes_when_capacity_sufficient
                    or not snapshot.foreign_processes
                )
            )
            target = _best_alternative(
                candidates,
                len(ordered),
                required_memory,
                self.policy.light_contention_utilization_percent,
            )
            if target is None:
                continue
            target_snapshots, target_band = target
            alternative_gpu_uuids = tuple(snapshot.uuid for snapshot in target_snapshots)
            planned_targets.update(alternative_gpu_uuids)
            plans.append(
                IdleActorEvictionPlan(
                    actor_id=actor_id,
                    allocations=ordered,
                    alternative_gpu_uuids=alternative_gpu_uuids,
                    required_memory_bytes=required_memory,
                    alternative_band=target_band,
                )
            )
        return tuple(plans)


def _complete_idle_replica(allocations: tuple[ActorAllocation, ...]) -> bool:
    return bool(
        allocations
        and all(allocation.state == "idle" for allocation in allocations)
        and len({allocation.model_key for allocation in allocations}) == 1
        and tuple(allocation.replica_rank for allocation in allocations)
        == tuple(range(len(allocations)))
        and len({allocation.gpu_uuid for allocation in allocations}) == len(allocations)
    )


def _resident_footprint_bytes(snapshot: GPUSnapshot) -> int:
    """Conservatively estimate the resident actor's per-device VRAM footprint."""

    owned_known = sum(process.used_memory_bytes or 0 for process in snapshot.topodrome_processes)
    external = snapshot.external_processes
    if any(process.used_memory_bytes is None for process in external):
        managed_estimate = snapshot.used_memory_bytes
    else:
        managed_estimate = max(
            0,
            snapshot.used_memory_bytes
            - sum(process.used_memory_bytes or 0 for process in external),
        )
    return max(owned_known, managed_estimate)


def _contention_band(
    snapshots: Sequence[GPUSnapshot],
    required_memory_bytes: int,
    light_utilization_percent: float,
) -> Literal["clean", "light", "heavy"]:
    """Mirror the scheduler's clean/light/heavy physical-safety bands."""

    if all(not snapshot.foreign_processes for snapshot in snapshots):
        return "clean"
    if all(
        snapshot.free_memory_bytes >= 2 * required_memory_bytes
        and snapshot.utilization_percent <= light_utilization_percent
        for snapshot in snapshots
    ):
        return "light"
    return "heavy"


def _best_alternative(
    candidates: tuple[GPUSnapshot, ...],
    gpu_count: int,
    required_memory_bytes: int,
    light_utilization_percent: float,
) -> tuple[tuple[GPUSnapshot, ...], Literal["clean", "light"]] | None:
    eligible: list[
        tuple[
            tuple[object, ...],
            tuple[GPUSnapshot, ...],
            Literal["clean", "light"],
        ]
    ] = []
    for selected in itertools.combinations(candidates, gpu_count):
        band = _contention_band(
            selected,
            required_memory_bytes,
            light_utilization_percent,
        )
        if band == "heavy":
            continue
        ordered = tuple(sorted(selected, key=lambda snapshot: snapshot.index))
        key: tuple[object, ...] = (
            0 if band == "clean" else 1,
            max(snapshot.utilization_percent for snapshot in ordered),
            -min(snapshot.free_memory_bytes for snapshot in ordered),
            tuple(snapshot.uuid for snapshot in ordered),
        )
        eligible.append((key, ordered, band))
    if not eligible:
        return None
    _, selected, band = min(eligible, key=lambda value: value[0])
    return selected, band


__all__ = [
    "IdleActorEvictionPlan",
    "IdleActorRebalancePolicy",
    "IdleActorRebalancer",
]
