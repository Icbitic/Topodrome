"""Sustained foreign-user pressure detection and safe process signalling."""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .inventory import GPUSnapshot, ProcessOwnershipRegistry

MIB = 1024**2


@dataclass(frozen=True, slots=True)
class PressurePolicy:
    enabled: bool = True
    sustained_seconds: float = 30.0
    foreign_memory_threshold_bytes: int = 256 * MIB
    utilization_threshold_percent: int = 20

    def __post_init__(self) -> None:
        if self.sustained_seconds < 0:
            raise ValueError("sustained pressure duration cannot be negative")
        if self.foreign_memory_threshold_bytes < 0:
            raise ValueError("foreign memory threshold cannot be negative")
        if not 0 <= self.utilization_threshold_percent <= 100:
            raise ValueError("utilization threshold must be between 0 and 100")

    def is_pressure(self, snapshot: GPUSnapshot) -> bool:
        if not self.enabled:
            return False
        foreign = snapshot.foreign_processes
        if not foreign:
            return False
        unknown_memory = any(process.used_memory_bytes is None for process in foreign)
        return (
            unknown_memory
            or snapshot.foreign_memory_bytes >= self.foreign_memory_threshold_bytes
            or snapshot.utilization_percent >= self.utilization_threshold_percent
        )


@dataclass(frozen=True, slots=True)
class PressureEvent:
    gpu_uuid: str
    first_observed_at: float
    observed_at: float
    foreign_pids: tuple[int, ...]


class SustainedPressureMonitor:
    """Emit one event after pressure remains continuously above policy."""

    def __init__(
        self,
        policy: PressurePolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self._clock = clock
        self._started_at: dict[str, float] = {}
        self._emitted: set[str] = set()

    def observe(
        self,
        snapshots: Sequence[GPUSnapshot],
        *,
        now: float | None = None,
    ) -> tuple[PressureEvent, ...]:
        observed_at = self._clock() if now is None else now
        present = {snapshot.uuid for snapshot in snapshots}
        for missing in set(self._started_at) - present:
            self._started_at.pop(missing, None)
            self._emitted.discard(missing)

        events: list[PressureEvent] = []
        for snapshot in snapshots:
            if not self.policy.is_pressure(snapshot):
                self._started_at.pop(snapshot.uuid, None)
                self._emitted.discard(snapshot.uuid)
                continue
            started_at = self._started_at.setdefault(snapshot.uuid, observed_at)
            if (
                snapshot.uuid not in self._emitted
                and observed_at - started_at >= self.policy.sustained_seconds
            ):
                self._emitted.add(snapshot.uuid)
                events.append(
                    PressureEvent(
                        gpu_uuid=snapshot.uuid,
                        first_observed_at=started_at,
                        observed_at=observed_at,
                        foreign_pids=tuple(process.pid for process in snapshot.foreign_processes),
                    )
                )
        return tuple(events)


class OwnedProcessSignaller:
    """Signal only a PID registered as Topodrome-owned.

    This guard is intentionally separate from pressure detection.  Foreign PIDs
    in a :class:`PressureEvent` are evidence that Topodrome should yield, never
    termination targets.
    """

    def __init__(
        self,
        ownership: ProcessOwnershipRegistry,
        *,
        kill: Callable[[int, int], None] = os.kill,
    ) -> None:
        self._ownership = ownership
        self._kill = kill

    def terminate(self, pid: int, *, force: bool = False) -> None:
        if not self._ownership.owns(pid):
            raise PermissionError(f"refusing to signal foreign process {pid}")
        self._kill(pid, signal.SIGKILL if force else signal.SIGTERM)


__all__ = [
    "OwnedProcessSignaller",
    "PressureEvent",
    "PressurePolicy",
    "SustainedPressureMonitor",
]
