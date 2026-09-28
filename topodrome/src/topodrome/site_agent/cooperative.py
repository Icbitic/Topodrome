"""Post-launch handoff checks for explicitly cooperative GPU scavengers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .inventory import InventoryProvider


@dataclass(frozen=True, slots=True)
class CooperativeExitPolicy:
    """Configuration exposed by the site-agent layer."""

    exit_timeout_seconds: float = 10.0
    poll_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.exit_timeout_seconds < 0:
            raise ValueError("cooperative exit timeout cannot be negative")
        if self.poll_interval_seconds <= 0:
            raise ValueError("cooperative exit poll interval must be positive")


@dataclass(frozen=True, slots=True)
class CooperativeExitResult:
    exited: bool
    remaining_pids: tuple[int, ...] = ()
    unexpected_foreign_pids: tuple[int, ...] = ()
    missing_gpu_uuids: tuple[str, ...] = ()


class CooperativeExitVerifier:
    """Wait for scavengers to self-exit without ever signalling their PIDs."""

    def __init__(
        self,
        inventory: InventoryProvider,
        policy: CooperativeExitPolicy | None = None,
    ) -> None:
        self.inventory = inventory
        self.policy = policy or CooperativeExitPolicy()

    async def wait(
        self,
        gpu_uuids: tuple[str, ...],
        *,
        expected_pids: frozenset[int] | None = None,
    ) -> CooperativeExitResult:
        expected = frozenset(gpu_uuids)
        if not expected:
            return CooperativeExitResult(True)
        deadline = time.monotonic() + self.policy.exit_timeout_seconds
        while True:
            snapshots = {
                snapshot.uuid: snapshot
                for snapshot in await asyncio.to_thread(self.inventory.sample)
            }
            missing = tuple(sorted(expected - snapshots.keys()))
            if missing:
                return CooperativeExitResult(False, missing_gpu_uuids=missing)
            ordinary_foreign = tuple(
                sorted(
                    process.pid
                    for uuid in expected
                    for process in snapshots[uuid].foreign_processes
                )
            )
            if ordinary_foreign:
                return CooperativeExitResult(False, unexpected_foreign_pids=ordinary_foreign)
            remaining = tuple(
                sorted(
                    process.pid
                    for uuid in expected
                    for process in snapshots[uuid].cooperative_processes
                )
            )
            if expected_pids is not None:
                visible_external = {
                    process.pid
                    for uuid in expected
                    for process in snapshots[uuid].external_processes
                }
                reclassified = tuple(sorted(expected_pids & visible_external))
                if reclassified:
                    remaining = tuple(sorted(set(remaining) | set(reclassified)))
            if not remaining:
                return CooperativeExitResult(True)
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                return CooperativeExitResult(False, remaining_pids=remaining)
            await asyncio.sleep(min(self.policy.poll_interval_seconds, remaining_time))


__all__ = [
    "CooperativeExitPolicy",
    "CooperativeExitResult",
    "CooperativeExitVerifier",
]
