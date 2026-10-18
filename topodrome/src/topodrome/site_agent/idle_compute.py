"""Bounded idle GPU load, never a job, actor, lease, or foreign-process killer.

The agent is the authority; a child receives expiring permissions through a
pipe. Admission stops the exact child before inspecting physical capacity.
Idle warm actors are permitted only through live PID/start-time proofs and a
complete idle allocation. Unknown workloads, reservations, and lost control
permission all fail closed. No CUDA library is loaded in the agent process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from topodrome.config import IdleComputeSettings

from .admission import GIB, LeaseFenceStore
from .inventory import (
    CooperativeProcessMatcher,
    GPUSnapshot,
    ProcessOwner,
    ProcessOwnershipRegistry,
)
from .sentinel import _minimal_environment
from .spool import SpoolEvent

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _Worker:
    process: asyncio.subprocess.Process
    started_at: float


class IdleComputeManager:
    def __init__(
        self,
        settings: IdleComputeSettings,
        fences: LeaseFenceStore,
        ownership: ProcessOwnershipRegistry,
        cooperative: CooperativeProcessMatcher,
    ) -> None:
        self.settings = settings
        self.fences = fences
        self.ownership = ownership
        self.cooperative = cooperative
        self.workers: dict[str, _Worker] = {}
        self._retry_after: dict[str, float] = {}
        self._blocked: set[str] = set()
        self._permit_until = 0.0

    def acknowledge(self, event: SpoolEvent, payload: Mapping[str, Any]) -> None:
        permission = payload.get("idle_compute_permission")
        # Never renew authority from replayed, stale, or delayed heartbeats.
        age = time.time() - event.created_at
        if (
            not self.settings.enabled or not isinstance(permission, dict)
            or payload.get("duplicate") or payload.get("stale")
            or not 0 <= age < 15 or permission.get("allowed") is not True
        ):
            self._permit_until = 0
            return
        blocked = permission.get("blocked_gpu_uuids")
        ttl = permission.get("valid_for_seconds")
        if (
            not isinstance(blocked, list) or any(not isinstance(x, str) for x in blocked)
            or not isinstance(ttl, (int, float)) or isinstance(ttl, bool) or not 0 < ttl <= 15
        ):
            self._permit_until = 0
            return
        self._blocked = set(blocked)
        self._permit_until = time.monotonic() + max(0, ttl - age)

    def eligible(
        self, snapshot: GPUSnapshot, *, own_pid: int | None = None,
    ) -> dict[int, int] | None:
        if (
            not self.settings.enabled or time.monotonic() >= self._permit_until
            or snapshot.uuid in self._blocked or snapshot.externally_reserved
            or snapshot.uuid in self.fences.active_claims()
            or snapshot.free_memory_bytes < self.settings.minimum_free_vram_gb * GIB
            or (own_pid is None and snapshot.utilization_percent > 10)
        ):
            return None
        allocation = self.fences.actor_allocation(snapshot.uuid)
        if allocation is not None and allocation.state != "idle":
            return None
        allowed: dict[int, int] = {}
        for process in snapshot.processes:
            if process.pid == own_pid:
                continue
            owner = self.ownership.exact_owner(process.pid, require_live=True)
            if (
                process.owner is not ProcessOwner.TOPODROME or allocation is None
                or allocation.state != "idle" or owner is None
                or owner.actor_id != allocation.actor_id or owner.start_ticks is None
            ):
                return None
            members = tuple(
                item.gpu_uuid for item in self.fences.actor_allocations()
                if item.actor_id == allocation.actor_id
            )
            if self.fences.actor_idle_since(
                allocation.actor_id, expected_gpu_uuids=members,
            ) is None:
                return None
            allowed[process.pid] = owner.start_ticks
        return allowed

    async def tick(self, snapshots: Sequence[GPUSnapshot], *, draining: bool = False) -> None:
        present = {item.uuid for item in snapshots}
        for gpu_uuid in set(self.workers) - present:
            await self.stop((gpu_uuid,))
        for snapshot in snapshots:
            worker = self.workers.get(snapshot.uuid)
            if worker is not None and worker.process.returncode is not None:
                self.cooperative.unregister_pid(worker.process.pid)
                self.workers.pop(snapshot.uuid)
                self._retry_after[snapshot.uuid] = time.monotonic() + 10
                worker = None
            allowed = None if draining else self.eligible(
                snapshot, own_pid=None if worker is None else worker.process.pid,
            )
            if allowed is None:
                await self.stop((snapshot.uuid,))
                continue
            if worker is None:
                if time.monotonic() < self._retry_after.get(snapshot.uuid, 0):
                    continue
                environment = _minimal_environment(snapshot.index)
                # Keep helpers inside the site's approved storage root without
                # inheriting its enrollment credentials or arbitrary settings.
                for key in (
                    "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "CUDA_CACHE_PATH",
                    "TRITON_CACHE_DIR", "TORCH_HOME", "HF_HOME",
                ):
                    if key in os.environ:
                        environment[key] = os.environ[key]
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "topodrome.site_agent.idle_compute",
                    "--worker", snapshot.uuid, str(self.settings.matrix_size),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    env=environment,
                )
                worker = _Worker(process, time.monotonic())
                self.workers[snapshot.uuid] = worker
                self.cooperative.register_pid(process.pid)
                log.info("idle compute started: GPU %s", snapshot.index)
            assert worker.process.stdin is not None
            try:
                worker.process.stdin.write(
                    json.dumps({"allowed_pids": allowed, "ttl": 3}).encode() + b"\n"
                )
                await worker.process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                await self.stop((snapshot.uuid,))

    async def stop(self, gpu_uuids: Sequence[str] | None = None) -> None:
        targets = tuple(self.workers) if gpu_uuids is None else tuple(gpu_uuids)
        for gpu_uuid in targets:
            worker = self.workers.pop(gpu_uuid, None)
            if worker is None:
                continue
            process = worker.process
            # Classification must remain cooperative until the context exits.
            try:
                if process.stdin is not None:
                    process.stdin.close()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=1)
                    except TimeoutError:
                        with suppress(ProcessLookupError):
                            process.kill()
                        await process.wait()
            finally:
                self.cooperative.unregister_pid(process.pid)


def run_worker(gpu_uuid: str, matrix_size: int) -> None:
    """Child-only CUDA loop, bounded memory and sub-second foreign-work yield."""
    # Imports deliberately remain here: the coordinator must never own CUDA.
    import pynvml  # type: ignore[import-untyped]

    stop = threading.Event()
    lock = threading.Lock()
    permission: dict[str, Any] = {"deadline": 0.0, "allowed": {}}

    def read_permissions() -> None:
        try:
            for line in sys.stdin.buffer:
                value = json.loads(line)
                allowed = {int(pid): int(ticks) for pid, ticks in value["allowed_pids"].items()}
                ttl = float(value["ttl"])
                if not 0 < ttl <= 3 or any(
                    pid <= 0 or ticks <= 0 for pid, ticks in allowed.items()
                ):
                    break
                with lock:
                    permission.update(deadline=time.monotonic() + ttl, allowed=allowed)
        except (ValueError, TypeError, KeyError):
            pass
        finally:
            stop.set()

    threading.Thread(target=read_permissions, daemon=True).start()
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
        registry = ProcessOwnershipRegistry()

        def permitted() -> bool:
            with lock:
                deadline, allowed = permission["deadline"], dict(permission["allowed"])
            if stop.is_set() or time.monotonic() >= deadline:
                return False
            try:
                processes = list(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
                with suppress(pynvml.NVMLError_NotSupported):
                    processes.extend(pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
                for process in processes:
                    if process.pid == os.getpid():
                        continue
                    ticks = allowed.get(process.pid)
                    if ticks is None:
                        return False
                    registry.register(process.pid, start_ticks=ticks)
                    if registry.exact_owner(process.pid, require_live=True) is None:
                        return False
                return True
            except pynvml.NVMLError:
                return False

        import torch

        # Importing Torch may be slow; the pipe reader keeps permissions fresh.
        if not permitted():
            return
        torch.set_num_threads(1)
        a = torch.randn((matrix_size, matrix_size), device="cuda", dtype=torch.float16)
        b = torch.randn_like(a)
        out = torch.empty_like(a)
        while permitted():
            until = time.monotonic() + 0.2
            while not stop.is_set() and time.monotonic() < until:
                # A bounded queue avoids CPU/synchronization bubbles without
                # leaving seconds of GPU work in flight during a handoff.
                for _ in range(8):
                    torch.mm(a, b, out=out)
                torch.cuda.synchronize()
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("idle compute must be launched by a site agent")
    size = int(sys.argv[3])
    if not 512 <= size <= 4096:
        raise SystemExit("invalid idle matrix size")
    run_worker(sys.argv[2], size)
