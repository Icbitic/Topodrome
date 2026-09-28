"""Owned CUDA sentinel lifecycle for cooperative GPU handoff."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


class CudaSentinel(Protocol):
    @property
    def pids(self) -> tuple[int, ...]: ...

    def is_alive(self) -> bool: ...

    async def stop(self) -> None: ...


class CudaSentinelFactory(Protocol):
    async def start(self, gpu_indices: Mapping[str, int]) -> CudaSentinel: ...


@dataclass(frozen=True, slots=True)
class SentinelPolicy:
    startup_timeout_seconds: float = 15.0
    shutdown_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.startup_timeout_seconds <= 0:
            raise ValueError("sentinel startup timeout must be positive")
        if self.shutdown_timeout_seconds <= 0:
            raise ValueError("sentinel shutdown timeout must be positive")


class SubprocessCudaSentinel:
    def __init__(
        self,
        processes: tuple[asyncio.subprocess.Process, ...],
        *,
        shutdown_timeout_seconds: float,
    ) -> None:
        self._processes = processes
        self._shutdown_timeout_seconds = shutdown_timeout_seconds

    @property
    def pids(self) -> tuple[int, ...]:
        return tuple(process.pid for process in self._processes)

    def is_alive(self) -> bool:
        return all(process.returncode is None for process in self._processes)

    async def stop(self) -> None:
        for process in self._processes:
            if process.stdin is not None:
                process.stdin.close()
        for process in self._processes:
            if process.returncode is not None:
                continue
            try:
                await asyncio.wait_for(process.wait(), timeout=self._shutdown_timeout_seconds)
            except TimeoutError:
                # These are PIDs created and retained by this object. No
                # scavenger or other foreign process can reach this branch.
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=self._shutdown_timeout_seconds)
                except TimeoutError:
                    process.kill()
                    await process.wait()


class SubprocessCudaSentinelFactory:
    """Launch one minimal owned CUDA context for every reserved GPU."""

    def __init__(self, policy: SentinelPolicy | None = None) -> None:
        self.policy = policy or SentinelPolicy()

    async def start(self, gpu_indices: Mapping[str, int]) -> SubprocessCudaSentinel:
        if not gpu_indices:
            raise ValueError("sentinel requires at least one GPU")
        processes: list[asyncio.subprocess.Process] = []
        try:
            for gpu_uuid, gpu_index in sorted(gpu_indices.items()):
                environment = _minimal_environment(gpu_index)
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "topodrome.site_agent.cuda_sentinel",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=environment,
                )
                processes.append(process)
                assert process.stdout is not None
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=self.policy.startup_timeout_seconds,
                    )
                except TimeoutError:
                    raise RuntimeError(f"CUDA sentinel did not start for {gpu_uuid}") from None
                if line.rstrip() != b"READY":
                    assert process.stderr is not None
                    error = (await process.stderr.read(2000)).decode(errors="replace")
                    raise RuntimeError(f"CUDA sentinel failed for {gpu_uuid}: {error.strip()}")
            return SubprocessCudaSentinel(
                tuple(processes),
                shutdown_timeout_seconds=self.policy.shutdown_timeout_seconds,
            )
        except BaseException:
            sentinel = SubprocessCudaSentinel(
                tuple(processes),
                shutdown_timeout_seconds=self.policy.shutdown_timeout_seconds,
            )
            await sentinel.stop()
            raise


def _minimal_environment(gpu_index: int) -> dict[str, str]:
    if gpu_index < 0:
        raise ValueError("GPU index cannot be negative")
    allowed = (
        "PATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "CUDA_HOME",
    )
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


__all__ = [
    "CudaSentinel",
    "CudaSentinelFactory",
    "SentinelPolicy",
    "SubprocessCudaSentinel",
    "SubprocessCudaSentinelFactory",
]
