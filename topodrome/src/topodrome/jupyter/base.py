"""Extension boundaries for durable sessions and site-specific launchers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from topodrome.jupyter.session import KernelSession, SessionLease


@dataclass(frozen=True, slots=True)
class KernelRuntime:
    runtime_id: str
    upstream_url: str
    site_id: str

    def __post_init__(self) -> None:
        if not self.runtime_id or not self.upstream_url or not self.site_id:
            raise ValueError("kernel runtime fields must not be empty")


@runtime_checkable
class SessionAllocator(Protocol):
    async def acquire(self, session: KernelSession) -> SessionLease: ...

    async def release(self, session_id: str, lease: SessionLease) -> None: ...


@runtime_checkable
class KernelLauncher(Protocol):
    async def start(self, session: KernelSession, lease: SessionLease) -> KernelRuntime: ...

    async def stop(self, runtime: KernelRuntime) -> None: ...


@runtime_checkable
class SessionStore(Protocol):
    async def create(self, session: KernelSession) -> None: ...

    async def save(self, session: KernelSession) -> None: ...

    async def get(self, session_id: str) -> KernelSession: ...

    async def list(self) -> tuple[KernelSession, ...]: ...


__all__ = ["KernelLauncher", "KernelRuntime", "SessionAllocator", "SessionStore"]
