"""Session orchestration independent of any particular browser relay."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from topodrome.domain.errors import NotFoundError
from topodrome.domain.specs import KernelRequest, KernelSummary, new_id
from topodrome.jupyter.base import KernelLauncher, KernelRuntime, SessionAllocator, SessionStore
from topodrome.jupyter.relay import RelayHandle, ReverseRelay
from topodrome.jupyter.session import KernelSession, SessionLease, SessionLeaseRegistry


class KernelGatewayError(RuntimeError):
    """A sanitized session start or cleanup failure."""


class InMemorySessionStore:
    """Hermetic store; production can provide a PostgreSQL-backed implementation."""

    def __init__(self) -> None:
        self._sessions: dict[str, KernelSession] = {}
        self._lock = asyncio.Lock()

    async def create(self, session: KernelSession) -> None:
        async with self._lock:
            if session.session_id in self._sessions:
                raise ValueError("session already exists")
            self._sessions[session.session_id] = copy.deepcopy(session)

    async def save(self, session: KernelSession) -> None:
        async with self._lock:
            if session.session_id not in self._sessions:
                raise NotFoundError("kernel session not found")
            self._sessions[session.session_id] = copy.deepcopy(session)

    async def get(self, session_id: str) -> KernelSession:
        async with self._lock:
            try:
                return copy.deepcopy(self._sessions[session_id])
            except KeyError as exc:
                raise NotFoundError("kernel session not found") from exc

    async def list(self) -> tuple[KernelSession, ...]:
        async with self._lock:
            return tuple(copy.deepcopy(session) for session in self._sessions.values())


@dataclass(frozen=True, slots=True)
class _ActiveResources:
    lease: SessionLease
    runtime: KernelRuntime
    relay: RelayHandle


class KernelGateway:
    """Acquire, launch, relay, expire, and tear down interactive sessions.

    This manager does not pretend to be an HTTP/WebSocket browser proxy. That
    responsibility is deliberately supplied through :class:`ReverseRelay`.
    """

    def __init__(
        self,
        *,
        allocator: SessionAllocator,
        launcher: KernelLauncher,
        relay: ReverseRelay,
        store: SessionStore | None = None,
        registry: SessionLeaseRegistry | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        id_factory: Callable[[], str] = lambda: new_id("kernel"),
    ) -> None:
        self._allocator = allocator
        self._launcher = launcher
        self._relay = relay
        self._store = store or InMemorySessionStore()
        self._registry = registry or SessionLeaseRegistry()
        self._clock = clock
        self._id_factory = id_factory
        self._active: dict[str, _ActiveResources] = {}
        self._operation_lock = asyncio.Lock()

    async def create(self, request: KernelRequest) -> KernelSummary:
        async with self._operation_lock:
            now = self._now()
            session = KernelSession(
                session_id=self._id_factory(),
                request=request,
                created_at=now,
                last_activity_at=now,
            )
            await self._store.create(session)
            session.begin_start()
            await self._store.save(session)

            lease: SessionLease | None = None
            runtime: KernelRuntime | None = None
            relay: RelayHandle | None = None
            reserved = False
            try:
                lease = await self._allocator.acquire(session)
                self._registry.reserve(
                    session.session_id,
                    session.request.mode,
                    lease,
                    requested_gpus=session.request.resources.gpu,
                )
                reserved = True
                runtime = await self._launcher.start(session, lease)
                relay = await self._relay.start(
                    session_id=session.session_id,
                    upstream_url=runtime.upstream_url,
                )
                session.mark_running(
                    lease=lease,
                    runtime_id=runtime.runtime_id,
                    relay_id=relay.relay_id,
                    connect_url=relay.public_url,
                    site_id=runtime.site_id,
                    now=self._now(),
                )
                self._active[session.session_id] = _ActiveResources(lease, runtime, relay)
                await self._store.save(session)
                return session.summary()
            except Exception:
                if relay is not None:
                    await self._best_effort(self._relay.stop(relay))
                if runtime is not None:
                    await self._best_effort(self._launcher.stop(runtime))
                if reserved:
                    self._registry.release(session.session_id)
                if lease is not None:
                    await self._best_effort(self._allocator.release(session.session_id, lease))
                session.mark_lost()
                await self._store.save(session)
                raise KernelGatewayError("kernel session failed to start") from None

    async def get(self, session_id: str) -> KernelSummary:
        return (await self._store.get(session_id)).summary()

    async def touch(self, session_id: str) -> KernelSummary:
        async with self._operation_lock:
            session = await self._store.get(session_id)
            session.touch(self._now())
            await self._store.save(session)
            return session.summary()

    async def mark_idle(self, session_id: str) -> KernelSummary:
        async with self._operation_lock:
            session = await self._store.get(session_id)
            session.mark_idle(self._now())
            await self._store.save(session)
            return session.summary()

    async def stop(self, session_id: str) -> KernelSummary:
        async with self._operation_lock:
            return await self._stop_locked(session_id)

    async def expire_idle(self) -> tuple[str, ...]:
        async with self._operation_lock:
            now = self._now()
            expired = [
                session.session_id
                for session in await self._store.list()
                if session.is_idle_expired(now)
            ]
            for session_id in expired:
                try:
                    await self._stop_locked(session_id)
                except KernelGatewayError:
                    # The session is already marked lost and all cleanup paths ran.
                    pass
            return tuple(expired)

    async def _stop_locked(self, session_id: str) -> KernelSummary:
        session = await self._store.get(session_id)
        if session.terminal:
            return session.summary()
        session.request_stop()
        await self._store.save(session)
        resources = self._active.pop(session_id, None)
        failures = 0
        if resources is not None:
            failures += not await self._try(self._relay.stop(resources.relay))
            failures += not await self._try(self._launcher.stop(resources.runtime))
            self._registry.release(session_id)
            failures += not await self._try(self._allocator.release(session_id, resources.lease))
        elif session.lease is not None:
            # A durable store can restore the lease after a gateway restart; its
            # production launcher/relay reconciler owns remote handle recovery.
            self._registry.release(session_id)
            failures += not await self._try(self._allocator.release(session_id, session.lease))

        if failures:
            session.mark_lost()
        else:
            session.mark_stopped()
        await self._store.save(session)
        if failures:
            raise KernelGatewayError("kernel session cleanup was incomplete")
        return session.summary()

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("kernel gateway clock must return a timezone-aware datetime")
        return now

    @staticmethod
    async def _try(operation: Awaitable[object]) -> bool:
        try:
            await operation
        except Exception:
            return False
        return True

    @classmethod
    async def _best_effort(cls, operation: Awaitable[object]) -> None:
        await cls._try(operation)


__all__ = ["InMemorySessionStore", "KernelGateway", "KernelGatewayError"]
