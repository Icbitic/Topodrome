from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TypeVar

from topodrome.domain.events import SchedulerEnvelope
from topodrome.domain.specs import AssignmentOffer

Sender = Callable[[dict[str, object]], Awaitable[None]]
Closer = Callable[[], Awaitable[None]]
ResultT = TypeVar("ResultT")


@dataclass(slots=True)
class AgentConnection:
    site_id: str
    session_id: str
    send: Sender
    close: Closer | None = field(default=None, repr=False)
    generation: str = field(default="", init=False)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


class AgentConnectionRegistry:
    def __init__(self) -> None:
        self._connections: dict[str, AgentConnection] = {}
        self._site_locks: dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def register(self, connection: AgentConnection) -> None:
        site_lock = await self._site_lock(connection.site_id)
        previous: AgentConnection | None = None
        async with site_lock:
            async with self._lock:
                current = self._connections.get(connection.site_id)
                if current is connection:
                    return
                # Random scheduler-local connection generations do not collide
                # after a scheduler restart, unlike an in-memory counter.
                connection.generation = secrets.token_urlsafe(24)
                self._connections[connection.site_id] = connection
                previous = current
        # The replacement becomes authoritative before the old socket is
        # closed.  Its receive loop therefore fails the current-connection
        # guard even if close delivery races one final inbound frame.
        if previous is not None and previous.close is not None:
            async with previous.send_lock:
                with suppress(Exception):
                    await previous.close()

    async def unregister(
        self,
        site_id: str,
        session_id: str,
        *,
        generation: str | None = None,
    ) -> None:
        site_lock = await self._site_lock(site_id)
        async with site_lock:
            async with self._lock:
                current = self._connections.get(site_id)
                if (
                    current
                    and current.session_id == session_id
                    and (generation is None or current.generation == generation)
                ):
                    self._connections.pop(site_id, None)

    async def connected_site_ids(self) -> frozenset[str]:
        async with self._lock:
            return frozenset(self._connections)

    async def agent_session_id(self, site_id: str) -> str | None:
        async with self._lock:
            connection = self._connections.get(site_id)
            return None if connection is None else connection.session_id

    async def identity(self, site_id: str) -> tuple[str, str] | None:
        async with self._lock:
            connection = self._connections.get(site_id)
            if connection is None:
                return None
            return connection.session_id, connection.generation

    async def identities(self) -> dict[str, tuple[str, str]]:
        async with self._lock:
            return {
                site_id: (connection.session_id, connection.generation)
                for site_id, connection in self._connections.items()
            }

    async def is_current(self, connection: AgentConnection) -> bool:
        async with self._lock:
            return self._connections.get(connection.site_id) is connection

    async def run_if_current(
        self,
        connection: AgentConnection,
        action: Callable[[], Awaitable[ResultT]],
    ) -> tuple[bool, ResultT | None]:
        """Serialize inbound handling with replacement for one physical site."""

        site_lock = await self._site_lock(connection.site_id)
        async with site_lock:
            async with self._lock:
                if self._connections.get(connection.site_id) is not connection:
                    return False, None
            return True, await action()

    async def run_if_disconnected(
        self,
        site_id: str,
        action: Callable[[], Awaitable[ResultT]],
    ) -> tuple[bool, ResultT | None]:
        """Run while registration of a replacement connection is excluded.

        This is the disconnected counterpart to :meth:`run_if_current`.  The
        per-site lock makes the absence check and the complete action atomic
        with respect to ``register``.  It is used for destructive provider
        recovery after an agent has acknowledged a drain and disconnected.
        """

        site_lock = await self._site_lock(site_id)
        async with site_lock:
            async with self._lock:
                if site_id in self._connections:
                    return False, None
            return True, await action()

    async def send(
        self,
        site_id: str,
        envelope: SchedulerEnvelope,
        *,
        expected_generation: str | None = None,
    ) -> bool:
        site_lock = await self._site_lock(site_id)
        await site_lock.acquire()
        try:
            async with self._lock:
                connection = self._connections.get(site_id)
            if connection is None or (
                expected_generation is not None and connection.generation != expected_generation
            ):
                return False
            # Acquire the per-socket writer before allowing replacement.  The
            # registry may then switch generations, but it waits on this lock
            # before closing the displaced socket.
            await connection.send_lock.acquire()
        finally:
            site_lock.release()
        try:
            await connection.send(envelope.model_dump(mode="json"))
        finally:
            connection.send_lock.release()
        return True

    async def offer(
        self,
        offer: AssignmentOffer,
        *,
        expected_generation: str | None = None,
    ) -> bool:
        return await self.send(
            offer.site_id,
            SchedulerEnvelope(
                site_id=offer.site_id,
                kind="offer_assignment",
                # Display/scheduling metadata stays on the control plane, so older
                # site agents can continue executing without a restart.
                payload=offer.model_dump(
                    mode="json",
                    exclude={"spec": {"name", "description", "estimated_runtime_seconds",
                                      "artifact_retention_seconds"}},
                ),
            ),
            expected_generation=expected_generation,
        )

    async def _site_lock(self, site_id: str) -> asyncio.Lock:
        async with self._lock:
            return self._site_locks.setdefault(site_id, asyncio.Lock())
