from __future__ import annotations

import asyncio
from typing import Any

import pytest

from topodrome.domain.events import SchedulerEnvelope
from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry

pytestmark = pytest.mark.unit


async def test_replacement_invalidates_and_closes_old_site_connection() -> None:
    registry = AgentConnectionRegistry()
    old_messages: list[dict[str, Any]] = []
    new_messages: list[dict[str, Any]] = []
    old_closed = asyncio.Event()

    async def old_send(message: dict[str, object]) -> None:
        old_messages.append(dict(message))

    async def new_send(message: dict[str, object]) -> None:
        new_messages.append(dict(message))

    async def close_old() -> None:
        old_closed.set()

    old = AgentConnection("lab1", "old-session", old_send, close=close_old)
    await registry.register(old)
    old_generation = old.generation
    replacement = AgentConnection("lab1", "new-session", new_send)
    await registry.register(replacement)

    assert old_closed.is_set()
    assert replacement.generation != old_generation
    assert await registry.identity("lab1") == (
        "new-session",
        replacement.generation,
    )
    assert not await registry.send(
        "lab1",
        SchedulerEnvelope(site_id="lab1", kind="cancel_attempt"),
        expected_generation=old_generation,
    )
    assert await registry.send(
        "lab1",
        SchedulerEnvelope(site_id="lab1", kind="cancel_attempt"),
        expected_generation=replacement.generation,
    )
    await registry.unregister("lab1", "old-session", generation=old_generation)

    assert old_messages == []
    assert len(new_messages) == 1
    assert await registry.identity("lab1") is not None


async def test_inbound_event_and_replacement_have_a_per_site_total_order() -> None:
    registry = AgentConnectionRegistry()
    action_started = asyncio.Event()
    release_action = asyncio.Event()
    old_closed = asyncio.Event()

    async def discard(_: dict[str, object]) -> None:
        return None

    async def close_old() -> None:
        old_closed.set()

    old = AgentConnection("lab1", "old-session", discard, close=close_old)
    await registry.register(old)

    async def blocked_inbound() -> str:
        action_started.set()
        await release_action.wait()
        return "committed"

    inbound = asyncio.create_task(registry.run_if_current(old, blocked_inbound))
    await action_started.wait()
    replacement = AgentConnection("lab1", "new-session", discard)
    replacing = asyncio.create_task(registry.register(replacement))
    await asyncio.sleep(0)
    assert not replacing.done()

    release_action.set()
    assert await inbound == (True, "committed")
    await replacing
    assert old_closed.is_set()
    assert await registry.run_if_current(old, blocked_inbound) == (False, None)


async def test_disconnected_action_excludes_replacement_registration() -> None:
    registry = AgentConnectionRegistry()
    action_started = asyncio.Event()
    release_action = asyncio.Event()

    async def discard(_: dict[str, object]) -> None:
        return None

    async def destructive_recovery() -> str:
        action_started.set()
        await release_action.wait()
        return "powered-off"

    recovery = asyncio.create_task(
        registry.run_if_disconnected("autodl", destructive_recovery)
    )
    await action_started.wait()
    replacement = AgentConnection("autodl", "replacement-session", discard)
    registering = asyncio.create_task(registry.register(replacement))
    await asyncio.sleep(0)
    assert not registering.done()

    release_action.set()
    assert await recovery == (True, "powered-off")
    await registering
    assert await registry.run_if_disconnected("autodl", destructive_recovery) == (
        False,
        None,
    )
