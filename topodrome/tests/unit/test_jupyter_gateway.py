from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from topodrome.domain.enums import KernelMode, KernelState
from topodrome.domain.specs import KernelRequest
from topodrome.jupyter.base import KernelRuntime
from topodrome.jupyter.gateway import InMemorySessionStore, KernelGateway, KernelGatewayError
from topodrome.jupyter.relay import RelayHandle
from topodrome.jupyter.session import KernelSession, SessionLease

pytestmark = pytest.mark.unit


class ManualClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 7, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class FakeAllocator:
    def __init__(self, *, raw: bool = False) -> None:
        self.raw = raw
        self.acquired: list[str] = []
        self.released: list[str] = []

    async def acquire(self, session: KernelSession) -> SessionLease:
        self.acquired.append(session.session_id)
        if self.raw:
            return SessionLease(
                workload_lease_id=f"lease-{session.session_id}",
                actor_id="private-actor",
                gpu_uuids=("GPU-1",),
                exclusive=True,
                private_actor=True,
            )
        return SessionLease(
            workload_lease_id=f"lease-{session.session_id}",
            actor_id="shared-actor",
        )

    async def release(self, session_id: str, lease: SessionLease) -> None:
        del lease
        self.released.append(session_id)


class FakeLauncher:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []

    async def start(self, session: KernelSession, lease: SessionLease) -> KernelRuntime:
        del lease
        self.started.append(session.session_id)
        return KernelRuntime(
            runtime_id=f"runtime-{session.session_id}",
            upstream_url=f"http://site.invalid/{session.session_id}",
            site_id="lab1",
        )

    async def stop(self, runtime: KernelRuntime) -> None:
        self.stopped.append(runtime.runtime_id.removeprefix("runtime-"))


class FakeRelay:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.started: list[str] = []
        self.stopped: list[str] = []

    async def start(self, *, session_id: str, upstream_url: str) -> RelayHandle:
        del upstream_url
        self.started.append(session_id)
        if self.fail:
            raise RuntimeError("relay token=should-not-leak")
        return RelayHandle(
            relay_id=f"relay-{session_id}",
            public_url=f"https://gateway.invalid/{session_id}",
        )

    async def stop(self, handle: RelayHandle) -> None:
        self.stopped.append(handle.relay_id.removeprefix("relay-"))


def _ids(*values: str):  # type: ignore[no-untyped-def]
    iterator = iter(values)
    return lambda: next(iterator)


async def test_gateway_allows_shared_service_actor_and_cleans_up() -> None:
    allocator = FakeAllocator()
    launcher = FakeLauncher()
    relay = FakeRelay()
    gateway = KernelGateway(
        allocator=allocator,
        launcher=launcher,
        relay=relay,
        id_factory=_ids("session-1", "session-2"),
    )

    first = await gateway.create(KernelRequest(model="llama", mode=KernelMode.SERVICE))
    second = await gateway.create(KernelRequest(model="llama", mode=KernelMode.SERVICE))

    assert first.state is second.state is KernelState.RUNNING
    assert await gateway.stop("session-1") == (await gateway.get("session-1"))
    assert (await gateway.get("session-1")).state is KernelState.STOPPED
    assert (await gateway.get("session-2")).state is KernelState.RUNNING
    assert allocator.released == ["session-1"]
    assert launcher.stopped == ["session-1"]
    assert relay.stopped == ["session-1"]


async def test_raw_sessions_cannot_share_gpu_or_private_actor() -> None:
    allocator = FakeAllocator(raw=True)
    store = InMemorySessionStore()
    gateway = KernelGateway(
        allocator=allocator,
        launcher=FakeLauncher(),
        relay=FakeRelay(),
        store=store,
        id_factory=_ids("raw-1", "raw-2"),
    )
    request = KernelRequest(model="llama", mode=KernelMode.RAW_MODEL)

    await gateway.create(request)
    with pytest.raises(KernelGatewayError):
        await gateway.create(request)

    assert (await gateway.get("raw-2")).state is KernelState.LOST
    assert allocator.released == ["raw-2"]


async def test_idle_expiry_uses_sliding_activity_deadline() -> None:
    clock = ManualClock()
    allocator = FakeAllocator()
    gateway = KernelGateway(
        allocator=allocator,
        launcher=FakeLauncher(),
        relay=FakeRelay(),
        clock=clock,
        id_factory=_ids("session-1"),
    )
    await gateway.create(
        KernelRequest(model="llama", mode=KernelMode.SERVICE, idle_timeout_seconds=60)
    )
    clock.now += timedelta(seconds=30)
    await gateway.touch("session-1")
    clock.now += timedelta(seconds=59)
    assert await gateway.expire_idle() == ()
    clock.now += timedelta(seconds=1)

    assert await gateway.expire_idle() == ("session-1",)
    assert (await gateway.get("session-1")).state is KernelState.STOPPED


async def test_start_failure_rolls_back_and_is_sanitized() -> None:
    allocator = FakeAllocator()
    launcher = FakeLauncher()
    gateway = KernelGateway(
        allocator=allocator,
        launcher=launcher,
        relay=FakeRelay(fail=True),
        id_factory=_ids("session-1"),
    )

    with pytest.raises(KernelGatewayError) as raised:
        await gateway.create(KernelRequest(model="llama"))

    assert "should-not-leak" not in str(raised.value)
    assert launcher.stopped == ["session-1"]
    assert allocator.released == ["session-1"]
    assert (await gateway.get("session-1")).state is KernelState.LOST
