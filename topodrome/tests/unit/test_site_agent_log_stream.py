from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from topodrome.site_agent.log_stream import (
    LogChunkAcknowledgement,
    stream_attempt_log,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_stream_retries_an_acknowledgement_lost_range_without_advancing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempt.log"
    path.write_bytes(b"first\n")
    stop = asyncio.Event()
    calls: list[tuple[int, int, bytes]] = []

    async def send(
        generation: int, offset: int, data: bytes
    ) -> LogChunkAcknowledgement | None:
        calls.append((generation, offset, data))
        if len(calls) == 1:
            return None
        stop.set()
        return LogChunkAcknowledgement(True, generation, offset + len(data))

    await asyncio.wait_for(
        stream_attempt_log(path, send, stop, poll_seconds=0.001), timeout=1
    )

    assert calls == [(0, 0, b"first\n"), (0, 0, b"first\n")]


@pytest.mark.asyncio
async def test_stream_increments_generation_when_the_file_is_rewritten(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempt.log"
    path.write_bytes(b"old")
    stop = asyncio.Event()
    calls: list[tuple[int, int, bytes]] = []

    async def send(
        generation: int, offset: int, data: bytes
    ) -> LogChunkAcknowledgement:
        calls.append((generation, offset, data))
        if generation == 0:
            path.write_bytes(b"new")
        else:
            stop.set()
        return LogChunkAcknowledgement(True, generation, offset + len(data))

    await asyncio.wait_for(
        stream_attempt_log(path, send, stop, poll_seconds=0.001), timeout=1
    )

    assert calls == [(0, 0, b"old"), (1, 0, b"new")]


@pytest.mark.asyncio
async def test_stream_stops_after_scheduler_caps_live_bytes(tmp_path: Path) -> None:
    path = tmp_path / "attempt.log"
    path.write_bytes(b"bounded")
    calls = 0

    async def send(
        generation: int, offset: int, data: bytes
    ) -> LogChunkAcknowledgement:
        nonlocal calls
        calls += 1
        return LogChunkAcknowledgement(
            False,
            generation,
            offset,
            stop=True,
        )

    await asyncio.wait_for(
        stream_attempt_log(path, send, asyncio.Event(), poll_seconds=0.001),
        timeout=1,
    )

    assert calls == 1
