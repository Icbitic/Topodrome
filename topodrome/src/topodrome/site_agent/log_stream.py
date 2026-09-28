"""Best-effort live tailing for an attempt's authoritative eventual log."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from topodrome.domain.logs import LIVE_LOG_CHUNK_BYTES, LOG_ANCHOR_BYTES


@dataclass(frozen=True, slots=True)
class LogChunkAcknowledgement:
    accepted: bool
    generation: int
    next_offset: int
    reset: bool = False
    stop: bool = False


LogChunkSender = Callable[
    [int, int, bytes], Awaitable[LogChunkAcknowledgement | None]
]


async def stream_attempt_log(
    path: Path,
    send: LogChunkSender,
    stop: asyncio.Event,
    *,
    poll_seconds: float = 0.2,
    chunk_bytes: int = LIVE_LOG_CHUNK_BYTES,
) -> None:
    """Follow one growing file without retaining output in process memory.

    An acknowledgement advances the local cursor.  A disconnect therefore
    retries the same byte range, while the scheduler's offset comparison makes
    an acknowledgement-lost retry idempotent.  This channel is non-authoritative:
    the runtime still flushes and verifies the complete log before reporting a
    terminal attempt.
    """

    if poll_seconds <= 0:
        raise ValueError("log polling interval must be positive")
    if not 1 <= chunk_bytes <= LIVE_LOG_CHUNK_BYTES:
        raise ValueError("log chunk size is outside the supported range")

    generation = 0
    offset = 0
    anchor_tail = b""
    anchor_sha256: str | None = None
    identity: tuple[int, int] | None = None

    while True:
        snapshot = await asyncio.to_thread(_snapshot, path, offset)
        if snapshot is None:
            if stop.is_set():
                return
            await _wait(stop, poll_seconds)
            continue
        current_identity, size, current_anchor = snapshot
        if identity is None:
            identity = current_identity
        elif current_identity != identity or size < offset or (
            offset > 0
            and anchor_sha256 is not None
            and current_anchor != anchor_sha256
        ):
            identity = current_identity
            generation += 1
            offset = 0
            anchor_tail = b""
            anchor_sha256 = None

        data = await asyncio.to_thread(_read, path, offset, chunk_bytes)
        if data:
            acknowledgement = await send(generation, offset, data)
            if acknowledgement is None:
                if stop.is_set():
                    return
            elif acknowledgement.stop:
                return
            elif acknowledgement.reset:
                generation = max(generation + 1, acknowledgement.generation)
                offset = 0
                anchor_tail = b""
                anchor_sha256 = None
                identity = None
            elif acknowledgement.accepted:
                expected = offset + len(data)
                if (
                    acknowledgement.generation != generation
                    or acknowledgement.next_offset != expected
                ):
                    if stop.is_set():
                        return
                else:
                    offset = expected
                    # Bind the cursor to the bytes the scheduler just
                    # acknowledged. The file may be rewritten from inside the
                    # send callback; re-reading here would then mistake those
                    # new bytes for the acknowledged old generation.
                    anchor_tail = (anchor_tail + data)[-LOG_ANCHOR_BYTES:]
                    anchor_sha256 = hashlib.sha256(anchor_tail).hexdigest()
                    continue
            elif acknowledgement.generation == generation:
                # The scheduler can ask for an earlier byte only when its live
                # sidecar was lost.  Never skip beyond bytes this call sent.
                expected = acknowledgement.next_offset
                if 0 <= expected <= offset + len(data):
                    if expected >= offset:
                        acknowledged = data[: expected - offset]
                        anchor_tail = (anchor_tail + acknowledged)[-LOG_ANCHOR_BYTES:]
                    else:
                        anchor_tail = await asyncio.to_thread(
                            _anchor_bytes, path, expected
                        )
                        # Re-establish the inode on the next pass. If this
                        # rewind raced a replacement, the sidecar overlap check
                        # will still force a fresh generation.
                        identity = None
                    offset = expected
                    anchor_sha256 = (
                        hashlib.sha256(anchor_tail).hexdigest()
                        if anchor_tail
                        else None
                    )

        elif stop.is_set():
            return
        await _wait(stop, poll_seconds)


def _snapshot(path: Path, offset: int) -> tuple[tuple[int, int], int, str | None] | None:
    try:
        stat = path.stat()
        if not path.is_file() or path.is_symlink():
            return None
        return (stat.st_dev, stat.st_ino), stat.st_size, _anchor(path, offset)
    except FileNotFoundError:
        return None


def _read(path: Path, offset: int, limit: int) -> bytes:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(limit)
    except FileNotFoundError:
        return b""


def _anchor(path: Path, offset: int) -> str | None:
    data = _anchor_bytes(path, offset)
    return hashlib.sha256(data).hexdigest() if data else None


def _anchor_bytes(path: Path, offset: int) -> bytes:
    if offset <= 0:
        return b""
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, offset - LOG_ANCHOR_BYTES), os.SEEK_SET)
            return handle.read(min(LOG_ANCHOR_BYTES, offset))
    except FileNotFoundError:
        return b""


async def _wait(stop: asyncio.Event, delay: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
    except TimeoutError:
        pass


__all__ = ["LogChunkAcknowledgement", "stream_attempt_log"]
