"""Retry-aware incremental rendering for ``submit --wait``."""

from __future__ import annotations

import codecs
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, TextIO

import httpx

from topodrome.domain.specs import JobSummary

from .client import AttemptLogChunk

_TERMINAL_JOBS = frozenset({"succeeded", "failed", "canceled"})
_TERMINAL_ATTEMPTS = frozenset(
    {"succeeded", "rejected", "preempted", "failed", "canceled", "lost"}
)


class LogFollowClient(Protocol):
    def status(self, job_id: str) -> JobSummary: ...

    def attempts(self, job_id: str) -> list[dict[str, object]]: ...

    def attempt_log_chunk(
        self,
        job_id: str,
        attempt_id: str,
        *,
        offset: int = 0,
        limit_bytes: int = 262144,
        anchor_sha256: str | None = None,
    ) -> AttemptLogChunk: ...


@dataclass(slots=True)
class _AttemptCursor:
    number: int
    state: str = ""
    offset: int = 0
    anchor_sha256: str | None = None
    decoder: codecs.IncrementalDecoder = field(
        default_factory=lambda: codecs.getincrementaldecoder("utf-8")("replace")
    )
    drained_terminal: bool = False
    decoder_finalized: bool = False


def follow_job(
    client: LogFollowClient,
    job_id: str,
    *,
    log_output: TextIO,
    status_output: TextIO,
    poll_seconds: float = 0.5,
    terminal_log_grace_seconds: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> JobSummary:
    """Write attempt bytes once as they arrive and return the terminal summary."""

    if poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    if terminal_log_grace_seconds < 0:
        raise ValueError("terminal log grace cannot be negative")
    cursors: dict[str, _AttemptCursor] = {}
    connection_interrupted = False
    terminal_deadline: float | None = None
    terminal_summary: JobSummary | None = None
    while True:
        try:
            summary = client.status(job_id)
            if summary.state.value in _TERMINAL_JOBS:
                terminal_summary = summary
                if terminal_deadline is None:
                    terminal_deadline = monotonic() + terminal_log_grace_seconds
            else:
                terminal_summary = None
                terminal_deadline = None
            attempts = _ordered_attempts(client.attempts(job_id))
            for attempt in attempts:
                attempt_id = _required_text(attempt, "attempt_id")
                number = _required_integer(attempt, "attempt_number")
                state = _required_text(attempt, "state")
                cursor = cursors.setdefault(attempt_id, _AttemptCursor(number=number))
                if cursor.number != number:
                    raise RuntimeError(
                        "Topodrome API returned inconsistent attempt metadata"
                    )
                if cursor.state != state:
                    cursor.state = state
                    site = attempt.get("site_id")
                    location = f" on {site}" if isinstance(site, str) and site else ""
                    status_output.write(
                        f"[topodrome] attempt {number} {state}{location}\n"
                    )
                    status_output.flush()
                if not cursor.drained_terminal:
                    complete = _drain_attempt(
                        client,
                        job_id,
                        attempt_id,
                        cursor,
                        log_output=log_output,
                        status_output=status_output,
                    )
                    if state == "rejected":
                        # Rejection happens before a runtime is started, so no
                        # attempt output can arrive later (including offer
                        # expiry performed centrally by the scheduler).
                        complete = True
                    if state in _TERMINAL_ATTEMPTS and complete:
                        _finish_decoder(cursor, log_output)
                        cursor.drained_terminal = True
            connection_interrupted = False
        except httpx.TransportError:
            if not connection_interrupted:
                status_output.write(
                    "[topodrome] API connection interrupted; retrying log follow\n"
                )
                status_output.flush()
                connection_interrupted = True
            if (
                terminal_summary is not None
                and terminal_deadline is not None
                and monotonic() >= terminal_deadline
            ):
                _finish_after_grace(
                    cursors,
                    job_id=job_id,
                    grace_seconds=terminal_log_grace_seconds,
                    log_output=log_output,
                    status_output=status_output,
                )
                return terminal_summary
            sleep(poll_seconds)
            continue

        if summary.state.value in _TERMINAL_JOBS:
            if all(
                cursor.state in _TERMINAL_ATTEMPTS and cursor.drained_terminal
                for cursor in cursors.values()
            ):
                _finish_decoders(cursors.values(), log_output)
                return summary
            assert terminal_deadline is not None
            if monotonic() >= terminal_deadline:
                _finish_after_grace(
                    cursors,
                    job_id=job_id,
                    grace_seconds=terminal_log_grace_seconds,
                    log_output=log_output,
                    status_output=status_output,
                )
                return summary
        sleep(poll_seconds)


def _drain_attempt(
    client: LogFollowClient,
    job_id: str,
    attempt_id: str,
    cursor: _AttemptCursor,
    *,
    log_output: TextIO,
    status_output: TextIO,
) -> bool:
    while True:
        chunk = client.attempt_log_chunk(
            job_id,
            attempt_id,
            offset=cursor.offset,
            anchor_sha256=cursor.anchor_sha256,
        )
        if chunk.reset:
            remainder = cursor.decoder.decode(b"", final=True)
            if remainder:
                log_output.write(remainder)
            cursor.decoder = codecs.getincrementaldecoder("utf-8")("replace")
            cursor.offset = chunk.offset
            cursor.anchor_sha256 = None
            status_output.write(
                f"[topodrome] attempt {cursor.number} log was replaced; following new bytes\n"
            )
            status_output.flush()
        elif chunk.offset != cursor.offset:
            raise RuntimeError("Topodrome API returned a discontinuous live-log range")

        if chunk.data:
            rendered = cursor.decoder.decode(chunk.data, final=False)
            if rendered:
                log_output.write(rendered)
                log_output.flush()
        cursor.offset = chunk.next_offset
        cursor.anchor_sha256 = chunk.anchor_sha256
        if not chunk.data or chunk.next_offset >= chunk.size_bytes:
            return chunk.complete and chunk.next_offset >= chunk.size_bytes


def _finish_decoder(cursor: _AttemptCursor, output: TextIO) -> None:
    if cursor.decoder_finalized:
        return
    remainder = cursor.decoder.decode(b"", final=True)
    if remainder:
        output.write(remainder)
    output.flush()
    cursor.decoder_finalized = True


def _finish_decoders(cursors: Iterable[_AttemptCursor], output: TextIO) -> None:
    for cursor in cursors:
        _finish_decoder(cursor, output)
    output.flush()


def _finish_after_grace(
    cursors: Mapping[str, _AttemptCursor],
    *,
    job_id: str,
    grace_seconds: float,
    log_output: TextIO,
    status_output: TextIO,
) -> None:
    pending = ", ".join(
        str(cursor.number)
        for cursor in cursors.values()
        if not cursor.drained_terminal
    )
    status_output.write(
        "[topodrome] final log verification did not arrive within "
        f"{grace_seconds:g}s"
        f"{f' (attempts {pending})' if pending else ''}; "
        f"use `topodrome logs {job_id}` to inspect late output\n"
    )
    status_output.flush()
    _finish_decoders(cursors.values(), log_output)


def _ordered_attempts(values: list[dict[str, object]]) -> list[Mapping[str, object]]:
    attempts: list[Mapping[str, object]] = list(values)
    attempts.sort(key=lambda value: _required_integer(value, "attempt_number"))
    return attempts


def _required_text(value: Mapping[str, object], key: str) -> str:
    candidate = value.get(key)
    if not isinstance(candidate, str) or not candidate:
        raise RuntimeError("Topodrome API returned invalid attempt metadata")
    return candidate


def _required_integer(value: Mapping[str, object], key: str) -> int:
    candidate = value.get(key)
    if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 1:
        raise RuntimeError("Topodrome API returned invalid attempt metadata")
    return candidate


__all__ = ["follow_job"]
