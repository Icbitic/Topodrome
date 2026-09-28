"""Storage-agnostic checkpoint requests and cooperative preemption."""

from __future__ import annotations

import hashlib
import pickle
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, RLock
from time import monotonic
from types import MappingProxyType
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    name: str
    sequence: int
    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("checkpoint name must not be empty")
        if self.sequence < 1:
            raise ValueError("checkpoint sequence must be positive")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class CheckpointRestore:
    """One scheduler-selected, durably verified checkpoint for a new attempt."""

    reference: str
    value: Any = field(default=None, repr=False)
    serialized_path: str | None = field(default=None, repr=False)
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("checkpoint restore reference must not be empty")
        object.__setattr__(self, "reference", self.reference.strip())
        if self.serialized_path is not None:
            if not Path(self.serialized_path).is_absolute():
                raise ValueError("checkpoint restore path must be absolute")
            if self.value is not None or self.sha256 is None or len(self.sha256) != 64:
                raise ValueError("serialized checkpoint requires a digest and no decoded value")

    def materialize(self) -> Any:
        """Deserialize only in the experiment's leased worker, never the agent."""
        if self.serialized_path is None:
            return self.value
        content = Path(self.serialized_path).read_bytes()
        if hashlib.sha256(content).hexdigest() != self.sha256:
            raise RuntimeError("checkpoint changed before worker restoration")
        return pickle.loads(content)  # noqa: S301 - verified trusted job checkpoint


class CheckpointHandler(Protocol):
    def __call__(self, checkpoint: CheckpointRecord) -> Any: ...


class CheckpointManager:
    """Numbers checkpoints and forwards them to the site's durable sink."""

    def __init__(
        self,
        handler: CheckpointHandler | None = None,
        *,
        interval_seconds: float | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if interval_seconds is not None and interval_seconds <= 0:
            raise ValueError("checkpoint interval must be positive")
        self._handler = handler
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._lock = RLock()
        self._sequence = 0
        # Checkpoints may contain complete optimizer/model state. Historical
        # payloads must not accumulate in a long-running worker's Python heap.
        self._records: deque[CheckpointRecord] = deque(maxlen=2)
        self._started_at = clock()
        self._last_saved_at: float | None = None
        self._latest_reference: str | None = None

    def save(
        self,
        value: Any,
        *,
        name: str = "checkpoint",
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointRecord:
        with self._lock:
            sequence = self._sequence + 1
            record = CheckpointRecord(
                name=name,
                sequence=sequence,
                value=value,
                metadata=metadata or {},
            )
            reference: Any = None
            if self._handler is not None:
                reference = self._handler(record)
            # Failed persistence handlers neither consume a sequence nor reset
            # interval accounting. The site backend applies the authoritative
            # durability barrier before publishing the reference.
            self._sequence = sequence
            self._records.append(record)
            self._last_saved_at = self._clock()
            if isinstance(reference, str) and reference:
                self._latest_reference = reference
            return record

    def due(self) -> bool:
        with self._lock:
            if self._interval_seconds is None:
                return False
            baseline = self._last_saved_at
            if baseline is None:
                baseline = self._started_at
            return self._clock() - baseline >= self._interval_seconds

    @property
    def records(self) -> tuple[CheckpointRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def latest_reference(self) -> str | None:
        with self._lock:
            return self._latest_reference


class PreemptionToken:
    """Thread-safe cooperative cancellation signal checked at safe boundaries."""

    def __init__(self) -> None:
        self._event = Event()
        self._lock = RLock()
        self._reason: str | None = None

    def request(self, reason: str = "preemption requested") -> bool:
        reason = reason.strip() if isinstance(reason, str) else str(reason)
        with self._lock:
            first_request = not self._event.is_set()
            if first_request:
                self._reason = reason or "preemption requested"
                self._event.set()
            return first_request

    def is_requested(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)
