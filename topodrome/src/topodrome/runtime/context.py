"""Experiment-facing runtime context."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType
from typing import Any

from topodrome.domain.progress import PROGRESS_PREFIX, JobProgress

from .checkpoint import (
    CheckpointHandler,
    CheckpointManager,
    CheckpointRecord,
    CheckpointRestore,
    PreemptionToken,
)
from .cleanup import CleanupHandle, CleanupStack


@dataclass(frozen=True, slots=True)
class ArtifactRegistration:
    source: Path = field(repr=False)
    logical_path: str
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.logical_path, str) or not self.logical_path.strip():
            raise ValueError("logical artifact path must be a safe relative path")
        logical = PurePosixPath(self.logical_path)
        if (
            not self.logical_path
            or logical.is_absolute()
            or ".." in logical.parts
            or logical == PurePosixPath(".")
        ):
            raise ValueError("logical artifact path must be a safe relative path")
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "logical_path", logical.as_posix())
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class ExperimentContext:
    """Objects and lifecycle hooks made available to one experiment attempt.

    Cleanup callbacks are LIFO and idempotent as a group.  Model hooks and
    temporary mutations should register their restoration callback immediately.
    The actor treats a failed callback, or an explicit :meth:`mark_dirty`, as a
    reason to destroy the warm model rather than reuse uncertain state.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: Any,
        job_id: str,
        attempt_id: str,
        workspace: Path | str | None = None,
        committed_inputs: Mapping[str, Path | str] | None = None,
        scratch: Path | str | None = None,
        preemption: PreemptionToken | None = None,
        preemption_probe: Callable[[], bool] | None = None,
        checkpoint_handler: CheckpointHandler | None = None,
        checkpoint_interval_seconds: float | None = None,
        checkpoint_restore: CheckpointRestore | None = None,
        artifact_handler: Callable[[ArtifactRegistration], Any] | None = None,
    ) -> None:
        if not job_id or not isinstance(job_id, str):
            raise ValueError("job_id must not be empty")
        if not attempt_id or not isinstance(attempt_id, str):
            raise ValueError("attempt_id must not be empty")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.job_id = job_id
        self.attempt_id = attempt_id
        self.workspace = Path(workspace) if workspace is not None else None
        self.scratch = Path(scratch) if scratch is not None else None
        self.committed_inputs = MappingProxyType(
            {key: Path(value) for key, value in (committed_inputs or {}).items()}
        )
        self._preemption = preemption or PreemptionToken()
        self._preemption_probe = preemption_probe
        self._checkpoints = CheckpointManager(
            checkpoint_handler,
            interval_seconds=checkpoint_interval_seconds,
        )
        self._checkpoint_restore = checkpoint_restore
        self._artifact_handler = artifact_handler
        self._artifacts: list[ArtifactRegistration] = []
        self._cleanup = CleanupStack()
        self._dirty_reasons: list[str] = []
        self._lock = RLock()
        self._progress_started = time.monotonic()
        self._progress_last = float("-inf")
        self._progress_baseline = None
        self._progress_samples = 0

    def report_progress(
        self,
        current: float = 0,
        total: float | None = None,
        *,
        stage: str = "",
        message: str = "",
    ) -> bool:
        """Report advisory progress, at most once per two seconds.

        Returns False when throttled or output is unavailable. This is not a
        checkpoint or completion signal. Distributed jobs should call on rank zero.
        """
        progress = JobProgress(current=current, total=total, stage=stage, message=message)
        if total is not None and current > total:
            raise ValueError("current must not exceed total")
        with self._lock:
            now = time.monotonic()
            if now - self._progress_last < 2:
                return False
            baseline = self._progress_baseline
            if baseline is None or baseline[:2] != (stage, total) or current < baseline[2]:
                baseline = (stage, total, current, now)
                self._progress_baseline = baseline
                self._progress_samples = 0
            self._progress_samples += 1
            progress.elapsed_seconds = max(0, now - self._progress_started)
            delta = current - baseline[2]
            elapsed = now - baseline[3]
            if total is not None and delta > 0 and elapsed >= 5 and self._progress_samples >= 3:
                progress.eta_seconds = elapsed * (total - current) / delta
            self._progress_last = now
            try:
                print("\n" + PROGRESS_PREFIX + progress.model_dump_json(), flush=True)
            except Exception:
                return False
            return True

    def should_preempt(self) -> bool:
        if self._preemption.is_requested():
            return True
        return bool(self._preemption_probe and self._preemption_probe())

    @property
    def preemption_reason(self) -> str | None:
        return self._preemption.reason

    def request_preemption(self, reason: str = "preemption requested") -> bool:
        """Signal this attempt; normally called by the owning site agent."""

        return self._preemption.request(reason)

    def checkpoint(
        self,
        value: Any,
        *,
        name: str = "checkpoint",
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointRecord:
        return self._checkpoints.save(value, name=name, metadata=metadata)

    def checkpoint_due(self) -> bool:
        """Return true at the configured interval or immediately on preemption."""

        return self.should_preempt() or self._checkpoints.due()

    @property
    def checkpoints(self) -> tuple[CheckpointRecord, ...]:
        return self._checkpoints.records

    @property
    def checkpoint_ref(self) -> str | None:
        restore = self._checkpoint_restore
        return None if restore is None else restore.reference

    @property
    def has_checkpoint(self) -> bool:
        return self._checkpoint_restore is not None

    def restore_checkpoint(self, default: Any = None) -> Any:
        """Return the scheduler-selected checkpoint, or ``default`` for a fresh run."""

        restore = self._checkpoint_restore
        if restore is None:
            return default
        with self._lock:
            restore = self._checkpoint_restore
            if getattr(restore, "serialized_path", None) is not None:
                restore = CheckpointRestore(restore.reference, restore.materialize())
                self._checkpoint_restore = restore
            return restore.value

    @property
    def latest_checkpoint_ref(self) -> str | None:
        return self._checkpoints.latest_reference

    def register_artifact(
        self,
        source: Path | str,
        *,
        logical_path: str | None = None,
        required: bool = True,
        metadata: Mapping[str, Any] | None = None,
        must_exist: bool = True,
    ) -> ArtifactRegistration:
        path = Path(source)
        if not path.is_absolute() and self.workspace is not None:
            path = self.workspace / path
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        registration = ArtifactRegistration(
            source=path,
            logical_path=Path(source).name if logical_path is None else logical_path,
            required=required,
            metadata=metadata or {},
        )
        if self._artifact_handler is not None:
            self._artifact_handler(registration)
        with self._lock:
            self._artifacts.append(registration)
        return registration

    @property
    def artifacts(self) -> tuple[ArtifactRegistration, ...]:
        with self._lock:
            return tuple(self._artifacts)

    def add_cleanup(
        self, callback: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> CleanupHandle:
        return self._cleanup.push(callback, *args, **kwargs)

    # A readable alias for experiment code that models cleanup as hook removal.
    on_cleanup = add_cleanup

    def mark_dirty(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("dirty reason must not be empty")
        with self._lock:
            self._dirty_reasons.append(reason.strip())

    @property
    def dirty(self) -> bool:
        with self._lock:
            return bool(self._dirty_reasons)

    @property
    def dirty_reasons(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._dirty_reasons)

    @property
    def closed(self) -> bool:
        return self._cleanup.closed

    def close(self) -> None:
        self._cleanup.close()

    async def aclose(self) -> None:
        await self._cleanup.aclose()

    def __enter__(self) -> ExperimentContext:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
