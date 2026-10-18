"""Attempt workspace preparation using logical global and local-scratch roots."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")


class WorkspaceStorage(Protocol):
    """Subset of the storage adapter needed before launching an attempt."""

    def healthcheck(self) -> bool: ...

    def resolve_writable(self, logical_path: str) -> Path: ...


@dataclass(frozen=True, slots=True)
class AttemptWorkspace:
    job_id: str
    attempt_id: str
    job_root: Path
    attempt_root: Path
    staging_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    committed_dir: Path
    scratch_dir: Path

    def environment(self) -> dict[str, str]:
        """Return only non-secret, workload-facing logical path variables."""

        return {
            "TOPODROME_JOB_ID": self.job_id,
            "TOPODROME_ATTEMPT_ID": self.attempt_id,
            "TOPODROME_JOB_ROOT": str(self.job_root),
            "TOPODROME_ATTEMPT_ROOT": str(self.attempt_root),
            "TOPODROME_STAGING_DIR": str(self.staging_dir),
            "TOPODROME_CHECKPOINT_DIR": str(self.checkpoint_dir),
            "TOPODROME_COMMITTED_DIR": str(self.committed_dir),
            "TOPODROME_SCRATCH_DIR": str(self.scratch_dir),
        }


class WorkspaceManager:
    def __init__(
        self,
        *,
        global_root: str | Path | None = None,
        scratch_root: str | Path,
        storage: WorkspaceStorage | None = None,
    ) -> None:
        if (global_root is None) == (storage is None):
            raise ValueError("provide exactly one of global_root or storage")
        self._global_root = None if global_root is None else Path(global_root)
        self._storage = storage
        self._scratch_root = Path(scratch_root)

    def prepare(self, job_id: str, attempt_id: str) -> AttemptWorkspace:
        _validate_component("job_id", job_id)
        _validate_component("attempt_id", attempt_id)
        if self._storage is not None:
            if not self._storage.healthcheck():
                raise RuntimeError("global experiment filesystem is unavailable")
            # Resolving an input tree may return a separate immutable cache.
            # Workspaces must use the canonical path consumed by the uploader.
            job_root = self._storage.resolve_writable(f"jobs/{job_id}")
        else:
            assert self._global_root is not None
            global_root = self._global_root.resolve()
            if not global_root.is_dir():
                raise RuntimeError("global experiment filesystem root is unavailable")
            job_root = (global_root / "jobs" / job_id).resolve()
            if global_root not in job_root.parents:
                raise ValueError("job workspace escaped the configured global root")

        attempt_root = job_root / "attempts" / attempt_id
        self._scratch_root.mkdir(parents=True, exist_ok=True)
        scratch_root = self._scratch_root.resolve()
        scratch_dir = (scratch_root / "jobs" / job_id / "attempts" / attempt_id).resolve()
        if scratch_root not in scratch_dir.parents:
            raise ValueError("attempt scratch escaped the configured scratch root")
        workspace = AttemptWorkspace(
            job_id=job_id,
            attempt_id=attempt_id,
            job_root=job_root,
            attempt_root=attempt_root,
            staging_dir=attempt_root / "staging",
            checkpoint_dir=attempt_root / "checkpoints",
            log_dir=attempt_root / "logs",
            committed_dir=job_root / "committed",
            scratch_dir=scratch_dir,
        )
        for directory in (
            workspace.staging_dir,
            workspace.checkpoint_dir,
            workspace.log_dir,
            workspace.committed_dir,
            workspace.scratch_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        try:
            workspace.scratch_dir.chmod(0o700)
        except OSError:
            # Some remote or container filesystems do not expose POSIX modes.
            pass
        return workspace

    def cleanup_scratch(self, workspace: AttemptWorkspace) -> None:
        root = self._scratch_root.resolve()
        target = workspace.scratch_dir.resolve()
        if target == root or root not in target.parents:
            raise ValueError("refusing to remove scratch outside the configured root")
        if target.exists():
            shutil.rmtree(target)


def _validate_component(label: str, value: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"unsafe {label}: {value!r}")


__all__ = ["AttemptWorkspace", "WorkspaceManager", "WorkspaceStorage"]
