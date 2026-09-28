"""Explicit durability boundary for site- or provider-persistent state."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from topodrome.storage.base import ObjectDigest, StorageUnavailableError
from topodrome.storage.local import fsync_and_digest
from topodrome.storage.paths import normalize_logical_path, resolve_beneath, validate_component

_REGIONAL_PREFIX = "regional://"


@dataclass(frozen=True, slots=True)
class VerifiedRegionalObject:
    reference: str
    local_path: Path
    digest: ObjectDigest


class RegionalStateStore:
    """Own region-affine checkpoints and shutdown durability markers.

    This is deliberately separate from the Lab1-authoritative StorageAdapter.
    References carry an explicit scheme, so a retry cannot accidentally treat
    provider-regional state as a portable global artifact.
    """

    def __init__(self, root: Path | str, *, create: bool = False) -> None:
        value = Path(root).expanduser()
        if not value.is_absolute():
            raise ValueError("regional persistent root must be absolute")
        if create:
            value.mkdir(parents=True, exist_ok=True)
        self.root = value.resolve()

    def healthcheck(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.R_OK | os.W_OK | os.X_OK)

    def checkpoint_directory(self, job_id: str, attempt_id: str) -> Path:
        job = validate_component(job_id, label="job_id")
        attempt = validate_component(attempt_id, label="attempt_id")
        target = resolve_beneath(
            self.root,
            f"jobs/{job}/attempts/{attempt}/checkpoints",
        )
        target.mkdir(parents=True, exist_ok=True)
        return target

    def reference_for(self, path: Path | str) -> str:
        candidate = Path(path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("regional object is outside the configured persistent root")
        relative = candidate.relative_to(self.root).as_posix()
        return f"{_REGIONAL_PREFIX}{normalize_logical_path(relative)}"

    def resolve(self, reference: str) -> Path:
        if not reference.startswith(_REGIONAL_PREFIX):
            raise ValueError("regional reference has the wrong scheme")
        logical = normalize_logical_path(reference.removeprefix(_REGIONAL_PREFIX))
        return resolve_beneath(self.root, logical)

    def flush_and_verify(self, reference: str) -> VerifiedRegionalObject:
        if not self.healthcheck():
            raise StorageUnavailableError("regional persistent storage is unavailable")
        path = self.resolve(reference)
        digest = fsync_and_digest(path)
        # Provider filesystems may buffer metadata below the ordinary file
        # descriptor. The per-file fsync above plus this filesystem-wide
        # barrier is completed before a durability ACK is emitted.
        os.sync()
        observed = fsync_and_digest(path)
        if observed != digest:
            raise RuntimeError("regional object changed across its durability barrier")
        return VerifiedRegionalObject(reference, path, observed)

    def durability_barrier(
        self,
        *,
        shutdown_generation: int,
        command_nonce: str,
    ) -> VerifiedRegionalObject:
        if shutdown_generation < 1 or not command_nonce:
            raise ValueError("shutdown generation and command nonce are required")
        if not self.healthcheck():
            raise StorageUnavailableError("regional persistent storage is unavailable")
        nonce_digest = hashlib.sha256(command_nonce.encode("utf-8")).hexdigest()
        marker_id = hashlib.sha256(
            f"{shutdown_generation}\0{nonce_digest}".encode()
        ).hexdigest()
        logical = f".topodrome/durability/{marker_id}.json"
        target = resolve_beneath(self.root, logical)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = (
            json.dumps(
                {
                    "format_version": 1,
                    "shutdown_generation": shutdown_generation,
                    "command_nonce_sha256": nonce_digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        try:
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return self.flush_and_verify(f"{_REGIONAL_PREFIX}{logical}")


def is_regional_reference(reference: str | None) -> bool:
    return bool(reference and reference.startswith(_REGIONAL_PREFIX))


__all__ = ["RegionalStateStore", "VerifiedRegionalObject", "is_regional_reference"]
