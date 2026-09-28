"""Durable adapter for an authoritative local filesystem."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from topodrome.storage.base import (
    ObjectDigest,
    StorageVerificationError,
    VerifiedObject,
)
from topodrome.storage.paths import normalize_logical_path, resolve_beneath
from topodrome.storage.progress import check_transfer_cancelled

_READ_SIZE = 1024 * 1024


def fsync_and_digest(path: Path) -> ObjectDigest:
    """Synchronize one regular file and hash the exact synchronized bytes."""

    try:
        with path.open("rb", buffering=0) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise StorageVerificationError("only regular files can be verified")
            digest = hashlib.sha256()
            size = 0
            while chunk := handle.read(_READ_SIZE):
                check_transfer_cancelled()
                digest.update(chunk)
                size += len(chunk)
            os.fsync(handle.fileno())
            after = os.fstat(handle.fileno())
    except FileNotFoundError:
        raise
    except StorageVerificationError:
        raise
    except OSError as exc:
        raise StorageVerificationError("unable to synchronize artifact") from exc

    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or size != after.st_size:
        raise StorageVerificationError("artifact changed while it was being verified")
    _fsync_directory(path.parent)
    return ObjectDigest(size_bytes=size, sha256=digest.hexdigest())


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise StorageVerificationError("unable to synchronize artifact directory") from exc


class LocalStorage:
    """A local directory that is itself the authoritative filesystem."""

    def __init__(self, root: Path | str, *, create: bool = True) -> None:
        self.root = Path(root).expanduser()
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()

    def healthcheck(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.R_OK | os.W_OK | os.X_OK)

    def resolve(self, logical_path: str) -> Path:
        return resolve_beneath(self.root, logical_path)

    def flush_and_verify(
        self,
        logical_path: str,
        *,
        expected_sha256: str | None = None,
    ) -> VerifiedObject:
        normalized = normalize_logical_path(logical_path)
        path = self.resolve(normalized)
        digest = fsync_and_digest(path)
        if expected_sha256 is not None and digest.sha256 != expected_sha256:
            raise StorageVerificationError("artifact checksum did not match the expected value")
        return VerifiedObject.now(
            logical_path=normalized,
            local_path=path,
            digest=digest,
        )


__all__ = ["LocalStorage", "fsync_and_digest"]
