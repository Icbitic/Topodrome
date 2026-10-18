"""Storage contracts shared by site agents and artifact publishers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from topodrome.domain.errors import DurabilityError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StorageError(RuntimeError):
    """Base class for storage failures that are safe to surface."""


class StorageUnavailableError(StorageError):
    """The configured storage endpoint is not currently usable."""


class StorageVerificationError(DurabilityError, StorageError):
    """An object could not be proven durable at the authoritative store."""


@dataclass(frozen=True, slots=True)
class ObjectDigest:
    """Content identity observed from one storage location."""

    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be a lowercase hexadecimal SHA-256 digest")


@dataclass(frozen=True, slots=True)
class VerifiedObject:
    """An object verified at the adapter's authoritative location."""

    logical_path: str
    local_path: Path
    digest: ObjectDigest
    verified_at: datetime

    @classmethod
    def now(
        cls,
        *,
        logical_path: str,
        local_path: Path,
        digest: ObjectDigest,
    ) -> VerifiedObject:
        return cls(
            logical_path=logical_path,
            local_path=local_path,
            digest=digest,
            verified_at=datetime.now(UTC),
        )


@runtime_checkable
class StorageAdapter(Protocol):
    """Logical filesystem with an explicit durability barrier."""

    def healthcheck(self) -> bool: ...

    def resolve(self, logical_path: str) -> Path: ...

    def resolve_writable(self, logical_path: str) -> Path:
        """Return the canonical local write path, without fetching remote data."""
        ...

    def flush_and_verify(
        self,
        logical_path: str,
        *,
        expected_sha256: str | None = None,
    ) -> VerifiedObject: ...


@runtime_checkable
class AuthoritativeDigestReader(Protocol):
    """Reads content identity without traversing a local VFS mount/cache."""

    def healthcheck(self) -> bool: ...

    def digest(self, logical_path: str) -> ObjectDigest | None: ...


__all__ = [
    "AuthoritativeDigestReader",
    "ObjectDigest",
    "StorageAdapter",
    "StorageError",
    "StorageUnavailableError",
    "StorageVerificationError",
    "VerifiedObject",
]
