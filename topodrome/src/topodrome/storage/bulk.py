"""A bulk transport moves bytes; the mirror owns integrity and publication."""

from pathlib import Path
from typing import BinaryIO, Protocol

from topodrome.storage.base import StorageUnavailableError


class BulkTransportError(StorageUnavailableError):
    """Safe failure classification, never server messages or credential details."""

    def __init__(
        self, reason: str, *, retryable: bool = False, retry_after_seconds: float | None = None
    ):
        self.transport_reason = reason
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"storage transport failed ({reason})")


def http_retry_after(response) -> float | None:
    """Honor bounded authority backpressure without trusting arbitrary delays."""
    if response.status_code not in {429, 503}:
        return None
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(0.1, min(5.0, seconds))


class BulkFileTransport(Protocol):
    def stat(self, logical: str) -> dict | None: ...
    def healthcheck(self) -> bool: ...
    def read(
        self,
        logical: str,
        output: BinaryIO,
        offset: int,
        count: int,
        *,
        expected_identity: str | None = None,
    ) -> None: ...
    def write(self, logical: str, source: Path, offset: int, size: int) -> None: ...
    def rename(self, source: str, destination: str) -> None: ...
    def remove(self, logical: str) -> None: ...
    def download_tree(self, logical: str, destination: Path) -> None: ...
    def close(self) -> None: ...
