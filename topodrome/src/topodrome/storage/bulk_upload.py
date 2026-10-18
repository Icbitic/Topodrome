"""Resumable, bounded HTTP writes to private authoritative upload prefixes.

This is a byte transport, not a publication API. The mirror still verifies the
whole prefix/file independently and fences promotion before declaring success.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from topodrome.storage.base import StorageVerificationError
from topodrome.storage.bulk import BulkTransportError, http_retry_after
from topodrome.storage.extent_encoding import MAX_EXTENT_BYTES, encode_extent
from topodrome.storage.paths import normalize_logical_path
from topodrome.storage.progress import check_transfer_cancelled, transfer_progress
from topodrome.storage.remote_digest import storage_api_origin

UPLOAD_CHUNK_BYTES = 128 * 1024
MAX_UPLOAD_CHUNK_BYTES = MAX_EXTENT_BYTES

if TYPE_CHECKING:
    from topodrome.storage.bulk_connections import AdaptiveBulkConnections


def private_upload_path(logical: str) -> str:
    path = normalize_logical_path(logical)
    parts = path.split("/")
    if (
        parts[0] not in {"jobs", "blobs", "artifacts"}
        or not re.fullmatch(r"\.[^/]+\.topodrome-upload-[a-f0-9]{32}", parts[-1])
        or len(parts) < 3
    ):
        raise ValueError("bulk writes require a private upload prefix")
    return path


class SchedulerBulkUploadClient:
    def __init__(
        self,
        scheduler_url: str,
        site_id: str,
        token: str,
        *,
        timeout: float = 120,
        transport: httpx.BaseTransport | None = None,
        bulk_connections: AdaptiveBulkConnections | None = None,
    ):
        self._url = storage_api_origin(scheduler_url) + "/upload-prefix"
        self._site_id = site_id
        self._gzip_supported = False
        self._bulk_connections = bulk_connections
        self._client = (
            None
            if bulk_connections is not None
            else httpx.Client(
                headers={"Authorization": f"Bearer {token}"},
                timeout=httpx.Timeout(timeout, connect=min(10, timeout), pool=min(10, timeout)),
                trust_env=False,
                follow_redirects=False,
                transport=transport,
                limits=httpx.Limits(
                    max_connections=2, max_keepalive_connections=2, keepalive_expiry=None
                ),
            )
        )

    def close(self):
        if self._client is not None:
            self._client.close()

    def _request(self, method, logical, *, content=None, **params):
        check_transfer_cancelled()
        try:
            wire_content = content
            encoding = "identity"
            if content is not None and self._gzip_supported:
                wire_content, encoding = encode_extent(content)
            headers = (
                {
                    "Content-Type": "application/octet-stream",
                    "X-Topodrome-Sha256": hashlib.sha256(content).hexdigest(),
                    "Content-Length": str(len(wire_content)),
                }
                if content is not None
                else {}
            )
            if encoding != "identity":
                headers.update(
                    {
                        "Content-Encoding": encoding,
                        "X-Topodrome-Uncompressed-Length": str(len(content)),
                    }
                )
            stream = (
                self._client.stream
                if self._bulk_connections is None
                else lambda *args, **kwargs: self._bulk_connections.stream(
                    "upload", *args, transfer_size=len(wire_content or b""), **kwargs
                )
            )
            with stream(
                method,
                self._url,
                params={
                    "site_id": self._site_id,
                    "logical_path": private_upload_path(logical),
                    **params,
                },
                headers=headers,
                content=wire_content,
            ) as response:
                # RFC 7694 response negotiation. A legacy authority remains
                # usable without an extra request or a blind encoded retry.
                if method == "GET" and response.status_code in {200, 404}:
                    self._gzip_supported = "gzip" in {
                        item.strip().lower()
                        for item in response.headers.get("accept-encoding", "").split(",")
                    }
                elif method == "PUT" and response.status_code == 200:
                    self._gzip_supported = "gzip" in {
                        item.strip().lower()
                        for item in response.headers.get("accept-encoding", "").split(",")
                    }
                if response.status_code == 404 and method in {"GET", "DELETE"}:
                    return None
                if response.status_code in {408, 429, 500, 502, 503, 504}:
                    raise BulkTransportError(
                        "HTTPTransient",
                        retryable=True,
                        retry_after_seconds=http_retry_after(response),
                    )
                if response.status_code in {400, 409, 412, 415, 422}:
                    raise StorageVerificationError("bulk upload prefix rejected")
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    check_transfer_cancelled()
                    body.extend(chunk)
                    if len(body) > 4096:
                        raise StorageVerificationError("oversized bulk upload acknowledgement")
                check_transfer_cancelled()
                return body
        except httpx.TransportError as exc:
            raise BulkTransportError(type(exc).__name__, retryable=True) from None
        except httpx.HTTPStatusError as exc:
            raise BulkTransportError(f"HTTP{exc.response.status_code}") from None

    def stat(self, logical: str) -> dict | None:
        body = self._request("GET", logical)
        if body is None:
            return None
        try:
            result = json.loads(body)
            if (
                result["IsDir"] is not False
                or type(result["Size"]) is not int
                or result["Size"] < 0
            ):
                raise ValueError("invalid prefix metadata")
            return result
        except (ValueError, TypeError, KeyError):
            raise StorageVerificationError("invalid upload prefix metadata") from None

    def remove(self, logical: str) -> None:
        self._request("DELETE", logical)

    def rename(self, logical: str, destination: str) -> None:
        self._request("POST", logical, destination=normalize_logical_path(destination))

    def write(self, logical: str, source: Path, offset: int, size: int) -> None:
        path = private_upload_path(logical)
        if not 0 <= offset <= size or not stat.S_ISREG(source.lstat().st_mode):
            raise StorageVerificationError("invalid bulk upload source")
        with source.open("rb") as stream:
            if stream.seek(offset) != offset:
                raise StorageVerificationError("bulk upload source offset changed")
            current = offset
            chunk_bytes = UPLOAD_CHUNK_BYTES
            while current < size or (size == 0 and current == 0):
                check_transfer_cancelled()
                content = stream.read(min(chunk_bytes, size - current))
                if not content and size:
                    raise StorageVerificationError("bulk upload source was truncated")
                checksum = hashlib.sha256(content).hexdigest()
                started = time.monotonic()
                try:
                    body = self._request("PUT", path, content=content, offset=current, total=size)
                    result = json.loads(body)
                    if (
                        result != {"offset": current + len(content), "sha256": checksum}
                        or type(result["offset"]) is not int
                    ):
                        raise StorageVerificationError("bulk upload acknowledgement mismatch")
                except (ValueError, TypeError, KeyError):
                    raise StorageVerificationError("invalid bulk upload acknowledgement") from None
                current += len(content)
                # Amortize RTT/fsync on fast links; use smaller retry extents on
                # slow links. Memory and retransmission remain strictly bounded.
                elapsed = max(0.001, time.monotonic() - started)
                chunk_bytes = max(
                    128 * 1024,
                    min(MAX_UPLOAD_CHUNK_BYTES, round(len(content) * 2 / elapsed), chunk_bytes * 2),
                )
                transfer_progress("upload", current, size)
                if not content:
                    break
