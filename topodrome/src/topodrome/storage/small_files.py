"""Immediate, bounded transfers on a persistent enrolled-site HTTP connection.

The HTTP acknowledgement is the authority's durability barrier, not just a
receipt for queued work. Bulk HTTP and digest requests use separate pools.
"""

from __future__ import annotations

import hashlib
import json
import stat
import time
from pathlib import Path

import httpx

from topodrome.storage.base import ObjectDigest, StorageUnavailableError, StorageVerificationError
from topodrome.storage.bulk import BulkTransportError, http_retry_after
from topodrome.storage.paths import normalize_logical_path
from topodrome.storage.progress import check_transfer_cancelled, transfer_progress
from topodrome.storage.remote_digest import DigestResponse, storage_api_origin

MAX_SMALL_FILE_BYTES = 4 * 1024**2
DEFAULT_SMALL_FILE_BYTES = 1024**2
SMALL_FILE_TIMEOUT_SECONDS = 30


class SmallFileFallback(Exception):
    """An ineligible object; use the identity-fenced HTTP streaming lane."""


def small_file_path(path: str) -> bool:
    # No credentials, private service/index metadata, notebooks or arbitrary
    # root files are exposed by the new transport.
    return path == "storage-layout.json" or path.split("/", 1)[0] in {
        "blobs",
        "jobs",
        "artifacts",
        "sources",
        "projects",
    }


class SchedulerSmallFileClient:
    def __init__(
        self,
        scheduler_url: str,
        site_id: str,
        token: str,
        *,
        max_bytes: int = DEFAULT_SMALL_FILE_BYTES,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not 0 < max_bytes <= MAX_SMALL_FILE_BYTES:
            raise ValueError("small-file limit is out of range")
        self.max_bytes = max_bytes
        self._site_id = site_id
        self._url = storage_api_origin(scheduler_url) + "/small-file"
        self._supported = True
        self._source_supported = True
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(SMALL_FILE_TIMEOUT_SECONDS, connect=5, pool=5),
            trust_env=False,
            follow_redirects=False,
            transport=transport,
            limits=httpx.Limits(
                max_connections=8, max_keepalive_connections=8, keepalive_expiry=None
            ),
        )

    def close(self) -> None:
        self._client.close()

    def accepts(self, logical_path: str, size_bytes: int | None = None) -> bool:
        return (
            self._supported
            and small_file_path(logical_path)
            and (size_bytes is None or size_bytes <= self.max_bytes)
        )

    def _check_response(self, response: httpx.Response, *, source_bundle: bool = False) -> None:
        def unsupported():
            if source_bundle:
                self._source_supported = False
            else:
                self._supported = False

        if response.status_code in {405, 501}:
            unsupported()
            raise SmallFileFallback
        if response.status_code == 404:
            # Distinguish a missing route on an older scheduler from a missing
            # file. Do not downgrade authentication or integrity failures.
            content = self._bounded_body(response, 4096)
            try:
                detail = json.loads(content).get("detail")
            except (ValueError, AttributeError):
                detail = None
            if detail == "Not Found":
                unsupported()
                raise SmallFileFallback
            if detail == "object not found":
                raise FileNotFoundError("small authoritative file was missing")
        if response.status_code == 413:
            raise SmallFileFallback
        if response.status_code in {408, 429, 500, 502, 503, 504}:
            raise BulkTransportError(
                "HTTPTransient", retryable=True, retry_after_seconds=http_retry_after(response)
            )
        response.raise_for_status()

    @staticmethod
    def _http_error(exc: httpx.HTTPError):
        if isinstance(exc, httpx.TransportError):
            return BulkTransportError(type(exc).__name__, retryable=True)
        return BulkTransportError(f"HTTP{exc.response.status_code}")

    def download_source_bundle(self, logical_path: str) -> bytes:
        from topodrome.storage.source_archive import source_bundle_path

        path = normalize_logical_path(logical_path)
        if not self._source_supported or not source_bundle_path(path):
            raise SmallFileFallback
        check_transfer_cancelled()
        try:
            with self._client.stream(
                "GET",
                self._url.removesuffix("small-file") + "source-bundle",
                params={
                    "site_id": self._site_id,
                    "logical_path": path,
                    "max_bytes": self.max_bytes,
                },
            ) as response:
                self._check_response(response, source_bundle=True)
                content = self._bounded_body(response, self.max_bytes)
                if response.headers.get(
                    "etag"
                ) != f'"{hashlib.sha256(content).hexdigest()}"' or response.headers.get(
                    "content-length"
                ) != str(len(content)):
                    raise StorageVerificationError("source archive transport failed verification")
        except httpx.HTTPError as exc:
            raise self._http_error(exc) from None
        transfer_progress("download", len(content), len(content))
        return content

    @staticmethod
    def _bounded_body(response: httpx.Response, maximum: int) -> bytes:
        content = bytearray()
        started = time.monotonic()
        for chunk in response.iter_bytes():
            check_transfer_cancelled()
            if time.monotonic() - started > SMALL_FILE_TIMEOUT_SECONDS:
                raise StorageUnavailableError("small-file response timed out")
            content.extend(chunk)
            if len(content) > maximum:
                raise StorageVerificationError("small-file response exceeded its bound")
        return bytes(content)

    def download(
        self,
        logical_path: str,
        *,
        expected: ObjectDigest | None = None,
        expected_size_bytes: int | None = None,
    ) -> tuple[bytes, ObjectDigest]:
        path = normalize_logical_path(logical_path)
        if not self.accepts(path, expected.size_bytes if expected else expected_size_bytes):
            raise SmallFileFallback
        check_transfer_cancelled()
        headers = {"If-Match": f'"{expected.sha256}"'} if expected else {}
        try:
            with self._client.stream(
                "GET",
                self._url,
                params={
                    "site_id": self._site_id,
                    "logical_path": path,
                    "max_bytes": self.max_bytes,
                },
                headers=headers,
            ) as response:
                self._check_response(response)
                content = self._bounded_body(response, self.max_bytes)
                digest = ObjectDigest(len(content), hashlib.sha256(content).hexdigest())
                if (
                    response.headers.get("etag") != f'"{digest.sha256}"'
                    or response.headers.get("content-length") != str(len(content))
                    or (expected is not None and digest != expected)
                    or (expected_size_bytes is not None and len(content) != expected_size_bytes)
                ):
                    raise StorageVerificationError(
                        "small-file download failed integrity verification"
                    )
        except httpx.HTTPError as exc:
            raise self._http_error(exc) from None
        transfer_progress("download", len(content), len(content))
        return content, digest

    def upload(
        self,
        logical_path: str,
        source: Path,
        digest: ObjectDigest,
        *,
        job_id: str | None = None,
        attempt_id: str | None = None,
    ) -> ObjectDigest:
        path = normalize_logical_path(logical_path)
        if not self.accepts(path, digest.size_bytes):
            raise SmallFileFallback
        check_transfer_cancelled()
        if not stat.S_ISREG(source.lstat().st_mode):
            raise StorageVerificationError("small-file source must be regular")
        with source.open("rb") as stream:
            content = stream.read(self.max_bytes + 1)
        if (
            len(content) != digest.size_bytes
            or hashlib.sha256(content).hexdigest() != digest.sha256
        ):
            raise StorageVerificationError("small-file source changed before upload")
        if job_id is None or attempt_id is None:
            parts = path.split("/")
            if len(parts) >= 5 and parts[0] == "jobs" and parts[2] == "attempts":
                job_id, attempt_id = parts[1], parts[3]
            elif (
                len(parts) == 4
                and parts[0] == "jobs"
                and parts[2:] == ["committed", "manifest.json"]
            ):
                try:
                    manifest = json.loads(content)
                    job_id, attempt_id = manifest["job_id"], manifest["attempt_id"]
                except (ValueError, KeyError, TypeError):
                    raise StorageVerificationError("invalid publication manifest") from None
            else:
                raise SmallFileFallback
        transfer_progress("upload", 0, len(content))
        try:
            with self._client.stream(
                "PUT",
                self._url,
                params={
                    "site_id": self._site_id,
                    "logical_path": path,
                    "job_id": job_id,
                    "attempt_id": attempt_id,
                },
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Topodrome-Sha256": digest.sha256,
                    "Content-Length": str(len(content)),
                },
                content=content,
            ) as response:
                self._check_response(response)
                result = DigestResponse.model_validate_json(self._bounded_body(response, 4096))
                observed = ObjectDigest(result.size_bytes, result.sha256)
                if observed != digest:
                    raise StorageVerificationError(
                        "small-file acknowledgement did not match its content"
                    )
        except httpx.HTTPError as exc:
            raise self._http_error(exc) from None
        except ValueError:
            raise StorageVerificationError("invalid small-file upload acknowledgement") from None
        transfer_progress("verify", len(content), len(content))
        return observed
