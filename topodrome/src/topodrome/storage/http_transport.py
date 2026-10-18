"""Persistent HTTP-only transport with separate short-request and bulk lanes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path
from typing import BinaryIO

import httpx

from topodrome.storage.base import ObjectDigest, StorageVerificationError
from topodrome.storage.bulk import BulkTransportError, http_retry_after
from topodrome.storage.bulk_connections import AdaptiveBulkConnections
from topodrome.storage.bulk_upload import SchedulerBulkUploadClient, private_upload_path
from topodrome.storage.local import fsync_and_digest
from topodrome.storage.paths import normalize_logical_path, validate_component
from topodrome.storage.progress import check_transfer_cancelled, transfer_progress
from topodrome.storage.remote_digest import DigestResponse, storage_api_origin
from topodrome.storage.small_files import SchedulerSmallFileClient, SmallFileFallback

MAX_METADATA_BYTES = 2 * 1024**2


class SchedulerFileTransport:
    """All storage operations use the authority's enrolled-agent HTTP API.

    Separate pools protect tiny metadata requests from long-lived streams. If
    SSH is used, ``bulk_url`` must use its independently supervised forward:
    different HTTP pools alone do not isolate a shared outer TCP connection.
    """

    def __init__(
        self,
        scheduler_url: str,
        site_id: str,
        token: str,
        *,
        bulk_url: str | None = None,
        timeout: float = 30,
        min_free_space_gb: float = 10,
        small_files: SchedulerSmallFileClient | None = None,
        transport: httpx.BaseTransport | None = None,
        bulk_connections: AdaptiveBulkConnections | None = None,
    ):
        self._url = storage_api_origin(scheduler_url)
        self._bulk_url = storage_api_origin(bulk_url or scheduler_url)
        self._site_id = site_id
        self._minimum_free_bytes = round(min_free_space_gb * 1024**3)
        self._small_files = small_files
        kwargs = dict(
            headers={"Authorization": f"Bearer {token}", "Accept-Encoding": "identity"},
            timeout=httpx.Timeout(timeout, connect=min(5, timeout), pool=min(5, timeout)),
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )
        self._client = httpx.Client(
            **kwargs,
            limits=httpx.Limits(
                max_connections=8, max_keepalive_connections=8, keepalive_expiry=None
            ),
        )
        self._bulk_connections = bulk_connections
        self._bulk = (
            None
            if bulk_connections is not None
            else httpx.Client(
                **kwargs,
                limits=httpx.Limits(
                    max_connections=2, max_keepalive_connections=2, keepalive_expiry=None
                ),
            )
        )
        self._uploads = SchedulerBulkUploadClient(
            bulk_url or scheduler_url,
            site_id,
            token,
            timeout=timeout,
            transport=transport,
            bulk_connections=bulk_connections,
        )

    def close(self):
        try:
            self._uploads.close()
        finally:
            try:
                if self._bulk is not None:
                    self._bulk.close()
                if self._bulk_connections is not None:
                    self._bulk_connections.close()
            finally:
                self._client.close()

    @staticmethod
    def _status(response, *, missing=False):
        if missing and response.status_code == 404:
            return False
        if response.status_code in {408, 429, 500, 502, 503, 504}:
            raise BulkTransportError(
                "HTTPTransient", retryable=True, retry_after_seconds=http_retry_after(response)
            )
        if response.status_code in {400, 409, 412, 416, 422}:
            raise StorageVerificationError("HTTP storage identity or request rejected")
        if not 200 <= response.status_code < 300:
            raise BulkTransportError(f"HTTP{response.status_code}")
        return True

    def _json(
        self,
        method: str,
        endpoint: str,
        *,
        logical: str | None = None,
        headers=None,
        payload=None,
        missing=False,
        **params,
    ):
        check_transfer_cancelled()
        try:
            query = {"site_id": self._site_id, **params}
            if logical is not None:
                query["logical_path"] = normalize_logical_path(logical)
            with self._client.stream(
                method, self._url + "/" + endpoint, params=query, headers=headers, json=payload
            ) as response:
                if not self._status(response, missing=missing):
                    return None
                body = bytearray()
                for chunk in response.iter_bytes():
                    check_transfer_cancelled()
                    if len(body) + len(chunk) > MAX_METADATA_BYTES:
                        raise StorageVerificationError("oversized HTTP storage metadata")
                    body.extend(chunk)
                check_transfer_cancelled()
                return json.loads(body)
        except httpx.TransportError as exc:
            raise BulkTransportError(type(exc).__name__, retryable=True) from None
        except (TypeError, ValueError):
            raise StorageVerificationError("invalid HTTP storage metadata") from None

    @staticmethod
    def _metadata(result) -> dict:
        if (
            not isinstance(result, dict)
            or type(result.get("IsDir")) is not bool
            or type(result.get("Size")) is not int
            or result["Size"] < 0
            or not isinstance(result.get("ID"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", result["ID"])
        ):
            raise StorageVerificationError("invalid HTTP storage object identity")
        return result

    def stat(self, logical: str) -> dict | None:
        try:
            private_upload_path(logical)
        except ValueError:
            result = self._json("GET", "stat", logical=logical, missing=True)
            return None if result is None else self._metadata(result)
        return self._uploads.stat(logical)

    def healthcheck(self) -> bool:
        result = self._json("GET", "health")
        return isinstance(result, dict) and result.get("available") is True

    def digest(self, logical_path: str) -> ObjectDigest | None:
        result = self._json(
            "POST",
            "digest",
            payload={
                "site_id": self._site_id,
                "logical_path": normalize_logical_path(logical_path),
            },
            missing=True,
        )
        if result is None:
            return None
        try:
            digest = DigestResponse.model_validate(result)
        except ValueError:
            raise StorageVerificationError("invalid authority digest") from None
        return ObjectDigest(digest.size_bytes, digest.sha256)

    def read(
        self,
        logical: str,
        output: BinaryIO,
        offset: int,
        count: int,
        *,
        expected_identity: str | None = None,
    ) -> None:
        if type(offset) is not int or type(count) is not int or offset < 0 or count < 0:
            raise ValueError("invalid HTTP range")
        if count == 0:
            return
        info = self.stat(logical)
        if info is None or info["IsDir"] or offset + count > info["Size"]:
            raise StorageVerificationError("HTTP range does not match the source")
        identity = expected_identity or info["ID"]
        if identity != info["ID"]:
            raise StorageVerificationError("HTTP source changed before resume")
        check_transfer_cancelled()
        headers = {"Range": f"bytes={offset}-{offset + count - 1}", "If-Match": f'"{identity}"'}
        try:
            stream = (
                self._bulk.stream
                if self._bulk_connections is None
                else lambda *args, **kwargs: self._bulk_connections.stream(
                    "download", *args, transfer_size=count, **kwargs
                )
            )
            with stream(
                "GET",
                self._bulk_url + "/file",
                params={"site_id": self._site_id, "logical_path": normalize_logical_path(logical)},
                headers=headers,
            ) as response:
                self._status(response)
                if (
                    response.status_code != 206
                    or response.headers.get("etag") != f'"{identity}"'
                    or response.headers.get("content-range")
                    != f"bytes {offset}-{offset + count - 1}/{info['Size']}"
                    or response.headers.get("content-length") != str(count)
                    or response.headers.get("content-encoding", "identity") != "identity"
                ):
                    raise StorageVerificationError("HTTP range acknowledgement mismatch")
                received = 0
                # No fixed iter_bytes chunk size: forward available bytes now,
                # rather than waiting for a MiB to accumulate on a slow link.
                for content in response.iter_raw():
                    check_transfer_cancelled()
                    if received + len(content) > count:
                        raise StorageVerificationError("HTTP file exceeded its pinned size")
                    written = output.write(content)
                    if written != len(content):
                        raise OSError("short local file write")
                    received += len(content)
                    if self._bulk_connections is not None:
                        self._bulk_connections.progress(response, len(content))
                    transfer_progress("downloading", offset + received, info["Size"])
                check_transfer_cancelled()
                if received != count:
                    raise BulkTransportError("HTTPTruncated", retryable=True)
        except httpx.TransportError as exc:
            raise BulkTransportError(type(exc).__name__, retryable=True) from None

    def write(self, logical, source, offset, size):
        return self._uploads.write(logical, source, offset, size)

    def rename(self, source, destination):
        return self._uploads.rename(source, destination)

    def remove(self, logical):
        # Only receipt-owned upload prefixes are removable through this API.
        return self._uploads.remove(private_upload_path(logical))

    def _directory(self, logical: str):
        after, identity = "", None
        while True:
            result = self._json(
                "GET",
                "list",
                logical=logical,
                after=after,
                headers=None if identity is None else {"If-Match": f'"{identity}"'},
            )
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("entries"), list)
                or len(result["entries"]) > 256
                or not isinstance(result.get("ID"), str)
            ):
                raise StorageVerificationError("invalid HTTP directory page")
            if identity is not None and result["ID"] != identity:
                raise StorageVerificationError("HTTP directory changed during pagination")
            identity = result["ID"]
            for member in result["entries"]:
                self._metadata(member)
                try:
                    name = validate_component(member["name"])
                except (KeyError, ValueError, TypeError):
                    raise StorageVerificationError("unsafe HTTP directory member") from None
                if name <= after:
                    raise StorageVerificationError("unordered HTTP directory page")
                after = name
                yield member
            cursor = result.get("next")
            if cursor is None:
                return
            if not result["entries"] or cursor != after:
                raise StorageVerificationError("invalid HTTP directory cursor")

    def download_tree(self, logical: str, destination: Path) -> None:
        """Safe paged tree copy; completed members and partials survive retries."""
        path = normalize_logical_path(logical)
        if destination.is_symlink() or not destination.is_dir():
            raise StorageVerificationError("unsafe HTTP tree staging directory")
        # Never place protocol partials among user-named tree members. A real
        # input named '.x.http-part' must not collide with the partial for 'x'.
        partials = destination.with_name(destination.name + ".http-parts")
        if partials.is_symlink():
            raise StorageVerificationError("unsafe HTTP tree partial directory")
        partials.mkdir(mode=0o700, exist_ok=True)
        pending = [(path, destination)]
        visited = 0
        while pending:
            parent, local = pending.pop()
            for member in self._directory(parent):
                check_transfer_cancelled()
                visited += 1
                if visited > 100_000:
                    raise StorageVerificationError("HTTP tree exceeds bounded member limit")
                child = local / member["name"]
                if child.is_symlink():
                    raise StorageVerificationError("unsafe HTTP tree cache member")
                remote = parent + "/" + member["name"]
                if member["IsDir"]:
                    child.mkdir(mode=0o700, exist_ok=True)
                    pending.append((remote, child))
                    continue
                expected = self.digest(remote)
                if expected is None or expected.size_bytes != member["Size"]:
                    raise StorageVerificationError("HTTP tree member changed")
                if (
                    child.exists()
                    and stat.S_ISREG(child.lstat().st_mode)
                    and fsync_and_digest(child) == expected
                ):
                    continue
                key = hashlib.sha256(remote.encode()).hexdigest()
                partial = partials / (key + ".part")
                try:
                    partial_status = partial.lstat()
                except FileNotFoundError:
                    partial_status = None
                if partial_status is not None and not stat.S_ISREG(partial_status.st_mode):
                    raise StorageVerificationError("unsafe HTTP tree partial")
                offset = partial_status.st_size if partial_status is not None else 0
                if offset > expected.size_bytes:
                    partial.unlink()
                    offset = 0
                if (
                    shutil.disk_usage(destination).free - (expected.size_bytes - offset)
                    < self._minimum_free_bytes
                ):
                    raise BulkTransportError("CacheDiskPressure")
                if (
                    offset == 0
                    and self._small_files is not None
                    and self._small_files.accepts(remote, expected.size_bytes)
                ):
                    try:
                        content, _ = self._small_files.download(
                            remote, expected=expected, expected_size_bytes=expected.size_bytes
                        )
                    except SmallFileFallback:
                        pass
                    else:
                        with partial.open("wb") as output:
                            output.write(content)
                            output.flush()
                            os.fsync(output.fileno())
                        offset = expected.size_bytes
                with partial.open("ab", buffering=0) as output:
                    try:
                        self.read(
                            remote,
                            output,
                            offset,
                            expected.size_bytes - offset,
                            expected_identity=member["ID"],
                        )
                    finally:
                        os.fsync(output.fileno())
                if fsync_and_digest(partial) != expected:
                    partial.unlink()
                    raise StorageVerificationError("HTTP tree member failed verification")
                os.replace(partial, child)
        # A completed copy has no pending members. Do not include protocol
        # state in the atomically published input directory.
        try:
            partials.rmdir()
        except OSError:
            pass
