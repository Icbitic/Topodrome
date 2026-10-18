"""Bounded append-only writes into private prefixes, never final job outputs.

This enrolled-agent endpoint cannot overwrite a published object, choose a shell
command, or report a job complete. Publication stays with the fenced mirror.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import stat
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from starlette.requests import ClientDisconnect

from topodrome.protocol.auth import extract_bearer, token_matches
from topodrome.scheduler.small_files import _create_parent, _identity
from topodrome.scheduler.storage_digest import _open_parent, hash_beneath
from topodrome.scheduler.storage_retention import finish_io
from topodrome.storage.bulk_upload import MAX_UPLOAD_CHUNK_BYTES, private_upload_path
from topodrome.storage.extent_encoding import decode_extent

UPLOAD_ENCODING_HEADERS = {"Accept-Encoding": "gzip, identity", "Cache-Control": "no-store"}


def append_encoded_prefix(
    root: Path,
    path: str,
    offset: int,
    total: int,
    content: bytes,
    checksum: str,
    encoding: str,
    raw_length: int,
):
    decoded = decode_extent(content, encoding, raw_length)
    return append_prefix(root, path, offset, total, decoded, checksum)


def append_prefix(root: Path, path: str, offset: int, total: int, content: bytes, checksum: str):
    path = private_upload_path(path)
    if (
        len(content) > MAX_UPLOAD_CHUNK_BYTES
        or offset + len(content) > total
        or (not content and total != 0)
        or hashlib.sha256(content).hexdigest() != checksum
    ):
        raise ValueError("invalid upload extent")
    if shutil.disk_usage(root).free < 1024**3 + len(content):
        raise HTTPException(507, "authority storage is below its free-space floor")
    parts = path.split("/")
    directory = _create_parent(root, parts)
    descriptor = None
    try:
        descriptor = os.open(
            parts[-1],
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
            dir_fd=directory,
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("unsafe upload prefix")
        if info.st_size == offset:
            written = 0
            while written < len(content):
                count = os.pwrite(descriptor, content[written:], offset + written)
                if count <= 0:
                    raise OSError("upload write did not advance")
                written += count
        elif info.st_size >= offset + len(content) and info.st_size <= total:
            # Lost acknowledgements are harmless; never blindly append twice.
            if os.pread(descriptor, len(content), offset) != content:
                raise ValueError("upload retry conflicts with its committed extent")
        else:
            raise ValueError("upload offset is not a contiguous prefix")
        os.fsync(descriptor)
        current = _open_parent(root, parts)
        try:
            if _identity(os.fstat(current))[:2] != _identity(os.fstat(directory))[:2] or _identity(
                os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
            ) != _identity(os.fstat(descriptor)):
                raise ValueError("upload prefix was replaced")
        finally:
            os.close(current)
        os.fsync(directory)
        return {"offset": offset + len(content), "sha256": checksum}
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def bulk_upload_router(root: Path, enrollment: Callable[[str], str | None]) -> APIRouter:
    router = APIRouter()
    slots = threading.BoundedSemaphore(2)
    # A slow body can occupy its reception slot for minutes. Prefix inspection
    # and publication must retain independently bounded capacity throughout.
    prefix_slots = threading.BoundedSemaphore(4)

    def authenticate(request: Request, site_id: str, logical_path: str) -> str:
        expected = enrollment(site_id)
        if expected is None or not token_matches(
            extract_bearer(request.headers.get("authorization")), expected
        ):
            raise HTTPException(403, "site enrollment rejected")
        try:
            return private_upload_path(logical_path)
        except ValueError:
            raise HTTPException(422, "invalid private upload prefix") from None

    async def prefix_io(operation, *args):
        if not prefix_slots.acquire(blocking=False):
            raise HTTPException(503, "bulk upload busy", headers={"Retry-After": "1"})
        try:
            return await finish_io(operation, *args)
        except FileNotFoundError:
            raise HTTPException(404, "upload prefix not found") from None
        except BlockingIOError:
            raise HTTPException(503, "upload prefix busy", headers={"Retry-After": "1"}) from None
        except (OSError, ValueError):
            raise HTTPException(409, "upload prefix could not be safely accessed") from None
        finally:
            prefix_slots.release()

    @router.get("/api/v1/agents/storage/upload-prefix")
    async def metadata(
        request: Request,
        site_id: str = Query(min_length=1, max_length=256),
        logical_path: str = Query(min_length=1, max_length=4096),
    ):
        path = authenticate(request, site_id, logical_path)
        try:
            result = await prefix_io(prefix_operation, root, path, "stat")
        except HTTPException as exc:
            if exc.status_code == 404:
                raise HTTPException(404, exc.detail, headers=UPLOAD_ENCODING_HEADERS) from None
            raise
        return Response(
            json.dumps(result), media_type="application/json", headers=UPLOAD_ENCODING_HEADERS
        )

    @router.delete("/api/v1/agents/storage/upload-prefix", status_code=204)
    async def remove(
        request: Request,
        site_id: str = Query(min_length=1, max_length=256),
        logical_path: str = Query(min_length=1, max_length=4096),
    ):
        path = authenticate(request, site_id, logical_path)
        await prefix_io(prefix_operation, root, path, "remove")
        return Response(status_code=204)

    @router.post("/api/v1/agents/storage/upload-prefix", status_code=204)
    async def promote(
        request: Request,
        site_id: str = Query(min_length=1, max_length=256),
        logical_path: str = Query(min_length=1, max_length=4096),
        destination: str = Query(min_length=1, max_length=4096),
    ):
        path = authenticate(request, site_id, logical_path)
        parent, _, name = path.rpartition("/")
        if destination != parent + "/" + name[1:].rsplit(".topodrome-upload-", 1)[0]:
            raise HTTPException(422, "promotion target does not belong to the prefix")
        await prefix_io(prefix_operation, root, path, "promote", destination.rsplit("/", 1)[1])
        return Response(status_code=204)

    @router.put("/api/v1/agents/storage/upload-prefix")
    async def upload(
        request: Request,
        site_id: str = Query(min_length=1, max_length=256),
        logical_path: str = Query(min_length=1, max_length=4096),
        offset: int = Query(ge=0, le=1024**4),
        total: int = Query(ge=0, le=1024**4),
        content_length: int = Header(ge=0, le=MAX_UPLOAD_CHUNK_BYTES),
        x_topodrome_sha256: str = Header(pattern=r"^[a-f0-9]{64}$"),
        content_encoding: str = Header(default="identity"),
        x_topodrome_uncompressed_length: int | None = Header(
            default=None, ge=0, le=MAX_UPLOAD_CHUNK_BYTES
        ),
    ):
        path = authenticate(request, site_id, logical_path)
        content_encoding = content_encoding.strip().lower()
        if content_encoding not in {"identity", "gzip"}:
            raise HTTPException(415, "unsupported upload encoding", headers=UPLOAD_ENCODING_HEADERS)
        raw_length = content_length
        if content_encoding == "gzip":
            if x_topodrome_uncompressed_length is None:
                raise HTTPException(422, "gzip upload requires decoded length")
            raw_length = x_topodrome_uncompressed_length
        elif x_topodrome_uncompressed_length not in {None, content_length}:
            raise HTTPException(422, "upload decoded length mismatch")
        try:
            if offset + raw_length > total or (raw_length == 0 and total != 0):
                raise ValueError("extent exceeds declared source")
        except ValueError:
            raise HTTPException(422, "invalid private upload extent") from None
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "bulk upload busy", headers={"Retry-After": "1"})
        try:
            content = bytearray()
            try:
                async with asyncio.timeout(120):
                    async for chunk in request.stream():
                        if len(content) + len(chunk) > content_length:
                            raise HTTPException(400, "upload length mismatch")
                        content.extend(chunk)
            except TimeoutError:
                raise HTTPException(408, "upload chunk reception timed out") from None
            except ClientDisconnect:
                raise HTTPException(408, "upload chunk reception disconnected") from None
            if len(content) != content_length:
                raise HTTPException(400, "upload length mismatch")
            result = await finish_io(
                append_encoded_prefix,
                root,
                path,
                offset,
                total,
                bytes(content),
                x_topodrome_sha256,
                content_encoding,
                raw_length,
            )
        except BlockingIOError:
            raise HTTPException(503, "upload prefix busy", headers={"Retry-After": "1"}) from None
        except (OSError, ValueError):
            raise HTTPException(409, "upload prefix could not be safely extended") from None
        finally:
            slots.release()
        return Response(
            json.dumps(result),
            media_type="application/json",
            headers=UPLOAD_ENCODING_HEADERS,
        )

    return router


def prefix_operation(root: Path, path: str, operation: str, destination: str | None = None):
    """Operate on the real filesystem, not a second storage view."""
    parts = private_upload_path(path).split("/")
    directory = _open_parent(root, parts)
    descriptor = None
    try:
        descriptor = os.open(parts[-1], os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("unsafe upload prefix")
        if _identity(os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)) != _identity(
            info
        ):
            raise ValueError("upload prefix changed")
        current = _open_parent(root, parts)
        try:
            if _identity(os.fstat(current))[:2] != _identity(os.fstat(directory))[:2]:
                raise ValueError("upload parent changed")
        finally:
            os.close(current)
        if operation == "stat":
            return {"IsDir": False, "Size": info.st_size, "ModTime": str(info.st_mtime_ns)}
        if operation == "remove":
            os.unlink(parts[-1], dir_fd=directory)
        elif operation == "promote":
            if destination is None:
                raise ValueError("missing promotion target")
            os.fsync(descriptor)
            # Only the small current-publication pointer is mutable. Payload
            # paths are immutable even when a second sender races deduplication.
            pointer = (
                parts[0] == "jobs" and parts[-2] == "committed" and destination == "manifest.json"
            )
            if pointer:
                try:
                    existing = os.stat(destination, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    if not stat.S_ISREG(existing.st_mode):
                        raise ValueError("unsafe publication pointer")
                os.replace(parts[-1], destination, src_dir_fd=directory, dst_dir_fd=directory)
            else:
                try:
                    os.link(
                        parts[-1],
                        destination,
                        src_dir_fd=directory,
                        dst_dir_fd=directory,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    target = "/".join([*parts[:-1], destination])
                    if hash_beneath(root, path) != hash_beneath(root, target):
                        raise ValueError("immutable upload target conflicts") from None
                os.unlink(parts[-1], dir_fd=directory)
        else:
            raise ValueError("unsupported prefix operation")
        os.fsync(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)
