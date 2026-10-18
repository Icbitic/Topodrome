"""Enrolled-agent HTTP reads: bounded metadata and identity-fenced range streams.

No shell, public listener or unbounded in-memory file bodies. This
is the same trusted storage boundary as the agent's digest/prefix endpoints;
it does not grant publication authority or mark a job successful.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import stat
import threading
from functools import partial
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException, Query, Request
from starlette.responses import Response, StreamingResponse

from topodrome.protocol.auth import extract_bearer, token_matches
from topodrome.scheduler.small_files import _identity
from topodrome.scheduler.storage_digest import _open_parent
from topodrome.storage.link_probe import LINK_PROBE_BYTES
from topodrome.storage.paths import normalize_logical_path, validate_component

READ_CHUNK_BYTES = 256 * 1024
MAX_LIST_ENTRIES = 256
MAX_DIRECTORY_ENTRIES = 100_000
READ_ROOTS = frozenset({"blobs", "jobs", "artifacts", "sources", "projects", "bootstrap"})


def read_path(logical: str) -> str:
    path = normalize_logical_path(logical)
    if path != "storage-layout.json" and path.split("/", 1)[0] not in READ_ROOTS:
        raise ValueError("outside the agent storage namespace")
    return path


def version(info: os.stat_result) -> str:
    # Opaque strong version of this immutable local object, NOT a claimed SHA
    # of its bytes. Pins and the authority's independent digest verify content.
    return hashlib.sha256(":".join(map(str, _identity(info))).encode()).hexdigest()


def metadata(info: os.stat_result) -> dict:
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError("not a regular file or directory")
    return {
        "IsDir": stat.S_ISDIR(info.st_mode),
        "Size": 0 if stat.S_ISDIR(info.st_mode) else info.st_size,
        "ModTime": str(info.st_mtime_ns),
        "ID": version(info),
    }


def open_read(root: Path, logical: str) -> tuple[int, os.stat_result]:
    parts = read_path(logical).split("/")
    parent = _open_parent(root, parts)
    try:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            info = os.fstat(fd)
            metadata(info)
            return fd, info
        except BaseException:
            os.close(fd)
            raise
    finally:
        os.close(parent)


def inspect(root: Path, logical: str) -> dict:
    fd, info = open_read(root, logical)
    try:
        return metadata(info)
    finally:
        os.close(fd)


def list_page(root: Path, logical: str, after: str, limit: int, match: str | None) -> dict:
    fd, before = open_read(root, logical)
    try:
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("not a directory")
        if match is not None and match != f'"{version(before)}"':
            raise HTTPException(412, "directory identity changed")
        names = []
        with os.scandir(fd) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > MAX_DIRECTORY_ENTRIES:
                    raise HTTPException(413, "directory exceeds bounded listing limit")
        selected = sorted(name for name in names if name > after)[: limit + 1]
        members = []
        for name in selected[:limit]:
            validate_component(name)
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            members.append({"name": name, **metadata(info)})
        if _identity(before) != _identity(os.fstat(fd)) or inspect(root, logical)["ID"] != version(
            before
        ):
            raise HTTPException(412, "directory changed during listing")
        return {
            "ID": version(before),
            "entries": members,
            "next": selected[limit - 1] if len(selected) > limit else None,
        }
    finally:
        os.close(fd)


def file_transport_router(root: Path, enrollment) -> APIRouter:
    router = APIRouter()
    metadata_slots = threading.BoundedSemaphore(8)
    stream_slots = threading.BoundedSemaphore(2)
    metadata_io = anyio.CapacityLimiter(8)
    stream_io = anyio.CapacityLimiter(2)
    probe_slots = asyncio.Semaphore(2)
    probe_body = os.urandom(LINK_PROBE_BYTES)
    probe_headers = {
        "Cache-Control": "private, no-store",
        "X-Topodrome-Sha256": hashlib.sha256(probe_body).hexdigest(),
    }

    async def io(function, *args, bulk=False, on_cancel=None):
        # Do not close a descriptor while an abandoned thread still uses it.
        task = asyncio.create_task(
            anyio.to_thread.run_sync(
                partial(function, *args), limiter=stream_io if bulk else metadata_io
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            result = task.result()
            if on_cancel is not None:
                on_cancel(result)
            raise

    def authorize(request: Request, site_id: str, logical: str) -> str:
        expected = enrollment(site_id)
        if expected is None or not token_matches(
            extract_bearer(request.headers.get("authorization")), expected
        ):
            raise HTTPException(403, "site enrollment rejected")
        try:
            return read_path(logical)
        except ValueError:
            raise HTTPException(422, "unsafe storage path") from None

    async def bounded_metadata(function, *args):
        if not metadata_slots.acquire(blocking=False):
            raise HTTPException(503, "storage metadata busy", headers={"Retry-After": "1"})
        try:
            return await io(function, *args)
        except FileNotFoundError:
            raise HTTPException(404, "object not found") from None
        except (ValueError, OSError):
            raise HTTPException(409, "object could not be safely inspected") from None
        finally:
            metadata_slots.release()

    @router.get("/api/v1/agents/storage/stat")
    async def stat_object(
        request: Request,
        site_id: str = Query(max_length=256),
        logical_path: str = Query(max_length=4096),
    ):
        path = authorize(request, site_id, logical_path)
        return await bounded_metadata(inspect, root, path)

    @router.get("/api/v1/agents/storage/health")
    async def health(request: Request, site_id: str = Query(max_length=256)):
        authorize(request, site_id, "storage-layout.json")
        await bounded_metadata(inspect, root, "storage-layout.json")
        return {"available": await io(lambda: shutil.disk_usage(root).free >= 1024**3)}

    @router.api_route("/api/v1/agents/storage/link-probe", methods=["GET", "POST"])
    async def link_probe(
        request: Request,
        site_id: str = Query(max_length=256),
        size: int = Query(default=LINK_PROBE_BYTES, ge=0, le=LINK_PROBE_BYTES),
    ):
        authorize(request, site_id, "storage-layout.json")
        if request.headers.get("content-encoding", "identity") != "identity":
            raise HTTPException(415, "link probes require identity encoding")
        if probe_slots.locked():
            raise HTTPException(429, "link probes busy", headers={"Retry-After": "10"})
        async with probe_slots:
            if request.method == "GET":
                if size not in {0, LINK_PROBE_BYTES}:
                    raise HTTPException(422, "invalid probe size")
                return Response(
                    probe_body[:size], headers=probe_headers, media_type="application/octet-stream"
                )
            if request.headers.get("content-length") != str(LINK_PROBE_BYTES):
                raise HTTPException(413, "invalid probe body length")
            digest, received = hashlib.sha256(), 0
            try:
                async with asyncio.timeout(15):
                    async for chunk in request.stream():
                        received += len(chunk)
                        if received > LINK_PROBE_BYTES:
                            raise HTTPException(413, "probe body too large")
                        digest.update(chunk)
            except TimeoutError:
                raise HTTPException(408, "link probe timeout") from None
            if received != LINK_PROBE_BYTES:
                raise HTTPException(422, "truncated probe body")
            return {"bytes": received, "sha256": digest.hexdigest()}

    @router.get("/api/v1/agents/storage/list")
    async def listing(
        request: Request,
        site_id: str = Query(max_length=256),
        logical_path: str = Query(max_length=4096),
        after: str = Query(default="", max_length=255),
        limit: int = Query(default=MAX_LIST_ENTRIES, ge=1, le=MAX_LIST_ENTRIES),
    ):
        path = authorize(request, site_id, logical_path)
        if after:
            try:
                validate_component(after)
            except ValueError:
                raise HTTPException(422, "invalid listing cursor") from None
        return await bounded_metadata(
            list_page, root, path, after, limit, request.headers.get("if-match")
        )

    @router.get("/api/v1/agents/storage/file")
    async def download(
        request: Request,
        site_id: str = Query(max_length=256),
        logical_path: str = Query(max_length=4096),
    ):
        path = authorize(request, site_id, logical_path)
        if not stream_slots.acquire(blocking=False):
            raise HTTPException(503, "storage streams busy", headers={"Retry-After": "1"})
        fd = None
        handed_off = False
        try:
            fd, info = await io(
                open_read, root, path, bulk=True, on_cancel=lambda result: os.close(result[0])
            )
            if not stat.S_ISREG(info.st_mode):
                raise HTTPException(409, "not a regular file")
            etag = f'"{version(info)}"'
            if request.headers.get("if-match", etag) != etag:
                raise HTTPException(412, "immutable object identity changed")
            start, end, status = 0, info.st_size - 1, 200
            requested = request.headers.get("range")
            if requested is not None:
                import re

                match = re.fullmatch(r"bytes=([0-9]+)-([0-9]*)", requested)
                if match is None:
                    raise HTTPException(416, "unsupported range")
                start = int(match[1])
                end = int(match[2]) if match[2] else end
                if start > end or end >= info.st_size:
                    raise HTTPException(
                        416,
                        "range outside object",
                        headers={"Content-Range": f"bytes */{info.st_size}"},
                    )
                status = 206
            headers = {
                "ETag": etag,
                "Accept-Ranges": "bytes",
                "Content-Length": str(max(0, end - start + 1)),
                "Cache-Control": "private, no-store",
            }
            if status == 206:
                headers["Content-Range"] = f"bytes {start}-{end}/{info.st_size}"
            descriptor = fd
            closed = False

            def close_stream():
                nonlocal closed
                if not closed:
                    closed = True
                    os.close(descriptor)
                    stream_slots.release()

            class FileResponse(StreamingResponse):
                async def __call__(self, scope, receive, send):
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        close_stream()

            async def body():
                try:
                    position = start
                    while position <= end:
                        if await request.is_disconnected():
                            return
                        content = await io(
                            os.pread,
                            descriptor,
                            min(READ_CHUNK_BYTES, end - position + 1),
                            position,
                            bulk=True,
                        )
                        if not content or _identity(
                            await io(os.fstat, descriptor, bulk=True)
                        ) != _identity(info):
                            raise RuntimeError("immutable object changed during stream")
                        position += len(content)
                        if position > end and (await io(inspect, root, path, bulk=True))[
                            "ID"
                        ] != version(info):
                            raise RuntimeError("immutable object changed during stream")
                        yield content
                finally:
                    close_stream()

            response = FileResponse(
                body(), status_code=status, headers=headers, media_type="application/octet-stream"
            )
            handed_off = True
            return response
        except FileNotFoundError:
            raise HTTPException(404, "object not found") from None
        except (ValueError, OSError):
            raise HTTPException(409, "object could not be safely read") from None
        finally:
            if not handed_off:
                if fd is not None:
                    os.close(fd)
                stream_slots.release()

    return router
