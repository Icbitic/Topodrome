"""A bounded immediate-transfer lane, independent of bulk hash/upload slots."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import stat
import threading
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import anyio
from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from sqlalchemy import select

from topodrome.db.models import AttemptRow, JobRow, LeaseRow
from topodrome.protocol.auth import extract_bearer, token_matches
from topodrome.scheduler.storage_digest import _open_parent, hash_beneath
from topodrome.storage.base import ObjectDigest, StorageVerificationError
from topodrome.storage.blobs import blob_path, blob_transaction, claim_blob, register_blob
from topodrome.storage.paths import normalize_logical_path, validate_component
from topodrome.storage.remote_digest import DigestResponse
from topodrome.storage.small_files import MAX_SMALL_FILE_BYTES, small_file_path
from topodrome.storage.source_archive import pack_source_bundle, source_bundle_path


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def read_small_file(root: Path, logical_path: str, maximum: int) -> tuple[bytes, ObjectDigest]:
    parts = normalize_logical_path(logical_path).split("/")
    directory = _open_parent(root, parts)
    try:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            os.close(fd)
            raise OverflowError("use bulk transport")
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if stat.S_ISDIR(before.st_mode) or before.st_size > maximum:
                raise OverflowError("use bulk transport")
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("not a regular file")
            content = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
            current_parent = _open_parent(root, parts)
            try:
                current = os.stat(parts[-1], dir_fd=current_parent, follow_symlinks=False)
            finally:
                os.close(current_parent)
            if (
                _identity(before) != _identity(after)
                or _identity(after) != _identity(current)
                or len(content) != after.st_size
                or len(content) > maximum
            ):
                raise ValueError("file changed during transfer")
            return content, ObjectDigest(len(content), hashlib.sha256(content).hexdigest())
    finally:
        os.close(directory)


def _create_parent(root: Path, parts: list[str]) -> int:
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory)
                os.fsync(directory)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return directory
    except BaseException:
        os.close(directory)
        raise


def install_small_file(
    root: Path,
    logical_path: str,
    content: bytes,
    digest: ObjectDigest,
    *,
    blob_owner: tuple[str, str] | None = None,
    replace_manifest: bool = False,
) -> ObjectDigest:
    """Verify independently and acknowledge only after file and namespace fsync.

    No network I/O is performed while holding the claim/GC order. Blobs and
    immutable attempt paths are never overwritten, including on an uncertain
    retry. Only the job's current publication pointer is replaceable.
    """
    if len(content) != digest.size_bytes or hashlib.sha256(content).hexdigest() != digest.sha256:
        raise StorageVerificationError("small-file upload failed integrity verification")
    parts = normalize_logical_path(logical_path).split("/")
    if shutil.disk_usage(root).free < 1024**3 + MAX_SMALL_FILE_BYTES * 8:
        raise HTTPException(507, "authority storage is below its free-space floor")
    directory = _create_parent(root, parts)
    temporary = f".{parts[-1]}.topodrome-small-{uuid4().hex}"
    try:
        fd = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        # Re-open from the authority, rather than publishing into a directory
        # detached/replaced since the first descriptor was acquired.
        current = _open_parent(root, parts)
        try:
            if _identity(os.fstat(current))[:2] != _identity(os.fstat(directory))[:2]:
                raise ValueError("publication parent changed")
        finally:
            os.close(current)
        with blob_transaction(root) if blob_owner is not None else nullcontext() as index:
            if blob_owner is not None:
                register_blob(index, digest)
                index.execute(
                    "INSERT OR IGNORE INTO claims VALUES(?,?,?)", (*blob_owner, digest.sha256)
                )
            if replace_manifest:
                # Refuse even an in-root symlink, FIFO or directory.
                try:
                    existing = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    if not stat.S_ISREG(existing.st_mode):
                        raise ValueError("unsafe publication target")
                os.replace(temporary, parts[-1], src_dir_fd=directory, dst_dir_fd=directory)
            else:
                try:
                    os.link(
                        temporary,
                        parts[-1],
                        src_dir_fd=directory,
                        dst_dir_fd=directory,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    _, stored = read_small_file(root, logical_path, MAX_SMALL_FILE_BYTES)
                    if stored != digest:
                        raise StorageVerificationError(
                            "immutable small-file target conflicts"
                        ) from None
            os.fsync(directory)
            # Independently inspect the published authority bytes, including
            # after deduplication, before returning the durable acknowledgement.
            stored = hash_beneath(root, logical_path)
            observed = ObjectDigest(stored.size_bytes, stored.sha256)
            if observed != digest:
                raise StorageVerificationError("published small-file verification failed")
        return observed
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
            os.fsync(directory)
        except FileNotFoundError:
            pass
        finally:
            os.close(directory)


async def _authorize_output(
    request: Request, site_id: str, job_id: str, attempt_id: str, *, log: bool, operation=None
):
    """Recheck current durable publication authority immediately before commit.

    The second call locks job -> attempt -> lease across *local* commit, never
    across network reception. Terminal log-outbox delivery remains supported.
    """
    async with request.app.state.database.transaction() as session:
        job = await session.scalar(select(JobRow).where(JobRow.id == job_id).with_for_update())
        attempt = await session.scalar(
            select(AttemptRow).where(AttemptRow.id == attempt_id).with_for_update()
        )
        if job is None or attempt is None or attempt.job_id != job.id or attempt.site_id != site_id:
            raise HTTPException(409, "attempt has no publication authority")
        if not log:
            lease = await session.scalar(
                select(LeaseRow).where(LeaseRow.attempt_id == attempt_id).with_for_update()
            )
            if (
                str(job.state) not in {"running", "assigning"}
                or job.cancel_requested
                or str(attempt.state)
                not in {
                    "running",
                    "preempt_requested",
                    "checkpointing",
                    "committing",
                }
                or lease is None
                or lease.site_id != site_id
                or str(lease.state) not in {"active", "releasing"}
                or lease.expires_at.replace(tzinfo=UTC) <= datetime.now(UTC)
            ):
                raise HTTPException(409, "attempt has no active publication authority")
        if operation is not None:
            return await operation()


def small_file_router(root: Path, enrollment: Callable[[str], str | None]) -> APIRouter:
    router = APIRouter()
    slots = threading.BoundedSemaphore(8)
    # Independent from the default thread limiter and the two long-running
    # authoritative digest slots. Bulk work cannot exhaust this lane's slots.
    io_limit = anyio.CapacityLimiter(8)

    async def io(function, *args, **kwargs):
        task = asyncio.create_task(
            anyio.to_thread.run_sync(
                partial(function, *args, **kwargs),
                limiter=io_limit,
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
            task.result()
            raise

    def authenticate(request: Request, site_id: str, logical_path: str) -> str:
        expected = enrollment(site_id)
        if expected is None or not token_matches(
            extract_bearer(request.headers.get("authorization")),
            expected,
        ):
            raise HTTPException(403, "site enrollment rejected")
        try:
            path = normalize_logical_path(logical_path)
            if not small_file_path(path):
                raise ValueError("path is outside the transport namespace")
        except ValueError:
            raise HTTPException(422, "unsafe storage path") from None
        return path

    @router.get("/api/v1/agents/storage/small-file")
    async def download(
        request: Request,
        site_id: str = Query(min_length=1, max_length=256),
        logical_path: str = Query(min_length=1, max_length=4096),
        max_bytes: int = Query(default=MAX_SMALL_FILE_BYTES, ge=1, le=MAX_SMALL_FILE_BYTES),
    ):
        path = authenticate(request, site_id, logical_path)
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "small-file transport busy", headers={"Retry-After": "1"})
        try:
            content, digest = await io(read_small_file, root, path, max_bytes)
        except FileNotFoundError:
            raise HTTPException(404, "object not found") from None
        except OverflowError:
            raise HTTPException(413, "use bulk transport") from None
        except (OSError, ValueError):
            raise HTTPException(409, "object could not be safely read") from None
        finally:
            slots.release()
        etag = f'"{digest.sha256}"'
        if request.headers.get("if-match", etag) != etag:
            raise HTTPException(412, "immutable object identity changed")
        return Response(
            content,
            media_type="application/octet-stream",
            headers={"ETag": etag, "Cache-Control": "private, no-store"},
        )

    @router.get("/api/v1/agents/storage/source-bundle")
    async def download_source(
        request: Request,
        site_id: str = Query(min_length=1, max_length=256),
        logical_path: str = Query(min_length=1, max_length=4096),
        max_bytes: int = Query(default=MAX_SMALL_FILE_BYTES, ge=1, le=MAX_SMALL_FILE_BYTES),
    ):
        path = authenticate(request, site_id, logical_path)
        if not source_bundle_path(path):
            raise HTTPException(422, "not an immutable source directory")
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "small-file transport busy", headers={"Retry-After": "1"})
        try:
            content = await io(
                pack_source_bundle,
                path,
                max_bytes,
                lambda member, bound: read_small_file(root, member, bound)[0],
            )
        except FileNotFoundError:
            raise HTTPException(404, "object not found") from None
        except OverflowError:
            raise HTTPException(413, "use bulk transport") from None
        except (OSError, ValueError, StorageVerificationError):
            raise HTTPException(409, "source bundle could not be safely read") from None
        finally:
            slots.release()
        return Response(
            content,
            media_type="application/zip",
            headers={
                "ETag": f'"{hashlib.sha256(content).hexdigest()}"',
                "Cache-Control": "private, no-store",
            },
        )

    @router.put("/api/v1/agents/storage/small-file", response_model=DigestResponse)
    async def upload(
        request: Request,
        site_id: str = Query(min_length=1, max_length=256),
        logical_path: str = Query(min_length=1, max_length=4096),
        job_id: str = Query(min_length=1, max_length=80),
        attempt_id: str = Query(min_length=1, max_length=80),
        content_length: int = Header(ge=0),
        x_topodrome_sha256: str = Header(pattern=r"^[a-f0-9]{64}$"),
    ):
        path = authenticate(request, site_id, logical_path)
        if content_length > MAX_SMALL_FILE_BYTES:
            raise HTTPException(413, "use bulk transport")
        try:
            validate_component(job_id)
            validate_component(attempt_id)
        except ValueError:
            raise HTTPException(422, "invalid publication identity") from None
        digest = ObjectDigest(content_length, x_topodrome_sha256)
        is_blob = path == blob_path(digest.sha256)
        prefix = f"jobs/{job_id}/attempts/{attempt_id}/"
        manifest = path == f"jobs/{job_id}/committed/manifest.json"
        log = path == prefix + "logs/attempt.log"
        if not (is_blob or path.startswith(prefix) or manifest):
            raise HTTPException(403, "output path does not belong to the attempt")
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "small-file transport busy", headers={"Retry-After": "1"})
        try:
            await _authorize_output(request, site_id, job_id, attempt_id, log=log)
            if is_blob:
                # Durable claim before the first uploaded payload is published.
                await io(claim_blob, root, job_id, attempt_id, digest)
            content = bytearray()
            try:
                async with asyncio.timeout(30):
                    async for chunk in request.stream():
                        if len(content) + len(chunk) > content_length:
                            raise HTTPException(400, "upload length mismatch")
                        content.extend(chunk)
            except TimeoutError:
                raise HTTPException(408, "small-file reception timed out") from None
            if len(content) != content_length:
                raise HTTPException(400, "upload length mismatch")
            if manifest:
                from topodrome.storage.artifacts import CommitManifest

                try:
                    publication = CommitManifest.from_bytes(bytes(content))
                    if publication.job_id != job_id or publication.attempt_id != attempt_id:
                        raise ValueError("publication identity mismatch")
                except ValueError:
                    raise HTTPException(422, "invalid publication manifest") from None
            observed = await _authorize_output(
                request,
                site_id,
                job_id,
                attempt_id,
                log=log,
                operation=lambda: io(
                    install_small_file,
                    root,
                    path,
                    bytes(content),
                    digest,
                    blob_owner=(job_id, attempt_id) if is_blob else None,
                    replace_manifest=manifest,
                ),
            )
        except (OSError, ValueError, StorageVerificationError):
            raise HTTPException(409, "object could not be durably published") from None
        finally:
            slots.release()
        return Response(
            DigestResponse(
                size_bytes=observed.size_bytes, sha256=observed.sha256
            ).model_dump_json(),
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )

    return router
