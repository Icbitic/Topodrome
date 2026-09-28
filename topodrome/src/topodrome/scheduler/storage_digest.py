"""Bounded server-local hashing for enrolled storage clients."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from topodrome.db.models import AttemptRow, JobRow
from topodrome.protocol.auth import extract_bearer, token_matches
from topodrome.storage.base import ObjectDigest
from topodrome.storage.blobs import claim_blob
from topodrome.storage.paths import normalize_logical_path
from topodrome.storage.remote_digest import DigestResponse


class DigestRequest(BaseModel):
    site_id: str = Field(min_length=1, max_length=256)
    logical_path: str = Field(min_length=1, max_length=4096)


class BlobClaimRequest(BaseModel):
    site_id: str = Field(min_length=1, max_length=256)
    job_id: str = Field(min_length=1, max_length=80)
    attempt_id: str = Field(min_length=1, max_length=80)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0, le=1024**4)


def _open_parent(root: Path, parts: list[str]) -> int:
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return directory
    except BaseException:
        os.close(directory)
        raise


def hash_beneath(root: Path, logical_path: str) -> DigestResponse:
    """Open each component relative to a pinned descriptor, rejecting symlinks."""
    parts = normalize_logical_path(logical_path).split("/")
    directory = _open_parent(root, parts)
    try:
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        with os.fdopen(descriptor, "rb", buffering=0) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("not a regular file")
            digest = hashlib.sha256()
            count = 0
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                count += len(chunk)
            os.fsync(stream.fileno())
            after = os.fstat(stream.fileno())
            # Rewalk from the authority root, detecting an ancestor replaced
            # while we hashed the previously opened descriptor.
            current_parent = _open_parent(root, parts)
            try:
                current = os.stat(parts[-1], dir_fd=current_parent, follow_symlinks=False)
            finally:
                os.close(current_parent)

            def identity(s: os.stat_result) -> tuple[int, ...]:
                return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

            if identity(before) != identity(after) or identity(after) != identity(current):
                raise ValueError("file changed during verification")
            if count != after.st_size:
                raise ValueError("file length changed")
            os.fsync(directory)
            return DigestResponse(size_bytes=count, sha256=digest.hexdigest())
    finally:
        os.close(directory)


def digest_router(root: Path, enrollment: Callable[[str], str | None]) -> APIRouter:
    router = APIRouter()
    slots = threading.BoundedSemaphore(2)

    @router.post("/api/v1/agents/storage/blobs/claim", status_code=204)
    async def blob_claim(body: BlobClaimRequest, request: Request):
        from topodrome.scheduler.storage_retention import finish_io

        expected = enrollment(body.site_id)
        if expected is None or not token_matches(
            extract_bearer(request.headers.get("authorization")), expected
        ):
            raise HTTPException(403, "site enrollment rejected")
        async with request.app.state.database.session() as session:
            attempt = await session.get(AttemptRow, body.attempt_id)
            job = await session.get(JobRow, body.job_id)
            if (
                attempt is None
                or job is None
                or attempt.job_id != job.id
                or attempt.site_id != body.site_id
                or str(job.state) not in {"running", "assigning"}
                or str(attempt.state) not in {"running", "checkpointing", "committing"}
            ):
                raise HTTPException(409, "attempt has no active publication authority")
        await finish_io(
            claim_blob,
            root,
            body.job_id,
            body.attempt_id,
            ObjectDigest(body.size_bytes, body.sha256),
        )
        return Response(status_code=204)

    @router.post("/api/v1/agents/storage/digest", response_model=DigestResponse)
    def digest(body: DigestRequest, request: Request, response: Response) -> DigestResponse:
        expected = enrollment(body.site_id)
        if expected is None or not token_matches(
            extract_bearer(request.headers.get("authorization")), expected
        ):
            raise HTTPException(403, "site enrollment rejected")
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "storage verification busy", headers={"Retry-After": "1"})
        try:
            result = hash_beneath(root, body.logical_path)
        except FileNotFoundError:
            raise HTTPException(404, "object not found") from None
        except (OSError, ValueError):
            raise HTTPException(409, "object could not be verified") from None
        finally:
            slots.release()
        response.headers["Cache-Control"] = "no-store"
        return result

    return router
