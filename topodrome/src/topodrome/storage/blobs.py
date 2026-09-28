"""Whole-file immutable blobs and an authority-local, durable object index.

The index enumerates owned objects and pending output claims. PostgreSQL job
state and authorized asset manifests remain the lifetime/permission authority.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import sqlite3
import stat
import time
from contextlib import contextmanager
from pathlib import Path

from topodrome.storage.base import ObjectDigest, StorageVerificationError
from topodrome.storage.local import fsync_and_digest
from topodrome.storage.paths import resolve_beneath

_SHA = re.compile(r"^[a-f0-9]{64}$")


def validate_layout(path: Path, expected: str) -> None:
    if not path.exists():
        return
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > 4096:
        raise ValueError("invalid storage layout marker")
    if json.loads(path.read_text()).get("layout") != expected:
        raise ValueError("configured storage layout differs from authority; upgrade required")


def blob_path(sha256: str) -> str:
    if not _SHA.fullmatch(sha256):
        raise ValueError("invalid blob digest")
    return f"blobs/sha256/{sha256[:2]}/{sha256}"


def is_blob_path(path: str, sha256: str) -> bool:
    return path == blob_path(sha256)


@contextmanager
def blob_transaction(root: Path):
    """Cross-process publication/claim/GC order; never held during network I/O."""
    directory = Path(root) / ".blob-index"
    directory.mkdir(mode=0o700, exist_ok=True)
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise ValueError("unsafe blob index directory")
    fd = os.open(directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        path = directory / "index.sqlite3"
        if path.is_symlink():
            raise ValueError("unsafe blob index")
        db = sqlite3.connect(path, timeout=60)
        try:
            os.chmod(path, 0o600)
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS objects (sha256 TEXT PRIMARY KEY, "
                "size INTEGER NOT NULL, touched REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS claims (job_id TEXT NOT NULL, "
                "attempt_id TEXT NOT NULL, sha256 TEXT NOT NULL, "
                "PRIMARY KEY(job_id,attempt_id,sha256))"
            )
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
    finally:
        os.close(fd)


def register_blob(db, digest: ObjectDigest) -> None:
    blob_path(digest.sha256)
    existing = db.execute("SELECT size FROM objects WHERE sha256=?", (digest.sha256,)).fetchone()
    if existing and existing[0] != digest.size_bytes:
        raise StorageVerificationError("blob identity collision")
    db.execute(
        "INSERT INTO objects VALUES(?,?,?) ON CONFLICT(sha256) DO UPDATE SET "
        "touched=excluded.touched",
        (digest.sha256, digest.size_bytes, time.time()),
    )


def claim_blob(root: Path, job_id: str, attempt_id: str, digest: ObjectDigest) -> None:
    with blob_transaction(root) as db:
        register_blob(db, digest)
        db.execute(
            "INSERT OR IGNORE INTO claims VALUES(?,?,?)", (job_id, attempt_id, digest.sha256)
        )


def install_local_blob(root: Path, source: Path, digest: ObjectDigest) -> Path:
    """Atomically link a private, already verified staged file into the CAS.

    Staging and blob roots must share a filesystem. Never overwrite an existing
    object, and verify existing bytes before deduplication. No caller may mutate a
    staged inode after handing it here.
    """
    if not stat.S_ISREG(source.lstat().st_mode):
        raise StorageVerificationError("blob source is not a regular file")
    target = resolve_beneath(root, blob_path(digest.sha256))
    target.parent.mkdir(parents=True, exist_ok=True)
    source.chmod(0o444)
    try:
        os.link(source, target)
    except FileExistsError:
        if not stat.S_ISREG(target.lstat().st_mode) or fsync_and_digest(target) != digest:
            raise StorageVerificationError("existing blob failed integrity verification") from None
    directory = target.parent
    while directory != root:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        directory = directory.parent
    return target


def publish_artifact_blob(storage, source: Path, *, job_id: str, attempt_id: str):
    from topodrome.storage.progress import reuse_verified_outputs

    digest = fsync_and_digest(source)
    # An authenticated claim precedes the first remotely visible byte.
    callback = getattr(storage, "claim_blob", None)
    if callback is None:
        claim_blob(storage.root, job_id, attempt_id, digest)
    else:
        callback(job_id, attempt_id, digest)
    install_local_blob(storage.root, source, digest)
    with reuse_verified_outputs():
        verified = storage.flush_and_verify(blob_path(digest.sha256), expected_sha256=digest.sha256)
    return verified
