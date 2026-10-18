"""Reference-aware, bounded GC of terminal attempts in authoritative storage.

No recursive directory removal, payload hashing, or remote command is used.
Only database-owned terminal attempts are candidates. The caller serializes
collection with submission/publication for the entire snapshot-to-unlink span.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
from collections import Counter
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, text

from topodrome.db.models import ArtifactRow, AttemptRow, JobRow, LeaseRow
from topodrome.domain.errors import AdmissionRejectedError
from topodrome.scheduler.assets import _read_manifest
from topodrome.storage.paths import normalize_logical_path, validate_component

TERMINAL_JOBS = {"succeeded", "failed", "canceled"}
TERMINAL_ATTEMPTS = {"succeeded", "failed", "canceled", "lost", "preempted", "rejected"}
LIVE_LEASES = {"offered", "accepted", "active", "releasing"}
CHECKPOINT = re.compile(r"^\d{8,}-[a-f0-9]{12}\.pkl$")
STORAGE_LOCK_ID = 0x544F504F53544F52


async def finish_io(function, *args, **kwargs):
    """Do not release reference locks while a canceled thread still mutates disk."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
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


def serialized_storage(function):
    @wraps(function)
    async def guarded(self, *args, **kwargs):
        async with self._storage_reference_lock:
            # Also orders independently constructed service instances/processes.
            async with self.database.transaction() as session:
                if self.database.engine.dialect.name == "postgresql":
                    await session.execute(
                        text("SELECT pg_advisory_xact_lock(:key)"), {"key": STORAGE_LOCK_ID}
                    )
                return await function(self, *args, **kwargs)

    return guarded


def regular_file(root, logical):
    """Reject symlinks in *every* path component, not merely the final file."""
    parts = Path(normalize_logical_path(logical)).parts
    path = root
    for index, part in enumerate(parts):
        path = path / part
        metadata = path.lstat()
        required = stat.S_ISREG if index == len(parts) - 1 else stat.S_ISDIR
        if not required(metadata.st_mode):
            raise ValueError("retention refuses symlink or non-regular storage path")
    return path, metadata


def require_members(root, members):
    for member in members:
        try:
            _, metadata = regular_file(root, member.logical_path)
            if metadata.st_size != member.size_bytes:
                raise ValueError("size changed")
        except (OSError, ValueError):
            raise AdmissionRejectedError(
                "requested artifact has expired or is unavailable; use a retained asset version"
            ) from None


def _stamp(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()


async def snapshot(database, *, max_rows=None):
    def bounded(query):
        return query.limit(max_rows + 1) if max_rows is not None else query

    async with database.session() as session:
        jobs = (
            (
                await session.execute(
                    bounded(
                        select(
                            JobRow.id,
                            JobRow.state,
                            JobRow.spec,
                            JobRow.owner_username,
                            JobRow.updated_at,
                        )
                    )
                )
            )
            .mappings()
            .all()
        )
        attempts = (
            (
                await session.execute(
                    bounded(
                        select(
                            AttemptRow.id,
                            AttemptRow.job_id,
                            AttemptRow.state,
                            AttemptRow.completed_at,
                            AttemptRow.updated_at,
                            AttemptRow.checkpoint_ref,
                            AttemptRow.restore_checkpoint_ref,
                        )
                    )
                )
            )
            .mappings()
            .all()
        )
        artifacts = (
            (
                await session.execute(
                    bounded(
                        select(
                            ArtifactRow.attempt_id,
                            ArtifactRow.logical_path,
                            ArtifactRow.size_bytes,
                            ArtifactRow.sha256,
                        )
                    )
                )
            )
            .mappings()
            .all()
        )
        leases = (
            (
                await session.execute(
                    bounded(
                        select(
                            LeaseRow.attempt_id,
                            LeaseRow.state,
                        )
                    )
                )
            )
            .mappings()
            .all()
        )
    if max_rows is not None and any(
        len(rows) > max_rows for rows in (jobs, attempts, artifacts, leases)
    ):
        raise ValueError("diagnostic snapshot budget exceeded")
    return [dict(v) for v in jobs], [dict(v) for v in attempts], artifacts, leases


def _identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _catalog_pins(root, check=lambda: None):
    from topodrome.storage.download_pins import download_protected_paths

    protected = download_protected_paths(root, check)
    catalog = root / "artifacts" / "catalog" / "v1"
    ancestor = catalog
    while ancestor != root:
        if ancestor.is_symlink():
            raise ValueError("symlink in asset catalog")
        ancestor = ancestor.parent
    if not catalog.exists():
        return protected

    def failed_scan(error):
        raise error

    # Any ambiguous catalog entry aborts the entire collection (fail closed).
    for directory, dirs, files in os.walk(catalog, followlinks=False, onerror=failed_scan):
        check()
        for name in dirs:
            check()
            if (Path(directory) / name).is_symlink():
                raise ValueError("symlink in asset catalog")
        for name in files:
            check()
            if name != "manifest.json":
                continue
            path = Path(directory) / name
            regular_file(root, path.relative_to(root).as_posix())
            manifest, _ = _read_manifest(path)
            protected.update(member.logical_path for member in manifest.members)
    return protected


def _safe_directory(root, logical):
    path = root
    try:
        for part in Path(normalize_logical_path(logical)).parts:
            path = path / part
            if not stat.S_ISDIR(path.lstat().st_mode):
                return False
    except OSError:
        return False
    return True


def plan(root, policy, data, *, now=None, check=lambda: None):
    root = Path(root).resolve()
    now = datetime.now(UTC).timestamp() if now is None else now
    jobs, attempts, artifacts, leases = data
    jobs = {j["id"]: j for j in jobs}
    protected = _catalog_pins(root, check)
    busy = {v["attempt_id"] for v in leases if str(v["state"]) in LIVE_LEASES}
    # Active consumer pins protect even expired producer artifacts. Preserve
    # terminal consumer inputs for their artifact-retention window as well.
    for job in jobs.values():
        check()
        retention = job["spec"].get("artifact_retention_seconds") or policy.artifact_seconds
        if str(job["state"]) not in TERMINAL_JOBS or now - _stamp(job["updated_at"]) < retention:
            for asset in job["spec"].get("assets", {}).values():
                if not isinstance(asset, dict) or not asset.get("members"):
                    raise ValueError("unrecognized persisted asset pin; collection deferred")
                protected.update(
                    normalize_logical_path(m["logical_path"]) for m in asset["members"]
                )
    for attempt in attempts:
        check()
        job = jobs.get(attempt["job_id"])
        if job is None:
            continue
        if (
            str(job["state"]) not in TERMINAL_JOBS
            or str(attempt["state"]) not in TERMINAL_ATTEMPTS
            or attempt["id"] in busy
            or (
                str(job["state"]) != "succeeded"
                and now - _stamp(job["updated_at"]) < policy.failed_checkpoint_seconds
            )
        ):
            for key in ("checkpoint_ref", "restore_checkpoint_ref"):
                ref = attempt[key]
                if ref and not ref.startswith("region://"):
                    protected.add(ref.split("#sha256=", 1)[0])
    registered = {}
    for item in artifacts:
        check()
        registered.setdefault(item["attempt_id"], []).append(item)
    candidates = []

    def add(logical, reason, job, attempt, expected_size=None):
        check()
        if logical in protected:
            return
        try:
            _, meta = regular_file(root, logical)
        except (OSError, ValueError):
            return
        if now - meta.st_mtime < policy.terminal_grace_seconds:
            return
        if expected_size is not None and expected_size != meta.st_size:
            return
        candidates.append(
            {
                "path": logical,
                "reason": reason,
                "job_id": job["id"],
                "attempt_id": attempt["id"],
                "owner": job["owner_username"],
                "bytes": meta.st_size,
                "allocated_bytes": meta.st_blocks * 512,
                "identity": _identity(meta),
            }
        )

    for attempt in attempts:
        check()
        job = jobs.get(attempt["job_id"])
        if (
            job is None
            or str(job["state"]) not in TERMINAL_JOBS
            or str(attempt["state"]) not in TERMINAL_ATTEMPTS
            or attempt["id"] in busy
        ):
            continue
        completed = _stamp(attempt["completed_at"])
        if completed is None:
            completed = _stamp(attempt["updated_at"])
        ended = max(completed, _stamp(job["updated_at"]))
        age = now - ended
        if age < policy.terminal_grace_seconds:
            continue
        prefix = (
            f"jobs/{validate_component(job['id'])}/attempts/{validate_component(attempt['id'])}"
        )
        ckpt_dir = root / prefix / "checkpoints"
        # Enumerate only exact writer-generated checkpoint filenames.
        checkpoints = []
        if _safe_directory(root, f"{prefix}/checkpoints"):
            for path in ckpt_dir.glob("*.pkl"):
                check()
                if CHECKPOINT.fullmatch(path.name):
                    checkpoints.append(path.name)
        checkpoints.sort()
        keep = set()
        if str(job["state"]) != "succeeded" and age < policy.failed_checkpoint_seconds:
            keep.update(checkpoints[-1:])
        for name in checkpoints:
            if name not in keep:
                add(f"{prefix}/checkpoints/{name}", "recovery_checkpoint", job, attempt)
        retention = job["spec"].get("artifact_retention_seconds") or policy.artifact_seconds
        if age >= retention:
            for artifact in registered.get(attempt["id"], ()):
                logical = normalize_logical_path(artifact["logical_path"])
                add(
                    f"{prefix}/committed/{logical}",
                    "expired_artifact",
                    job,
                    attempt,
                    artifact["size_bytes"],
                )
            # Unregistered committed files, logs and metadata are not candidates.
            stage = root / prefix / "staging"
            if not _safe_directory(root, f"{prefix}/staging"):
                continue
            for directory, dirs, files in os.walk(stage, followlinks=False):
                check()
                dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
                for name in files:
                    add(
                        (Path(directory) / name).relative_to(root).as_posix(),
                        "expired_staging",
                        job,
                        attempt,
                    )
    # Largest first reclaims meaningful capacity within each bounded pass.
    candidates.sort(key=lambda v: (-v["allocated_bytes"], v["path"]))
    return candidates


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def collect(root, policy, candidates, *, apply=False):
    root = Path(root).resolve()
    totals = Counter()
    for item in candidates:
        totals[item["reason"]] += item["allocated_bytes"]
    report = {
        "dry_run": not apply,
        "candidate_files": len(candidates),
        "candidate_allocated_bytes": sum(totals.values()),
        "bytes_by_reason": dict(totals),
        "deleted_files": 0,
        "unlinked_allocated_bytes": 0,
        "skipped_changed_files": 0,
        "sample": candidates[:20],
        "policy": policy.model_dump(),
    }
    if not apply or not candidates:
        return report
    selected = []
    total = 0
    for item in candidates:
        if len(selected) >= policy.max_files_per_pass:
            break
        if selected and total + item["allocated_bytes"] > policy.max_bytes_per_pass:
            continue
        selected.append(item)
        total += item["allocated_bytes"]
    audit_root = root / ".topodrome-retention"
    if audit_root.is_symlink():
        raise ValueError("unsafe retention journal")
    audit_root.mkdir(mode=0o700, exist_ok=True)
    audit = audit_root / f"{uuid4().hex}.json"
    journal = {"state": "intent", "created_at": datetime.now(UTC).isoformat(), "files": selected}
    _atomic_json(audit, journal)  # Durable exact targets before the first unlink.
    for item in selected:
        try:
            path, meta = regular_file(root, item["path"])
        except (OSError, ValueError):
            report["skipped_changed_files"] += 1
            continue
        if _identity(meta) != tuple(item["identity"]):
            report["skipped_changed_files"] += 1
            continue
        path.unlink()
        report["deleted_files"] += 1
        # Do not claim physically freed blocks while another hardlink exists.
        if meta.st_nlink == 1:
            report["unlinked_allocated_bytes"] += meta.st_blocks * 512
    journal.update(state="completed", report=report)
    _atomic_json(audit, journal)
    report["audit_path"] = str(audit)
    return report


def plan_blob_storage(root, policy, data, index, *, now=None, check=lambda: None):
    """Shared eligibility planner; callers supply a read-only or GC transaction."""
    from topodrome.storage.blobs import blob_path

    now = datetime.now(UTC).timestamp() if now is None else now
    jobs, attempts, artifacts, leases = data
    by_job = {j["id"]: j for j in jobs}
    by_attempt = {a["id"]: a for a in attempts}
    busy = {r["attempt_id"] for r in leases if str(r["state"]) in LIVE_LEASES}
    protected = _catalog_pins(root, check)

    def retained(job, attempt=None):
        if job is None:
            return True
        retention = job["spec"].get("artifact_retention_seconds") or policy.artifact_seconds
        if str(job["state"]) not in TERMINAL_JOBS:
            return True
        ended = _stamp(job["updated_at"])
        if attempt is not None:
            if attempt["id"] in busy or str(attempt["state"]) not in TERMINAL_ATTEMPTS:
                return True
            ended = max(ended, _stamp(attempt["completed_at"] or attempt["updated_at"]))
        return now - ended < max(retention, policy.terminal_grace_seconds)

    for job in jobs:
        check()
        if retained(job):
            for asset in job["spec"].get("assets", {}).values():
                if not isinstance(asset, dict) or not asset.get("members"):
                    raise ValueError("unrecognized persisted asset pin")
                protected.update(m["logical_path"] for m in asset["members"])
    for item in artifacts:
        check()
        attempt = by_attempt.get(item["attempt_id"])
        if attempt is None or retained(by_job.get(attempt["job_id"]), attempt):
            protected.add(blob_path(item["sha256"]))
    for job_id, attempt_id, digest in index.execute("SELECT * FROM claims"):
        check()
        if retained(by_job.get(job_id), by_attempt.get(attempt_id)):
            protected.add(blob_path(digest))

    candidates = plan(root, policy, data, now=now, check=check)
    for digest, size, touched in index.execute("SELECT * FROM objects"):
        check()
        logical = blob_path(digest)
        if logical in protected or now - touched < policy.terminal_grace_seconds:
            continue
        try:
            _, meta = regular_file(root, logical)
        except FileNotFoundError:
            continue
        if meta.st_size != size or now - meta.st_mtime < policy.terminal_grace_seconds:
            continue
        candidates.append(
            {
                "path": logical,
                "reason": "unreferenced_blob",
                "job_id": None,
                "attempt_id": None,
                "owner": "@blob-store",
                "bytes": size,
                "allocated_bytes": meta.st_blocks * 512,
                "identity": _identity(meta),
            }
        )
    candidates.sort(key=lambda v: (-v["allocated_bytes"], v["path"]))
    return candidates


def collect_blob_storage(root, policy, data, *, apply=False):
    """Index-driven CAS collection, serialized with every blob claim/publication."""
    from topodrome.storage.blobs import blob_transaction

    root = Path(root).resolve()
    with blob_transaction(root) as index:
        candidates = plan_blob_storage(root, policy, data, index)
        report = collect(root, policy, candidates, apply=apply)
        if apply:
            for item in candidates:
                if item["reason"] != "unreferenced_blob":
                    continue
                try:
                    regular_file(root, item["path"])
                except FileNotFoundError:
                    digest = item["path"].rsplit("/", 1)[-1]
                    index.execute("DELETE FROM claims WHERE sha256=?", (digest,))
                    index.execute("DELETE FROM objects WHERE sha256=?", (digest,))
        return report
