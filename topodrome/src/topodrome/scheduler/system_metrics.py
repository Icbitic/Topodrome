"""Cached, advisory management telemetry. Never deletes or grants GC authority."""

from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import stat
import threading
import time
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from topodrome.db.session import CURRENT_SCHEMA_REVISION
from topodrome.scheduler.storage_retention import plan, plan_blob_storage, regular_file, snapshot


class ScanBudget:
    def __init__(self, stop=None, seconds=20, entries=250_000):
        self.stop = stop or threading.Event()
        self.deadline = time.monotonic() + seconds
        self.remaining = entries

    def __call__(self):
        self.remaining -= 1
        if self.stop.is_set() or time.monotonic() > self.deadline or self.remaining < 0:
            raise TimeoutError("storage metrics scan budget exceeded")


def disk_usage(root, check):
    """Allocated regular-file bytes, unique inodes, no symlinks or submounts."""
    root = Path(root)
    meta = root.lstat()
    if not stat.S_ISDIR(meta.st_mode):
        raise ValueError("unsafe storage root")
    categories = Counter()
    seen = set()
    logical = 0
    skipped = 0

    def failed(error):
        raise error

    for directory, dirs, files, fd in os.fwalk(root, follow_symlinks=False, onerror=failed):
        check()
        for name in list(dirs):
            check()
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or info.st_dev != meta.st_dev:
                dirs.remove(name)
                skipped += 1
        dirs.sort()
        for name in sorted(files):
            check()
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                continue  # Live files may disappear while sampling.
            if not stat.S_ISREG(info.st_mode) or info.st_dev != meta.st_dev:
                skipped += 1
                continue
            identity = info.st_dev, info.st_ino
            if identity in seen:
                continue
            seen.add(identity)
            parts = (Path(directory) / name).relative_to(root).parts
            category = {
                "blobs": "Immutable blobs",
                "sources": "Source bundles",
                "artifacts": "Asset metadata / uploads",
                ".blob-index": "Blob index",
                ".topodrome-retention": "Retention journal",
            }.get(parts[0], "Other central files")
            if parts[0] == "jobs":
                category = (
                    "Checkpoints"
                    if "checkpoints" in parts
                    else "Staged outputs"
                    if "staging" in parts
                    else "Logs"
                    if name.endswith(".log")
                    else "Job outputs / metadata"
                )
            categories[category] += info.st_blocks * 512
            logical += info.st_size
    fs = shutil.disk_usage(root)
    return {
        "allocated_bytes": sum(categories.values()),
        "logical_bytes": logical,
        "files": len(seen),
        "categories": dict(categories),
        "skipped_entries": skipped,
        "filesystem_total_bytes": fs.total,
        "filesystem_free_bytes": fs.free,
    }


class IndexSnapshot:
    def __init__(self, root, check):
        path, _ = regular_file(root, ".blob-index/index.sqlite3")
        self.tables = {}
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.2)) as db:
            try:
                db.execute("PRAGMA query_only=ON")
                db.execute("BEGIN")
                for query in ("SELECT * FROM claims", "SELECT * FROM objects"):
                    rows = []
                    for row in db.execute(query):
                        check()
                        rows.append(row)
                    self.tables[query] = rows
            finally:
                db.rollback()

    def execute(self, query):
        return self.tables[query]


def reclaimable(root, candidates, check):
    """Do not count shared inodes unless every hardlink is eligible."""
    groups = {}
    for item in candidates:
        check()
        try:
            _, meta = regular_file(root, item["path"])
        except FileNotFoundError:
            continue
        group = groups.setdefault((meta.st_dev, meta.st_ino), [meta, set()])
        group[1].add(item["path"])
    return sum(
        meta.st_blocks * 512 for meta, paths in groups.values() if len(paths) == meta.st_nlink
    )


def sample_storage(root, layout, policy, data, stop=None):
    check = ScanBudget(stop)
    root = Path(root).absolute()
    now = datetime.now(UTC).timestamp()
    report = disk_usage(root, check)
    try:
        index = IndexSnapshot(root, check) if layout == "blobs-v1" else None

        def candidates(when):
            if index is not None:
                return plan_blob_storage(root, policy, data, index, now=when, check=check)
            return plan(root, policy, data, now=when, check=check)

        ready = candidates(now)
        tomorrow = candidates(now + 86400)
        eligible_now = reclaimable(root, ready, check)
        eligible_day = reclaimable(root, tomorrow, check)
        report.update(
            eligible_now_bytes=eligible_now,
            eligible_24h_bytes=eligible_day,
            newly_eligible_24h_bytes=max(0, eligible_day - eligible_now),
            eligible_24h_files=len(tomorrow),
            estimate_state="ready",
        )
    except (OSError, ValueError, sqlite3.Error, TimeoutError, KeyError):
        report["estimate_state"] = "unavailable"
    report["sampled_at"] = datetime.now(UTC).isoformat()
    return report


class SystemMetrics:
    """One bounded scanner per service, independent of dispatcher/reference locks."""

    interval_seconds = 300

    def __init__(self, service):
        self.service = service
        self.started = time.monotonic()
        self.last_attempt = None
        self.last_error = None
        self.storage = None
        self.task = None
        self.stop = threading.Event()

    async def refresh(self):
        self.last_attempt = time.monotonic()
        try:
            async with asyncio.timeout(15):
                data = await snapshot(self.service.database, max_rows=50_000)
            self.storage = await asyncio.to_thread(
                sample_storage,
                self.service.settings.global_root,
                self.service.settings.storage_layout,
                self.service.settings.storage_retention,
                data,
                self.stop,
            )
            self.last_error = None
        except Exception as exc:
            self.last_error = type(exc).__name__  # No paths, credentials, or DB exception text.

    async def get(self):
        if (
            not self.stop.is_set()
            and (self.task is None or self.task.done())
            and (
                self.last_attempt is None
                or time.monotonic() - self.last_attempt >= self.interval_seconds
            )
        ):
            self.task = asyncio.create_task(self.refresh(), name="topodrome-storage-metrics")
        settings = self.service.settings
        try:
            async with asyncio.timeout(3):
                ready = await self.service.database.ready(
                    expected_schema_revision=CURRENT_SCHEMA_REVISION
                )
        except TimeoutError:
            ready = False
        try:
            load = list(os.getloadavg())
            statm = await asyncio.to_thread(Path("/proc/self/statm").read_text)
            rss = int(statm.split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            load, rss = None, None
        return {
            "sampled_at": datetime.now(UTC).isoformat(),
            "uptime_seconds": time.monotonic() - self.started,
            "database_ready": ready,
            "schema_revision": CURRENT_SCHEMA_REVISION,
            "scheduler_rss_bytes": rss,
            "host_load_average": load,
            "connected_agents": len(await self.service.connections.identities()),
            "policy": await self.service.policy_status(),
            "upgrades": [r.summary() for r in self.service.upgrades.store.read().records[-10:]],
            "retention": settings.storage_retention.model_dump(),
            "storage_layout": settings.storage_layout,
            "storage": self.storage,
            "storage_state": "stale"
            if self.last_error and self.storage
            else "error"
            if self.last_error
            else "refreshing"
            if self.task and not self.task.done()
            else "ready",
            "storage_error": self.last_error,
            "storage_refresh_seconds": self.interval_seconds,
        }

    async def close(self):
        self.stop.set()
        if self.task and not self.task.done():
            await self.task
