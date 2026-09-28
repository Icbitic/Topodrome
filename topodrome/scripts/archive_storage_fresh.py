"""Offline archive/reset. Never starts services; a failure leaves maintenance closed.

Run only after stopping scheduler and site agents. The default is a read-only
inventory. --apply requires an unused archive path outside the live root.
"""
# This offline operator utility deliberately runs fixed argv synchronously.
# ruff: noqa: ASYNC221, ASYNC240, S603

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import yaml
from sqlalchemy import delete, func, select
from sqlalchemy.engine import make_url

from topodrome.db.models import (
    ArtifactRow,
    GPUReservationRow,
    JobRow,
    KernelSessionRow,
    LeaseRow,
    SiteRow,
)
from topodrome.db.session import Database
from topodrome.scheduler.storage_retention import _atomic_json

DATA_DIRECTORIES = (
    "jobs",
    "artifacts",
    "projects",
    ".topodrome-live-logs",
    ".topodrome-retention",
    ".blob-index",
    "blobs",
)


async def inventory(db):
    async with db.session() as session:
        for model, terminal in (
            (JobRow, {"succeeded", "failed", "canceled"}),
            (KernelSessionRow, {"stopped", "lost"}),
            (LeaseRow, {"released", "expired", "lost", "rejected"}),
        ):
            states = (await session.execute(select(model.state))).scalars().all()
            if any(str(v) not in terminal for v in states):
                raise RuntimeError(f"nonterminal {model.__tablename__}; archive refused")
        result = {}
        for model in (JobRow, ArtifactRow, KernelSessionRow, SiteRow, GPUReservationRow):
            result[model.__tablename__] = await session.scalar(
                select(func.count()).select_from(model)
            )
        return result


def archive_directories(root, archive):
    """Journal and move exact owned trees; no recursive deletion or byte copies."""
    if archive == root or root in archive.parents or archive in root.parents:
        raise ValueError("archive must be outside the live root")
    entries = []
    for name in DATA_DIRECTORIES:
        source = root / name
        if source.is_symlink():
            raise ValueError("refusing symlinked storage tree")
        if source.exists():
            if not source.is_dir():
                raise ValueError("storage entry is not a directory")
            entries.append(name)
    archive.mkdir(mode=0o700, parents=False, exist_ok=False)
    fd = os.open(archive.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if archive.stat().st_dev != root.stat().st_dev:
        raise ValueError("archive must be on the same filesystem for atomic renames")
    _atomic_json(archive / "files-journal.json", {"state": "intent", "entries": entries})
    for name in entries:
        (root / name).rename(archive / name)
        _atomic_json(
            archive / "files-journal.json",
            {
                "state": "moving",
                "entries": entries,
                "moved": [n for n in entries if (archive / n).exists()],
            },
        )
    _atomic_json(archive / "files-journal.json", {"state": "moved", "entries": entries})
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


async def main(args):
    config = yaml.safe_load(args.config.read_text())
    root = Path(config["global_root"]).resolve()
    archive = args.archive.absolute()
    db = Database(config["database_dsn"])
    try:
        counts = await inventory(db)
        print(
            json.dumps(
                {"counts": counts, "root": str(root), "archive": str(archive), "apply": args.apply}
            ),
            flush=True,
        )
        if not args.apply:
            return
        for service in ("topodrome-scheduler.service", "topodrome-agent-lab1.service"):
            result = subprocess.run(
                ["/usr/bin/systemctl", "--user", "is-active", service],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip() not in {"inactive", "failed"}:
                raise RuntimeError(f"stop {service} before archiving")
        if archive.exists() or archive.is_symlink():
            raise ValueError("archive destination already exists")
        # Prepare the backup beside the archive, not in a directory that moves.
        backup_dir = archive.with_name(archive.name + "-metadata")
        backup_dir.mkdir(mode=0o700, exist_ok=False)
        url = make_url(config["database_dsn"])
        env = {
            **os.environ,
            "PGHOST": str(url.query.get("host") or url.host or "localhost"),
            "PGPORT": str(url.query.get("port") or url.port or 5432),
            "PGUSER": url.username or "",
            "PGDATABASE": url.database or "",
        }
        if url.password:
            env["PGPASSWORD"] = url.password
        dump = backup_dir / "database.dump"
        subprocess.run(
            [str(args.pg_bin / "pg_dump"), "-Fc", "-f", str(dump)], env=env, check=True, timeout=120
        )
        dump.chmod(0o600)
        fd = os.open(dump, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        verify_name = "topodrome_archive_verify_" + uuid4().hex[:12]
        subprocess.run(
            [str(args.pg_bin / "createdb"), verify_name], env=env, check=True, timeout=30
        )
        try:
            subprocess.run(
                [
                    str(args.pg_bin / "pg_restore"),
                    "--exit-on-error",
                    "--no-owner",
                    "--dbname",
                    verify_name,
                    str(dump),
                ],
                env=env,
                check=True,
                timeout=120,
            )
            verification = Database(
                url.set(database=verify_name).render_as_string(hide_password=False)
            )
            try:
                if await inventory(verification) != counts:
                    raise RuntimeError("restored backup inventory differs")
            finally:
                await verification.close()
        finally:
            subprocess.run(
                [str(args.pg_bin / "dropdb"), verify_name], env=env, check=True, timeout=30
            )
        print("Database backup restored and verified in a disposable database", flush=True)
        _atomic_json(
            backup_dir / "archive-state.json",
            {
                "state": "backup_verified",
                "counts": counts,
                "live_root": str(root),
                "archive": str(archive),
            },
        )
        await inventory(db)
        archive_directories(root, archive)
        async with db.transaction() as session:
            await session.execute(delete(KernelSessionRow))
            await session.execute(delete(JobRow))
        after = await inventory(db)
        assert after["jobs"] == after["artifacts"] == after["kernel_sessions"] == 0
        assert after["sites"] == counts["sites"]
        assert after["gpu_reservations"] == counts["gpu_reservations"]
        for name in ("jobs", "artifacts", "projects", "blobs"):
            (root / name).mkdir(mode=0o700, exist_ok=True)
        _atomic_json(root / "storage-layout.json", {"layout": "blobs-v1", "archive": str(archive)})
        shutil.copy2(args.config, backup_dir / "scheduler-before.yaml")
        _atomic_json(
            backup_dir / "archive-state.json",
            {
                "state": "complete",
                "before": counts,
                "after": after,
                "live_root": str(root),
                "archive": str(archive),
            },
        )
        print(
            json.dumps(
                {"archived": str(archive), "metadata": str(backup_dir), "active_counts": after}
            ),
            flush=True,
        )
    finally:
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pg-bin", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    os.umask(0o077)
    asyncio.run(main(parser.parse_args()))
