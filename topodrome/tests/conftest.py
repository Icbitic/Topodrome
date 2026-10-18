from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from topodrome.db.session import Database


@pytest.fixture
def ray_alias_tmpfs(monkeypatch, tmp_path):
    """Keep tiny AF_UNIX aliases independent of a busy/full system root disk."""
    from topodrome.site_agent.ray_cluster import RayClusterConfig

    short_root = Path("/dev/shm")  # noqa: S108 -- aliases remain exact-owner validated
    uid = os.geteuid()
    original = RayClusterConfig.__post_init__

    def configure(self):
        if self.short_temp_root == Path("/tmp"):  # noqa: S108 -- redirect test aliases only
            object.__setattr__(self, "short_temp_root", short_root)
        original(self)

    monkeypatch.setattr(RayClusterConfig, "__post_init__", configure)
    yield
    private = short_root / f".tdr-{uid}"
    if private.is_dir() and not private.is_symlink():
        for alias in private.iterdir():
            if alias.is_symlink() and alias.lstat().st_uid == uid:
                target = alias.readlink()
                if target.is_absolute() and target.is_relative_to(tmp_path):
                    alias.unlink()


@pytest.fixture
def pre_staged_inputs(monkeypatch):
    """Policy-only tests model a worker whose immutable inputs are already ready.

    The real two-phase protocol is exercised in test_lifecycle_reliability;
    these tests intentionally don't start a storage worker/control event loop.
    """
    from topodrome.scheduler import input_staging

    original = input_staging.prepare_inputs

    async def ready(session, job, site_id, **kwargs):
        available, command = await original(session, job, site_id, **kwargs)
        if command is not None:
            await input_staging.receive_stage(
                session,
                site_id,
                {
                    **command.payload,
                    "state": "ready",
                    "progress": {"current": 0, "stage": "inputs-ready"},
                },
            )
            return True, None
        return available, command

    monkeypatch.setattr(input_staging, "prepare_inputs", ready)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'scheduler.sqlite3'}")
    await database.create_schema()
    yield database
    await database.close()
