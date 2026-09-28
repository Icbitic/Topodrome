from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from topodrome.db.session import Database


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
            await input_staging.receive_stage(session, site_id, {
                **command.payload, "state": "ready",
                "progress": {"current": 0, "stage": "inputs-ready"},
            })
            return True, None
        return available, command

    monkeypatch.setattr(input_staging, "prepare_inputs", ready)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'scheduler.sqlite3'}")
    await database.create_schema()
    yield database
    await database.close()
