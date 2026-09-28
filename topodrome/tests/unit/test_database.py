from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from topodrome.db.session import CURRENT_SCHEMA_REVISION, Database

pytestmark = pytest.mark.unit


async def test_schema_aware_readiness_requires_exact_alembic_head(tmp_path: Path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'readiness.sqlite3'}")
    try:
        assert await database.ready()
        assert not await database.ready(expected_schema_revision=CURRENT_SCHEMA_REVISION)

        async with database.engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE TABLE alembic_version "
                    "(version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
                )
            )
            await connection.execute(
                text("INSERT INTO alembic_version(version_num) VALUES ('0008_model_replicas')")
            )
        assert not await database.ready(expected_schema_revision=CURRENT_SCHEMA_REVISION)

        async with database.engine.begin() as connection:
            await connection.execute(
                text("UPDATE alembic_version SET version_num = :revision"),
                {"revision": CURRENT_SCHEMA_REVISION},
            )
        assert await database.ready(expected_schema_revision=CURRENT_SCHEMA_REVISION)

        async with database.engine.begin() as connection:
            await connection.execute(
                text("INSERT INTO alembic_version(version_num) VALUES ('unexpected_branch')")
            )
        assert not await database.ready(expected_schema_revision=CURRENT_SCHEMA_REVISION)
    finally:
        await database.close()
