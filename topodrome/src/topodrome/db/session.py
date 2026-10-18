from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from topodrome.db.models import Base

# Update this alongside every new Alembic head. Production readiness uses the
# revision as a deployment contract instead of accepting a merely reachable,
# empty, or stale database.
CURRENT_SCHEMA_REVISION = "0020_repository_models"


class Database:
    def __init__(self, dsn: str, *, echo: bool = False) -> None:
        kwargs: dict[str, object] = {"echo": echo, "pool_pre_ping": True}
        if dsn.startswith("sqlite+"):
            kwargs.pop("pool_pre_ping")
        self.engine: AsyncEngine = create_async_engine(dsn, **kwargs)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False, autoflush=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessions() as session:
            yield session

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        async with self.sessions.begin() as session:
            yield session

    async def ready(self, *, expected_schema_revision: str | None = None) -> bool:
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
                if expected_schema_revision is not None:
                    result = await session.execute(text("SELECT version_num FROM alembic_version"))
                    revisions = [str(value) for value in result.scalars().all()]
                    if revisions != [expected_schema_revision]:
                        return False
            return True
        except Exception:
            return False

    async def create_schema(self) -> None:
        """Test/development helper; production uses Alembic migrations."""
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def drop_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        await self.engine.dispose()
