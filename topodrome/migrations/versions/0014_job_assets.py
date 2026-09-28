"""Normalize persisted job specifications to the global-asset schema.

Revision ID: 0014_job_assets
Revises: 0013_global_assets

The legacy dataset field was a sequence and therefore did not carry the input
aliases required by the global-asset mapping.  Empty legacy values have an
unambiguous representation; non-empty values deliberately fail the migration
instead of inventing aliases or silently dropping an input.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "0014_job_assets"
down_revision = "0013_global_assets"
branch_labels = None
depends_on = None


def _jobs_table() -> sa.TableClause:
    return sa.table(
        "jobs",
        sa.column("id", sa.String(length=80)),
        sa.column("spec", sa.JSON()),
    )


def _upgraded_specs(bind: Any, jobs: sa.TableClause) -> list[tuple[str, dict[str, Any]]]:
    rewrites: list[tuple[str, dict[str, Any]]] = []
    rows = list(bind.execute(sa.select(jobs.c.id, jobs.c.spec)).mappings())
    for row in rows:
        raw_spec = row["spec"]
        if not isinstance(raw_spec, dict):
            raise RuntimeError("job asset migration requires every spec to be a JSON object")
        spec = dict(raw_spec)
        if "datasets" in spec:
            legacy = spec.pop("datasets")
            if not isinstance(legacy, (list, dict)) or legacy:
                raise RuntimeError("job asset migration cannot convert non-empty legacy datasets")
        spec.setdefault("assets", {})
        if spec != raw_spec:
            rewrites.append((str(row["id"]), spec))
    return rewrites


def _downgraded_specs(
    bind: Any,
    jobs: sa.TableClause,
) -> list[tuple[str, dict[str, Any]]]:
    rewrites: list[tuple[str, dict[str, Any]]] = []
    rows = list(bind.execute(sa.select(jobs.c.id, jobs.c.spec)).mappings())
    for row in rows:
        raw_spec = row["spec"]
        if not isinstance(raw_spec, dict):
            raise RuntimeError("job asset downgrade requires every spec to be a JSON object")
        spec = dict(raw_spec)
        assets = spec.pop("assets", {})
        if not isinstance(assets, dict) or assets:
            raise RuntimeError("job asset downgrade cannot convert non-empty assets")
        if "datasets" in spec:
            raise RuntimeError("job asset downgrade found an unexpected legacy datasets field")
        spec["datasets"] = []
        if spec != raw_spec:
            rewrites.append((str(row["id"]), spec))
    return rewrites


def _write_specs(
    bind: Any,
    jobs: sa.TableClause,
    rewrites: list[tuple[str, dict[str, Any]]],
) -> None:
    for job_id, spec in rewrites:
        bind.execute(sa.update(jobs).where(jobs.c.id == job_id).values(spec=spec))


def upgrade() -> None:
    bind = op.get_bind()
    jobs = _jobs_table()
    # Validate every row before the first write.  This keeps the migration
    # fail-closed even on databases whose DDL transaction behavior is weaker
    # than PostgreSQL's.
    _write_specs(bind, jobs, _upgraded_specs(bind, jobs))


def downgrade() -> None:
    bind = op.get_bind()
    jobs = _jobs_table()
    _write_specs(bind, jobs, _downgraded_specs(bind, jobs))
