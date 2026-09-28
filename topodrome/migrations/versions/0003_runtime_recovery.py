"""Persist retry deadlines and checkpoint lineage.

Revision ID: 0003_runtime_recovery
Revises: 0002_kernel_runtime_links

The initial migration historically imported live ORM metadata. The guards are
retained for databases created during that period that already have these
columns; the now-frozen initial revision does not.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_runtime_recovery"
down_revision = "0002_kernel_runtime_links"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    return {str(column["name"]) for column in sa.inspect(op.get_bind()).get_columns(table)}


def _indexes(table: str) -> set[str]:
    return {
        str(index["name"])
        for index in sa.inspect(op.get_bind()).get_indexes(table)
        if index.get("name") is not None
    }


def upgrade() -> None:
    if "retry_not_before" not in _columns("jobs"):
        op.add_column(
            "jobs",
            sa.Column("retry_not_before", sa.DateTime(timezone=True), nullable=True),
        )
    if "ix_jobs_retry_not_before" not in _indexes("jobs"):
        op.create_index("ix_jobs_retry_not_before", "jobs", ["retry_not_before"])
    if "restore_checkpoint_ref" not in _columns("attempts"):
        op.add_column(
            "attempts",
            sa.Column("restore_checkpoint_ref", sa.String(length=1000), nullable=True),
        )


def downgrade() -> None:
    if "restore_checkpoint_ref" in _columns("attempts"):
        op.drop_column("attempts", "restore_checkpoint_ref")
    if "ix_jobs_retry_not_before" in _indexes("jobs"):
        op.drop_index("ix_jobs_retry_not_before", table_name="jobs")
    if "retry_not_before" in _columns("jobs"):
        op.drop_column("jobs", "retry_not_before")
