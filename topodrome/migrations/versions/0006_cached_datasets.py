"""Persist site-reported verified dataset locality.

Revision ID: 0006_cached_datasets
Revises: 0005_shutdown_generations
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_cached_datasets"
down_revision = "0005_shutdown_generations"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table not in set(inspector.get_table_names()):
        return set()
    return {str(column["name"]) for column in inspector.get_columns(table)}


def upgrade() -> None:
    columns = _columns("sites")
    if columns and "cached_datasets" not in columns:
        op.add_column(
            "sites",
            sa.Column(
                "cached_datasets",
                sa.JSON(),
                nullable=False,
                server_default="[]",
            ),
        )


def downgrade() -> None:
    if "cached_datasets" in _columns("sites"):
        op.drop_column("sites", "cached_datasets")
