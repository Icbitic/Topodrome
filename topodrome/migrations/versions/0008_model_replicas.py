"""Persist warm model-replica identity and ordered GPU membership.

Revision ID: 0008_model_replicas
Revises: 0007_runtime_resources
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_model_replicas"
down_revision = "0007_runtime_resources"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table not in set(inspector.get_table_names()):
        return set()
    return {str(column["name"]) for column in inspector.get_columns(table)}


def upgrade() -> None:
    columns = _columns("model_capabilities")
    if columns and "replica_id" not in columns:
        op.add_column(
            "model_capabilities",
            sa.Column(
                "replica_id",
                sa.String(length=160),
                nullable=False,
                server_default="",
            ),
        )
    if columns and "replica_gpu_uuids" not in columns:
        op.add_column(
            "model_capabilities",
            sa.Column(
                "replica_gpu_uuids",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'[]'"),
            ),
        )


def downgrade() -> None:
    columns = _columns("model_capabilities")
    if "replica_gpu_uuids" in columns:
        op.drop_column("model_capabilities", "replica_gpu_uuids")
    if "replica_id" in columns:
        op.drop_column("model_capabilities", "replica_id")
