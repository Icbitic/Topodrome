"""Persist host-capacity telemetry and warm-runtime compatibility.

Revision ID: 0007_runtime_resources
Revises: 0006_cached_datasets
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_runtime_resources"
down_revision = "0006_cached_datasets"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table not in set(inspector.get_table_names()):
        return set()
    return {str(column["name"]) for column in inspector.get_columns(table)}


def upgrade() -> None:
    site_columns = _columns("sites")
    for name, column_type in (
        ("total_cpu", sa.Float()),
        ("available_cpu", sa.Float()),
        ("total_memory_bytes", sa.BigInteger()),
        ("available_memory_bytes", sa.BigInteger()),
    ):
        if site_columns and name not in site_columns:
            op.add_column("sites", sa.Column(name, column_type, nullable=True))

    capability_columns = _columns("model_capabilities")
    if capability_columns and "runtime_fingerprint" not in capability_columns:
        op.add_column(
            "model_capabilities",
            sa.Column(
                "runtime_fingerprint",
                sa.String(length=64),
                nullable=False,
                server_default="",
            ),
        )


def downgrade() -> None:
    if "runtime_fingerprint" in _columns("model_capabilities"):
        op.drop_column("model_capabilities", "runtime_fingerprint")
    site_columns = _columns("sites")
    for name in (
        "available_memory_bytes",
        "total_memory_bytes",
        "available_cpu",
        "total_cpu",
    ):
        if name in site_columns:
            op.drop_column("sites", name)
