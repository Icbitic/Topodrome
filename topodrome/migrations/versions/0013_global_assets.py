"""Rename site cache telemetry for generic immutable assets.

Revision ID: 0013_global_assets
Revises: 0012_multi_user_ownership
"""

from __future__ import annotations

from alembic import op

revision = "0013_global_assets"
down_revision = "0012_multi_user_ownership"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "sites",
        "cached_datasets",
        new_column_name="cached_assets",
    )


def downgrade() -> None:
    op.alter_column(
        "sites",
        "cached_assets",
        new_column_name="cached_datasets",
    )
