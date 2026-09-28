"""Persist validated foreign GPU reservation state.

Revision ID: 0010_gpu_reservations
Revises: 0009_schema_parity
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010_gpu_reservations"
down_revision = "0009_schema_parity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "gpus",
        sa.Column(
            "externally_reserved",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("gpus", "externally_reserved")
