"""Persist job and kernel ownership for whitelisted API users.

Revision ID: 0012_multi_user_ownership
Revises: 0011_timed_gpu_reservations
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012_multi_user_ownership"
down_revision = "0011_timed_gpu_reservations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Every resource created before multi-user access belonged to the sole
    # operator. Keeping a server default also makes rolling deployments safe
    # while an older scheduler process is still draining.
    op.add_column(
        "jobs",
        sa.Column(
            "owner_username",
            sa.String(length=64),
            nullable=False,
            server_default="@operator",
        ),
    )
    op.create_index("ix_jobs_owner_username", "jobs", ["owner_username"])
    op.add_column(
        "kernel_sessions",
        sa.Column(
            "owner_username",
            sa.String(length=64),
            nullable=False,
            server_default="@operator",
        ),
    )
    op.create_index(
        "ix_kernel_sessions_owner_username",
        "kernel_sessions",
        ["owner_username"],
    )


def downgrade() -> None:
    op.drop_index("ix_kernel_sessions_owner_username", table_name="kernel_sessions")
    op.drop_column("kernel_sessions", "owner_username")
    op.drop_index("ix_jobs_owner_username", table_name="jobs")
    op.drop_column("jobs", "owner_username")
