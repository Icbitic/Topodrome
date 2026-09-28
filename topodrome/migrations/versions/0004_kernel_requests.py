"""Persist kernel access generations and request idempotency.

Revision ID: 0004_kernel_requests
Revises: 0003_runtime_recovery

The initial migration historically imported live ORM metadata. Guards remain
for databases created during that period that may already contain these
objects; the now-frozen initial revision does not.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_kernel_requests"
down_revision = "0003_runtime_recovery"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table not in set(inspector.get_table_names()):
        return set()
    return {str(column["name"]) for column in inspector.get_columns(table)}


def upgrade() -> None:
    if "access_generation" not in _columns("kernel_sessions"):
        op.add_column(
            "kernel_sessions",
            sa.Column("access_generation", sa.Integer(), nullable=False, server_default="0"),
        )
    if not _columns("kernel_channel_requests"):
        op.create_table(
            "kernel_channel_requests",
            sa.Column("id", sa.String(length=120), nullable=False),
            sa.Column("session_id", sa.String(length=80), nullable=False),
            sa.Column("request_id", sa.String(length=200), nullable=False),
            sa.Column("request_digest", sa.LargeBinary(length=32), nullable=False),
            sa.Column("state", sa.String(length=20), nullable=False),
            sa.Column("execution_count", sa.Integer(), nullable=False),
            sa.Column("agent_session_id", sa.String(length=120), nullable=True),
            sa.Column("result_digest", sa.LargeBinary(length=32), nullable=True),
            sa.Column("error_code", sa.String(length=80), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(
                ["session_id"], ["kernel_sessions.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "session_id", "request_id", name="uq_kernel_channel_request"
            ),
        )
        op.create_index(
            "ix_kernel_channel_requests_session_id",
            "kernel_channel_requests",
            ["session_id"],
        )
        op.create_index(
            "ix_kernel_channel_requests_state",
            "kernel_channel_requests",
            ["state"],
        )


def downgrade() -> None:
    if _columns("kernel_channel_requests"):
        op.drop_table("kernel_channel_requests")
    if "access_generation" in _columns("kernel_sessions"):
        op.drop_column("kernel_sessions", "access_generation")
