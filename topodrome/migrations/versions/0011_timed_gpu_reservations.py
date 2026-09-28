"""Persist durable, timed operator GPU reservations.

Revision ID: 0011_timed_gpu_reservations
Revises: 0010_gpu_reservations
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_timed_gpu_reservations"
down_revision = "0010_gpu_reservations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_capabilities",
        sa.Column(
            "replica_key_hash",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
    )
    op.create_table(
        "gpu_reservations",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("site_id", sa.String(length=120), nullable=False),
        sa.Column("gpu_uuid", sa.String(length=200), nullable=False),
        sa.Column("gpu_index", sa.Integer(), nullable=False),
        sa.Column("requested_by", sa.String(length=200), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("duration_seconds", sa.Integer(), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=True),
        sa.Column(
            "state",
            sa.Enum(
                "pending",
                "ready",
                "canceled",
                "expired",
                name="gpureservationstate",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("canceled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("eviction_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "idempotency_key",
            name="uq_gpu_reservations_idempotency_key",
        ),
    )
    op.create_index(
        "ix_gpu_reservations_site_id",
        "gpu_reservations",
        ["site_id"],
    )
    op.create_index(
        "ix_gpu_reservations_state",
        "gpu_reservations",
        ["state"],
    )
    op.create_index(
        "ix_gpu_reservations_expires_at",
        "gpu_reservations",
        ["expires_at"],
    )
    op.create_index(
        "uq_active_gpu_reservation",
        "gpu_reservations",
        ["site_id", "gpu_uuid"],
        unique=True,
        postgresql_where=sa.text("state IN ('pending', 'ready')"),
        sqlite_where=sa.text("state IN ('pending', 'ready')"),
    )


def downgrade() -> None:
    op.drop_index("uq_active_gpu_reservation", table_name="gpu_reservations")
    op.drop_index("ix_gpu_reservations_expires_at", table_name="gpu_reservations")
    op.drop_index("ix_gpu_reservations_state", table_name="gpu_reservations")
    op.drop_index("ix_gpu_reservations_site_id", table_name="gpu_reservations")
    op.drop_table("gpu_reservations")
    op.drop_column("model_capabilities", "replica_key_hash")
