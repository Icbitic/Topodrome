"""Fence site connections and provider shutdown acknowledgements.

Revision ID: 0005_shutdown_generations
Revises: 0004_kernel_requests

The initial migration historically imported live ORM metadata. Conditional
additions remain compatible with databases created during that period as well
as the now-frozen revision chain.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_shutdown_generations"
down_revision = "0004_kernel_requests"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table not in set(inspector.get_table_names()):
        return set()
    return {str(column["name"]) for column in inspector.get_columns(table)}


def upgrade() -> None:
    site_columns = _columns("sites")
    if site_columns and "agent_connection_generation" not in site_columns:
        op.add_column(
            "sites",
            sa.Column("agent_connection_generation", sa.String(length=120), nullable=True),
        )
    kernel_request_columns = _columns("kernel_channel_requests")
    if (
        kernel_request_columns
        and "agent_connection_generation" not in kernel_request_columns
    ):
        op.add_column(
            "kernel_channel_requests",
            sa.Column("agent_connection_generation", sa.String(length=120), nullable=True),
        )

    provider_columns = _columns("provider_instances")
    additions = (
        sa.Column(
            "shutdown_generation",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("shutdown_agent_session_id", sa.String(length=120), nullable=True),
        sa.Column(
            "shutdown_connection_generation", sa.String(length=120), nullable=True
        ),
        sa.Column("drain_command_nonce", sa.String(length=120), nullable=True),
        sa.Column("drain_ack_nonce", sa.String(length=120), nullable=True),
        sa.Column("durability_command_nonce", sa.String(length=120), nullable=True),
        sa.Column("durability_ack_nonce", sa.String(length=120), nullable=True),
    )
    for column in additions:
        if provider_columns and column.name not in provider_columns:
            op.add_column("provider_instances", column)


def downgrade() -> None:
    provider_columns = _columns("provider_instances")
    for name in (
        "durability_ack_nonce",
        "durability_command_nonce",
        "drain_ack_nonce",
        "drain_command_nonce",
        "shutdown_connection_generation",
        "shutdown_agent_session_id",
        "shutdown_generation",
    ):
        if name in provider_columns:
            op.drop_column("provider_instances", name)
    if "agent_connection_generation" in _columns("kernel_channel_requests"):
        op.drop_column("kernel_channel_requests", "agent_connection_generation")
    if "agent_connection_generation" in _columns("sites"):
        op.drop_column("sites", "agent_connection_generation")
