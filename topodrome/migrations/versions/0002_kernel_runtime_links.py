"""Persist kernel-to-attempt and runtime identities.

Revision ID: 0002_kernel_runtime_links
Revises: 0001_initial
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_kernel_runtime_links"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The guards remain for databases created while revision 0001 still
    # imported live ORM metadata and could already contain these columns.
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("kernel_sessions")}
    with op.batch_alter_table("kernel_sessions") as batch:
        if "attempt_id" not in columns:
            batch.add_column(sa.Column("attempt_id", sa.String(length=80), nullable=True))
            batch.create_foreign_key(
                "fk_kernel_sessions_attempt_id_attempts",
                "attempts",
                ["attempt_id"],
                ["id"],
                ondelete="SET NULL",
            )
            batch.create_index("ix_kernel_sessions_attempt_id", ["attempt_id"])
        if "runtime_id" not in columns:
            batch.add_column(sa.Column("runtime_id", sa.String(length=160), nullable=True))
        if "actor_id" not in columns:
            batch.add_column(sa.Column("actor_id", sa.String(length=160), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("kernel_sessions")}
    with op.batch_alter_table("kernel_sessions") as batch:
        if "actor_id" in columns:
            batch.drop_column("actor_id")
        if "runtime_id" in columns:
            batch.drop_column("runtime_id")
        if "attempt_id" in columns:
            batch.drop_index("ix_kernel_sessions_attempt_id")
            batch.drop_constraint("fk_kernel_sessions_attempt_id_attempts", type_="foreignkey")
            batch.drop_column("attempt_id")
