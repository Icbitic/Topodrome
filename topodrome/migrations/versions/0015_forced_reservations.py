"""Explicit forced reservations and durable attempt retry attribution."""

import sqlalchemy as sa
from alembic import op

revision = "0015_forced_reservations"
down_revision = "0014_job_assets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("SET LOCAL lock_timeout = '5s'")
        op.execute("SET LOCAL statement_timeout = '30s'")
    op.add_column(
        "gpu_reservations",
        sa.Column("force", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("attempts", sa.Column("forced_reservation_id", sa.String(80), nullable=True))


def downgrade() -> None:
    op.drop_column("attempts", "forced_reservation_id")
    op.drop_column("gpu_reservations", "force")
