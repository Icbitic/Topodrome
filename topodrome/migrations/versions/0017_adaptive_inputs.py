"""Adaptive, lease-free input placement and measured scheduling costs."""

import sqlalchemy as sa
from alembic import op

revision = "0017_adaptive_inputs"
down_revision = "0016_input_staging"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "input_stages", sa.Column("placement_data", sa.JSON(), nullable=False, server_default="{}")
    )
    op.add_column(
        "sites",
        sa.Column("storage_authoritative", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_table(
        "scheduling_measurements",
        sa.Column("site_id", sa.String(120), sa.ForeignKey("sites.id"), primary_key=True),
        sa.Column("metric", sa.String(32), primary_key=True),
        sa.Column("context_key", sa.String(64), primary_key=True),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("samples", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade():
    op.drop_table("scheduling_measurements")
    op.drop_column("sites", "storage_authoritative")
    op.drop_column("input_stages", "placement_data")
