"""Independent, bounded input staging with no GPU leases."""

import sqlalchemy as sa
from alembic import op

revision = "0016_input_staging"
down_revision = "0015_forced_reservations"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "input_stages",
        sa.Column("job_id", sa.String(80), sa.ForeignKey("jobs.id", ondelete="CASCADE"),
                  primary_key=True),
        sa.Column("stage_id", sa.String(80), nullable=False, unique=True),
        sa.Column("site_id", sa.String(120), sa.ForeignKey("sites.id"), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("progress", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_input_stages_site_id", "input_stages", ["site_id"])


def downgrade():
    op.drop_table("input_stages")
