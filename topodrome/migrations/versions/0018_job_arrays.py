"""Durable arrays and scheduler-observed phase timings."""

import sqlalchemy as sa
from alembic import op

revision = "0018_job_arrays"
down_revision = "0017_adaptive_inputs"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "job_arrays",
        sa.Column("id", sa.String(80), primary_key=True),
        sa.Column("owner_username", sa.String(64), nullable=False, index=True),
        sa.Column("name", sa.String(160)),
        sa.Column("max_parallel", sa.Integer(), nullable=False),
        sa.Column("held", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("idempotency_key", sa.String(200), unique=True),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(sa.Column("array_id", sa.String(80)))
        batch.create_foreign_key("fk_jobs_array", "job_arrays", ["array_id"], ["id"])
        batch.add_column(sa.Column("array_index", sa.Integer()))
        batch.create_unique_constraint("uq_array_element", ["array_id", "array_index"])
        batch.create_index("ix_jobs_array_id", ["array_id"])
        batch.add_column(sa.Column("eligible_since", sa.DateTime(timezone=True)))
        batch.create_index("ix_jobs_eligible_since", ["eligible_since"])
        batch.add_column(sa.Column("blocked_reason", sa.String(200)))
        batch.add_column(sa.Column("phase", sa.String(32), nullable=False, server_default="queued"))
        batch.add_column(sa.Column("phase_started_at", sa.DateTime(timezone=True)))
        batch.add_column(sa.Column("phase_seconds", sa.JSON(), nullable=False, server_default="{}"))
    op.execute("UPDATE jobs SET phase='untracked', phase_started_at=updated_at")
    op.execute("UPDATE jobs SET eligible_since=updated_at WHERE state='queued'")
    op.add_column("attempts", sa.Column("compute_finished_at", sa.DateTime(timezone=True)))


def downgrade():
    op.drop_column("attempts", "compute_finished_at")
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("fk_jobs_array", type_="foreignkey")
        batch.drop_constraint("uq_array_element", type_="unique")
        batch.drop_index("ix_jobs_array_id")
        batch.drop_index("ix_jobs_eligible_since")
        for name in (
            "array_id",
            "array_index",
            "eligible_since",
            "blocked_reason",
            "phase",
            "phase_started_at",
            "phase_seconds",
        ):
            batch.drop_column(name)
    op.drop_table("job_arrays")
