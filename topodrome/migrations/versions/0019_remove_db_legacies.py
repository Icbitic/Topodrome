"""Remove unused log indexes and persisted kernel URLs.

Runtime job replay now requires owner-scoped keys. Refuse unexpected legacy
data rather than silently dropping it or breaking replay. No historical rows
are rewritten; installations with such data need an explicit archival decision.
"""

import sqlalchemy as sa
from alembic import op

revision = "0019_remove_db_legacies"
down_revision = "0018_job_arrays"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        # Bound maintenance locks and keep the preflight and DDL atomic.
        op.execute("SET LOCAL lock_timeout = '5s'")
        op.execute("SET LOCAL statement_timeout = '30s'")
        op.execute("LOCK TABLE log_indexes, kernel_sessions IN ACCESS EXCLUSIVE MODE")

    checks = (
        ("SELECT 1 FROM log_indexes LIMIT 1", "log-index rows"),
        (
            "SELECT 1 FROM kernel_sessions WHERE connect_url IS NOT NULL LIMIT 1",
            "stored kernel URLs",
        ),
        (
            "SELECT 1 FROM jobs WHERE idempotency_key IS NOT NULL "
            "AND idempotency_key NOT LIKE 'owner-v1:%' LIMIT 1",
            "unscoped job idempotency keys",
        ),
    )
    for query, label in checks:
        if connection.execute(sa.text(query)).first() is not None:
            raise RuntimeError(
                f"database cleanup requires explicit archival of {label}; no data removed"
            )

    op.drop_table("log_indexes")
    with op.batch_alter_table("kernel_sessions") as batch:
        batch.drop_column("connect_url")


def downgrade() -> None:
    # Upgrade required these structures to be empty; no payload reconstruction
    # or credential restoration is needed for a schema rollback.
    op.add_column("kernel_sessions", sa.Column("connect_url", sa.String(2000)))
    op.create_table(
        "log_indexes",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("job_id", sa.String(80), nullable=False),
        sa.Column("attempt_id", sa.String(80), nullable=False),
        sa.Column("logical_path", sa.String(1200), nullable=False),
        sa.Column("start_offset", sa.BigInteger(), nullable=False),
        sa.Column("end_offset", sa.BigInteger(), nullable=False),
        sa.Column("first_timestamp", sa.DateTime(timezone=True)),
        sa.Column("last_timestamp", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_log_indexes_job_id", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["attempt_id"], ["attempts.id"],
            name="fk_log_indexes_attempt_id", ondelete="CASCADE",
        ),
    )
    op.create_index("ix_log_indexes_job_id", "log_indexes", ["job_id"])
    op.create_index("ix_log_indexes_attempt_id", "log_indexes", ["attempt_id"])
