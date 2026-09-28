"""Create the original Topodrome scheduler schema.

Revision ID: 0001_initial
Revises: None

This revision is deliberately self-contained. Historical migrations must not
import live ORM metadata: doing so makes the meaning of ``0001_initial`` change
whenever a model is added or altered.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def _enum(name: str, *values: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False)


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column(
            "state",
            _enum(
                "jobstate",
                "queued",
                "assigning",
                "running",
                "retry_wait",
                "canceling",
                "succeeded",
                "failed",
                "canceled",
            ),
            nullable=False,
        ),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key", name="uq_jobs_idempotency_key"),
    )
    op.create_index("ix_jobs_state", "jobs", ["state"])

    op.create_table(
        "sites",
        sa.Column("id", sa.String(length=120), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column(
            "kind",
            _enum("sitekind", "lab", "autodl", "jupyter"),
            nullable=False,
        ),
        sa.Column(
            "state",
            _enum(
                "sitestate",
                "provisioning",
                "connecting",
                "ready",
                "draining",
                "stopped",
                "starting",
                "error",
                "offline",
            ),
            nullable=False,
        ),
        sa.Column("agent_session_id", sa.String(length=120), nullable=True),
        sa.Column("last_sequence", sa.Integer(), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("storage_available", sa.Boolean(), nullable=False),
        sa.Column("supported_workloads", sa.JSON(), nullable=False),
        sa.Column("queue_depth", sa.Integer(), nullable=False),
        sa.Column("reliability", sa.Float(), nullable=False),
        sa.Column("gpu_safety_margin_bytes", sa.BigInteger(), nullable=False),
        sa.Column("paid", sa.Boolean(), nullable=False),
        sa.Column("region", sa.String(length=120), nullable=True),
        sa.Column("provider_instance_id", sa.String(length=120), nullable=True),
        sa.Column("job_running", sa.Boolean(), nullable=False),
        sa.Column("assignment_queued_locally", sa.Boolean(), nullable=False),
        sa.Column("interactive_kernel_active", sa.Boolean(), nullable=False),
        sa.Column("artifact_flush_pending", sa.Boolean(), nullable=False),
        sa.Column("staging_operation_active", sa.Boolean(), nullable=False),
        sa.Column("drain_ready", sa.Boolean(), nullable=False),
        sa.Column("durability_ready", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sites_heartbeat_at", "sites", ["heartbeat_at"])
    op.create_index("ix_sites_provider_instance_id", "sites", ["provider_instance_id"])
    op.create_index("ix_sites_state", "sites", ["state"])

    op.create_table(
        "provider_instances",
        sa.Column("id", sa.String(length=120), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("external_id", sa.String(length=120), nullable=True),
        sa.Column("site_id", sa.String(length=120), nullable=True),
        sa.Column(
            "state",
            _enum(
                "providerinstancestate",
                "provisioning",
                "running",
                "draining",
                "stopping",
                "stopped",
                "starting",
                "starting_failed",
                "durability_confirmed",
                "releasing",
                "released",
                "error",
            ),
            nullable=False,
        ),
        sa.Column("region", sa.String(length=120), nullable=True),
        sa.Column("gpu_spec", sa.String(length=120), nullable=False),
        sa.Column("image_uuid", sa.String(length=120), nullable=False),
        sa.Column("durable_state_confirmed", sa.Boolean(), nullable=False),
        sa.Column("idle_since", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id", name="uq_provider_instances_external_id"),
    )
    op.create_index("ix_provider_instances_provider", "provider_instances", ["provider"])
    op.create_index("ix_provider_instances_region", "provider_instances", ["region"])
    op.create_index("ix_provider_instances_site_id", "provider_instances", ["site_id"])
    op.create_index("ix_provider_instances_state", "provider_instances", ["state"])

    op.create_table(
        "gpu_fence_counters",
        sa.Column("site_id", sa.String(length=120), nullable=False),
        sa.Column("gpu_uuid", sa.String(length=200), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("site_id", "gpu_uuid"),
    )

    op.create_table(
        "agent_event_acks",
        sa.Column("message_id", sa.String(length=120), nullable=False),
        sa.Column("site_id", sa.String(length=120), nullable=False),
        sa.Column("agent_session_id", sa.String(length=120), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("payload_digest", sa.LargeBinary(length=32), nullable=False),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("message_id"),
        sa.UniqueConstraint(
            "site_id",
            "agent_session_id",
            "sequence",
            name="uq_agent_sequence",
        ),
    )
    op.create_index("ix_agent_event_acks_site_id", "agent_event_acks", ["site_id"])

    op.create_table(
        "job_state_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=80), nullable=False),
        sa.Column("from_state", sa.String(length=40), nullable=True),
        sa.Column("to_state", sa.String(length=40), nullable=False),
        sa.Column("attempt_id", sa.String(length=80), nullable=True),
        sa.Column("reason_code", sa.String(length=100), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_job_state_events_job_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_job_state_events_created_at", "job_state_events", ["created_at"])
    op.create_index("ix_job_state_events_job_id", "job_state_events", ["job_id"])

    op.create_table(
        "gpus",
        sa.Column("site_id", sa.String(length=120), nullable=False),
        sa.Column("uuid", sa.String(length=200), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("total_memory_bytes", sa.BigInteger(), nullable=False),
        sa.Column("free_memory_bytes", sa.BigInteger(), nullable=False),
        sa.Column("utilization_percent", sa.Float(), nullable=False),
        sa.Column("temperature_c", sa.Float(), nullable=True),
        sa.Column("foreign_process_count", sa.Integer(), nullable=False),
        sa.Column("foreign_memory_bytes", sa.BigInteger(), nullable=False),
        sa.Column("healthy", sa.Boolean(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["site_id"], ["sites.id"], name="fk_gpus_site_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("site_id", "uuid"),
        sa.UniqueConstraint("site_id", "index", name="uq_gpu_site_index"),
    )
    op.create_index("ix_gpus_observed_at", "gpus", ["observed_at"])

    op.create_table(
        "model_capabilities",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.String(length=120), nullable=False),
        sa.Column("alias", sa.String(length=200), nullable=False),
        sa.Column("revision", sa.String(length=200), nullable=False),
        sa.Column("backend", sa.String(length=100), nullable=False),
        sa.Column("backend_version", sa.String(length=100), nullable=False),
        sa.Column("dtype", sa.String(length=50), nullable=False),
        sa.Column("quantization", sa.String(length=100), nullable=False),
        sa.Column("adapter", sa.String(length=500), nullable=False),
        sa.Column(
            "availability",
            _enum("modelavailability", "warm", "cached", "obtainable"),
            nullable=False,
        ),
        sa.Column("gpu_uuid", sa.String(length=200), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["site_id"],
            ["sites.id"],
            name="fk_model_capabilities_site_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "site_id",
            "alias",
            "revision",
            "backend",
            "backend_version",
            "dtype",
            "quantization",
            "adapter",
            "gpu_uuid",
            name="uq_model_capability_identity",
        ),
    )
    op.create_index("ix_model_capabilities_alias", "model_capabilities", ["alias"])
    op.create_index(
        "ix_model_capabilities_availability", "model_capabilities", ["availability"]
    )
    op.create_index("ix_model_capabilities_site_id", "model_capabilities", ["site_id"])

    op.create_table(
        "model_actors",
        sa.Column("id", sa.String(length=120), nullable=False),
        sa.Column("site_id", sa.String(length=120), nullable=False),
        sa.Column("gpu_uuid", sa.String(length=200), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("key_data", sa.JSON(), nullable=False),
        sa.Column(
            "state",
            _enum(
                "actorstate",
                "loading",
                "idle",
                "busy",
                "evicting",
                "dirty",
                "stopping",
                "stopped",
            ),
            nullable=False,
        ),
        sa.Column("active_allocation", sa.Boolean(), nullable=False),
        sa.Column("owner_attempt_id", sa.String(length=80), nullable=True),
        sa.Column("process_ids", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["site_id"], ["sites.id"], name="fk_model_actors_site_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_actors_key_hash", "model_actors", ["key_hash"])
    op.create_index("ix_model_actors_site_id", "model_actors", ["site_id"])
    op.create_index("ix_model_actors_state", "model_actors", ["state"])
    op.create_index(
        "uq_active_actor_gpu",
        "model_actors",
        ["site_id", "gpu_uuid"],
        unique=True,
        postgresql_where=sa.text("active_allocation"),
        sqlite_where=sa.text("active_allocation = 1"),
    )

    op.create_table(
        "attempts",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("job_id", sa.String(length=80), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column(
            "state",
            _enum(
                "attemptstate",
                "offered",
                "accepted",
                "starting",
                "running",
                "preempt_requested",
                "checkpointing",
                "committing",
                "succeeded",
                "rejected",
                "preempted",
                "failed",
                "canceled",
                "lost",
            ),
            nullable=False,
        ),
        sa.Column("site_id", sa.String(length=120), nullable=True),
        sa.Column("reason_code", sa.String(length=100), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("checkpoint_ref", sa.String(length=1000), nullable=True),
        sa.Column("offer_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fencing_token_digest", sa.String(length=64), nullable=True),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_attempts_job_id", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["site_id"], ["sites.id"], name="fk_attempts_site_id", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "number", name="uq_attempt_job_number"),
    )
    op.create_index("ix_attempts_job_id", "attempts", ["job_id"])
    op.create_index("ix_attempts_offer_expires_at", "attempts", ["offer_expires_at"])
    op.create_index("ix_attempts_site_id", "attempts", ["site_id"])
    op.create_index("ix_attempts_state", "attempts", ["state"])

    op.create_table(
        "leases",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("attempt_id", sa.String(length=80), nullable=False),
        sa.Column("site_id", sa.String(length=120), nullable=False),
        sa.Column(
            "state",
            _enum(
                "leasestate",
                "offered",
                "accepted",
                "active",
                "releasing",
                "released",
                "rejected",
                "expired",
                "lost",
            ),
            nullable=False,
        ),
        sa.Column("fencing_token_digest", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["attempt_id"], ["attempts.id"], name="fk_leases_attempt_id", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["site_id"], ["sites.id"], name="fk_leases_site_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("attempt_id", name="uq_leases_attempt_id"),
    )
    op.create_index("ix_leases_expires_at", "leases", ["expires_at"])
    op.create_index("ix_leases_site_id", "leases", ["site_id"])
    op.create_index("ix_leases_state", "leases", ["state"])

    op.create_table(
        "lease_gpu_claims",
        sa.Column("lease_id", sa.String(length=80), nullable=False),
        sa.Column("site_id", sa.String(length=120), nullable=False),
        sa.Column("gpu_uuid", sa.String(length=200), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["lease_id"],
            ["leases.id"],
            name="fk_lease_gpu_claims_lease_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("lease_id", "site_id", "gpu_uuid"),
    )
    op.create_index("ix_lease_gpu_claims_active", "lease_gpu_claims", ["active"])
    op.create_index(
        "uq_active_lease_gpu",
        "lease_gpu_claims",
        ["site_id", "gpu_uuid"],
        unique=True,
        postgresql_where=sa.text("active"),
        sqlite_where=sa.text("active = 1"),
    )

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=80), nullable=False),
        sa.Column("attempt_id", sa.String(length=80), nullable=False),
        sa.Column("logical_path", sa.String(length=1200), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "state",
            _enum(
                "artifactstate", "staging", "flushing", "verified", "committed", "failed"
            ),
            nullable=False,
        ),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["attempts.id"],
            name="fk_artifacts_attempt_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_artifacts_job_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("attempt_id", "logical_path", name="uq_artifact_attempt_path"),
    )
    op.create_index("ix_artifacts_attempt_id", "artifacts", ["attempt_id"])
    op.create_index("ix_artifacts_job_id", "artifacts", ["job_id"])
    op.create_index("ix_artifacts_state", "artifacts", ["state"])

    op.create_table(
        "log_indexes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=80), nullable=False),
        sa.Column("attempt_id", sa.String(length=80), nullable=False),
        sa.Column("logical_path", sa.String(length=1200), nullable=False),
        sa.Column("start_offset", sa.BigInteger(), nullable=False),
        sa.Column("end_offset", sa.BigInteger(), nullable=False),
        sa.Column("first_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["attempts.id"],
            name="fk_log_indexes_attempt_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_log_indexes_job_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_log_indexes_attempt_id", "log_indexes", ["attempt_id"])
    op.create_index("ix_log_indexes_job_id", "log_indexes", ["job_id"])

    op.create_table(
        "kernel_sessions",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column(
            "state",
            _enum(
                "kernelstate",
                "requested",
                "starting",
                "running",
                "idle",
                "stopping",
                "stopped",
                "lost",
            ),
            nullable=False,
        ),
        sa.Column("mode", _enum("kernelmode", "service", "raw-model"), nullable=False),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("job_id", sa.String(length=80), nullable=True),
        sa.Column("site_id", sa.String(length=120), nullable=True),
        sa.Column("relay_id", sa.String(length=120), nullable=True),
        sa.Column("connect_url", sa.String(length=2000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_kernel_sessions_job_id", ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["site_id"], ["sites.id"], name="fk_kernel_sessions_site_id", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("relay_id", name="uq_kernel_sessions_relay_id"),
    )
    op.create_index("ix_kernel_sessions_expires_at", "kernel_sessions", ["expires_at"])
    op.create_index("ix_kernel_sessions_site_id", "kernel_sessions", ["site_id"])
    op.create_index("ix_kernel_sessions_state", "kernel_sessions", ["state"])

    op.create_table(
        "provider_operations",
        sa.Column("id", sa.String(length=120), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("instance_id", sa.String(length=120), nullable=True),
        sa.Column("external_instance_id", sa.String(length=120), nullable=True),
        sa.Column(
            "action",
            _enum("provideraction", "create", "power_on", "power_off", "release"),
            nullable=False,
        ),
        sa.Column(
            "state",
            _enum(
                "provideroperationstate",
                "intent",
                "calling",
                "applied",
                "reconciling",
                "failed",
            ),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("request_redacted", sa.JSON(), nullable=False),
        sa.Column("external_request_id", sa.String(length=200), nullable=True),
        sa.Column("result_external_instance_id", sa.String(length=120), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["instance_id"],
            ["provider_instances.id"],
            name="fk_provider_operations_instance_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key", name="uq_provider_operations_idempotency_key"),
    )
    op.create_index(
        "ix_provider_operations_external_instance_id",
        "provider_operations",
        ["external_instance_id"],
    )
    op.create_index(
        "ix_provider_operations_instance_id", "provider_operations", ["instance_id"]
    )
    op.create_index("ix_provider_operations_provider", "provider_operations", ["provider"])
    op.create_index("ix_provider_operations_state", "provider_operations", ["state"])


def downgrade() -> None:
    op.drop_table("provider_operations")
    op.drop_table("kernel_sessions")
    op.drop_table("log_indexes")
    op.drop_table("artifacts")
    op.drop_table("lease_gpu_claims")
    op.drop_table("leases")
    op.drop_table("attempts")
    op.drop_table("model_actors")
    op.drop_table("model_capabilities")
    op.drop_table("gpus")
    op.drop_table("job_state_events")
    op.drop_table("agent_event_acks")
    op.drop_table("gpu_fence_counters")
    op.drop_table("provider_instances")
    op.drop_table("sites")
    op.drop_table("jobs")
