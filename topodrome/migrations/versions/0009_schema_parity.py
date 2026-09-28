"""Repair known migration omissions and require schema parity.

Revision ID: 0009_schema_parity
Revises: 0008_model_replicas

Early versions of ``0001_initial`` imported live ORM metadata, while several
later revisions silently skipped changes when a table was absent.  This
revision repairs only changes that are unambiguous and safe to backfill.  It
fails closed on damage to the original schema so Alembic cannot stamp an
incomplete database as current.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "0009_schema_parity"
down_revision = "0008_model_replicas"
branch_labels = None
depends_on = None


_EXPECTED_COLUMNS = {
    "jobs": {
        "id",
        "state",
        "spec",
        "idempotency_key",
        "priority",
        "cancel_requested",
        "retry_not_before",
        "created_at",
        "updated_at",
    },
    "job_state_events": {
        "id",
        "job_id",
        "from_state",
        "to_state",
        "attempt_id",
        "reason_code",
        "message",
        "created_at",
    },
    "sites": {
        "id",
        "display_name",
        "kind",
        "state",
        "agent_session_id",
        "agent_connection_generation",
        "last_sequence",
        "heartbeat_at",
        "storage_available",
        "supported_workloads",
        "cached_datasets",
        "queue_depth",
        "reliability",
        "gpu_safety_margin_bytes",
        "total_cpu",
        "available_cpu",
        "total_memory_bytes",
        "available_memory_bytes",
        "paid",
        "region",
        "provider_instance_id",
        "job_running",
        "assignment_queued_locally",
        "interactive_kernel_active",
        "artifact_flush_pending",
        "staging_operation_active",
        "drain_ready",
        "durability_ready",
        "created_at",
        "updated_at",
    },
    "gpus": {
        "site_id",
        "uuid",
        "index",
        "name",
        "total_memory_bytes",
        "free_memory_bytes",
        "utilization_percent",
        "temperature_c",
        "foreign_process_count",
        "foreign_memory_bytes",
        "healthy",
        "observed_at",
    },
    "model_capabilities": {
        "id",
        "site_id",
        "alias",
        "revision",
        "backend",
        "backend_version",
        "dtype",
        "quantization",
        "adapter",
        "availability",
        "gpu_uuid",
        "runtime_fingerprint",
        "replica_id",
        "replica_gpu_uuids",
        "verified_at",
    },
    "model_actors": {
        "id",
        "site_id",
        "gpu_uuid",
        "key_hash",
        "key_data",
        "state",
        "active_allocation",
        "owner_attempt_id",
        "process_ids",
        "created_at",
        "last_used_at",
    },
    "attempts": {
        "id",
        "job_id",
        "number",
        "state",
        "site_id",
        "reason_code",
        "message",
        "checkpoint_ref",
        "restore_checkpoint_ref",
        "offer_expires_at",
        "fencing_token_digest",
        "accepted_at",
        "started_at",
        "completed_at",
        "created_at",
        "updated_at",
    },
    "leases": {
        "id",
        "attempt_id",
        "site_id",
        "state",
        "fencing_token_digest",
        "expires_at",
        "created_at",
        "updated_at",
    },
    "gpu_fence_counters": {"site_id", "gpu_uuid", "epoch"},
    "lease_gpu_claims": {"lease_id", "site_id", "gpu_uuid", "epoch", "active"},
    "artifacts": {
        "id",
        "job_id",
        "attempt_id",
        "logical_path",
        "size_bytes",
        "sha256",
        "state",
        "committed_at",
    },
    "log_indexes": {
        "id",
        "job_id",
        "attempt_id",
        "logical_path",
        "start_offset",
        "end_offset",
        "first_timestamp",
        "last_timestamp",
    },
    "kernel_sessions": {
        "id",
        "state",
        "mode",
        "request",
        "job_id",
        "attempt_id",
        "site_id",
        "relay_id",
        "runtime_id",
        "actor_id",
        "access_generation",
        "connect_url",
        "created_at",
        "last_activity_at",
        "expires_at",
    },
    "kernel_channel_requests": {
        "id",
        "session_id",
        "request_id",
        "request_digest",
        "state",
        "execution_count",
        "agent_session_id",
        "agent_connection_generation",
        "result_digest",
        "error_code",
        "created_at",
        "updated_at",
    },
    "provider_instances": {
        "id",
        "provider",
        "external_id",
        "site_id",
        "state",
        "region",
        "gpu_spec",
        "image_uuid",
        "durable_state_confirmed",
        "shutdown_generation",
        "shutdown_agent_session_id",
        "shutdown_connection_generation",
        "drain_command_nonce",
        "drain_ack_nonce",
        "durability_command_nonce",
        "durability_ack_nonce",
        "idle_since",
        "metadata",
        "created_at",
        "updated_at",
    },
    "provider_operations": {
        "id",
        "provider",
        "instance_id",
        "external_instance_id",
        "action",
        "state",
        "idempotency_key",
        "request_redacted",
        "external_request_id",
        "result_external_instance_id",
        "error_code",
        "created_at",
        "updated_at",
    },
    "agent_event_acks": {
        "message_id",
        "site_id",
        "agent_session_id",
        "sequence",
        "payload_digest",
        "acknowledged_at",
    },
}

_REPAIRABLE_COLUMNS = {
    "jobs": {"retry_not_before"},
    "attempts": {"restore_checkpoint_ref"},
    "kernel_sessions": {"attempt_id", "runtime_id", "actor_id", "access_generation"},
    "kernel_channel_requests": {"agent_connection_generation"},
    "sites": {
        "agent_connection_generation",
        "cached_datasets",
        "total_cpu",
        "available_cpu",
        "total_memory_bytes",
        "available_memory_bytes",
    },
    "model_capabilities": {
        "runtime_fingerprint",
        "replica_id",
        "replica_gpu_uuids",
    },
    "provider_instances": {
        "shutdown_generation",
        "shutdown_agent_session_id",
        "shutdown_connection_generation",
        "drain_command_nonce",
        "drain_ack_nonce",
        "durability_command_nonce",
        "durability_ack_nonce",
    },
}

_NULLABLE_COLUMNS = {
    "jobs": {"idempotency_key", "retry_not_before"},
    "job_state_events": {"from_state", "attempt_id", "reason_code", "message"},
    "sites": {
        "agent_session_id",
        "agent_connection_generation",
        "heartbeat_at",
        "total_cpu",
        "available_cpu",
        "total_memory_bytes",
        "available_memory_bytes",
        "region",
        "provider_instance_id",
    },
    "gpus": {"temperature_c"},
    "model_capabilities": set(),
    "model_actors": {"owner_attempt_id"},
    "attempts": {
        "site_id",
        "reason_code",
        "message",
        "checkpoint_ref",
        "restore_checkpoint_ref",
        "offer_expires_at",
        "fencing_token_digest",
        "accepted_at",
        "started_at",
        "completed_at",
    },
    "leases": set(),
    "gpu_fence_counters": set(),
    "lease_gpu_claims": set(),
    "artifacts": {"committed_at"},
    "log_indexes": {"first_timestamp", "last_timestamp"},
    "kernel_sessions": {
        "job_id",
        "attempt_id",
        "site_id",
        "relay_id",
        "runtime_id",
        "actor_id",
        "connect_url",
        "expires_at",
    },
    "kernel_channel_requests": {
        "agent_session_id",
        "agent_connection_generation",
        "result_digest",
        "error_code",
    },
    "provider_instances": {
        "external_id",
        "site_id",
        "region",
        "shutdown_agent_session_id",
        "shutdown_connection_generation",
        "drain_command_nonce",
        "drain_ack_nonce",
        "durability_command_nonce",
        "durability_ack_nonce",
        "idle_since",
    },
    "provider_operations": {
        "instance_id",
        "external_instance_id",
        "external_request_id",
        "result_external_instance_id",
        "error_code",
    },
    "agent_event_acks": set(),
}

# Coarse, dialect-independent SQL type families. Enums intentionally fall
# through to String because all Topodrome enums use ``native_enum=False``.
_JSON_COLUMNS = {
    "jobs.spec",
    "sites.supported_workloads",
    "sites.cached_datasets",
    "model_capabilities.replica_gpu_uuids",
    "model_actors.key_data",
    "model_actors.process_ids",
    "kernel_sessions.request",
    "provider_instances.metadata",
    "provider_operations.request_redacted",
}
_BOOLEAN_COLUMNS = {
    "jobs.cancel_requested",
    "sites.storage_available",
    "sites.paid",
    "sites.job_running",
    "sites.assignment_queued_locally",
    "sites.interactive_kernel_active",
    "sites.artifact_flush_pending",
    "sites.staging_operation_active",
    "sites.drain_ready",
    "sites.durability_ready",
    "gpus.healthy",
    "model_actors.active_allocation",
    "lease_gpu_claims.active",
    "provider_instances.durable_state_confirmed",
}
_DATETIME_COLUMNS = {
    "jobs.retry_not_before",
    "jobs.created_at",
    "jobs.updated_at",
    "job_state_events.created_at",
    "sites.heartbeat_at",
    "sites.created_at",
    "sites.updated_at",
    "gpus.observed_at",
    "model_capabilities.verified_at",
    "model_actors.created_at",
    "model_actors.last_used_at",
    "attempts.offer_expires_at",
    "attempts.accepted_at",
    "attempts.started_at",
    "attempts.completed_at",
    "attempts.created_at",
    "attempts.updated_at",
    "leases.expires_at",
    "leases.created_at",
    "leases.updated_at",
    "artifacts.committed_at",
    "log_indexes.first_timestamp",
    "log_indexes.last_timestamp",
    "kernel_sessions.created_at",
    "kernel_sessions.last_activity_at",
    "kernel_sessions.expires_at",
    "kernel_channel_requests.created_at",
    "kernel_channel_requests.updated_at",
    "provider_instances.idle_since",
    "provider_instances.created_at",
    "provider_instances.updated_at",
    "provider_operations.created_at",
    "provider_operations.updated_at",
    "agent_event_acks.acknowledged_at",
}
_BIG_INTEGER_COLUMNS = {
    "sites.gpu_safety_margin_bytes",
    "sites.total_memory_bytes",
    "sites.available_memory_bytes",
    "gpus.total_memory_bytes",
    "gpus.free_memory_bytes",
    "gpus.foreign_memory_bytes",
    "artifacts.size_bytes",
    "log_indexes.start_offset",
    "log_indexes.end_offset",
}
_INTEGER_COLUMNS = {
    "jobs.priority",
    "job_state_events.id",
    "sites.last_sequence",
    "sites.queue_depth",
    "gpus.index",
    "gpus.foreign_process_count",
    "model_capabilities.id",
    "attempts.number",
    "gpu_fence_counters.epoch",
    "lease_gpu_claims.epoch",
    "artifacts.id",
    "log_indexes.id",
    "kernel_sessions.access_generation",
    "kernel_channel_requests.execution_count",
    "provider_instances.shutdown_generation",
    "agent_event_acks.sequence",
}
_FLOAT_COLUMNS = {
    "sites.reliability",
    "sites.total_cpu",
    "sites.available_cpu",
    "gpus.utilization_percent",
    "gpus.temperature_c",
}
_BINARY_COLUMNS = {
    "kernel_channel_requests.request_digest",
    "kernel_channel_requests.result_digest",
    "agent_event_acks.payload_digest",
}
_TEXT_COLUMNS = {"job_state_events.message", "attempts.message"}

_PRIMARY_KEYS = {
    "jobs": ("id",),
    "job_state_events": ("id",),
    "sites": ("id",),
    "gpus": ("site_id", "uuid"),
    "model_capabilities": ("id",),
    "model_actors": ("id",),
    "attempts": ("id",),
    "leases": ("id",),
    "gpu_fence_counters": ("site_id", "gpu_uuid"),
    "lease_gpu_claims": ("lease_id", "site_id", "gpu_uuid"),
    "artifacts": ("id",),
    "log_indexes": ("id",),
    "kernel_sessions": ("id",),
    "kernel_channel_requests": ("id",),
    "provider_instances": ("id",),
    "provider_operations": ("id",),
    "agent_event_acks": ("message_id",),
}

_UNIQUE_CONSTRAINTS = {
    "jobs": (("uq_jobs_idempotency_key", ("idempotency_key",)),),
    "gpus": (("uq_gpu_site_index", ("site_id", "index")),),
    "model_capabilities": (
        (
            "uq_model_capability_identity",
            (
                "site_id",
                "alias",
                "revision",
                "backend",
                "backend_version",
                "dtype",
                "quantization",
                "adapter",
                "gpu_uuid",
            ),
        ),
    ),
    "attempts": (("uq_attempt_job_number", ("job_id", "number")),),
    "leases": (("uq_leases_attempt_id", ("attempt_id",)),),
    "artifacts": (("uq_artifact_attempt_path", ("attempt_id", "logical_path")),),
    "kernel_sessions": (("uq_kernel_sessions_relay_id", ("relay_id",)),),
    "kernel_channel_requests": (
        ("uq_kernel_channel_request", ("session_id", "request_id")),
    ),
    "provider_instances": (
        ("uq_provider_instances_external_id", ("external_id",)),
    ),
    "provider_operations": (
        ("uq_provider_operations_idempotency_key", ("idempotency_key",)),
    ),
    "agent_event_acks": (
        ("uq_agent_sequence", ("site_id", "agent_session_id", "sequence")),
    ),
}

# name, local columns, remote table, remote columns, on-delete action
_FOREIGN_KEYS = {
    "job_state_events": (
        ("fk_job_state_events_job_id", ("job_id",), "jobs", ("id",), "CASCADE"),
    ),
    "gpus": (("fk_gpus_site_id", ("site_id",), "sites", ("id",), "CASCADE"),),
    "model_capabilities": (
        (
            "fk_model_capabilities_site_id",
            ("site_id",),
            "sites",
            ("id",),
            "CASCADE",
        ),
    ),
    "model_actors": (
        ("fk_model_actors_site_id", ("site_id",), "sites", ("id",), "CASCADE"),
    ),
    "attempts": (
        ("fk_attempts_job_id", ("job_id",), "jobs", ("id",), "CASCADE"),
        ("fk_attempts_site_id", ("site_id",), "sites", ("id",), "SET NULL"),
    ),
    "leases": (
        ("fk_leases_attempt_id", ("attempt_id",), "attempts", ("id",), "CASCADE"),
        ("fk_leases_site_id", ("site_id",), "sites", ("id",), "CASCADE"),
    ),
    "lease_gpu_claims": (
        ("fk_lease_gpu_claims_lease_id", ("lease_id",), "leases", ("id",), "CASCADE"),
    ),
    "artifacts": (
        ("fk_artifacts_job_id", ("job_id",), "jobs", ("id",), "CASCADE"),
        (
            "fk_artifacts_attempt_id",
            ("attempt_id",),
            "attempts",
            ("id",),
            "CASCADE",
        ),
    ),
    "log_indexes": (
        ("fk_log_indexes_job_id", ("job_id",), "jobs", ("id",), "CASCADE"),
        (
            "fk_log_indexes_attempt_id",
            ("attempt_id",),
            "attempts",
            ("id",),
            "CASCADE",
        ),
    ),
    "kernel_sessions": (
        ("fk_kernel_sessions_job_id", ("job_id",), "jobs", ("id",), "SET NULL"),
        (
            "fk_kernel_sessions_attempt_id_attempts",
            ("attempt_id",),
            "attempts",
            ("id",),
            "SET NULL",
        ),
        ("fk_kernel_sessions_site_id", ("site_id",), "sites", ("id",), "SET NULL"),
    ),
    "kernel_channel_requests": (
        (
            "fk_kernel_channel_requests_session_id",
            ("session_id",),
            "kernel_sessions",
            ("id",),
            "CASCADE",
        ),
    ),
    "provider_operations": (
        (
            "fk_provider_operations_instance_id",
            ("instance_id",),
            "provider_instances",
            ("id",),
            "SET NULL",
        ),
    ),
}

# name, columns, unique, partial predicate (if any)
_INDEXES = {
    "jobs": (
        ("ix_jobs_state", ("state",), False, None),
        ("ix_jobs_retry_not_before", ("retry_not_before",), False, None),
    ),
    "job_state_events": (
        ("ix_job_state_events_job_id", ("job_id",), False, None),
        ("ix_job_state_events_created_at", ("created_at",), False, None),
    ),
    "sites": (
        ("ix_sites_state", ("state",), False, None),
        ("ix_sites_heartbeat_at", ("heartbeat_at",), False, None),
        ("ix_sites_provider_instance_id", ("provider_instance_id",), False, None),
    ),
    "gpus": (("ix_gpus_observed_at", ("observed_at",), False, None),),
    "model_capabilities": (
        ("ix_model_capabilities_site_id", ("site_id",), False, None),
        ("ix_model_capabilities_alias", ("alias",), False, None),
        ("ix_model_capabilities_availability", ("availability",), False, None),
    ),
    "model_actors": (
        ("ix_model_actors_site_id", ("site_id",), False, None),
        ("ix_model_actors_key_hash", ("key_hash",), False, None),
        ("ix_model_actors_state", ("state",), False, None),
        (
            "uq_active_actor_gpu",
            ("site_id", "gpu_uuid"),
            True,
            "active_allocation",
        ),
    ),
    "attempts": (
        ("ix_attempts_job_id", ("job_id",), False, None),
        ("ix_attempts_state", ("state",), False, None),
        ("ix_attempts_site_id", ("site_id",), False, None),
        ("ix_attempts_offer_expires_at", ("offer_expires_at",), False, None),
    ),
    "leases": (
        ("ix_leases_site_id", ("site_id",), False, None),
        ("ix_leases_state", ("state",), False, None),
        ("ix_leases_expires_at", ("expires_at",), False, None),
    ),
    "lease_gpu_claims": (
        ("ix_lease_gpu_claims_active", ("active",), False, None),
        ("uq_active_lease_gpu", ("site_id", "gpu_uuid"), True, "active"),
    ),
    "artifacts": (
        ("ix_artifacts_job_id", ("job_id",), False, None),
        ("ix_artifacts_attempt_id", ("attempt_id",), False, None),
        ("ix_artifacts_state", ("state",), False, None),
    ),
    "log_indexes": (
        ("ix_log_indexes_job_id", ("job_id",), False, None),
        ("ix_log_indexes_attempt_id", ("attempt_id",), False, None),
    ),
    "kernel_sessions": (
        ("ix_kernel_sessions_state", ("state",), False, None),
        ("ix_kernel_sessions_attempt_id", ("attempt_id",), False, None),
        ("ix_kernel_sessions_site_id", ("site_id",), False, None),
        ("ix_kernel_sessions_expires_at", ("expires_at",), False, None),
    ),
    "kernel_channel_requests": (
        ("ix_kernel_channel_requests_session_id", ("session_id",), False, None),
        ("ix_kernel_channel_requests_state", ("state",), False, None),
    ),
    "provider_instances": (
        ("ix_provider_instances_provider", ("provider",), False, None),
        ("ix_provider_instances_site_id", ("site_id",), False, None),
        ("ix_provider_instances_state", ("state",), False, None),
        ("ix_provider_instances_region", ("region",), False, None),
    ),
    "provider_operations": (
        ("ix_provider_operations_provider", ("provider",), False, None),
        ("ix_provider_operations_instance_id", ("instance_id",), False, None),
        (
            "ix_provider_operations_external_instance_id",
            ("external_instance_id",),
            False,
            None,
        ),
        ("ix_provider_operations_state", ("state",), False, None),
    ),
    "agent_event_acks": (
        ("ix_agent_event_acks_site_id", ("site_id",), False, None),
    ),
}


def _column_names(inspector: sa.Inspector, table: str) -> set[str]:
    return {str(column["name"]) for column in inspector.get_columns(table)}


def _unique_signatures(inspector: sa.Inspector, table: str) -> set[tuple[str, ...]]:
    return {
        tuple(str(column) for column in constraint["column_names"])
        for constraint in inspector.get_unique_constraints(table)
    }


def _foreign_key_signatures(
    inspector: sa.Inspector,
    table: str,
) -> set[tuple[tuple[str, ...], str, tuple[str, ...], str]]:
    signatures = set()
    for constraint in inspector.get_foreign_keys(table):
        options = constraint.get("options") or {}
        signatures.add(
            (
                tuple(str(column) for column in constraint["constrained_columns"]),
                str(constraint["referred_table"]),
                tuple(str(column) for column in constraint["referred_columns"]),
                str(options.get("ondelete", "")).upper(),
            )
        )
    return signatures


def _expected_type(table: str, column: str) -> type[sa.types.TypeEngine[Any]]:
    qualified = f"{table}.{column}"
    if qualified in _JSON_COLUMNS:
        return sa.JSON
    if qualified in _BOOLEAN_COLUMNS:
        return sa.Boolean
    if qualified in _DATETIME_COLUMNS:
        return sa.DateTime
    if qualified in _BIG_INTEGER_COLUMNS:
        return sa.BigInteger
    if qualified in _INTEGER_COLUMNS:
        return sa.Integer
    if qualified in _FLOAT_COLUMNS:
        return sa.Float
    if qualified in _BINARY_COLUMNS:
        return sa.LargeBinary
    if qualified in _TEXT_COLUMNS:
        return sa.Text
    return sa.String


def _type_is_compatible(
    actual: sa.types.TypeEngine[Any],
    expected: type[sa.types.TypeEngine[Any]],
) -> bool:
    # BigInteger and Text subclass Integer and String respectively, so exact
    # family checks for those two prevent accepting a materially narrower type.
    if expected is sa.BigInteger:
        return isinstance(actual, sa.BigInteger)
    if expected is sa.Text:
        return isinstance(actual, sa.Text)
    return isinstance(actual, expected)


def _normalize_predicate(value: object | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"[\s()`\"']", "", str(value)).lower()


def _index_predicate(index: Mapping[str, Any], dialect_name: str) -> str | None:
    options = index.get("dialect_options") or {}
    return _normalize_predicate(options.get(f"{dialect_name}_where"))


def _expected_index_predicate(
    predicate: str | None,
    dialect_name: str,
) -> str | None:
    if predicate is None:
        return None
    expression = f"{predicate} = 1" if dialect_name == "sqlite" else predicate
    return _normalize_predicate(expression)


def _repair_columns() -> dict[str, tuple[sa.Column[Any], ...]]:
    return {
        "jobs": (
            sa.Column("retry_not_before", sa.DateTime(timezone=True), nullable=True),
        ),
        "attempts": (
            sa.Column("restore_checkpoint_ref", sa.String(length=1000), nullable=True),
        ),
        "kernel_sessions": (
            sa.Column("attempt_id", sa.String(length=80), nullable=True),
            sa.Column("runtime_id", sa.String(length=160), nullable=True),
            sa.Column("actor_id", sa.String(length=160), nullable=True),
            sa.Column(
                "access_generation",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
        ),
        "kernel_channel_requests": (
            sa.Column(
                "agent_connection_generation",
                sa.String(length=120),
                nullable=True,
            ),
        ),
        "sites": (
            sa.Column(
                "agent_connection_generation",
                sa.String(length=120),
                nullable=True,
            ),
            sa.Column(
                "cached_datasets",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'[]'"),
            ),
            sa.Column("total_cpu", sa.Float(), nullable=True),
            sa.Column("available_cpu", sa.Float(), nullable=True),
            sa.Column("total_memory_bytes", sa.BigInteger(), nullable=True),
            sa.Column("available_memory_bytes", sa.BigInteger(), nullable=True),
        ),
        "model_capabilities": (
            sa.Column(
                "runtime_fingerprint",
                sa.String(length=64),
                nullable=False,
                server_default=sa.text("''"),
            ),
            sa.Column(
                "replica_id",
                sa.String(length=160),
                nullable=False,
                server_default=sa.text("''"),
            ),
            sa.Column(
                "replica_gpu_uuids",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'[]'"),
            ),
        ),
        "provider_instances": (
            sa.Column(
                "shutdown_generation",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            sa.Column("shutdown_agent_session_id", sa.String(length=120), nullable=True),
            sa.Column(
                "shutdown_connection_generation",
                sa.String(length=120),
                nullable=True,
            ),
            sa.Column("drain_command_nonce", sa.String(length=120), nullable=True),
            sa.Column("drain_ack_nonce", sa.String(length=120), nullable=True),
            sa.Column("durability_command_nonce", sa.String(length=120), nullable=True),
            sa.Column("durability_ack_nonce", sa.String(length=120), nullable=True),
        ),
    }


def _create_kernel_channel_requests() -> None:
    op.create_table(
        "kernel_channel_requests",
        sa.Column("id", sa.String(length=120), nullable=False),
        sa.Column("session_id", sa.String(length=80), nullable=False),
        sa.Column("request_id", sa.String(length=200), nullable=False),
        sa.Column("request_digest", sa.LargeBinary(length=32), nullable=False),
        sa.Column("state", sa.String(length=20), nullable=False),
        sa.Column("execution_count", sa.Integer(), nullable=False),
        sa.Column("agent_session_id", sa.String(length=120), nullable=True),
        sa.Column("agent_connection_generation", sa.String(length=120), nullable=True),
        sa.Column("result_digest", sa.LargeBinary(length=32), nullable=True),
        sa.Column("error_code", sa.String(length=80), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["kernel_sessions.id"],
            name="fk_kernel_channel_requests_session_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "session_id", "request_id", name="uq_kernel_channel_request"
        ),
    )


def _assert_original_shape(inspector: sa.Inspector) -> None:
    repair_table = "kernel_channel_requests"
    original_tables = set(_EXPECTED_COLUMNS) - {repair_table}
    actual_tables = set(inspector.get_table_names())
    missing_tables = sorted(original_tables - actual_tables)
    if missing_tables:
        raise RuntimeError(
            "Topodrome schema repair refused: missing original tables: "
            + ", ".join(missing_tables)
        )

    for table, expected in _EXPECTED_COLUMNS.items():
        if table not in actual_tables:
            continue
        actual = _column_names(inspector, table)
        unrecoverable = sorted(expected - actual - _REPAIRABLE_COLUMNS.get(table, set()))
        if unrecoverable:
            raise RuntimeError(
                f"Topodrome schema repair refused: {table} is missing original columns: "
                + ", ".join(unrecoverable)
            )

        primary_key = tuple(
            str(column) for column in inspector.get_pk_constraint(table)["constrained_columns"]
        )
        if primary_key != _PRIMARY_KEYS[table]:
            raise RuntimeError(
                f"Topodrome schema repair refused: {table} has primary key {primary_key!r}; "
                f"expected {_PRIMARY_KEYS[table]!r}"
            )

        by_name = {str(column["name"]): column for column in inspector.get_columns(table)}
        for name in expected & actual:
            expected_nullable = name in _NULLABLE_COLUMNS[table]
            if bool(by_name[name]["nullable"]) != expected_nullable:
                raise RuntimeError(
                    f"Topodrome schema repair refused: {table}.{name} has incompatible "
                    "nullability"
                )
            expected_type = _expected_type(table, name)
            if not _type_is_compatible(by_name[name]["type"], expected_type):
                raise RuntimeError(
                    f"Topodrome schema repair refused: {table}.{name} has incompatible "
                    f"SQL type {by_name[name]['type']!s}"
                )


def _add_missing_constraints(bind: sa.Connection) -> None:
    for table in _EXPECTED_COLUMNS:
        inspector = sa.inspect(bind)
        actual_unique = _unique_signatures(inspector, table)
        actual_foreign = _foreign_key_signatures(inspector, table)
        expected_foreign_by_columns: dict[
            tuple[str, ...],
            tuple[tuple[str, ...], str, tuple[str, ...], str],
        ] = {
            constraint[1]: (constraint[1], constraint[2], constraint[3], constraint[4])
            for constraint in _FOREIGN_KEYS.get(table, ())
        }
        for actual_constraint in actual_foreign:
            local_columns = actual_constraint[0]
            expected_constraint = expected_foreign_by_columns.get(local_columns)
            if expected_constraint is not None and actual_constraint != expected_constraint:
                raise RuntimeError(
                    f"Topodrome schema repair refused: {table} has a conflicting foreign "
                    f"key on {local_columns!r}"
                )
        missing_unique = [
            constraint
            for constraint in _UNIQUE_CONSTRAINTS.get(table, ())
            if constraint[1] not in actual_unique
        ]
        missing_foreign = [
            constraint
            for constraint in _FOREIGN_KEYS.get(table, ())
            if (constraint[1], constraint[2], constraint[3], constraint[4])
            not in actual_foreign
        ]
        if not missing_unique and not missing_foreign:
            continue
        with op.batch_alter_table(table) as batch:
            for name, columns in missing_unique:
                batch.create_unique_constraint(name, list(columns))
            for name, columns, remote_table, remote_columns, ondelete in missing_foreign:
                batch.create_foreign_key(
                    name,
                    remote_table,
                    list(columns),
                    list(remote_columns),
                    ondelete=ondelete,
                )


def _add_missing_indexes(bind: sa.Connection) -> None:
    for table, expected_indexes in _INDEXES.items():
        inspector = sa.inspect(bind)
        actual = {
            str(index["name"]): index
            for index in inspector.get_indexes(table)
            if index.get("name") is not None
        }
        for name, columns, unique, predicate in expected_indexes:
            existing = actual.get(name)
            if existing is not None:
                actual_columns = tuple(str(column) for column in existing["column_names"])
                actual_predicate = _index_predicate(existing, bind.dialect.name)
                expected_predicate = _expected_index_predicate(
                    predicate,
                    bind.dialect.name,
                )
                if (
                    actual_columns != columns
                    or bool(existing["unique"]) != unique
                    or actual_predicate != expected_predicate
                ):
                    raise RuntimeError(
                        f"Topodrome schema repair refused: index {name} has an "
                        "incompatible definition"
                    )
                continue
            dialect_options: dict[str, Any] = {}
            if predicate is not None:
                dialect_options = {
                    "postgresql_where": sa.text(predicate),
                    "sqlite_where": sa.text(f"{predicate} = 1"),
                }
            op.create_index(
                name,
                table,
                list(columns),
                unique=unique,
                **dialect_options,
            )


def _assert_head_parity(bind: sa.Connection) -> None:
    inspector = sa.inspect(bind)
    missing_tables = sorted(set(_EXPECTED_COLUMNS) - set(inspector.get_table_names()))
    if missing_tables:
        raise RuntimeError(
            "Topodrome schema parity failed: missing tables: " + ", ".join(missing_tables)
        )

    problems: list[str] = []
    for table, expected_columns in _EXPECTED_COLUMNS.items():
        columns_by_name = {
            str(column["name"]): column for column in inspector.get_columns(table)
        }
        if missing := sorted(expected_columns - set(columns_by_name)):
            problems.append(f"{table} missing columns {missing!r}")
        for name in expected_columns & set(columns_by_name):
            column = columns_by_name[name]
            if bool(column["nullable"]) != (name in _NULLABLE_COLUMNS[table]):
                problems.append(f"{table}.{name} has incompatible nullability")
            if not _type_is_compatible(column["type"], _expected_type(table, name)):
                problems.append(f"{table}.{name} has incompatible SQL type {column['type']!s}")

        primary_key = tuple(
            str(column) for column in inspector.get_pk_constraint(table)["constrained_columns"]
        )
        if primary_key != _PRIMARY_KEYS[table]:
            problems.append(f"{table} has primary key {primary_key!r}")

        actual_unique = _unique_signatures(inspector, table)
        for _name, columns in _UNIQUE_CONSTRAINTS.get(table, ()):
            if columns not in actual_unique:
                problems.append(f"{table} missing unique constraint on {columns!r}")

        actual_foreign = _foreign_key_signatures(inspector, table)
        for _name, columns, remote_table, remote_columns, ondelete in _FOREIGN_KEYS.get(
            table, ()
        ):
            signature = (columns, remote_table, remote_columns, ondelete)
            if signature not in actual_foreign:
                problems.append(f"{table} missing foreign key {signature!r}")

        actual_indexes = {
            str(index["name"]): (
                tuple(str(column) for column in index["column_names"]),
                bool(index["unique"]),
                _index_predicate(index, bind.dialect.name),
            )
            for index in inspector.get_indexes(table)
            if index.get("name") is not None
        }
        for name, columns, unique, predicate in _INDEXES.get(table, ()):
            expected_index = (
                columns,
                unique,
                _expected_index_predicate(predicate, bind.dialect.name),
            )
            if actual_indexes.get(name) != expected_index:
                problems.append(f"{table} missing or incompatible index {name}")

    if bind.dialect.name == "sqlite":
        violations = list(bind.exec_driver_sql("PRAGMA foreign_key_check"))
        if violations:
            problems.append(f"foreign-key violations: {violations!r}")

    if problems:
        raise RuntimeError("Topodrome schema parity failed: " + "; ".join(problems))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    _assert_original_shape(inspector)

    if "kernel_channel_requests" not in set(inspector.get_table_names()):
        _create_kernel_channel_requests()

    inspector = sa.inspect(bind)
    for table, columns in _repair_columns().items():
        actual = _column_names(inspector, table)
        for column in columns:
            if column.name not in actual:
                op.add_column(table, column)

    _add_missing_constraints(bind)
    _add_missing_indexes(bind)
    _assert_head_parity(bind)


def downgrade() -> None:
    # This revision restores objects that revision 0008 already claimed were
    # present.  Removing them would recreate the invalid state it repairs.
    pass
