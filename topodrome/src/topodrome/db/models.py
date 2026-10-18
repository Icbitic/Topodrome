from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    false,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from topodrome.domain.enums import (
    ActorState,
    ArtifactState,
    AttemptState,
    GPUReservationState,
    JobState,
    KernelMode,
    KernelState,
    LeaseState,
    ModelAvailability,
    ProviderAction,
    ProviderInstanceState,
    ProviderOperationState,
    SiteKind,
    SiteState,
)
from topodrome.domain.specs import utc_now


class Base(DeclarativeBase):
    pass


def enum_column(enum_type: type[Any]) -> Enum:
    return Enum(
        enum_type,
        native_enum=False,
        values_callable=lambda members: [item.value for item in members],
    )


class JobArrayRow(Base):
    __tablename__ = "job_arrays"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    owner_username: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str | None] = mapped_column(String(160))
    max_parallel: Mapped[int] = mapped_column(Integer)
    held: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    idempotency_key: Mapped[str | None] = mapped_column(String(200), unique=True)
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    owner_username: Mapped[str] = mapped_column(
        String(64), default="@operator", server_default="@operator", index=True
    )
    state: Mapped[JobState] = mapped_column(enum_column(JobState), index=True)
    spec: Mapped[dict[str, Any]] = mapped_column(JSON)
    idempotency_key: Mapped[str | None] = mapped_column(String(200), unique=True)
    array_id: Mapped[str | None] = mapped_column(ForeignKey("job_arrays.id"), index=True)
    array_index: Mapped[int | None] = mapped_column(Integer)
    eligible_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    blocked_reason: Mapped[str | None] = mapped_column(String(200))
    phase: Mapped[str] = mapped_column(String(32), default="queued", server_default="queued")
    phase_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    phase_seconds: Mapped[dict[str, float]] = mapped_column(JSON, default=dict, server_default="{}")
    priority: Mapped[int] = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    retry_not_before: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    attempts: Mapped[list[AttemptRow]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    __table_args__ = (UniqueConstraint("array_id", "array_index", name="uq_array_element"),)


class JobStateEventRow(Base):
    __tablename__ = "job_state_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    from_state: Mapped[str | None] = mapped_column(String(40))
    to_state: Mapped[str] = mapped_column(String(40))
    attempt_id: Mapped[str | None] = mapped_column(String(80))
    reason_code: Mapped[str | None] = mapped_column(String(100))
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )


class InputStageRow(Base):
    """Durable transfer intent; never grants physical GPU ownership."""

    __tablename__ = "input_stages"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    stage_id: Mapped[str] = mapped_column(String(80), unique=True)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id"), index=True)
    state: Mapped[str] = mapped_column(String(32), default="downloading")
    progress: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    placement_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, server_default="{}")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SchedulingMeasurementRow(Base):
    """Bounded EWMA observations, without job contents or credentials."""

    __tablename__ = "scheduling_measurements"
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id"), primary_key=True)
    metric: Mapped[str] = mapped_column(String(32), primary_key=True)
    context_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[float] = mapped_column(Float)
    samples: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SiteRow(Base):
    __tablename__ = "sites"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[SiteKind] = mapped_column(enum_column(SiteKind))
    state: Mapped[SiteState] = mapped_column(enum_column(SiteState), index=True)
    agent_session_id: Mapped[str | None] = mapped_column(String(120))
    agent_connection_generation: Mapped[str | None] = mapped_column(String(120))
    last_sequence: Mapped[int] = mapped_column(Integer, default=-1)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    storage_available: Mapped[bool] = mapped_column(Boolean, default=False)
    storage_authoritative: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false()
    )
    supported_workloads: Mapped[list[str]] = mapped_column(JSON, default=list)
    cached_assets: Mapped[list[str]] = mapped_column(JSON, default=list)
    queue_depth: Mapped[int] = mapped_column(Integer, default=0)
    reliability: Mapped[float] = mapped_column(Float, default=1.0)
    gpu_safety_margin_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    total_cpu: Mapped[float | None] = mapped_column(Float)
    available_cpu: Mapped[float | None] = mapped_column(Float)
    total_memory_bytes: Mapped[int | None] = mapped_column(BigInteger)
    available_memory_bytes: Mapped[int | None] = mapped_column(BigInteger)
    paid: Mapped[bool] = mapped_column(Boolean, default=False)
    region: Mapped[str | None] = mapped_column(String(120))
    provider_instance_id: Mapped[str | None] = mapped_column(String(120), index=True)
    job_running: Mapped[bool] = mapped_column(Boolean, default=False)
    assignment_queued_locally: Mapped[bool] = mapped_column(Boolean, default=False)
    interactive_kernel_active: Mapped[bool] = mapped_column(Boolean, default=False)
    artifact_flush_pending: Mapped[bool] = mapped_column(Boolean, default=False)
    staging_operation_active: Mapped[bool] = mapped_column(Boolean, default=False)
    drain_ready: Mapped[bool] = mapped_column(Boolean, default=False)
    durability_ready: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    gpus: Mapped[list[GPURow]] = relationship(back_populates="site", cascade="all, delete-orphan")
    models: Mapped[list[ModelCapabilityRow]] = relationship(
        back_populates="site", cascade="all, delete-orphan"
    )


class GPURow(Base):
    __tablename__ = "gpus"

    site_id: Mapped[str] = mapped_column(
        ForeignKey("sites.id", ondelete="CASCADE"), primary_key=True
    )
    uuid: Mapped[str] = mapped_column(String(200), primary_key=True)
    index: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(200))
    total_memory_bytes: Mapped[int] = mapped_column(BigInteger)
    free_memory_bytes: Mapped[int] = mapped_column(BigInteger)
    utilization_percent: Mapped[float] = mapped_column(Float)
    temperature_c: Mapped[float | None] = mapped_column(Float)
    foreign_process_count: Mapped[int] = mapped_column(Integer, default=0)
    foreign_memory_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    externally_reserved: Mapped[bool] = mapped_column(Boolean, default=False)
    healthy: Mapped[bool] = mapped_column(Boolean, default=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    site: Mapped[SiteRow] = relationship(back_populates="gpus")

    __table_args__ = (UniqueConstraint("site_id", "index", name="uq_gpu_site_index"),)


class GPUReservationRow(Base):
    """Operator intent for one exact physical GPU, independent of heartbeat rows."""

    __tablename__ = "gpu_reservations"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    # Do not reference GPURow: heartbeat ingestion replaces those rows.  The
    # target is validated against the current inventory when this intent is
    # created, then retained as an audit record even if the site disappears.
    site_id: Mapped[str] = mapped_column(String(120), index=True)
    gpu_uuid: Mapped[str] = mapped_column(String(200))
    gpu_index: Mapped[int] = mapped_column(Integer)
    requested_by: Mapped[str] = mapped_column(String(200))
    reason: Mapped[str] = mapped_column(Text)
    force: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    duration_seconds: Mapped[int] = mapped_column(Integer)
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    idempotency_key: Mapped[str | None] = mapped_column(String(200))
    state: Mapped[GPUReservationState] = mapped_column(enum_column(GPUReservationState), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    canceled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    eviction_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    __table_args__ = (
        UniqueConstraint(
            "idempotency_key",
            name="uq_gpu_reservations_idempotency_key",
        ),
        Index(
            "uq_active_gpu_reservation",
            "site_id",
            "gpu_uuid",
            unique=True,
            postgresql_where=text("state IN ('pending', 'ready')"),
            sqlite_where=text("state IN ('pending', 'ready')"),
        ),
    )


class ModelCapabilityRow(Base):
    __tablename__ = "model_capabilities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    repo_id: Mapped[str] = mapped_column(String(200), index=True)
    revision: Mapped[str] = mapped_column(String(200))
    subfolder: Mapped[str] = mapped_column(String(500), default="", server_default="")
    backend: Mapped[str] = mapped_column(String(100))
    backend_version: Mapped[str] = mapped_column(String(100))
    dtype: Mapped[str] = mapped_column(String(50))
    quantization: Mapped[str] = mapped_column(String(100), default="")
    adapter: Mapped[str] = mapped_column(String(500), default="")
    availability: Mapped[ModelAvailability] = mapped_column(
        enum_column(ModelAvailability), index=True
    )
    gpu_uuid: Mapped[str] = mapped_column(String(200), default="")
    runtime_fingerprint: Mapped[str] = mapped_column(String(64), default="")
    replica_id: Mapped[str] = mapped_column(String(160), default="")
    replica_gpu_uuids: Mapped[list[str]] = mapped_column(JSON, default=list)
    replica_key_hash: Mapped[str] = mapped_column(String(64), default="")
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    site: Mapped[SiteRow] = relationship(back_populates="models")

    __table_args__ = (
        UniqueConstraint(
            "site_id",
            "repo_id",
            "revision",
            "subfolder",
            "backend",
            "backend_version",
            "dtype",
            "quantization",
            "adapter",
            "gpu_uuid",
            name="uq_model_capability_identity",
        ),
    )


class ModelActorRow(Base):
    __tablename__ = "model_actors"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    gpu_uuid: Mapped[str] = mapped_column(String(200))
    key_hash: Mapped[str] = mapped_column(String(64), index=True)
    key_data: Mapped[dict[str, Any]] = mapped_column(JSON)
    state: Mapped[ActorState] = mapped_column(enum_column(ActorState), index=True)
    active_allocation: Mapped[bool] = mapped_column(Boolean, default=True)
    owner_attempt_id: Mapped[str | None] = mapped_column(String(80))
    process_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_used_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    __table_args__ = (
        Index(
            "uq_active_actor_gpu",
            "site_id",
            "gpu_uuid",
            unique=True,
            postgresql_where=text("active_allocation"),
            sqlite_where=text("active_allocation = 1"),
        ),
    )


class AttemptRow(Base):
    __tablename__ = "attempts"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    state: Mapped[AttemptState] = mapped_column(enum_column(AttemptState), index=True)
    site_id: Mapped[str | None] = mapped_column(
        ForeignKey("sites.id", ondelete="SET NULL"), index=True
    )
    reason_code: Mapped[str | None] = mapped_column(String(100))
    forced_reservation_id: Mapped[str | None] = mapped_column(String(80))
    message: Mapped[str | None] = mapped_column(Text)
    checkpoint_ref: Mapped[str | None] = mapped_column(String(1000))
    restore_checkpoint_ref: Mapped[str | None] = mapped_column(String(1000))
    offer_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    fencing_token_digest: Mapped[str | None] = mapped_column(String(64))
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    compute_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    job: Mapped[JobRow] = relationship(back_populates="attempts")
    lease: Mapped[LeaseRow | None] = relationship(
        back_populates="attempt", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (UniqueConstraint("job_id", "number", name="uq_attempt_job_number"),)


class LeaseRow(Base):
    __tablename__ = "leases"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    attempt_id: Mapped[str] = mapped_column(
        ForeignKey("attempts.id", ondelete="CASCADE"), unique=True
    )
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    state: Mapped[LeaseState] = mapped_column(enum_column(LeaseState), index=True)
    fencing_token_digest: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    attempt: Mapped[AttemptRow] = relationship(back_populates="lease")
    claims: Mapped[list[LeaseGPUClaimRow]] = relationship(
        back_populates="lease", cascade="all, delete-orphan"
    )


class GPUFenceCounterRow(Base):
    __tablename__ = "gpu_fence_counters"

    site_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    gpu_uuid: Mapped[str] = mapped_column(String(200), primary_key=True)
    epoch: Mapped[int] = mapped_column(Integer, default=0)


class LeaseGPUClaimRow(Base):
    __tablename__ = "lease_gpu_claims"

    lease_id: Mapped[str] = mapped_column(
        ForeignKey("leases.id", ondelete="CASCADE"), primary_key=True
    )
    site_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    gpu_uuid: Mapped[str] = mapped_column(String(200), primary_key=True)
    epoch: Mapped[int] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    lease: Mapped[LeaseRow] = relationship(back_populates="claims")

    __table_args__ = (
        Index(
            "uq_active_lease_gpu",
            "site_id",
            "gpu_uuid",
            unique=True,
            postgresql_where=text("active"),
            sqlite_where=text("active = 1"),
        ),
    )


class ArtifactRow(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    attempt_id: Mapped[str] = mapped_column(
        ForeignKey("attempts.id", ondelete="CASCADE"), index=True
    )
    logical_path: Mapped[str] = mapped_column(String(1200))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    state: Mapped[ArtifactState] = mapped_column(enum_column(ArtifactState), index=True)
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("attempt_id", "logical_path", name="uq_artifact_attempt_path"),
    )


class KernelSessionRow(Base):
    __tablename__ = "kernel_sessions"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    owner_username: Mapped[str] = mapped_column(
        String(64), default="@operator", server_default="@operator", index=True
    )
    state: Mapped[KernelState] = mapped_column(enum_column(KernelState), index=True)
    mode: Mapped[KernelMode] = mapped_column(enum_column(KernelMode))
    request: Mapped[dict[str, Any]] = mapped_column(JSON)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"))
    attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("attempts.id", ondelete="SET NULL"), index=True
    )
    site_id: Mapped[str | None] = mapped_column(
        ForeignKey("sites.id", ondelete="SET NULL"), index=True
    )
    relay_id: Mapped[str | None] = mapped_column(String(120), unique=True)
    runtime_id: Mapped[str | None] = mapped_column(String(160))
    actor_id: Mapped[str | None] = mapped_column(String(160))
    access_generation: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    attempt: Mapped[AttemptRow | None] = relationship()


class KernelChannelRequestRow(Base):
    """Bounded durable idempotency record for one notebook execution request."""

    __tablename__ = "kernel_channel_requests"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("kernel_sessions.id", ondelete="CASCADE"), index=True
    )
    request_id: Mapped[str] = mapped_column(String(200))
    request_digest: Mapped[bytes] = mapped_column(LargeBinary(32))
    state: Mapped[str] = mapped_column(String(20), index=True)
    execution_count: Mapped[int] = mapped_column(Integer)
    agent_session_id: Mapped[str | None] = mapped_column(String(120))
    agent_connection_generation: Mapped[str | None] = mapped_column(String(120))
    result_digest: Mapped[bytes | None] = mapped_column(LargeBinary(32))
    error_code: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    __table_args__ = (
        UniqueConstraint("session_id", "request_id", name="uq_kernel_channel_request"),
    )


class ProviderInstanceRow(Base):
    __tablename__ = "provider_instances"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), index=True)
    external_id: Mapped[str | None] = mapped_column(String(120), unique=True)
    site_id: Mapped[str | None] = mapped_column(String(120), index=True)
    state: Mapped[ProviderInstanceState] = mapped_column(
        enum_column(ProviderInstanceState), index=True
    )
    region: Mapped[str | None] = mapped_column(String(120), index=True)
    gpu_spec: Mapped[str] = mapped_column(String(120))
    image_uuid: Mapped[str] = mapped_column(String(120))
    durable_state_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    shutdown_generation: Mapped[int] = mapped_column(Integer, default=0)
    shutdown_agent_session_id: Mapped[str | None] = mapped_column(String(120))
    shutdown_connection_generation: Mapped[str | None] = mapped_column(String(120))
    drain_command_nonce: Mapped[str | None] = mapped_column(String(120))
    drain_ack_nonce: Mapped[str | None] = mapped_column(String(120))
    durability_command_nonce: Mapped[str | None] = mapped_column(String(120))
    durability_ack_nonce: Mapped[str | None] = mapped_column(String(120))
    idle_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class ProviderOperationRow(Base):
    __tablename__ = "provider_operations"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), index=True)
    instance_id: Mapped[str | None] = mapped_column(
        ForeignKey("provider_instances.id", ondelete="SET NULL"), index=True
    )
    external_instance_id: Mapped[str | None] = mapped_column(String(120), index=True)
    action: Mapped[ProviderAction] = mapped_column(enum_column(ProviderAction))
    state: Mapped[ProviderOperationState] = mapped_column(
        enum_column(ProviderOperationState), index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True)
    request_redacted: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    external_request_id: Mapped[str | None] = mapped_column(String(200))
    result_external_instance_id: Mapped[str | None] = mapped_column(String(120))
    error_code: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class AgentEventAckRow(Base):
    __tablename__ = "agent_event_acks"

    message_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    site_id: Mapped[str] = mapped_column(String(120), index=True)
    agent_session_id: Mapped[str] = mapped_column(String(120))
    sequence: Mapped[int] = mapped_column(Integer)
    payload_digest: Mapped[bytes] = mapped_column(LargeBinary(32))
    acknowledged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    __table_args__ = (
        UniqueConstraint("site_id", "agent_session_id", "sequence", name="uq_agent_sequence"),
    )
