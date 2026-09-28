"""Manual SSH shutdown activity and durable journal value types.

Historical action enum values remain readable in existing audit records.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from topodrome.providers.autodl.redaction import contains_sensitive_key


@dataclass(frozen=True, slots=True)
class IdleActivity:
    """The architecture's exact five-part definition of AutoDL idleness."""

    job_running: bool = False
    assignment_queued_locally: bool = False
    interactive_kernel_active: bool = False
    artifact_flush_pending: bool = False
    staging_operation_active: bool = False

    @property
    def is_idle(self) -> bool:
        return not any(
            (
                self.job_running,
                self.assignment_queued_locally,
                self.interactive_kernel_active,
                self.artifact_flush_pending,
                self.staging_operation_active,
            )
        )


class OperationKind(StrEnum):
    CREATE = "create"
    POWER_ON = "power_on"
    POWER_OFF = "power_off"
    RELEASE = "release"


class OperationState(StrEnum):
    INTENT_RECORDED = "intent_recorded"
    IN_FLIGHT = "in_flight"
    SUCCEEDED = "succeeded"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProviderOperationIntent:
    operation_id: str
    idempotency_key: str
    kind: OperationKind
    request: Mapping[str, object]
    instance_uuid: str | None = None
    instance_name: str | None = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.operation_id or not self.idempotency_key:
            raise ValueError("operation_id and idempotency_key are required")
        if contains_sensitive_key(self.request):
            raise ValueError("provider operation intents must not contain credentials")


@dataclass(frozen=True, slots=True)
class ProviderOperationRecord:
    intent: ProviderOperationIntent
    state: OperationState
    provider_request_id: str | None = None
    result_instance_uuid: str | None = None
    failure_code: str | None = None


@dataclass(frozen=True, slots=True)
class OperationClaim:
    record: ProviderOperationRecord
    created: bool
