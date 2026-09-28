from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from topodrome.domain.specs import StrictModel, new_id, utc_now

AgentMessageKind = Literal[
    "hello",
    "heartbeat",
    "inventory",
    "capability_report",
    "assignment_accepted",
    "assignment_rejected",
    "lease_renewal",
    "attempt_event",
    "kernel_event",
    "kernel_result",
    "artifact_event",
    "log_chunk",
    "attempt_log_chunk",
    "drain_ready",
    "durability_ready",
    "command_ack",
    "input_stage",
    "recover_finalization",
    "attempt_log_complete",
]

SchedulerMessageKind = Literal[
    "ack",
    "offer_assignment",
    "request_checkpoint",
    "cancel_attempt",
    "evict_actor",
    "start_kernel",
    "kernel_execute",
    "kernel_interrupt",
    "stop_kernel",
    "begin_drain",
    "cancel_drain",
    "verify_durability",
    "stop_agent",
    "stage_inputs",
    "cancel_input_stage",
]


class AgentEnvelope(StrictModel):
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    site_id: str
    agent_session_id: str
    sequence: int = Field(ge=0)
    kind: AgentMessageKind
    payload: dict[str, Any] = Field(default_factory=dict)
    sent_at: datetime = Field(default_factory=utc_now)


class SchedulerEnvelope(StrictModel):
    message_id: str = Field(default_factory=lambda: new_id("cmd"))
    site_id: str
    kind: SchedulerMessageKind
    payload: dict[str, Any] = Field(default_factory=dict)
    sent_at: datetime = Field(default_factory=utc_now)
