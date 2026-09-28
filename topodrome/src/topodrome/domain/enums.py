from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    QUEUED = "queued"
    ASSIGNING = "assigning"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    CANCELING = "canceling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class AttemptState(StrEnum):
    OFFERED = "offered"
    ACCEPTED = "accepted"
    STARTING = "starting"
    RUNNING = "running"
    PREEMPT_REQUESTED = "preempt_requested"
    CHECKPOINTING = "checkpointing"
    COMMITTING = "committing"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    PREEMPTED = "preempted"
    FAILED = "failed"
    CANCELED = "canceled"
    LOST = "lost"


class LeaseState(StrEnum):
    OFFERED = "offered"
    ACCEPTED = "accepted"
    ACTIVE = "active"
    RELEASING = "releasing"
    RELEASED = "released"
    REJECTED = "rejected"
    EXPIRED = "expired"
    LOST = "lost"


class GPUReservationState(StrEnum):
    """Durable lifecycle of an operator-owned physical GPU reservation."""

    PENDING = "pending"
    READY = "ready"
    CANCELED = "canceled"
    EXPIRED = "expired"


class SiteState(StrEnum):
    PROVISIONING = "provisioning"
    CONNECTING = "connecting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    STARTING = "starting"
    ERROR = "error"
    OFFLINE = "offline"


class SiteKind(StrEnum):
    LAB = "lab"
    AUTODL = "autodl"
    JUPYTER = "jupyter"


class ActorState(StrEnum):
    LOADING = "loading"
    IDLE = "idle"
    BUSY = "busy"
    EVICTING = "evicting"
    DIRTY = "dirty"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ArtifactState(StrEnum):
    STAGING = "staging"
    FLUSHING = "flushing"
    VERIFIED = "verified"
    COMMITTED = "committed"
    FAILED = "failed"


class KernelState(StrEnum):
    REQUESTED = "requested"
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    STOPPING = "stopping"
    STOPPED = "stopped"
    LOST = "lost"


class KernelMode(StrEnum):
    SERVICE = "service"
    RAW_MODEL = "raw-model"


class WorkloadKind(StrEnum):
    BATCH = "batch"
    INTERACTIVE = "interactive"
    ADAPTER_TRAINING = "adapter_training"


class Portability(StrEnum):
    PORTABLE = "portable"
    SITE_AFFINE = "site-affine"
    REGION_AFFINE = "region-affine"
    RECOMPUTABLE = "recomputable"


class ModelAvailability(StrEnum):
    WARM = "warm"
    CACHED = "cached"
    OBTAINABLE = "obtainable"


class ProviderInstanceState(StrEnum):
    PROVISIONING = "provisioning"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"
    STARTING = "starting"
    STARTING_FAILED = "starting_failed"
    DURABILITY_CONFIRMED = "durability_confirmed"
    RELEASING = "releasing"
    RELEASED = "released"
    ERROR = "error"


class ProviderOperationState(StrEnum):
    INTENT = "intent"
    CALLING = "calling"
    APPLIED = "applied"
    RECONCILING = "reconciling"
    FAILED = "failed"


class ProviderAction(StrEnum):
    CREATE = "create"
    POWER_ON = "power_on"
    POWER_OFF = "power_off"
    RELEASE = "release"
