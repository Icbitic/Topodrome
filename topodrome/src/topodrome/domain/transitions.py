from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TypeVar

from topodrome.domain.enums import (
    ActorState,
    ArtifactState,
    AttemptState,
    JobState,
    KernelState,
    LeaseState,
    ProviderInstanceState,
)
from topodrome.domain.errors import InvalidTransitionError

E = TypeVar("E", bound=StrEnum)

JOB_TRANSITIONS: Mapping[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.ASSIGNING, JobState.CANCELING}),
    JobState.ASSIGNING: frozenset(
        {JobState.RUNNING, JobState.QUEUED, JobState.CANCELING, JobState.FAILED}
    ),
    JobState.RUNNING: frozenset(
        {JobState.SUCCEEDED, JobState.RETRY_WAIT, JobState.CANCELING, JobState.FAILED}
    ),
    JobState.RETRY_WAIT: frozenset({JobState.QUEUED, JobState.CANCELING, JobState.FAILED}),
    JobState.CANCELING: frozenset({JobState.CANCELED}),
    JobState.SUCCEEDED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELED: frozenset(),
}

ATTEMPT_TRANSITIONS: Mapping[AttemptState, frozenset[AttemptState]] = {
    AttemptState.OFFERED: frozenset(
        {AttemptState.ACCEPTED, AttemptState.REJECTED, AttemptState.CANCELED, AttemptState.LOST}
    ),
    AttemptState.ACCEPTED: frozenset(
        {
            AttemptState.STARTING,
            AttemptState.PREEMPT_REQUESTED,
            AttemptState.CANCELED,
            AttemptState.LOST,
        }
    ),
    AttemptState.STARTING: frozenset(
        {
            AttemptState.RUNNING,
            AttemptState.PREEMPT_REQUESTED,
            AttemptState.FAILED,
            AttemptState.CANCELED,
            AttemptState.LOST,
        }
    ),
    AttemptState.RUNNING: frozenset(
        {
            AttemptState.COMMITTING,
            AttemptState.PREEMPT_REQUESTED,
            AttemptState.FAILED,
            AttemptState.CANCELED,
            AttemptState.LOST,
        }
    ),
    AttemptState.PREEMPT_REQUESTED: frozenset(
        {AttemptState.CHECKPOINTING, AttemptState.PREEMPTED, AttemptState.FAILED}
    ),
    AttemptState.CHECKPOINTING: frozenset({AttemptState.PREEMPTED, AttemptState.FAILED}),
    AttemptState.COMMITTING: frozenset(
        {AttemptState.SUCCEEDED, AttemptState.FAILED, AttemptState.CANCELED, AttemptState.LOST}
    ),
    AttemptState.SUCCEEDED: frozenset(),
    AttemptState.REJECTED: frozenset(),
    AttemptState.PREEMPTED: frozenset(),
    AttemptState.FAILED: frozenset(),
    AttemptState.CANCELED: frozenset(),
    AttemptState.LOST: frozenset(),
}

LEASE_TRANSITIONS: Mapping[LeaseState, frozenset[LeaseState]] = {
    LeaseState.OFFERED: frozenset({LeaseState.ACCEPTED, LeaseState.REJECTED, LeaseState.EXPIRED}),
    LeaseState.ACCEPTED: frozenset(
        {LeaseState.ACTIVE, LeaseState.RELEASED, LeaseState.EXPIRED, LeaseState.LOST}
    ),
    LeaseState.ACTIVE: frozenset({LeaseState.RELEASING, LeaseState.LOST}),
    LeaseState.RELEASING: frozenset({LeaseState.RELEASED, LeaseState.LOST}),
    LeaseState.RELEASED: frozenset(),
    LeaseState.REJECTED: frozenset(),
    LeaseState.EXPIRED: frozenset(),
    LeaseState.LOST: frozenset(),
}

ACTOR_TRANSITIONS: Mapping[ActorState, frozenset[ActorState]] = {
    ActorState.LOADING: frozenset({ActorState.IDLE, ActorState.DIRTY, ActorState.STOPPING}),
    ActorState.IDLE: frozenset({ActorState.BUSY, ActorState.EVICTING, ActorState.DIRTY}),
    ActorState.BUSY: frozenset({ActorState.IDLE, ActorState.DIRTY, ActorState.STOPPING}),
    ActorState.EVICTING: frozenset({ActorState.STOPPING}),
    ActorState.DIRTY: frozenset({ActorState.STOPPING}),
    ActorState.STOPPING: frozenset({ActorState.STOPPED}),
    ActorState.STOPPED: frozenset(),
}

ARTIFACT_TRANSITIONS: Mapping[ArtifactState, frozenset[ArtifactState]] = {
    ArtifactState.STAGING: frozenset({ArtifactState.FLUSHING, ArtifactState.FAILED}),
    ArtifactState.FLUSHING: frozenset({ArtifactState.VERIFIED, ArtifactState.FAILED}),
    ArtifactState.VERIFIED: frozenset({ArtifactState.COMMITTED, ArtifactState.FAILED}),
    ArtifactState.COMMITTED: frozenset(),
    ArtifactState.FAILED: frozenset(),
}

KERNEL_TRANSITIONS: Mapping[KernelState, frozenset[KernelState]] = {
    KernelState.REQUESTED: frozenset(
        {KernelState.STARTING, KernelState.STOPPING, KernelState.STOPPED, KernelState.LOST}
    ),
    KernelState.STARTING: frozenset(
        {KernelState.RUNNING, KernelState.STOPPING, KernelState.STOPPED, KernelState.LOST}
    ),
    KernelState.RUNNING: frozenset({KernelState.IDLE, KernelState.STOPPING, KernelState.LOST}),
    KernelState.IDLE: frozenset({KernelState.RUNNING, KernelState.STOPPING, KernelState.LOST}),
    KernelState.STOPPING: frozenset({KernelState.STOPPED, KernelState.LOST}),
    KernelState.STOPPED: frozenset(),
    KernelState.LOST: frozenset(),
}

PROVIDER_INSTANCE_TRANSITIONS: Mapping[ProviderInstanceState, frozenset[ProviderInstanceState]] = {
    ProviderInstanceState.PROVISIONING: frozenset(
        {ProviderInstanceState.RUNNING, ProviderInstanceState.ERROR}
    ),
    ProviderInstanceState.RUNNING: frozenset(
        {ProviderInstanceState.DRAINING, ProviderInstanceState.ERROR}
    ),
    ProviderInstanceState.DRAINING: frozenset(
        {ProviderInstanceState.STOPPING, ProviderInstanceState.RUNNING, ProviderInstanceState.ERROR}
    ),
    ProviderInstanceState.STOPPING: frozenset(
        {ProviderInstanceState.STOPPED, ProviderInstanceState.ERROR}
    ),
    ProviderInstanceState.STOPPED: frozenset(
        {ProviderInstanceState.STARTING, ProviderInstanceState.DURABILITY_CONFIRMED}
    ),
    ProviderInstanceState.STARTING: frozenset(
        {ProviderInstanceState.RUNNING, ProviderInstanceState.STARTING_FAILED}
    ),
    ProviderInstanceState.STARTING_FAILED: frozenset({ProviderInstanceState.DURABILITY_CONFIRMED}),
    ProviderInstanceState.DURABILITY_CONFIRMED: frozenset({ProviderInstanceState.RELEASING}),
    ProviderInstanceState.RELEASING: frozenset(
        {ProviderInstanceState.RELEASED, ProviderInstanceState.ERROR}
    ),
    ProviderInstanceState.RELEASED: frozenset(),
    ProviderInstanceState.ERROR: frozenset(
        {ProviderInstanceState.DRAINING, ProviderInstanceState.STOPPING}
    ),
}


def require_transition(current: E, target: E, allowed: Mapping[E, frozenset[E]]) -> None:
    if target not in allowed.get(current, frozenset()):
        raise InvalidTransitionError(f"cannot transition {current.value} to {target.value}")
