"""Stable domain types shared by the scheduler, agents, and clients."""

from topodrome.domain.enums import (
    ActorState,
    ArtifactState,
    AttemptState,
    JobState,
    KernelMode,
    KernelState,
    LeaseState,
    ModelAvailability,
    Portability,
    SiteKind,
    SiteState,
    WorkloadKind,
)
from topodrome.domain.specs import (
    AssetMember,
    AssetRef,
    DistributedParallelism,
    JobSpec,
    ModelRequirement,
    ModelShardedParallelism,
    ResourceRequirements,
    SingleGPUParallelism,
)

__all__ = [
    "ActorState",
    "ArtifactState",
    "AssetMember",
    "AssetRef",
    "AttemptState",
    "DistributedParallelism",
    "JobSpec",
    "JobState",
    "KernelMode",
    "KernelState",
    "LeaseState",
    "ModelAvailability",
    "ModelRequirement",
    "ModelShardedParallelism",
    "Portability",
    "ResourceRequirements",
    "SingleGPUParallelism",
    "SiteKind",
    "SiteState",
    "WorkloadKind",
]
