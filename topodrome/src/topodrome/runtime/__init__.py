"""Topodrome's site-local model execution runtime.

Importing this package has no heavyweight framework side effects.  In
particular, Ray and model backends are loaded only by their explicit adapters.
"""

from .actor import (
    ActorCleanupError,
    ActorDescription,
    ActorDisposalError,
    ActorError,
    ActorManager,
    ActorState,
    ActorUnavailableError,
    IncompatibleActorBusyError,
    LoadedModel,
    LocalModelActor,
    ModelLoader,
)
from .checkpoint import CheckpointRecord, CheckpointRestore, PreemptionToken
from .cleanup import CleanupError, CleanupFailure, CleanupHandle, CleanupStack
from .context import ArtifactRegistration, ExperimentContext
from .experiment import Experiment, invoke_experiment, resolve_entrypoint
from .loaders import (
    HuggingFaceModelAcquirer,
    ModelAcquisitionError,
    ModelLoadError,
    SecretProvider,
    TransformersLoader,
)
from .model_key import ActorKey, ModelKey
from .model_payload import (
    ModelPayloadVerificationError,
    VerifiedModelPayload,
    record_huggingface_local_dir_manifest,
    verify_transformers_payload,
)
from .model_registry import (
    AmbiguousModelError,
    DuplicateModelError,
    ModelAvailability,
    ModelCapability,
    ModelRegistration,
    ModelRegistry,
    ModelRegistryError,
    UnknownModelError,
)
from .ray_actor import (
    DetachedRayActorAdapter,
    RayActorError,
    RayActorHandle,
    RayActorSpec,
    RayActorTimeoutError,
    RayDisposalError,
    RayOperationTimeouts,
    RayUnavailableError,
    RayWorkerGroupHandle,
    RayWorkerGroupSpec,
    ray_gpu_resource_name,
)
from .service import (
    AsyncRuntimeExecutionService,
    ExecutionTransport,
    PreparedActor,
    RuntimeExecutionError,
    RuntimeExecutionService,
)

__all__ = [
    "ActorCleanupError",
    "ActorDescription",
    "ActorDisposalError",
    "ActorError",
    "ActorKey",
    "ActorManager",
    "ActorState",
    "ActorUnavailableError",
    "AmbiguousModelError",
    "ArtifactRegistration",
    "AsyncRuntimeExecutionService",
    "CheckpointRecord",
    "CheckpointRestore",
    "CleanupError",
    "CleanupFailure",
    "CleanupHandle",
    "CleanupStack",
    "DetachedRayActorAdapter",
    "DuplicateModelError",
    "ExecutionTransport",
    "Experiment",
    "ExperimentContext",
    "HuggingFaceModelAcquirer",
    "IncompatibleActorBusyError",
    "LoadedModel",
    "LocalModelActor",
    "ModelAcquisitionError",
    "ModelAvailability",
    "ModelCapability",
    "ModelKey",
    "ModelLoadError",
    "ModelLoader",
    "ModelPayloadVerificationError",
    "ModelRegistration",
    "ModelRegistry",
    "ModelRegistryError",
    "PreemptionToken",
    "PreparedActor",
    "RayActorError",
    "RayActorHandle",
    "RayActorSpec",
    "RayActorTimeoutError",
    "RayDisposalError",
    "RayOperationTimeouts",
    "RayUnavailableError",
    "RayWorkerGroupHandle",
    "RayWorkerGroupSpec",
    "RuntimeExecutionError",
    "RuntimeExecutionService",
    "SecretProvider",
    "TransformersLoader",
    "UnknownModelError",
    "VerifiedModelPayload",
    "invoke_experiment",
    "ray_gpu_resource_name",
    "record_huggingface_local_dir_manifest",
    "resolve_entrypoint",
    "verify_transformers_payload",
]
