from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import unquote
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from topodrome.domain.enums import (
    AttemptState,
    GPUReservationState,
    JobState,
    KernelMode,
    KernelState,
    ModelAvailability,
    Portability,
    SiteKind,
    SiteState,
    WorkloadKind,
)
from topodrome.runtime_references import (
    validate_job_working_dir_reference,
    validate_pinned_runtime_requirements,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AssetMember(StrictModel):
    """One exact immutable file in a scheduler-pinned global asset."""

    logical_path: str = Field(min_length=1, max_length=2000)
    relative_path: str = Field(min_length=1, max_length=1200)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("logical_path", "relative_path")
    @classmethod
    def paths_are_logical(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or value in {"", "."}
            or value != path.as_posix()
            or "\\" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("asset member paths must be normalized relative logical paths")
        return value


class AssetRef(StrictModel):
    """A global asset reference, optionally resolved to an immutable file set.

    Users submit only ``source_ref`` (usually with the compact string form).
    Before dispatch the scheduler replaces it with the complete content pin.
    Site agents accept only that pinned form and never resolve a mutable name.
    """

    source_ref: str = Field(min_length=1, max_length=2000)
    kind: Literal["file", "directory"] | None = None
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    total_bytes: int | None = Field(default=None, ge=0)
    members: tuple[AssetMember, ...] = ()

    @classmethod
    def parse(cls, value: str | AssetRef | dict[str, Any]) -> AssetRef:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(source_ref=value)
        return cls.model_validate(value)

    @field_validator("source_ref")
    @classmethod
    def source_reference_is_safe(cls, value: str) -> str:
        if _contains_embedded_credential(value):
            raise ValueError("asset references must not contain credentials")
        if any(character.isspace() or ord(character) < 32 for character in value):
            raise ValueError("asset references must not contain whitespace or control characters")
        return value

    @model_validator(mode="after")
    def pin_is_complete(self) -> AssetRef:
        has_pin = (
            self.kind is not None
            or self.digest is not None
            or self.total_bytes is not None
            or bool(self.members)
        )
        if not has_pin:
            return self
        if self.kind is None or self.digest is None or self.total_bytes is None or not self.members:
            raise ValueError("pinned assets require kind, digest, total_bytes, and members")
        if self.kind == "file" and len(self.members) != 1:
            raise ValueError("file assets require exactly one member")
        if self.total_bytes != sum(member.size_bytes for member in self.members):
            raise ValueError("asset total_bytes must equal the sum of member sizes")
        logical_paths = tuple(member.logical_path for member in self.members)
        relative_paths = tuple(member.relative_path for member in self.members)
        if len(set(logical_paths)) != len(logical_paths):
            raise ValueError("asset member logical paths must be unique")
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("asset member relative paths must be unique")
        return self

    @property
    def pinned(self) -> bool:
        return self.digest is not None

    @property
    def cache_key(self) -> str | None:
        """Exact site-cache identity for this immutable pin.

        The content digest deliberately ignores global storage coordinates so
        identical trees have the same content identity.  A reusable cache
        view also depends on its file/directory semantics and exact source
        paths, so placement telemetry uses this stricter key.
        """

        if not self.pinned:
            return None
        assert self.kind is not None
        assert self.digest is not None
        assert self.total_bytes is not None
        payload = {
            "schema_version": 1,
            "kind": self.kind,
            "digest": self.digest,
            "total_bytes": self.total_bytes,
            "members": [
                {
                    "logical_path": member.logical_path,
                    "relative_path": member.relative_path,
                    "size_bytes": member.size_bytes,
                    "sha256": member.sha256,
                }
                for member in sorted(self.members, key=lambda item: item.relative_path)
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class SingleGPUParallelism(StrictModel):
    """One model process bound to one physical GPU slot."""

    mode: Literal["single"] = "single"
    degree: Literal[1] = 1
    engine: Literal["none"] = "none"


class DistributedParallelism(StrictModel):
    """One private Ray worker per GPU for collective-aware experiment code."""

    mode: Literal["distributed"]
    degree: int = Field(ge=2, le=64)
    engine: Literal["ray_worker_group"] = "ray_worker_group"


class ModelShardedParallelism(StrictModel):
    """One reusable model replica sharded across an ordered GPU gang."""

    mode: Literal["model_sharded"]
    degree: int = Field(ge=2, le=64)
    engine: Literal["transformers_device_map"] = "transformers_device_map"
    strategy: Literal["balanced", "balanced_low_0", "sequential"] = "balanced"
    max_memory_gb_per_gpu: float = Field(gt=0)


ModelParallelism = Annotated[
    SingleGPUParallelism | DistributedParallelism | ModelShardedParallelism,
    Field(discriminator="mode"),
]


class ModelRequirement(StrictModel):
    alias: str = Field(min_length=1, max_length=200)
    revision: str = Field(default="configured", min_length=1, max_length=200)
    backend: str = Field(default="transformers", min_length=1, max_length=100)
    backend_version: str | None = Field(default=None, max_length=100)
    dtype: str = Field(default="bfloat16", max_length=50)
    quantization: str | None = Field(default=None, max_length=100)
    adapter: str | None = Field(default=None, max_length=500)
    parallelism: ModelParallelism = Field(default_factory=SingleGPUParallelism)


class ResourceRequirements(StrictModel):
    gpu: int = Field(default=1, ge=0, le=64)
    min_vram_gb: float = Field(default=0, ge=0)
    cpu: float = Field(default=1, gt=0)
    memory_gb: float = Field(default=1, gt=0)

    @property
    def gpu_count(self) -> int:
        return self.gpu


def _validate_parallelism_resources(
    model: ModelRequirement,
    resources: ResourceRequirements,
) -> None:
    parallelism = model.parallelism
    if resources.gpu != parallelism.degree:
        raise ValueError(
            "resources.gpu must equal model.parallelism.degree; multi-GPU intent must be explicit"
        )
    if isinstance(parallelism, ModelShardedParallelism):
        if model.backend.casefold() != "transformers":
            raise ValueError(
                "transformers_device_map model sharding requires the transformers backend"
            )
        if resources.min_vram_gb > 0 and parallelism.max_memory_gb_per_gpu > resources.min_vram_gb:
            raise ValueError(
                "model-sharded max_memory_gb_per_gpu cannot exceed resources.min_vram_gb"
            )


class RetryPolicy(StrictModel):
    max_attempts: int = Field(default=3, ge=1, le=100)
    backoff_seconds: float = Field(default=1, ge=0, le=86400)
    retry_on: set[str] = Field(default_factory=lambda: {"lost", "preempted", "runtime_error"})


class CheckpointPolicy(StrictModel):
    enabled: bool = True
    interval_seconds: float | None = Field(default=None, gt=0)
    grace_seconds: float = Field(default=30, ge=0, le=3600)


class RuntimeEnvironment(StrictModel):
    pip: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    working_dir_ref: str | None = Field(default=None, max_length=1000)

    @field_validator("pip")
    @classmethod
    def reject_credential_bearing_requirements(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(_contains_embedded_credential(item) for item in value):
            raise ValueError("runtime pip requirements must not contain credentials")
        return validate_pinned_runtime_requirements(value)

    @field_validator("working_dir_ref")
    @classmethod
    def reject_credential_bearing_working_directory(cls, value: str | None) -> str | None:
        if value is not None and _contains_embedded_credential(value):
            raise ValueError("runtime working_dir_ref must not contain credentials")
        return validate_job_working_dir_reference(value)

    @field_validator("env")
    @classmethod
    def reject_secret_environment(cls, value: dict[str, str]) -> dict[str, str]:
        secret_words = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY", "AUTHORIZATION")
        offenders = [key for key in value if any(word in key.upper() for word in secret_words)]
        if offenders:
            raise ValueError("credentials cannot be forwarded in a runtime environment")
        reserved = [
            key
            for key in value
            if key.upper() == "CUDA_VISIBLE_DEVICES"
            or key.upper().startswith(("RAY_", "TOPODROME_"))
        ]
        if reserved:
            raise ValueError("runtime environment contains a Topodrome-reserved key")
        return value


class JobSpec(StrictModel):
    name: str | None = Field(default=None, max_length=160)
    description: str | None = Field(default=None, max_length=4000)
    entrypoint: str = Field(min_length=1, max_length=1000)
    source_ref: str | None = Field(default=None, max_length=500)
    model: ModelRequirement
    assets: dict[str, AssetRef] = Field(default_factory=dict)
    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    runtime: RuntimeEnvironment = Field(default_factory=RuntimeEnvironment)
    checkpoint: CheckpointPolicy = Field(default_factory=CheckpointPolicy)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    required_outputs: tuple[str, ...] = ()
    workload_kind: WorkloadKind = WorkloadKind.BATCH
    portability: Portability = Portability.PORTABLE
    affinity_site_id: str | None = None
    affinity_region: str | None = None
    priority: int = Field(default=0, ge=-1000, le=1000)
    artifact_retention_seconds: int | None = Field(default=None, ge=3600, le=31536000)
    estimated_runtime_seconds: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, strict=True
    )

    @field_validator("name", "description")
    @classmethod
    def readable_metadata(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        allowed = "\n\t" if info.field_name == "description" else ""
        if any(unicodedata.category(c).startswith("C") and c not in allowed for c in value):
            raise ValueError("job metadata must not contain control characters")
        return value.strip() or None

    @field_validator("model", mode="before")
    @classmethod
    def parse_model(cls, value: Any) -> Any:
        return {"alias": value} if isinstance(value, str) else value

    @field_validator("assets", mode="before")
    @classmethod
    def parse_assets(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("assets must map input aliases to asset references")
        return {str(alias): AssetRef.parse(reference) for alias, reference in value.items()}

    @field_validator("entrypoint")
    @classmethod
    def entrypoint_is_logical(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("entrypoint must be a relative logical path")
        if path.suffix != ".py":
            raise ValueError("entrypoint must name a Python file")
        return str(path)

    @field_validator("required_outputs")
    @classmethod
    def outputs_are_logical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for raw in value:
            path = PurePosixPath(raw)
            if path.is_absolute() or ".." in path.parts or raw in {"", "."}:
                raise ValueError("required outputs must be relative logical paths")
        return value

    @model_validator(mode="after")
    def validate_affinity(self) -> JobSpec:
        if self.portability is Portability.SITE_AFFINE and not self.affinity_site_id:
            raise ValueError("site-affine jobs require affinity_site_id")
        if self.portability is Portability.REGION_AFFINE and not self.affinity_region:
            raise ValueError("region-affine jobs require affinity_region")
        if self.workload_kind is WorkloadKind.ADAPTER_TRAINING and self.resources.gpu < 1:
            raise ValueError("adapter training requires a GPU")
        _validate_parallelism_resources(self.model, self.resources)
        if _job_kwargs_contain_credentials(self.kwargs):
            raise ValueError("credentials cannot be persisted in job kwargs")
        for alias in self.assets:
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", alias) is None:
                raise ValueError("asset aliases must be safe single path components")
        return self

    @property
    def runtime_fingerprint(self) -> str:
        """Stable identity of the environment and resources held by a warm actor."""

        payload = {
            "pip": list(self.runtime.pip),
            "env": dict(sorted(self.runtime.env.items())),
            "working_dir_ref": self.runtime.working_dir_ref,
            "cpu": format(self.resources.cpu, ".12g"),
            "memory_bytes": round(self.resources.memory_gb * 1024**3),
            "gpu_count": self.resources.gpu_count,
            "model_parallelism": self.model.parallelism.model_dump(mode="json"),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class GPUProcess(StrictModel):
    pid: int = Field(gt=0)
    used_memory_bytes: int = Field(default=0, ge=0)
    process_name: str = ""
    username: str | None = None
    script_name: str | None = None
    owner: str | None = None
    topodrome_owned: bool = False
    reclaimable: bool = False


class GPUInventory(StrictModel):
    uuid: str = Field(min_length=1, max_length=200)
    index: int = Field(ge=0)
    name: str
    total_memory_bytes: int = Field(ge=0)
    free_memory_bytes: int = Field(ge=0)
    utilization_percent: float = Field(ge=0, le=100)
    temperature_c: float | None = None
    healthy: bool = True
    externally_reserved: bool = False
    processes: tuple[GPUProcess, ...] = ()
    observed_at: datetime = Field(default_factory=utc_now)

    @property
    def foreign_processes(self) -> tuple[GPUProcess, ...]:
        return tuple(process for process in self.processes if not process.topodrome_owned)

    @property
    def blocking_foreign_processes(self) -> tuple[GPUProcess, ...]:
        return tuple(
            process
            for process in self.processes
            if not process.topodrome_owned and not process.reclaimable
        )

    @property
    def reclaimable_processes(self) -> tuple[GPUProcess, ...]:
        return tuple(process for process in self.processes if process.reclaimable)

    @property
    def free_vram_gb(self) -> float:
        return self.free_memory_bytes / (1024**3)


class GPUReservationRequest(StrictModel):
    """Reserve one exact physical GPU for a bounded operator maintenance window."""

    site_id: str = Field(min_length=1, max_length=120)
    gpu_uuid: str | None = Field(default=None, min_length=1, max_length=200)
    gpu_index: int | None = Field(default=None, ge=0)
    duration_seconds: int = Field(ge=1, le=31_536_000)
    reason: str = Field(min_length=1, max_length=2000)
    force: bool = False

    @model_validator(mode="after")
    def require_exactly_one_gpu_selector(self) -> GPUReservationRequest:
        if (self.gpu_uuid is None) == (self.gpu_index is None):
            raise ValueError("provide exactly one of gpu_uuid or gpu_index")
        return self


class GPUReservationExtensionRequest(StrictModel):
    """Compare-and-set an existing expiry; replays never add time twice."""

    expected_expires_at: AwareDatetime
    additional_seconds: int = Field(ge=1, le=31_536_000)


class GPUReservationSummary(StrictModel):
    reservation_id: str
    site_id: str
    gpu_uuid: str
    gpu_index: int
    requested_by: str
    reason: str
    force: bool = False
    duration_seconds: int
    state: GPUReservationState
    created_at: datetime
    expires_at: datetime
    canceled_at: datetime | None = None
    eviction_requested_at: datetime | None = None


class ModelCapability(StrictModel):
    alias: str
    revision: str
    backend: str
    backend_version: str
    dtype: str
    quantization: str | None = None
    adapter: str | None = None
    availability: ModelAvailability
    gpu_uuid: str | None = None
    runtime_fingerprint: str | None = Field(
        default=None,
        pattern=r"^(?:[0-9a-f]{64}|legacy-runtime)$",
    )
    replica_id: str | None = Field(default=None, min_length=1, max_length=160)
    replica_gpu_uuids: tuple[str, ...] = ()
    replica_key_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Site-authoritative start of the current continuous idle interval. This
    # originates in the durable local actor-allocation fence, not scheduler
    # receipt time, so central actor expiry survives scheduler restarts.
    idle_since: datetime | None = None
    verified_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_replica_membership(self) -> ModelCapability:
        if len(set(self.replica_gpu_uuids)) != len(self.replica_gpu_uuids):
            raise ValueError("replica GPU UUIDs must be unique")
        if self.replica_gpu_uuids:
            if self.gpu_uuid is None or self.gpu_uuid not in self.replica_gpu_uuids:
                raise ValueError("warm capability GPU must belong to its replica")
            if self.availability is not ModelAvailability.WARM:
                raise ValueError("only warm capabilities may name replica GPU members")
            if len(self.replica_gpu_uuids) > 1 and self.replica_id is None:
                raise ValueError("multi-GPU warm capabilities require a replica identity")
        if self.replica_id is not None and self.availability is not ModelAvailability.WARM:
            raise ValueError("only warm capabilities may name a replica identity")
        if self.replica_key_hash is not None and self.availability is not ModelAvailability.WARM:
            raise ValueError("only warm capabilities may name a replica key hash")
        if self.idle_since is not None and (
            self.availability is not ModelAvailability.WARM or self.replica_id is None
        ):
            raise ValueError("only an identified warm replica may publish idle_since")
        return self

    def identity_tuple(self) -> tuple[str | None, ...]:
        return (
            self.alias,
            self.revision,
            self.backend,
            self.backend_version,
            self.dtype,
            self.quantization,
            self.adapter,
        )


class SiteHeartbeat(StrictModel):
    site_id: str
    site_kind: SiteKind
    state: SiteState = SiteState.READY
    agent_session_id: str
    sequence: int = Field(ge=0)
    gpus: tuple[GPUInventory, ...] = ()
    models: tuple[ModelCapability, ...] = ()
    cached_assets: set[str] = Field(default_factory=set)
    supported_workloads: set[WorkloadKind] = Field(default_factory=lambda: set(WorkloadKind))
    storage_available: bool = True
    storage_authoritative: bool = False
    queue_depth: int = Field(default=0, ge=0)
    reliability: float = Field(default=1, ge=0, le=1)
    gpu_safety_margin_bytes: int = Field(default=0, ge=0)
    total_cpu: float | None = Field(default=None, gt=0)
    available_cpu: float | None = Field(default=None, ge=0)
    total_memory_bytes: int | None = Field(default=None, gt=0)
    available_memory_bytes: int | None = Field(default=None, ge=0)
    provider_instance_id: str | None = None
    region: str | None = None
    job_running: bool = False
    assignment_queued_locally: bool = False
    interactive_kernel_active: bool = False
    artifact_flush_pending: bool = False
    staging_operation_active: bool = False
    sent_at: datetime = Field(default_factory=utc_now)

    @field_validator("cached_assets")
    @classmethod
    def cached_asset_keys_are_digests(cls, value: set[str]) -> set[str]:
        if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in value):
            raise ValueError("cached asset keys must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def validate_host_capacity(self) -> SiteHeartbeat:
        values = (
            self.total_cpu,
            self.available_cpu,
            self.total_memory_bytes,
            self.available_memory_bytes,
        )
        if any(value is not None for value in values) and any(value is None for value in values):
            raise ValueError("host capacity telemetry must be reported as a complete set")
        if self.total_cpu is not None:
            assert self.available_cpu is not None
            assert self.total_memory_bytes is not None
            assert self.available_memory_bytes is not None
            if self.available_cpu > self.total_cpu:
                raise ValueError("available CPU cannot exceed total CPU")
            if self.available_memory_bytes > self.total_memory_bytes:
                raise ValueError("available memory cannot exceed total memory")
        return self


class FenceClaim(StrictModel):
    gpu_uuid: str
    epoch: int = Field(gt=0)


class AssignmentOffer(StrictModel):
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    job_id: str
    attempt_id: str
    attempt_number: int = Field(gt=0)
    site_id: str
    spec: JobSpec
    claims: tuple[FenceClaim, ...]
    fencing_token: str = Field(repr=False)
    expires_at: datetime
    lease_ttl_seconds: float = Field(default=90, gt=0)
    checkpoint_ref: str | None = Field(default=None, max_length=1000)
    # Set only when placement selected GPUs occupied by incompatible, proven
    # idle Topodrome actors. The site still atomically proves and fences every
    # complete source replica before admitting this exact target lease.
    replace_idle_actors: bool = False

    @model_validator(mode="after")
    def validate_claims(self) -> AssignmentOffer:
        claim_uuids = tuple(claim.gpu_uuid for claim in self.claims)
        if len(set(claim_uuids)) != len(claim_uuids):
            raise ValueError("assignment claims must name unique GPU UUIDs")
        if len(claim_uuids) != self.spec.resources.gpu_count:
            raise ValueError("assignment claim count must equal the job's requested GPU count")
        return self


class AttemptLogDescriptor(StrictModel):
    """Agent-reported coordinates independently verified by the scheduler."""

    logical_path: str = Field(min_length=1, max_length=1000)
    start_offset: Literal[0] = 0
    end_offset: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AttemptTimings(StrictModel):
    startup_seconds: float | None = Field(default=None, ge=0, le=86400, allow_inf_nan=False)
    compute_seconds: float | None = Field(default=None, ge=0, le=31536000, allow_inf_nan=False)
    finalization_seconds: float | None = Field(default=None, ge=0, le=31536000, allow_inf_nan=False)


class InputStageMetrics(StrictModel):
    remaining_bytes: int = Field(default=0, ge=0, le=1024**5)
    downloaded_bytes: int = Field(default=0, ge=0, le=1024**5)
    download_seconds: float = Field(default=0, ge=0, le=31536000, allow_inf_nan=False)


class AttemptReport(StrictModel):
    message_id: str = Field(default_factory=lambda: new_id("evt"))
    attempt_id: str
    job_id: str
    site_id: str
    fencing_token: str = Field(repr=False)
    state: AttemptState
    gpu_released: bool = False
    timings: AttemptTimings | None = None
    reason_code: str | None = Field(default=None, max_length=100)
    message: str | None = Field(default=None, max_length=2000)
    checkpoint_ref: str | None = Field(default=None, max_length=1000)
    artifacts: tuple[ArtifactDescriptor, ...] = ()
    log: AttemptLogDescriptor | None = None
    log_complete: bool = False
    emitted_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_log_completion(self) -> AttemptReport:
        terminal = self.state in {
            AttemptState.SUCCEEDED,
            AttemptState.PREEMPTED,
            AttemptState.FAILED,
            AttemptState.CANCELED,
            AttemptState.LOST,
        }
        if self.log is not None and not self.log_complete:
            raise ValueError("an attempt log descriptor requires log_complete")
        if self.log_complete and not terminal:
            raise ValueError("only a terminal attempt can complete its log")
        return self


class LeaseRenewal(StrictModel):
    attempt_id: str
    fencing_token: str = Field(repr=False)


class LeaseRenewalGrant(StrictModel):
    attempt_id: str
    fencing_token: str = Field(repr=False)
    fencing_token_expires_at: datetime
    lease_expires_at: datetime


class ArtifactDescriptor(StrictModel):
    logical_path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    durable: bool = False

    @field_validator("logical_path")
    @classmethod
    def logical_path_only(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact path must be logical")
        return str(path)


class JobSummary(StrictModel):
    job_id: str
    owner_username: str = "@operator"
    state: JobState
    spec: JobSpec
    attempts: int = 0
    retry_not_before: datetime | None = None
    array_id: str | None = None
    array_index: int | None = None
    blocked_reason: str | None = None
    timing: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class KernelRequest(StrictModel):
    model: ModelRequirement
    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)
    mode: KernelMode = KernelMode.SERVICE
    idle_timeout_seconds: int = Field(default=1800, ge=60, le=86400)

    @field_validator("model", mode="before")
    @classmethod
    def parse_model(cls, value: Any) -> Any:
        return {"alias": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def require_raw_gpu(self) -> KernelRequest:
        if self.mode is KernelMode.RAW_MODEL and self.resources.gpu < 1:
            raise ValueError("raw-model sessions require at least one GPU")
        _validate_parallelism_resources(self.model, self.resources)
        return self


class KernelSummary(StrictModel):
    session_id: str
    owner_username: str = "@operator"
    state: KernelState
    mode: KernelMode
    model: ModelRequirement
    site_id: str | None = None
    connect_url: str | None = None
    created_at: datetime
    expires_at: datetime | None = None


class KernelOperationRequest(StrictModel):
    """One credential-free operation sent through the scheduler relay."""

    operation: str = Field(min_length=1, max_length=120)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments")
    @classmethod
    def reject_sensitive_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        if contains_sensitive_key(value):
            raise ValueError("credentials cannot be forwarded to a kernel operation")
        return value


class KernelOperationResult(StrictModel):
    operation_id: str
    session_id: str
    result: Any = None
    execution_count: int | None = Field(default=None, ge=0)


class KernelLifecycleReport(StrictModel):
    session_id: str
    attempt_id: str
    fencing_token: str = Field(repr=False)
    state: KernelState
    relay_id: str | None = None
    runtime_id: str | None = None
    actor_id: str | None = None
    message: str | None = Field(default=None, max_length=1000)


class KernelOperationReport(StrictModel):
    operation_id: str
    session_id: str
    attempt_id: str
    fencing_token: str = Field(repr=False)
    succeeded: bool
    result: Any = None
    message: str | None = Field(default=None, max_length=1000)


class HealthStatus(StrictModel):
    status: Literal["ok", "unavailable"]
    database: bool | None = None


_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:token|secrets?|passwords?|credentials?|authorizations?|api[_-]?keys?)(?:$|[_-])",
    re.I,
)
_EMBEDDED_CREDENTIAL = re.compile(
    r"(?i)(?:"
    r"[a-z][a-z0-9+.-]*://[^\s/@]+@|"
    r"(?:^|[?&#;\s])(?:token|access[_-]?token|refresh[_-]?token|secret|"
    r"client[_-]?secret|password|credentials?|authorization|api[_-]?key|"
    r"x-amz-credential|x-amz-signature)\s*=|"
    r"(?:^|[\r\n;])\s*(?:proxy[-_])?authorization\s*:\s*"
    r"(?:basic|bearer|digest|aws4-hmac-sha256)\s+\S+|"
    r"(?:^|[\r\n;])\s*(?:x[-_])?api[-_]key\s*:\s*\S+|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)
_KWARGS_CREDENTIAL_SCAN_MAX_DEPTH = 32
_KWARGS_CREDENTIAL_SCAN_MAX_NODES = 20_000
_KWARGS_CREDENTIAL_SCAN_MAX_STRING_CHARS = 1_000_000


def _contains_embedded_credential(value: str) -> bool:
    if _EMBEDDED_CREDENTIAL.search(value):
        return True
    # Percent-encoding must not turn query parameter names into a bypass. One
    # decoding pass is sufficient for ordinary persisted URLs and remains
    # bounded by the caller's string-size limit.
    decoded = unquote(value) if "%" in value else value
    return decoded != value and bool(_EMBEDDED_CREDENTIAL.search(decoded))


def _job_kwargs_contain_credentials(value: Any) -> bool:
    """Inspect a bounded JSON-like kwargs tree for keys or string material.

    The explicit limits make validation resistant to adversarially deep or
    enormous request objects. Exceeding a limit fails closed rather than
    leaving an uninspected suffix that could be persisted to PostgreSQL.
    """

    stack: list[tuple[Any, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > _KWARGS_CREDENTIAL_SCAN_MAX_NODES:
            raise ValueError("job kwargs exceed bounded credential inspection limits")
        if isinstance(current, str):
            if len(current) > _KWARGS_CREDENTIAL_SCAN_MAX_STRING_CHARS:
                raise ValueError("job kwargs exceed bounded credential inspection limits")
            if _contains_embedded_credential(current):
                return True
            continue
        if isinstance(current, dict):
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            if current and depth >= _KWARGS_CREDENTIAL_SCAN_MAX_DEPTH:
                raise ValueError("job kwargs exceed bounded credential inspection limits")
            if visited + len(stack) + len(current) > _KWARGS_CREDENTIAL_SCAN_MAX_NODES:
                raise ValueError("job kwargs exceed bounded credential inspection limits")
            for key, item in current.items():
                key_text = str(key)
                if len(key_text) > _KWARGS_CREDENTIAL_SCAN_MAX_STRING_CHARS:
                    raise ValueError("job kwargs exceed bounded credential inspection limits")
                if _SENSITIVE_KEY.search(key_text) or _contains_embedded_credential(key_text):
                    return True
                stack.append((item, depth + 1))
            continue
        if isinstance(current, (list, tuple, set, frozenset)):
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            if current and depth >= _KWARGS_CREDENTIAL_SCAN_MAX_DEPTH:
                raise ValueError("job kwargs exceed bounded credential inspection limits")
            if visited + len(stack) + len(current) > _KWARGS_CREDENTIAL_SCAN_MAX_NODES:
                raise ValueError("job kwargs exceed bounded credential inspection limits")
            stack.extend((item, depth + 1) for item in current)
    return False


def contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _SENSITIVE_KEY.search(str(key)) or contains_sensitive_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(contains_sensitive_key(item) for item in value)
    return False


AttemptReport.model_rebuild()
