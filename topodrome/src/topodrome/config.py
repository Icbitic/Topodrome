from __future__ import annotations

import os
import re
import stat
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from topodrome.domain.enums import SiteKind
from topodrome.source_limits import SourceLimits


class AutoDLSettings(BaseModel):
    """Manually registered capacity only; no provider API credentials or rentals."""

    model_config = SettingsConfigDict(extra="forbid")
    manual_ssh: bool = False
    manual_offline_unregister_seconds: int = Field(default=600, ge=60)
    manual_state_dir: Path | None = None
    manual_bootstrap_dir: Path | None = None
    site_id: str = "autodl"
    idle_timeout_seconds: int = Field(default=300, ge=60, le=86400)
    reconcile_interval_seconds: float = Field(default=15, ge=1, le=300)
    drain_timeout_seconds: float = Field(default=120, ge=1, le=1800)
    startup_heartbeat_timeout_seconds: int = Field(default=600, ge=300, le=3600)


_CREDENTIAL_ARGUMENT = re.compile(
    r"(?i)(?:bearer\s+|authorization\s*[:=]|"
    r"(?:auth|token|secret|passw(?:or)?d|credential|api[-_]?key)\s*[:=]|"
    r"--(?:[a-z0-9]+[-_])*(?:auth(?:orization)?|access[-_]?token|token|secret|"
    r"passw(?:or)?d|credential|api[-_]?key)"
    r"(?:=|\s+)|"
    r"(?:^|[\s;&|])(?:export\s+)?(?:[A-Za-z0-9]+_)*"
    r"(?:AUTH|AUTHORIZATION|ACCESS_TOKEN|TOKEN|SECRET|PASSWORD|PASSWD|"
    r"CREDENTIAL|API_KEY|APIKEY)\s*=)"
)


def _validate_credential_free_command(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value or "\0" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} must be a non-empty single line")
    if _CREDENTIAL_ARGUMENT.search(value):
        raise ValueError(f"{field_name} must not contain credentials")
    parsed_urls = re.findall(r"https?://[^\s'\"]+", value, flags=re.IGNORECASE)
    for url in parsed_urls:
        parts = urlsplit(url)
        query_keys = {key.casefold() for key in re.findall(r"(?:^|[?&])([^=&]+)=", url)}
        if (
            parts.username
            or parts.password
            or query_keys
            & {
                "access_token",
                "api_key",
                "apikey",
                "auth",
                "authorization",
                "credential",
                "key",
                "password",
                "secret",
                "token",
            }
        ):
            raise ValueError(f"{field_name} must not contain credential-bearing URLs")
    return value


class JupyterBootstrapSettings(BaseModel):
    """Opt-in recovery for an outbound agent reachable only through Jupyter."""

    enabled: bool = False
    site_id: str = Field(default="remote-jupyter-1", min_length=1, max_length=120)
    command: str | None = Field(default=None, max_length=2048)
    ready_marker: str = Field(
        default="TOPODROME_AGENT_READY",
        min_length=1,
        max_length=200,
    )
    attempt_timeout_seconds: float = Field(default=90, gt=0, le=1800)
    initial_retry_seconds: float = Field(default=5, gt=0, le=1800)
    max_retry_seconds: float = Field(default=300, gt=0, le=3600)

    @field_validator("site_id", "ready_marker")
    @classmethod
    def _non_whitespace_value(cls, value: str) -> str:
        value = value.strip()
        if not value or "\0" in value or "\n" in value or "\r" in value:
            raise ValueError("value must be a non-empty single line")
        return value

    @field_validator("command")
    @classmethod
    def _credential_free_command(cls, value: str | None) -> str | None:
        return _validate_credential_free_command(value, field_name="bootstrap command")

    @model_validator(mode="after")
    def _validate_enabled_configuration(self) -> JupyterBootstrapSettings:
        if self.enabled and self.command is None:
            raise ValueError("jupyter_bootstrap.command is required when enabled")
        if self.command is not None and self.ready_marker in self.command:
            raise ValueError("ready marker must not occur in the bootstrap command")
        if self.max_retry_seconds < self.initial_retry_seconds:
            raise ValueError("max_retry_seconds must be at least initial_retry_seconds")
        return self


class InputPlacementSettings(BaseModel):
    """Advisory estimates; never bypass admission or move an active lease."""

    enabled: bool = True
    min_saving_seconds: float = Field(default=30, ge=0, le=86400, allow_inf_nan=False)
    min_saving_fraction: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    cooldown_seconds: float = Field(default=60, ge=1, le=86400, allow_inf_nan=False)
    history_max_age_seconds: float = Field(default=604800, ge=60, le=2592000)
    default_download_mib_per_second: float = Field(default=1, gt=0, le=100000)
    local_read_mib_per_second: float = Field(default=512, gt=0, le=100000)
    preparation_overhead_seconds: float = Field(default=5, ge=0, le=3600)


class SchedulingPolicySettings(BaseModel):
    """Operator-owned placement policy for heterogeneous institutional GPUs."""

    # An empty allow-list deliberately treats every eligible Lab/Jupyter GPU as
    # fast.  That fail-closed default prevents an omitted classification from
    # causing paid AutoDL capacity to outrank an institutional GPU.
    fast_gpu_name_patterns: tuple[str, ...] = ()
    # Blocking foreign processes make a GPU ineligible unless the operator has
    # explicitly declared that the named site safely supports capacity sharing.
    # This must agree with the corresponding site-agent admission policy; the
    # empty default deliberately fails closed.
    foreign_capacity_sharing_site_ids: tuple[str, ...] = ()
    # Opt in through versioned reload; a priority-only change activates at
    # the next dispatch boundary without disturbing active work.
    prefer_autodl_over_foreign_processes: bool = False
    input_placement: InputPlacementSettings = Field(default_factory=InputPlacementSettings)
    starvation_seconds: float = Field(default=1800, ge=30, le=604800, allow_inf_nan=False)
    # Queue depth is reported as a count, while cold-start estimates are in
    # seconds.  This conservative default converts the count into the same
    # estimated-ready-time scale without requiring per-site tuning.
    queue_delay_seconds_per_job: float = Field(default=30, ge=0, le=86_400)
    # Threshold used only for explicitly permitted foreign-capacity sharing.
    light_contention_utilization_percent: float = Field(default=10, ge=0, le=100)
    # Global warm-actor maintenance is scheduler-owned so these values can be
    # changed through the drain-before-swap policy reload without restarting a
    # site agent.  Counts are physical GPU memberships, not Ray actor objects:
    # one two-GPU sharded replica therefore counts as two occupied GPUs.
    actor_policy_enabled: bool = True
    institutional_actor_floor_gpu_count: int = Field(default=1, ge=0, le=128)
    contended_actor_shedding_threshold_gpu_count: int = Field(
        default=5,
        ge=1,
        le=128,
    )
    actor_policy_stability_seconds: float = Field(default=30, ge=0, le=3_600)
    actor_policy_cooldown_seconds: float = Field(default=300, ge=0, le=86_400)
    institutional_warm_actor_idle_timeout_seconds: float | None = Field(
        default=1_200,
        ge=60,
        le=604_800,
    )
    institutional_warm_actor_idle_timeout_seconds_by_site: dict[
        str, Annotated[float, Field(ge=60, le=604_800)] | None
    ] = Field(default_factory=dict)

    @field_validator("institutional_warm_actor_idle_timeout_seconds_by_site")
    @classmethod
    def _valid_idle_timeout_sites(cls, values: dict[str, float | None]) -> dict[str, float | None]:
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", key) for key in values):
            raise ValueError("idle timeout overrides require valid site IDs")
        return values

    # Source admission is part of the same immutable, crash-safe policy
    # snapshot. A reload may raise or lower these bounds without invalidating
    # content-addressed sources that were accepted by an earlier generation.
    source_limits: SourceLimits = Field(default_factory=SourceLimits)

    @model_validator(mode="after")
    def _actor_policy_threshold_preserves_floor(self) -> SchedulingPolicySettings:
        if (
            self.contended_actor_shedding_threshold_gpu_count
            <= self.institutional_actor_floor_gpu_count
        ):
            raise ValueError(
                "contended actor shedding threshold must exceed the institutional actor floor"
            )
        return self

    @field_validator("fast_gpu_name_patterns")
    @classmethod
    def _valid_gpu_name_patterns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not value or any(character in value for character in ("\0", "\r", "\n")):
                raise ValueError("fast GPU name patterns must be non-empty single lines")
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"invalid fast GPU name pattern {value!r}: {exc}") from None
        return values

    @field_validator("foreign_capacity_sharing_site_ids")
    @classmethod
    def _valid_foreign_capacity_sharing_site_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        seen: set[str] = set()
        for value in values:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,119}", value):
                raise ValueError(
                    "foreign capacity-sharing site IDs must be 1-120 characters and "
                    "contain only letters, digits, '.', '_', ':', or '-'"
                )
            if value in seen:
                raise ValueError("foreign capacity-sharing site IDs must be unique")
            seen.add(value)
        return values


class ManagedJupyterSettings(BaseModel):
    """Opt-in stock Jupyter Server bridge for IDE and notebook clients."""

    enabled: bool = False
    executable: str = Field(default="topodrome-managed-jupyter", min_length=1, max_length=500)
    bind_host: str = Field(default="127.0.0.1", min_length=1, max_length=255)
    public_url_template: str = "http://127.0.0.1:{port}"
    notebooks_root: Path | None = None
    runtime_root: Path | None = None
    port_start: int = Field(default=8888, ge=1024, le=65535)
    port_end: int = Field(default=8999, ge=1024, le=65535)
    max_servers: int = Field(default=8, ge=1, le=128)
    startup_timeout_seconds: float = Field(default=30, gt=0, le=300)
    shutdown_timeout_seconds: float = Field(default=10, gt=0, le=120)
    idle_timeout_seconds: int = Field(default=1800, ge=60, le=86400)

    @field_validator("executable", "bind_host")
    @classmethod
    def _single_line_value(cls, value: str) -> str:
        value = value.strip()
        if not value or "\0" in value or "\n" in value or "\r" in value:
            raise ValueError("value must be a non-empty single line")
        return value

    @field_validator("public_url_template")
    @classmethod
    def _valid_public_url_template(cls, value: str) -> str:
        if value.count("{port}") != 1 or "{" in value.replace("{port}", ""):
            raise ValueError("public_url_template must contain only one {port} placeholder")
        try:
            rendered = value.format(port=12345)
            parts = urlsplit(rendered)
        except (KeyError, ValueError):
            raise ValueError("public_url_template is invalid") from None
        if (
            parts.scheme not in {"http", "https"}
            or not parts.netloc
            or parts.query
            or parts.fragment
            or parts.username
            or parts.password
            or parts.path not in {"", "/"}
        ):
            raise ValueError("public_url_template must be a credential-free HTTP origin")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _valid_ranges(self) -> ManagedJupyterSettings:
        if self.port_end < self.port_start:
            raise ValueError("managed_jupyter.port_end must be at least port_start")
        return self


class UserAccessSettings(BaseModel):
    """Per-Unix-user bearer credentials for the loopback scheduler API."""

    enabled: bool = False
    credential_store_path: Path | None = None
    require_local_account: bool = True

    @field_validator("credential_store_path")
    @classmethod
    def _absolute_credential_store(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        expanded = value.expanduser()
        if not expanded.is_absolute():
            raise ValueError("user credential store path must be absolute")
        return expanded

    @model_validator(mode="after")
    def _enabled_store_is_explicit(self) -> UserAccessSettings:
        if self.enabled and self.credential_store_path is None:
            raise ValueError("enabled multi-user access requires a credential_store_path")
        return self


class StorageRetentionSettings(BaseModel):
    """Authoritative retention; remote cache budgets are independent."""

    model_config = SettingsConfigDict(extra="forbid")
    enabled: bool = False
    interval_seconds: int = Field(default=600, ge=30)
    artifact_seconds: int = Field(default=604800, ge=3600)
    failed_checkpoint_seconds: int = Field(default=604800, ge=3600)
    terminal_grace_seconds: int = Field(default=3600, ge=300)
    max_files_per_pass: int = Field(default=1000, ge=1, le=10000)
    max_bytes_per_pass: int = Field(default=64 * 1024**3, ge=1)


class SchedulerSettings(BaseSettings):
    storage_layout: Literal["attempt-v1", "blobs-v1"] = "attempt-v1"
    model_config = SettingsConfigDict(
        env_prefix="TOPODROME_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    database_dsn: str = "postgresql+asyncpg://topodrome@127.0.0.1/topodrome"
    api_host: str = "127.0.0.1"
    api_port: int = 8042
    public_url: str = "http://127.0.0.1:8042"
    # Backwards-compatible local development remains usable, but this mode is
    # constrained to loopback listeners and origins. Any deployable/non-loopback
    # configuration must explicitly disable it and provide authenticated HTTPS.
    insecure_development_mode: bool = True
    global_root: Path = Path("/srv/topodrome")
    storage_retention: StorageRetentionSettings = Field(default_factory=StorageRetentionSettings)
    heartbeat_ttl_seconds: int = 30
    offer_ttl_seconds: int = 30
    lease_ttl_seconds: int = 90
    finalization_recovery_grace_seconds: int = Field(default=86_400, ge=60, le=604_800)
    fencing_token_ttl_seconds: int = 10800
    dispatch_interval_seconds: float = 1.0
    kernel_connect_ttl_seconds: int = Field(default=300, ge=1, le=3600)
    kernel_operation_timeout_seconds: float = Field(default=120, gt=0, le=3600)
    kernel_max_request_bytes: int = Field(default=262_144, ge=4096, le=4_194_304)
    kernel_max_output_bytes: int = Field(default=262_144, ge=4096, le=4_194_304)
    kernel_request_cache_size: int = Field(default=256, ge=8, le=4096)
    api_token: SecretStr | None = None
    # This key fences work across scheduler restarts, so an ephemeral default is
    # unsafe. Production and tests must provide the same stable value to the
    # scheduler and the corresponding site verifier.
    fence_signing_key: SecretStr = Field(min_length=32)
    agent_tokens: dict[str, SecretStr] = Field(default_factory=dict)
    allow_unenrolled_agents: bool = False
    secret_file: Path = Path("/home/xiangkun/.config/topodrome/secrets.env")
    model_config_path: Path | None = Field(default=None, alias="model_config")
    scheduling: SchedulingPolicySettings = Field(default_factory=SchedulingPolicySettings)
    user_access: UserAccessSettings = Field(default_factory=UserAccessSettings)
    autodl: AutoDLSettings = Field(default_factory=AutoDLSettings)
    jupyter_bootstrap: JupyterBootstrapSettings = Field(default_factory=JupyterBootstrapSettings)
    managed_jupyter: ManagedJupyterSettings = Field(default_factory=ManagedJupyterSettings)

    @field_validator("public_url")
    @classmethod
    def _valid_public_url(cls, value: str) -> str:
        value = value.rstrip("/")
        parts = urlsplit(value)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.netloc
            or parts.query
            or parts.fragment
            or parts.username
            or parts.password
        ):
            raise ValueError("public_url must be a credential-free HTTP URL")
        return value

    @model_validator(mode="after")
    def _deployment_security_is_fail_closed(self) -> SchedulerSettings:
        public = urlsplit(self.public_url)
        managed = urlsplit(self.managed_jupyter.public_url_template.format(port=12345))
        # Both built-in servers speak plaintext HTTP.  They may therefore be
        # exposed only on loopback and published through a TLS reverse proxy;
        # an HTTPS public URL alone does not secure a non-loopback bind.
        if not _is_loopback_host(self.api_host):
            raise ValueError(
                "the built-in scheduler listener is plaintext; a loopback API listener is required"
            )
        if self.managed_jupyter.enabled and not _is_loopback_host(self.managed_jupyter.bind_host):
            raise ValueError(
                "the built-in managed-Jupyter listener is plaintext and must bind to loopback"
            )
        if self.insecure_development_mode:
            if not _is_loopback_host(public.hostname):
                raise ValueError(
                    "insecure_development_mode permits only loopback API listeners and origins"
                )
            if self.managed_jupyter.enabled and (not _is_loopback_host(managed.hostname)):
                raise ValueError(
                    "insecure_development_mode permits only loopback managed-Jupyter "
                    "listeners and origins"
                )
            return self

        if self.api_token is None or not self.api_token.get_secret_value().strip():
            raise ValueError("production scheduler configuration requires api_token")
        if public.scheme != "https":
            raise ValueError("production scheduler public_url must use HTTPS")
        if self.managed_jupyter.enabled and managed.scheme != "https":
            raise ValueError("production managed-Jupyter public_url_template must use HTTPS")
        return self

    def validate_api_bind_host(self, host: str) -> str:
        """Validate a runtime CLI bind override against the configured mode."""

        if not _is_loopback_host(host):
            raise ValueError(
                "the built-in scheduler listener is plaintext; a loopback API listener is required"
            )
        return host

    @classmethod
    def from_yaml(cls, path: Path | str) -> SchedulerSettings:
        raw = _read_yaml(path)
        return cls(**raw)


class RcloneSettings(BaseModel):
    # Enable only when this remote is the scheduler's authoritative global root.
    verification_via_scheduler: bool = False
    mode: Literal["mount", "mirror"] = "mount"
    mountpoint: Path
    remote: str
    cache_dir: Path
    binary: str = "rclone"
    vfs_cache_max_size_gb: float = Field(default=20, gt=0)
    min_free_space_gb: float = Field(default=10, gt=0)
    write_back_seconds: float = Field(default=5, gt=0)
    poll_interval_seconds: float = Field(default=5, gt=0)
    startup_timeout_seconds: float = Field(default=30, gt=0)
    verification_timeout_seconds: float = Field(default=120, gt=0)
    transfer_timeout_seconds: float = Field(default=6 * 60 * 60, gt=0)


class SSHLocalForwardSettings(BaseModel):
    """Protected outbound route from a remote site to the Lab1 scheduler."""

    identity_file: Path
    known_hosts_file: Path
    remote_host: str
    remote_user: str
    remote_port: int = Field(default=22, ge=1, le=65_535)
    local_host: Literal["127.0.0.1", "::1"] = "127.0.0.1"
    local_port: int = Field(ge=1, le=65_535)
    target_host: str = "127.0.0.1"
    target_port: int = Field(ge=1, le=65_535)
    state_directory: Path
    ssh_executable: Path = Path("/usr/bin/ssh")
    startup_timeout_seconds: float = Field(default=15, gt=0, le=300)
    shutdown_timeout_seconds: float = Field(default=10, gt=0, le=300)

    @field_validator(
        "identity_file",
        "known_hosts_file",
        "state_directory",
        "ssh_executable",
    )
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        expanded = value.expanduser()
        if not expanded.is_absolute():
            raise ValueError("SSH local-forward paths must be absolute")
        return expanded


class SiteSettings(BaseSettings):
    storage_layout: Literal["attempt-v1", "blobs-v1"] = "attempt-v1"
    model_config = SettingsConfigDict(
        env_prefix="TOPODROME_SITE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    site_id: str
    display_name: str
    scheduler_url: str
    site_kind: SiteKind = SiteKind.LAB
    model_catalog_site_id: str | None = None
    provider_instance_id: str | None = None
    region: str | None = None
    enrollment_token: SecretStr | None = None
    fence_verification_key: SecretStr = Field(min_length=32)
    allowed_gpu_indices: tuple[int, ...] | None = None
    global_root: Path
    scratch_root: Path
    # Ray's timestamped AF_UNIX sockets require a short lexical temp path.
    # Long scratch roots are reached through a private, validated alias here.
    ray_short_temp_root: Path | None = Path("/tmp")  # noqa: S108 -- validated below UID root
    persistent_root: Path | None = None
    spool_path: Path
    heartbeat_interval_seconds: float = 5
    gpu_safety_margin_gb: float = 1
    allow_foreign_processes_when_capacity_sufficient: bool = False
    cooperative_process_patterns: tuple[str, ...] = ()
    cooperative_exit_timeout_seconds: float = 10
    # Optional cooperative reservation protocol used by shared Jupyter hosts.
    # Each exact ``gpu_<index>.lock`` entry is validated locally before it can
    # remove that physical GPU from scheduling consideration.
    gpu_reservation_lock_dir: Path | None = None
    foreign_pressure_grace_seconds: float = 30
    # Proactive contention rebalancing unloads only an exact idle warm actor;
    # normal demand later recreates it through the central scheduler.
    idle_actor_rebalance_enabled: bool = True
    idle_actor_rebalance_cooldown_seconds: float = Field(default=300, ge=0, le=86_400)
    # Optional residency limit for a clean warm actor after its last workload
    # lease is released. This unloads the actor only; the site agent and Ray
    # cluster remain available for later cold starts.
    warm_actor_idle_timeout_seconds: float | None = Field(
        default=None,
        ge=60,
        le=604_800,
    )
    light_contention_utilization_percent: float = Field(default=10, ge=0, le=100)
    checkpoint_grace_seconds: float = 30
    checkpoint_keep_snapshots: int = Field(default=2, ge=2, le=1000)
    # Explicit user cancellation is bounded independently of a job's
    # checkpoint policy. A submitted workload cannot extend this operator
    # safety limit through its raw specification.
    cancellation_grace_seconds: float = Field(default=30, ge=0, le=300)
    finalization_concurrency: int = Field(default=2, ge=1, le=32)
    finalization_backlog_limit: int = Field(default=4, ge=1, le=128)
    asset_cache_max_age_seconds: int = Field(default=604_800, ge=60)
    asset_cache_max_bytes: int = Field(default=100 * 1024**3, ge=0)
    asset_cache_gc_interval_seconds: int = Field(default=600, ge=30)
    model_config_path: Path | None = Field(default=None, alias="model_config")
    # Full LFS-content verification is the secure default. Operators may turn
    # it off only for trusted, immutable model trees that were preseeded out of
    # band; acquired and staged payloads retain their independent verification.
    verify_preseeded_model_hashes: bool = True
    rclone: RcloneSettings | None = None
    ssh_local_forward: SSHLocalForwardSettings | None = None

    @field_validator("scheduler_url")
    @classmethod
    def _secure_scheduler_url(cls, value: str) -> str:
        value = value.rstrip("/")
        parts = urlsplit(value)
        if (
            parts.scheme not in {"http", "https", "ws", "wss"}
            or not parts.netloc
            or parts.hostname is None
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or any(character in value for character in ("\0", "\r", "\n"))
        ):
            raise ValueError("scheduler_url must be a credential-free HTTP(S) or WebSocket URL")
        try:
            _ = parts.port
        except ValueError:
            raise ValueError("scheduler_url has an invalid port") from None
        if parts.scheme in {"http", "ws"} and not _is_loopback_host(parts.hostname):
            raise ValueError("non-loopback scheduler_url must use HTTPS or WSS")
        return value

    @field_validator("gpu_reservation_lock_dir")
    @classmethod
    def _absolute_gpu_reservation_lock_dir(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        expanded = value.expanduser()
        if not expanded.is_absolute():
            raise ValueError("gpu_reservation_lock_dir must be an absolute path")
        return expanded

    @model_validator(mode="after")
    def _scheduler_uses_configured_local_forward(self) -> SiteSettings:
        if self.storage_layout == "blobs-v1" and self.rclone and self.rclone.mode == "mount":
            raise ValueError("blobs-v1 requires native storage or rclone mirror mode")
        tunnel = self.ssh_local_forward
        if tunnel is None:
            return self
        scheduler = urlsplit(self.scheduler_url)
        if (
            scheduler.scheme not in {"http", "ws"}
            or scheduler.hostname != tunnel.local_host
            or scheduler.port != tunnel.local_port
        ):
            raise ValueError(
                "scheduler_url must use the configured SSH local-forward loopback port"
            )
        return self

    @model_validator(mode="after")
    def _global_root_matches_rclone_namespace(self) -> SiteSettings:
        if self.rclone is not None and (
            self.global_root.expanduser().absolute()
            != self.rclone.mountpoint.expanduser().absolute()
        ):
            raise ValueError("global_root must equal the managed rclone namespace")
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> SiteSettings:
        raw = _read_yaml(path)
        # An explicit YAML null is commonly used as a deployment placeholder,
        # but BaseSettings gives init values priority over environment values.
        # Drop null identity hints so protected bootstrap environment values
        # remain authoritative for a recreated provider instance.
        for field_name in ("provider_instance_id", "region"):
            if raw.get(field_name) is None:
                raw.pop(field_name, None)
        settings = cls(**raw)
        manual = os.environ.get("TOPODROME_AUTODL_MANUAL_SITE_ID")
        if manual:
            from topodrome.autodl_identity import manual_token, valid_manual_site

            if not valid_manual_site(manual) or settings.site_kind is not SiteKind.AUTODL:
                raise ValueError("invalid manual AutoDL identity")
            if settings.enrollment_token is None:
                raise ValueError("manual AutoDL requires enrollment")
            settings = settings.model_copy(
                update={
                    "site_id": manual,
                    "display_name": manual,
                    "provider_instance_id": manual,
                    "model_catalog_site_id": settings.site_id,
                    "spool_path": settings.spool_path.with_name(manual + ".sqlite3"),
                    "enrollment_token": SecretStr(
                        manual_token(settings.enrollment_token.get_secret_value(), manual)
                    ),
                }
            )
        return settings


class SecretBundle(BaseModel):
    """Scheduler-only integration secrets with redacted repr/serialization."""

    jupyter_bootstrap_url: SecretStr | None = None
    jupyter_bootstrap_token: SecretStr | None = None
    managed_jupyter_server_token: SecretStr | None = None
    test_jupyter_url: str | None = None
    test_jupyter_token: SecretStr | None = None
    hf_token: SecretStr | None = None

    def model_dump(self, *_: Any, **__: Any) -> dict[str, str]:
        raise TypeError("SecretBundle must never be serialized")


class SiteModelLocation(BaseModel):
    path: Path | None = None
    origin: str | None = None
    persistent_path: Path | None = None
    local_path: Path | None = None
    revision_manifest: Path | None = None
    endpoint: str | None = None
    local_files_only: bool = False
    exclude: tuple[str, ...] = ()
    # Scheduler-side estimate for constructing a cold actor from this site's
    # configured model location. Warm selected actors always have zero startup
    # regardless of this value.
    estimated_startup_seconds: float | None = Field(default=None, ge=0, le=86_400)

    @model_validator(mode="after")
    def _external_manifest_is_for_direct_preseeded_models(self) -> SiteModelLocation:
        if self.revision_manifest is None:
            return self
        if not self.revision_manifest.expanduser().is_absolute():
            raise ValueError("revision_manifest must be an absolute path")
        if self.path is None or any(
            value is not None for value in (self.origin, self.persistent_path, self.local_path)
        ):
            raise ValueError(
                "revision_manifest is supported only with one direct preseeded model path"
            )
        return self


class ModelDefinition(BaseModel):
    alias: str
    revision: str
    backend: str = "transformers"
    backend_version: str = "configured"
    dtype: str = "bfloat16"
    quantization: str | None = None
    min_vram_gb: float = 0
    sites: dict[str, SiteModelLocation]


class ModelCatalog(BaseModel):
    models: dict[str, ModelDefinition]

    @classmethod
    def from_yaml(cls, path: Path | str) -> ModelCatalog:
        raw = _read_yaml(path)
        definitions: dict[str, ModelDefinition] = {}
        for alias, value in raw.get("models", {}).items():
            definitions[alias] = ModelDefinition(alias=alias, **value)
        return cls(models=definitions)

    def resolve(self, alias: str) -> ModelDefinition:
        try:
            return self.models[alias]
        except KeyError as exc:
            raise ValueError(f"unknown model alias: {alias}") from exc


def load_secret_bundle(path: Path | str) -> SecretBundle:
    secret_path = Path(path).expanduser().resolve()
    file_mode = stat.S_IMODE(secret_path.stat().st_mode)
    parent_mode = stat.S_IMODE(secret_path.parent.stat().st_mode)
    if file_mode & 0o077:
        raise PermissionError(f"secret file {secret_path} must have mode 0600 or stricter")
    if parent_mode & 0o077:
        raise PermissionError(
            f"secret directory {secret_path.parent} must have mode 0700 or stricter"
        )
    values: dict[str, str] = {}
    for raw_line in secret_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return SecretBundle(
        jupyter_bootstrap_url=values.get("TOPODROME_JUPYTER_BOOTSTRAP_URL"),
        jupyter_bootstrap_token=values.get("TOPODROME_JUPYTER_BOOTSTRAP_TOKEN"),
        managed_jupyter_server_token=values.get("TOPODROME_MANAGED_JUPYTER_SERVER_TOKEN"),
        test_jupyter_url=values.get("TOPODROME_TEST_JUPYTER_URL"),
        test_jupyter_token=values.get("TOPODROME_TEST_JUPYTER_TOKEN"),
        hf_token=values.get("TOPODROME_HF_TOKEN") or values.get("HF_TOKEN"),
    )


def _read_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def _is_loopback_host(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().strip("[]").casefold()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False
