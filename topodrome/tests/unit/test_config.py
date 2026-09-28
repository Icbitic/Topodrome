from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from topodrome.config import ModelCatalog, SchedulerSettings, SiteModelLocation, SiteSettings
from topodrome.site_agent.inventory import CooperativeProcessMatcher
from topodrome.source_limits import (
    HARD_MAX_SOURCE_FILE_BYTES,
    HARD_MAX_SOURCE_TOTAL_BYTES,
    MAX_SOURCE_FILE_BYTES,
    MAX_SOURCE_TOTAL_BYTES,
)

pytestmark = pytest.mark.unit


def test_scheduler_requires_explicit_stable_fencing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOPODROME_FENCE_SIGNING_KEY", raising=False)

    with pytest.raises(ValidationError, match="fence_signing_key"):
        SchedulerSettings()
    with pytest.raises(ValidationError, match="at least 32"):
        SchedulerSettings(fence_signing_key="too-short")


def test_scheduler_fast_gpu_classification_is_explicit_and_validated() -> None:
    safe_default = SchedulerSettings(fence_signing_key="f" * 32)
    assert safe_default.scheduling.fast_gpu_name_patterns == ()
    assert safe_default.scheduling.foreign_capacity_sharing_site_ids == ()
    assert safe_default.scheduling.queue_delay_seconds_per_job == 30
    assert safe_default.scheduling.light_contention_utilization_percent == 10
    assert safe_default.scheduling.actor_policy_enabled
    assert safe_default.scheduling.institutional_actor_floor_gpu_count == 1
    assert safe_default.scheduling.contended_actor_shedding_threshold_gpu_count == 5
    assert safe_default.scheduling.actor_policy_stability_seconds == 30
    assert safe_default.scheduling.actor_policy_cooldown_seconds == 300
    assert safe_default.scheduling.institutional_warm_actor_idle_timeout_seconds == 1_200

    configured = SchedulerSettings(
        fence_signing_key="f" * 32,
        scheduling={
            "fast_gpu_name_patterns": [r"^NVIDIA A800", r"RTX 4090$"],
            "foreign_capacity_sharing_site_ids": ["lab1"],
            "queue_delay_seconds_per_job": 12.5,
            "light_contention_utilization_percent": 8,
            "institutional_actor_floor_gpu_count": 3,
            "contended_actor_shedding_threshold_gpu_count": 6,
            "actor_policy_stability_seconds": 20,
            "actor_policy_cooldown_seconds": 240,
            "institutional_warm_actor_idle_timeout_seconds": 900,
        },
    )
    assert configured.scheduling.fast_gpu_name_patterns == (
        r"^NVIDIA A800",
        r"RTX 4090$",
    )
    assert configured.scheduling.foreign_capacity_sharing_site_ids == ("lab1",)
    assert configured.scheduling.queue_delay_seconds_per_job == 12.5
    assert configured.scheduling.light_contention_utilization_percent == 8
    assert configured.scheduling.institutional_actor_floor_gpu_count == 3
    assert configured.scheduling.contended_actor_shedding_threshold_gpu_count == 6
    assert configured.scheduling.actor_policy_stability_seconds == 20
    assert configured.scheduling.actor_policy_cooldown_seconds == 240
    assert configured.scheduling.institutional_warm_actor_idle_timeout_seconds == 900

    with pytest.raises(ValidationError, match="invalid fast GPU name pattern"):
        SchedulerSettings(
            fence_signing_key="f" * 32,
            scheduling={"fast_gpu_name_patterns": [r"["]},
        )

    for invalid_site_ids in ([""], [" remote-jupyter-1"], ["lab/1"], ["lab1", "lab1"]):
        with pytest.raises(ValidationError, match="foreign capacity-sharing site IDs"):
            SchedulerSettings(
                fence_signing_key="f" * 32,
                scheduling={"foreign_capacity_sharing_site_ids": invalid_site_ids},
            )

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SchedulerSettings(
            fence_signing_key="f" * 32,
            scheduling={"queue_delay_seconds_per_job": -1},
        )

    with pytest.raises(ValidationError, match="less than or equal to 100"):
        SchedulerSettings(
            fence_signing_key="f" * 32,
            scheduling={"light_contention_utilization_percent": 101},
        )

    with pytest.raises(ValidationError, match="shedding threshold must exceed"):
        SchedulerSettings(
            fence_signing_key="f" * 32,
            scheduling={
                "institutional_actor_floor_gpu_count": 5,
                "contended_actor_shedding_threshold_gpu_count": 5,
            },
        )


def test_scheduler_source_limits_are_bounded_and_cross_validated() -> None:
    defaults = SchedulerSettings(fence_signing_key="f" * 32).scheduling.source_limits
    assert defaults.max_file_bytes == MAX_SOURCE_FILE_BYTES
    assert defaults.max_total_bytes == MAX_SOURCE_TOTAL_BYTES

    production = SchedulerSettings(
        fence_signing_key="f" * 32,
        scheduling={
            "source_limits": {
                "max_file_bytes": HARD_MAX_SOURCE_FILE_BYTES,
                "max_total_bytes": HARD_MAX_SOURCE_TOTAL_BYTES,
            }
        },
    ).scheduling.source_limits
    assert production.max_file_bytes == 256 * 1024 * 1024
    assert production.max_total_bytes == 1024 * 1024 * 1024

    for values in (
        {"max_file_bytes": 0},
        {"max_file_bytes": HARD_MAX_SOURCE_FILE_BYTES + 1},
        {"max_total_bytes": HARD_MAX_SOURCE_TOTAL_BYTES + 1},
        {"max_file_bytes": 5, "max_total_bytes": 4},
    ):
        with pytest.raises(ValidationError):
            SchedulerSettings(
                fence_signing_key="f" * 32,
                scheduling={"source_limits": values},
            )


def test_per_site_idle_timeout_validation() -> None:
    field = "institutional_warm_actor_idle_timeout_seconds_by_site"
    settings = SchedulerSettings(
        fence_signing_key="f" * 32,
        scheduling={
            field: {"lab1": 300, "remote-jupyter-1": 300, "disabled": None},
        },
    )
    assert getattr(settings.scheduling, field)["lab1"] == 300
    assert settings.scheduling.institutional_warm_actor_idle_timeout_seconds == 1200
    for overrides in ({"lab1": 59}, {"lab1": 604801}, {"bad site": 300}):
        with pytest.raises(ValidationError):
            SchedulerSettings(fence_signing_key="f" * 32, scheduling={field: overrides})


def test_multi_user_access_requires_an_absolute_credential_store(tmp_path: Path) -> None:
    base = {"fence_signing_key": "f" * 32}
    with pytest.raises(ValidationError, match="credential_store_path"):
        SchedulerSettings(
            **base,
            user_access={"enabled": True},
        )
    with pytest.raises(ValidationError, match="absolute"):
        SchedulerSettings(
            **base,
            user_access={
                "enabled": True,
                "credential_store_path": "users.json",
            },
        )

    configured = SchedulerSettings(
        **base,
        user_access={
            "enabled": True,
            "credential_store_path": tmp_path / "users.json",
        },
    )
    assert configured.user_access.enabled
    assert configured.user_access.credential_store_path == tmp_path / "users.json"


def test_model_catalog_carries_bounded_per_site_startup_estimates() -> None:
    repository = Path(__file__).resolve().parents[2]
    catalog = ModelCatalog.from_yaml(repository / "configs/models.example.yaml")
    locations = catalog.resolve("meta-llama-3-8b-instruct").sites

    assert locations["lab1"].estimated_startup_seconds == 2
    assert locations["remote-jupyter-1"].estimated_startup_seconds == 120
    assert locations["autodl"].estimated_startup_seconds == 30

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SiteModelLocation(path=Path("/model"), estimated_startup_seconds=-1)


def test_insecure_development_mode_is_strictly_loopback() -> None:
    configured = SchedulerSettings(fence_signing_key="f" * 32)
    assert configured.insecure_development_mode
    with pytest.raises(ValueError, match="loopback API listener"):
        configured.validate_api_bind_host("192.0.2.1")

    for values in (
        {"api_host": "192.0.2.1"},
        {"public_url": "http://scheduler.example"},
        {
            "managed_jupyter": {
                "enabled": True,
                "bind_host": "192.0.2.1",
                "public_url_template": "http://127.0.0.1:{port}",
            }
        },
        {
            "managed_jupyter": {
                "enabled": True,
                "bind_host": "127.0.0.1",
                "public_url_template": "http://notebooks.example:{port}",
            }
        },
    ):
        with pytest.raises(ValidationError, match="loopback"):
            SchedulerSettings(fence_signing_key="f" * 32, **values)


def test_production_scheduler_requires_token_and_https_origins() -> None:
    base = {
        "fence_signing_key": "f" * 32,
        "insecure_development_mode": False,
        "public_url": "https://scheduler.example/base",
    }
    with pytest.raises(ValidationError, match="requires api_token"):
        SchedulerSettings(**base)
    with pytest.raises(ValidationError, match="public_url must use HTTPS"):
        SchedulerSettings(**{**base, "api_token": "api-secret", "public_url": "http://host"})
    with pytest.raises(ValidationError, match=r"managed-Jupyter.*HTTPS"):
        SchedulerSettings(
            **base,
            api_token="api-secret",  # noqa: S106 - synthetic test credential
            managed_jupyter={
                "enabled": True,
                "public_url_template": "http://127.0.0.1:{port}",
            },
        )

    configured = SchedulerSettings(
        **base,
        api_token="api-secret",  # noqa: S106 - synthetic test credential
        managed_jupyter={
            "enabled": True,
            "public_url_template": "https://notebooks.example:{port}",
        },
    )
    assert configured.api_token is not None

    with pytest.raises(ValidationError, match=r"scheduler listener.*loopback"):
        SchedulerSettings(
            **base,
            api_host="0.0.0.0",  # noqa: S104 - must be rejected
            api_token="api-secret",  # noqa: S106 - synthetic test credential
        )
    with pytest.raises(ValidationError, match=r"managed-Jupyter listener.*loopback"):
        SchedulerSettings(
            **base,
            api_token="api-secret",  # noqa: S106 - synthetic test credential
            managed_jupyter={
                "enabled": True,
                "bind_host": "192.0.2.10",
                "public_url_template": "https://notebooks.example:{port}",
            },
        )
    with pytest.raises(ValueError, match=r"scheduler listener.*loopback"):
        configured.validate_api_bind_host("0.0.0.0")  # noqa: S104 - must be rejected


@pytest.mark.parametrize(
    "command",
    [
        "/opt/start --token VALUE",
        "/opt/start --token=VALUE",
        "/opt/start --api-key VALUE",
        "/opt/start --hf-token VALUE",
        "/opt/start --jupyter_token=VALUE",
        "/opt/start --access-token VALUE",
        "/opt/start;--hf-token VALUE",
        "TOPODROME_HF_TOKEN=VALUE /opt/start",
        "curl https://host/path?access_token=VALUE",
    ],
)
def test_bootstrap_commands_reject_credential_argument_forms(command: str) -> None:
    with pytest.raises(ValidationError, match="must not contain credentials"):
        SchedulerSettings(
            fence_signing_key="f" * 32,
            jupyter_bootstrap={"command": command},
        )


def test_noncredential_tokenizer_option_is_not_a_false_positive() -> None:
    command = "TOKENIZERS_PARALLELISM=false /opt/topodrome/bin/start-agent"
    settings = SchedulerSettings(
        fence_signing_key="f" * 32,
        jupyter_bootstrap={"command": command},
    )
    assert settings.jupyter_bootstrap.command == command


def test_site_requires_explicit_stable_fencing_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TOPODROME_SITE_FENCE_VERIFICATION_KEY", raising=False)
    values = {
        "site_id": "lab1",
        "display_name": "Lab1",
        "scheduler_url": "http://127.0.0.1:8042",
        "global_root": tmp_path / "global",
        "scratch_root": tmp_path / "scratch",
        "spool_path": tmp_path / "spool.sqlite3",
    }

    with pytest.raises(ValidationError, match="fence_verification_key"):
        SiteSettings(**values)
    with pytest.raises(ValidationError, match="at least 32"):
        SiteSettings(**values, fence_verification_key="too-short")

    configured = SiteSettings(**values, fence_verification_key="s" * 32)
    assert configured.fence_verification_key.get_secret_value() == "s" * 32
    assert configured.ray_short_temp_root == Path("/tmp")  # noqa: S108


def test_warm_actor_idle_timeout_is_optional_bounded_and_environment_configurable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    field = "TOPODROME_SITE_WARM_ACTOR_IDLE_TIMEOUT_SECONDS"
    monkeypatch.delenv(field, raising=False)
    values = {
        "site_id": "site",
        "display_name": "Site",
        "scheduler_url": "https://scheduler.example",
        "global_root": tmp_path / "global",
        "scratch_root": tmp_path / "scratch",
        "spool_path": tmp_path / "spool.sqlite3",
        "fence_verification_key": "s" * 32,
    }

    assert SiteSettings(**values).warm_actor_idle_timeout_seconds is None
    assert (
        SiteSettings(
            **values,
            warm_actor_idle_timeout_seconds=7200,
        ).warm_actor_idle_timeout_seconds
        == 7200
    )
    assert (
        SiteSettings(**values, warm_actor_idle_timeout_seconds=60).warm_actor_idle_timeout_seconds
        == 60
    )
    assert (
        SiteSettings(
            **values, warm_actor_idle_timeout_seconds=604_800
        ).warm_actor_idle_timeout_seconds
        == 604_800
    )
    for invalid in (59, 604_801):
        with pytest.raises(ValidationError, match="warm_actor_idle_timeout_seconds"):
            SiteSettings(**values, warm_actor_idle_timeout_seconds=invalid)

    monkeypatch.setenv(field, "3600")
    assert SiteSettings(**values).warm_actor_idle_timeout_seconds == 3600


def test_explicit_cancellation_grace_defaults_to_thirty_seconds_and_is_bounded(
    tmp_path: Path,
) -> None:
    values = {
        "site_id": "site",
        "display_name": "Site",
        "scheduler_url": "https://scheduler.example",
        "global_root": tmp_path / "global",
        "scratch_root": tmp_path / "scratch",
        "spool_path": tmp_path / "spool.sqlite3",
        "fence_verification_key": "s" * 32,
    }

    assert SiteSettings(**values).cancellation_grace_seconds == 30
    assert SiteSettings(**values, cancellation_grace_seconds=0).cancellation_grace_seconds == 0
    with pytest.raises(ValidationError, match="cancellation_grace_seconds"):
        SiteSettings(**values, cancellation_grace_seconds=301)


@pytest.mark.parametrize(
    "scheduler_url",
    [
        "http://scheduler.example:8042",
        "ws://scheduler.example:8042",
        "https://user:password@scheduler.example",
        "https://scheduler.example?token=secret",
        "https://scheduler.example#fragment",
    ],
)
def test_site_scheduler_url_requires_tls_off_loopback(
    scheduler_url: str,
    tmp_path: Path,
) -> None:
    values = {
        "site_id": "site",
        "display_name": "Site",
        "scheduler_url": scheduler_url,
        "global_root": tmp_path / "global",
        "scratch_root": tmp_path / "scratch",
        "spool_path": tmp_path / "spool.sqlite3",
        "fence_verification_key": "s" * 32,
    }
    with pytest.raises(ValidationError, match=r"scheduler_url|non-loopback"):
        SiteSettings(**values)

    values["scheduler_url"] = "wss://scheduler.example/base"
    assert SiteSettings(**values).scheduler_url == "wss://scheduler.example/base"
    values["scheduler_url"] = "http://[::1]:8042/"
    assert SiteSettings(**values).scheduler_url == "http://[::1]:8042"


def test_null_yaml_provider_identity_does_not_shadow_bootstrap_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "site.yaml"
    config.write_text(
        "site_id: autodl\n"
        "display_name: AutoDL\n"
        "scheduler_url: https://scheduler.invalid\n"
        "provider_instance_id: null\n"
        "region: null\n"
        f"global_root: {tmp_path / 'global'}\n"
        f"scratch_root: {tmp_path / 'scratch'}\n"
        f"spool_path: {tmp_path / 'spool.sqlite3'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TOPODROME_SITE_PROVIDER_INSTANCE_ID", "provider-from-env")
    monkeypatch.setenv("TOPODROME_SITE_REGION", "beijingDC2")
    monkeypatch.setenv("TOPODROME_SITE_FENCE_VERIFICATION_KEY", "s" * 32)

    configured = SiteSettings.from_yaml(config)

    assert configured.provider_instance_id == "provider-from-env"
    assert configured.region == "beijingDC2"


def test_external_revision_manifest_requires_one_absolute_direct_path(
    tmp_path: Path,
) -> None:
    manifest = (tmp_path / "manifest.json").resolve()
    model = (tmp_path / "model").resolve()

    configured = SiteModelLocation(
        path=model,
        revision_manifest=manifest,
        local_files_only=True,
    )
    assert configured.revision_manifest == manifest

    with pytest.raises(ValidationError, match="absolute path"):
        SiteModelLocation(path=model, revision_manifest=Path("manifest.json"))
    with pytest.raises(ValidationError, match="direct preseeded"):
        SiteModelLocation(
            path=model,
            origin="owner/model",
            revision_manifest=manifest,
        )


def test_jupyter_deployment_config_shares_vram_and_only_reclaims_exact_idle_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOPODROME_SITE_FENCE_VERIFICATION_KEY", "s" * 32)
    config = Path(__file__).resolve().parents[2] / "configs/site.jupyter.example.yaml"

    settings = SiteSettings.from_yaml(config)
    matcher = CooperativeProcessMatcher(command_line_patterns=settings.cooperative_process_patterns)

    assert settings.allow_foreign_processes_when_capacity_sufficient
    assert not settings.idle_actor_rebalance_enabled
    assert settings.idle_actor_rebalance_cooldown_seconds == 300
    assert settings.warm_actor_idle_timeout_seconds is None
    assert settings.light_contention_utilization_percent == 10
    assert settings.gpu_reservation_lock_dir is None
    assert matcher.matches(
        600,
        command_line="/opt/conda/bin/python /home/jovyan/gpu_idle_load.py",
    )
    assert not matcher.matches(
        601,
        command_line="/opt/conda/bin/python /home/jovyan/gpu_idle_load.py.backup",
    )
    assert not matcher.matches(
        602,
        command_line="/opt/conda/bin/python /home/jovyan/train.py --name gpu_idle_load.py",
    )


def test_institutional_site_examples_delegate_warm_retention_to_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOPODROME_SITE_FENCE_VERIFICATION_KEY", "s" * 32)
    monkeypatch.setenv("TOPODROME_API_TOKEN", "synthetic-api-value")
    monkeypatch.setenv("TOPODROME_FENCE_SIGNING_KEY", "f" * 32)
    repository = Path(__file__).resolve().parents[2]

    lab1 = SiteSettings.from_yaml(repository / "configs/site.lab1.example.yaml")
    jupyter = SiteSettings.from_yaml(repository / "configs/site.jupyter.example.yaml")

    assert lab1.warm_actor_idle_timeout_seconds is None
    assert jupyter.warm_actor_idle_timeout_seconds is None
    assert not lab1.idle_actor_rebalance_enabled
    assert not jupyter.idle_actor_rebalance_enabled

    scheduler = SchedulerSettings.from_yaml(repository / "configs/scheduler.example.yaml")
    assert scheduler.scheduling.institutional_actor_floor_gpu_count == 1
    assert scheduler.scheduling.contended_actor_shedding_threshold_gpu_count == 5
    assert scheduler.scheduling.institutional_warm_actor_idle_timeout_seconds == 1_200


def test_autodl_examples_use_manual_capacity_and_safe_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("TOPODROME_API_TOKEN", "synthetic-api-value")
    monkeypatch.setenv("TOPODROME_FENCE_SIGNING_KEY", "f" * 32)
    scheduler = SchedulerSettings.from_yaml(repository / "configs/scheduler.example.yaml")

    assert not scheduler.autodl.manual_ssh
    assert scheduler.autodl.idle_timeout_seconds == 300
    assert scheduler.autodl.startup_heartbeat_timeout_seconds == 1800
    assert scheduler.scheduling.foreign_capacity_sharing_site_ids == (
        "lab1",
        "remote-jupyter-1",
    )

    monkeypatch.setenv("TOPODROME_SITE_FENCE_VERIFICATION_KEY", "f" * 32)
    site = SiteSettings.from_yaml(repository / "configs/site.autodl.example.yaml")
    assert site.scheduler_url == "http://127.0.0.1:18042"
    assert site.rclone is not None
    assert site.rclone.mode == "mirror"
    assert site.rclone.remote == "lab1-global:"
    assert site.global_root.is_relative_to(Path("/root/autodl-tmp"))
    assert site.ssh_local_forward is not None
    assert site.ssh_local_forward.remote_host == "111.127.53.162"
    assert site.ssh_local_forward.remote_port == 2201
    assert site.ssh_local_forward.local_port == 18042
    assert site.ssh_local_forward.target_port == scheduler.api_port


def test_gpu_reservation_lock_directory_must_be_absolute(tmp_path: Path) -> None:
    values = {
        "site_id": "jupyter",
        "display_name": "Jupyter",
        "scheduler_url": "https://scheduler.invalid",
        "global_root": tmp_path / "global",
        "scratch_root": tmp_path / "scratch",
        "spool_path": tmp_path / "spool.sqlite3",
        "fence_verification_key": "s" * 32,
    }

    with pytest.raises(ValidationError, match=r"gpu_reservation_lock_dir.*absolute"):
        SiteSettings(**values, gpu_reservation_lock_dir=Path("relative/locks"))

    configured = SiteSettings(
        **values,
        gpu_reservation_lock_dir=tmp_path / "gpu_runtime_locks",
    )
    assert configured.gpu_reservation_lock_dir == tmp_path / "gpu_runtime_locks"


def test_site_ssh_forward_requires_matching_loopback_scheduler_url(tmp_path: Path) -> None:
    values = {
        "site_id": "jupyter",
        "display_name": "Jupyter",
        "scheduler_url": "http://127.0.0.1:18042",
        "global_root": tmp_path / "global",
        "scratch_root": tmp_path / "scratch",
        "spool_path": tmp_path / "spool.sqlite3",
        "fence_verification_key": "s" * 32,
        "ssh_local_forward": {
            "identity_file": tmp_path / "id_ed25519",
            "known_hosts_file": tmp_path / "known_hosts",
            "remote_host": "lab1.example",
            "remote_user": "topodrome",
            "remote_port": 2201,
            "local_port": 18042,
            "target_port": 8042,
            "state_directory": tmp_path / "tunnel-state",
        },
    }

    configured = SiteSettings(**values)
    assert configured.ssh_local_forward is not None
    assert configured.ssh_local_forward.local_port == 18042

    with pytest.raises(ValidationError, match="configured SSH local-forward"):
        SiteSettings(**{**values, "scheduler_url": "http://127.0.0.1:18043"})
    with pytest.raises(ValidationError, match="SSH local-forward paths must be absolute"):
        SiteSettings(
            **{
                **values,
                "ssh_local_forward": {
                    **values["ssh_local_forward"],
                    "identity_file": "relative-key",
                },
            }
        )


def test_site_rclone_namespace_must_match_global_root(tmp_path: Path) -> None:
    values = {
        "site_id": "jupyter",
        "display_name": "Jupyter",
        "scheduler_url": "https://scheduler.invalid",
        "global_root": tmp_path / "global",
        "scratch_root": tmp_path / "scratch",
        "spool_path": tmp_path / "spool.sqlite3",
        "fence_verification_key": "s" * 32,
        "rclone": {
            "mode": "mirror",
            "mountpoint": tmp_path / "different-cache",
            "remote": "lab1-global:",
            "cache_dir": tmp_path / "vfs-cache",
        },
    }

    with pytest.raises(ValidationError, match="managed rclone namespace"):
        SiteSettings(**values)

    values["rclone"]["mountpoint"] = tmp_path / "global"
    assert SiteSettings(**values).rclone is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("enabled", True),
        ("base_url", "https://unused.invalid"),
        ("provision_on_demand", True),
        ("max_gpu_count", 4),
    ],
)
def test_removed_autodl_api_settings_are_rejected(field, value):
    from topodrome.config import AutoDLSettings

    with pytest.raises(ValidationError, match="Extra inputs"):
        AutoDLSettings(**{field: value})
