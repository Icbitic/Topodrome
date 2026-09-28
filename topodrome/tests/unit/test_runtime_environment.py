from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

import topodrome.site_agent.ray_cluster as ray_cluster
from topodrome.runtime.environment import (
    RAY_ALLOWED_JOB_KEYS_ENVIRONMENT,
    RAY_DISABLE_FAILURE_SIGNAL_HANDLER_ENVIRONMENT,
    configure_ray_driver_environment,
    minimal_ray_head_environment,
    sanitize_configured_ray_worker_environment,
    sanitize_ray_worker_environment,
)
from topodrome.site_agent.ray_cluster import RayClusterConfig, SubprocessRayLifecycle

pytestmark = pytest.mark.unit


def _credential_bearing_environment() -> dict[str, str]:
    return {
        "PATH": "/runtime/bin",
        "LD_LIBRARY_PATH": "/cuda/lib64",
        "CUDA_HOME": "/cuda",
        "CUDA_VISIBLE_DEVICES": "2",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "TOPODROME_SITE_ENROLLMENT_TOKEN": "enrollment-canary",
        "TOPODROME_SITE_FENCE_VERIFICATION_KEY": "fence-canary",
        "TOPODROME_AUTODL_TOKEN": "provider-canary",
        "TOPODROME_TEST_JUPYTER_TOKEN": "jupyter-canary",
        "TOPODROME_HF_TOKEN": "hub-canary",
        "HF_TOKEN": "hub-alias-canary",
        "AWS_SECRET_ACCESS_KEY": "cloud-canary",
        "UNRELATED_VALUE": "must-not-be-inherited",
    }


def test_minimal_ray_head_environment_preserves_runtime_but_excludes_credentials() -> None:
    environment = minimal_ray_head_environment(_credential_bearing_environment())

    assert environment == {
        "CUDA_HOME": "/cuda",
        "CUDA_VISIBLE_DEVICES": "2",
        "LD_LIBRARY_PATH": "/cuda/lib64",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "PATH": "/runtime/bin",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "RAY_prestart_worker_first_driver": "0",
    }
    assert not any("canary" in value for value in environment.values())


def test_ray_driver_disables_native_failure_handler_without_changing_head_env() -> None:
    environment = {
        "PATH": "/runtime/bin",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "1",
        RAY_DISABLE_FAILURE_SIGNAL_HANDLER_ENVIRONMENT: "0",
    }

    configure_ray_driver_environment(environment)

    assert environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"
    assert environment[RAY_DISABLE_FAILURE_SIGNAL_HANDLER_ENVIRONMENT] == "1"
    assert RAY_DISABLE_FAILURE_SIGNAL_HANDLER_ENVIRONMENT not in (
        minimal_ray_head_environment(environment)
    )


def test_ray_worker_setup_replaces_inherited_environment_with_safe_values() -> None:
    environment = {
        **_credential_bearing_environment(),
        "RAY_JOB_ID": "ray-job",
        "RAY_AUTH_TOKEN": "ray-secret-canary",
        "OMP_NUM_THREADS": "4",
    }

    sanitize_ray_worker_environment(environment)

    assert environment["RAY_JOB_ID"] == "ray-job"
    assert environment["RAY_prestart_worker_first_driver"] == "0"
    assert environment["OMP_NUM_THREADS"] == "4"
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert "RAY_AUTH_TOKEN" not in environment
    assert "UNRELATED_VALUE" not in environment
    assert not any("canary" in value for value in environment.values())


def test_ray_prestart_cannot_be_reenabled_by_parent_environment() -> None:
    source = {
        "PATH": "/runtime/bin",
        "RAY_prestart_worker_first_driver": "1",
    }

    environment = minimal_ray_head_environment(source)

    assert environment["RAY_prestart_worker_first_driver"] == "0"


def test_importable_ray_worker_hook_consumes_fail_closed_key_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "PATH": "/runtime/bin",
        "SAFE_JOB_VALUE": "preserved",
        "UNLISTED_VALUE": "removed",
        "TOPODROME_HF_TOKEN": "secret-canary",
        RAY_ALLOWED_JOB_KEYS_ENVIRONMENT: '["SAFE_JOB_VALUE"]',
    }
    monkeypatch.setattr(os, "environ", environment)

    sanitize_configured_ray_worker_environment()

    assert environment["SAFE_JOB_VALUE"] == "preserved"
    assert "UNLISTED_VALUE" not in environment
    assert "TOPODROME_HF_TOKEN" not in environment
    assert RAY_ALLOWED_JOB_KEYS_ENVIRONMENT not in environment
    assert not any("secret-canary" in value for value in environment.values())


async def test_ray_lifecycle_passes_only_minimal_environment_to_every_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, str]] = []

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def create_subprocess_exec(*_arguments: str, **options: Any) -> Process:
        captured.append(dict(options["env"]))
        assert options["stdout"] is asyncio.subprocess.PIPE
        assert options["stderr"] is asyncio.subprocess.PIPE
        return Process()

    for key, value in _credential_bearing_environment().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    lifecycle = SubprocessRayLifecycle(RayClusterConfig(gpu_uuids=("GPU-one",)))

    assert await lifecycle._run("ray", "status", check=False) == 0

    assert captured == [minimal_ray_head_environment(os.environ)]
    assert not any("canary" in value for value in captured[0].values())


async def test_managed_ray_head_binds_locally_and_disables_dashboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "ray"
    lifecycle = SubprocessRayLifecycle(
        RayClusterConfig(gpu_uuids=("GPU-one",), temp_dir=target)
    )
    commands: list[tuple[str, ...]] = []

    async def is_running() -> bool:
        return False

    class Process:
        pid = 424_246
        returncode: int | None = None

    async def spawn(
        arguments: tuple[str, ...],
        *,
        log_directory: Path,
    ) -> Process:
        assert log_directory == target
        commands.append(arguments)
        return Process()

    async def wait_for_start(_process: object) -> None:
        return None

    async def stable_owned_head(
        _process: object,
        spawned: ray_cluster._OwnedRayHead,
    ) -> ray_cluster._OwnedRayHead:
        return spawned

    monkeypatch.setattr(lifecycle, "is_running", is_running)
    monkeypatch.setattr(lifecycle, "_spawn_owned_head", spawn)
    monkeypatch.setattr(lifecycle, "_wait_for_stable_owned_head", stable_owned_head)
    monkeypatch.setattr(lifecycle, "_wait_for_owned_start", wait_for_start)
    monkeypatch.setattr(
        ray_cluster,
        "_capture_owned_head",
        lambda pid, *, address, temp_dir: ray_cluster._OwnedRayHead(
            pid=pid,
            process_group_id=pid,
            start_ticks=123,
            boot_id="test-boot",
            command_sha256="d" * 64,
            address=address,
            temp_dir=temp_dir,
        ),
    )

    assert await lifecycle.ensure_running()
    command = commands[0]
    assert "--block" in command
    assert "--node-ip-address=127.0.0.1" in command
    assert "--dashboard-host=127.0.0.1" in command
    assert "--include-dashboard=false" in command


async def test_cold_ray_probe_skips_slow_cli_when_endpoint_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = SubprocessRayLifecycle(RayClusterConfig(gpu_uuids=("GPU-one",)))
    cli_called = False

    async def open_connection(_host: str, _port: int) -> tuple[object, object]:
        raise ConnectionRefusedError

    async def run(*_arguments: str, check: bool) -> int:
        nonlocal cli_called
        del check
        cli_called = True
        return 0

    monkeypatch.setattr(asyncio, "open_connection", open_connection)
    monkeypatch.setattr(lifecycle, "_run", run)

    assert not await lifecycle.is_running()
    assert not cli_called
