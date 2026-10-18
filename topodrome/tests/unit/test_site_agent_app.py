from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import signal
from pathlib import Path
from typing import Any

import pytest
import yaml

import topodrome.site_agent.app as site_agent_app
from topodrome.config import SiteSettings
from topodrome.runtime import ModelAvailability
from topodrome.site_agent.app import (
    AgentApplication,
    _control_url,
    _model_registry,
    _ray_cluster_config,
    _ray_executable,
    _storage,
    build_application,
)
from topodrome.site_agent.inventory import GPUReservationLockReader

pytestmark = pytest.mark.unit


def _git_blob_id(content: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(content)}\0".encode("ascii"))
    digest.update(content)
    return digest.hexdigest()


def _payload(root: Path, revision: str) -> None:
    files = {
        "config.json": b"{}\n",
        "tokenizer.json": b"{}\n",
        "model.safetensors": b"weights",
    }
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    tree = root / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    tree.parent.mkdir(parents=True)
    tree.write_text(
        json.dumps(
            {
                "format_version": 1,
                "files": {
                    name: {
                        "size": len(content),
                        "blob_id": _git_blob_id(content),
                        **(
                            {"lfs_sha256": hashlib.sha256(content).hexdigest()}
                            if name.endswith(".safetensors")
                            else {}
                        ),
                    }
                    for name, content in files.items()
                },
            }
        ),
        encoding="utf-8",
    )


def _settings(tmp_path: Path, model_config: Path | None = None, **values: Any) -> SiteSettings:
    return SiteSettings(
        site_id="test-site",
        display_name="Test",
        scheduler_url="https://scheduler.invalid",
        global_root=tmp_path / "global",
        scratch_root=tmp_path / "scratch",
        spool_path=tmp_path / "spool.sqlite3",
        fence_verification_key="f" * 32,
        model_config=model_config,
        **values,
    )


def test_control_url_rejects_cleartext_remote_and_credential_bearing_urls() -> None:
    assert _control_url(
        "http://127.0.0.1:8042/base",
        site_id="lab1",
        session_id="session",
    ).startswith("ws://127.0.0.1:8042/base/api/v1/agents/control?")
    assert _control_url(
        "https://scheduler.example/base",
        site_id="remote",
        session_id="session",
    ).startswith("wss://scheduler.example/base/api/v1/agents/control?")

    for invalid in (
        "http://scheduler.example",
        "ws://scheduler.example",
        "https://user:password@scheduler.example",
        "https://scheduler.example?token=secret",
        "https://scheduler.example#fragment",
    ):
        with pytest.raises(ValueError, match=r"credential-free|HTTPS"):
            _control_url(invalid, site_id="site", session_id="session")


def test_model_registry_verifies_payload_before_advertising_cached(tmp_path: Path) -> None:
    revision = "a" * 40
    incomplete = tmp_path / "local-model"
    incomplete.mkdir()
    persistent = tmp_path / "persistent-model"
    persistent.mkdir()
    _payload(persistent, revision)
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "test/model": {
                        "revision": revision,
                        "backend": "transformers",
                        "backend_version": importlib.metadata.version("transformers"),
                        "sites": {
                            "test-site": {
                                "local_path": str(incomplete),
                                "persistent_path": str(persistent),
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    registry, _ = _model_registry(_settings(tmp_path, config))

    capability = registry.public_capabilities()[0]
    registration = registry.get(capability.key)
    assert capability.availability is ModelAvailability.CACHED
    assert registration.local_path == persistent
    assert registration.payload_identity is not None
    assert "path" not in capability.as_dict()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, True), (False, False)],
)
def test_model_registry_passes_preseeded_hash_policy_only_to_startup_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: bool | None,
    expected: bool,
) -> None:
    revision = "e" * 40
    model = tmp_path / "preseeded-model"
    model.mkdir()
    _payload(model, revision)
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "test/model": {
                        "revision": revision,
                        "backend_version": importlib.metadata.version("transformers"),
                        "sites": {"test-site": {"path": str(model)}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    settings_values = {} if configured is None else {"verify_preseeded_model_hashes": configured}
    settings = _settings(tmp_path, config, **settings_values)
    observed: list[bool] = []
    original = site_agent_app.verify_transformers_payload

    def track_verification(*args: Any, **kwargs: Any) -> Any:
        observed.append(bool(kwargs["verify_lfs_sha256"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(site_agent_app, "verify_transformers_payload", track_verification)

    registry, _ = _model_registry(settings)

    assert settings.verify_preseeded_model_hashes is expected
    assert registry.public_capabilities()[0].availability is ModelAvailability.CACHED
    assert observed == [expected]


def test_model_acquisition_targets_persistent_storage_not_ephemeral_copy(
    tmp_path: Path,
) -> None:
    revision = "b" * 40
    persistent = tmp_path / "persistent-model"
    local = tmp_path / "ephemeral-model"
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "test/model": {
                        "revision": revision,
                        "backend": "transformers",
                        "backend_version": importlib.metadata.version("transformers"),
                        "sites": {
                            "test-site": {
                                "origin": "owner/model",
                                "persistent_path": str(persistent),
                                "local_path": str(local),
                                "exclude": ["original/*"],
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    registry, router = _model_registry(_settings(tmp_path, config))
    registration = registry.public_capabilities()[0]
    acquirer = router._acquirers[registration.key]

    assert acquirer.destination == persistent
    assert acquirer.ignore_patterns == ("original/*",)
    assert registration.availability is ModelAvailability.OBTAINABLE


def test_persistent_model_is_atomically_staged_to_configured_local_ssd(
    tmp_path: Path,
) -> None:
    revision = "c" * 40
    persistent = tmp_path / "persistent-model"
    persistent.mkdir()
    _payload(persistent, revision)
    local = (tmp_path / "local-model").resolve()
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "test/model": {
                        "revision": revision,
                        "backend": "transformers",
                        "backend_version": importlib.metadata.version("transformers"),
                        "sites": {
                            "test-site": {
                                "persistent_path": str(persistent),
                                "local_path": str(local),
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    registry, router = _model_registry(_settings(tmp_path, config))
    original = registry.get(registry.public_capabilities()[0].key)
    staged = router(original)

    assert original.local_path == persistent
    assert staged.local_path == local
    assert staged.payload_identity == original.payload_identity
    assert (local / "model.safetensors").read_bytes() == b"weights"
    assert not tuple(local.parent.glob(f".{local.name}.topodrome-stage-*"))


def test_unverified_model_without_origin_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "incomplete"
    model.mkdir()
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "test/model": {
                        "revision": "c" * 40,
                        "backend_version": importlib.metadata.version("transformers"),
                        "sites": {"test-site": {"path": str(model)}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no verified local payload"):
        _model_registry(_settings(tmp_path, config))


def test_read_only_preseeded_model_uses_operator_revision_manifest(
    tmp_path: Path,
) -> None:
    revision = "d" * 40
    model = tmp_path / "read-only-model"
    model.mkdir()
    _payload(model, revision)
    internal = model / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    manifest = tmp_path / "model-revision-manifest.json"
    internal.replace(manifest)
    manifest.chmod(0o600)
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "test/model": {
                        "revision": revision,
                        "backend_version": importlib.metadata.version("transformers"),
                        "sites": {
                            "test-site": {
                                "path": str(model),
                                "revision_manifest": str(manifest),
                                "local_files_only": True,
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    registry, _ = _model_registry(_settings(tmp_path, config))

    capability = registry.public_capabilities()[0]
    assert capability.availability is ModelAvailability.CACHED
    assert registry.get(capability.key).payload_identity is not None


def test_storage_wires_http_only_without_external_executables(tmp_path: Path) -> None:
    from topodrome.storage.http_mirror import HttpMirrorStorage
    from topodrome.storage.http_transport import SchedulerFileTransport

    settings = _settings(
        tmp_path,
        enrollment_token="synthetic",  # noqa: S106 -- synthetic fixture
        storage={
            "authority_id": "lab1-global:",
            "small_file_max_bytes": 4096,
            "min_free_space_gb": 3,
            "verification_timeout_seconds": 19,
        },
    )
    storage = _storage(settings)
    try:
        assert isinstance(storage, HttpMirrorStorage)
        assert isinstance(storage._bulk, SchedulerFileTransport)
        assert storage._reader is storage._bulk
        assert storage._small_files.max_bytes == 4096
        assert storage.root == (tmp_path / "global").resolve()
        assert storage.remote_root == "lab1-global:"
    finally:
        storage.close()
    assert storage._small_files._client.is_closed
    assert storage._bulk._client.is_closed
    assert storage._bulk._bulk.is_closed
    assert storage._bulk._uploads._client.is_closed
    settings.storage.small_file_max_bytes = 0
    storage = _storage(settings)
    try:
        assert storage._small_files is None
    finally:
        storage.close()


async def test_application_closes_only_its_owned_storage() -> None:
    closed: list[str] = []

    class Agent:
        async def run(self) -> None:
            closed.append("ran")

        async def shutdown(self, *, stop_ray: bool) -> None:
            assert stop_ray

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    class Storage:
        def close(self) -> None:
            closed.append("storage")

    application = AgentApplication(
        agent=Agent(),  # type: ignore[arg-type]
        inventory=Resource("inventory"),  # type: ignore[arg-type]
        fences=Resource("fences"),  # type: ignore[arg-type]
        spool=Resource("spool"),  # type: ignore[arg-type]
        storage_close=Storage().close,
    )

    await application.run()

    assert closed == ["ran", "inventory", "fences", "spool", "storage"]


async def test_application_configures_ray_signals_and_coalesces_repeated_sigterm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: dict[int, Any] = {}
    stop = asyncio.Event()

    class Agent:
        shutdown_calls = 0

        async def run(self) -> None:
            assert os.environ["RAY_DISABLE_FAILURE_SIGNAL_HANDLER"] == "1"
            handlers[signal.SIGTERM]()
            handlers[signal.SIGTERM]()
            await stop.wait()

        async def shutdown(self, *, stop_ray: bool) -> None:
            assert stop_ray
            self.shutdown_calls += 1
            await asyncio.sleep(0)
            stop.set()

    class Resource:
        def close(self) -> None:
            return None

    loop = asyncio.get_running_loop()
    monkeypatch.setenv("RAY_DISABLE_FAILURE_SIGNAL_HANDLER", "0")
    monkeypatch.setattr(
        loop,
        "add_signal_handler",
        lambda signum, callback: handlers.__setitem__(signum, callback),
    )
    agent = Agent()
    application = AgentApplication(
        agent=agent,  # type: ignore[arg-type]
        inventory=Resource(),  # type: ignore[arg-type]
        fences=Resource(),  # type: ignore[arg-type]
        spool=Resource(),  # type: ignore[arg-type]
    )

    await application.run()

    assert agent.shutdown_calls == 1


async def test_application_owns_control_tunnel_for_exact_agent_lifetime() -> None:
    events: list[str] = []

    class Agent:
        async def run(self) -> None:
            events.append("agent-run")

        async def shutdown(self, *, stop_ray: bool) -> None:
            assert stop_ray

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(self.name)

    class Tunnel:
        def ensure_running(self) -> None:
            events.append("tunnel-start")

        def stop(self) -> bool:
            events.append("tunnel-stop")
            return True

    application = AgentApplication(
        agent=Agent(),  # type: ignore[arg-type]
        inventory=Resource("inventory"),  # type: ignore[arg-type]
        fences=Resource("fences"),  # type: ignore[arg-type]
        spool=Resource("spool"),  # type: ignore[arg-type]
        control_tunnel=Tunnel(),  # type: ignore[arg-type]
    )

    await application.run()

    assert events == [
        "tunnel-start",
        "agent-run",
        "inventory",
        "fences",
        "spool",
        "tunnel-stop",
    ]


async def test_slow_bulk_maintenance_cannot_block_control_supervision(monkeypatch):
    import threading

    started, release = threading.Event(), threading.Event()
    ready = asyncio.Event()
    loop = asyncio.get_running_loop()
    control_ticks = []
    original_wait = asyncio.wait_for

    async def quick_wait(awaitable, timeout):  # noqa: ASYNC109 -- matching asyncio API for test clock
        return await original_wait(awaitable, 0.01 if timeout == 5 else timeout)

    monkeypatch.setattr(site_agent_app.asyncio, "wait_for", quick_wait)

    def maintenance():
        started.set()
        assert release.wait(3)

    class Agent:
        async def run(self):
            try:
                async with asyncio.timeout(2):
                    await ready.wait()
            finally:
                release.set()

        async def shutdown(self, *, stop_ray):
            pytest.fail("bulk maintenance must not restart computation")

    class Resource:
        def close(self):
            pass

    class Tunnel:
        def ensure_running(self):
            control_ticks.append(True)
            if started.is_set() and len(control_ticks) >= 4:
                loop.call_soon_threadsafe(ready.set)

        def stop(self):
            pass

    resource = Resource()
    application = AgentApplication(
        agent=Agent(),
        inventory=resource,
        fences=resource,
        spool=resource,
        control_tunnel=Tunnel(),
        storage_maintenance=maintenance,
    )
    await application.run()
    assert started.is_set() and len(control_ticks) >= 4


async def test_abnormal_application_exit_stops_runtime_before_local_cleanup() -> None:
    events: list[str] = []

    class Agent:
        async def run(self) -> None:
            events.append("run")
            raise RuntimeError("background failure")

        async def shutdown(self, *, stop_ray: bool) -> None:
            assert stop_ray
            events.append("shutdown-ray")

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(self.name)

    class Storage:
        def close(self) -> None:
            events.append("storage")

    application = AgentApplication(
        agent=Agent(),  # type: ignore[arg-type]
        inventory=Resource("inventory"),  # type: ignore[arg-type]
        fences=Resource("fences"),  # type: ignore[arg-type]
        spool=Resource("spool"),  # type: ignore[arg-type]
        storage_close=Storage().close,
    )

    with pytest.raises(RuntimeError, match="background failure"):
        await application.run()

    assert events == [
        "run",
        "shutdown-ray",
        "inventory",
        "fences",
        "spool",
        "storage",
    ]


def test_partial_application_build_rolls_back_storage_and_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    inventory_options: dict[str, object] = {}

    class Inventory:
        def sample(self) -> tuple[object, ...]:
            return (object(),)

        def close(self) -> None:
            events.append("inventory-close")

    class Storage:
        def healthcheck(self) -> bool:
            return True

        def resolve(self, logical):
            return tmp_path / "global" / logical

        def resolve_writable(self, logical):
            return tmp_path / "global" / logical

        def close(self):
            events.append("storage-close")

    inventory = Inventory()

    def inventory_factory(*_args: object, **options: object) -> Inventory:
        inventory_options.update(options)
        return inventory

    monkeypatch.setattr(site_agent_app, "NVMLInventoryProvider", inventory_factory)
    monkeypatch.setattr(site_agent_app, "_storage", lambda _settings: Storage())
    monkeypatch.setattr(
        site_agent_app,
        "_model_registry",
        lambda _settings: (_ for _ in ()).throw(RuntimeError("injected build failure")),
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "enrollment_token": "enrollment-for-test",
            "gpu_reservation_lock_dir": tmp_path / "gpu_runtime_locks",
        }
    )

    with pytest.raises(RuntimeError, match="injected build failure"):
        build_application(settings)

    assert events == ["storage-close", "inventory-close"]
    assert isinstance(
        inventory_options["reservation_detector"],
        GPUReservationLockReader,
    )


def test_ray_cluster_is_wired_to_exact_uv_interpreter_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_directory = tmp_path / "venv" / "bin"
    binary_directory.mkdir(parents=True)
    python = binary_directory / "python"
    ray = binary_directory / "ray"
    python.touch()
    ray.write_text("#!/bin/sh\n", encoding="utf-8")
    ray.chmod(0o700)
    monkeypatch.setattr(site_agent_app.sys, "executable", str(python))

    assert _ray_executable() == str(ray)
    short_temp_root = tmp_path / "short-ray"
    config = _ray_cluster_config(
        ("GPU-A", "GPU-B"),
        tmp_path / "ray-state",
        short_temp_root=short_temp_root,
    )
    assert config.ray_executable == str(ray)
    assert config.command_prefix == (str(python), "-m", "ray.scripts.scripts")
    assert config.gpu_uuids == ("GPU-A", "GPU-B")
    assert config.short_temp_root == short_temp_root


def test_ray_cluster_overlay_does_not_follow_external_console_script_shebang(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_directory = tmp_path / "venv" / "bin"
    binary_directory.mkdir(parents=True)
    python = binary_directory / "python"
    python.touch()
    system_ray = tmp_path / "usr" / "local" / "bin" / "ray"
    system_ray.parent.mkdir(parents=True)
    system_ray.write_text("#!/usr/bin/python3.12\n", encoding="utf-8")
    system_ray.chmod(0o700)
    ray = binary_directory / "ray"
    ray.symlink_to(system_ray)
    monkeypatch.setattr(site_agent_app.sys, "executable", str(python))

    assert _ray_executable() == str(ray)
    config = _ray_cluster_config(("GPU-A",), tmp_path / "ray-state")

    assert config.ray_executable == str(ray)
    assert config.command_prefix == (str(python), "-m", "ray.scripts.scripts")
    assert config.command_prefix[0] != str(ray.resolve())
