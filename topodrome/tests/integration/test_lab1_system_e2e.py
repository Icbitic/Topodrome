from __future__ import annotations

import asyncio
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import psutil
import pytest
import yaml

from topodrome.client import Client
from topodrome.db.session import Database
from topodrome.domain.enums import JobState
from topodrome.domain.specs import JobSpec
from topodrome.site_agent import NVMLInventoryProvider, ProcessOwnershipRegistry
from topodrome.storage import LocalStorage, resolve_committed_artifact

pytestmark = [pytest.mark.gpu, pytest.mark.ray, pytest.mark.slow]

_ACK = "RUN_LAB1_SYSTEM_E2E"
_SHARDED_ACK = "RUN_LAB1_SYSTEM_SHARDED_E2E"
_MODEL = Path("/ssd/models/Meta-Llama-3-8B-Instruct")
_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
_GIB = 1024**3


def _require_opt_in() -> None:
    if os.environ.get("TOPODROME_RUN_LAB1_SYSTEM_E2E") != _ACK:
        pytest.skip(
            "set TOPODROME_RUN_LAB1_SYSTEM_E2E=RUN_LAB1_SYSTEM_E2E "
            "to run the complete Lab1 scheduler/agent/GPU smoke"
        )


def _model_sharded() -> bool:
    value = os.environ.get("TOPODROME_RUN_LAB1_SYSTEM_SHARDED_E2E")
    if value is None:
        return False
    if value != _SHARDED_ACK:
        pytest.fail(
            "TOPODROME_RUN_LAB1_SYSTEM_SHARDED_E2E must use its exact opt-in value"
        )
    return True


def _eligible_gpu() -> tuple[int, str]:
    forced = os.environ.get("TOPODROME_LAB1_GPU_INDEX")
    allowed = (int(forced),) if forced is not None else tuple(range(8))
    required = round(
        float(os.environ.get("TOPODROME_SYSTEM_E2E_FREE_VRAM_GIB", "20")) * _GIB
    )
    inventory = NVMLInventoryProvider(
        ProcessOwnershipRegistry(),
        allowed_gpu_indices=allowed,
    )
    try:
        candidates = tuple(
            snapshot
            for snapshot in inventory.sample()
            if snapshot.free_memory_bytes >= required
        )
    finally:
        inventory.close()
    if not candidates:
        pytest.skip("no Lab1 GPU currently has enough physical free VRAM")
    selected = max(candidates, key=lambda value: (value.free_memory_bytes, -value.index))
    return selected.index, selected.uuid


def _eligible_gpu_group() -> tuple[tuple[int, str], ...]:
    raw_forced = os.environ.get("TOPODROME_LAB1_SYSTEM_SHARDED_GPU_INDICES")
    if raw_forced is None:
        allowed = tuple(range(8))
    else:
        try:
            allowed = tuple(
                int(value.strip())
                for value in raw_forced.split(",")
                if value.strip()
            )
        except ValueError:
            pytest.fail("sharded GPU indices must be comma-separated integers")
        if len(allowed) != 2 or len(set(allowed)) != 2 or min(allowed) < 0:
            pytest.fail("the sharded system smoke requires exactly two unique GPU indices")
    required = round(
        float(
            os.environ.get(
                "TOPODROME_SYSTEM_E2E_SHARDED_FREE_VRAM_GIB",
                "11",
            )
        )
        * _GIB
    )
    maximum_utilization = float(
        os.environ.get("TOPODROME_SYSTEM_E2E_MAX_UTILIZATION_PERCENT", "5")
    )
    if not 0 <= maximum_utilization <= 100:
        pytest.fail("the system smoke utilization threshold must be between 0 and 100")
    inventory = NVMLInventoryProvider(
        ProcessOwnershipRegistry(),
        allowed_gpu_indices=allowed,
    )
    try:
        candidates = tuple(
            snapshot
            for snapshot in inventory.sample()
            if snapshot.free_memory_bytes >= required
            and snapshot.utilization_percent <= maximum_utilization
        )
    finally:
        inventory.close()
    if len(candidates) < 2:
        pytest.skip(
            "fewer than two Lab1 GPUs meet the sharded smoke's physical VRAM "
            "and utilization limits"
        )
    selected = sorted(
        candidates,
        key=lambda value: (
            value.free_memory_bytes,
            -value.utilization_percent,
            -value.index,
        ),
        reverse=True,
    )[:2]
    return tuple(
        (snapshot.index, snapshot.uuid)
        for snapshot in sorted(selected, key=lambda value: value.index)
    )


def _ray_processes() -> dict[int, str]:
    result: dict[int, str] = {}
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            name = str(process.info.get("name") or "")
            command = " ".join(process.info.get("cmdline") or ())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if name in {"gcs_server", "raylet"} or "runtime_env_agent" in command:
            result[int(process.info["pid"])] = command
    return result


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _safe_process_output(path: Path, secrets_to_redact: tuple[str, ...]) -> str:
    try:
        value = path.read_text(encoding="utf-8", errors="replace")[-20_000:]
    except OSError:
        return "<process output unavailable>"
    for secret in secrets_to_redact:
        value = value.replace(secret, "<redacted>")
    return value


def _stop_process(process: subprocess.Popen[bytes], *, timeout: float = 90) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


@contextmanager
def _process(
    arguments: list[str],
    *,
    environment: dict[str, str],
    output: Path,
) -> Iterator[subprocess.Popen[bytes]]:
    with output.open("wb") as stream:
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter/module arguments
            arguments,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            yield process
        finally:
            _stop_process(process)


def _wait_until(
    operation: Callable[[], Any],
    predicate: Callable[[Any], bool],
    *,
    timeout: float,
    processes: tuple[subprocess.Popen[bytes], ...] = (),
) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"managed process exited with {process.returncode}")
        try:
            last = operation()
            if predicate(last):
                return last
        except (httpx.HTTPError, RuntimeError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"condition was not reached; last observation: {last!r}")


def _artifact(global_root: Path, job_id: str) -> dict[str, Any]:
    path = resolve_committed_artifact(
        LocalStorage(global_root, create=False),
        job_id=job_id,
        logical_path="actor-identity.json",
    )
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _model_device_ordinals(identity: dict[str, Any]) -> set[int]:
    raw = identity.get("model_device_map")
    if not isinstance(raw, dict):
        raise AssertionError("actor artifact has no model device map")
    ordinals: set[int] = set()
    for device in raw.values():
        value = str(device).casefold()
        if value.isdigit():
            ordinals.add(int(value))
        elif value.startswith("cuda:") and value.removeprefix("cuda:").isdigit():
            ordinals.add(int(value.removeprefix("cuda:")))
        else:
            raise AssertionError(f"model device map escaped CUDA: {value!r}")
    return ordinals


def _write_configuration(
    root: Path,
    *,
    scheduler_port: int,
    gpu_indices: tuple[int, ...],
    model_sharded: bool,
) -> tuple[Path, Path, Path, Path]:
    global_root = root / "global"
    scratch_root = root / "scratch"
    source_root = root / "source"
    for directory in (global_root, scratch_root, source_root):
        directory.mkdir(mode=0o700, parents=True)
    model_config = root / "models.yaml"
    model_config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "meta-llama-3-8b-instruct": {
                        "revision": _REVISION,
                        "backend": "transformers",
                        "backend_version": "4.57.6",
                        "dtype": "bfloat16",
                        "min_vram_gb": 15,
                        "sites": {
                            "lab1-system-e2e": {
                                "path": str(_MODEL),
                                "local_files_only": True,
                            }
                        },
                    }
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    database_path = root / "scheduler.sqlite3"
    scheduler = root / "scheduler.yaml"
    scheduler.write_text(
        yaml.safe_dump(
            {
                "database_dsn": f"sqlite+aiosqlite:///{database_path}",
                "api_host": "127.0.0.1",
                "api_port": scheduler_port,
                "public_url": f"http://127.0.0.1:{scheduler_port}",
                "global_root": str(global_root),
                "heartbeat_ttl_seconds": 10,
                "offer_ttl_seconds": 30,
                "lease_ttl_seconds": 120,
                "dispatch_interval_seconds": 0.2,
                "model_config": str(model_config),
                "autodl": {"enabled": False},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    site = root / "site.yaml"
    site.write_text(
        yaml.safe_dump(
            {
                "site_id": "lab1-system-e2e",
                "display_name": "Lab1 system E2E",
                "scheduler_url": f"http://127.0.0.1:{scheduler_port}",
                "site_kind": "lab",
                "allowed_gpu_indices": list(gpu_indices),
                "global_root": str(global_root),
                "scratch_root": str(scratch_root),
                # Pytest's temporary root can exceed Ray's AF_UNIX socket
                # budget after Ray appends its session and sockets paths.
                # Keep Ray's data in scratch through the validated short alias.
                "ray_short_temp_root": "/tmp",  # noqa: S108 -- private validated alias
                "spool_path": str(root / "agent-spool.sqlite3"),
                "heartbeat_interval_seconds": 0.5,
                # The sharded manifest reserves 10 GiB per device and admits
                # at 11 GiB, leaving its own explicit one-GiB headroom.
                "gpu_safety_margin_gb": 0 if model_sharded else 5,
                "allow_foreign_processes_when_capacity_sufficient": True,
                # The harness never signals foreign processes. Keep a
                # capacity-safe low-utilization selection from being displaced
                # by the normal shared-site pressure policy during this smoke.
                "foreign_pressure_grace_seconds": 7200,
                "model_config": str(model_config),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source = '''from __future__ import annotations

import json
import os


def run(context, *, label, selected_gpu_uuids):
    raw_device_map = getattr(context.model, "hf_device_map", {})
    device_map = (
        {str(name): str(device) for name, device in raw_device_map.items()}
        if isinstance(raw_device_map, dict)
        else {}
    )
    value = {
        "label": label,
        "worker_pid": os.getpid(),
        "model_object_id": id(context.model),
        "job_id": context.job_id,
        "attempt_id": context.attempt_id,
        "device": str(context.device),
        "selected_gpu_uuids": list(selected_gpu_uuids),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","),
        "model_device_map": device_map,
    }
    output = context.workspace / "actor-identity.json"
    output.write_text(json.dumps(value, sort_keys=True) + "\\n", encoding="utf-8")
    context.register_artifact(output, logical_path="actor-identity.json")
    print(f"TOPODROME_SYSTEM_E2E_EXECUTED:{label}")
'''
    (source_root / "experiment.py").write_text(source, encoding="utf-8")
    return scheduler, site, model_config, source_root


def _job(
    label: str,
    *,
    selected_gpu_uuids: tuple[str, ...],
    model_sharded: bool,
) -> JobSpec:
    model: str | dict[str, Any] = "meta-llama-3-8b-instruct"
    resources: dict[str, int] = {
        "gpu": 1,
        "min_vram_gb": 15,
        "cpu": 1,
        "memory_gb": 2,
    }
    if model_sharded:
        model = {
            "alias": "meta-llama-3-8b-instruct",
            "parallelism": {
                "mode": "model_sharded",
                "degree": 2,
                "engine": "transformers_device_map",
                "strategy": "balanced",
                "max_memory_gb_per_gpu": 10,
            },
        }
        resources = {
            "gpu": 2,
            "min_vram_gb": 11,
            "cpu": 1,
            "memory_gb": 2,
        }
    return JobSpec(
        entrypoint="experiment.py",
        model=model,
        resources=resources,
        kwargs={
            "label": label,
            "selected_gpu_uuids": list(selected_gpu_uuids),
        },
        required_outputs=("actor-identity.json",),
        affinity_site_id="lab1-system-e2e",
        retry={"max_attempts": 1},
    )


def test_lab1_scheduler_agent_python_and_cli_reuse_one_real_llama_actor(
    tmp_path: Path,
) -> None:
    _require_opt_in()
    if not (_MODEL / "config.json").is_file():
        pytest.fail("the configured Lab1 Llama payload is unavailable")
    existing_ray = _ray_processes()
    if existing_ray:
        pytest.skip("a pre-existing Ray cluster is outside this test's ownership boundary")
    model_sharded = _model_sharded()
    selected_gpus = _eligible_gpu_group() if model_sharded else (_eligible_gpu(),)
    gpu_indices = tuple(index for index, _uuid in selected_gpus)
    gpu_uuids = tuple(uuid for _index, uuid in selected_gpus)
    port = _free_port()
    scheduler_config, site_config, _models, source_root = _write_configuration(
        tmp_path,
        scheduler_port=port,
        gpu_indices=gpu_indices,
        model_sharded=model_sharded,
    )
    global_root = tmp_path / "global"
    dsn = f"sqlite+aiosqlite:///{tmp_path / 'scheduler.sqlite3'}"

    async def create_schema() -> None:
        database = Database(dsn)
        try:
            await database.create_schema()
        finally:
            await database.close()

    asyncio.run(create_schema())
    fence_key = secrets.token_urlsafe(32)
    enrollment = secrets.token_urlsafe(32)
    base_url = f"http://127.0.0.1:{port}"
    base_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TOPODROME_")
        and key not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
    }
    base_environment["PYTHONUNBUFFERED"] = "1"
    scheduler_environment = {
        **base_environment,
        "TOPODROME_FENCE_SIGNING_KEY": fence_key,
        "TOPODROME_AGENT_TOKENS": json.dumps(
            {"lab1-system-e2e": enrollment}, separators=(",", ":")
        ),
    }
    agent_environment = {
        **base_environment,
        "TOPODROME_SITE_ENROLLMENT_TOKEN": enrollment,
        "TOPODROME_SITE_FENCE_VERIFICATION_KEY": fence_key,
    }
    scheduler_log = tmp_path / "scheduler.log"
    agent_log = tmp_path / "agent.log"
    redactions = (fence_key, enrollment)
    identity_one: dict[str, Any] | None = None
    identity_two: dict[str, Any] | None = None
    first_members: tuple[str, ...] = ()
    second_members: tuple[str, ...] = ()
    first_warm_members: set[str] = set()
    second_warm_members: set[str] = set()

    try:
        with _process(
            [
                sys.executable,
                "-m",
                "topodrome.scheduler.main",
                "--config",
                str(scheduler_config),
            ],
            environment=scheduler_environment,
            output=scheduler_log,
        ) as scheduler_process:
            _wait_until(
                lambda: httpx.get(f"{base_url}/health/ready", timeout=2),
                lambda response: response.status_code == 200,
                timeout=30,
                processes=(scheduler_process,),
            )
            with _process(
                [
                    sys.executable,
                    "-m",
                    "topodrome.site_agent.app",
                    "--config",
                    str(site_config),
                ],
                environment=agent_environment,
                output=agent_log,
            ) as agent_process:
                processes = (scheduler_process, agent_process)

                def site_ready() -> list[dict[str, Any]]:
                    with Client(base_url) as client:
                        return client.sites()

                sites = _wait_until(
                    site_ready,
                    lambda rows: bool(rows) and rows[0]["state"] == "ready",
                    timeout=120,
                    processes=processes,
                )
                assert sites[0]["site_id"] == "lab1-system-e2e"
                with Client(base_url) as client:
                    reported_gpu_uuids = {
                        str(gpu["uuid"])
                        for gpu in client.gpus()
                        if gpu["site_id"] == "lab1-system-e2e"
                    }
                    assert reported_gpu_uuids == set(gpu_uuids)
                    first = client.submit(
                        _job(
                            "python-sdk",
                            selected_gpu_uuids=gpu_uuids,
                            model_sharded=model_sharded,
                        ),
                        source_dir=source_root,
                        idempotency_key="lab1-system-e2e-python",
                    )
                    first_summary = first.wait(poll_seconds=0.5, timeout=900)
                    first_attempts = first.attempts()
                    assert first_summary.state is JobState.SUCCEEDED, first_attempts
                    assert len(first_attempts) == 1
                    first_members = tuple(first_attempts[0]["gpu_uuids"])
                    assert "TOPODROME_SYSTEM_E2E_EXECUTED:python-sdk" in first.logs()
                    identity_one = _artifact(global_root, first.job_id)

                def models() -> list[dict[str, Any]]:
                    with Client(base_url) as client:
                        return client.models()

                first_model_rows = _wait_until(
                    models,
                    lambda rows: len(
                        {
                            row["gpu_uuid"]
                            for row in rows
                            if row["availability"] == "warm"
                            and row["gpu_uuid"] is not None
                        }
                    )
                    == len(gpu_uuids),
                    timeout=30,
                    processes=processes,
                )
                first_warm_members = {
                    str(row["gpu_uuid"])
                    for row in first_model_rows
                    if row["availability"] == "warm"
                    and row["gpu_uuid"] is not None
                }
                manifest = tmp_path / "cli-job.yaml"
                manifest.write_text(
                    yaml.safe_dump(
                        {
                            "source_dir": "source",
                            **_job(
                                "cli",
                                selected_gpu_uuids=gpu_uuids,
                                model_sharded=model_sharded,
                            ).model_dump(mode="json", exclude={"source_ref"}),
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                cli_environment = {**base_environment, "TOPODROME_URL": base_url}
                completed = subprocess.run(  # noqa: S603 - fixed interpreter/module
                    [
                        sys.executable,
                        "-m",
                        "topodrome.client.cli",
                        "submit",
                        str(manifest),
                        "--wait",
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    env=cli_environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                assert completed.returncode == 0, completed.stderr[-4000:]
                second_summary = json.loads(completed.stdout)
                assert second_summary["state"] == "succeeded"
                second_job_id = str(second_summary["job_id"])
                identity_two = _artifact(global_root, second_job_id)
                with Client(base_url) as client:
                    assert "TOPODROME_SYSTEM_E2E_EXECUTED:cli" in client.logs(second_job_id)
                    second_attempts = client.attempts(second_job_id)
                    assert len(second_attempts) == 1
                    second_members = tuple(second_attempts[0]["gpu_uuids"])
                    second_warm_members = {
                        str(row["gpu_uuid"])
                        for row in client.models()
                        if row["availability"] == "warm"
                        and row["gpu_uuid"] is not None
                    }
    except Exception as exc:
        pytest.fail(
            f"{type(exc).__name__}: {exc}\n"
            f"scheduler:\n{_safe_process_output(scheduler_log, redactions)}\n"
            f"agent:\n{_safe_process_output(agent_log, redactions)}"
        )
    finally:
        # The site agent normally stops its Ray cluster. If a test-side crash
        # interrupts that path, touch only processes whose command line names
        # this exact pytest-owned scratch root.
        marker = str((tmp_path / "scratch").resolve())
        leftovers = {
            pid: command
            for pid, command in _ray_processes().items()
            if marker in command and pid not in existing_ray
        }
        for pid in leftovers:
            try:
                psutil.Process(pid).terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
        _gone, alive = psutil.wait_procs(
            [psutil.Process(pid) for pid in leftovers if psutil.pid_exists(pid)],
            timeout=10,
        )
        for process in alive:
            try:
                process.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass

    assert identity_one is not None and identity_two is not None
    assert identity_one["worker_pid"] == identity_two["worker_pid"]
    assert identity_one["model_object_id"] == identity_two["model_object_id"]
    assert identity_one["attempt_id"] != identity_two["attempt_id"]
    assert len(first_members) == len(gpu_uuids)
    assert second_members == first_members
    assert set(first_members) == set(gpu_uuids)
    assert first_warm_members == set(gpu_uuids)
    assert second_warm_members == set(gpu_uuids)
    assert set(identity_one["selected_gpu_uuids"]) == set(gpu_uuids)
    assert identity_two["selected_gpu_uuids"] == identity_one["selected_gpu_uuids"]
    if model_sharded:
        index_by_uuid = {uuid: index for index, uuid in selected_gpus}
        expected_visible = [str(index_by_uuid[uuid]) for uuid in first_members]
        assert identity_one["cuda_visible_devices"] == expected_visible
        assert identity_two["cuda_visible_devices"] == expected_visible
        assert _model_device_ordinals(identity_one) == {0, 1}
        assert _model_device_ordinals(identity_two) == {0, 1}
        assert identity_two["model_device_map"] == identity_one["model_device_map"]
