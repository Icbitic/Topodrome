from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import suppress
from pathlib import Path

import pytest

from scripts.integration_support import (
    REMOTE_JUPYTER_ACK,
    ManagedJupyterClient,
    create_source_archive,
    decode_result,
    integration_secrets,
    new_run_id,
    opted_in,
    python_terminal_command,
)

pytestmark = [pytest.mark.remote, pytest.mark.gpu]

GIB = 1024**3
GPU_RESULT_PREFIX = "TOPODROME_GPU_RESULT="
MODEL_RESULT_PREFIX = "TOPODROME_MODEL_RESULT="
MODEL_ERROR_PREFIX = "TOPODROME_MODEL_ERROR="
COMMAND_RESULT_PREFIX = "TOPODROME_COMMAND_RESULT="
STAGE_PREFIX = "TOPODROME_STAGE="
JUPYTER_MODEL_PATH = "/mnt/model/ckpts/Meta-Llama-3-8B-Instruct"
LAB1_MODEL_PATH = Path("/ssd/models/Meta-Llama-3-8B-Instruct")
MODEL_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
LAB1_REVISION_MANIFEST = (
    LAB1_MODEL_PATH / ".cache" / "huggingface" / "trees" / f"{MODEL_REVISION}.json"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _remote_probe_source(ready_marker: str) -> str:
    return f"""
import base64
import hashlib
import json
import pathlib
import subprocess

query = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)
gpus = []
for raw in query.stdout.splitlines():
    index, uuid, name, total, free, utilization = [part.strip() for part in raw.split(",", 5)]
    gpus.append({{
        "index": int(index),
        "uuid": uuid,
        "name": name,
        "total_mib": int(total),
        "free_mib": int(free),
        "utilization_percent": int(utilization),
    }})
model_root = pathlib.Path({JUPYTER_MODEL_PATH!r})
def digest(name):
    return hashlib.sha256(model_root.joinpath(name).read_bytes()).hexdigest()
payload = {{
    "gpus": gpus,
    "model_config": model_root.joinpath("config.json").is_file(),
    "config_sha256": digest("config.json"),
    "index_sha256": digest("model.safetensors.index.json"),
    "safetensor_shards": len(tuple(model_root.glob("model-*.safetensors"))),
}}
encoded = base64.urlsafe_b64encode(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).decode()
print({GPU_RESULT_PREFIX!r} + encoded, flush=True)
print({ready_marker!r}, flush=True)
"""


def _checked_shell_command(command: str, ready_marker: str) -> str:
    source = f"""
import base64
import json
import subprocess

completed = subprocess.run(["/bin/sh", "-c", {command!r}], check=False)
payload = base64.urlsafe_b64encode(
    json.dumps({{"exit_code": completed.returncode}}).encode()
).decode()
print({COMMAND_RESULT_PREFIX!r} + payload, flush=True)
print({ready_marker!r}, flush=True)
"""
    return python_terminal_command(source)


def _stage(name: str) -> str:
    return python_terminal_command(f"print({(STAGE_PREFIX + name)!r}, flush=True)")


def _require_remote_opt_in() -> None:
    if not opted_in("TOPODROME_RUN_REMOTE_JUPYTER_TESTS", REMOTE_JUPYTER_ACK):
        pytest.skip(
            "set TOPODROME_RUN_REMOTE_JUPYTER_TESTS=RUN_REMOTE_JUPYTER_TESTS "
            "to contact the configured Jupyter server"
        )


def _lab1_identity() -> tuple[str, str]:
    def digest(name: str) -> str:
        return hashlib.sha256(LAB1_MODEL_PATH.joinpath(name).read_bytes()).hexdigest()

    return digest("config.json"), digest("model.safetensors.index.json")


async def _client() -> ManagedJupyterClient:
    secrets = integration_secrets()
    if secrets.test_jupyter_url is None or secrets.test_jupyter_token is None:
        pytest.fail("protected secret file lacks the remote Jupyter URL or token")
    return ManagedJupyterClient(secrets.test_jupyter_url, secrets.test_jupyter_token)


async def test_terminal_api_probe_finds_shareable_gpu_and_local_model() -> None:
    _require_remote_opt_in()
    run_id = new_run_id()
    ready = f"topodrome_ready_{new_run_id()}"
    client = await _client()
    async with client:
        await client.cleanup_registered_terminals()
        result = await client.run_terminal(
            python_terminal_command(_remote_probe_source(ready)),
            ready_marker=ready,
            run_id=run_id,
            timeout_seconds=60,
        )
    payload = decode_result(result.output, GPU_RESULT_PREFIX)
    gpus = payload.get("gpus")
    assert isinstance(gpus, list) and gpus
    assert any(int(gpu["free_mib"]) >= 20 * 1024 for gpu in gpus)
    assert payload["model_config"] is True
    assert int(payload["safetensor_shards"]) == 4
    local_config, local_index = await asyncio.to_thread(_lab1_identity)
    assert payload["config_sha256"] == local_config
    assert payload["index_sha256"] == local_index


@pytest.mark.ray
@pytest.mark.slow
async def test_terminal_bootstrap_runs_local_llama_through_reused_ray_actor() -> None:
    _require_remote_opt_in()
    if os.environ.get("TOPODROME_RUN_REMOTE_JUPYTER_MODEL_TESTS") != "RUN_REMOTE_MODEL_TEST":
        pytest.skip(
            "set TOPODROME_RUN_REMOTE_JUPYTER_MODEL_TESTS=RUN_REMOTE_MODEL_TEST "
            "to install the uv runtime and load the remote model"
        )
    run_id = new_run_id()
    remote_root = f"topodrome_it/{run_id}"
    archive_path = f"{remote_root}/source.tar.gz"
    ready_setup = f"topodrome_setup_{new_run_id()}"
    ready_model = f"topodrome_model_ready_{new_run_id()}"
    gpu_count = int(os.environ.get("TOPODROME_REMOTE_JUPYTER_MODEL_GPU_COUNT", "1"))
    if gpu_count not in {1, 2}:
        pytest.fail("remote Jupyter model smoke GPU count must be one or two")
    raw_indices = os.environ.get("TOPODROME_JUPYTER_SHARDED_GPU_INDICES", "2,3")
    try:
        sharded_indices = tuple(
            int(value) for value in raw_indices.split(",") if value.strip()
        )
    except ValueError:
        pytest.fail("remote Jupyter sharded GPU indices must be comma-separated integers")
    if len(sharded_indices) != 2 or len(set(sharded_indices)) != 2 or min(sharded_indices) < 0:
        pytest.fail("remote Jupyter sharded smoke requires exactly two unique GPU indices")
    safe_indices = ",".join(str(value) for value in sharded_indices)
    gpu_arguments = (
        "--min-free-vram-gib 20"
        if gpu_count == 1
        else (
            "--gpu-count 2 "
            f"--allowed-gpus {safe_indices} "
            "--min-free-vram-gib 11 --sharded-max-memory-gib 10"
        )
    )
    setup_source = f"""
import pathlib
path = pathlib.Path({remote_root!r})
path.mkdir(mode=0o700, parents=True, exist_ok=True)
path.chmod(0o700)
print({ready_setup!r}, flush=True)
"""
    command = " && ".join(
        (
            "set -eu",
            "set +x",
            f"cd {remote_root}",
            "tar -xzf source.tar.gz",
            "export UV_CACHE_DIR=/tmp/topodrome-uv-cache",
            "export UV_NO_PROGRESS=1",
            _stage("venv"),
            "python3 -m uv venv --clear --system-site-packages --python 3.12 .venv",
            _stage("topodrome_install"),
            # Install Topodrome's lightweight base dependencies as well as the
            # editable package. The remote system environment supplies the
            # heavy Torch/Ray stack, but is not required to preinstall e.g.
            # structlog or pydantic-settings.
            "python3 -m uv pip install --python .venv/bin/python -e .",
            _stage("transformers_install"),
            "python3 -m uv pip install --python .venv/bin/python "
            "'transformers==4.57.6'",
            _stage("model_smoke"),
            ".venv/bin/python scripts/remote_model_smoke.py "
            f"--site-id remote-jupyter-1 --run-id {run_id} "
            f"--model-path {JUPYTER_MODEL_PATH} "
            '--revision-manifest "$PWD/revision-manifest.json" '
            f"{gpu_arguments}",
        )
    )

    client = await _client()
    async with client:
        await client.cleanup_registered_terminals()
        try:
            await client.run_terminal(
                python_terminal_command(setup_source),
                ready_marker=ready_setup,
                run_id=run_id,
                timeout_seconds=60,
            )
            await client.put_file(archive_path, create_source_archive(REPOSITORY_ROOT))
            # This preseeded checkpoint predates Topodrome's local-dir
            # metadata. Keep the trust anchor outside its read-only tree and
            # hash every required remote file against the Lab1 manifest.
            await client.put_file(
                f"{remote_root}/revision-manifest.json",
                await asyncio.to_thread(LAB1_REVISION_MANIFEST.read_bytes),
            )
            result = await client.run_terminal(
                _checked_shell_command(command, ready_model),
                ready_marker=ready_model,
                run_id=run_id,
                timeout_seconds=int(
                    os.environ.get("TOPODROME_REMOTE_MODEL_TIMEOUT_SECONDS", "3600")
                ),
            )
        finally:
            with suppress(Exception):
                await client.delete_path(remote_root)
    command_result = decode_result(result.output, COMMAND_RESULT_PREFIX)
    if int(command_result["exit_code"]) != 0:
        try:
            failure = decode_result(result.output, MODEL_ERROR_PREFIX)
            pytest.fail(
                f"remote model smoke failed during {failure['stage']} "
                f"with {failure['error_type']}"
            )
        except pytest.fail.Exception:
            raise
        except Exception:
            stages = [
                line[position + len(STAGE_PREFIX) :].strip()
                for line in result.output.replace("\r", "\n").splitlines()
                if (position := line.find(STAGE_PREFIX)) >= 0
            ]
            stage = stages[-1] if stages else "bootstrap"
            pytest.fail(f"remote model smoke command failed during {stage}")
    payload = decode_result(result.output, MODEL_RESULT_PREFIX)
    assert payload["actor_id_equal"] is True
    assert payload["reused"] is True
    assert int(payload["executions"]) == 2
    assert int(payload["entrypoint_calls"]) == 2
    assert int(payload["parameter_count"]) > 7_000_000_000
    assert int(payload["gpu_count"]) == gpu_count
    if gpu_count == 2:
        observed = {
            int(str(device).removeprefix("cuda:"))
            for device in payload["model_devices"]
        }
        assert observed == {0, 1}
