from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.integration_support import (
    LAB1_GPU_ACK,
    decode_result,
    new_run_id,
    opted_in,
)
from topodrome.site_agent import NVMLInventoryProvider, ProcessOwnershipRegistry

pytestmark = [pytest.mark.gpu, pytest.mark.ray, pytest.mark.slow]

GIB = 1024**3
MODEL_PATH = Path("/ssd/models/Meta-Llama-3-8B-Instruct")
RESULT_PREFIX = "TOPODROME_MODEL_RESULT="


def _require_lab1_opt_in() -> None:
    if not opted_in("TOPODROME_RUN_LAB1_GPU_TESTS", LAB1_GPU_ACK):
        pytest.skip(
            "set TOPODROME_RUN_LAB1_GPU_TESTS=RUN_LAB1_GPU_TESTS to use a Lab1 GPU"
        )


def _eligible_gpu() -> tuple[int, str, int]:
    forced = os.environ.get("TOPODROME_LAB1_GPU_INDEX")
    allowed = (int(forced),) if forced is not None else tuple(range(8))
    required = int(float(os.environ.get("TOPODROME_MODEL_SMOKE_FREE_VRAM_GIB", "20")) * GIB)
    inventory = NVMLInventoryProvider(
        ProcessOwnershipRegistry(),
        allowed_gpu_indices=allowed,
    )
    try:
        candidates = tuple(gpu for gpu in inventory.sample() if gpu.free_memory_bytes >= required)
    finally:
        inventory.close()
    if not candidates:
        pytest.skip("no Lab1 GPU currently has the required physical free VRAM")
    selected = max(candidates, key=lambda gpu: (gpu.free_memory_bytes, -gpu.index))
    return selected.index, selected.uuid, selected.free_memory_bytes


def _eligible_gpu_group() -> tuple[int, ...]:
    forced = os.environ.get("TOPODROME_LAB1_SHARDED_GPU_INDICES", "")
    allowed = (
        tuple(int(value) for value in forced.split(",") if value.strip())
        if forced
        else tuple(range(8))
    )
    required = int(
        float(os.environ.get("TOPODROME_SHARDED_SMOKE_FREE_VRAM_GIB", "11")) * GIB
    )
    inventory = NVMLInventoryProvider(
        ProcessOwnershipRegistry(),
        allowed_gpu_indices=allowed,
    )
    try:
        candidates = tuple(
            gpu for gpu in inventory.sample() if gpu.free_memory_bytes >= required
        )
    finally:
        inventory.close()
    if len(candidates) < 2:
        pytest.skip("fewer than two Lab1 GPUs have sharded-smoke VRAM capacity")
    selected = sorted(
        candidates,
        key=lambda gpu: (gpu.free_memory_bytes, -gpu.index),
        reverse=True,
    )[:2]
    return tuple(sorted(gpu.index for gpu in selected))


def test_lab1_inventory_classifies_unregistered_processes_as_foreign() -> None:
    _require_lab1_opt_in()
    index, _uuid, free_memory = _eligible_gpu()
    assert index in range(8)
    assert free_memory >= 20 * GIB


def test_lab1_llama_ray_actor_is_reused_without_a_second_load() -> None:
    _require_lab1_opt_in()
    if not MODEL_PATH.joinpath("config.json").is_file():
        pytest.fail("configured Lab1 model payload is unavailable")
    index, _uuid, _free_memory = _eligible_gpu()
    run_id = new_run_id()
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and repository-owned script
        [
            sys.executable,
            str(repository_root / "scripts" / "remote_model_smoke.py"),
            "--site-id",
            "lab1",
            "--run-id",
            run_id,
            "--model-path",
            str(MODEL_PATH),
            "--gpu-index",
            str(index),
            "--min-free-vram-gib",
            os.environ.get("TOPODROME_MODEL_SMOKE_FREE_VRAM_GIB", "20"),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("TOPODROME_MODEL_SMOKE_TIMEOUT_SECONDS", "1800")),
    )
    if completed.returncode != 0:
        pytest.fail(f"Lab1 model smoke failed (exit {completed.returncode})")
    result = decode_result(completed.stdout, RESULT_PREFIX)
    assert result["actor_id_equal"] is True
    assert result["reused"] is True
    assert int(result["executions"]) == 2
    assert int(result["first_generated_tokens"]) == 1
    assert int(result["parameter_count"]) > 7_000_000_000


def test_lab1_llama_forced_model_sharding_reuses_one_two_gpu_replica() -> None:
    _require_lab1_opt_in()
    if os.environ.get("TOPODROME_RUN_LAB1_SHARDED_MODEL_TEST") != "RUN_SHARDED_MODEL_TEST":
        pytest.skip("set the dedicated sharded-model opt-in to use two Lab1 GPUs")
    if not MODEL_PATH.joinpath("config.json").is_file():
        pytest.fail("configured Lab1 model payload is unavailable")
    indices = _eligible_gpu_group()
    run_id = new_run_id()
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(  # noqa: S603 - repository-owned script
        [
            sys.executable,
            str(repository_root / "scripts" / "remote_model_smoke.py"),
            "--site-id",
            "lab1",
            "--run-id",
            run_id,
            "--model-path",
            str(MODEL_PATH),
            "--gpu-count",
            "2",
            "--allowed-gpus",
            ",".join(str(index) for index in indices),
            "--min-free-vram-gib",
            os.environ.get("TOPODROME_SHARDED_SMOKE_FREE_VRAM_GIB", "11"),
            "--sharded-max-memory-gib",
            os.environ.get("TOPODROME_SHARDED_SMOKE_MAX_MEMORY_GIB", "10"),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("TOPODROME_MODEL_SMOKE_TIMEOUT_SECONDS", "1800")),
    )
    if completed.returncode != 0:
        pytest.fail(f"Lab1 sharded model smoke failed (exit {completed.returncode})")
    result = decode_result(completed.stdout, RESULT_PREFIX)
    assert result["gpu_indices"] == list(indices)
    assert result["actor_id_equal"] is True
    assert result["reused"] is True
    assert int(result["entrypoint_calls"]) == 2
    observed = {
        int(str(device).removeprefix("cuda:")) for device in result["model_devices"]
    }
    assert observed == {0, 1}
    assert int(result["parameter_count"]) > 7_000_000_000
