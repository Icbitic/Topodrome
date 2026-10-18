"""Run one real model twice through a reusable site-local Ray actor.

This script is copied to disposable workers by the integration harness.  It
prints one base64-encoded JSON result and never prints credentials or model
filesystem paths.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from topodrome.config import load_secret_bundle
from topodrome.runtime import (
    ActorKey,
    DetachedRayActorAdapter,
    HuggingFaceModelAcquirer,
    ModelKey,
    ModelPayloadVerificationError,
    ModelRegistration,
    RayActorSpec,
    TransformersLoader,
    ray_gpu_resource_name,
    verify_transformers_payload,
)
from topodrome.site_agent import NVMLInventoryProvider, ProcessOwnershipRegistry

RESULT_PREFIX = "TOPODROME_MODEL_RESULT="
ERROR_PREFIX = "TOPODROME_MODEL_ERROR="
GIB = 1024**3
_STAGE = "startup"


def _inference(context: Any) -> dict[str, Any]:
    torch = importlib.import_module("torch")

    tokenizer = context.tokenizer
    model = context.model
    inputs = tokenizer("Reply with one word: ready", return_tensors="pt")
    input_device = next(model.parameters()).device
    device_inputs = {name: value.to(input_device) for name, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **device_inputs,
            do_sample=False,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return {
        "device": str(input_device),
        "generated_tokens": int(generated.shape[-1] - device_inputs["input_ids"].shape[-1]),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_devices": sorted(
            {str(device) for device in getattr(model, "hf_device_map", {}).values()}
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision-manifest", type=Path)
    parser.add_argument("--origin")
    parser.add_argument("--secret-file", type=Path)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--ignore-pattern", action="append", default=[])
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--allowed-gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--min-free-vram-gib", type=float, default=20.0)
    parser.add_argument(
        "--sharding-strategy",
        choices=("balanced", "balanced_low_0", "sequential"),
        default="balanced",
    )
    parser.add_argument("--sharded-max-memory-gib", type=float, default=10.0)
    return parser


def _select_gpus(args: argparse.Namespace) -> tuple[Any, ...]:
    if not 1 <= args.gpu_count <= 4:
        raise ValueError("GPU count must be between one and four")
    if args.gpu_index is not None and args.gpu_count != 1:
        raise ValueError("a fixed GPU index is valid only for a one-GPU smoke")
    allowed = tuple(int(value) for value in args.allowed_gpus.split(",") if value.strip())
    if args.gpu_index is not None:
        allowed = (args.gpu_index,)
    inventory = NVMLInventoryProvider(
        ProcessOwnershipRegistry(),
        allowed_gpu_indices=allowed,
    )
    try:
        candidates = tuple(
            gpu
            for gpu in inventory.sample()
            if gpu.free_memory_bytes >= int(args.min_free_vram_gib * GIB)
        )
    finally:
        inventory.close()
    if len(candidates) < args.gpu_count:
        raise RuntimeError("too few admitted GPUs have the required physical free VRAM")
    selected = sorted(
        candidates,
        key=lambda gpu: (gpu.free_memory_bytes, -gpu.index),
        reverse=True,
    )[: args.gpu_count]
    return tuple(sorted(selected, key=lambda gpu: gpu.index))


def _registration(args: argparse.Namespace, key: ModelKey) -> ModelRegistration:
    registration = ModelRegistration(
        key=key,
        origin=args.origin,
        local_path=args.model_path,
    )
    try:
        verified = verify_transformers_payload(
            args.model_path,
            revision=key.revision,
            verify_lfs_sha256=True,
            revision_manifest=args.revision_manifest,
        )
    except ModelPayloadVerificationError:
        verified = None
    if verified is not None:
        return ModelRegistration(
            key=key,
            origin=args.origin,
            local_path=args.model_path,
            payload_identity=verified.identity_digest,
        )
    if not args.origin or args.secret_file is None:
        raise RuntimeError(
            "model cache is incomplete and no credential-backed origin was configured"
        )
    secrets = load_secret_bundle(args.secret_file)
    if secrets.hf_token is None:
        raise RuntimeError("Hugging Face token is unavailable")
    acquirer = HuggingFaceModelAcquirer(
        destination=args.model_path,
        endpoint=args.endpoint,
        token_provider=lambda: secrets.hf_token,
        ignore_patterns=tuple(args.ignore_pattern),
    )
    return acquirer.acquire(registration)


def main() -> int:
    global _STAGE

    args = _parser().parse_args()
    if len(args.run_id) != 12 or any(
        character not in "0123456789abcdef" for character in args.run_id
    ):
        raise ValueError("run ID must be twelve lowercase hexadecimal characters")
    _STAGE = "physical_gpu_inventory"
    gpus = _select_gpus(args)
    key = ModelKey(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        backend="transformers",
        backend_version="4.57.6",
        dtype="bfloat16",
    )
    _STAGE = "model_registration"
    registration = _registration(args, key)
    loader = TransformersLoader(registration, device="cuda:0")
    if len(gpus) > 1:
        if args.sharded_max_memory_gib <= 0:
            raise ValueError("sharded per-GPU memory must be positive")
        loader = loader.for_model_sharding(
            degree=len(gpus),
            strategy=args.sharding_strategy,
            max_memory_bytes_per_gpu=round(args.sharded_max_memory_gib * GIB),
        )

    # Site environments are already resolved and locked by uv. Ray 2.58's
    # automatic uv-run propagation can otherwise create a nested worker env
    # that omits Ray itself when this driver was launched by ``uv run``.
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray = importlib.import_module("ray")

    resources = {ray_gpu_resource_name(gpu.uuid): 1 for gpu in gpus}
    _STAGE = "ray_start"
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        resources=resources,
        include_dashboard=False,
        logging_level=logging.ERROR,
    )
    adapter = DetachedRayActorAdapter(ray)
    gpu_uuids = tuple(str(gpu.uuid) for gpu in gpus)
    spec = RayActorSpec(
        key=ActorKey(
            site_id=args.site_id,
            gpu_uuid=gpu_uuids[0],
            model=key,
            replica_gpu_uuids=gpu_uuids,
        ),
        cuda_visible_device=",".join(gpu_uuids),
        namespace=f"topodrome_it_{args.run_id}",
    )
    first = None
    try:
        _STAGE = "actor_prepare"
        first = adapter.get_or_create(spec, loader)
        _STAGE = "first_inference"
        first_result = first.run(_inference, job_id="smoke", attempt_id="smoke-1")
        _STAGE = "actor_reuse"
        second = adapter.get_or_create(spec, loader)
        _STAGE = "second_inference"
        second_result = second.run(_inference, job_id="smoke", attempt_id="smoke-2")
        first_description = first.describe()
        second_description = second.describe()
        executions = int(second_description["executions"])
        result = {
            "actor_id_equal": first_description["actor_id"] == second_description["actor_id"],
            # A sharded replica is one actor, so this is also the exact number
            # of user-entrypoint invocations rather than degree * requests.
            "executions": executions,
            "entrypoint_calls": executions,
            "first_generated_tokens": int(first_result["generated_tokens"]),
            "gpu_count": len(gpus),
            "gpu_free_before_bytes": int(gpus[0].free_memory_bytes),
            "gpu_free_before_bytes_all": [int(gpu.free_memory_bytes) for gpu in gpus],
            "gpu_index": int(gpus[0].index),
            "gpu_indices": [int(gpu.index) for gpu in gpus],
            "gpu_name": str(gpus[0].name),
            "gpu_names": [str(gpu.name) for gpu in gpus],
            "model_devices": list(second_result["model_devices"]),
            "parameter_count": int(second_result["parameter_count"]),
            "reused": bool(second.reused),
        }
        encoded = base64.urlsafe_b64encode(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        print(f"{RESULT_PREFIX}{encoded}", flush=True)
        _STAGE = "complete"
        return 0
    finally:
        if first is not None:
            with suppress(Exception):
                adapter.evict(spec, force=True)
        ray.shutdown()


if __name__ == "__main__":
    try:
        exit_code = main()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        failure = {
            "error_type": type(error).__name__,
            "stage": _STAGE,
        }
        encoded_failure = base64.urlsafe_b64encode(
            json.dumps(failure, sort_keys=True, separators=(",", ":")).encode()
        ).decode("ascii")
        print(f"{ERROR_PREFIX}{encoded_failure}", flush=True)
        exit_code = 1
    raise SystemExit(exit_code)
