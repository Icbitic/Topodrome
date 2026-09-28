from __future__ import annotations

import json


def _model_devices(model: object) -> list[str]:
    """Return the Transformers placement without assuming a sharded model."""

    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return []
    return sorted({str(device) for device in device_map.values()})


def run(context, *, label: str = "llama-smoke") -> dict[str, object]:
    """Write one small, durable artifact without copying model weights."""

    summary: dict[str, object] = {
        "label": label,
        "job_id": context.job_id,
        "attempt_id": context.attempt_id,
        "model_class": type(context.model).__name__,
        "device": str(context.device),
        "model_devices": _model_devices(context.model),
    }
    output = context.workspace / "model-summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    context.register_artifact(output, logical_path="model-summary.json")
    return summary
