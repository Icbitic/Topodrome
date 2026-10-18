"""Small GPU-only context example; supports a single process or native ranks."""

import json
from datetime import timedelta

import torch


def run(context):
    assert context.model is None and context.tokenizer is None
    assert all(isinstance(device, torch.device) for device in context.devices)
    if context.world_size > 1:
        import torch.distributed as dist

        dist.init_process_group(
            "nccl",
            rank=context.rank,
            world_size=context.world_size,
            timeout=timedelta(seconds=60),
        )
        context.add_cleanup(dist.destroy_process_group)

    results = []
    for device in context.devices:
        # Use the assigned worker-local device object, never a host GPU index.
        value = torch.full((32, 32), context.rank + 1.0, device=device)
        result = value @ value
        if context.world_size > 1:
            dist.all_reduce(result)
        results.append(float(result[0, 0].item()))
    if context.is_main_process:
        context.report_progress(1, 1, stage="compute", message="GPU computation complete")
        context.checkpoint({"completed": True}, name="gpu-compute")
        output = context.workspace / "gpu-result.json"
        output.write_text(
            json.dumps(
                {
                    "devices": [str(device) for device in context.devices],
                    "rank": context.rank,
                    "world_size": context.world_size,
                    "local_rank": context.local_rank,
                    "results": results,
                }
            )
            + "\n"
        )
        context.register_artifact(output, logical_path="gpu-result.json")
