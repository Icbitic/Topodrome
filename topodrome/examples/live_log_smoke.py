from __future__ import annotations

import json
import time


def run(context, *, delay_seconds: float = 2.0) -> dict[str, object]:
    """Emit two separated log records and publish one tiny durable artifact."""

    print("topodrome-live: started", flush=True)
    time.sleep(delay_seconds)
    result: dict[str, object] = {
        "attempt_id": context.attempt_id,
        "delay_seconds": delay_seconds,
        "device": str(context.device),
        "job_id": context.job_id,
    }
    output = context.workspace / "live-log-smoke.json"
    output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    context.register_artifact(output, logical_path="live-log-smoke.json")
    print("topodrome-live: artifact registered", flush=True)
    return result
