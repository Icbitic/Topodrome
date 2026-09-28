"""Safe index expansion and aggregate array following (no mixed log streams)."""

import re
import time
from pathlib import Path

import yaml

from topodrome.domain.arrays import ArrayElement, JobArraySpec


def parse_indices(value: str) -> tuple[ArrayElement, ...]:
    indices = []
    for segment in value.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+)(?::(\d+))?)?", segment.strip())
        if match is None:
            raise ValueError("indices must be like 0-99, 0-20:2, or 1,3,7")
        start = int(match[1])
        end = int(match[2]) if match[2] is not None else start
        step = int(match[3]) if match[3] is not None else 1
        if step < 1 or end < start or end > 2**31 - 1:
            raise ValueError("array index range is invalid")
        count = (end - start) // step + 1
        if len(indices) + count > 1000:
            raise ValueError("an array may have at most 1000 elements")
        indices.extend(range(start, end + 1, step))
    if len(indices) != len(set(indices)):
        raise ValueError("array indices must be unique")
    return tuple(ArrayElement(index=index) for index in indices)


def load_array_file(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("array manifest must be a mapping")
    source = raw.pop("source_dir", None)
    source_dir = (path.parent / str(source)).resolve() if source is not None else None
    if source_dir is not None and not source_dir.is_dir():
        raise ValueError("source_dir must be an existing directory")
    if "indices" in raw:
        if "elements" in raw:
            raise ValueError("provide indices or elements, not both")
        raw["elements"] = parse_indices(str(raw.pop("indices")))
    return JobArraySpec.model_validate(raw), source_dir


def follow_array(client, array_id, *, output, poll_seconds=2):
    previous = None
    while True:
        summary = client.array(array_id, limit=1)
        marker = (summary["state"], sorted(summary["counts"].items()))
        if marker != previous:
            counts = ", ".join(
                f"{name}={count}" for name, count in sorted(summary["counts"].items())
            )
            output.write(f"[topodrome] {array_id}: {counts}\n")
            output.flush()
            previous = marker
        if summary["terminal"] == summary["total"]:
            return summary
        time.sleep(poll_seconds)
