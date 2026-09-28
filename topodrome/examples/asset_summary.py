"""Small consumer used to verify an immutable Topodrome asset input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run(context: Any, *, asset_alias: str = "input") -> dict[str, object]:
    root = Path(context.committed_inputs[asset_alias])
    files = (root,) if root.is_file() else tuple(path for path in root.rglob("*") if path.is_file())
    summary: dict[str, object] = {
        "alias": asset_alias,
        "kind": "file" if root.is_file() else "directory",
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
    }
    output = context.workspace / "asset-summary.json"
    output.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    context.register_artifact(output, logical_path="asset-summary.json")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary
