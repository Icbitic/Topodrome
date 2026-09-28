from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from topodrome.domain.specs import JobSpec


def load_job_file(path: str | Path) -> tuple[JobSpec, Path | None]:
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("experiment manifest must be a mapping")
    source_dir_value = raw.pop("source_dir", None)
    source_dir = None
    if source_dir_value is not None:
        source_dir = (manifest_path.parent / str(source_dir_value)).resolve()
        if not source_dir.is_dir():
            raise ValueError(f"source_dir does not exist: {source_dir}")
    return JobSpec.model_validate(raw), source_dir
