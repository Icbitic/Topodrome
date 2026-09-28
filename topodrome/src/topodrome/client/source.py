from __future__ import annotations

import base64
from pathlib import Path

from topodrome.source_limits import LEGACY_SOURCE_LIMITS, SourceLimits


def read_source_files(
    root: str | Path,
    *,
    limits: SourceLimits = LEGACY_SOURCE_LIMITS,
) -> dict[str, str]:
    """Read one source tree into the scheduler's content-addressed wire format."""

    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError(f"source_dir is not a directory: {source_root}")

    files: dict[str, str] = {}
    total_bytes = 0
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"source_dir cannot contain symbolic links: {relative.as_posix()}")
        if not path.is_file():
            continue
        logical_path = relative.as_posix()
        if len(logical_path) > limits.max_path_chars:
            raise ValueError("source_dir contains a path longer than the source limit")
        if len(files) >= limits.max_files:
            raise ValueError(f"source_dir exceeds the {limits.max_files}-file limit")
        stat_size = path.stat().st_size
        if stat_size > limits.max_file_bytes:
            raise ValueError("source_dir contains a file larger than the per-file limit")
        if total_bytes + stat_size > limits.max_total_bytes:
            raise ValueError("source_dir exceeds the total decoded-size limit")
        body = path.read_bytes()
        # Recheck after reading so a concurrently growing file cannot bypass
        # the pre-read bound.
        if len(body) > limits.max_file_bytes:
            raise ValueError("source_dir contains a file larger than the per-file limit")
        if total_bytes + len(body) > limits.max_total_bytes:
            raise ValueError("source_dir exceeds the total decoded-size limit")
        files[logical_path] = base64.b64encode(body).decode()
        total_bytes += len(body)

    if not files:
        raise ValueError(f"source_dir contains no publishable files: {source_root}")
    return files


__all__ = ["read_source_files"]
