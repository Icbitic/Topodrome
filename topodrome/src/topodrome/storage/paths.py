"""Validation for machine-independent paths in the global namespace."""

from __future__ import annotations

import stat
from pathlib import Path, PurePosixPath


class LogicalPathError(ValueError):
    """A logical path is empty, ambiguous, or could escape its configured root."""


def normalize_logical_path(value: str | PurePosixPath) -> str:
    """Return a strict relative POSIX path without silently normalizing input."""

    raw = str(value)
    if not raw or raw in {".", ".."}:
        raise LogicalPathError("logical path must not be empty")
    if "\\" in raw:
        raise LogicalPathError("logical paths must use forward slashes")
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise LogicalPathError("logical paths must not contain control characters")
    if raw.startswith("/"):
        raise LogicalPathError("logical path must be relative")

    components = raw.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise LogicalPathError("logical path contains an unsafe component")

    path = PurePosixPath(*components)
    if path.is_absolute() or ".." in path.parts:
        raise LogicalPathError("logical path escapes its configured root")
    return path.as_posix()


def resolve_beneath(root: Path | str, logical_path: str | PurePosixPath) -> Path:
    """Resolve a logical path and reject existing symlinks that leave ``root``."""

    normalized = normalize_logical_path(logical_path)
    resolved_root = Path(root).expanduser().resolve()
    candidate = (resolved_root / normalized).resolve(strict=False)
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise LogicalPathError("logical path resolves outside its configured root")
    return candidate


def validate_component(value: str, *, label: str = "identifier") -> str:
    """Validate an identifier used as exactly one logical path component."""

    normalized = normalize_logical_path(value)
    if "/" in normalized:
        raise LogicalPathError(f"{label} must be one path component")
    return normalized


def resolve_writable_beneath(root: Path | str, logical_path: str | PurePosixPath) -> Path:
    """Resolve a canonical write location, refusing aliases and special files.

    This does not create, download or materialize anything. Read-only directory
    caches must never determine where an attempt writes its local state.
    """
    normalized = normalize_logical_path(logical_path)
    candidate = Path(root).expanduser().resolve()
    parts = normalized.split("/")
    for index, part in enumerate(parts):
        candidate = candidate / part
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise LogicalPathError("writable path must not contain symbolic links")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise LogicalPathError("writable path parent must be a directory")
        elif not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise LogicalPathError("writable path must be a regular file or directory")
    return candidate


__all__ = [
    "LogicalPathError",
    "normalize_logical_path",
    "resolve_beneath",
    "resolve_writable_beneath",
    "validate_component",
]
