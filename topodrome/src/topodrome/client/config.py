"""Private, local defaults for the Topodrome client."""

from __future__ import annotations

import os
import stat
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

DEFAULT_URL = "http://127.0.0.1:8042"


def client_connection(
    base_url: str | None,
    token: str | None,
) -> tuple[str, str | None]:
    """Resolve explicit, environment, then owner-only file configuration."""

    environment_url = os.environ.get("TOPODROME_URL")
    environment_token = os.environ.get("TOPODROME_API_TOKEN")
    values: dict[str, str] = {}
    if (base_url is None and environment_url is None) or (
        token is None and environment_token is None
    ):
        values = _read_client_environment(_client_environment_path())
    resolved_url = base_url or environment_url or values.get("TOPODROME_URL") or DEFAULT_URL
    resolved_token = token or environment_token or values.get("TOPODROME_API_TOKEN")
    return resolved_url.rstrip("/"), resolved_token


def _client_environment_path() -> Path:
    override = os.environ.get("TOPODROME_CLIENT_ENV_FILE")
    if override:
        return Path(override).expanduser()
    config_root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_root / "topodrome" / "client.env"


def _read_client_environment(path: Path) -> dict[str, str]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {}
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PermissionError(f"client environment {path} must be a regular file")
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PermissionError(f"client environment {path} must be owner-only (mode 0600)")
    parent = path.parent
    parent_metadata = parent.stat()
    if parent_metadata.st_uid != os.getuid() or stat.S_IMODE(parent_metadata.st_mode) & 0o077:
        raise PermissionError(f"client environment directory {parent} must be owner-only")

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"TOPODROME_URL", "TOPODROME_API_TOKEN"}:
            values[key] = value.strip().strip('"').strip("'")
    return values


def save_client_connection(base_url: str, token: str) -> Path:
    """Atomically install one user's private scheduler credential."""

    normalized_url = base_url.strip().rstrip("/")
    parsed = urlsplit(normalized_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(character in normalized_url for character in ("\0", "\r", "\n"))
    ):
        raise ValueError("Topodrome URL must be a credential-free HTTP(S) origin")
    if not token or len(token) > 4096 or any(character in token for character in "\r\n\0"):
        raise ValueError("Topodrome token is invalid")
    path = _client_environment_path()
    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent.chmod(0o700)
    metadata = parent.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PermissionError("client environment directory must be owner-only")
    encoded = (
        f"TOPODROME_URL={normalized_url}\n"
        f"TOPODROME_API_TOKEN={token}\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    replaced = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        replaced = True
        directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not replaced:
            with suppress(FileNotFoundError):
                temporary.unlink()
    return path


__all__ = ["DEFAULT_URL", "client_connection", "save_client_connection"]
