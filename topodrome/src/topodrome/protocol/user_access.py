"""Owner-only credentials for whitelisted Unix users."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pwd
import re
import secrets
import stat
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

_USERNAME = re.compile(r"^[a-z_][a-z0-9_.-]{0,63}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class UserPrincipal:
    username: str
    is_admin: bool = False


@dataclass(frozen=True, slots=True)
class GrantedCredential:
    username: str
    token: str


class UserCredentialStore:
    """Atomic digest-only whitelist with immediate grant/revoke visibility."""

    _SCHEMA_VERSION = 1
    _MAXIMUM_BYTES = 1_048_576

    def __init__(self, path: Path, *, require_local_account: bool = True) -> None:
        expanded = path.expanduser()
        if not expanded.is_absolute() or expanded == Path("/"):
            raise ValueError("user credential store must be an absolute scoped path")
        self.path = expanded
        self.require_local_account = require_local_account
        self._lock = threading.RLock()

    def authenticate(self, token: str) -> UserPrincipal | None:
        if not isinstance(token, str) or not token or len(token) > 4096:
            return None
        supplied = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            users = self._read_users()
        matched: str | None = None
        # Scan every digest so a match position does not become a useful timing
        # signal. The store is intentionally a small human-managed whitelist.
        for username, record in sorted(users.items()):
            digest = str(record["token_sha256"])
            if hmac.compare_digest(supplied, digest):
                matched = username
        return None if matched is None else UserPrincipal(matched)

    def list_users(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._read_users()))

    def is_whitelisted(self, username: str) -> bool:
        """Return current whitelist membership without trusting cached state."""

        return self.authorization_revision(username) is not None

    def authorization_revision(self, username: str) -> str | None:
        """Return the non-secret revision that changes on every grant/rotation."""

        try:
            normalized = self._validate_username(username, require_account=False)
        except ValueError:
            return None
        with self._lock:
            record = self._read_users().get(normalized)
        return None if record is None else record["revision"]

    def grant(self, username: str) -> GrantedCredential:
        normalized = self._validate_username(username)
        token = secrets.token_urlsafe(32)
        record = {
            "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
            "created_at": datetime.now(UTC).isoformat(),
            "revision": secrets.token_hex(16),
        }
        with self._lock:
            users = self._read_users()
            users[normalized] = record
            self._write_users(users)
        return GrantedCredential(username=normalized, token=token)

    def revoke(self, username: str) -> bool:
        normalized = self._validate_username(username, require_account=False)
        with self._lock:
            users = self._read_users()
            existed = users.pop(normalized, None) is not None
            if existed:
                self._write_users(users)
        return existed

    def _validate_username(self, username: str, *, require_account: bool = True) -> str:
        normalized = username.strip() if isinstance(username, str) else ""
        if not _USERNAME.fullmatch(normalized) or normalized.startswith("."):
            raise ValueError("username is not a valid local account name")
        if require_account and self.require_local_account:
            try:
                account = pwd.getpwnam(normalized)
            except KeyError:
                raise ValueError("username is not an existing local account") from None
            if account.pw_name != normalized:
                raise ValueError("username does not resolve to an exact local account")
        return normalized

    def _read_users(self) -> dict[str, dict[str, str]]:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except FileNotFoundError:
            return {}
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) & 0o077
                or metadata.st_size > self._MAXIMUM_BYTES
            ):
                raise PermissionError(
                    "user credential store must be an owner-only regular file"
                )
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = -1
                payload = json.load(handle)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != self._SCHEMA_VERSION
            or not isinstance(payload.get("users"), dict)
        ):
            raise ValueError("user credential store schema is invalid")
        users: dict[str, dict[str, str]] = {}
        for username, raw_record in payload["users"].items():
            normalized = self._validate_username(str(username), require_account=False)
            if normalized != username or not isinstance(raw_record, dict):
                raise ValueError("user credential store entry is invalid")
            digest = raw_record.get("token_sha256")
            created_at = raw_record.get("created_at")
            revision = raw_record.get("revision")
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not isinstance(created_at, str)
                or len(created_at) > 80
            ):
                raise ValueError("user credential store entry is invalid")
            if revision is None:
                # Stores written before credential revisions remain readable.
                # The digest already changes on rotation and has full token
                # entropy, so it is a safe deterministic compatibility epoch.
                revision = f"legacy-{digest}"
            if not isinstance(revision, str) or not re.fullmatch(
                r"(?:[0-9a-f]{32}|legacy-[0-9a-f]{64})",
                revision,
            ):
                raise ValueError("user credential store entry is invalid")
            users[normalized] = {
                "token_sha256": digest,
                "created_at": created_at,
                "revision": revision,
            }
        return users

    def _write_users(self, users: dict[str, dict[str, str]]) -> None:
        parent = self.path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent_metadata = parent.stat()
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(parent_metadata.st_mode) & 0o077
        ):
            raise PermissionError(
                "user credential store directory must be owner-only"
            )
        payload: dict[str, Any] = {
            "schema_version": self._SCHEMA_VERSION,
            "users": dict(sorted(users.items())),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(encoded) > self._MAXIMUM_BYTES:
            raise ValueError("user credential store exceeds its size limit")
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        replaced = False
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(encoded)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
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


__all__ = ["GrantedCredential", "UserCredentialStore", "UserPrincipal"]
