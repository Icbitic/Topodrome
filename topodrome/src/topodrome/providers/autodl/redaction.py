"""Credential-safe rendering for AutoDL responses and failures."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "<redacted>"
_SENSITIVE_KEY = re.compile(
    r"(?:authorization|token|password|passwd|secret|credential|api[_-]?key)",
    re.IGNORECASE,
)
_LABELED_SECRET = re.compile(
    r"(?i)\b(authorization|token|password|passwd|secret|credential|api[_-]?key)"
    r"\s*([:=])\s*([^\s,;&]+)"
)
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:token|access_token|auth|authorization|api[_-]?key|key|"
    r"password|passwd|secret|credential)=)([^&#\s]+)"
)
_FLAG_SECRET = re.compile(
    r"(?i)(?:^|[\s;&|])--(?:[a-z0-9]+[-_])*(?:auth(?:orization)?|access[-_]?token|"
    r"token|secret|passw(?:or)?d|credential|api[-_]?key)(?:=|\s+)[^\s;&|]+"
)
_ENV_SECRET = re.compile(
    r"(?i)(?:^|[\s;&|])(?:export\s+)?(?:[A-Za-z0-9]+_)*"
    r"(?:AUTH|AUTHORIZATION|ACCESS_TOKEN|TOKEN|SECRET|PASSWORD|PASSWD|"
    r"CREDENTIAL|API_KEY|APIKEY)\s*="
)


def redact_text(value: str, *, known_secrets: Sequence[str] = ()) -> str:
    """Remove known credentials and common labeled/query secret forms."""

    redacted = value
    for secret in known_secrets:
        if secret:
            redacted = redacted.replace(secret, REDACTED)
    redacted = _LABELED_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", redacted
    )
    return _QUERY_SECRET.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)


def redact_value(value: Any, *, known_secrets: Sequence[str] = ()) -> Any:
    """Recursively produce a log-safe copy of a JSON-like value."""

    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED
                if _SENSITIVE_KEY.search(str(key))
                else redact_value(item, known_secrets=known_secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, str):
        return redact_text(value, known_secrets=known_secrets)
    if isinstance(value, tuple):
        return tuple(redact_value(item, known_secrets=known_secrets) for item in value)
    if isinstance(value, list):
        return [redact_value(item, known_secrets=known_secrets) for item in value]
    return value


def contains_sensitive_key(value: Any) -> bool:
    """Return whether a JSON-like value contains credential-shaped material."""

    if isinstance(value, Mapping):
        return any(
            _SENSITIVE_KEY.search(str(key)) is not None or contains_sensitive_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(contains_sensitive_key(item) for item in value)
    if isinstance(value, str):
        return any(
            pattern.search(value) is not None
            for pattern in (_LABELED_SECRET, _QUERY_SECRET, _FLAG_SECRET, _ENV_SECRET)
        )
    return False


__all__ = ["REDACTED", "contains_sensitive_key", "redact_text", "redact_value"]
