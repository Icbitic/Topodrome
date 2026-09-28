from __future__ import annotations

import hmac


def extract_bearer(header: str | None) -> str | None:
    if not header:
        return None
    scheme, separator, value = header.partition(" ")
    if not separator or scheme.lower() != "bearer" or not value:
        return None
    return value


def token_matches(provided: str | None, expected: str | None) -> bool:
    if expected is None:
        return True
    return provided is not None and hmac.compare_digest(provided, expected)
