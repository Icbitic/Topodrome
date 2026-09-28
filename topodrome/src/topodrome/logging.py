from __future__ import annotations

import logging
import re
from collections.abc import Mapping, MutableMapping
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import structlog

_KEY = re.compile(r"(?:authorization|token|secret|password|credential|api[_-]?key)", re.I)
_BEARER = re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:access|access_token|auth|authorization|token|api[_-]?key)=)[^&#\s]+"
)


def redact_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return "<redacted-url>"
    if not parts.scheme or not parts.netloc:
        return value
    query = [
        (key, "<redacted>" if _KEY.search(key) else item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
    ]
    # URL userinfo is an authentication field. Drop it completely rather than
    # retaining even a username, and defensively redact credential fragments.
    netloc = parts.netloc.rsplit("@", 1)[-1]
    fragment = "<redacted>" if _KEY.search(parts.fragment) else parts.fragment
    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(query), fragment))


def redact(value: Any, known_secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if _KEY.search(str(key)) else redact(item, known_secrets)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact(item, known_secrets) for item in value)
    if isinstance(value, list):
        return [redact(item, known_secrets) for item in value]
    if isinstance(value, str):
        result = redact_url(value) if "://" in value else value
        # Access logs generally contain a relative request target rather than
        # an absolute URL, so redact credential-shaped query values there too.
        result = _QUERY_SECRET.sub(r"\1<redacted>", result)
        result = _BEARER.sub(r"\1<redacted>", result)
        result = _JWT.sub("<redacted-jwt>", result)
        for secret in known_secrets:
            if secret:
                result = result.replace(secret, "<redacted>")
        return result
    return value


def structlog_redactor(
    _: Any, __: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    return cast(MutableMapping[str, Any], redact(event_dict))


class RedactingFilter(logging.Filter):
    def __init__(self, known_secrets: tuple[str, ...] = ()) -> None:
        super().__init__()
        self._known_secrets = known_secrets

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.msg), self._known_secrets)
        if record.args:
            record.args = tuple(redact(item, self._known_secrets) for item in record.args)
        return True


def configure_logging(level: int = logging.INFO, known_secrets: tuple[str, ...] = ()) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter(known_secrets))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # ``basicConfig(force=True)`` replaces handlers but deliberately preserves
    # the ``disabled`` bit on named loggers.  A library/test that previously
    # applied a ``dictConfig`` can therefore leave Uvicorn access logging
    # disabled, bypassing both observability and the redacting root handler.
    # This is the one named logger whose records Topodrome explicitly owns.
    logging.getLogger("uvicorn.access").disabled = False
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog_redactor,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )
