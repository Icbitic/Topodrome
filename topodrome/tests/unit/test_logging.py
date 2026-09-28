from __future__ import annotations

import io
import logging

import pytest

from topodrome.logging import configure_logging, redact

pytestmark = pytest.mark.unit


def test_redact_hides_kernel_tokens_in_absolute_and_access_log_urls() -> None:
    capability = "signed-kernel-capability"

    absolute = redact(f"https://scheduler.invalid/kernel?token={capability}&safe=one")
    request_line = redact(f'127.0.0.1 - "GET /kernel?token={capability}&safe=one HTTP/1.1"')

    assert capability not in absolute
    assert capability not in request_line
    assert "safe=one" in absolute
    assert "safe=one" in request_line


def test_redact_url_removes_userinfo_and_credential_fragments() -> None:
    rendered = redact(
        "https://operator:password@example.invalid/path?safe=one#token=fragment-secret"
    )

    assert "operator" not in rendered
    assert "password" not in rendered
    assert "fragment-secret" not in rendered
    assert rendered.startswith("https://example.invalid/path?safe=one")


def test_uvicorn_style_access_record_is_redacted_by_the_root_handler() -> None:
    capability = "signed-access-log-capability"
    stream = io.StringIO()
    access = logging.getLogger("uvicorn.access")
    access.disabled = True
    configure_logging()
    root_handler = logging.getLogger().handlers[0]
    assert isinstance(root_handler, logging.StreamHandler)
    root_handler.setStream(stream)
    assert not access.disabled
    access.handlers.clear()
    access.propagate = True
    access.setLevel(logging.INFO)

    access.info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:1234",
        "GET",
        f"/api/v1/kernel-relay/session?token={capability}",
        "1.1",
        200,
    )

    rendered = stream.getvalue()
    assert capability not in rendered
    assert "token=<redacted>" in rendered
