"""Shared validation and wire limits for live batch output."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac

from pydantic import Field, model_validator

from topodrome.domain.specs import StrictModel

LIVE_LOG_CHUNK_BYTES = 64 * 1024
LOG_READ_LIMIT_BYTES = 1024 * 1024
LOG_ANCHOR_BYTES = 4096
_MAX_ENCODED_CHUNK_BYTES = 4 * ((LIVE_LOG_CHUNK_BYTES + 2) // 3)


class LiveLogChunkReport(StrictModel):
    """One fenced, contiguous byte range from a site-local attempt log."""

    site_id: str = Field(min_length=1, max_length=200)
    job_id: str = Field(min_length=1, max_length=80)
    attempt_id: str = Field(min_length=1, max_length=80)
    fencing_token: str = Field(min_length=1, repr=False)
    generation: int = Field(default=0, ge=0, le=2**31 - 1)
    offset: int = Field(ge=0)
    data_base64: str = Field(min_length=1, max_length=_MAX_ENCODED_CHUNK_BYTES, repr=False)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_data(self) -> LiveLogChunkReport:
        try:
            data = base64.b64decode(self.data_base64, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("live log chunk is not valid base64") from None
        if not data or len(data) > LIVE_LOG_CHUNK_BYTES:
            raise ValueError("live log chunk has an invalid decoded size")
        if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), self.sha256):
            raise ValueError("live log chunk digest does not match its bytes")
        return self

    def decoded(self) -> bytes:
        return base64.b64decode(self.data_base64, validate=True)


__all__ = [
    "LIVE_LOG_CHUNK_BYTES",
    "LOG_ANCHOR_BYTES",
    "LOG_READ_LIMIT_BYTES",
    "LiveLogChunkReport",
]
