from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime, timedelta

from pydantic import Field

from topodrome.domain.errors import StaleFenceError
from topodrome.domain.specs import FenceClaim, StrictModel, utc_now


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class FencePayload(StrictModel):
    token_id: str
    attempt_id: str
    site_id: str
    claims: tuple[FenceClaim, ...]
    issued_at: datetime
    expires_at: datetime


class FencingAuthority:
    def __init__(self, signing_key: bytes | str) -> None:
        key = signing_key.encode() if isinstance(signing_key, str) else signing_key
        if len(key) < 32:
            raise ValueError("fence signing key must be at least 32 bytes")
        self._key = key

    def issue(
        self,
        *,
        attempt_id: str,
        site_id: str,
        claims: tuple[FenceClaim, ...],
        ttl: timedelta,
        now: datetime | None = None,
        token_id: str | None = None,
    ) -> tuple[str, FencePayload]:
        issued_at = now or utc_now()
        payload = FencePayload(
            token_id=token_id or secrets.token_urlsafe(18),
            attempt_id=attempt_id,
            site_id=site_id,
            claims=claims,
            issued_at=issued_at,
            expires_at=issued_at + ttl,
        )
        raw = json.dumps(
            payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        signature = hmac.digest(self._key, raw, "sha256")
        return f"{_encode(raw)}.{_encode(signature)}", payload

    def verify(
        self,
        token: str,
        *,
        attempt_id: str | None = None,
        site_id: str | None = None,
        now: datetime | None = None,
    ) -> FencePayload:
        payload = self.authenticate(token, attempt_id=attempt_id, site_id=site_id)
        current = now or datetime.now(UTC)
        if payload.expires_at <= current:
            raise StaleFenceError("fencing token expired")
        return payload

    def authenticate(
        self,
        token: str,
        *,
        attempt_id: str | None = None,
        site_id: str | None = None,
    ) -> FencePayload:
        """Authenticate immutable token identity without applying its time window.

        Normal authorization must use :meth:`verify`. This narrower primitive is
        reserved for replay after the durable event journal proves the exact
        request committed, or for a terminal-attempt no-op that cannot revive an
        expired lease.
        """

        try:
            encoded_payload, encoded_signature = token.split(".", 1)
            raw = _decode(encoded_payload)
            signature = _decode(encoded_signature)
        except (ValueError, TypeError) as exc:
            raise StaleFenceError("malformed fencing token") from exc
        expected = hmac.digest(self._key, raw, "sha256")
        if not hmac.compare_digest(signature, expected):
            raise StaleFenceError("invalid fencing signature")
        try:
            payload = FencePayload.model_validate(json.loads(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            raise StaleFenceError("invalid fencing payload") from exc
        if attempt_id is not None and not hmac.compare_digest(payload.attempt_id, attempt_id):
            raise StaleFenceError("fencing token belongs to another attempt")
        if site_id is not None and not hmac.compare_digest(payload.site_id, site_id):
            raise StaleFenceError("fencing token belongs to another site")
        return payload

    @staticmethod
    def digest(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()


class FenceEpochs(StrictModel):
    values: dict[str, int] = Field(default_factory=dict)

    def accept(self, claims: tuple[FenceClaim, ...]) -> None:
        stale = [
            claim.gpu_uuid for claim in claims if claim.epoch <= self.values.get(claim.gpu_uuid, 0)
        ]
        if stale:
            raise StaleFenceError(f"stale fencing epoch for {', '.join(stale)}")
        for claim in claims:
            self.values[claim.gpu_uuid] = claim.epoch
