"""Short-lived authentication for scheduler-hosted kernel relay URLs."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import quote, urlencode


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


@dataclass(frozen=True, slots=True)
class KernelConnectClaims:
    session_id: str
    relay_id: str
    generation: int
    expires_at: datetime


class KernelConnectSigner:
    """Mint opaque, expiring URLs without storing their bearer credential."""

    def __init__(self, key: str | bytes, *, public_url: str, ttl_seconds: int) -> None:
        material = key.encode() if isinstance(key, str) else key
        if len(material) < 32:
            raise ValueError("kernel URL signing key must contain at least 32 bytes")
        if ttl_seconds <= 0:
            raise ValueError("kernel URL lifetime must be positive")
        self._key = material
        self.public_url = public_url.rstrip("/")
        self.ttl_seconds = ttl_seconds

    def issue(
        self,
        *,
        session_id: str,
        relay_id: str,
        generation: int,
        session_expires_at: datetime,
        now: datetime,
    ) -> tuple[str, datetime]:
        current = _aware(now)
        session_expiry = _aware(session_expires_at)
        expires_at = min(
            session_expiry,
            current + timedelta(seconds=self.ttl_seconds),
        )
        if expires_at <= current:
            raise ValueError("kernel session has expired")
        payload = {
            "e": int(expires_at.timestamp()),
            "g": generation,
            "r": relay_id,
            "s": session_id,
        }
        encoded = _b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        signature = _b64encode(hmac.digest(self._key, encoded.encode(), hashlib.sha256))
        access = f"{encoded}.{signature}"
        path = f"/api/v1/kernel-relay/{quote(session_id, safe='')}"
        return f"{self.public_url}{path}?{urlencode({'token': access})}", expires_at

    def verify(
        self,
        access: str,
        *,
        session_id: str,
        relay_id: str,
        generation: int,
        now: datetime,
    ) -> KernelConnectClaims:
        try:
            encoded, signature = access.split(".", 1)
            expected = _b64encode(hmac.digest(self._key, encoded.encode(), hashlib.sha256))
            if not hmac.compare_digest(signature, expected):
                raise ValueError("kernel access signature is invalid")
            raw = json.loads(_b64decode(encoded))
            claimed_session = str(raw["s"])
            claimed_relay = str(raw["r"])
            claimed_generation = int(raw["g"])
            expires_at = datetime.fromtimestamp(int(raw["e"]), UTC)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("kernel access credential is invalid") from exc
        if (
            claimed_session != session_id
            or claimed_relay != relay_id
            or claimed_generation != generation
        ):
            raise ValueError("kernel access credential has the wrong scope")
        if expires_at <= _aware(now):
            raise ValueError("kernel access credential has expired")
        return KernelConnectClaims(
            claimed_session,
            claimed_relay,
            claimed_generation,
            expires_at,
        )


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["KernelConnectClaims", "KernelConnectSigner"]
