"""Small enrolled-agent link probes; no artifact writes or public endpoints."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from typing import Any

import httpx

from topodrome.storage.base import StorageVerificationError

LINK_PROBE_BYTES = 128 * 1024


class LinkProbeDeferred(RuntimeError):
    """Unsupported/busy authorities are not measurements of a faulty route."""


@dataclass(frozen=True)
class LinkRates:
    download: float
    upload: float


def probe_link(
    client: httpx.Client,
    origin: str,
    site_id: str,
    *,
    timeout_seconds: float = 5,
    clock: Callable[[], float] = time.monotonic,
) -> LinkRates | None:
    """Two uncached samples per direction, excluding TCP setup via warmup.

    Unsupported authorities/backpressure defer the entire comparison. A
    network timeout supplies no rate; a verified candidate may then recover.
    Authentication and integrity errors are never treated as a slow route.
    """
    endpoint = origin + "/link-probe"
    deadline = clock() + timeout_seconds
    payload = os.urandom(LINK_PROBE_BYTES)
    digest = hashlib.sha256(payload).hexdigest()
    samples: dict[str, list[float]] = {"download": [], "upload": []}

    def request(method: str, *, limit: int, **kwargs: Any) -> tuple[bytes, httpx.Headers]:
        remaining = deadline - clock()
        if remaining <= 0:
            raise httpx.ReadTimeout("link probe observation budget exhausted")
        # Stream and bound responses rather than buffering an untrusted body.
        # The budget spans all samples; individual socket operations also have
        # timeouts, including a stalled peer before it produces a body chunk.
        with client.stream(method, endpoint, timeout=remaining, **kwargs) as response:
            if response.status_code in {404, 408, 429, 500, 502, 503, 504}:
                raise LinkProbeDeferred("storage authority unavailable for link comparison")
            if response.status_code != 200:
                raise StorageVerificationError("link probe authorization/protocol rejected")
            if response.headers.get("content-encoding", "identity") != "identity":
                raise StorageVerificationError("link probe requires identity encoding")
            declared = response.headers.get("content-length")
            if declared is not None and (not declared.isdecimal() or int(declared) > limit):
                raise StorageVerificationError("link probe response is oversized")
            body = bytearray()
            for chunk in response.iter_bytes():
                if clock() >= deadline:
                    raise httpx.ReadTimeout("link probe observation budget exhausted")
                if len(body) + len(chunk) > limit:
                    raise StorageVerificationError("link probe response is oversized")
                body.extend(chunk)
            if clock() >= deadline:
                raise httpx.ReadTimeout("link probe observation budget exhausted")
            return bytes(body), response.headers

    try:
        request("GET", limit=0, params={"site_id": site_id, "size": 0})
        for _ in range(2):
            for direction in samples:
                started = clock()
                if direction == "download":
                    response = request("GET", limit=LINK_PROBE_BYTES, params={"site_id": site_id})
                else:
                    response = request(
                        "POST",
                        limit=4096,
                        params={"site_id": site_id},
                        content=payload,
                        headers={"Content-Type": "application/octet-stream"},
                    )
                elapsed = max(0.000001, clock() - started)
                body, headers = response
                if direction == "download":
                    if len(body) != LINK_PROBE_BYTES or hashlib.sha256(
                        body
                    ).hexdigest() != headers.get("x-topodrome-sha256"):
                        raise StorageVerificationError("link probe bytes failed verification")
                elif json.loads(body) != {"bytes": LINK_PROBE_BYTES, "sha256": digest}:
                    raise StorageVerificationError("link probe acknowledgement mismatch")
                samples[direction].append(LINK_PROBE_BYTES / elapsed)
    except httpx.TransportError:
        return None
    except (ValueError, TypeError):
        raise StorageVerificationError("malformed link probe acknowledgement") from None
    return LinkRates(median(samples["download"]), median(samples["upload"]))
