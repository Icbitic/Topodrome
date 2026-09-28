"""Small authenticated digest requests; payload bytes stay at the authority."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from topodrome.storage.base import AuthoritativeDigestReader, ObjectDigest, StorageUnavailableError
from topodrome.storage.paths import normalize_logical_path


class DigestResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SchedulerDigestReader:
    """Use the enrolled site's control origin, never a user-supplied URL."""

    def __init__(
        self,
        scheduler_url: str,
        site_id: str,
        token: str,
        *,
        health_reader: AuthoritativeDigestReader,
        timeout: float = 120,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(scheduler_url)
        path = parsed.path.rstrip("/").removesuffix("/api/v1/agents/control")
        self._url = urlunsplit(
            (
                {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme),
                parsed.netloc,
                path + "/api/v1/agents/storage/digest",
                "",
                "",
            )
        )
        self._site_id = site_id
        self._token = token
        self._timeout = timeout
        self._transport = transport
        self._health_reader = health_reader

    def healthcheck(self) -> bool:
        # Transport health is probed separately by the rclone adapter.
        return self._health_reader.healthcheck()

    def claim_blob(self, job_id: str, attempt_id: str, digest: ObjectDigest) -> None:
        try:
            with httpx.Client(timeout=self._timeout, trust_env=False,
                              transport=self._transport, follow_redirects=False) as client:
                response = client.post(
                    self._url.removesuffix("digest") + "blobs/claim",
                    headers={"Authorization": f"Bearer {self._token}"},
                    json={"site_id": self._site_id, "job_id": job_id,
                          "attempt_id": attempt_id, "sha256": digest.sha256,
                          "size_bytes": digest.size_bytes},
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise StorageUnavailableError("blob upload claim rejected or unavailable") from exc

    def digest(self, logical_path: str) -> ObjectDigest | None:
        path = normalize_logical_path(logical_path)
        try:
            with httpx.Client(
                timeout=self._timeout,
                trust_env=False,
                transport=self._transport,
                follow_redirects=False,
            ) as client:
                response = client.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._token}"},
                    json={"site_id": self._site_id, "logical_path": path},
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                if len(response.content) > 4096:
                    raise ValueError("oversized digest response")
                result = DigestResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            raise StorageUnavailableError("authoritative digest request failed") from exc
        return ObjectDigest(size_bytes=result.size_bytes, sha256=result.sha256)
