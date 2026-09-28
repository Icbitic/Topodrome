"""Reverse-relay contract for browser-facing Jupyter connectivity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from topodrome.logging import redact_url


class BrowserRelayUnavailable(RuntimeError):
    """No authenticated HTTP/WebSocket relay implementation is configured."""


@dataclass(frozen=True, slots=True)
class RelayHandle:
    relay_id: str
    public_url: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.relay_id or not self.public_url:
            raise ValueError("relay identity and public URL must not be empty")

    def __repr__(self) -> str:
        public_url = redact_url(self.public_url)
        return f"RelayHandle(relay_id={self.relay_id!r}, public_url={public_url!r})"


@runtime_checkable
class ReverseRelay(Protocol):
    """Publishes an authenticated HTTP and WebSocket route to one upstream."""

    async def start(self, *, session_id: str, upstream_url: str) -> RelayHandle: ...

    async def stop(self, handle: RelayHandle) -> None: ...


class UnavailableReverseRelay:
    """Safe default that makes the unimplemented production proxy explicit."""

    async def start(self, *, session_id: str, upstream_url: str) -> RelayHandle:
        del session_id, upstream_url
        raise BrowserRelayUnavailable(
            "configure a production ReverseRelay that proxies both HTTP and WebSocket traffic"
        )

    async def stop(self, handle: RelayHandle) -> None:
        del handle


__all__ = [
    "BrowserRelayUnavailable",
    "RelayHandle",
    "ReverseRelay",
    "UnavailableReverseRelay",
]
