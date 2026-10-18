"""Make-before-break renewal of bounded bulk HTTP pools and owned SSH routes.

Only two configured loopback slots can exist. A generation remains owned while
any stream uses it. Upload extents choose a generation at each request boundary;
downloads retain their stream until EOF/failure and then use normal pin/resume.
No control pool, job, actor, or provider operation participates in renewal.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

from topodrome.config import BulkReconnectSettings
from topodrome.deployment.ssh_tunnel import (
    SSHLocalForwardConfig,
    SSHLocalForwardStatus,
    SSHLocalForwardSupervisor,
)
from topodrome.storage.link_probe import LinkRates, probe_link
from topodrome.storage.remote_digest import storage_api_origin

logger = logging.getLogger(__name__)


class _LinkProbe(Protocol):
    def __call__(
        self, client: httpx.Client, origin: str, site_id: str, *, timeout_seconds: float
    ) -> LinkRates | None: ...


@dataclass(eq=False)
class _Generation:
    slot: int
    number: int
    origin: str
    forward: SSHLocalForwardSupervisor
    clients: dict[str, httpx.Client]
    refs: int = 0
    data_refs: int = 0
    bytes: int = 0
    closed: bool = False


class AdaptiveBulkConnections:
    def __init__(
        self,
        config: SSHLocalForwardConfig,
        origin: str,
        site_id: str,
        token: str,
        policy: BulkReconnectSettings,
        *,
        timeout: float = 30,
        forward_factory: Callable[
            [SSHLocalForwardConfig], SSHLocalForwardSupervisor
        ] = SSHLocalForwardSupervisor,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
        probe: _LinkProbe = probe_link,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not policy.enabled or policy.candidate_port in {None, config.local_port}:
            raise ValueError(
                "adaptive bulk forwarding requires a distinct configured candidate port"
            )
        self._policy, self._site_id = policy, site_id
        self._clock, self._client_factory, self._probe = clock, client_factory, probe
        self._origin = storage_api_origin(origin)
        parsed = urlsplit(self._origin)
        if parsed.hostname != config.local_host or parsed.port != config.local_port:
            raise ValueError("bulk origin must match its owned loopback SSH route")
        candidate = replace(
            config,
            local_port=policy.candidate_port,
            state_directory=config.state_directory / "renewal",
        )
        self._forwards = (forward_factory(config), forward_factory(candidate))
        hostname = f"[{config.local_host}]" if ":" in config.local_host else config.local_host
        self._origins = (
            self._origin,
            urlunsplit(parsed._replace(netloc=f"{hostname}:{policy.candidate_port}")),
        )
        self._kwargs: dict[str, Any] = dict(
            headers={"Authorization": f"Bearer {token}", "Accept-Encoding": "identity"},
            timeout=httpx.Timeout(timeout, connect=min(5, timeout), pool=min(5, timeout)),
            trust_env=False,
            follow_redirects=False,
        )
        self._lock = threading.RLock()
        self._renew_lock, self._cleanup_lock = threading.Lock(), threading.Lock()
        self._current: _Generation | None = None
        self._retired: list[_Generation] = []
        self._number, self._closed = 0, False
        self._last_tick, self._last_bytes = clock(), 0
        self._slow_since: float | None = None
        self._last_attempt = float("-inf")
        self._last_activity = float("-inf")

    def _generation(self, slot: int) -> _Generation:
        forward = self._forwards[slot]
        forward.ensure_running()
        clients = {}
        try:
            for lane in ("download", "upload"):
                clients[lane] = self._client_factory(
                    **self._kwargs,
                    limits=httpx.Limits(
                        max_connections=2,
                        max_keepalive_connections=2,
                        keepalive_expiry=None,
                    ),
                )
        except BaseException:
            for client in clients.values():
                client.close()
            forward.stop()
            raise
        self._number += 1
        return _Generation(slot, self._number, self._origins[slot], forward, clients)

    def ensure_running(self) -> SSHLocalForwardStatus:
        with self._lock:
            if self._closed:
                raise RuntimeError("bulk connections are closed")
            if self._current is None:
                # A crashed agent's candidate is adopted/stopped only by its
                # exact durable SSH ownership record. Unknown listeners refuse.
                self._forwards[1].stop()
                self._current = self._generation(0)
            return self._current.forward.ensure_running()

    @contextmanager
    def stream(
        self, lane: str, method: str, url: str, *, transfer_size: int = 0, **kwargs: Any
    ) -> Iterator[httpx.Response]:
        with self._lock:
            if self._closed or self._current is None:
                raise RuntimeError("bulk connections are unavailable")
            generation = self._current
            generation.refs += 1
            generation.data_refs += bool(transfer_size)
        try:
            if not url.startswith(self._origin + "/"):
                raise ValueError("bulk request must reach the configured storage authority")
            rewritten = generation.origin + url[len(self._origin) :]
            with generation.clients[lane].stream(method, rewritten, **kwargs) as response:
                response.extensions["topodrome_bulk_generation"] = generation
                yield response
                if transfer_size and lane == "upload" and response.status_code == 200:
                    self.progress(response, transfer_size)
        finally:
            with self._lock:
                generation.refs -= 1
                generation.data_refs -= bool(transfer_size)
            self._cleanup()

    def progress(self, response: httpx.Response, byte_count: int) -> None:
        generation = response.extensions.get("topodrome_bulk_generation")
        if byte_count <= 0 or not isinstance(generation, _Generation):
            return
        with self._lock:
            generation.bytes += byte_count
            self._last_activity = self._clock()

    def _cleanup(self) -> None:
        if not self._cleanup_lock.acquire(blocking=False):
            return
        try:
            with self._lock:
                ready = [g for g in self._retired if not g.refs]
            for generation in ready:
                try:
                    if not generation.closed:
                        for client in generation.clients.values():
                            client.close()
                        generation.closed = True
                    generation.forward.stop()
                except Exception as exc:
                    logger.warning("bulk retirement deferred: %s", type(exc).__name__)
                    continue
                with self._lock:
                    self._retired.remove(generation)
        finally:
            self._cleanup_lock.release()

    @staticmethod
    def _better(old: LinkRates | None, fresh: LinkRates | None, speedup: float) -> bool:
        if fresh is None:
            return False
        if old is None:
            return True
        return (
            min(fresh.download, fresh.upload) >= min(old.download, old.upload) * speedup
            and fresh.download >= old.download * 0.8
            and fresh.upload >= old.upload * 0.8
        )

    def _measure(self, generation: _Generation) -> LinkRates | None:
        with self._client_factory(
            **{**self._kwargs, "timeout": self._policy.probe_timeout_seconds},
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        ) as client:
            return self._probe(
                client,
                generation.origin,
                self._site_id,
                timeout_seconds=self._policy.probe_timeout_seconds,
            )

    def maintain(self) -> bool:
        """Called off the event loop; one bounded candidate after sustained slowness."""
        self._cleanup()
        with self._lock:
            current, now = self._current, self._clock()
            if self._closed or current is None:
                return False
            elapsed = now - self._last_tick
            if elapsed < 5:
                return False
            rate = max(0, current.bytes - self._last_bytes) / elapsed
            self._last_tick, self._last_bytes = now, current.bytes
            active = current.data_refs or now - self._last_activity <= elapsed
            if not active or rate >= self._policy.slow_mib_per_second * 1024**2:
                self._slow_since = None
                return False
            if self._slow_since is None:
                self._slow_since = now
            if (
                now - self._slow_since < self._policy.sustained_seconds
                or now - self._last_attempt < self._policy.cooldown_seconds
                or self._retired
            ):
                return False
        if not self._renew_lock.acquire(blocking=False):
            return False
        candidate = None
        try:
            with self._lock:
                if self._closed or current is not self._current or self._retired:
                    return False
                self._last_attempt, self._slow_since = now, None
                current.refs += 1  # Probes also pin the old route through comparison.
            try:
                candidate = self._generation(1 - current.slot)
                old, fresh = self._measure(current), self._measure(candidate)
                if not self._better(old, fresh, self._policy.min_speedup):
                    return False
                with self._lock:
                    if self._closed or current is not self._current:
                        return False
                    self._current = candidate
                    self._retired.append(current)
                    candidate = None
                    self._last_tick, self._last_bytes = self._clock(), 0
                logger.info("bulk connection renewed after verified link improvement")
                return True
            finally:
                with self._lock:
                    current.refs -= 1
        except Exception as exc:
            # No endpoint, key path, header or raw HTTP error enters logs.
            logger.warning("bulk renewal declined: %s", type(exc).__name__)
            return False
        finally:
            if candidate is not None:
                with self._lock:
                    self._retired.append(candidate)
            self._cleanup()
            self._renew_lock.release()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._current is not None:
                    self._retired.append(self._current)
                    self._current = None
        self._cleanup()

    def stop(self) -> bool:
        self.close()
        return True
