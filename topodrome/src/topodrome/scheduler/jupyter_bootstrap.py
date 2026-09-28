"""Recover terminal-only Jupyter site agents without exposing their credentials."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import stat
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import structlog

from topodrome.config import (
    JupyterBootstrapSettings,
    SchedulerSettings,
    load_secret_bundle,
)
from topodrome.db.models import SiteRow
from topodrome.db.session import Database
from topodrome.domain.specs import utc_now
from topodrome.jupyter.terminal import (
    BootstrapResult,
    JupyterTerminalClient,
    RemoteTerminal,
)
from topodrome.scheduler.control import AgentConnectionRegistry

log = structlog.get_logger(__name__)


class TerminalBootstrapClient(Protocol):
    async def bootstrap(
        self,
        command: str,
        *,
        ready_marker: str,
        timeout_seconds: float | None = None,
        cleanup_terminal: bool = True,
        keep_terminal_on_success: bool = False,
        terminal_name: str | None = None,
    ) -> BootstrapResult: ...

    async def delete_terminal(self, terminal: RemoteTerminal) -> None: ...

    async def aclose(self) -> None: ...


class JupyterBootstrapController:
    """Single-flight, retrying recovery for one configured physical site."""

    def __init__(
        self,
        *,
        database: Database,
        connections: AgentConnectionRegistry,
        settings: JupyterBootstrapSettings,
        heartbeat_ttl_seconds: int,
        client: TerminalBootstrapClient,
        ownership_path: Path,
        terminal_origin: str,
        now: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not settings.enabled or settings.command is None:
            raise ValueError("Jupyter bootstrap controller requires enabled settings")
        self.database = database
        self.connections = connections
        self.settings = settings
        self._heartbeat_ttl = timedelta(seconds=heartbeat_ttl_seconds)
        self._client = client
        self._now = now
        self._monotonic = monotonic
        self._ownership_path = ownership_path.expanduser().absolute()
        site_key = hashlib.sha256(settings.site_id.encode("utf-8")).hexdigest()[:16]
        self._endpoint_fingerprint = hashlib.sha256(
            terminal_origin.encode("utf-8")
        ).hexdigest()
        # Jupyter Server 2.x terminal WebSocket routes accept ``\w`` names.
        # Hyphens can be created through REST but cannot subsequently be opened
        # as a terminal WebSocket on those releases.
        self._terminal_prefix = f"topodrome_bootstrap_{site_key}_"
        self._prepare_ownership_root()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._retained_terminal = self._load_owned_terminal()
        self._consecutive_attempts = 0
        self._next_attempt_at = 0.0
        self._closed = False

    @property
    def attempt_in_flight(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def consecutive_attempts(self) -> int:
        return self._consecutive_attempts

    @property
    def next_attempt_at(self) -> float:
        return self._next_attempt_at

    async def reconcile_once(self) -> None:
        """Start at most one due terminal bootstrap attempt."""

        healthy = await self._site_is_healthy()
        cancel_task: asyncio.Task[None] | None = None
        async with self._lock:
            if self._closed:
                return
            if healthy:
                if self._task is not None:
                    if self._task.done():
                        with suppress(asyncio.CancelledError):
                            self._task.result()
                    else:
                        self._task.cancel()
                        cancel_task = self._task
                    self._task = None
                self._consecutive_attempts = 0
                self._next_attempt_at = 0.0
            else:
                if self._task is not None:
                    if not self._task.done():
                        return
                    # Attempts handle their own errors, so retrieving the result is
                    # safe and prevents task exceptions from being left unobserved.
                    self._task.result()
                    self._task = None
                if self._monotonic() < self._next_attempt_at:
                    return
                self._task = asyncio.create_task(
                    self._run_attempt(),
                    name=f"jupyter-bootstrap-{self.settings.site_id}",
                )
        if cancel_task is not None:
            with suppress(asyncio.CancelledError):
                await cancel_task

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            task = self._task
            self._task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        elif task is not None:
            task.result()
        # A successfully retained terminal owns the now-independent remote
        # agent and intentionally survives a scheduler process restart.
        await self._client.aclose()

    async def _site_is_healthy(self) -> bool:
        identity_before = await self.connections.identity(self.settings.site_id)
        if identity_before is None:
            return False
        async with self.database.session() as session:
            row = await session.get(SiteRow, self.settings.site_id)
        identity_after = await self.connections.identity(self.settings.site_id)
        if identity_before != identity_after or identity_after is None or row is None:
            return False
        if row.heartbeat_at is None:
            return False
        heartbeat_at = _as_utc(row.heartbeat_at)
        if heartbeat_at < _as_utc(self._now()) - self._heartbeat_ttl:
            return False
        return (
            row.agent_session_id,
            row.agent_connection_generation,
        ) == identity_after

    async def _run_attempt(self) -> None:
        schedule_retry = True
        try:
            previous = self._retained_terminal
            if previous is not None:
                await self._client.delete_terminal(previous)
                self._retained_terminal = None
                self._clear_owned_terminal()
            terminal = RemoteTerminal(
                f"{self._terminal_prefix}{secrets.token_hex(12)}"
            )
            # Journal the intended exact name before issuing POST. This closes
            # the ambiguity window where Jupyter creates the terminal but the
            # scheduler exits before receiving its response.
            self._save_owned_terminal(terminal)
            self._retained_terminal = terminal
            result = await self._client.bootstrap(
                self.settings.command or "",
                ready_marker=self.settings.ready_marker,
                timeout_seconds=self.settings.attempt_timeout_seconds,
                cleanup_terminal=True,
                keep_terminal_on_success=True,
                terminal_name=terminal.name,
            )
            if result.terminal != terminal:
                raise RuntimeError("Jupyter returned an unexpected terminal identity")
            self._retained_terminal = result.terminal
            log.info(
                "jupyter_bootstrap_submitted",
                site_id=self.settings.site_id,
                attempt=self._consecutive_attempts + 1,
            )
        except asyncio.CancelledError:
            schedule_retry = False
            raise
        except Exception as exc:
            # Never include exception text: HTTP/client exceptions can contain
            # URLs, response bodies, or credentials from a remote server.
            log.warning(
                "jupyter_bootstrap_failed",
                site_id=self.settings.site_id,
                attempt=self._consecutive_attempts + 1,
                error_type=type(exc).__name__[:80],
            )
        finally:
            if schedule_retry and not self._closed:
                self._consecutive_attempts += 1
                delay = min(
                    self.settings.initial_retry_seconds
                    * (2 ** min(self._consecutive_attempts - 1, 30)),
                    self.settings.max_retry_seconds,
                )
                self._next_attempt_at = self._monotonic() + delay

    def _prepare_ownership_root(self) -> None:
        parent = self._ownership_path.parent
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if parent.is_symlink():
            raise ValueError("Jupyter bootstrap ownership directory cannot be a symlink")
        parent.chmod(0o700)
        if self._ownership_path.is_symlink():
            raise ValueError("Jupyter bootstrap ownership state cannot be a symlink")

    def _load_owned_terminal(self) -> RemoteTerminal | None:
        try:
            metadata = self._ownership_path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Jupyter bootstrap ownership state must be a regular file")
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise PermissionError(
                "Jupyter bootstrap ownership state must be owner-only"
            )
        try:
            payload = json.loads(self._ownership_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError
            if (
                payload.get("version") != 1
                or payload.get("site_id") != self.settings.site_id
                or payload.get("endpoint_fingerprint") != self._endpoint_fingerprint
            ):
                raise ValueError
            terminal = RemoteTerminal(str(payload["terminal_name"]))
            if not self._is_owned_terminal(terminal):
                raise ValueError
            return terminal
        except (KeyError, TypeError, json.JSONDecodeError, UnicodeError, ValueError):
            # An ambiguous/corrupt record must stop recovery rather than widen
            # deletion to a guessed or remotely enumerated terminal.
            raise ValueError("invalid Jupyter bootstrap ownership state") from None

    def _save_owned_terminal(self, terminal: RemoteTerminal) -> None:
        if not self._is_owned_terminal(terminal):
            raise ValueError("refusing to persist an unowned Jupyter terminal")
        payload = json.dumps(
            {
                "version": 1,
                "site_id": self.settings.site_id,
                "endpoint_fingerprint": self._endpoint_fingerprint,
                "terminal_name": terminal.name,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        temporary = self._ownership_path.with_name(
            f".{self._ownership_path.name}.{secrets.token_hex(8)}.tmp"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._ownership_path)
            self._ownership_path.chmod(0o600)
            _fsync_directory(self._ownership_path.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def _clear_owned_terminal(self) -> None:
        self._ownership_path.unlink(missing_ok=True)
        _fsync_directory(self._ownership_path.parent)

    def _is_owned_terminal(self, terminal: RemoteTerminal) -> bool:
        if not terminal.name.startswith(self._terminal_prefix):
            return False
        suffix = terminal.name.removeprefix(self._terminal_prefix)
        return _OWNED_TERMINAL_SUFFIX.fullmatch(suffix) is not None


def build_jupyter_bootstrap_controller(
    settings: SchedulerSettings,
    database: Database,
    connections: AgentConnectionRegistry,
) -> JupyterBootstrapController | None:
    """Build the optional controller, reading secrets only when it is enabled."""

    bootstrap = settings.jupyter_bootstrap
    if not bootstrap.enabled:
        return None
    secrets = load_secret_bundle(settings.secret_file)
    if secrets.jupyter_bootstrap_url is None:
        raise ValueError(
            "TOPODROME_JUPYTER_BOOTSTRAP_URL is required when Jupyter bootstrap is enabled"
        )
    if secrets.jupyter_bootstrap_token is None:
        raise ValueError(
            "TOPODROME_JUPYTER_BOOTSTRAP_TOKEN is required when Jupyter bootstrap is enabled"
        )
    base_url = secrets.jupyter_bootstrap_url.get_secret_value()
    parts = urlsplit(base_url)
    if parts.query or parts.fragment or parts.username or parts.password:
        raise ValueError("Jupyter bootstrap URL must not contain credentials, query, or fragment")
    client = JupyterTerminalClient(base_url, secrets.jupyter_bootstrap_token)
    site_key = hashlib.sha256(bootstrap.site_id.encode()).hexdigest()[:16]
    ownership_path = (
        settings.secret_file.expanduser().absolute().parent
        / "state"
        / f"jupyter-bootstrap-{site_key}.json"
    )
    return JupyterBootstrapController(
        database=database,
        connections=connections,
        settings=bootstrap,
        heartbeat_ttl_seconds=settings.heartbeat_ttl_seconds,
        client=client,
        ownership_path=ownership_path,
        terminal_origin=base_url,
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_OWNED_TERMINAL_SUFFIX = re.compile(r"[a-f0-9]{24}")


__all__ = [
    "JupyterBootstrapController",
    "TerminalBootstrapClient",
    "build_jupyter_bootstrap_controller",
]
