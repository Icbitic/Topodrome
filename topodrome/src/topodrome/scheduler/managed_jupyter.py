"""Supervise stock Jupyter Servers backed by scoped Topodrome gateways.

The IDE-facing server token is stable for the life of a Topodrome kernel and
one owner-whitelist credential revision. The credential that the child server
uses against the scheduler gateway is a separate HMAC capability scoped to one
child-process generation. It is passed only in the child environment and
accepted only while that exact process is authoritative.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import select
import shutil
import signal
import socket
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import httpx
import structlog
from pydantic import SecretStr

from topodrome.config import ManagedJupyterSettings, SchedulerSettings, load_secret_bundle
from topodrome.domain.enums import KernelState
from topodrome.domain.errors import ConflictError, NotFoundError
from topodrome.domain.specs import KernelSummary, utc_now

if TYPE_CHECKING:
    from topodrome.scheduler.service import SchedulerService

log = structlog.get_logger(__name__)

_OWNER_USERNAME = re.compile(r"^[a-z_][a-z0-9_.-]{0,63}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class JupyterServerProbe:
    ready: bool
    gateway_ready: bool
    last_activity_at: datetime
    connections: int


class ManagedProcess(Protocol):
    pid: int

    @property
    def returncode(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


class ManagedProcessLauncher(Protocol):
    async def launch(
        self,
        argv: Sequence[str],
        *,
        environment: Mapping[str, str],
        cwd: Path,
    ) -> ManagedProcess: ...


class AsyncioProcessLauncher:
    async def launch(
        self,
        argv: Sequence[str],
        *,
        environment: Mapping[str, str],
        cwd: Path,
    ) -> ManagedProcess:
        return await asyncio.create_subprocess_exec(
            *argv,
            env=dict(environment),
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )


class ManagedKernelService(Protocol):
    async def get_kernel(self, session_id: str) -> KernelSummary: ...

    async def stop_kernel(self, session_id: str) -> KernelSummary: ...

    async def refresh_managed_kernel_connection(
        self, session_id: str
    ) -> KernelSummary: ...


ProbeFunction = Callable[
    [str, int, str, bool], Awaitable[JupyterServerProbe | None]
]


@dataclass(slots=True)
class _ServerState:
    session_id: str
    port: int
    owner_username: str | None = None
    owner_access_revision: str | None = None
    desired: bool = True
    pid: int | None = None
    process_start_ticks: int | None = None
    bridge_generation: str | None = None
    started_at: datetime | None = None
    last_activity_at: datetime | None = None
    failure_count: int = 0
    retry_at: datetime | None = None
    process: ManagedProcess | None = field(default=None, repr=False)


class ManagedJupyterController:
    """Own IDE-facing Jupyter Server processes without owning GPU execution."""

    def __init__(
        self,
        *,
        settings: ManagedJupyterSettings,
        global_root: Path,
        scheduler_public_url: str,
        server_token_key: SecretStr,
        kernels: ManagedKernelService,
        launcher: ManagedProcessLauncher | None = None,
        probe: ProbeFunction | None = None,
        now: Callable[[], datetime] = utc_now,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        generation_factory: Callable[[], str] | None = None,
        owner_authorizer: Callable[[str], bool] | None = None,
        owner_revision_provider: Callable[[str], str | None] | None = None,
    ) -> None:
        if not settings.enabled:
            raise ValueError("managed Jupyter controller requires enabled settings")
        key = server_token_key.get_secret_value()
        if len(key.encode()) < 32:
            raise ValueError("managed Jupyter server token must contain at least 32 bytes")
        self.settings = settings
        self.kernels = kernels
        self.scheduler_public_url = _validated_url(scheduler_public_url)
        self.notebooks_root = (
            settings.notebooks_root or global_root / "notebooks"
        ).expanduser().resolve()
        self.runtime_root = (
            settings.runtime_root or global_root / ".topodrome-jupyter-runtime"
        ).expanduser().resolve()
        self._state_root = self.runtime_root / "state"
        self._process_root = self.runtime_root / "servers"
        self._server_token_key = key.encode()
        self._launcher = launcher or AsyncioProcessLauncher()
        self._probe_override = probe
        self._now = now
        self._sleep = sleep
        self._generation_factory = generation_factory or (
            lambda: secrets.token_urlsafe(24)
        )
        self._owner_authorizer = owner_authorizer or (
            lambda _owner_username: True
        )
        self._owner_revision_provider = owner_revision_provider or (
            lambda _owner_username: "legacy"
        )
        self._lock = asyncio.Lock()
        self._states: dict[str, _ServerState] = {}
        self._closed = False
        self._prepare_roots()
        self._load_states()

    def set_owner_authorizer(
        self,
        authorizer: Callable[[str], bool] | None,
        revision_provider: Callable[[str], str | None] | None = None,
    ) -> None:
        self._owner_authorizer = authorizer or (lambda _owner_username: True)
        self._owner_revision_provider = revision_provider or (
            lambda _owner_username: "legacy"
        )

    async def ensure_server(self, session_id: str) -> dict[str, Any]:
        """Start or recover one stable stock server for an active kernel."""

        async with self._lock:
            self._require_open()
            kernel = await self.kernels.get_kernel(session_id)
            self._require_running_kernel(kernel)
            state = self._states.get(session_id)
            try:
                owner_username, owner_access_revision = (
                    self._authorized_kernel_owner(kernel)
                )
            except ConflictError:
                if state is not None:
                    await self._remove_terminal_state(state)
                raise
            if state is None:
                desired_count = sum(item.desired for item in self._states.values())
                if desired_count >= self.settings.max_servers:
                    raise ConflictError("managed Jupyter server limit has been reached")
                state = _ServerState(
                    session_id=session_id,
                    port=self._select_port(),
                    owner_username=owner_username,
                    owner_access_revision=owner_access_revision,
                )
                self._states[session_id] = state
                self._save_state(state)
            else:
                try:
                    legacy_root_migration = self._bind_owner(
                        state,
                        owner_username,
                        owner_access_revision,
                    )
                except ConflictError:
                    await self._remove_terminal_state(state)
                    raise
                state.desired = True
                self._save_state(state)
                if legacy_root_migration and self._process_matches(state):
                    # A pre-multi-user child for an ordinary account was rooted
                    # at the shared legacy directory. Revoke and replace only
                    # that exact child before returning any IDE credential.
                    await self._stop_process(state, preserve_desired=True)

            # Never replace a healthy process merely because its upstream
            # gateway is temporarily unreachable. That would break IDE state.
            if self._process_matches(state) and state.bridge_generation is not None:
                probe = await self._probe(state, verify_gateway=True)
                if probe is None:
                    raise ConflictError("managed Jupyter Server is temporarily unavailable")
                state.last_activity_at = probe.last_activity_at
                state.failure_count = 0
                self._save_state(state)
                if not probe.gateway_ready:
                    raise ConflictError("managed Jupyter kernel gateway is unavailable")
                return self._public_info(state, ready=True)

            await self._stop_process(state, preserve_desired=True)
            await self._start_process(state)
            return self._public_info(state, ready=True)

    async def server_info(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            state = self._states.get(session_id)
            if state is None or not state.desired:
                raise NotFoundError(f"managed Jupyter server for {session_id} does not exist")
            kernel = await self.kernels.get_kernel(session_id)
            self._require_running_kernel(kernel)
            try:
                owner_username, owner_access_revision = (
                    self._authorized_kernel_owner(kernel)
                )
                root_migration = self._bind_owner(
                    state,
                    owner_username,
                    owner_access_revision,
                )
            except ConflictError:
                await self._remove_terminal_state(state)
                raise
            if root_migration:
                await self._remove_terminal_state(state)
                raise ConflictError(
                    "managed Jupyter Server must be restarted for owner isolation"
                )
            self._save_state(state)
            ready = await self._is_ready(state, verify_gateway=False)
            return self._public_info(state, ready=ready)

    async def stop_server(
        self,
        session_id: str,
        *,
        stop_kernel: bool = True,
    ) -> dict[str, Any]:
        """Revoke the bridge, stop its exact process, and optionally stop the kernel."""

        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                raise NotFoundError(f"managed Jupyter server for {session_id} does not exist")
            await self._stop_process(state, preserve_desired=False)
            state.desired = False
            info = self._public_info(state, ready=False, include_token=False)
            self._states.pop(session_id, None)
        if stop_kernel:
            with suppress(ConflictError, NotFoundError):
                summary = await self.kernels.get_kernel(session_id)
                if summary.state not in {KernelState.STOPPED, KernelState.LOST}:
                    await self.kernels.stop_kernel(session_id)
        return info

    def verify_bridge_credential(self, session_id: str, credential: str) -> bool:
        """Validate a child-only gateway capability against live process state.

        This performs no I/O or lock acquisition so the scheduler can call it
        while holding a database transaction. Mutations occur on the same
        asyncio event-loop thread; the HMAC comparison is constant-time.
        """

        state = self._states.get(session_id)
        if (
            self._closed
            or state is None
            or not state.desired
            or state.bridge_generation is None
            or not self._process_matches(state)
        ):
            return False
        expected = self._bridge_token(session_id, state.bridge_generation)
        return hmac.compare_digest(credential, expected)

    async def revoke_owner(self, owner_username: str) -> int:
        """Revoke and reap managed children for one owner, preserving kernels."""

        normalized = _validated_owner_username(owner_username)
        reaped = 0
        async with self._lock:
            for state in tuple(self._states.values()):
                try:
                    kernel = await self.kernels.get_kernel(state.session_id)
                except NotFoundError:
                    await self._remove_terminal_state(state)
                    continue
                kernel_owner = _validated_owner_username(kernel.owner_username)
                if state.owner_username is not None and not hmac.compare_digest(
                    state.owner_username,
                    kernel_owner,
                ):
                    # A persisted owner mismatch can never be served safely.
                    await self._remove_terminal_state(state)
                    continue
                if hmac.compare_digest(kernel_owner, normalized):
                    await self._remove_terminal_state(state)
                    reaped += 1
        return reaped

    async def reconcile_once(self) -> None:
        async with self._lock:
            if self._closed:
                return
            session_ids = tuple(self._states)
        for session_id in session_ids:
            stop_kernel = False
            async with self._lock:
                if self._closed:
                    return
                state = self._states.get(session_id)
                if state is None:
                    continue
                if not state.desired:
                    # Older schedulers could persist desired=false before the
                    # owned child was actually terminated. Retain and safely
                    # reap that exact recovered identity instead of orphaning
                    # it or widening cleanup to process enumeration.
                    await self._remove_terminal_state(state)
                    continue
                try:
                    kernel = await self.kernels.get_kernel(session_id)
                except NotFoundError:
                    await self._remove_terminal_state(state)
                    continue
                if kernel.state not in {KernelState.RUNNING, KernelState.IDLE}:
                    await self._remove_terminal_state(state)
                    continue
                try:
                    owner_username, owner_access_revision = (
                        self._authorized_kernel_owner(kernel)
                    )
                    legacy_root_migration = self._bind_owner(
                        state,
                        owner_username,
                        owner_access_revision,
                    )
                    self._save_state(state)
                except ConflictError:
                    # Access revocation or an ownership mismatch invalidates
                    # only the IDE child. The underlying GPU kernel continues.
                    await self._remove_terminal_state(state)
                    continue
                if legacy_root_migration:
                    await self._stop_process(state, preserve_desired=True)
                    await self._start_with_backoff(state, _as_utc(self._now()))
                    continue

                now = _as_utc(self._now())
                if not self._process_matches(state) or state.bridge_generation is None:
                    if state.retry_at is None or _as_utc(state.retry_at) <= now:
                        await self._stop_process(state, preserve_desired=True)
                        await self._start_with_backoff(state, now)
                    continue

                if state.retry_at is not None and _as_utc(state.retry_at) > now:
                    continue

                probe = await self._probe(state, verify_gateway=True)
                if probe is None:
                    # Give a matching but temporarily unavailable PID three
                    # bounded reconciliation windows before replacement.
                    state.failure_count += 1
                    if state.failure_count >= 3:
                        await self._stop_process(state, preserve_desired=True)
                        await self._start_with_backoff(state, now)
                    else:
                        state.retry_at = now + timedelta(seconds=2 ** state.failure_count)
                        self._save_state(state)
                    continue

                state.failure_count = 0
                state.retry_at = None
                state.last_activity_at = probe.last_activity_at
                self._save_state(state)
                if probe.connections > 0:
                    try:
                        await self.kernels.refresh_managed_kernel_connection(session_id)
                    except (ConflictError, NotFoundError):
                        # The kernel changed state after the first read. Revoke
                        # the generation and stop only this exact managed child.
                        await self._remove_terminal_state(state)
                        continue
                idle_for = now - probe.last_activity_at
                if (
                    probe.connections == 0
                    and idle_for >= timedelta(seconds=self.settings.idle_timeout_seconds)
                ):
                    await self._remove_terminal_state(state)
                    stop_kernel = True
                elif not probe.gateway_ready:
                    # An upstream outage must not rotate a healthy IDE server.
                    log.warning(
                        "managed_jupyter_gateway_unavailable",
                        session_id=session_id,
                        pid=state.pid,
                    )
            if stop_kernel:
                with suppress(ConflictError, NotFoundError):
                    await self.kernels.stop_kernel(session_id)

    async def close(self, *, terminate: bool = False) -> None:
        """Close supervision, preserving active IDE servers unless requested.

        Children run in their own process session and have durable non-secret
        state, so a replacement scheduler can recover the same PID and port.
        Passing terminate=True is useful to service managers and hermetic tests.
        """

        async with self._lock:
            if self._closed:
                return
            if terminate:
                for state in tuple(self._states.values()):
                    await self._stop_process(state, preserve_desired=True)
            else:
                for state in self._states.values():
                    self._save_state(state)
            self._closed = True

    def _prepare_roots(self) -> None:
        if self.runtime_root == Path("/") or self.notebooks_root == Path("/"):
            raise ValueError("managed Jupyter roots must be scoped directories")
        if self.runtime_root.is_relative_to(
            self.notebooks_root
        ) or self.notebooks_root.is_relative_to(self.runtime_root):
            raise ValueError("managed Jupyter runtime and notebook roots must not overlap")
        self.notebooks_root.mkdir(parents=True, exist_ok=True)
        self._state_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._process_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        if self.notebooks_root.is_symlink():
            raise ValueError("managed Jupyter notebook root cannot be a symlink")
        for path in (self.runtime_root, self._state_root, self._process_root):
            if path.is_symlink():
                raise ValueError("managed Jupyter runtime directories cannot be symlinks")
            path.chmod(0o700)

    def _load_states(self) -> None:
        for path in sorted(self._state_root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                session_id = str(raw["session_id"])
                if path.name != f"{_state_key(session_id)}.json":
                    raise ValueError
                state = _ServerState(
                    session_id=session_id,
                    port=int(raw["port"]),
                    owner_username=_optional_string(raw.get("owner_username")),
                    owner_access_revision=_optional_string(
                        raw.get("owner_access_revision")
                    ),
                    desired=bool(raw["desired"]),
                    pid=_optional_int(raw.get("pid")),
                    process_start_ticks=_optional_int(raw.get("process_start_ticks")),
                    bridge_generation=_optional_string(raw.get("bridge_generation")),
                    started_at=_optional_datetime(raw.get("started_at")),
                    last_activity_at=_optional_datetime(raw.get("last_activity_at")),
                    failure_count=max(0, int(raw.get("failure_count", 0))),
                    retry_at=_optional_datetime(raw.get("retry_at")),
                )
                if not self.settings.port_start <= state.port <= self.settings.port_end:
                    raise ValueError
                if state.desired or state.pid is not None:
                    self._states[session_id] = state
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                log.warning("managed_jupyter_state_ignored", state_file=path.name[:100])

    async def _remove_terminal_state(self, state: _ServerState) -> None:
        await self._stop_process(state, preserve_desired=False)
        state.desired = False
        self._states.pop(state.session_id, None)

    async def _start_with_backoff(self, state: _ServerState, now: datetime) -> None:
        try:
            await self._start_process(state)
        except Exception as exc:
            state.failure_count += 1
            delay = min(2 ** min(state.failure_count - 1, 8), 300)
            state.retry_at = now + timedelta(seconds=delay)
            self._save_state(state)
            log.warning(
                "managed_jupyter_recovery_failed",
                session_id=state.session_id,
                error_type=type(exc).__name__[:80],
                retry_seconds=delay,
            )

    async def _start_process(self, state: _ServerState) -> None:
        kernel = await self.kernels.get_kernel(state.session_id)
        self._require_running_kernel(kernel)
        owner_username, owner_access_revision = self._authorized_kernel_owner(kernel)
        self._bind_owner(state, owner_username, owner_access_revision)
        if not _port_is_available(self.settings.bind_host, state.port):
            state.port = self._select_port(exclude={state.port})

        generation = self._generation_factory()
        if not generation or len(generation) > 200 or any(ch.isspace() for ch in generation):
            raise ValueError("managed Jupyter bridge generation is invalid")
        state.bridge_generation = generation
        runtime_dir = self._runtime_dir(state.session_id)
        self._clear_runtime_dir(runtime_dir)
        runtime_dir.mkdir(mode=0o700)
        config_dir = runtime_dir / "config"
        data_dir = runtime_dir / "data"
        jupyter_runtime = runtime_dir / "runtime"
        for path in (config_dir, data_dir, jupyter_runtime):
            path.mkdir(mode=0o700)

        server_token = self._server_token(state)
        gateway_url, gateway_ws_url = self._gateway_urls(state.session_id)
        notebooks_root = self._notebooks_root_for_owner(owner_username)
        environment = _minimal_environment()
        environment.update(
            {
                "JUPYTER_CONFIG_DIR": str(config_dir),
                "JUPYTER_DATA_DIR": str(data_dir),
                "JUPYTER_RUNTIME_DIR": str(jupyter_runtime),
                "JUPYTER_TOKEN": server_token,
                "JUPYTER_GATEWAY_URL": gateway_url,
                "JUPYTER_GATEWAY_WS_URL": gateway_ws_url,
                "JUPYTER_GATEWAY_AUTH_SCHEME": "token",
                "JUPYTER_GATEWAY_AUTH_TOKEN": self._bridge_token(
                    state.session_id, generation
                ),
                "JUPYTER_GATEWAY_ALLOWED_ENVS": "",
            }
        )
        argv = (
            self.settings.executable,
            "--no-browser",
            f"--ServerApp.ip={self.settings.bind_host}",
            f"--ServerApp.port={state.port}",
            "--ServerApp.port_retries=0",
            f"--ServerApp.root_dir={notebooks_root}",
            "--ServerApp.open_browser=False",
            "--ServerApp.no_browser_open_file=True",
            "--ServerApp.allow_unauthenticated_access=False",
            "--ServerApp.disable_check_xsrf=False",
            "--ServerApp.allow_remote_access=True",
            "--ServerApp.terminals_enabled=False",
            (
                "--ServerApp.kernel_manager_class="
                "topodrome.jupyter.server_bridge."
                "AuthenticatedGatewayMappingKernelManager"
            ),
            (
                "--ServerApp.session_manager_class="
                "topodrome.jupyter.server_bridge.ManagedGatewaySessionManager"
            ),
        )
        process = await self._launcher.launch(
            argv,
            environment=environment,
            cwd=notebooks_root,
        )
        process_start_ticks = _pid_start_ticks(process.pid)
        if (
            process_start_ticks is None
            or process.returncode is not None
            or _pid_start_ticks(process.pid) != process_start_ticks
        ):
            # The direct child handle is still authoritative even though a
            # restart-safe PID generation could not be captured. Terminate it
            # before persisting any PID; otherwise a scheduler crash could
            # leave an authenticated server that can never be safely reaped.
            await self._terminate_direct_process(process)
            state.process = None
            state.pid = None
            state.process_start_ticks = None
            state.bridge_generation = None
            self._save_state(state)
            self._clear_runtime_dir(runtime_dir)
            raise ConflictError(
                "managed Jupyter Server process identity could not be proven"
            )
        state.process = process
        state.pid = process.pid
        state.process_start_ticks = process_start_ticks
        state.started_at = _as_utc(self._now())
        state.last_activity_at = state.started_at
        state.retry_at = None
        self._save_state(state)
        deadline = asyncio.get_running_loop().time() + self.settings.startup_timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                break
            probe = await self._probe(state, verify_gateway=True)
            if probe is not None and probe.gateway_ready:
                state.failure_count = 0
                state.last_activity_at = probe.last_activity_at
                self._save_state(state)
                log.info(
                    "managed_jupyter_started",
                    session_id=state.session_id,
                    port=state.port,
                    pid=state.pid,
                )
                return
            await self._sleep(0.1)
        await self._stop_process(state, preserve_desired=True)
        raise ConflictError("managed Jupyter Server failed its readiness check")

    async def _is_ready(self, state: _ServerState, *, verify_gateway: bool) -> bool:
        probe = await self._probe(state, verify_gateway=verify_gateway)
        return probe is not None and probe.ready and (
            probe.gateway_ready or not verify_gateway
        )

    async def _probe(
        self,
        state: _ServerState,
        *,
        verify_gateway: bool,
    ) -> JupyterServerProbe | None:
        if state.pid is None or not self._process_matches(state):
            return None
        server_token = self._server_token(state)
        if self._probe_override is not None:
            return await self._probe_override(
                state.session_id,
                state.port,
                server_token,
                verify_gateway,
            )
        internal_url = f"http://{_probe_host(self.settings.bind_host)}:{state.port}"
        headers = {"Authorization": f"token {server_token}"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(1.5), headers=headers) as client:
                status_response = await client.get(
                    f"{internal_url}/api/status",
                    params={"no_track_activity": "1"},
                )
                status_response.raise_for_status()
                payload = status_response.json()
                gateway_ready = True
                if verify_gateway:
                    gateway_response = await client.get(
                        f"{internal_url}/api/kernelspecs",
                        params={"no_track_activity": "1"},
                    )
                    gateway_ready = gateway_response.status_code == 200
            return JupyterServerProbe(
                ready=True,
                gateway_ready=gateway_ready,
                last_activity_at=_parse_datetime(payload["last_activity"]),
                connections=max(0, int(payload.get("connections", 0))),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    async def _stop_process(self, state: _ServerState, *, preserve_desired: bool) -> None:
        # Revocation precedes termination. No delayed request from this child
        # can reach the kernel once its generation is cleared.
        state.bridge_generation = None
        # Persist revocation before any signal. If the scheduler exits during
        # cleanup, its replacement can recover the exact PID/start identity but
        # can never reactivate the old child's gateway capability.
        if state.pid is not None:
            self._save_state(state)
        process = state.process
        if process is not None and process.returncode is None:
            await self._terminate_direct_process(process)
        elif state.pid is not None and self._process_matches(state):
            await self._stop_recovered_process(state)
        state.process = None
        state.pid = None
        state.process_start_ticks = None
        state.retry_at = None
        if preserve_desired:
            state.desired = True
            self._save_state(state)
        else:
            self._delete_state(state)
            self._clear_runtime_dir(self._runtime_dir(state.session_id))

    async def _terminate_direct_process(self, process: ManagedProcess) -> None:
        """Terminate the exact child represented by an unreleased process handle."""

        if process.returncode is not None:
            return
        process.terminate()
        try:
            async with asyncio.timeout(self.settings.shutdown_timeout_seconds):
                await process.wait()
        except TimeoutError:
            process.kill()
            await process.wait()

    async def _stop_recovered_process(self, state: _ServerState) -> None:
        """Signal one recovered child through a race-free Linux pidfd.

        A numeric PID can be recycled between a ``/proc`` identity check and
        ``os.kill``. Opening the pidfd first pins the kernel process object; the
        subsequent start-tick check proves that object is the persisted child.
        """

        assert state.pid is not None
        assert state.process_start_ticks is not None
        pidfd = _open_verified_pidfd(state.pid, state.process_start_ticks)
        if pidfd is None:
            return
        try:
            _send_pidfd_signal(pidfd, signal.SIGTERM)
            deadline = (
                asyncio.get_running_loop().time()
                + self.settings.shutdown_timeout_seconds
            )
            while not _pidfd_has_exited(pidfd) and asyncio.get_running_loop().time() < deadline:
                await self._sleep(0.05)
            if _pidfd_has_exited(pidfd):
                return
            _send_pidfd_signal(pidfd, signal.SIGKILL)
            kill_deadline = (
                asyncio.get_running_loop().time()
                + self.settings.shutdown_timeout_seconds
            )
            while (
                not _pidfd_has_exited(pidfd)
                and asyncio.get_running_loop().time() < kill_deadline
            ):
                await self._sleep(0.05)
            if not _pidfd_has_exited(pidfd):
                raise RuntimeError("managed Jupyter process did not exit after SIGKILL")
        finally:
            os.close(pidfd)

    def _process_matches(self, state: _ServerState) -> bool:
        if state.process is not None:
            return (
                state.process.returncode is None
                and state.pid == state.process.pid
                and state.process_start_ticks is not None
                and _pid_start_ticks(state.process.pid) == state.process_start_ticks
            )
        return (
            state.pid is not None
            and state.process_start_ticks is not None
            and _pid_start_ticks(state.pid) == state.process_start_ticks
        )

    def _select_port(self, *, exclude: set[int] | None = None) -> int:
        excluded = set(exclude or ())
        excluded.update(state.port for state in self._states.values() if state.desired)
        for port in range(self.settings.port_start, self.settings.port_end + 1):
            if port not in excluded and _port_is_available(self.settings.bind_host, port):
                return port
        raise ConflictError("no managed Jupyter Server port is available")

    def _public_info(
        self,
        state: _ServerState,
        *,
        ready: bool | None = None,
        include_token: bool = True,
    ) -> dict[str, Any]:
        running = self._process_matches(state) if ready is None else ready
        url = self.settings.public_url_template.format(port=state.port)
        token = self._server_token(state) if include_token else None
        return {
            "session_id": state.session_id,
            "state": "running" if running else "stopped",
            "url": url,
            "token": token,
            "token_url": (
                f"{url}/?{urlencode({'token': token})}" if token is not None else None
            ),
            "started_at": state.started_at,
            "last_activity_at": state.last_activity_at,
        }

    def _server_token(self, state: _ServerState) -> str:
        scope = f"managed-jupyter-server:{state.session_id}"
        if state.owner_username not in {None, "@operator"}:
            if state.owner_access_revision is None:
                raise ConflictError("managed Jupyter owner revision is unavailable")
            scope = (
                f"{scope}:{state.owner_username}:"
                f"{state.owner_access_revision}"
            )
        return _derived_token(
            self._server_token_key,
            scope,
        )

    def _bridge_token(self, session_id: str, generation: str) -> str:
        return _derived_token(
            self._server_token_key,
            f"managed-jupyter-bridge:{session_id}:{generation}",
        )

    def _gateway_urls(self, session_id: str) -> tuple[str, str]:
        parsed = urlsplit(self.scheduler_public_url)
        path = (
            f"{parsed.path.rstrip('/')}/api/v1/kernel-relay/"
            f"{quote(session_id, safe='')}/gateway"
        )
        base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        websocket_scheme = "wss" if parsed.scheme == "https" else "ws"
        ws_url = urlunsplit((websocket_scheme, parsed.netloc, path, "", ""))
        return base_url, ws_url

    def _runtime_dir(self, session_id: str) -> Path:
        return self._process_root / _state_key(session_id)

    def _notebooks_root_for_owner(self, owner_username: str) -> Path:
        owner = _validated_owner_username(owner_username)
        if owner == "@operator":
            # Preserve the pre-multi-user path for the operator so existing
            # notebooks and IDE configurations continue to work.
            return self.notebooks_root
        users_root = self.notebooks_root / "users"
        owner_root = users_root / owner
        for path in (users_root, owner_root):
            path.mkdir(mode=0o700, exist_ok=True)
            if path.is_symlink():
                raise ConflictError("managed Jupyter owner root cannot be a symlink")
            path.chmod(0o700)
        resolved_base = self.notebooks_root.resolve()
        resolved_owner = owner_root.resolve()
        if not resolved_owner.is_relative_to(resolved_base):
            raise ConflictError("managed Jupyter owner root escaped its base")
        return owner_root

    def _authorized_kernel_owner(self, kernel: KernelSummary) -> tuple[str, str | None]:
        owner = _validated_owner_username(kernel.owner_username)
        if owner == "@operator":
            return owner, None
        try:
            authorized = self._owner_authorizer(owner)
            revision = self._owner_revision_provider(owner) if authorized else None
        except Exception as exc:
            log.warning(
                "managed_jupyter_owner_authorization_failed",
                owner_username=owner,
                error_type=type(exc).__name__[:80],
            )
            authorized = False
            revision = None
        if not authorized or not _valid_owner_access_revision(revision):
            raise ConflictError("managed Jupyter owner access has been revoked")
        return owner, revision

    @staticmethod
    def _bind_owner(
        state: _ServerState,
        owner_username: str,
        owner_access_revision: str | None,
    ) -> bool:
        """Bind recovered ownership and reject stale credential revisions."""

        owner = _validated_owner_username(owner_username)
        if state.owner_username is None:
            state.owner_username = owner
            state.owner_access_revision = owner_access_revision
            return owner != "@operator"
        if not hmac.compare_digest(state.owner_username, owner):
            raise ConflictError("managed Jupyter state owner does not match its kernel")
        if owner == "@operator":
            state.owner_access_revision = None
            return False
        if state.owner_access_revision is None:
            state.owner_access_revision = owner_access_revision
            return True
        if owner_access_revision is None or not hmac.compare_digest(
            state.owner_access_revision,
            owner_access_revision,
        ):
            raise ConflictError("managed Jupyter owner credential was rotated")
        return False

    def _save_state(self, state: _ServerState) -> None:
        payload = {
            "session_id": state.session_id,
            "port": state.port,
            "owner_username": state.owner_username,
            "owner_access_revision": state.owner_access_revision,
            "desired": state.desired,
            "pid": state.pid,
            "process_start_ticks": state.process_start_ticks,
            "bridge_generation": state.bridge_generation,
            "started_at": _format_datetime(state.started_at),
            "last_activity_at": _format_datetime(state.last_activity_at),
            "failure_count": state.failure_count,
            "retry_at": _format_datetime(state.retry_at),
        }
        path = self._state_path(state.session_id)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
        _fsync_directory(self._state_root)

    def _delete_state(self, state: _ServerState) -> None:
        path = self._state_path(state.session_id)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(self._state_root)

    def _state_path(self, session_id: str) -> Path:
        return self._state_root / f"{_state_key(session_id)}.json"

    def _clear_runtime_dir(self, path: Path) -> None:
        resolved_parent = self._process_root.resolve()
        if path.parent != self._process_root or not path.resolve().is_relative_to(resolved_parent):
            raise RuntimeError("managed Jupyter runtime path escaped its root")
        if path.exists():
            shutil.rmtree(path)

    @staticmethod
    def _require_running_kernel(kernel: KernelSummary) -> None:
        if kernel.state not in {KernelState.RUNNING, KernelState.IDLE}:
            raise ConflictError("kernel must be running before starting its Jupyter server")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("managed Jupyter controller is closed")


def build_managed_jupyter_controller(
    settings: SchedulerSettings,
    kernels: SchedulerService,
) -> ManagedJupyterController | None:
    managed = settings.managed_jupyter
    if not managed.enabled:
        return None
    secrets_bundle = load_secret_bundle(settings.secret_file)
    if secrets_bundle.managed_jupyter_server_token is None:
        raise ValueError(
            "TOPODROME_MANAGED_JUPYTER_SERVER_TOKEN is required when managed Jupyter is enabled"
        )
    return ManagedJupyterController(
        settings=managed,
        global_root=settings.global_root,
        scheduler_public_url=settings.public_url,
        server_token_key=secrets_bundle.managed_jupyter_server_token,
        kernels=kernels,
        owner_authorizer=getattr(kernels, "is_owner_authorized", None),
        owner_revision_provider=getattr(kernels, "owner_access_revision", None),
    )


def _derived_token(key: bytes, scope: str) -> str:
    digest = hmac.digest(key, scope.encode(), hashlib.sha256)
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _validated_url(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("scheduler public URL must be a credential-free HTTP URL")
    return value.rstrip("/")


def _state_key(session_id: str) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()


def _validated_owner_username(value: object) -> str:
    owner = value if isinstance(value, str) else ""
    if owner == "@operator":
        return owner
    if not _OWNER_USERNAME.fullmatch(owner) or owner.startswith("."):
        raise ConflictError("kernel owner is invalid")
    return owner


def _valid_owner_access_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 200
        and not any(character.isspace() for character in value)
    )


def _minimal_environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONUNBUFFERED": "1",
    }


def _port_is_available(host: str, port: int) -> bool:
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    for family, kind, protocol, _, address in addresses:
        probe = socket.socket(family, kind, protocol)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(address)
            return True
        except OSError:
            continue
        finally:
            probe.close()
    return False


def _probe_host(bind_host: str) -> str:
    if bind_host in {"0.0.0.0", "::", "*"}:  # noqa: S104 - map wildcard to loopback
        return "127.0.0.1"
    return f"[{bind_host}]" if ":" in bind_host and not bind_host.startswith("[") else bind_host


def _pid_start_ticks(pid: int) -> int | None:
    try:
        body = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = body[body.rfind(")") + 2 :].split()
        return int(fields[19])
    except (IndexError, OSError, ValueError):
        return None


def _open_verified_pidfd(pid: int, expected_start_ticks: int) -> int | None:
    """Open the current PID and prove it is the persisted process identity."""

    opener = getattr(os, "pidfd_open", None)
    sender = getattr(signal, "pidfd_send_signal", None)
    if not callable(opener) or not callable(sender):
        raise RuntimeError(
            "safe recovered-process signaling requires Linux pidfd support"
        )
    try:
        pidfd = int(opener(pid, 0))
    except ProcessLookupError:
        return None
    except OSError as exc:
        raise RuntimeError("could not open managed Jupyter process pidfd") from exc
    if _pid_start_ticks(pid) != expected_start_ticks:
        os.close(pidfd)
        return None
    return pidfd


def _send_pidfd_signal(pidfd: int, signum: int) -> None:
    sender = getattr(signal, "pidfd_send_signal", None)
    if not callable(sender):  # pragma: no cover - checked while opening
        raise RuntimeError(
            "safe recovered-process signaling requires Linux pidfd support"
        )
    try:
        sender(pidfd, signum, None, 0)
    except ProcessLookupError:
        # The pinned process exited after the liveness poll. No replacement PID
        # can be targeted through this descriptor.
        return
    except OSError as exc:
        raise RuntimeError("could not signal managed Jupyter process pidfd") from exc


def _pidfd_has_exited(pidfd: int) -> bool:
    try:
        readable, _, _ = select.select((pidfd,), (), (), 0)
    except OSError as exc:
        raise RuntimeError("could not poll managed Jupyter process pidfd") from exc
    return bool(readable)


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("invalid datetime")
    return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _optional_datetime(value: Any) -> datetime | None:
    return None if value is None else _parse_datetime(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _format_datetime(value: datetime | None) -> str | None:
    return None if value is None else _as_utc(value).isoformat()


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "AsyncioProcessLauncher",
    "JupyterServerProbe",
    "ManagedJupyterController",
    "ManagedProcess",
    "ManagedProcessLauncher",
    "build_managed_jupyter_controller",
]
