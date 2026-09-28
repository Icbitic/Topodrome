"""Optional local kernel executor using ``jupyter-client`` lazily."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, cast

from topodrome.jupyter.terminal import JupyterDependencyUnavailable

_SECRET_KEY = re.compile(
    r"(?:authorization|token|secret|password|credential|api[_-]?key)", re.IGNORECASE
)


class _KernelClient(Protocol):
    def start_channels(self) -> None: ...

    def stop_channels(self) -> None: ...

    async def wait_for_ready(self, wait_seconds: float) -> None: ...


class _KernelManager(Protocol):
    connection_file: str

    async def start_kernel(self, *, env: dict[str, str] | None = None) -> None: ...

    def client(self) -> _KernelClient: ...

    async def shutdown_kernel(self, *, now: bool = False) -> None: ...


ManagerFactory = Callable[[str], _KernelManager]


@dataclass(frozen=True, slots=True)
class LocalKernelHandle:
    session_id: str
    connection_file: str


@dataclass(slots=True)
class _RunningKernel:
    manager: _KernelManager
    client: _KernelClient


class LocalJupyterClientExecutor:
    """Starts local kernels but does not provide a browser HTTP proxy."""

    def __init__(
        self,
        *,
        kernel_name: str = "python3",
        ready_timeout_seconds: float = 30,
        manager_factory: ManagerFactory | None = None,
    ) -> None:
        if not kernel_name or ready_timeout_seconds <= 0:
            raise ValueError("kernel name and ready timeout must be valid")
        self._kernel_name = kernel_name
        self._ready_timeout = ready_timeout_seconds
        self._manager_factory = manager_factory
        self._running: dict[str, _RunningKernel] = {}

    async def start(
        self,
        session_id: str,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> LocalKernelHandle:
        if not session_id:
            raise ValueError("session_id must not be empty")
        if session_id in self._running:
            raise RuntimeError("session already has a local kernel")
        environment_dict = dict(environment or {})
        if any(_SECRET_KEY.search(key) for key in environment_dict):
            raise ValueError("credentials cannot be forwarded into notebook kernels")

        manager = self._new_manager()
        client: _KernelClient | None = None
        try:
            await manager.start_kernel(env=environment_dict or None)
            client = manager.client()
            client.start_channels()
            await client.wait_for_ready(self._ready_timeout)
        except Exception:
            if client is not None:
                client.stop_channels()
            with suppress(Exception):
                await manager.shutdown_kernel(now=True)
            raise
        self._running[session_id] = _RunningKernel(manager=manager, client=client)
        return LocalKernelHandle(session_id=session_id, connection_file=manager.connection_file)

    async def stop(self, session_id: str) -> None:
        running = self._running.pop(session_id, None)
        if running is None:
            return
        running.client.stop_channels()
        await running.manager.shutdown_kernel(now=True)

    def _new_manager(self) -> _KernelManager:
        if self._manager_factory is not None:
            return self._manager_factory(self._kernel_name)
        try:
            module = importlib.import_module("jupyter_client")
            manager_type = module.AsyncKernelManager
        except (ImportError, AttributeError):
            raise JupyterDependencyUnavailable(
                "install the 'jupyter' extra to run a local notebook kernel"
            ) from None
        return cast(_KernelManager, manager_type(kernel_name=self._kernel_name))


__all__ = ["LocalJupyterClientExecutor", "LocalKernelHandle", "ManagerFactory"]
