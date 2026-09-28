from __future__ import annotations

import pytest

from topodrome.jupyter.executor import LocalJupyterClientExecutor

pytestmark = pytest.mark.unit


class FakeClient:
    def __init__(self) -> None:
        self.started = False
        self.ready_timeout: float | None = None
        self.stopped = False

    def start_channels(self) -> None:
        self.started = True

    def stop_channels(self) -> None:
        self.stopped = True

    async def wait_for_ready(self, wait_seconds: float) -> None:
        self.ready_timeout = wait_seconds


class FakeManager:
    connection_file = "/tmp/fake-kernel.json"  # noqa: S108

    def __init__(self) -> None:
        self.kernel_name = ""
        self.environment: dict[str, str] | None = None
        self.client_instance = FakeClient()
        self.shutdown = False

    async def start_kernel(self, *, env: dict[str, str] | None = None) -> None:
        self.environment = env

    def client(self) -> FakeClient:
        return self.client_instance

    async def shutdown_kernel(self, *, now: bool = False) -> None:
        self.shutdown = now


async def test_local_executor_uses_injected_manager_without_jupyter_dependency() -> None:
    manager = FakeManager()

    def factory(kernel_name: str) -> FakeManager:
        manager.kernel_name = kernel_name
        return manager

    executor = LocalJupyterClientExecutor(
        manager_factory=factory,
        ready_timeout_seconds=4,
    )

    handle = await executor.start("session-1", environment={"EXPERIMENT": "safe"})

    assert handle.connection_file == "/tmp/fake-kernel.json"  # noqa: S108
    assert manager.kernel_name == "python3"
    assert manager.environment == {"EXPERIMENT": "safe"}
    assert manager.client_instance.started
    assert manager.client_instance.ready_timeout == 4
    await executor.stop("session-1")
    assert manager.client_instance.stopped
    assert manager.shutdown


async def test_local_executor_rejects_credentials_in_kernel_environment() -> None:
    executor = LocalJupyterClientExecutor(manager_factory=lambda _name: FakeManager())

    try:
        await executor.start("session-1", environment={"API_TOKEN": "secret"})
    except ValueError as error:
        assert "credentials" in str(error)
    else:
        raise AssertionError("credential-shaped environment key was accepted")
