"""Interactive session lifecycle and remote Jupyter integration."""

from topodrome.jupyter.base import KernelLauncher, KernelRuntime, SessionAllocator, SessionStore
from topodrome.jupyter.executor import LocalJupyterClientExecutor, LocalKernelHandle
from topodrome.jupyter.gateway import InMemorySessionStore, KernelGateway, KernelGatewayError
from topodrome.jupyter.relay import (
    BrowserRelayUnavailable,
    RelayHandle,
    ReverseRelay,
    UnavailableReverseRelay,
)
from topodrome.jupyter.session import (
    KernelSession,
    SessionLease,
    SessionLeaseConflict,
    SessionLeaseRegistry,
    SessionStateError,
)
from topodrome.jupyter.terminal import (
    BootstrapResult,
    JupyterDependencyUnavailable,
    JupyterTerminalClient,
    JupyterTerminalError,
    RemoteTerminal,
    TerminalSocket,
    TerminalSocketConnector,
    WebsocketsConnector,
)

__all__ = [
    "BootstrapResult",
    "BrowserRelayUnavailable",
    "InMemorySessionStore",
    "JupyterDependencyUnavailable",
    "JupyterTerminalClient",
    "JupyterTerminalError",
    "KernelGateway",
    "KernelGatewayError",
    "KernelLauncher",
    "KernelRuntime",
    "KernelSession",
    "LocalJupyterClientExecutor",
    "LocalKernelHandle",
    "RelayHandle",
    "RemoteTerminal",
    "ReverseRelay",
    "SessionAllocator",
    "SessionLease",
    "SessionLeaseConflict",
    "SessionLeaseRegistry",
    "SessionStateError",
    "SessionStore",
    "TerminalSocket",
    "TerminalSocketConnector",
    "UnavailableReverseRelay",
    "WebsocketsConnector",
]
