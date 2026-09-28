"""Hardened stock Jupyter Server entry point for managed IDE sessions.

The IDE token is supplied to the child through its minimal environment.  Stock
Jupyter Server writes that token into ``jpserver-<pid>.json`` by default, which
would turn an intentionally ephemeral credential into runtime state.  This
small subclass keeps the stock contents, sessions, and kernel APIs unchanged
while redacting that one field from server discovery metadata.
"""

from __future__ import annotations

from typing import Any

from jupyter_server.serverapp import ServerApp
from traitlets import default


class ManagedJupyterServerApp(ServerApp):
    """Stock Jupyter Server with credential-free discovery metadata."""

    name = "topodrome-managed-jupyter"

    @default("terminals_enabled")
    def _terminals_are_disabled(self) -> bool:
        # IDE clients need contents, sessions, kernelspecs, kernels, and native
        # channels.  A scheduler-host shell is outside that surface and would
        # also inherit the child-only gateway capability.
        return False

    @default("session_manager_class")
    def _managed_session_manager_class(self) -> str:
        # Closing a notebook in an IDE deletes its stock Jupyter session.  The
        # associated GPU kernel remains owned by Topodrome's explicit lifecycle
        # rather than being shut down through the gateway as a side effect.
        return "topodrome.jupyter.server_bridge.ManagedGatewaySessionManager"

    @default("no_browser_open_file")
    def _browser_open_file_is_disabled(self) -> bool:
        # Stock Jupyter otherwise writes an HTML redirect containing the IDE
        # token even when automatic browser launch itself is disabled.
        return True

    def _write_cookie_secret_file(self, secret: bytes) -> None:
        # The stock default already generated ``secret`` from fresh OS entropy.
        # Keep it process-local: a surviving child retains its cookies across a
        # scheduler restart, while a replacement intentionally revokes them.
        del secret

    def server_info(self) -> dict[str, Any]:
        info = super().server_info()
        # Keep the key for compatibility with ``jupyter server list`` while
        # ensuring the plaintext IDE bearer is never written to disk.
        info["token"] = ""
        return info


def main() -> None:
    ManagedJupyterServerApp.launch_instance()


__all__ = ["ManagedJupyterServerApp", "main"]


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    main()
