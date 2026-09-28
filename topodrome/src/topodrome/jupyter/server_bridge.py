"""Jupyter Server gateway managers that authenticate native kernel channels.

Jupyter Server 2.x applies GatewayClient connection headers to REST requests,
but its GatewayKernelClient does not forward them when it opens the native
WebSocket. Topodrome configures this narrow subclass for managed IDE servers so
the generation-scoped bridge capability is present on both transports.
"""

from __future__ import annotations

from threading import Thread

import websocket
from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_server.gateway.gateway_client import GatewayClient
from jupyter_server.gateway.managers import (
    GatewayKernelClient,
    GatewayKernelManager,
    GatewayMappingKernelManager,
    GatewaySessionManager,
)
from jupyter_server.services.sessions.sessionmanager import KernelSessionRecord
from jupyter_server.utils import url_path_join
from tornado.escape import url_escape
from tornado.web import HTTPError
from traitlets import DottedObjectName, Type, default


class AuthenticatedGatewayKernelClient(GatewayKernelClient):
    """Forward the configured GatewayClient authorization to its WebSocket."""

    async def start_channels(  # type: ignore[override]
        self,
        shell: bool = True,
        iopub: bool = True,
        stdin: bool = True,
        hb: bool = True,
        control: bool = True,
    ) -> None:
        client = GatewayClient.instance()
        ws_url = url_path_join(
            client.ws_url or "",
            client.kernels_endpoint,
            url_escape(self.kernel_id),
            "channels",
        )
        connection_args = client.load_connection_args()  # type: ignore[no-untyped-call]
        configured_headers = connection_args.get("headers", {})
        if not isinstance(configured_headers, dict):
            raise RuntimeError("Jupyter gateway headers must be a mapping")
        headers = [f"{key}: {value}" for key, value in configured_headers.items()]
        ssl_options = {
            "ca_certs": client.ca_certs,
            "certfile": client.client_cert,
            "keyfile": client.client_key,
        }
        self.channel_socket = websocket.create_connection(
            ws_url,
            timeout=client.KERNEL_LAUNCH_TIMEOUT,
            enable_multithread=True,
            sslopt=ssl_options,
            header=headers,
        )
        # Bypass GatewayKernelClient.start_channels: that upstream method opens
        # a second WebSocket without its configured Authorization header.
        AsyncKernelClient.start_channels(
            self,
            shell=shell,
            iopub=iopub,
            stdin=stdin,
            hb=hb,
            control=control,
        )
        self.response_router = Thread(target=self._route_responses)
        self.response_router.start()


class AuthenticatedGatewayKernelManager(GatewayKernelManager):
    """Construct the authenticated Topodrome gateway kernel client."""

    client_class = DottedObjectName(
        "topodrome.jupyter.server_bridge.AuthenticatedGatewayKernelClient"
    )
    client_factory = Type(klass=AuthenticatedGatewayKernelClient)


class AuthenticatedGatewayMappingKernelManager(GatewayMappingKernelManager):
    """Use Topodrome's authenticated manager for every remote kernel."""

    @default("kernel_manager_class")
    def _default_kernel_manager_class(self) -> str:
        return "topodrome.jupyter.server_bridge.AuthenticatedGatewayKernelManager"


class ManagedGatewaySessionManager(GatewaySessionManager):
    """Detach IDE sessions without stopping their Topodrome-owned kernel.

    Stock Jupyter treats deleting its last notebook session as authority to
    shut down the associated gateway kernel.  A managed Topodrome kernel has a
    longer, explicit lifecycle: direct kernel stop, managed-server stop, and
    idle expiry are the operations allowed to release its lease.  Session
    deletion therefore removes only Jupyter Server's local session record.
    """

    async def delete_session(self, session_id: str) -> None:
        """Delete a stock session record without issuing gateway DELETE."""
        record = KernelSessionRecord(session_id=session_id)
        self._pending_sessions.update(record)
        try:
            row = self.cursor.execute(
                "SELECT 1 FROM session WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise HTTPError(404, f"Session not found: session_id={session_id!r}")
            self.cursor.execute(
                "DELETE FROM session WHERE session_id=?", (session_id,)
            )
        finally:
            self._pending_sessions.remove(record)


__all__ = [
    "AuthenticatedGatewayKernelClient",
    "AuthenticatedGatewayKernelManager",
    "AuthenticatedGatewayMappingKernelManager",
    "ManagedGatewaySessionManager",
]
