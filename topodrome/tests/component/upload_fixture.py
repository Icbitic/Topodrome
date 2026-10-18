"""Loopback authoritative API without a network or an extra payload copy."""

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from topodrome.scheduler.bulk_upload import bulk_upload_router
from topodrome.storage.bulk_upload import SchedulerBulkUploadClient


def upload_client(root):
    app = FastAPI()
    app.include_router(
        bulk_upload_router(root, lambda site: "enrolled" if site == "site" else None)
    )

    class Transport(httpx.BaseTransport):
        def __init__(self):
            self.client = TestClient(app)
            self.client.__enter__()

        def handle_request(self, request):
            response = self.client.request(
                request.method,
                str(request.url),
                headers=dict(request.headers),
                content=request.read(),
            )
            return httpx.Response(
                response.status_code, headers=response.headers, content=response.content
            )

        def close(self):
            self.client.__exit__(None, None, None)

    return SchedulerBulkUploadClient("http://testserver", "site", "enrolled", transport=Transport())
