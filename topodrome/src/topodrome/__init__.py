"""Topodrome public Python API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topodrome.client.client import Client, JobHandle, KernelHandle

__all__ = ["Client", "JobHandle", "KernelHandle", "kernel", "submit"]
__version__ = "0.1.0"


def submit(
    *,
    source_dir: str | Path | None = None,
    idempotency_key: str | None = None,
    **job: Any,
) -> JobHandle:
    """Upload source and submit through a default environment-configured client."""

    client = Client()
    handle = client.submit(
        source_dir=source_dir,
        idempotency_key=idempotency_key,
        **job,
    )
    # The handle deliberately owns the client until the caller finishes using it.
    return handle


def kernel(**request: Any) -> KernelHandle:
    """Create an interactive session through the default scheduler."""
    client = Client()
    return client.kernel(**request)
