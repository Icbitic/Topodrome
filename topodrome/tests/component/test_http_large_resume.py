"""Real HTTP socket acceptance, including >2-GiB files and adapter restart."""

from __future__ import annotations

import hashlib
import os
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI

from topodrome.scheduler.bulk_upload import bulk_upload_router
from topodrome.scheduler.file_transport import file_transport_router
from topodrome.scheduler.storage_digest import digest_router
from topodrome.storage.http_mirror import HttpMirrorStorage
from topodrome.storage.http_transport import SchedulerFileTransport
from topodrome.storage.progress import TransferCancelled, cancellable_transfers, observe_transfers

pytestmark = pytest.mark.component


def file_hash(path: Path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while content := stream.read(4 * 1024**2):
            result.update(content)
    return result.hexdigest()


def test_http_only_two_gib_download_and_upload_survive_adapter_restart(tmp_path):
    authority = tmp_path / "authority"
    authority.mkdir()
    source = authority / "artifacts/input/two-gib.bin"
    source.parent.mkdir(parents=True)
    size = 2 * 1024**3 + 17 * 1024
    with source.open("wb") as stream:
        stream.truncate(size)
        stream.seek(size - 1)
        stream.write(b"x")
    expected = file_hash(source)
    app = FastAPI()

    def enrollment(site):
        return "synthetic" if site == "site" else None

    for factory in (file_transport_router, bulk_upload_router, digest_router):
        app.include_router(factory(authority, enrollment))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    origin = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="critical", lifespan="off", access_log=False)
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started

    def adapter():
        transport = SchedulerFileTransport(origin, "site", "synthetic", min_free_space_gb=0.001)
        return HttpMirrorStorage(
            tmp_path / "cache",
            "http-authority",
            authoritative_reader=transport,
            bulk_transport=transport,
            min_free_space_gb=0.001,
        )

    storage = adapter()
    try:
        interrupted = threading.Event()

        def stop_download(stage, current, total):
            if stage == "downloading" and current >= 256 * 1024:
                interrupted.set()

        with cancellable_transfers(interrupted), observe_transfers(stop_download):
            with pytest.raises(TransferCancelled):
                storage.resolve_pinned(
                    "artifacts/input/two-gib.bin",
                    expected_size_bytes=size,
                    expected_sha256=expected,
                )
        assert not (storage.root / "artifacts/input/two-gib.bin").exists()
        prefix = next(storage._partial_root.glob("*.part"))
        assert 0 < prefix.stat().st_size < size
        storage.close()
        storage = adapter()
        resumed = []
        with observe_transfers(
            lambda stage, current, total: (
                resumed.append(current) if stage == "downloading" and not resumed else None
            )
        ):
            local = storage.resolve_pinned(
                "artifacts/input/two-gib.bin", expected_size_bytes=size, expected_sha256=expected
            )
        assert resumed[0] > 0
        assert local.stat().st_size == size and file_hash(local) == expected
        logical = "jobs/synthetic/attempts/test/committed/two-gib.bin"
        output = storage.root / logical
        output.parent.mkdir(parents=True)
        os.link(local, output)
        interrupted.clear()

        def stop_upload(stage, current, total):
            if stage == "upload" and current >= 128 * 1024:
                interrupted.set()

        with cancellable_transfers(interrupted), observe_transfers(stop_upload):
            with pytest.raises(TransferCancelled):
                storage.flush_and_verify(logical, expected_sha256=expected)
        assert not (authority / logical).exists()
        assert list(storage._upload_root.glob("*.json"))
        storage.close()
        storage = adapter()
        resumed.clear()
        with observe_transfers(
            lambda stage, current, total: (
                resumed.append(current) if stage == "upload" and not resumed else None
            )
        ):
            result = storage.flush_and_verify(logical, expected_sha256=expected)
        assert resumed[0] > 0 and result.digest.size_bytes == size
        assert file_hash(authority / logical) == expected
        assert not list(storage._upload_root.glob("*.json"))
    finally:
        storage.close()
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
    assert not thread.is_alive()
