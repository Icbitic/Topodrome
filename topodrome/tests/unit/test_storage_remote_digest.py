import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from topodrome.scheduler.storage_digest import digest_router, hash_beneath
from topodrome.storage.base import StorageUnavailableError
from topodrome.storage.remote_digest import SchedulerDigestReader


def test_digest_is_authoritative_and_authenticated(tmp_path: Path):
    (tmp_path / "result").write_bytes(b"stored on lab1")
    app = FastAPI()
    app.include_router(digest_router(tmp_path, lambda site: "secret" if site == "sqz" else None))
    with TestClient(app) as client:
        body = {"site_id": "sqz", "logical_path": "result"}
        url = "/api/v1/agents/storage/digest"
        assert client.post(url, json=body).status_code == 403
        auth = {"Authorization": "Bearer secret"}
        response = client.post(url, json=body, headers=auth)
        assert response.json() == {
            "size_bytes": 14,
            "sha256": hashlib.sha256(b"stored on lab1").hexdigest(),
        }
        assert response.headers["cache-control"] == "no-store"
        (tmp_path / "result").write_bytes(b"changed")
        assert client.post(url, json=body, headers=auth).json()["size_bytes"] == 7
        assert client.post(url, json={**body, "site_id": "other"}, headers=auth).status_code == 403
        assert (
            client.post(url, json={**body, "logical_path": "missing"}, headers=auth).status_code
            == 404
        )


@pytest.mark.parametrize(
    "path", ["../outside", "/etc/passwd", "a/../b", "link", "dirlink/file", "fifo"]
)
def test_digest_rejects_unsafe_objects(tmp_path: Path, path: str):
    (tmp_path / "link").symlink_to("/etc/passwd")
    (tmp_path / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises((OSError, ValueError)):
        hash_beneath(tmp_path, path)


class Healthy:
    def healthcheck(self):
        return False


def test_digest_rejects_replaced_parent(tmp_path: Path, monkeypatch):
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "result").write_bytes(b"original")
    original_fsync = os.fsync
    replaced = False

    def replace_during_sync(fd):
        nonlocal replaced
        if not replaced:
            replaced = True
            folder.rename(tmp_path / "previous")
            folder.mkdir()
            (folder / "result").write_bytes(b"different")
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", replace_during_sync)
    with pytest.raises(ValueError, match="changed"):
        hash_beneath(tmp_path, "folder/result")


def test_reader_uses_small_post_not_payload_download():
    sha = hashlib.sha256(b"x").hexdigest()

    def request(req):
        assert req.method == "POST"
        assert str(req.url) == "https://scheduler/prefix/api/v1/agents/storage/digest"
        assert req.headers["authorization"] == "Bearer secret"
        assert len(req.content) < 200
        return httpx.Response(200, json={"size_bytes": 5 * 1024**3, "sha256": sha})

    reader = SchedulerDigestReader(
        "wss://scheduler/prefix/api/v1/agents/control",
        "sqz",
        "secret",
        health_reader=Healthy(),
        transport=httpx.MockTransport(request),
    )
    assert reader.digest("jobs/result").size_bytes == 5 * 1024**3
    assert reader.healthcheck() is False


@pytest.mark.parametrize(
    "code,payload", [(403, {}), (503, {}), (302, {}), (200, {"size_bytes": True, "sha256": "bad"})]
)
def test_reader_fails_closed(code, payload):
    reader = SchedulerDigestReader(
        "https://scheduler",
        "sqz",
        "secret",
        health_reader=Healthy(),
        transport=httpx.MockTransport(lambda req: httpx.Response(code, json=payload)),
    )
    with pytest.raises(StorageUnavailableError):
        reader.digest("jobs/result")


@pytest.mark.parametrize("corrupt", [False, True])
def test_artifact_commit_verifies_on_authority_without_readback(tmp_path, corrupt):
    from tests.unit.test_storage_http_mirror import FakeAuthority
    from topodrome.storage import (
        AttemptArtifactStore,
        HttpMirrorStorage,
        StorageVerificationError,
    )

    remote = FakeAuthority()
    requests = []
    authority = tmp_path / "lab1"
    authority.mkdir()

    def digest_request(request):
        requests.append(request)
        assert request.method == "POST" and len(request.content) < 512
        logical = json.loads(request.content)["logical_path"]
        if logical not in remote.objects:
            return httpx.Response(404)
        stored = authority / logical
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_bytes(remote.objects[logical] + (b"corrupt" if corrupt else b""))
        return httpx.Response(200, json=hash_beneath(authority, logical).model_dump())

    reader = SchedulerDigestReader(
        "http://scheduler",
        "autodl-manual-" + "a" * 24,
        "test-credential",
        health_reader=remote,
        transport=httpx.MockTransport(digest_request),
    )
    mirror = HttpMirrorStorage(
        tmp_path / "worker",
        remote.remote_root,
        bulk_transport=remote,
        authoritative_reader=reader,
        min_free_space_gb=0.001,
        verification_timeout_seconds=0.01,
        poll_interval_seconds=0.01,
    )
    store = AttemptArtifactStore(mirror, job_id="job_test", attempt_id="attempt_test")
    store.stage_bytes("result.bin", b"payload" * 10000)
    if corrupt:
        with pytest.raises(StorageVerificationError):
            store.commit(["result.bin"])
        assert store.published_manifest_path not in remote.objects
    else:
        manifest = store.commit(["result.bin"])
        assert manifest.artifact("result.bin").size_bytes == 70000
        assert store.published_manifest_path in remote.objects
    assert requests
    assert not remote.reads
