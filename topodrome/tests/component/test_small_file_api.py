from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import threading
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from topodrome.config import SchedulerSettings
from topodrome.db.models import AttemptRow, JobRow, LeaseRow, SiteRow
from topodrome.domain.enums import AttemptState, JobState, LeaseState, SiteKind, SiteState
from topodrome.domain.specs import JobSpec
from topodrome.scheduler.app import create_app
from topodrome.scheduler.source_store import SourceBundle, SourceStore
from topodrome.storage.blobs import blob_path, blob_transaction
from topodrome.storage.source_archive import unpack_source_bundle

pytestmark = pytest.mark.component
URL = "/api/v1/agents/storage/small-file"
AUTH = {"Authorization": "Bearer test-site"}
PAYLOAD = b"immediately usable result"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
async def api(database, tmp_path):
    app = create_app(
        SchedulerSettings(
            global_root=tmp_path / "authority",
            storage_layout="blobs-v1",
            fence_signing_key="f" * 32,
            agent_tokens={"site": "test-site", "other": "test-other"},
        ),
        database=database,
        run_dispatcher=False,
    )
    async with database.transaction() as session:
        session.add(
            SiteRow(id="site", display_name="site", kind=SiteKind.LAB, state=SiteState.READY)
        )
        session.add(
            JobRow(
                id="job",
                state=JobState.RUNNING,
                spec=JobSpec(entrypoint="run.py", model="test/model").model_dump(mode="json"),
            )
        )
        session.add(
            AttemptRow(
                id="attempt", job_id="job", site_id="site", number=1, state=AttemptState.COMMITTING
            )
        )
        session.add(
            LeaseRow(
                id="lease",
                attempt_id="attempt",
                site_id="site",
                state=LeaseState.RELEASING,
                fencing_token_digest="x" * 64,
                expires_at=datetime.now(UTC) + timedelta(minutes=5),
            )
        )
    root = app.state.settings.global_root
    root.mkdir(exist_ok=True)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, root, app


async def put(c, *, content=PAYLOAD, path=None, **kwargs):
    headers = {**AUTH, "X-Topodrome-Sha256": DIGEST, "Content-Length": str(len(PAYLOAD))}
    headers.update(kwargs.pop("headers", {}))
    params = {
        "site_id": "site",
        "job_id": "job",
        "attempt_id": "attempt",
        "logical_path": path or blob_path(DIGEST),
    }
    params.update(kwargs.pop("params", {}))
    return await c.put(URL, params=params, headers=headers, content=content, **kwargs)


async def test_source_archive_uses_one_bounded_enrolled_request(api, tmp_path):
    c, root, _ = api
    result = SourceStore(root).put(
        SourceBundle(
            files={
                "run.py": base64.b64encode(b"print('hi')").decode(),
                "lib/empty": "",
            }
        )
    )
    path = "projects/.topodrome-sources/" + result.source_ref.removeprefix("sha256:")
    response = await c.get(
        "/api/v1/agents/storage/source-bundle",
        params={"site_id": "site", "logical_path": path},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["etag"] == f'"{hashlib.sha256(response.content).hexdigest()}"'
    destination = tmp_path / "source-cache"
    destination.mkdir()
    unpack_source_bundle(path, response.content, destination, 1024**2)
    assert (destination / "run.py").read_bytes() == b"print('hi')"


@pytest.mark.parametrize("case", ["auth", "namespace", "missing", "oversize", "symlink", "corrupt"])
async def test_source_archive_invalid_cases_fail_closed(api, case):
    c, root, _ = api
    store = SourceStore(root)
    result = store.put(SourceBundle(files={"run.py": base64.b64encode(b"code").decode()}))
    path = "projects/.topodrome-sources/" + result.source_ref.removeprefix("sha256:")
    params = {"site_id": "site", "logical_path": path}
    headers = AUTH
    expected = 409
    if case == "auth":
        headers, expected = {"Authorization": "Bearer wrong"}, 403
    elif case == "namespace":
        params["logical_path"], expected = "jobs/job", 422
    elif case == "missing":
        params["logical_path"], expected = "projects/.topodrome-sources/" + "f" * 64, 404
    elif case == "oversize":
        params["max_bytes"], expected = 1, 413
    elif case == "symlink":
        member = root / path / "run.py"
        member.unlink()
        member.symlink_to(root / path / ".topodrome/manifest.json")
    else:
        (root / path / "run.py").write_bytes(b"altered")
    response = await c.get("/api/v1/agents/storage/source-bundle", params=params, headers=headers)
    assert response.status_code == expected


async def test_one_put_claims_verifies_and_durably_publishes(api):
    c, root, _ = api
    r = await put(c)
    assert r.status_code == 200
    assert r.json() == {"size_bytes": len(PAYLOAD), "sha256": DIGEST}
    assert r.headers["cache-control"] == "no-store"
    path = root / blob_path(DIGEST)
    assert path.read_bytes() == PAYLOAD
    inode = path.stat().st_ino
    assert (await put(c)).status_code == 200
    assert path.stat().st_ino == inode
    assert not list(path.parent.glob("*.topodrome-small-*"))
    with blob_transaction(root) as db:
        assert db.execute("SELECT job_id,attempt_id,sha256 FROM claims").fetchall() == [
            ("job", "attempt", DIGEST),
        ]
    r = await c.get(
        URL,
        params={"site_id": "site", "logical_path": blob_path(DIGEST)},
        headers={**AUTH, "If-Match": f'"{DIGEST}"'},
    )
    assert r.status_code == 200 and r.content == PAYLOAD
    assert r.headers["etag"] == f'"{DIGEST}"'


@pytest.mark.parametrize("case", ["auth", "other-site", "other-job", "other-attempt", "path"])
async def test_upload_authorization_cannot_cross_attempts(api, case):
    c, root, _ = api
    kwargs = {}
    if case == "auth":
        kwargs["headers"] = {"Authorization": "Bearer invalid"}
    if case == "other-site":
        kwargs.update(params={"site_id": "other"}, headers={"Authorization": "Bearer test-other"})
    if case == "other-job":
        kwargs["params"] = {"job_id": "other"}
    if case == "other-attempt":
        kwargs["params"] = {"attempt_id": "other"}
    if case == "path":
        kwargs["path"] = "jobs/other/attempts/attempt/committed/result"
    assert (await put(c, **kwargs)).status_code in {403, 409}
    assert not (root / blob_path(DIGEST)).exists()


@pytest.mark.parametrize("case", ["terminal", "cancel", "expired", "released"])
async def test_live_authority_required_and_rechecked(api, database, case):
    c, root, _ = api
    async with database.transaction() as session:
        job = await session.get(JobRow, "job")
        lease = await session.get(LeaseRow, "lease")
        if case == "terminal":
            job.state = JobState.CANCELED
        elif case == "cancel":
            job.cancel_requested = True
        elif case == "expired":
            lease.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        else:
            lease.state = LeaseState.RELEASED
    assert (await put(c)).status_code == 409
    assert not (root / blob_path(DIGEST)).exists()


async def test_cancel_during_body_reception_prevents_publication(api, database):
    c, root, _ = api

    async def body():
        async with database.transaction() as session:
            job = await session.get(JobRow, "job")
            job.cancel_requested = True
        yield PAYLOAD

    assert (await put(c, content=body())).status_code == 409
    assert not (root / blob_path(DIGEST)).exists()


async def test_failed_job_log_outbox_can_complete_without_a_lease(api, database):
    c, root, _ = api
    async with database.transaction() as session:
        job = await session.get(JobRow, "job")
        attempt = await session.get(AttemptRow, "attempt")
        lease = await session.get(LeaseRow, "lease")
        job.state, attempt.state, lease.state = (
            JobState.FAILED,
            AttemptState.FAILED,
            LeaseState.RELEASED,
        )
    path = "jobs/job/attempts/attempt/logs/attempt.log"
    assert (await put(c, path=path)).status_code == 200
    assert (root / path).read_bytes() == PAYLOAD


@pytest.mark.parametrize(
    "case", ["hash", "short", "extra", "oversized", "symlink", "corrupt-existing"]
)
async def test_invalid_uploads_never_receive_success(api, case):
    c, root, _ = api
    kwargs = {}
    if case == "hash":
        kwargs["content"] = b"x" * len(PAYLOAD)
    elif case == "short":
        kwargs["content"] = PAYLOAD[:-1]
    elif case == "extra":
        kwargs["content"] = PAYLOAD + b"x"
    elif case == "oversized":
        kwargs["headers"] = {"Content-Length": str(4 * 1024**2 + 1)}
    elif case == "symlink":
        (root / "blobs").symlink_to(root, target_is_directory=True)
    else:
        path = root / blob_path(DIGEST)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"bad previous upload")
    assert (await put(c, **kwargs)).status_code in {400, 409, 413}
    if case not in {"symlink", "corrupt-existing"}:
        assert not (root / blob_path(DIGEST)).exists()


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/etc/passwd",
        ".blob-index/index.sqlite3",
        "jobs/link",
        "jobs/fifo",
        "jobs/dirlink/file",
    ],
)
async def test_read_is_safe_and_bounded(api, path):
    c, root, _ = api
    (root / "jobs").mkdir(exist_ok=True)
    (root / "jobs/link").symlink_to("/etc/passwd")
    (root / "jobs/dirlink").symlink_to(root)
    os.mkfifo(root / "jobs/fifo")
    r = await c.get(URL, params={"site_id": "site", "logical_path": path}, headers=AUTH)
    assert r.status_code in {409, 422}


async def test_large_files_and_directories_select_bulk_but_bad_etag_does_not(api):
    c, root, _ = api
    (root / "jobs").mkdir(exist_ok=True)
    (root / "jobs/result").write_bytes(PAYLOAD)
    for path, extra in [("jobs", {}), ("jobs/result", {"max_bytes": 1})]:
        r = await c.get(
            URL, params={"site_id": "site", "logical_path": path, **extra}, headers=AUTH
        )
        assert r.status_code == 413
    r = await c.get(
        URL,
        params={"site_id": "site", "logical_path": "jobs/result"},
        headers={**AUTH, "If-Match": '"wrong"'},
    )
    assert r.status_code == 412


async def test_durable_ack_waits_for_authority_verification(api, monkeypatch):
    from topodrome.scheduler import small_files

    c, _, _ = api

    def failed_sync(*_):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(small_files, "hash_beneath", failed_sync)
    assert (await put(c)).status_code == 409


async def test_bulk_hash_slots_cannot_block_small_file_download(api, monkeypatch):
    from topodrome.scheduler import storage_digest

    c, root, _ = api
    (root / "jobs").mkdir(exist_ok=True)
    (root / "jobs/result").write_bytes(PAYLOAD)
    release = threading.Event()
    entered = threading.Event()
    count = 0
    lock = threading.Lock()
    original = storage_digest.hash_beneath

    def blocked(*args):
        nonlocal count
        with lock:
            count += 1
            if count == 2:
                entered.set()
        assert release.wait(3)
        return original(*args)

    monkeypatch.setattr(storage_digest, "hash_beneath", blocked)
    requests = [
        asyncio.create_task(
            c.post(
                "/api/v1/agents/storage/digest",
                headers=AUTH,
                json={"site_id": "site", "logical_path": "jobs/result"},
            )
        )
        for _ in range(2)
    ]
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        r = await asyncio.wait_for(
            c.get(URL, headers=AUTH, params={"site_id": "site", "logical_path": "jobs/result"}), 1
        )
        assert r.status_code == 200 and r.content == PAYLOAD
    finally:
        release.set()
        await asyncio.gather(*requests)


async def test_real_mirror_commit_uses_http_for_payload_and_manifests(api, tmp_path):
    from tests.unit.test_storage_http_mirror import FakeAuthority
    from topodrome.storage import AttemptArtifactStore, HttpMirrorStorage
    from topodrome.storage.small_files import SchedulerSmallFileClient

    c, root, _ = api
    loop = asyncio.get_running_loop()
    requests = []

    def send(request):
        requests.append((request.method, request.url.path))
        result = asyncio.run_coroutine_threadsafe(
            c.request(
                request.method, str(request.url), headers=request.headers, content=request.content
            ),
            loop,
        ).result(timeout=5)
        return httpx.Response(result.status_code, content=result.content, headers=result.headers)

    small = SchedulerSmallFileClient(
        "http://test", "site", "test-site", transport=httpx.MockTransport(send)
    )
    remote = FakeAuthority()
    mirror = HttpMirrorStorage(
        tmp_path / "cache",
        remote.remote_root,
        bulk_transport=remote,
        authoritative_reader=remote,
        small_file_client=small,
        min_free_space_gb=0.001,
    )
    mirror.content_addressed = True
    try:
        store = AttemptArtifactStore(mirror, job_id="job", attempt_id="attempt")
        await asyncio.to_thread(store.stage_bytes, "result", PAYLOAD)
        manifest = await asyncio.to_thread(store.commit, ["result"])
        assert manifest.artifact("result").sha256 == DIGEST
        assert (root / blob_path(DIGEST)).read_bytes() == PAYLOAD
        assert (root / store.published_manifest_path).is_file()
        assert not remote.calls
        assert sum(method == "PUT" for method, _ in requests) == 3
        assert all(path == URL for _, path in requests)
    finally:
        mirror.close()


async def test_cached_job_retry_uploads_canonical_files_with_live_authority(api, tmp_path):
    from types import SimpleNamespace

    from topodrome.site_agent.runtime_backend import RuntimeExecutionBackend
    from topodrome.site_agent.workspace import WorkspaceManager
    from topodrome.storage import HttpMirrorStorage
    from topodrome.storage.http_transport import SchedulerFileTransport
    from topodrome.storage.small_files import SchedulerSmallFileClient

    c, root, _ = api
    (root / "storage-layout.json").write_text('{"layout":"blobs-v1"}')
    old_log = root / "jobs/job/attempts/old/logs/attempt.log"
    old_log.parent.mkdir(parents=True)
    old_log.write_bytes(b"previous failed attempt")
    loop = asyncio.get_running_loop()
    requests = []

    def send(request):
        requests.append((request.method, request.url.path))
        result = asyncio.run_coroutine_threadsafe(
            c.request(
                request.method, str(request.url), headers=request.headers, content=request.content
            ), loop,
        ).result(timeout=5)
        return httpx.Response(result.status_code, content=result.content, headers=result.headers)

    transport = httpx.MockTransport(send)
    small = SchedulerSmallFileClient("http://test", "site", "test-site", transport=transport)
    bulk = SchedulerFileTransport(
        "http://test", "site", "test-site", transport=transport, small_files=small,
        min_free_space_gb=0.001,
    )
    storage = HttpMirrorStorage(
        tmp_path / "cache", "same-authority", authoritative_reader=bulk,
        bulk_transport=bulk, small_file_client=small, min_free_space_gb=0.001,
    )
    try:
        read_view = await asyncio.to_thread(storage.resolve, "jobs/job")
        manager = WorkspaceManager(storage=storage, scratch_root=tmp_path / "scratch")
        workspace = await asyncio.to_thread(manager.prepare, "job", "attempt")
        assert workspace.job_root != read_view
        log = workspace.log_dir / "attempt.log"
        log.write_bytes(b"completed expensive workload\n")
        checkpoint = workspace.checkpoint_dir / "00000001-recovery.pkl"
        checkpoint.write_bytes(b"latest resumable state")
        assignment = SimpleNamespace(
            job_id="job", attempt_id="attempt",
            raw_spec={"entrypoint": "run.py", "model": "test/model"},
        )
        backend = SimpleNamespace(storage=storage)
        descriptor = await asyncio.to_thread(
            RuntimeExecutionBackend._verify_log, backend, assignment, workspace,
        )
        reference = await asyncio.to_thread(
            RuntimeExecutionBackend._verify_checkpoints, backend, assignment, workspace,
        )
        assert (root / descriptor["logical_path"]).read_bytes() == log.read_bytes()
        assert descriptor["sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()
        assert (root / reference.split("#sha256=")[0]).read_bytes() == checkpoint.read_bytes()
        assert old_log.read_bytes() == b"previous failed attempt"
        assert (read_view / "attempts/old/logs/attempt.log").read_bytes() == old_log.read_bytes()
        assert not (read_view / "attempts/attempt").exists()
        assert [path for method, path in requests if method == "PUT"] == [URL, URL]
    finally:
        storage.close()
