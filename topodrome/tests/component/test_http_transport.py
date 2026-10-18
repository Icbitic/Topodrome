from __future__ import annotations

import hashlib
import io
import os
import threading

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from topodrome.scheduler.bulk_upload import bulk_upload_router
from topodrome.scheduler.file_transport import file_transport_router
from topodrome.scheduler.small_files import small_file_router
from topodrome.scheduler.storage_digest import digest_router
from topodrome.storage.base import StorageVerificationError
from topodrome.storage.bulk import BulkTransportError
from topodrome.storage.http_mirror import HttpMirrorStorage
from topodrome.storage.http_transport import SchedulerFileTransport
from topodrome.storage.progress import TransferCancelled, cancellable_transfers, observe_transfers
from topodrome.storage.small_files import SchedulerSmallFileClient

pytestmark = pytest.mark.component
BASE = "/api/v1/agents/storage/"
AUTH = {"Authorization": "Bearer synthetic"}


@pytest.mark.parametrize("direction", ["download", "upload"])
def test_renewed_bulk_route_preserves_verified_resume_and_durable_ack(api, tmp_path, direction):
    from topodrome.config import BulkReconnectSettings
    from topodrome.deployment.ssh_tunnel import SSHLocalForwardConfig
    from topodrome.storage.bulk_connections import AdaptiveBulkConnections
    from topodrome.storage.link_probe import LinkRates

    client, root = api
    now, switched, extents, ranges = [0.0], [], [], []
    data = os.urandom(2 * 1024**2 + 123)
    logical = "jobs/synthetic/attempts/test/committed/renewed"

    class Forward:
        def __init__(self, _config):
            pass

        def ensure_running(self):
            pass

        def stop(self):
            pass

    def renew():
        now[0] = 5
        assert not route.maintain()
        now[0] = 35
        assert route.maintain()
        switched.append(True)

    def fault(request, result):
        if direction == "upload" and request.method == "PUT":
            extents.append(
                (request.url.port, int(request.url.params["offset"]), len(request.content))
            )
            if not switched:
                renew()
                raise httpx.ReadError("lost ACK after durable write and route renewal")
        if direction == "download" and request.url.path.endswith("/file"):
            ranges.append((request.url.port, request.headers["range"]))
            if not switched:
                # Do not interrupt the active body to switch generations; this
                # independent fault then exercises its normal verified resume.
                renew()
                return httpx.Response(
                    206, headers=result.headers, stream=BrokenBody(result.content[: 256 * 1024])
                )

    transport = bridge(client, fault)
    route = AdaptiveBulkConnections(
        SSHLocalForwardConfig(
            identity_file=tmp_path / "key",
            known_hosts_file=tmp_path / "hosts",
            remote_host="lab1.example",
            remote_user="site",
            local_port=18043,
            target_port=8042,
            state_directory=tmp_path / "bulk",
        ),
        "http://127.0.0.1:18043",
        "site",
        "synthetic",
        BulkReconnectSettings(enabled=True, candidate_port=18044),
        clock=lambda: now[0],
        forward_factory=Forward,
        client_factory=lambda **kwargs: httpx.Client(**kwargs, transport=transport),
        probe=lambda _c, origin, _s, **kw: (
            LinkRates(100_000, 100_000)
            if httpx.URL(origin).port == 18043
            else LinkRates(2_000_000, 2_000_000)
        ),
    )
    route.ensure_running()
    bulk = SchedulerFileTransport(
        "http://127.0.0.1",
        "site",
        "synthetic",
        bulk_url="http://127.0.0.1:18043",
        bulk_connections=route,
        transport=transport,
        min_free_space_gb=0.001,
    )
    storage = HttpMirrorStorage(
        tmp_path / "cache",
        "same-authority",
        authoritative_reader=bulk,
        bulk_transport=bulk,
        min_free_space_gb=0.001,
    )
    try:
        if direction == "download":
            put_file(root, logical, data)
            assert (
                storage.resolve_pinned(
                    logical,
                    expected_size_bytes=len(data),
                    expected_sha256=hashlib.sha256(data).hexdigest(),
                ).read_bytes()
                == data
            )
            assert ranges == [
                (18043, f"bytes=0-{len(data) - 1}"),
                (18044, f"bytes=262144-{len(data) - 1}"),
            ]
        else:
            put_file(storage.root, logical, data)
            assert (
                storage.flush_and_verify(logical).digest.sha256 == hashlib.sha256(data).hexdigest()
            )
            assert (root / logical).read_bytes() == data
            assert extents[0][0] == 18043 and extents[1][0] == 18044
            assert extents[1][1] == extents[0][2]
            assert sum(size for _, _, size in extents) == len(data)
        assert switched == [True]
        assert not route._retired
    finally:
        storage.close()


def test_link_probes_are_enrolled_bounded_and_do_not_touch_storage(api):
    from topodrome.storage.link_probe import LINK_PROBE_BYTES, probe_link

    client, root = api
    endpoint = BASE + "link-probe"
    params = {"site_id": "site"}
    before = sorted(root.rglob("*"))
    assert client.get(endpoint, params=params).status_code == 403
    assert client.get(endpoint, params={"site_id": "foreign"}, headers=AUTH).status_code == 403
    assert client.get(endpoint, params={**params, "size": 1}, headers=AUTH).status_code == 422
    assert client.post(endpoint, params=params, headers=AUTH, content=b"short").status_code == 413
    assert (
        client.post(
            endpoint,
            params=params,
            headers={**AUTH, "Content-Encoding": "gzip"},
            content=os.urandom(LINK_PROBE_BYTES),
        ).status_code
        == 415
    )
    with httpx.Client(transport=bridge(client), headers=AUTH) as local:
        rates = probe_link(local, "http://testserver/api/v1/agents/storage", "site")
    assert rates.download > 0 and rates.upload > 0
    assert sorted(root.rglob("*")) == before


@pytest.fixture
def api(tmp_path):
    root = tmp_path / "authority"
    root.mkdir()
    (root / "storage-layout.json").write_text('{"layout":"blobs-v1"}')
    app = FastAPI()

    def enrollment(site):
        return "synthetic" if site == "site" else None

    for factory in (file_transport_router, bulk_upload_router, small_file_router, digest_router):
        app.include_router(factory(root, enrollment))
    with TestClient(app) as client:
        yield client, root


class BrokenBody(httpx.SyncByteStream):
    def __init__(self, prefix):
        self.prefix = prefix

    def __iter__(self):
        yield self.prefix
        raise httpx.ReadError("synthetic disconnected stream")


def bridge(client, fault=None):
    def handle(request):
        result = client.request(
            request.method, str(request.url), headers=request.headers, content=request.read()
        )
        if fault is not None:
            response = fault(request, result)
            if response is not None:
                return response
        return httpx.Response(
            result.status_code, headers=result.headers, stream=httpx.ByteStream(result.content)
        )

    return httpx.MockTransport(handle)


def mirror(tmp_path, client, fault=None, *, small=False):
    transport = bridge(client, fault)
    short = (
        SchedulerSmallFileClient("http://127.0.0.1", "site", "synthetic", transport=transport)
        if small
        else None
    )
    bulk = SchedulerFileTransport(
        "http://127.0.0.1",
        "site",
        "synthetic",
        min_free_space_gb=0.001,
        small_files=short,
        transport=transport,
    )
    return HttpMirrorStorage(
        tmp_path / "cache",
        "same-authority",
        authoritative_reader=bulk,
        bulk_transport=bulk,
        small_file_client=short,
        min_free_space_gb=0.001,
    )


def put_file(root, logical, content):
    path = root / logical
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_retry_workspace_uploads_log_and_checkpoint_outside_downloaded_tree(api, tmp_path):
    from types import SimpleNamespace

    from topodrome.site_agent.runtime_backend import RuntimeExecutionBackend
    from topodrome.site_agent.workspace import WorkspaceManager

    client, root = api
    old_log = "jobs/job-retry/attempts/attempt-old/logs/attempt.log"
    put_file(root, old_log, b"previous attempt diagnostic")
    storage = mirror(tmp_path, client)
    try:
        downloaded = storage.resolve("jobs/job-retry")
        workspace = WorkspaceManager(storage=storage, scratch_root=tmp_path / "scratch").prepare(
            "job-retry", "attempt-new"
        )
        assert workspace.job_root != downloaded
        payload = b"workload complete; preserve all diagnostics\n"
        (workspace.log_dir / "attempt.log").write_bytes(payload)
        checkpoint = workspace.checkpoint_dir / "00000001-recovery.pkl"
        checkpoint.write_bytes(b"resumable checkpoint")
        assignment = SimpleNamespace(
            job_id="job-retry", attempt_id="attempt-new",
            raw_spec={"entrypoint": "run.py", "model": "test/model"},
        )
        backend = SimpleNamespace(storage=storage)
        descriptor = RuntimeExecutionBackend._verify_log(backend, assignment, workspace)
        assert (root / descriptor["logical_path"]).read_bytes() == payload
        assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
        reference = RuntimeExecutionBackend._verify_checkpoints(backend, assignment, workspace)
        assert (root / reference.split("#sha256=")[0]).read_bytes() == checkpoint.read_bytes()
        assert (root / old_log).read_bytes() == b"previous attempt diagnostic"
        assert (
            downloaded / "attempts/attempt-old/logs/attempt.log"
        ).read_bytes() == b"previous attempt diagnostic"
        assert not (downloaded / "attempts/attempt-new").exists()
    finally:
        storage.close()


def test_range_identity_and_rejections_release_stream_slots(api):
    client, root = api
    put_file(root, "artifacts/input/data", b"abcdefgh")
    params = {"site_id": "site", "logical_path": "artifacts/input/data"}
    info = client.get(BASE + "stat", params=params, headers=AUTH).json()
    etag = f'"{info["ID"]}"'
    response = client.get(
        BASE + "file", params=params, headers={**AUTH, "Range": "bytes=3-6", "If-Match": etag}
    )
    assert response.status_code == 206 and response.content == b"defg"
    assert response.headers["content-range"] == "bytes 3-6/8"
    for headers, status in [
        ({"If-Match": '"changed"'}, 412),
        ({"Range": "bytes=9-"}, 416),
        ({"Range": "bytes=0-1,3-4"}, 416),
    ]:
        assert (
            client.get(BASE + "file", params=params, headers={**AUTH, **headers}).status_code
            == status
        )
    for _ in range(5):
        assert (
            client.get(
                BASE + "file", params={**params, "logical_path": "artifacts/missing"}, headers=AUTH
            ).status_code
            == 404
        )
    assert client.get(BASE + "file", params=params, headers=AUTH).content == b"abcdefgh"
    assert client.get(BASE + "file", params=params).status_code == 403


@pytest.mark.parametrize("kind", ["symlink", "parent", "fifo", "namespace", "traversal"])
def test_http_reads_never_escape_or_follow_non_regular_objects(api, kind):
    client, root = api
    source = put_file(root, "artifacts/private/data", b"private")
    path = "artifacts/private/data"
    if kind == "symlink":
        source.unlink()
        source.symlink_to(root / "storage-layout.json")
    elif kind == "parent":
        (root / "artifacts" / "alias").symlink_to(source.parent)
        path = "artifacts/alias/data"
    elif kind == "fifo":
        source.unlink()
        os.mkfifo(source)
    elif kind == "namespace":
        path = "private-key"
    else:
        path = "artifacts/../storage-layout.json"
    response = client.get(
        BASE + "file", params={"site_id": "site", "logical_path": path}, headers=AUTH
    )
    assert response.status_code in {409, 422}


def test_http_large_download_resumes_disconnection_then_verifies(api, tmp_path):
    client, root = api
    data = os.urandom(3 * 1024**2 + 123)
    logical = "artifacts/input/large"
    put_file(root, logical, data)
    ranges = []

    def fault(request, result):
        if request.url.path.endswith("/file"):
            ranges.append(request.headers["range"])
            if len(ranges) == 1:
                return httpx.Response(
                    206, headers=result.headers, stream=BrokenBody(result.content[: 256 * 1024])
                )

    storage = mirror(tmp_path, client, fault)
    try:
        result = storage.resolve_pinned(
            logical, expected_size_bytes=len(data), expected_sha256=hashlib.sha256(data).hexdigest()
        )
        assert result.read_bytes() == data
        assert ranges == [f"bytes=0-{len(data) - 1}", f"bytes=262144-{len(data) - 1}"]
    finally:
        storage.close()


def test_canceled_http_download_survives_a_new_adapter_process(api, tmp_path):
    client, root = api
    data = os.urandom(2 * 1024**2)
    logical = "artifacts/input/resume"
    put_file(root, logical, data)
    ranges = []

    def fault(request, result):
        if request.url.path.endswith("/file"):
            ranges.append(request.headers["range"])
            if len(ranges) == 1:

                class Chunks(httpx.SyncByteStream):
                    def __iter__(self):
                        yield result.content[: 256 * 1024]
                        yield result.content[256 * 1024 :]

                return httpx.Response(206, headers=result.headers, stream=Chunks())

    canceled = threading.Event()
    storage = mirror(tmp_path, client, fault)
    try:
        with (
            cancellable_transfers(canceled),
            observe_transfers(
                lambda stage, current, total: canceled.set() if current >= 256 * 1024 else None
            ),
        ):
            with pytest.raises(TransferCancelled):
                storage.resolve(logical)
        assert not (storage.root / logical).exists()
    finally:
        storage.close()
    storage = mirror(tmp_path, client, fault)
    try:
        assert storage.resolve(logical).read_bytes() == data
        assert ranges[1].startswith("bytes=262144-")
    finally:
        storage.close()


def test_corrupt_pinned_http_stream_is_not_published(api, tmp_path):
    client, root = api
    data = os.urandom(2 * 1024**2)
    logical = "artifacts/input/corrupt"
    put_file(root, logical, data)

    def fault(request, result):
        if request.url.path.endswith("/file"):
            return httpx.Response(
                206, headers=result.headers, stream=httpx.ByteStream(b"x" * len(data))
            )

    storage = mirror(tmp_path, client, fault)
    try:
        with pytest.raises(StorageVerificationError):
            storage.resolve_pinned(
                logical,
                expected_size_bytes=len(data),
                expected_sha256=hashlib.sha256(data).hexdigest(),
            )
        assert not (storage.root / logical).exists()
        assert not list(storage._partial_root.glob("*.part"))
    finally:
        storage.close()


def test_http_upload_recovers_lost_chunk_and_promotion_ack(api, tmp_path):
    client, root = api
    data = os.urandom(10 * 1024**2 + 123)
    logical = "jobs/synthetic/attempts/test/committed/output"
    lost, extents = set(), []

    def fault(request, result):
        if request.url.path.endswith("/upload-prefix"):
            if request.method == "PUT":
                extents.append((int(request.url.params["offset"]), len(request.content)))
            if request.method in {"PUT", "POST"} and request.method not in lost:
                lost.add(request.method)
                raise httpx.ReadError("synthetic lost durable acknowledgement")

    storage = mirror(tmp_path, client, fault)
    try:
        source = put_file(storage.root, logical, data)
        verified = storage.flush_and_verify(logical)
        assert verified.digest.sha256 == hashlib.sha256(data).hexdigest()
        assert (root / logical).read_bytes() == source.read_bytes() == data
        assert sum(size for _, size in extents) == len(data)
        assert extents[0][0] == 0 and extents[1][0] == extents[0][1]
        assert not list(storage._upload_root.glob("*.json"))
    finally:
        storage.close()


def test_paged_tree_is_atomic_and_tiny_requests_are_one_get(api, tmp_path):
    client, root = api
    for index in range(260):
        put_file(root, f"artifacts/tree/{index:04d}.txt", str(index).encode())
    (root / "artifacts/tree/empty").mkdir()
    requests = []

    def record(request, result):
        requests.append(request.url.path)

    storage = mirror(tmp_path, client, record, small=True)
    try:
        tiny = storage.resolve("artifacts/tree/0001.txt")
        assert tiny.read_bytes() == b"1"
        assert requests == [BASE + "small-file"]
        requests.clear()
        result = storage.resolve("artifacts/tree")
        assert len(list(result.glob("*.txt"))) == 260
        assert (result / "empty").is_dir()
        assert requests.count(BASE + "list") == 3
    finally:
        storage.close()


def test_http_client_rejects_ignored_or_incorrect_ranges():
    metadata = {"IsDir": False, "Size": 4, "ID": "a" * 64}

    def handler(request):
        if request.url.path.endswith("/stat"):
            return httpx.Response(200, json=metadata)
        return httpx.Response(200, content=b"data")

    client = SchedulerFileTransport(
        "http://127.0.0.1", "site", "synthetic", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(StorageVerificationError):
            client.read("artifacts/input/data", io.BytesIO(), 0, 4)
    finally:
        client.close()


def test_tree_partial_names_cannot_collide_with_input_members(api, tmp_path):
    client, root = api
    data = {"x": b"payload", ".x.http-part": b"real hidden input"}
    for name, content in data.items():
        put_file(root, "artifacts/collision/" + name, content)
    storage = mirror(tmp_path, client)
    try:
        result = storage.resolve("artifacts/collision")
        assert {member.name: member.read_bytes() for member in result.iterdir()} == data
        assert not list(result.parent.glob("*.http-parts"))
    finally:
        storage.close()


def test_authority_backpressure_does_not_exhaust_the_network_retry_budget(api, tmp_path):
    client, root = api
    logical = "artifacts/input/busy"
    data = b"safe input"
    put_file(root, logical, data)
    requests = []

    def fault(request, result):
        if request.url.path.endswith("/small-file"):
            requests.append(request)
            if len(requests) <= 4:
                return httpx.Response(503, headers={"Retry-After": "0.1"})

    storage = mirror(tmp_path, client, fault, small=True)
    try:
        assert storage.resolve(logical).read_bytes() == data
        assert len(requests) == 5
    finally:
        storage.close()


def test_authority_backpressure_wait_is_bounded_and_cancellable(api, tmp_path):
    client, _ = api
    storage = mirror(tmp_path, client)
    clock = [0.0]
    storage._monotonic = lambda: clock[0]
    storage._sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    requests = []

    def busy():
        requests.append(clock[0])
        raise BulkTransportError("HTTPTransient", retryable=True, retry_after_seconds=5)

    try:
        with pytest.raises(BulkTransportError):
            storage._retry_bulk(busy)
        assert requests[-1] == pytest.approx(120)
        canceled = threading.Event()
        storage._sleep = lambda _seconds: canceled.set()
        with cancellable_transfers(canceled), pytest.raises(TransferCancelled):
            storage._retry_bulk(busy)
    finally:
        storage.close()
