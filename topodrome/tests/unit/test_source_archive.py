from __future__ import annotations

import base64
import hashlib
import io
import json
import stat
import zipfile

import httpx
import pytest

from tests.unit.test_storage_http_mirror import FakeAuthority
from topodrome.scheduler.small_files import read_small_file
from topodrome.scheduler.source_store import SourceBundle, SourceStore
from topodrome.storage.base import StorageUnavailableError, StorageVerificationError
from topodrome.storage.http_mirror import HttpMirrorStorage
from topodrome.storage.small_files import SchedulerSmallFileClient, SmallFileFallback
from topodrome.storage.source_archive import pack_source_bundle, unpack_source_bundle

pytestmark = pytest.mark.unit


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "authority"
    files = {"run.py": b"print('hello')", "helpers/utils.py": b"value=1", "empty": b""}
    result = SourceStore(root).put(
        SourceBundle(
            files={name: base64.b64encode(body).decode() for name, body in files.items()},
        )
    )
    path = "projects/.topodrome-sources/" + result.source_ref.removeprefix("sha256:")
    content = pack_source_bundle(
        path, 1024**2, lambda member, bound: read_small_file(root, member, bound)[0]
    )
    return root, path, content, files


def test_standard_stored_zip_preserves_existing_source_identity(bundle, tmp_path):
    root, path, content, files = bundle
    destination = tmp_path / "staging"
    destination.mkdir()
    unpack_source_bundle(path, content, destination, 1024**2)
    assert {name: (destination / name).read_bytes() for name in files} == files
    assert json.loads((destination / ".topodrome/manifest.json").read_bytes())["source_ref"] == (
        "sha256:" + path.rsplit("/", 1)[-1]
    )
    assert SourceStore(root).resolve("sha256:" + path.rsplit("/", 1)[-1]).is_dir()


def test_oversized_member_downgrades_without_reading_large_bytes(bundle):
    root, path, _, _ = bundle
    bounds = []

    def read(member, bound):
        bounds.append(bound)
        return read_small_file(root, member, bound)[0]

    with pytest.raises(OverflowError):
        pack_source_bundle(path, 300, read)
    assert max(bounds) == 300


@pytest.mark.parametrize(
    "case",
    [
        "content",
        "extra",
        "duplicate",
        "symlink",
        "compressed",
        "truncated",
        "wrong-source",
        "overflow",
    ],
)
def test_invalid_archive_cannot_expose_any_cache_files(bundle, tmp_path, case):
    _, path, content, _ = bundle
    original = zipfile.ZipFile(io.BytesIO(content))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for entry in original.infolist():
            body = original.read(entry)
            if entry.filename == "run.py":
                if case == "content":
                    body = b"corrupt"
                if case == "symlink":
                    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                if case == "compressed":
                    entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, body)
        if case == "extra":
            archive.writestr("../outside", b"not allowed")
        if case == "duplicate":
            with pytest.warns(UserWarning):
                archive.writestr("run.py", b"again")
    content = output.getvalue()
    if case == "truncated":
        content = content[:100]
    if case == "wrong-source":
        path = "projects/.topodrome-sources/" + "f" * 64
    target = tmp_path / "cache"
    target.mkdir()
    with pytest.raises(StorageVerificationError):
        unpack_source_bundle(path, content, target, 10 if case == "overflow" else 1024**2)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("filename", ["../outside", "/absolute", "a/../b", "a//b", ".topodrome/x"])
def test_manifest_paths_cannot_escape_or_collide_with_metadata(tmp_path, filename):
    filehash = hashlib.sha256(b"data").hexdigest()
    identity = hashlib.sha256(filename.encode() + b"\0" + bytes.fromhex(filehash)).hexdigest()
    path = "projects/.topodrome-sources/" + identity
    manifest = json.dumps(
        {"source_ref": "sha256:" + identity, "format_version": 2, "files": {filename: filehash}}
    ).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(".topodrome/manifest.json", manifest)
        archive.writestr(filename, b"data")
    with pytest.raises(StorageVerificationError):
        unpack_source_bundle(path, output.getvalue(), tmp_path, 1024**2)
    assert list(tmp_path.iterdir()) == []


def test_source_directory_takes_one_http_request_and_no_bulk_command(bundle, tmp_path):
    _, path, content, _ = bundle
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, content=content, headers={"ETag": f'"{hashlib.sha256(content).hexdigest()}"'}
        )

    client = SchedulerSmallFileClient(
        "http://scheduler", "site", "private-test", transport=httpx.MockTransport(respond)
    )

    authority = FakeAuthority()

    mirror = HttpMirrorStorage(
        tmp_path / "cache",
        "lab1:/global",
        small_file_client=client,
        authoritative_reader=authority,
        bulk_transport=authority,
        min_free_space_gb=1e-7,
    )
    try:
        assert (mirror.resolve(path) / "run.py").read_bytes() == b"print('hello')"
        mirror.resolve(path)
        assert len(requests) == 1
        assert requests[0].url.path.endswith("/source-bundle")
    finally:
        mirror.close()


def test_unsupported_bundle_endpoint_does_not_disable_small_files(bundle):
    _, path, _, _ = bundle
    body = b"still fast"

    def respond(request):
        if request.url.path.endswith("source-bundle"):
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(
            200, content=body, headers={"ETag": f'"{hashlib.sha256(body).hexdigest()}"'}
        )

    client = SchedulerSmallFileClient(
        "http://scheduler", "site", "private-test", transport=httpx.MockTransport(respond)
    )
    try:
        with pytest.raises(SmallFileFallback):
            client.download_source_bundle(path)
        assert client.download("jobs/job/file")[0] == body
    finally:
        client.close()


def test_source_authentication_failure_never_downgrades(bundle):
    _, path, _, _ = bundle
    client = SchedulerSmallFileClient(
        "http://scheduler",
        "site",
        "private-test",
        transport=httpx.MockTransport(lambda _: httpx.Response(403, json={"detail": "rejected"})),
    )
    try:
        with pytest.raises(StorageUnavailableError):
            client.download_source_bundle(path)
        assert client._source_supported
    finally:
        client.close()
