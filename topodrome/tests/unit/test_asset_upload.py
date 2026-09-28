from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from topodrome.asset_upload import CHUNK_BYTES, AssetUploadHeader
from topodrome.client.asset_upload import prepare_asset_upload, upload_chunks
from topodrome.domain.errors import AdmissionRejectedError, ConflictError, NotFoundError
from topodrome.scheduler.asset_upload import receive_asset
from topodrome.scheduler.assets import AssetCatalog, AssetUploadRequest

pytestmark = pytest.mark.unit


async def _stream(body: bytes):
    for start in range(0, len(body), 7):
        yield body[start : start + 7]


def _body(path: Path) -> bytes:
    return b"".join(upload_chunks(*prepare_asset_upload(path)))


@pytest.mark.asyncio
async def test_upload_without_producer_is_private_immutable_and_pinned(tmp_path: Path) -> None:
    source = tmp_path / "params.pt"
    source.write_bytes(b"first")
    catalog = AssetCatalog(tmp_path / "global")
    request = AssetUploadRequest(name="lens", version="v1")
    first = await receive_asset(catalog, request, "alice", _stream(_body(source)))
    assert first.created
    assert first.asset.producer_job_id is None
    assert first.asset.upload_id is not None
    pin = catalog.resolve(first.asset.ref, requester_username="alice")
    assert (catalog.global_root / pin.members[0].logical_path).read_bytes() == b"first"
    with pytest.raises(NotFoundError):
        catalog.get("lens", "v1", requester_username="bob")
    repeated = await receive_asset(catalog, request, "alice", _stream(_body(source)))
    assert not repeated.created
    assert repeated.asset == first.asset
    assert len(list((catalog.global_root / "artifacts/uploads").iterdir())) == 1
    source.write_bytes(b"second")
    with pytest.raises(ConflictError):
        await receive_asset(catalog, request, "alice", _stream(_body(source)))
    second = await receive_asset(
        catalog, request.model_copy(update={"version": "v2"}), "alice", _stream(_body(source))
    )
    assert second.asset.digest != first.asset.digest
    assert (catalog.global_root / pin.members[0].logical_path).read_bytes() == b"first"


@pytest.mark.asyncio
async def test_directory_shared_upload_and_bounded_chunks(tmp_path: Path) -> None:
    source = tmp_path / "input"
    (source / "nested").mkdir(parents=True)
    (source / "empty").write_bytes(b"")
    (source / "nested/data").write_bytes(b"x" * (CHUNK_BYTES * 3 + 17))
    header, root = prepare_asset_upload(source)
    assert header.kind == "directory"
    assert max(map(len, upload_chunks(header, root))) <= CHUNK_BYTES

    async def stream():
        for chunk in upload_chunks(header, root):
            yield chunk

    catalog = AssetCatalog(tmp_path / "global")
    result = await receive_asset(
        catalog,
        AssetUploadRequest(name="data", version="v1", visibility="shared"),
        "alice",
        stream(),
    )
    assert catalog.get("data", "v1", visibility="shared", requester_username="bob") == result.asset
    assert [item.relative_path for item in result.asset.files] == ["empty", "nested/data"]


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["truncated", "checksum", "trailing", "huge_header", "path"])
async def test_invalid_upload_never_publishes(tmp_path: Path, damage: str) -> None:
    source = tmp_path / "file"
    source.write_bytes(b"payload")
    body = _body(source)
    if damage == "truncated":
        body = body[:-1]
    elif damage == "checksum":
        body = body[:-1] + b"X"
    elif damage == "trailing":
        body += b"extra"
    elif damage == "huge_header":
        body = (2**40).to_bytes(8, "big")
    else:
        size = int.from_bytes(body[:8], "big")
        header = json.loads(body[8 : 8 + size])
        header["files"][0]["relative_path"] = "../escape"
        encoded = json.dumps(header).encode()
        body = len(encoded).to_bytes(8, "big") + encoded + body[8 + size :]
    catalog = AssetCatalog(tmp_path / "global")
    with pytest.raises(AdmissionRejectedError):
        await receive_asset(
            catalog, AssetUploadRequest(name="data", version="v1"), "alice", _stream(body)
        )
    assert catalog.list_visible(requester_username="alice") == []
    assert not list(catalog.global_root.glob("artifacts/uploads/.pending-*"))


@pytest.mark.asyncio
async def test_disconnect_cleans_partial_and_does_not_publish(tmp_path: Path) -> None:
    source = tmp_path / "file"
    source.write_bytes(b"payload")
    body = _body(source)

    async def disconnect():
        yield body[:-1]
        raise asyncio.CancelledError

    catalog = AssetCatalog(tmp_path / "global")
    with pytest.raises(asyncio.CancelledError):
        await receive_asset(
            catalog, AssetUploadRequest(name="data", version="v1"), "alice", disconnect()
        )
    assert not list(catalog.global_root.glob("artifacts/uploads/.pending-*"))
    assert catalog.list_visible(requester_username="alice") == []


def test_local_symlinks_and_file_directory_conflicts_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symbolic"):
        prepare_asset_upload(source)
    with pytest.raises(ValueError, match="conflicting"):
        AssetUploadHeader(
            kind="directory",
            files=[
                {"relative_path": path, "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
                for path in ("file", "file/nested")
            ],
        )


def test_multi_gib_file_metadata_is_not_subject_to_source_limits() -> None:
    header = AssetUploadHeader(
        kind="file",
        files=[{"relative_path": "params.pt", "size_bytes": 5 * 1024**3, "sha256": "a" * 64}],
    )
    assert header.files[0].size_bytes == 5 * 1024**3


def test_local_manifest_uses_string_order_for_nested_paths(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a/z").write_bytes(b"nested")
    (tmp_path / "a.txt").write_bytes(b"file")
    header, _ = prepare_asset_upload(tmp_path)
    assert [item.relative_path for item in header.files] == ["a.txt", "a/z"]
