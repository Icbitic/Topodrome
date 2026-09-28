"""Receive verified assets without GPU admission, job creation, or body buffering."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from collections.abc import AsyncIterator, Callable
from functools import partial
from pathlib import Path
from typing import BinaryIO, TypeVar
from uuid import uuid4

from topodrome.asset_upload import CHUNK_BYTES, MAX_HEADER_BYTES, MIN_FREE_BYTES, AssetUploadHeader
from topodrome.domain.errors import AdmissionRejectedError, ConflictError
from topodrome.scheduler.assets import AssetCatalog, AssetPublishResult, AssetUploadRequest
from topodrome.storage.base import ObjectDigest
from topodrome.storage.blobs import blob_transaction, install_local_blob, register_blob

T = TypeVar("T")


async def _io(operation: Callable[[], T]) -> T:
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def _sync_file(output: BinaryIO) -> None:
    output.flush()
    os.fsync(output.fileno())


class _BodyReader:
    def __init__(self, stream: AsyncIterator[bytes]) -> None:
        self.stream = stream
        self.pending = memoryview(b"")

    async def read(self, size: int) -> bytes:
        while not self.pending:
            try:
                self.pending = memoryview(await anext(self.stream))
            except StopAsyncIteration:
                return b""
        result = bytes(self.pending[:size])
        self.pending = self.pending[len(result) :]
        return result

    async def exact(self, size: int) -> bytes:
        parts = bytearray()
        while len(parts) < size:
            block = await self.read(min(CHUNK_BYTES, size - len(parts)))
            if not block:
                raise AdmissionRejectedError("asset upload was truncated")
            parts.extend(block)
        return bytes(parts)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


async def receive_asset(
    catalog: AssetCatalog,
    request: AssetUploadRequest,
    owner: str,
    stream: AsyncIterator[bytes],
) -> AssetPublishResult:
    reader = _BodyReader(stream)
    header_size = int.from_bytes(await reader.exact(8), "big")
    if not 0 < header_size <= MAX_HEADER_BYTES:
        raise AdmissionRejectedError("invalid asset upload header size")
    try:
        header = AssetUploadHeader.model_validate_json(await reader.exact(header_size))
    except ValueError:
        raise AdmissionRejectedError("invalid asset upload file manifest") from None

    root = catalog.global_root / "artifacts" / "uploads"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    await _io(lambda: _fsync_directory(root.parent))
    await _io(lambda: _fsync_directory(catalog.global_root))
    upload_id = uuid4().hex
    stage = root / f".pending-{upload_id}"
    stage.mkdir(mode=0o700)
    destination = root / upload_id
    try:
        remaining = sum(item.size_bytes for item in header.files)
        if shutil.disk_usage(root).free < remaining + MIN_FREE_BYTES:
            raise AdmissionRejectedError("insufficient storage for asset upload")
        for item in header.files:
            target = stage / "files" / item.relative_path
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            digest = hashlib.sha256()
            with target.open("xb") as output:
                target.chmod(0o600)
                remaining = item.size_bytes
                while remaining:
                    block = await reader.read(min(CHUNK_BYTES, remaining))
                    if not block:
                        raise AdmissionRejectedError("asset upload was truncated")
                    if shutil.disk_usage(root).free < len(block) + MIN_FREE_BYTES:
                        raise AdmissionRejectedError("asset upload storage floor reached")
                    # The write is bounded; never collect the complete body in RAM.
                    await _io(partial(output.write, block))
                    digest.update(block)
                    remaining -= len(block)
                await _io(lambda: _sync_file(output))
            if digest.hexdigest() != item.sha256:
                raise AdmissionRejectedError("asset upload checksum mismatch")
            target.chmod(0o444)
        if await reader.read(1):
            raise AdmissionRejectedError("asset upload contains trailing bytes")

        def publish() -> AssetPublishResult:
            if catalog.content_addressed:
                with blob_transaction(catalog.global_root) as db:
                    for item in header.files:
                        register_blob(db, ObjectDigest(item.size_bytes, item.sha256))
                    db.commit()  # Journal ownership before installing any payload.
                    for item in header.files:
                        digest = ObjectDigest(size_bytes=item.size_bytes, sha256=item.sha256)
                        install_local_blob(catalog.global_root,
                                           stage / "files" / item.relative_path, digest)
                    return catalog.publish_uploaded(
                        request, owner_username=owner, upload_id=upload_id,
                        kind=header.kind, files=header.files, _blob_lock_held=True,
                    )
            for directory, _, _ in os.walk(stage, topdown=False):
                _fsync_directory(Path(directory))
            stage.rename(destination)
            _fsync_directory(root)
            # If publication raises after linking a manifest, retain the promoted
            # payload: it may already be referenced. Never delete it speculatively.
            try:
                result = catalog.publish_uploaded(
                    request,
                    owner_username=owner,
                    upload_id=upload_id,
                    kind=header.kind,
                    files=header.files,
                )
            except ConflictError:
                # A known immutable-name conflict proves our payload was not linked.
                shutil.rmtree(destination)
                raise
            if not result.created:
                shutil.rmtree(destination)  # Only this redundant unique upload.
            return result

        task = asyncio.create_task(asyncio.to_thread(publish))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Do not remove staging underneath a publication still using it.
            await task
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
