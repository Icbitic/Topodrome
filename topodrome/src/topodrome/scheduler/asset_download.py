"""Authenticated, descriptor-pinned reads of named asset members."""

from __future__ import annotations

import asyncio
import os
import re
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from topodrome.domain.errors import NotFoundError
from topodrome.scheduler.assets import ResolvedAssetMember
from topodrome.storage.paths import normalize_logical_path

CHUNK_BYTES = 1024 * 1024


def open_member(root: Path, member: ResolvedAssetMember) -> BinaryIO:
    """Walk only manifest-owned paths, refusing symlinks at every component."""
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    descriptor = -1
    try:
        parts = normalize_logical_path(member.logical_path).split("/")
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != member.size_bytes:
            raise OSError("asset member does not match committed metadata")
        result = os.fdopen(descriptor, "rb")
        descriptor = -1
        return result
    except OSError:
        raise NotFoundError("asset file is unavailable") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory)


class AssetFileResponse(StreamingResponse):
    def __init__(self, handle: BinaryIO, member: ResolvedAssetMember) -> None:
        self.handle = handle

        def chunks() -> Iterator[bytes]:
            remaining = member.size_bytes
            while remaining:
                block = handle.read(min(CHUNK_BYTES, remaining))
                if not block:
                    raise OSError("asset file became truncated")
                remaining -= len(block)
                yield block

        super().__init__(
            chunks(),
            media_type="application/octet-stream",
            headers={
                "Content-Length": str(member.size_bytes),
                "ETag": f'"{member.sha256}"',
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.handle.close()


class DownloadFileResponse(StreamingResponse):
    """Range-capable immutable read, renewing the short transfer reference."""

    def __init__(self, handle, member, *, range_header=None, if_match=None, renew=None):
        self.handle = handle
        self.renew = renew
        etag = f'"{member.sha256}"'
        headers = {
            "ETag": etag,
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        }
        start = 0
        code = 200
        if if_match is not None and if_match != etag:
            code = 412
        elif range_header:
            match = re.fullmatch(r"bytes=(\d{1,20})-", range_header)
            if match is None or int(match[1]) >= member.size_bytes:
                code = 416
                headers["Content-Range"] = f"bytes */{member.size_bytes}"
            else:
                code = 206
                start = int(match[1])
                headers["Content-Range"] = (
                    f"bytes {start}-{member.size_bytes - 1}/{member.size_bytes}"
                )
        count = member.size_bytes - start if code in {200, 206} else 0
        headers["Content-Length"] = str(count)

        async def chunks():
            from .storage_retention import finish_io

            await finish_io(handle.seek, start)
            remaining = count
            while remaining:
                block = await finish_io(handle.read, min(CHUNK_BYTES, remaining))
                if not block:
                    raise OSError("download file became truncated")
                remaining -= len(block)
                yield block

        super().__init__(
            chunks(), status_code=code, media_type="application/octet-stream", headers=headers
        )

    async def __call__(self, scope, receive, send):
        async def keepalive():
            while True:
                await asyncio.sleep(60)
                await self.renew()

        keeper = asyncio.create_task(keepalive()) if self.renew else None
        stream = asyncio.create_task(super().__call__(scope, receive, send))
        try:
            done, _ = await asyncio.wait(
                [stream, keeper] if keeper else [stream],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                task.result()
        finally:
            if keeper:
                keeper.cancel()
            stream.cancel()
            await asyncio.gather(
                *([stream, keeper] if keeper else [stream]), return_exceptions=True
            )
            self.handle.close()
