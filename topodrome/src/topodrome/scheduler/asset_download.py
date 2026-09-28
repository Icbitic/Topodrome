"""Authenticated, descriptor-pinned reads of named asset members."""

from __future__ import annotations

import os
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
