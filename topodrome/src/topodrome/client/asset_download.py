"""Verify downloads in a private staging tree before exposing an editable copy."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path, PureWindowsPath
from typing import BinaryIO, TypeVar

from topodrome.domain.specs import AssetMember
from topodrome.scheduler.assets import AssetFileSummary, AssetSummary
from topodrome.storage.assets import asset_manifest_digest

CHUNK_BYTES = 1024 * 1024
T = TypeVar("T")


async def download_io(operation: Callable[[], T]) -> T:
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class AssetDownload:
    def __init__(self, asset: AssetSummary, destination: str | Path) -> None:
        files = asset.files
        paths = [item.relative_path for item in files]
        if not paths or paths != sorted(set(paths)):
            raise ValueError("asset manifest contains empty, duplicate, or unordered paths")
        if asset.kind == "file" and len(files) != 1:
            raise ValueError("file asset must contain exactly one member")
        for name in paths:
            if PureWindowsPath(name).drive:
                raise ValueError("asset path is not portable")
            parts = name.split("/")
            if any("/".join(parts[:i]) in paths for i in range(1, len(parts))):
                raise ValueError("asset manifest contains conflicting paths")
        members = [AssetMember(logical_path=f.relative_path, **f.model_dump()) for f in files]
        if (
            asset_manifest_digest(members) != asset.digest
            or sum(f.size_bytes for f in files) != asset.total_size_bytes
        ):
            raise ValueError("asset manifest content identity is invalid")
        self.asset = asset
        self.destination = Path(destination).expanduser().absolute()
        if os.path.lexists(self.destination):
            raise FileExistsError(f"destination already exists: {self.destination}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Path(
            tempfile.mkdtemp(prefix=".topodrome-download-", dir=self.destination.parent)
        )
        self.payload = self.stage / "payload"
        if asset.kind == "directory":
            self.payload.mkdir(mode=0o700)

    def __enter__(self) -> AssetDownload:
        return self

    def __exit__(self, *_: object) -> None:
        shutil.rmtree(self.stage)

    def open_file(self, item: AssetFileSummary) -> BinaryIO:
        path = self.payload / item.relative_path if self.asset.kind == "directory" else self.payload
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        return os.fdopen(descriptor, "wb")

    @staticmethod
    def verify_file(handle: BinaryIO, item: AssetFileSummary, count: int, digest: str) -> None:
        if count != item.size_bytes or digest != item.sha256:
            raise ValueError(
                f"asset download failed size/checksum verification: {item.relative_path}"
            )
        handle.flush()
        os.fsync(handle.fileno())

    def commit(self) -> Path:
        if self.asset.kind == "file":
            os.link(self.payload, self.destination)  # Atomic, never overwrites.
        else:
            # Claim an absent destination first; a concurrently created path wins.
            self.destination.mkdir(mode=0o700)
            placeholder = self.destination.stat()
            try:
                self.payload.rename(self.destination)
            except BaseException:
                # Only remove our empty placeholder, never somebody else's data.
                try:
                    current = self.destination.lstat()
                    if (current.st_dev, current.st_ino) == (placeholder.st_dev, placeholder.st_ino):
                        self.destination.rmdir()
                except OSError:
                    pass
                raise
        return self.destination
