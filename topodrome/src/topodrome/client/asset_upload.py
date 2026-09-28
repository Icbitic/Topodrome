"""Hash local regular files and stream their immutable asset upload."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Generator
from pathlib import Path

from topodrome.asset_upload import (
    CHUNK_BYTES,
    MAX_FILE_BYTES,
    MAX_HEADER_BYTES,
    MAX_TOTAL_BYTES,
    AssetUploadHeader,
)
from topodrome.scheduler.assets import AssetFileSummary


def prepare_asset_upload(path: str | Path) -> tuple[AssetUploadHeader, Path]:
    source = Path(path).expanduser().absolute()
    if source.is_symlink():
        raise ValueError("asset upload does not follow symbolic links")
    kind = "directory" if source.is_dir() else "file"
    files = source.rglob("*") if kind == "directory" else iter([source])
    members: list[AssetFileSummary] = []
    total = 0
    for file in files:
        mode = file.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("asset upload does not follow symbolic links")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("asset upload requires regular files")
        relative = file.relative_to(source).as_posix() if kind == "directory" else file.name
        with file.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            total += size
            if len(members) >= 10000 or size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError("asset upload exceeds file count or size safety limits")
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        members.append(AssetFileSummary(relative_path=relative, size_bytes=size, sha256=digest))
    members.sort(key=lambda item: item.relative_path)
    return AssetUploadHeader(kind=kind, files=tuple(members)), source


def upload_chunks(header: AssetUploadHeader, source: Path) -> Generator[bytes, None, None]:
    encoded = header.model_dump_json().encode()
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError("asset upload manifest exceeds metadata limit")
    yield len(encoded).to_bytes(8, "big")
    yield encoded
    for member in header.files:
        file = source / member.relative_path if header.kind == "directory" else source
        fd = os.open(file, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != member.size_bytes:
                raise ValueError("asset changed during upload; retry with a stable local copy")
            remaining = member.size_bytes
            while remaining:
                block = handle.read(min(CHUNK_BYTES, remaining))
                if not block:
                    raise ValueError("asset changed during upload")
                yield block
                remaining -= len(block)
            if handle.read(1):
                raise ValueError("asset changed during upload")
