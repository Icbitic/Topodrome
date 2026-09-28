"""Bounded streaming wire format for direct assets, separate from source JSON."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from topodrome.domain.specs import StrictModel
from topodrome.scheduler.assets import AssetFileSummary, AssetKind

CONTENT_TYPE = "application/vnd.topodrome.asset-upload"
CHUNK_BYTES = 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_FILE_BYTES = 256 * 1024**3
MAX_TOTAL_BYTES = 1024**4
MIN_FREE_BYTES = 1024**3


class AssetUploadHeader(StrictModel):
    schema_version: Literal[1] = 1
    kind: AssetKind
    files: tuple[AssetFileSummary, ...] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def validate_files(self) -> AssetUploadHeader:
        paths = [item.relative_path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("upload paths must be sorted and unique")
        if self.kind == "file" and (len(paths) != 1 or "/" in paths[0]):
            raise ValueError("file upload requires exactly one basename")
        names = set(paths)
        for path in paths:
            parts = path.split("/")
            if any("/".join(parts[:i]) in names for i in range(1, len(parts))):
                raise ValueError("upload contains conflicting file/directory paths")
        if any(item.size_bytes > MAX_FILE_BYTES for item in self.files):
            raise ValueError("asset file exceeds 256 GiB")
        if sum(item.size_bytes for item in self.files) > MAX_TOTAL_BYTES:
            raise ValueError("asset upload exceeds 1 TiB")
        return self
