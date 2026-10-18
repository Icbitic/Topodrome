"""Public immutable download descriptors; never expose authority filesystem paths."""

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from topodrome.domain.specs import StrictModel
from topodrome.storage.paths import normalize_logical_path


class DownloadFile(StrictModel):
    relative_path: str = Field(min_length=1, max_length=1200)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("relative_path")
    @classmethod
    def safe_path(cls, value):
        return normalize_logical_path(value)


class DownloadRequest(StrictModel):
    reference: str = Field(min_length=1, max_length=1600)


class DownloadManifest(StrictModel):
    download_id: str = Field(pattern=r"^download_[a-f0-9]{32}$")
    ref: str
    kind: Literal["file", "directory"]
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    total_size_bytes: int = Field(ge=0)
    files: tuple[DownloadFile, ...] = Field(min_length=1, max_length=10000)
    expires_at: datetime


class DownloadRenewal(StrictModel):
    download_id: str = Field(pattern=r"^download_[a-f0-9]{32}$")
    expires_at: datetime
