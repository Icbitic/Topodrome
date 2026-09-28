"""Shared, bounded source-bundle admission limits."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# These legacy defaults keep a new client conservative when it talks to a
# scheduler version that predates limit discovery.
MAX_SOURCE_FILES = 1024
MAX_SOURCE_PATH_CHARS = 500
MAX_SOURCE_FILE_BYTES = 16 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_SOURCE_ENCODED_FILE_CHARS = 4 * ((MAX_SOURCE_FILE_BYTES + 2) // 3)

# Runtime limits may be changed without replacing code, but never beyond these
# process-safety ceilings. Source upload currently uses JSON/base64 and therefore
# materializes both encoded and decoded bytes in memory.
HARD_MAX_SOURCE_FILES = 4096
HARD_MAX_SOURCE_PATH_CHARS = 4096
HARD_MAX_SOURCE_FILE_BYTES = 256 * 1024 * 1024
HARD_MAX_SOURCE_TOTAL_BYTES = 1024 * 1024 * 1024
HARD_MAX_SOURCE_ENCODED_FILE_CHARS = 4 * ((HARD_MAX_SOURCE_FILE_BYTES + 2) // 3)


class SourceLimits(BaseModel):
    """One immutable source-upload admission snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    max_files: int = Field(default=MAX_SOURCE_FILES, ge=1, le=HARD_MAX_SOURCE_FILES)
    max_path_chars: int = Field(
        default=MAX_SOURCE_PATH_CHARS,
        ge=1,
        le=HARD_MAX_SOURCE_PATH_CHARS,
    )
    max_file_bytes: int = Field(
        default=MAX_SOURCE_FILE_BYTES,
        ge=1,
        le=HARD_MAX_SOURCE_FILE_BYTES,
    )
    max_total_bytes: int = Field(
        default=MAX_SOURCE_TOTAL_BYTES,
        ge=1,
        le=HARD_MAX_SOURCE_TOTAL_BYTES,
    )

    @model_validator(mode="after")
    def _total_covers_one_file(self) -> SourceLimits:
        if self.max_total_bytes < self.max_file_bytes:
            raise ValueError("source max_total_bytes must be at least max_file_bytes")
        return self

    @property
    def max_encoded_file_chars(self) -> int:
        return 4 * ((self.max_file_bytes + 2) // 3)


LEGACY_SOURCE_LIMITS = SourceLimits()

__all__ = [
    "HARD_MAX_SOURCE_ENCODED_FILE_CHARS",
    "HARD_MAX_SOURCE_FILES",
    "HARD_MAX_SOURCE_FILE_BYTES",
    "HARD_MAX_SOURCE_PATH_CHARS",
    "HARD_MAX_SOURCE_TOTAL_BYTES",
    "LEGACY_SOURCE_LIMITS",
    "MAX_SOURCE_ENCODED_FILE_CHARS",
    "MAX_SOURCE_FILES",
    "MAX_SOURCE_FILE_BYTES",
    "MAX_SOURCE_PATH_CHARS",
    "MAX_SOURCE_TOTAL_BYTES",
    "SourceLimits",
]
