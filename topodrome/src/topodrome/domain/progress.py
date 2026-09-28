"""Small advisory progress records transported through fenced attempt logs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROGRESS_PREFIX = "TOPODROME_PROGRESS_V1 "
PROGRESS_RECORD_BYTES = 4096


class JobProgress(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    current: float = Field(ge=0)
    total: float | None = Field(default=None, gt=0)
    stage: str = Field(default="", max_length=64)
    message: str = Field(default="", max_length=256)
    elapsed_seconds: float = Field(default=0, ge=0)
    eta_seconds: float | None = Field(default=None, ge=0)
    unit: Literal["items", "bytes"] = "items"

    @field_validator("stage", "message")
    @classmethod
    def printable(cls, value: str) -> str:
        if any(not c.isprintable() for c in value):
            raise ValueError("progress text must be printable")
        return value
