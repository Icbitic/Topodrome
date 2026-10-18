"""Operator-owned, optimistic live placement preferences."""

from __future__ import annotations

import re

from pydantic import Field, field_validator

from topodrome.domain.specs import StrictModel


def validate_preferred_sites(values: tuple[str, ...]) -> tuple[str, ...]:
    if len(values) > 64 or len(set(values)) != len(values):
        raise ValueError("preferred sites must be unique and contain at most 64 IDs")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,119}", v) for v in values):
        raise ValueError("preferred sites require valid site IDs")
    return values


class SitePreferenceRequest(StrictModel):
    preferred_site_ids: tuple[str, ...] = Field(max_length=64)
    expected_generation: int = Field(ge=1)

    _valid_sites = field_validator("preferred_site_ids")(validate_preferred_sites)
