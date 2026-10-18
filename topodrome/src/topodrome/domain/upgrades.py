"""Operator-only, immutable idle-upgrade plans and sanitized status."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from topodrome.domain.specs import StrictModel, new_id, utc_now

PauseScope = Literal["none", "targets", "all"]
UpgradeState = Literal[
    "queued",
    "preparing",
    "waiting",
    "applying",
    "verifying",
    "rolling_back",
    "verifying_rollback",
    "needs_attention",
    "succeeded",
    "rolled_back",
    "failed",
    "canceled",
]
TERMINAL_UPGRADES = frozenset({"succeeded", "rolled_back", "failed", "canceled"})


class UpgradeStep(StrictModel):
    # Trusted operator commands, not experiment payloads. Never exposed in status.
    argv: tuple[str, ...] = Field(min_length=1, max_length=64)
    timeout_seconds: float = Field(default=300, gt=0, le=3600)

    @model_validator(mode="after")
    def safe_arguments(self) -> UpgradeStep:
        if any(not item or "\0" in item or len(item) > 4096 for item in self.argv):
            raise ValueError("upgrade arguments must be bounded nonempty strings")
        executable = self.argv[0].replace("{release}", "/release")
        if not Path(executable).is_absolute():
            raise ValueError("upgrade executables must have absolute paths")
        return self


class UpgradePlan(StrictModel):
    name: str = Field(min_length=1, max_length=160)
    scheduler: bool = False
    sites: tuple[str, ...] = Field(default=(), max_length=64)
    files: tuple[str, ...] = Field(default=(), max_length=64)
    prepare: tuple[UpgradeStep, ...] = Field(default=(), max_length=16)
    apply: tuple[UpgradeStep, ...] = Field(min_length=1, max_length=16)
    verify: tuple[UpgradeStep, ...] = Field(min_length=1, max_length=16)
    rollback: tuple[UpgradeStep, ...] = Field(default=(), max_length=16)
    verify_rollback: tuple[UpgradeStep, ...] = Field(default=(), max_length=16)
    require_new_agent_session: bool = True
    health_timeout_seconds: float = Field(default=180, ge=10, le=1800)

    @model_validator(mode="after")
    def scoped(self) -> UpgradePlan:
        if not self.scheduler and not self.sites:
            raise ValueError("an upgrade must name the scheduler or exact site IDs")
        if len(set(self.sites)) != len(self.sites):
            raise ValueError("upgrade sites must be unique")
        if any(not site or len(site) > 120 for site in self.sites):
            raise ValueError("invalid upgrade site ID")
        if bool(self.rollback) != bool(self.verify_rollback):
            raise ValueError("rollback needs both apply and verification commands")
        if len(set(self.files)) != len(self.files):
            raise ValueError("upgrade files must be unique")
        for name in self.files:
            path = Path(name)
            if path.is_absolute() or not path.parts or any(p in {".", ".."} for p in path.parts):
                raise ValueError("upgrade files must stay inside the plan directory")
        return self


class UpgradeRequest(StrictModel):
    plan_path: Path
    pause: PauseScope = "targets"
    idempotency_key: str = Field(min_length=1, max_length=120)


class UpgradeClaim(StrictModel):
    runner_id: str = Field(min_length=1, max_length=120)


class UpgradeRecovery(StrictModel):
    action: Literal["verify", "rollback"]


class UpgradePauseRequest(StrictModel):
    pause: PauseScope
    expected_pause: PauseScope


class UpgradeRecord(StrictModel):
    upgrade_id: str = Field(default_factory=lambda: new_id("upgrade"))
    plan: UpgradePlan
    pause: PauseScope
    idempotency_key: str
    fingerprint: str
    files: dict[str, str] = Field(default_factory=dict)
    state: UpgradeState = "queued"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    runner_id: str | None = None
    claimed_at: datetime | None = None
    sessions_before: dict[str, str] = Field(default_factory=dict)
    scheduler_before: str | None = None
    mutation_started: bool = False
    blockers: dict[str, int] = Field(default_factory=dict)
    error_type: str | None = None

    def summary(self) -> dict[str, object]:
        return {
            "upgrade_id": self.upgrade_id,
            "name": self.plan.name,
            "level": "cluster"
            if self.plan.scheduler and self.plan.sites
            else "scheduler"
            if self.plan.scheduler
            else "sites",
            "sites": list(self.plan.sites),
            "pause": self.pause,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "blockers": self.blockers,
            "error_type": self.error_type,
            "release_digest": self.fingerprint,
        }


class UpgradeJournal(StrictModel):
    schema_version: Literal[1] = 1
    records: tuple[UpgradeRecord, ...] = ()

    @model_validator(mode="after")
    def one_live_fence(self) -> UpgradeJournal:
        if len({r.upgrade_id for r in self.records}) != len(self.records):
            raise ValueError("upgrade identities must be unique")
        if sum(r.state not in TERMINAL_UPGRADES for r in self.records) > 1:
            raise ValueError("only one pending upgrade is supported")
        return self
