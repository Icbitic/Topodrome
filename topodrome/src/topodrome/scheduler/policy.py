"""Validated immutable scheduling-policy snapshots and hot reload."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from topodrome.config import ModelCatalog, SchedulerSettings, SchedulingPolicySettings
from topodrome.domain.model_address import model_label
from topodrome.domain.specs import utc_now


@dataclass(frozen=True, slots=True)
class SchedulingPolicySnapshot:
    """One atomic policy view captured by a future dispatch pass."""

    settings: SchedulingPolicySettings
    startup_seconds_by_model: Mapping[str, Mapping[str, float]]
    generation: int
    loaded_at: datetime

    def startup_seconds(
        self, repo_id: str, revision: str | None = None, subfolder: str | None = None,
    ) -> dict[str, float]:
        key = model_label(repo_id, revision, subfolder)
        return dict(self.startup_seconds_by_model.get(key, {}))

    def status(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "loaded_at": self.loaded_at,
        }


@dataclass(frozen=True, slots=True)
class SchedulingPolicyState:
    active: SchedulingPolicySnapshot
    pending: SchedulingPolicySnapshot | None = None


class PolicyStateCommitUncertainError(OSError):
    """The atomic replace happened but its directory fsync did not finish."""

    def __init__(self, state: SchedulingPolicyState) -> None:
        super().__init__("scheduler policy state commit durability is uncertain")
        self.state = state


class SchedulingPolicyStore:
    """Crash-safe owner-only state for an active and staged policy pair."""

    _SCHEMA_VERSION = 1
    _MAXIMUM_BYTES = 1_048_576

    def __init__(self, global_root: Path) -> None:
        root = global_root.expanduser().resolve()
        if root == Path("/"):
            raise ValueError("scheduler global root must be a scoped directory")
        root.mkdir(parents=True, exist_ok=True)
        self.directory = root / ".scheduler-state"
        self.directory.mkdir(mode=0o700, exist_ok=True)
        if self.directory.is_symlink():
            raise ValueError("scheduler policy state directory cannot be a symlink")
        metadata = self.directory.stat()
        if metadata.st_uid != os.geteuid():
            raise PermissionError("scheduler policy state directory has an unexpected owner")
        self.directory.chmod(0o700)
        self.path = self.directory / "scheduling-policy.json"

    def load_or_seed(
        self,
        initial: SchedulingPolicySnapshot,
    ) -> SchedulingPolicyState:
        state = self._read()
        if state is not None:
            return state
        state = SchedulingPolicyState(active=initial)
        self.save(state)
        return state

    def save(self, state: SchedulingPolicyState) -> None:
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "active": _snapshot_payload(state.active),
            "pending": (_snapshot_payload(state.pending) if state.pending is not None else None),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > self._MAXIMUM_BYTES:
            raise ValueError("scheduler policy state exceeds its size limit")
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        replaced = False
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(encoded)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            replaced = True
            # The temporary inode was created 0600; chmod after replace would
            # introduce a failure point after the visible commit.
            _fsync_directory(self.directory)
        except Exception as exc:
            if replaced:
                # The path now names the candidate. Tell the caller to pause
                # dispatch even if the final durability proof is ambiguous.
                # A later identical save retries the fsync safely.
                raise PolicyStateCommitUncertainError(state) from exc
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                temporary.unlink()

    def _read(self) -> SchedulingPolicyState | None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except FileNotFoundError:
            return None
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) & 0o077
                or metadata.st_size > self._MAXIMUM_BYTES
            ):
                raise PermissionError("scheduler policy state is not a private regular file")
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = -1
                payload = json.load(handle)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if not isinstance(payload, dict) or payload.get("schema_version") != self._SCHEMA_VERSION:
            raise ValueError("scheduler policy state schema is invalid")
        active = _snapshot_from_payload(payload.get("active"))
        pending_payload = payload.get("pending")
        pending = None if pending_payload is None else _snapshot_from_payload(pending_payload)
        if pending is not None and pending.generation != active.generation + 1:
            raise ValueError("pending scheduler policy generation is invalid")
        return SchedulingPolicyState(active=active, pending=pending)


def initial_policy_snapshot(
    settings: SchedulerSettings,
    model_catalog: ModelCatalog | None,
) -> SchedulingPolicySnapshot:
    return SchedulingPolicySnapshot(
        settings=settings.scheduling.model_copy(deep=True),
        startup_seconds_by_model=_startup_estimates(model_catalog),
        generation=1,
        loaded_at=utc_now(),
    )


def load_policy_snapshot(
    settings_path: Path,
    *,
    current_settings: SchedulerSettings,
    current_model_catalog: ModelCatalog | None,
    previous: SchedulingPolicySnapshot,
) -> SchedulingPolicySnapshot:
    """Validate both files before returning a replacement policy pointer.

    Only ``scheduling`` and model-location ``estimated_startup_seconds`` may
    differ. Comparisons never serialize scheduler secrets, and the caller does
    not mutate its current snapshot until this function returns successfully.
    """

    candidate_settings = SchedulerSettings.from_yaml(settings_path)
    candidate_non_policy = candidate_settings.model_copy(
        update={"scheduling": current_settings.scheduling}
    )
    if candidate_non_policy != current_settings:
        raise ValueError("scheduler reload contains a non-policy configuration change")

    candidate_catalog = (
        ModelCatalog.from_yaml(candidate_settings.model_config_path)
        if candidate_settings.model_config_path is not None
        else None
    )
    if _catalog_without_startup_estimates(candidate_catalog) != (
        _catalog_without_startup_estimates(current_model_catalog)
    ):
        raise ValueError("model reload contains a change other than startup estimates")

    candidate_policy = candidate_settings.scheduling.model_copy(deep=True)
    # An omitted bootstrap setting must not undo a live operator choice when
    # reloading unrelated settings. An explicitly configured order can replace it.
    if "preferred_site_ids" not in candidate_policy.model_fields_set:
        candidate_policy = candidate_policy.model_copy(
            update={"preferred_site_ids": previous.settings.preferred_site_ids}
        )
    return SchedulingPolicySnapshot(
        settings=candidate_policy,
        startup_seconds_by_model=_startup_estimates(candidate_catalog),
        generation=previous.generation + 1,
        loaded_at=utc_now(),
    )


def same_policy_content(
    first: SchedulingPolicySnapshot,
    second: SchedulingPolicySnapshot,
) -> bool:
    return first.settings == second.settings and {
        alias: dict(estimates) for alias, estimates in first.startup_seconds_by_model.items()
    } == {alias: dict(estimates) for alias, estimates in second.startup_seconds_by_model.items()}


def is_live_priority_change(
    first: SchedulingPolicySnapshot,
    second: SchedulingPolicySnapshot,
) -> bool:
    """Order changes and stricter future admission cannot invalidate active work."""

    fields = {
        "preferred_site_ids",
        "prefer_autodl_over_foreign_processes",
        "foreign_capacity_sharing_site_ids",
        # Idle-only eviction revalidates every exact replica and lease at the
        # site. Changing these timers cannot stop an active job or kernel.
        "institutional_warm_actor_idle_timeout_seconds_by_site",
    }
    return (
        not same_policy_content(first, second)
        and set(second.settings.foreign_capacity_sharing_site_ids).issubset(
            first.settings.foreign_capacity_sharing_site_ids
        )
        and first.settings.model_dump(exclude=fields) == second.settings.model_dump(exclude=fields)
    )


def _startup_estimates(
    model_catalog: ModelCatalog | None,
) -> Mapping[str, Mapping[str, float]]:
    if model_catalog is None:
        return MappingProxyType({})
    estimates = {}
    for definition in model_catalog.variants():
        estimates[model_label(definition.repo_id, definition.revision, definition.subfolder)] = (
            MappingProxyType({
                site_id: float(location.estimated_startup_seconds)
                for site_id, location in definition.sites.items()
                if location.estimated_startup_seconds is not None
            })
        )
    return MappingProxyType(estimates)


def _catalog_without_startup_estimates(
    model_catalog: ModelCatalog | None,
) -> dict[str, Any] | None:
    if model_catalog is None:
        return None
    payload = model_catalog.model_dump(mode="json")
    for definition in payload["models"].values():
        for snapshot in (definition, *definition.get("snapshots", ())):
            for location in snapshot["sites"].values():
                location.pop("estimated_startup_seconds", None)
    return payload


def _snapshot_payload(snapshot: SchedulingPolicySnapshot) -> dict[str, object]:
    return {
        "settings": snapshot.settings.model_dump(mode="json"),
        "startup_seconds_by_model": {
            alias: dict(estimates) for alias, estimates in snapshot.startup_seconds_by_model.items()
        },
        "generation": snapshot.generation,
        "loaded_at": snapshot.loaded_at.isoformat(),
    }


def _snapshot_from_payload(payload: object) -> SchedulingPolicySnapshot:
    if not isinstance(payload, dict):
        raise ValueError("scheduler policy snapshot is invalid")
    generation = payload.get("generation")
    loaded_at_value = payload.get("loaded_at")
    estimates_value = payload.get("startup_seconds_by_model")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
        or not isinstance(loaded_at_value, str)
        or not isinstance(estimates_value, dict)
    ):
        raise ValueError("scheduler policy snapshot coordinates are invalid")
    loaded_at = datetime.fromisoformat(loaded_at_value)
    if loaded_at.tzinfo is None:
        raise ValueError("scheduler policy timestamp must be timezone-aware")
    estimates: dict[str, Mapping[str, float]] = {}
    for alias, site_values in estimates_value.items():
        if not isinstance(alias, str) or not alias or not isinstance(site_values, dict):
            raise ValueError("scheduler policy startup estimates are invalid")
        parsed_sites: dict[str, float] = {}
        for site_id, value in site_values.items():
            if (
                not isinstance(site_id, str)
                or not site_id
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 86_400
            ):
                raise ValueError("scheduler policy startup estimate is invalid")
            parsed_sites[site_id] = float(value)
        estimates[alias] = MappingProxyType(parsed_sites)
    return SchedulingPolicySnapshot(
        settings=SchedulingPolicySettings.model_validate(payload.get("settings")),
        startup_seconds_by_model=MappingProxyType(estimates),
        generation=generation,
        loaded_at=loaded_at,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "PolicyStateCommitUncertainError",
    "SchedulingPolicySnapshot",
    "SchedulingPolicyState",
    "SchedulingPolicyStore",
    "initial_policy_snapshot",
    "is_live_priority_change",
    "load_policy_snapshot",
    "same_policy_content",
]
