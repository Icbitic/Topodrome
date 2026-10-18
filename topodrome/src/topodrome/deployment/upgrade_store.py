"""Private, fsynced update journal shared by the scheduler and external runner.

No deployment command is blindly replayed after an ambiguous crash. The journal
is outside release directories, so its admission fence survives code replacement.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from topodrome.domain.errors import ConflictError, NotFoundError
from topodrome.domain.specs import utc_now
from topodrome.domain.upgrades import (
    TERMINAL_UPGRADES,
    UpgradeJournal,
    UpgradePlan,
    UpgradeRecord,
    UpgradeRequest,
)


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise PermissionError("upgrade directory must be owned by the service account")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("upgrade directory must be private")


def _open_regular(path: Path, *, private: bool = False) -> int:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o022
        or (private and stat.S_IMODE(info.st_mode) & 0o077)
    ):
        os.close(fd)
        raise PermissionError("upgrade input is not a protected owned regular file")
    return fd


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class UpgradeStore:
    def __init__(self, global_root: Path):
        self.root = global_root / ".topodrome-upgrades"
        self.path = self.root / "state.json"

    @contextmanager
    def locked(self) -> Iterator[None]:
        _private_directory(self.root)
        fd = os.open(self.root / "state.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise PermissionError("unsafe upgrade lock")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def read(self) -> UpgradeJournal:
        if self.root.is_symlink():
            raise PermissionError("upgrade journal cannot be a symlink")
        if not self.root.exists():
            return UpgradeJournal()
        _private_directory(self.root)
        try:
            fd = _open_regular(self.path, private=True)
        except FileNotFoundError:
            return UpgradeJournal()
        with os.fdopen(fd, "rb") as stream:
            data = stream.read(1024**2 + 1)
        if len(data) > 1024**2:
            raise ValueError("upgrade journal exceeds its bound")
        try:
            return UpgradeJournal.model_validate_json(data)
        except ValueError:
            raise ValueError("upgrade journal is invalid; admission remains fenced") from None

    def _save(self, journal: UpgradeJournal) -> None:
        data = journal.model_dump_json().encode()
        if len(data) > 1024**2:
            raise ValueError("upgrade journal exceeds its bound")
        temporary = self.root / (".state-" + uuid4().hex)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            _sync_directory(self.root)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()

    def get(self, upgrade_id: str) -> UpgradeRecord:
        for record in self.read().records:
            if record.upgrade_id == upgrade_id:
                return record
        raise NotFoundError("upgrade does not exist")

    def current(self) -> UpgradeRecord | None:
        return next((r for r in self.read().records if r.state not in TERMINAL_UPGRADES), None)

    def exclusions(self) -> tuple[bool, frozenset[str]]:
        record = self.current()
        if record is None or record.state in {"queued", "preparing"}:
            return False, frozenset()
        if not record.mutation_started and record.pause == "none":
            return False, frozenset()
        if record.pause == "all" or record.plan.scheduler:
            return True, frozenset()
        return False, frozenset(record.plan.sites)

    def blocked(self, site_id: str) -> bool:
        all_sites, sites = self.exclusions()
        return all_sites or site_id in sites

    def schedule(self, request: UpgradeRequest) -> UpgradeRecord:
        path = request.plan_path
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("upgrade plan must be an absolute protected file")
        with os.fdopen(_open_regular(path, private=True), "rb") as stream:
            payload = stream.read(128 * 1024 + 1)
        if len(payload) > 128 * 1024:
            raise ValueError("upgrade plan exceeds its bound")
        try:
            plan = UpgradePlan.model_validate(yaml.safe_load(payload))
        except (ValueError, yaml.YAMLError):
            raise ValueError("upgrade plan is invalid") from None
        digests = {}
        total_bytes = 0
        # Hash/validate before deciding idempotency; mutable inputs cannot change
        # the meaning of an already queued release.
        for name in plan.files:
            source = path.parent / name
            if any(p.is_symlink() for p in (source, *source.parents)):
                raise ValueError("upgrade files cannot traverse symlinks")
            digest = hashlib.sha256()
            with os.fdopen(_open_regular(source), "rb") as stream:
                if os.fstat(stream.fileno()).st_size > 1024**3:
                    raise ValueError("upgrade file exceeds 1 GiB")
                total_bytes += os.fstat(stream.fileno()).st_size
                if total_bytes > 1024**3:
                    raise ValueError("upgrade release exceeds 1 GiB in total")
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests[name] = digest.hexdigest()
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "plan": plan.model_dump(mode="json"),
                    "files": digests,
                    "pause": request.pause,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        with self.locked():
            journal = self.read()
            for old in journal.records:
                if old.idempotency_key == request.idempotency_key:
                    if old.fingerprint != fingerprint:
                        raise ConflictError("upgrade idempotency key has different content")
                    return old
            if any(r.state not in TERMINAL_UPGRADES for r in journal.records):
                raise ConflictError("an upgrade is already pending; cancel or resolve it first")
            if len(journal.records) >= 100:
                raise ConflictError("upgrade audit limit reached; archive completed records first")
            record = UpgradeRecord(
                plan=plan,
                pause=request.pause,
                idempotency_key=request.idempotency_key,
                fingerprint=fingerprint,
                files=digests,
            )
            release = self.release_path(record)
            _private_directory(release)
            if shutil.disk_usage(self.root).free < total_bytes + 1024**3:
                raise OSError("insufficient free space to freeze an upgrade release")
            for name, expected in digests.items():
                target = release / name
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with os.fdopen(_open_regular(path.parent / name), "rb") as source_stream:
                    fd = os.open(
                        target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
                    )
                    with os.fdopen(fd, "wb") as destination:
                        for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                            destination.write(chunk)
                            digest.update(chunk)
                        destination.flush()
                        os.fsync(destination.fileno())
                if digest.hexdigest() != expected:
                    raise ValueError("upgrade input changed while freezing the release")
                _sync_directory(target.parent)
            _sync_directory(release.parent)
            self._save(journal.model_copy(update={"records": (*journal.records, record)}))
            return record

    def release_path(self, record: UpgradeRecord) -> Path:
        return self.root / "releases" / record.upgrade_id

    def validate_release(self, record: UpgradeRecord) -> None:
        for name, expected in record.files.items():
            digest = hashlib.sha256()
            with os.fdopen(
                _open_regular(self.release_path(record) / name, private=True), "rb"
            ) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("frozen upgrade release has changed")

    def change(self, upgrade_id: str, *, expected: set[str], **updates: Any) -> UpgradeRecord:
        with self.locked():
            journal = self.read()
            record = next((r for r in journal.records if r.upgrade_id == upgrade_id), None)
            if record is None:
                raise NotFoundError("upgrade does not exist")
            if record.state not in expected:
                raise ConflictError("upgrade phase changed")
            updated = UpgradeRecord.model_validate(
                {
                    **record.model_dump(),
                    **updates,
                    "updated_at": utc_now(),
                }
            )
            self._save(
                journal.model_copy(
                    update={
                        "records": tuple(
                            updated if r.upgrade_id == upgrade_id else r for r in journal.records
                        )
                    }
                )
            )
            return updated
