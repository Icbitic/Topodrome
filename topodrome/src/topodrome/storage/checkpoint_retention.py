"""Attempt-local snapshot rotation, independent of remote cache eviction.

Selection for verification durably pins a filename before releasing the short
cross-process lock. Network transfers never hold this lock. Pins are conservative
and survive failed verification/restart; terminal-job retention owns their expiry.
"""

from __future__ import annotations

import fcntl
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path

SNAPSHOT_NAME = re.compile(r"^([0-9]{8,})-[a-f0-9]{12}\.pkl$")


def _sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _lock(directory: Path, *, blocking: bool):
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise ValueError("checkpoint directory must not be a symlink")
    fd = os.open(directory / ".retention.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        meta = os.fstat(fd)
        if not stat.S_ISREG(meta.st_mode) or meta.st_uid != os.getuid() or meta.st_nlink != 1:
            raise ValueError("unsafe checkpoint retention lock")
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield
    finally:
        os.close(fd)


def pin_latest_checkpoint(directory: Path) -> Path | None:
    """Keep selected bytes through verification, retries and scheduler restores."""
    if not directory.exists():
        return None
    with _lock(directory, blocking=True):
        candidates = sorted(directory.glob("*.pkl"), key=_order)
        if not candidates:
            return None
        path = candidates[-1]
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("checkpoint must be a regular file")
        pins = directory / ".retention-pins"
        pins.mkdir(mode=0o700, exist_ok=True)
        if not stat.S_ISDIR(pins.lstat().st_mode):
            raise ValueError("unsafe checkpoint retention pins")
        fd = os.open(pins / path.name, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            meta = os.fstat(fd)
            if not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1:
                raise ValueError("unsafe checkpoint pin")
            os.fsync(fd)
        finally:
            os.close(fd)
        _sync_directory(pins)
        _sync_directory(directory)
        return path


def _order(path: Path):
    prefix = path.name.split("-", 1)[0]
    return (int(prefix) if prefix.isdecimal() else -1, path.name)


def rotate_checkpoints(directory: Path, *, keep: int = 2) -> int:
    """Best-effort bounded deletion after a new snapshot has been fsynced/hashed.

    Skip rather than stall computation if a verifier is selecting a snapshot.
    Unknown files, hardlinks, symlinks, and every pinned snapshot are preserved.
    """
    if keep < 2:
        raise ValueError("checkpoint retention must keep at least two snapshots")
    try:
        with _lock(directory, blocking=False):
            pins = directory / ".retention-pins"
            if pins.is_symlink():
                raise ValueError("unsafe checkpoint retention pins")
            protected = {p.name for p in pins.iterdir()} if pins.exists() else set()
            paths = sorted(
                (p for p in directory.iterdir() if SNAPSHOT_NAME.fullmatch(p.name)), key=_order
            )
            removed = 0
            for path in paths[:-keep]:
                if path.name in protected:
                    continue
                meta = path.lstat()
                if not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1:
                    continue
                path.unlink()
                removed += 1
                if removed >= 100:
                    break
            if removed:
                _sync_directory(directory)
            return removed
    except BlockingIOError:
        return 0
