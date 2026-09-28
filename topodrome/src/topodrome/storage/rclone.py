"""rclone VFS adapter with verification against the remote, not its cache."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from topodrome.storage.base import (
    AuthoritativeDigestReader,
    ObjectDigest,
    StorageUnavailableError,
    StorageVerificationError,
    VerifiedObject,
)
from topodrome.storage.local import fsync_and_digest
from topodrome.storage.paths import normalize_logical_path, resolve_beneath
from topodrome.storage.rclone_environment import rclone_process_environment
from topodrome.storage.transfer_process import run_transfer

_RCLONE_NOT_FOUND = 3
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_HASHSUM_BYTES = 16 * 1024
_SHA256 = re.compile(rb"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _RemoteObjectStat:
    size_bytes: int
    modification_time: str | None
    object_id: str | None


class RcloneCommandReader:
    """Read authoritative objects with direct ``rclone`` commands.

    The commands address the configured remote directly. The content stream is
    hashed inside rclone, never read from the VFS mountpoint or accumulated in
    Python, so a match proves that write-back reached the authoritative side.
    """

    def __init__(
        self,
        remote_root: str,
        *,
        binary: Path | str = "rclone",
        command_timeout_seconds: float = 120,
        transfer_timeout_seconds: float = 6 * 60 * 60,
        run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        if not remote_root or any(character in remote_root for character in "\r\n\0"):
            raise ValueError("remote_root must be a non-empty rclone remote path")
        if command_timeout_seconds <= 0:
            raise ValueError("command timeout must be positive")
        if transfer_timeout_seconds <= 0:
            raise ValueError("transfer timeout must be positive")
        resolved_binary = shutil.which(str(binary))
        if resolved_binary is None:
            raise FileNotFoundError(f"rclone executable not found: {binary}")
        self._binary = str(Path(resolved_binary).resolve())
        self._remote_root = remote_root.rstrip("/")
        self._timeout = command_timeout_seconds
        self._transfer_timeout = transfer_timeout_seconds
        self._run = run_transfer if run is subprocess.run else run

    def healthcheck(self) -> bool:
        try:
            completed = self._run(
                [self._binary, "lsf", self._remote_root, "--max-depth", "1"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=rclone_process_environment(),
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0

    def digest(self, logical_path: str) -> ObjectDigest | None:
        normalized = normalize_logical_path(logical_path)
        remote_path = f"{self._remote_root}/{normalized}"
        before = self._stat(remote_path)
        if before is None:
            return None
        try:
            # Let rclone consume and hash the stream. Only its one-line checksum
            # is returned to Python, so verifying a multi-GiB object does not
            # allocate a second object-sized buffer in this process. ``--download``
            # is required because SFTP and several other remotes cannot supply a
            # server-side SHA-256.
            completed = self._run(
                [self._binary, "hashsum", "sha256", remote_path, "--download"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=rclone_process_environment(),
                timeout=self._transfer_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise StorageUnavailableError("authoritative rclone hash timed out") from exc
        except OSError as exc:
            raise StorageUnavailableError("authoritative rclone hash failed") from exc
        if completed.returncode == _RCLONE_NOT_FOUND:
            return None
        if completed.returncode != 0:
            raise StorageUnavailableError("authoritative rclone hash failed")
        if len(completed.stdout) > _MAX_HASHSUM_BYTES:
            raise StorageUnavailableError("authoritative rclone hash response was too large")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise StorageUnavailableError("authoritative rclone hash response was invalid")
        digest_text = lines[0].split(maxsplit=1)[0].lower()
        if not _SHA256.fullmatch(digest_text):
            raise StorageUnavailableError("authoritative rclone hash response was invalid")

        # A remote object replaced during the streamed hash could otherwise
        # combine one version's size with another version's digest. Immutable
        # Topodrome paths should never change, but fail closed if they do.
        after = self._stat(remote_path)
        if after is None or after != before:
            raise StorageUnavailableError("authoritative object changed while hashing")
        return ObjectDigest(size_bytes=before.size_bytes, sha256=digest_text.decode("ascii"))

    def _stat(self, remote_path: str) -> _RemoteObjectStat | None:
        try:
            completed = self._run(
                [self._binary, "lsjson", remote_path, "--stat"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=rclone_process_environment(),
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise StorageUnavailableError("authoritative rclone stat timed out") from exc
        except OSError as exc:
            raise StorageUnavailableError("authoritative rclone stat failed") from exc
        if completed.returncode == _RCLONE_NOT_FOUND:
            return None
        if completed.returncode != 0:
            raise StorageUnavailableError("authoritative rclone stat failed")
        if len(completed.stdout) > _MAX_METADATA_BYTES:
            raise StorageUnavailableError("authoritative rclone stat response was too large")
        try:
            raw = json.loads(completed.stdout)
            if (
                not isinstance(raw, dict)
                or raw.get("IsDir") is not False
                or isinstance(raw.get("Size"), bool)
                or not isinstance(raw.get("Size"), int)
                or raw["Size"] < 0
            ):
                raise ValueError
            modification_time = raw.get("ModTime")
            object_id = raw.get("ID")
            if modification_time is not None and not isinstance(modification_time, str):
                raise ValueError
            if object_id is not None and not isinstance(object_id, str):
                raise ValueError
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StorageUnavailableError("authoritative rclone stat response was invalid") from exc
        return _RemoteObjectStat(
            size_bytes=raw["Size"],
            modification_time=modification_time,
            object_id=object_id,
        )


class RcloneMountStorage:
    """A mounted logical root whose authority is an independently read remote."""

    def __init__(
        self,
        mountpoint: Path | str,
        authoritative_reader: AuthoritativeDigestReader,
        *,
        verification_timeout_seconds: float = 120,
        poll_interval_seconds: float = 0.25,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if verification_timeout_seconds <= 0:
            raise ValueError("verification timeout must be positive")
        if poll_interval_seconds < 0:
            raise ValueError("poll interval must be non-negative")
        self.root = Path(mountpoint).expanduser().resolve()
        self._reader = authoritative_reader
        self._timeout = verification_timeout_seconds
        self._poll_interval = poll_interval_seconds
        self._monotonic = monotonic
        self._sleep = sleep

    def healthcheck(self) -> bool:
        return self.root.is_dir() and self._reader.healthcheck()

    def resolve(self, logical_path: str) -> Path:
        return resolve_beneath(self.root, logical_path)

    def flush_and_verify(
        self,
        logical_path: str,
        *,
        expected_sha256: str | None = None,
    ) -> VerifiedObject:
        normalized = normalize_logical_path(logical_path)
        local_path = self.resolve(normalized)
        local_digest = fsync_and_digest(local_path)
        if expected_sha256 is not None and local_digest.sha256 != expected_sha256:
            raise StorageVerificationError("artifact checksum did not match the expected value")

        deadline = self._monotonic() + self._timeout
        last_observation: ObjectDigest | None = None
        while True:
            try:
                last_observation = self._reader.digest(normalized)
            except StorageUnavailableError:
                last_observation = None
            if last_observation == local_digest:
                return VerifiedObject.now(
                    logical_path=normalized,
                    local_path=local_path,
                    digest=local_digest,
                )
            if self._monotonic() >= deadline:
                detail = "missing" if last_observation is None else "different content"
                raise StorageVerificationError(
                    f"authoritative object was {detail} after the verification timeout"
                )
            self._sleep(self._poll_interval)


__all__ = ["RcloneCommandReader", "RcloneMountStorage"]
