"""Lifecycle supervision for a site-owned rclone VFS mount."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from topodrome.storage.rclone_environment import rclone_process_environment


class MountHealthReader(Protocol):
    def healthcheck(self) -> bool: ...


class RcloneMountError(RuntimeError):
    """A managed rclone mount could not be started or verified."""


class RcloneMountSupervisor:
    """Start, verify, and stop one exact rclone mount without a shell.

    Existing healthy mounts are accepted as externally supervised and are
    never unmounted by this object.  Only a subprocess created by this object
    may trigger cleanup of the mountpoint, which keeps unrelated FUSE mounts
    and user processes outside Topodrome's ownership boundary.
    """

    def __init__(
        self,
        *,
        remote: str,
        mountpoint: Path | str,
        cache_dir: Path | str,
        health_reader: MountHealthReader,
        binary: Path | str = "rclone",
        vfs_cache_max_size_gb: float = 20,
        min_free_space_gb: float = 10,
        write_back_seconds: float = 5,
        poll_interval_seconds: float = 5,
        startup_timeout_seconds: float = 30,
        is_mount: Callable[[str], bool] = os.path.ismount,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        if not remote or any(character in remote for character in "\r\n\0"):
            raise ValueError("rclone remote must be non-empty and contain no control bytes")
        for value, label in (
            (vfs_cache_max_size_gb, "VFS cache maximum"),
            (min_free_space_gb, "minimum free space"),
            (write_back_seconds, "write-back delay"),
            (poll_interval_seconds, "poll interval"),
            (startup_timeout_seconds, "startup timeout"),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        resolved_binary = shutil.which(str(binary))
        if resolved_binary is None:
            raise FileNotFoundError(f"rclone executable not found: {binary}")
        self.remote = remote.rstrip("/")
        self.mountpoint = Path(mountpoint).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if self.mountpoint == self.cache_dir or self.mountpoint in self.cache_dir.parents:
            raise ValueError("rclone cache directory must not be inside the mountpoint")
        self.binary = str(Path(resolved_binary).resolve())
        self.health_reader = health_reader
        self.vfs_cache_max_size_gb = vfs_cache_max_size_gb
        self.min_free_space_gb = min_free_space_gb
        self.write_back_seconds = write_back_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self._is_mount = is_mount
        self._monotonic = monotonic
        self._sleep = sleep
        self._popen = popen
        self._run = run
        self._process: subprocess.Popen[bytes] | None = None
        self._external = False

    @property
    def managed(self) -> bool:
        return self._process is not None

    @property
    def command(self) -> tuple[str, ...]:
        return (
            self.binary,
            "mount",
            self.remote,
            str(self.mountpoint),
            "--cache-dir",
            str(self.cache_dir),
            "--vfs-cache-mode",
            "full",
            "--vfs-cache-max-size",
            f"{self.vfs_cache_max_size_gb:g}G",
            "--vfs-cache-min-free-space",
            f"{self.min_free_space_gb:g}G",
            "--vfs-write-back",
            f"{self.write_back_seconds:g}s",
            "--poll-interval",
            f"{self.poll_interval_seconds:g}s",
        )

    def start(self) -> None:
        if self.healthcheck():
            return
        if self._is_mount(str(self.mountpoint)):
            raise RcloneMountError("an existing rclone mount is not authoritative or healthy")
        self.mountpoint.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._require_cache_capacity()
        try:
            self._process = self._popen(
                list(self.command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=rclone_process_environment(),
                start_new_session=True,
            )
        except OSError as exc:
            raise RcloneMountError("rclone mount process could not be started") from exc

        deadline = self._monotonic() + self.startup_timeout_seconds
        while self._monotonic() < deadline:
            assert self._process is not None
            status = self._process.poll()
            if status is not None:
                self._process = None
                raise RcloneMountError("rclone mount process exited during startup")
            if self.healthcheck():
                return
            self._sleep(min(0.1, self.startup_timeout_seconds))
        self.stop()
        raise RcloneMountError("rclone mount did not become healthy before timeout")

    def healthcheck(self) -> bool:
        mounted = self._is_mount(str(self.mountpoint))
        if not mounted:
            return False
        process = self._process
        if process is not None and process.poll() is not None:
            return False
        try:
            self._require_cache_capacity()
        except RcloneMountError:
            return False
        healthy = self.health_reader.healthcheck()
        if healthy and process is None:
            self._external = True
        return healthy

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if self._is_mount(str(self.mountpoint)):
            fusermount = shutil.which("fusermount3") or shutil.which("fusermount")
            if fusermount is not None:
                self._run(
                    [fusermount, "-u", str(self.mountpoint)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=rclone_process_environment(),
                    timeout=15,
                    check=False,
                )

    def _require_cache_capacity(self) -> None:
        anchor = self.cache_dir if self.cache_dir.exists() else self.cache_dir.parent
        while not anchor.exists() and anchor != anchor.parent:
            anchor = anchor.parent
        try:
            free = shutil.disk_usage(anchor).free
        except OSError as exc:
            raise RcloneMountError("rclone cache filesystem is unavailable") from exc
        required = round(self.min_free_space_gb * 1024**3)
        if free < required:
            raise RcloneMountError("rclone cache filesystem is below minimum free space")


__all__ = ["RcloneMountError", "RcloneMountSupervisor"]
