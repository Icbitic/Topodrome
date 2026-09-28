from __future__ import annotations

import hashlib
import json
import subprocess
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from topodrome.storage import (
    ObjectDigest,
    RcloneCommandReader,
    RcloneMountStorage,
    StorageUnavailableError,
    StorageVerificationError,
)
from topodrome.storage.rclone_environment import rclone_process_environment

pytestmark = pytest.mark.unit


class ManualTime:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, delay: float) -> None:
        self.value += max(delay, 0.01)


class ScriptedRemote:
    def __init__(self, observations: list[ObjectDigest | None], *, healthy: bool = True) -> None:
        self.observations = deque(observations)
        self.healthy = healthy
        self.paths: list[str] = []

    def healthcheck(self) -> bool:
        return self.healthy

    def digest(self, logical_path: str) -> ObjectDigest | None:
        self.paths.append(logical_path)
        if len(self.observations) > 1:
            return self.observations.popleft()
        return self.observations[0] if self.observations else None


class HashsumRclone:
    def __init__(self, *, size_bytes: int, sha256: str) -> None:
        self.size_bytes = size_bytes
        self.sha256 = sha256
        self.modification_times = deque(["2026-09-13T00:00:00Z"])
        self.commands: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []

    def __call__(
        self,
        args: Sequence[str],
        **options: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        command = tuple(args)
        self.commands.append(command)
        self.timeouts.append(options["timeout"])
        if command[1] == "lsjson":
            modification_time = (
                self.modification_times.popleft()
                if len(self.modification_times) > 1
                else self.modification_times[0]
            )
            payload = {
                "IsDir": False,
                "Size": self.size_bytes,
                "ModTime": modification_time,
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(payload).encode(), b"")
        if command[1] == "hashsum":
            return subprocess.CompletedProcess(
                args,
                0,
                f"{self.sha256}  large.bin\n".encode(),
                b"",
            )
        raise AssertionError(f"unexpected fake rclone operation: {command[1]}")


def test_rclone_waits_for_independent_authoritative_match(tmp_path: Path) -> None:
    mountpoint = tmp_path / "mount"
    path = mountpoint / "jobs/job-1/result"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"durable result")
    correct = ObjectDigest(
        size_bytes=len(b"durable result"),
        sha256=hashlib.sha256(b"durable result").hexdigest(),
    )
    wrong = ObjectDigest(size_bytes=3, sha256=hashlib.sha256(b"old").hexdigest())
    remote = ScriptedRemote([None, wrong, correct])
    clock = ManualTime()
    storage = RcloneMountStorage(
        mountpoint,
        remote,
        verification_timeout_seconds=2,
        poll_interval_seconds=0.25,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    verified = storage.flush_and_verify("jobs/job-1/result")

    assert verified.digest == correct
    assert remote.paths == ["jobs/job-1/result"] * 3
    assert clock.value == 0.5


def test_local_close_is_not_treated_as_rclone_durability(tmp_path: Path) -> None:
    mountpoint = tmp_path / "mount"
    path = mountpoint / "artifact"
    mountpoint.mkdir()
    path.write_bytes(b"only cached")
    clock = ManualTime()
    storage = RcloneMountStorage(
        mountpoint,
        ScriptedRemote([None]),
        verification_timeout_seconds=0.5,
        poll_interval_seconds=0.25,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    with pytest.raises(StorageVerificationError, match="authoritative object was missing"):
        storage.flush_and_verify("artifact")


def test_rclone_health_requires_mount_and_authoritative_remote(tmp_path: Path) -> None:
    mountpoint = tmp_path / "mount"
    mountpoint.mkdir()
    remote = ScriptedRemote([], healthy=False)
    assert not RcloneMountStorage(mountpoint, remote).healthcheck()


def test_rclone_environment_does_not_inherit_other_site_credentials() -> None:
    environment = rclone_process_environment(
        {
            "PATH": "/runtime/bin",
            "RCLONE_CONFIG": "/protected/rclone.conf",
            "RCLONE_CONFIG_LAB1_PASS": "rclone-secret",
            "TOPODROME_SITE_ENROLLMENT_TOKEN": "agent-secret",
            "TOPODROME_SITE_FENCE_VERIFICATION_KEY": "fence-secret",
            "TOPODROME_HF_TOKEN": "model-secret",
            "TOPODROME_AUTODL_TOKEN": "provider-secret",
        }
    )

    assert environment == {
        "PATH": "/runtime/bin",
        "RCLONE_CONFIG": "/protected/rclone.conf",
        "RCLONE_CONFIG_LAB1_PASS": "rclone-secret",
    }


def test_authoritative_digest_streams_inside_rclone_with_long_transfer_timeout() -> None:
    size_bytes = 5 * 1024**3
    checksum = "a" * 64
    runner = HashsumRclone(size_bytes=size_bytes, sha256=checksum)
    reader = RcloneCommandReader(
        "lab1:/global",
        binary="/usr/bin/true",
        command_timeout_seconds=3,
        transfer_timeout_seconds=999,
        run=runner,
    )

    observed = reader.digest("artifacts/large.bin")

    assert observed == ObjectDigest(size_bytes=size_bytes, sha256=checksum)
    assert [command[1] for command in runner.commands] == ["lsjson", "hashsum", "lsjson"]
    assert runner.commands[1][-1] == "--download"
    assert all(command[1] != "cat" for command in runner.commands)
    assert runner.timeouts == [3, 999, 3]


def test_authoritative_digest_rejects_object_changed_during_hash() -> None:
    runner = HashsumRclone(size_bytes=1024, sha256="b" * 64)
    runner.modification_times = deque(["2026-09-13T00:00:00Z", "2026-09-13T00:01:00Z"])
    reader = RcloneCommandReader(
        "lab1:/global",
        binary="/usr/bin/true",
        run=runner,
    )

    with pytest.raises(StorageUnavailableError, match="changed while hashing"):
        reader.digest("artifacts/large.bin")
