from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from topodrome.storage import (
    AttemptArtifactStore,
    ObjectDigest,
    RcloneMirrorStorage,
    StorageAdapter,
    StorageUnavailableError,
    StorageVerificationError,
    resolve_committed_artifact,
)

pytestmark = pytest.mark.unit


def test_commit_transfer_phases_and_verification_barrier(tmp_path: Path) -> None:
    remote = FakeRclone()
    storage = _storage(tmp_path, remote)
    store = AttemptArtifactStore(storage, job_id="job-1", attempt_id="attempt-1")
    store.stage_bytes("a", b"abc")
    store.stage_bytes("b", b"defgh")
    events = []
    store.commit(["b", "a"], progress=lambda *args: events.append(args))
    assert ("upload", 0, 8) in events
    assert ("upload", 3, 8) in events
    assert ("promote", 8, 8) in events
    assert events[-1] == ("manifest", 8, 8)
    assert "jobs/job-1/committed/manifest.json" in remote.objects

    # Upload reaching 100% must not bypass the independent digest check.
    store = AttemptArtifactStore(storage, job_id="job-2", attempt_id="attempt-2")
    store.stage_bytes("a", b"abc")
    remote.digest_observations = [None]
    events.clear()
    with pytest.raises(StorageUnavailableError):
        store.commit(["a"], progress=lambda *args: events.append(args))
    assert ("promote", 3, 3) in events
    assert not any(stage == "manifest" for stage, _, _ in events)
    assert "jobs/job-2/committed/manifest.json" not in remote.objects


class ManualTime:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, delay: float) -> None:
        self.value += max(delay, 0.01)


class FakeRclone:
    def __init__(self, *, remote_root: str = "lab1:/global") -> None:
        self.remote_root = remote_root
        self.objects: dict[str, bytes] = {}
        self.commands: list[tuple[str, ...]] = []
        self.healthy = True
        self.fail_operation: str | None = None
        self.cat_fail_after_bytes: int | None = None
        self.modification_time = "2026-09-13T00:00:00Z"
        self.file_size_override: int | None = None
        self.digest_observations: list[ObjectDigest | None] | None = None
        self.timeouts: list[float | None] = []

    def healthcheck(self) -> bool:
        return self.healthy

    def digest(self, logical_path: str) -> ObjectDigest | None:
        if self.digest_observations is not None:
            if len(self.digest_observations) > 1:
                return self.digest_observations.pop(0)
            return self.digest_observations[0]
        content = self.objects.get(logical_path)
        if content is None:
            return None
        return ObjectDigest(
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )

    def __call__(
        self,
        args: Sequence[str],
        **options: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        command = tuple(args)
        self.commands.append(command)
        self.timeouts.append(options.get("timeout"))
        operation = command[1]
        if operation == self.fail_operation:
            return subprocess.CompletedProcess(args, 1, b"", b"")
        if operation == "lsjson":
            return self._stat(command[2], args)
        if operation == "copy":
            return self._copy_directory(command[2], Path(command[3]), args)
        if operation == "copyto":
            return self._copy_file(command[2], command[3], args)
        if operation == "cat":
            return self._cat(command, options["stdout"], args)
        if operation == "moveto":
            source = self._logical(command[2])
            destination = self._logical(command[3])
            content = self.objects.pop(source, None)
            if content is None:
                return subprocess.CompletedProcess(args, 3, b"", b"")
            self.objects[destination] = content
            return subprocess.CompletedProcess(args, 0, b"", b"")
        if operation == "deletefile":
            self.objects.pop(self._logical(command[2]), None)
            return subprocess.CompletedProcess(args, 0, b"", b"")
        raise AssertionError(f"unexpected fake rclone operation: {operation}")

    def _stat(
        self,
        remote_path: str,
        args: Sequence[str],
    ) -> subprocess.CompletedProcess[bytes]:
        logical = self._logical(remote_path)
        content = self.objects.get(logical)
        if content is not None:
            payload = {
                "IsDir": False,
                "Size": (
                    len(content) if self.file_size_override is None else self.file_size_override
                ),
                "ModTime": self.modification_time,
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(payload).encode(), b"")
        prefix = f"{logical}/"
        if any(name.startswith(prefix) for name in self.objects):
            payload = {"IsDir": True, "Size": -1}
            return subprocess.CompletedProcess(args, 0, json.dumps(payload).encode(), b"")
        return subprocess.CompletedProcess(args, 3, b"", b"")

    def _copy_directory(
        self,
        remote_path: str,
        destination: Path,
        args: Sequence[str],
    ) -> subprocess.CompletedProcess[bytes]:
        logical = self._logical(remote_path)
        prefix = f"{logical}/"
        for name, content in self.objects.items():
            if not name.startswith(prefix):
                continue
            target = destination / name.removeprefix(prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return subprocess.CompletedProcess(args, 0, b"", b"")

    def _cat(
        self,
        command: tuple[str, ...],
        output: Any,
        args: Sequence[str],
    ) -> subprocess.CompletedProcess[bytes]:
        content = self.objects.get(self._logical(command[2]))
        if content is None:
            return subprocess.CompletedProcess(args, 3, b"", b"")
        offset = int(command[command.index("--offset") + 1])
        count = int(command[command.index("--count") + 1])
        body = content[offset : offset + count]
        if self.cat_fail_after_bytes is not None:
            body = body[: self.cat_fail_after_bytes]
            output.write(body)
            self.cat_fail_after_bytes = None
            return subprocess.CompletedProcess(args, 1, b"", b"")
        output.write(body)
        return subprocess.CompletedProcess(args, 0, b"", b"")

    def _copy_file(
        self,
        source: str,
        destination: str,
        args: Sequence[str],
    ) -> subprocess.CompletedProcess[bytes]:
        if source.startswith(f"{self.remote_root}/"):
            content = self.objects.get(self._logical(source))
            if content is None:
                return subprocess.CompletedProcess(args, 3, b"", b"")
            target = Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        else:
            self.objects[self._logical(destination)] = Path(source).read_bytes()
        return subprocess.CompletedProcess(args, 0, b"", b"")

    def _logical(self, remote_path: str) -> str:
        prefix = f"{self.remote_root}/"
        if not remote_path.startswith(prefix):
            raise AssertionError(f"path is outside fake remote: {remote_path}")
        return remote_path.removeprefix(prefix)


def _storage(
    tmp_path: Path,
    remote: FakeRclone,
    *,
    clock: ManualTime | None = None,
) -> RcloneMirrorStorage:
    clock = clock or ManualTime()
    return RcloneMirrorStorage(
        tmp_path / "cache",
        remote.remote_root,
        binary="/usr/bin/true",
        min_free_space_gb=0.0000001,
        healthcheck_cache_seconds=0,
        verification_timeout_seconds=0.5,
        transfer_timeout_seconds=7,
        poll_interval_seconds=0.25,
        authoritative_reader=remote,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        run=remote,
    )


def test_checkpoint_lock_stripes_do_not_block_diagnostic_logs(tmp_path: Path) -> None:
    storage = _storage(tmp_path, FakeRclone())
    log = "jobs/job-1/attempts/attempt-1/logs/attempt.log"
    stripe = hash(log) % 64
    checkpoint = next(
        f"jobs/job-1/attempts/attempt-1/checkpoints/{i}.pkl"
        for i in range(10000)
        if hash(f"jobs/job-1/attempts/attempt-1/checkpoints/{i}.pkl") % 64 == stripe
    )
    assert storage._path_lock(log) is not storage._path_lock(checkpoint)
    assert storage._path_lock(log) is storage._path_lock(log)


@pytest.mark.parametrize("target", ["projects/.topodrome-sources/source", "assets/input.bin"])
def test_slow_checkpoint_does_not_block_colliding_download(tmp_path: Path, target: str) -> None:
    uploading = threading.Event()
    release = threading.Event()

    class SlowCheckpoint(FakeRclone):
        def __call__(
            self, args: Sequence[str], **options: Any
        ) -> subprocess.CompletedProcess[bytes]:
            if args[1] == "copyto":
                uploading.set()
                assert release.wait(5), "test did not release checkpoint upload"
            return super().__call__(args, **options)

    remote = SlowCheckpoint()
    remote.objects[target] = b"source or asset"
    storage = _storage(tmp_path, remote)
    checkpoint = next(
        f"jobs/old/attempts/old/checkpoints/{i}.pkl"
        for i in range(10000)
        if hash(f"jobs/old/attempts/old/checkpoints/{i}.pkl") % 64 == hash(target) % 64
    )
    local = storage.root / checkpoint
    local.parent.mkdir(parents=True)
    local.write_bytes(b"checkpoint")
    with ThreadPoolExecutor(max_workers=2) as pool:
        upload = pool.submit(storage.flush_and_verify, checkpoint)
        try:
            assert uploading.wait(2)
            download = pool.submit(storage.resolve, target)
            assert download.result(timeout=2).read_bytes() == b"source or asset"
            assert not upload.done()
        finally:
            release.set()
        assert upload.result(timeout=2).logical_path == checkpoint


def test_exact_path_lock_serializes_waiters_and_reclaims_unused_entries(tmp_path: Path) -> None:
    storage = _storage(tmp_path, FakeRclone())
    first = storage._path_lock("assets/same")
    first_id = id(first)
    assert storage._path_lock("assets/same") is first
    entered = threading.Event()
    acquired = threading.Event()

    def wait_for_lock() -> None:
        lock = storage._path_lock("assets/same")
        assert id(lock) == first_id
        entered.set()
        with lock:
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with first:
            pending = pool.submit(wait_for_lock)
            assert entered.wait(2)
            assert not acquired.is_set()
        pending.result(timeout=2)
    assert acquired.is_set()
    del first
    assert not storage._path_locks


def test_path_lock_is_reentrant_and_released_after_exception(tmp_path: Path) -> None:
    storage = _storage(tmp_path, FakeRclone())
    lock = storage._path_lock("assets/same")
    with pytest.raises(RuntimeError), lock, storage._path_lock("assets/same"):
        raise RuntimeError("transfer failed")
    with ThreadPoolExecutor(max_workers=1) as pool:

        def acquire_again() -> None:
            with storage._path_lock("assets/same"):
                pass

        pool.submit(acquire_again).result(timeout=2)


def test_missing_remote_file_is_downloaded_atomically(tmp_path: Path) -> None:
    remote = FakeRclone()
    remote.objects["assets/tiny/v1/data.json"] = b'{"answer": 42}\n'
    storage = _storage(tmp_path, remote)

    resolved = storage.resolve("assets/tiny/v1/data.json")

    assert resolved.read_bytes() == b'{"answer": 42}\n'
    assert [command[1] for command in remote.commands] == ["lsjson", "cat", "lsjson"]
    assert remote.timeouts == [120, 7, 120]
    assert not list(resolved.parent.glob(".*.topodrome-download-*"))
    assert isinstance(storage, StorageAdapter)


def test_interrupted_file_download_resumes_from_deterministic_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = FakeRclone()
    content = b"0123456789"
    remote.objects["artifacts/large.bin"] = content
    remote.cat_fail_after_bytes = 4
    storage = _storage(tmp_path, remote)

    with pytest.raises(StorageUnavailableError, match="download failed"):
        storage.resolve("artifacts/large.bin")

    partials = list(storage._partial_root.glob("*.part"))
    assert len(partials) == 1
    assert partials[0].read_bytes() == content[:4]

    # Existing partial bytes already consume disk. Admission therefore reserves
    # exactly the six bytes still needed, in addition to the configured floor.
    floor = round(0.0000001 * 1024**3)
    usage = shutil_disk_usage(total=10_000, used=10_000 - floor - 6, free=floor + 6)
    monkeypatch.setattr("topodrome.storage.rclone_mirror.shutil.disk_usage", lambda _: usage)
    remote.commands.clear()

    resolved = storage.resolve("artifacts/large.bin")

    assert resolved.read_bytes() == content
    cat_command = next(command for command in remote.commands if command[1] == "cat")
    assert cat_command[cat_command.index("--offset") + 1] == "4"
    assert cat_command[cat_command.index("--count") + 1] == "6"
    assert not list(storage._partial_root.glob("*.part"))
    assert not list(storage._partial_root.glob("*.json"))


def test_changed_remote_identity_discards_partial_instead_of_splicing_versions(
    tmp_path: Path,
) -> None:
    remote = FakeRclone()
    logical_path = "artifacts/large.bin"
    remote.objects[logical_path] = b"old-version"
    remote.cat_fail_after_bytes = 3
    storage = _storage(tmp_path, remote)

    with pytest.raises(StorageUnavailableError):
        storage.resolve(logical_path)

    remote.objects[logical_path] = b"new-version"
    remote.modification_time = "2026-09-13T00:01:00Z"
    remote.commands.clear()

    resolved = storage.resolve(logical_path)

    assert resolved.read_bytes() == b"new-version"
    cat_command = next(command for command in remote.commands if command[1] == "cat")
    assert cat_command[cat_command.index("--offset") + 1] == "0"


def test_file_download_requires_exact_remote_size_above_free_space_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = FakeRclone()
    remote.objects["artifacts/large.bin"] = b"0123456789"
    storage = _storage(tmp_path, remote)
    floor = round(0.0000001 * 1024**3)
    usage = shutil_disk_usage(total=10_000, used=10_000 - floor - 9, free=floor + 9)
    monkeypatch.setattr("topodrome.storage.rclone_mirror.shutil.disk_usage", lambda _: usage)

    with pytest.raises(StorageUnavailableError, match="below minimum free space"):
        storage.resolve("artifacts/large.bin")

    assert [command[1] for command in remote.commands] == ["lsjson"]


def test_file_download_rejects_unknown_or_negative_authoritative_size(
    tmp_path: Path,
) -> None:
    remote = FakeRclone()
    remote.objects["artifacts/large.bin"] = b"content"
    remote.file_size_override = -1
    storage = _storage(tmp_path, remote)

    with pytest.raises(StorageUnavailableError, match="stat response was invalid"):
        storage.resolve("artifacts/large.bin")


def test_missing_remote_directory_is_downloaded_recursively(tmp_path: Path) -> None:
    remote = FakeRclone()
    remote.objects["projects/.topodrome-sources/abc/manifest.json"] = b"{}"
    remote.objects["projects/.topodrome-sources/abc/package/main.py"] = b"answer = 42\n"
    storage = _storage(tmp_path, remote)

    source = storage.resolve("projects/.topodrome-sources/abc")

    assert (source / "manifest.json").read_bytes() == b"{}"
    assert (source / "package/main.py").read_bytes() == b"answer = 42\n"
    assert [command[1] for command in remote.commands] == ["lsjson", "copy"]


def test_remote_miss_returns_safe_local_write_path(tmp_path: Path) -> None:
    remote = FakeRclone()
    storage = _storage(tmp_path, remote)

    target = storage.resolve("jobs/job-1/attempts/attempt-1/staging/result.json")

    assert target == storage.root / "jobs/job-1/attempts/attempt-1/staging/result.json"
    assert not target.exists()


def test_cached_object_can_be_used_without_an_extra_remote_command(tmp_path: Path) -> None:
    remote = FakeRclone()
    storage = _storage(tmp_path, remote)
    target = storage.root / "assets/tiny/v1"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"cached")

    assert storage.resolve("assets/tiny/v1") == target
    assert remote.commands == []


def test_resolve_pinned_replaces_a_wrong_sized_regular_cache_file(tmp_path: Path) -> None:
    remote = FakeRclone()
    logical_path = "jobs/job-1/committed/model.bin"
    remote.objects[logical_path] = b"authoritative"
    storage = _storage(tmp_path, remote)
    cached = storage.root / logical_path
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"stale")

    resolved = storage.resolve_pinned(
        logical_path,
        expected_size_bytes=len(remote.objects[logical_path]),
    )

    assert resolved.read_bytes() == b"authoritative"
    assert [command[1] for command in remote.commands] == ["lsjson", "cat", "lsjson"]


def test_resolve_pinned_rejects_wrong_remote_size_before_download(tmp_path: Path) -> None:
    remote = FakeRclone()
    logical_path = "jobs/job-1/committed/model.bin"
    remote.objects[logical_path] = b"unexpectedly large payload"
    storage = _storage(tmp_path, remote)

    with pytest.raises(StorageVerificationError, match="size"):
        storage.resolve_pinned(logical_path, expected_size_bytes=4)

    assert [command[1] for command in remote.commands] == ["lsjson"]
    assert not (storage.root / logical_path).exists()


def test_resolve_pinned_rejects_an_authoritative_directory(tmp_path: Path) -> None:
    remote = FakeRclone()
    remote.objects["jobs/job-1/committed/tree/member"] = b"content"
    storage = _storage(tmp_path, remote)

    with pytest.raises(StorageVerificationError, match="regular file"):
        storage.resolve_pinned("jobs/job-1/committed/tree", expected_size_bytes=7)

    assert [command[1] for command in remote.commands] == ["lsjson"]


def test_invalidate_cached_removes_only_regular_non_symlink_files(tmp_path: Path) -> None:
    remote = FakeRclone()
    storage = _storage(tmp_path, remote)
    regular = storage.root / "assets/model.bin"
    regular.parent.mkdir(parents=True)
    regular.write_bytes(b"bad cache")

    storage.invalidate_cached("assets/model.bin")
    storage.invalidate_cached("assets/absent.bin")

    assert not regular.exists()
    directory = storage.root / "assets/tree"
    directory.mkdir()
    with pytest.raises(StorageVerificationError, match="regular"):
        storage.invalidate_cached("assets/tree")
    assert directory.is_dir()

    target = storage.root / "assets/target.bin"
    target.write_bytes(b"keep")
    symbolic_link = storage.root / "assets/link.bin"
    symbolic_link.symlink_to(target.name)
    with pytest.raises(StorageVerificationError, match="regular"):
        storage.invalidate_cached("assets/link.bin")
    assert symbolic_link.is_symlink()
    assert target.read_bytes() == b"keep"


def test_non_not_found_stat_failure_is_not_treated_as_a_writable_miss(
    tmp_path: Path,
) -> None:
    remote = FakeRclone()
    remote.fail_operation = "lsjson"
    storage = _storage(tmp_path, remote)

    with pytest.raises(StorageUnavailableError, match="stat failed"):
        storage.resolve("jobs/job-1/new-output")


def test_flush_uploads_then_promotes_and_verifies_authority(tmp_path: Path) -> None:
    remote = FakeRclone()
    storage = _storage(tmp_path, remote)
    target = storage.resolve("jobs/job-1/attempts/attempt-1/committed/result.json")
    target.parent.mkdir(parents=True)
    target.write_bytes(b'{"ok": true}\n')

    verified = storage.flush_and_verify("jobs/job-1/attempts/attempt-1/committed/result.json")

    assert verified.digest.sha256 == hashlib.sha256(b'{"ok": true}\n').hexdigest()
    assert (
        remote.objects["jobs/job-1/attempts/attempt-1/committed/result.json"] == b'{"ok": true}\n'
    )
    operations = [command[1] for command in remote.commands]
    assert operations[-2:] == ["copyto", "moveto"]
    assert remote.commands[-1][-1] == "--ignore-times"
    assert remote.timeouts[-2:] == [7, 120]
    assert not any("topodrome-upload" in name for name in remote.objects)


def test_flush_rejects_expected_checksum_before_upload(tmp_path: Path) -> None:
    remote = FakeRclone()
    storage = _storage(tmp_path, remote)
    target = storage.resolve("jobs/job-1/result")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"actual")
    remote.commands.clear()

    with pytest.raises(StorageVerificationError, match="checksum"):
        storage.flush_and_verify("jobs/job-1/result", expected_sha256="0" * 64)

    assert remote.commands == []


def test_flush_materializes_a_missing_authoritative_checkpoint_first(
    tmp_path: Path,
) -> None:
    remote = FakeRclone()
    content = b"portable checkpoint"
    logical_path = "jobs/job-1/attempts/attempt-1/checkpoints/0001.pkl"
    remote.objects[logical_path] = content
    storage = _storage(tmp_path, remote)

    verified = storage.flush_and_verify(
        logical_path,
        expected_sha256=hashlib.sha256(content).hexdigest(),
    )

    assert verified.local_path.read_bytes() == content
    assert [command[1] for command in remote.commands] == ["lsjson", "cat", "lsjson"]


def test_failed_promotion_attempts_to_remove_only_its_remote_temporary(
    tmp_path: Path,
) -> None:
    remote = FakeRclone()
    storage = _storage(tmp_path, remote)
    target = storage.resolve("jobs/job-1/result")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"result")
    remote.commands.clear()
    remote.fail_operation = "moveto"

    with pytest.raises(StorageVerificationError, match="promotion"):
        storage.flush_and_verify("jobs/job-1/result")

    assert [command[1] for command in remote.commands] == [
        "copyto",
        "moveto",
        "deletefile",
    ]
    assert not any("topodrome-upload" in name for name in remote.objects)


def test_authoritative_mismatch_fails_after_bounded_verification(tmp_path: Path) -> None:
    remote = FakeRclone()
    clock = ManualTime()
    storage = _storage(tmp_path, remote, clock=clock)
    target = storage.resolve("jobs/job-1/result")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"result")
    wrong = ObjectDigest(size_bytes=3, sha256=hashlib.sha256(b"old").hexdigest())
    remote.digest_observations = [wrong]

    with pytest.raises(StorageVerificationError, match="different content"):
        storage.flush_and_verify("jobs/job-1/result")

    assert clock.value == 0.5


def test_healthcheck_requires_both_remote_and_cache_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = FakeRclone()
    storage = _storage(tmp_path, remote)
    assert storage.healthcheck()

    remote.healthy = False
    assert not storage.healthcheck()

    remote.healthy = True
    usage = shutil_disk_usage(total=1000, used=999, free=1)
    monkeypatch.setattr("topodrome.storage.rclone_mirror.shutil.disk_usage", lambda _: usage)
    assert not storage.healthcheck()


class shutil_disk_usage(NamedTuple):
    total: int
    used: int
    free: int


@pytest.mark.parametrize(
    "remote_root",
    ["--config=attacker", ":sftp,host=attacker:/global", "relative/path", "lab1:../global"],
)
def test_remote_root_must_be_a_named_safe_remote(
    tmp_path: Path,
    remote_root: str,
) -> None:
    with pytest.raises(ValueError, match="named rclone remote"):
        RcloneMirrorStorage(tmp_path / "cache", remote_root)


def test_logical_paths_cannot_escape_the_cache(tmp_path: Path) -> None:
    remote = FakeRclone()
    storage = _storage(tmp_path, remote)
    with pytest.raises(ValueError):
        storage.resolve("../outside")
    with pytest.raises(ValueError, match="reserved namespace"):
        storage.resolve(".topodrome-mirror-partials/foreign")


def test_named_remote_root_can_be_scoped_by_the_sftp_server(tmp_path: Path) -> None:
    storage = RcloneMirrorStorage(
        tmp_path / "cache",
        "lab1-global:",
        binary="/usr/bin/true",
        min_free_space_gb=0.0000001,
        authoritative_reader=FakeRclone(),
    )

    assert storage.remote_root == "lab1-global:"


def test_committed_artifacts_round_trip_through_a_fresh_site_cache(
    tmp_path: Path,
) -> None:
    remote = FakeRclone()
    writer = _storage(tmp_path / "writer", remote)
    store = AttemptArtifactStore(writer, job_id="job-1", attempt_id="attempt-1")
    store.stage_bytes("result.json", b'{"answer": 42}\n')

    manifest = store.commit(["result.json"])

    assert (
        manifest.artifact("result.json").sha256 == hashlib.sha256(b'{"answer": 42}\n').hexdigest()
    )
    reader = _storage(tmp_path / "reader", remote)
    artifact = resolve_committed_artifact(
        reader,
        job_id="job-1",
        logical_path="result.json",
    )
    assert artifact.read_bytes() == b'{"answer": 42}\n'
