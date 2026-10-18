from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from topodrome.storage import AttemptArtifactStore, ObjectDigest, resolve_committed_artifact
from topodrome.storage.base import StorageUnavailableError
from topodrome.storage.http_mirror import HttpMirrorStorage

pytestmark = pytest.mark.unit


class FakeAuthority:
    """In-memory byte authority; no commands, processes or protocol fallback."""

    def __init__(self):
        self.remote_root = "lab1-authority"
        self.objects = {}
        self.calls = []
        self.reads = []
        self.writes = []
        self.digest_observations = None
        self.healthy = True

    def close(self):
        pass

    def healthcheck(self):
        return self.healthy

    def stat(self, logical):
        self.calls.append(("stat", logical))
        content = self.objects.get(logical)
        if content is not None:
            return {"IsDir": False, "Size": len(content), "ID": hashlib.sha256(content).hexdigest()}
        if any(path.startswith(logical + "/") for path in self.objects):
            return {"IsDir": True, "Size": 0, "ID": "d" * 64}
        return None

    def digest(self, logical):
        if self.digest_observations is not None:
            if len(self.digest_observations) > 1:
                return self.digest_observations.pop(0)
            return self.digest_observations[0]
        content = self.objects.get(logical)
        return (
            None
            if content is None
            else ObjectDigest(len(content), hashlib.sha256(content).hexdigest())
        )

    def read(self, logical, output, offset, count, *, expected_identity=None):
        self.calls.append(("read", logical))
        self.reads.append((logical, offset, count))
        output.write(self.objects[logical][offset : offset + count])

    def write(self, logical, source, offset, size):
        self.calls.append(("write", logical))
        self.writes.append((logical, offset, size))
        self.objects[logical] = (
            self.objects.get(logical, b"")[:offset] + source.read_bytes()[offset:size]
        )

    def rename(self, source, destination):
        self.calls.append(("rename", source))
        self.objects[destination] = self.objects.pop(source)

    def remove(self, logical):
        self.calls.append(("remove", logical))
        self.objects.pop(logical, None)

    def download_tree(self, logical, destination):
        for name, content in self.objects.items():
            if name.startswith(logical + "/"):
                target = destination / name[len(logical) + 1 :]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)


class ManualTime:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += max(seconds, 0.01)


def _storage(tmp_path, remote):
    clock = ManualTime()
    return HttpMirrorStorage(
        tmp_path / "cache",
        remote.remote_root,
        authoritative_reader=remote,
        bulk_transport=remote,
        min_free_space_gb=1e-7,
        healthcheck_cache_seconds=0,
        verification_timeout_seconds=0.5,
        poll_interval_seconds=0.25,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


def test_commit_transfer_phases_and_verification_barrier(tmp_path):
    remote = FakeAuthority()
    storage = _storage(tmp_path, remote)
    store = AttemptArtifactStore(storage, job_id="job-1", attempt_id="attempt-1")
    store.stage_bytes("a", b"abc")
    store.stage_bytes("b", b"defgh")
    events = []
    store.commit(["b", "a"], progress=lambda *args: events.append(args))
    assert ("upload", 0, 8) in events
    assert events[-1] == ("manifest", 8, 8)
    assert store.published_manifest_path in remote.objects
    store = AttemptArtifactStore(storage, job_id="job-2", attempt_id="attempt-2")
    store.stage_bytes("a", b"abc")
    remote.digest_observations = [None]
    with pytest.raises(StorageUnavailableError):
        store.commit(["a"])
    assert store.published_manifest_path not in remote.objects


def test_committed_outputs_round_trip_through_a_fresh_cache(tmp_path):
    remote = FakeAuthority()
    storage = _storage(tmp_path / "first", remote)
    store = AttemptArtifactStore(storage, job_id="job", attempt_id="attempt")
    store.stage_bytes("results/output", b"durable")
    store.commit(["results/output"])
    fresh = _storage(tmp_path / "second", remote)
    result = resolve_committed_artifact(fresh, job_id="job", logical_path="results/output")
    assert result.read_bytes() == b"durable"


def test_individually_cached_child_does_not_claim_a_complete_tree(tmp_path):
    remote = FakeAuthority()
    remote.objects.update({"artifacts/tree/a": b"a", "artifacts/tree/b": b"b"})
    storage = _storage(tmp_path, remote)
    storage.resolve("artifacts/tree/a")
    whole = storage.resolve("artifacts/tree")
    assert {p.name for p in whole.iterdir()} == {"a", "b"}


def test_exact_path_locks_are_reentrant_and_reclaimed(tmp_path):
    storage = _storage(tmp_path, FakeAuthority())
    first = storage._path_lock("jobs/a")
    assert storage._path_lock("jobs/a") is first
    with first, first:
        assert storage._path_lock("jobs/b") is not first
    del first
    assert len(storage._path_locks) == 0


def test_slow_output_upload_does_not_block_an_unrelated_input(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class SlowAuthority(FakeAuthority):
        def write(self, *args):
            entered.set()
            assert release.wait(3)
            return super().write(*args)

    remote = SlowAuthority()
    remote.objects["artifacts/input"] = b"fast"
    storage = _storage(tmp_path, remote)
    output = storage.resolve("jobs/test/attempts/one/committed/output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"slow")
    with ThreadPoolExecutor(max_workers=2) as workers:
        upload = workers.submit(storage.flush_and_verify, "jobs/test/attempts/one/committed/output")
        try:
            assert entered.wait(2)
            download = workers.submit(storage.resolve, "artifacts/input")
            assert download.result(timeout=2).read_bytes() == b"fast"
        finally:
            release.set()
        upload.result(timeout=3)
