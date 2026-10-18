import threading
import time

import pytest

from topodrome.runtime.model_key import ModelKey
from topodrome.runtime.model_payload import verify_transformers_payload
from topodrome.runtime.model_prefetch import ModelPrefetcher
from topodrome.runtime.model_registry import ModelRegistration


def checkpoint(tmp_path, revision="first"):
    root = tmp_path / revision
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"x" * 4096)
    key = ModelKey("test/model", revision, "transformers", "configured")
    verified = verify_transformers_payload(root, revision=revision)
    return ModelRegistration(key, local_path=root, payload_identity=verified.identity_digest)


def drained(prefetch):
    deadline = time.monotonic() + 3
    while prefetch.active and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not prefetch.active


def test_read_ahead_reuses_existing_files_and_is_deduplicated(tmp_path):
    registration = checkpoint(tmp_path)
    before = {p.name: p.read_bytes() for p in registration.local_path.iterdir()}
    prefetch = ModelPrefetcher(max_bytes=8192, memory_floor_bytes=1, memory_available=lambda: 8192)
    assert prefetch.schedule(registration)
    drained(prefetch)
    assert prefetch.bytes_read == 4096
    assert not prefetch.schedule(registration)
    assert {p.name: p.read_bytes() for p in registration.local_path.iterdir()} == before
    prefetch.close()


@pytest.mark.parametrize("case", ["disabled", "memory", "oversized", "unverified"])
def test_read_ahead_is_optional_and_memory_byte_bounded(tmp_path, case):
    registration = checkpoint(tmp_path)
    if case == "unverified":
        registration = ModelRegistration(registration.key, local_path=registration.local_path)
    prefetch = ModelPrefetcher(
        max_bytes=0 if case == "disabled" else (1024 if case == "oversized" else 8192),
        memory_floor_bytes=1,
        memory_available=lambda: 0 if case == "memory" else 8192,
    )
    prefetch.schedule(registration)
    drained(prefetch)
    assert prefetch.bytes_read == 0
    prefetch.close()


def test_site_wide_concurrency_is_bounded_and_close_cancels(tmp_path, monkeypatch):
    first, second = checkpoint(tmp_path), checkpoint(tmp_path, "second")
    started, release = threading.Event(), threading.Event()

    def available():
        started.set()
        assert release.wait(3)
        return 8192

    prefetch = ModelPrefetcher(
        max_bytes=8192, concurrency=1, memory_floor_bytes=1, memory_available=available
    )
    assert prefetch.schedule(first)
    assert started.wait(3)
    assert not prefetch.schedule(second)
    prefetch.close()
    release.set()
    drained(prefetch)
    assert prefetch.bytes_read == 0
    assert not prefetch.schedule(second)
