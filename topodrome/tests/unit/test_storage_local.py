from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from topodrome.storage import LocalStorage, StorageVerificationError

pytestmark = pytest.mark.unit


def test_local_flush_fsyncs_and_hashes_exact_content(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "global")
    path = storage.resolve("artifacts/result.bin")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"topodrome-result")

    verified = storage.flush_and_verify("artifacts/result.bin")

    assert verified.local_path == path
    assert verified.digest.size_bytes == len(b"topodrome-result")
    assert verified.digest.sha256 == hashlib.sha256(b"topodrome-result").hexdigest()
    assert verified.verified_at.tzinfo is not None


def test_local_flush_rejects_wrong_expected_checksum(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "global")
    path = storage.resolve("artifacts/result.bin")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"actual")

    with pytest.raises(StorageVerificationError, match="checksum"):
        storage.flush_and_verify("artifacts/result.bin", expected_sha256="0" * 64)


def test_storage_resolution_rechecks_symlink_containment(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "global")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (storage.root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside"):
        storage.resolve("linked/result")
