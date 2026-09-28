from __future__ import annotations

from pathlib import Path

import pytest

from topodrome.site_agent.regional import RegionalStateStore, is_regional_reference

pytestmark = pytest.mark.unit


def test_regional_checkpoint_reference_and_verification(tmp_path: Path) -> None:
    store = RegionalStateStore(tmp_path / "regional", create=True)
    directory = store.checkpoint_directory("job-1", "attempt-1")
    checkpoint = directory / "00000001-state.pkl"
    checkpoint.write_bytes(b"checkpoint")

    reference = store.reference_for(checkpoint)
    verified = store.flush_and_verify(reference)

    assert is_regional_reference(reference)
    assert store.resolve(reference) == checkpoint
    assert verified.digest.size_bytes == len(b"checkpoint")


def test_regional_shutdown_marker_is_hashed_and_durable(tmp_path: Path) -> None:
    store = RegionalStateStore(tmp_path / "regional", create=True)
    marker = store.durability_barrier(
        shutdown_generation=4,
        command_nonce="opaque-command-nonce",
    )

    content = marker.local_path.read_text(encoding="utf-8")
    assert marker.reference.startswith("regional://.topodrome/durability/")
    assert "opaque-command-nonce" not in content
    assert '"shutdown_generation":4' in content


def test_regional_store_rejects_escape_and_wrong_scheme(tmp_path: Path) -> None:
    store = RegionalStateStore(tmp_path / "regional", create=True)
    with pytest.raises(ValueError):
        store.resolve("regional://../escape")
    with pytest.raises(ValueError, match="wrong scheme"):
        store.resolve("jobs/job/checkpoint")
    with pytest.raises(ValueError, match="outside"):
        store.reference_for(tmp_path / "outside")
