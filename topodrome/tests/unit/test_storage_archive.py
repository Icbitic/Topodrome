from __future__ import annotations

import json

import pytest

from scripts.archive_storage_fresh import archive_directories
from topodrome.storage.blobs import validate_layout

pytestmark = pytest.mark.unit


def test_archive_moves_only_named_data_without_copying(tmp_path):
    root = tmp_path / "global"
    archive = tmp_path / "archive"
    for name in ("jobs", "artifacts", "projects", "notebooks", "bootstrap", ".scheduler-state"):
        (root / name).mkdir(parents=True)
        (root / name / "file").write_text(name)
    inode = (root / "artifacts/file").stat().st_ino
    archive_directories(root, archive)
    assert (archive / "artifacts/file").stat().st_ino == inode
    assert (archive / "artifacts/file").read_text() == "artifacts"
    assert not (root / "jobs").exists()
    assert (root / "notebooks/file").exists()
    assert (root / "bootstrap/file").exists()
    assert (root / ".scheduler-state/file").exists()
    assert json.loads((archive / "files-journal.json").read_text())["state"] == "moved"
    with pytest.raises(FileExistsError):
        archive_directories(root, archive)


def test_archive_refuses_symlinks_and_nested_targets(tmp_path):
    root = tmp_path / "global"
    root.mkdir()
    (root / "jobs").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        archive_directories(root, tmp_path / "archive")
    with pytest.raises(ValueError):
        archive_directories(root, root / "archive")
    assert (root / "jobs").is_symlink()


def test_storage_layout_marker_rejects_old_agent_configuration(tmp_path):
    marker = tmp_path / "storage-layout.json"
    marker.write_text('{"layout":"blobs-v1"}')
    validate_layout(marker, "blobs-v1")
    with pytest.raises(ValueError, match="upgrade required"):
        validate_layout(marker, "attempt-v1")
