from __future__ import annotations

from pathlib import Path

import pytest

from topodrome.storage.paths import LogicalPathError, normalize_logical_path, resolve_beneath

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "/absolute", "a/../b", "a/./b", "a//b", "a\\b", "a\0b", "a\nb"],
)
def test_unsafe_logical_paths_are_rejected(value: str) -> None:
    with pytest.raises(LogicalPathError):
        normalize_logical_path(value)


def test_logical_paths_remain_machine_independent() -> None:
    assert normalize_logical_path("jobs/job-1/results/table.json") == (
        "jobs/job-1/results/table.json"
    )


def test_existing_symlink_cannot_escape_storage_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(LogicalPathError):
        resolve_beneath(root, "escape/secret.txt")


def test_symlink_that_stays_beneath_root_is_allowed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "target"
    target.mkdir(parents=True)
    (root / "alias").symlink_to(target, target_is_directory=True)

    assert resolve_beneath(root, "alias/data.json") == target / "data.json"
