from __future__ import annotations

import hashlib
import shlex
import subprocess
from pathlib import Path

import pytest

from topodrome.deployment.ssh_files import receiver_command


def _receive(root: Path, payload: bytes, *, size=None, digest=None, name="release.tar.gz"):
    command = receiver_command(
        str(root),
        name,
        len(payload) if size is None else size,
        hashlib.sha256(payload).hexdigest() if digest is None else digest,
        0o640,
    )
    return subprocess.run(shlex.split(command), input=payload, capture_output=True)  # noqa: S603


def test_receiver_atomically_publishes_exact_verified_content(tmp_path):
    root = tmp_path / "private" / "install"
    result = _receive(root, b"release bytes")
    assert result.returncode == 0, result.stderr
    target = root / "release.tar.gz"
    assert target.read_bytes() == b"release bytes"
    assert target.stat().st_mode & 0o777 == 0o640
    assert not list(root.glob(".ssh-upload-*"))


@pytest.mark.parametrize("size,digest", [(100, None), (1, None), (None, "0" * 64)])
def test_receiver_preserves_old_release_on_truncation_excess_or_corruption(tmp_path, size, digest):
    root = tmp_path / "install"
    root.mkdir(mode=0o700)
    target = root / "release.tar.gz"
    target.write_bytes(b"old release")
    result = _receive(root, b"new release", size=size, digest=digest)
    assert result.returncode != 0
    assert target.read_bytes() == b"old release"
    assert not list(root.glob(".ssh-upload-*"))


def test_receiver_refuses_symlink_target_and_parent(tmp_path):
    root = tmp_path / "install"
    root.mkdir(mode=0o700)
    unrelated = tmp_path / "other-users-file"
    unrelated.write_bytes(b"untouched")
    (root / "release.tar.gz").symlink_to(unrelated)
    assert _receive(root, b"new release").returncode != 0
    assert unrelated.read_bytes() == b"untouched"
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    assert _receive(link, b"new release", name="another").returncode != 0
    assert not (root / "another").exists()


@pytest.mark.parametrize(
    "root,name,mode",
    [
        ("/", "release", 0o600),
        ("/private/../escape", "release", 0o600),
        ("/private", "../release", 0o600),
        ("/private", "release", 0o4600),
    ],
)
def test_receiver_command_rejects_unsafe_targets(root, name, mode):
    with pytest.raises(ValueError):
        receiver_command(root, name, 1, "a" * 64, mode)
