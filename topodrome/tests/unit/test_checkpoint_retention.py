from __future__ import annotations

import fcntl
import gc
import pickle
import threading
import weakref

import pytest

from topodrome.runtime.checkpoint import CheckpointManager, CheckpointRecord
from topodrome.site_agent.runtime_backend import _CheckpointWriter
from topodrome.storage.checkpoint_retention import pin_latest_checkpoint, rotate_checkpoints

pytestmark = pytest.mark.unit


def writer(directory, **kwargs):
    return _CheckpointWriter(directory, "jobs/j/attempts/a/checkpoints", **kwargs)


def save(w, n):
    return w(CheckpointRecord(name="progress", sequence=n, value={"step": n}))


def states(directory):
    return sorted(pickle.loads(p.read_bytes())["step"] for p in directory.glob("*.pkl"))  # noqa: S301


def test_rotation_keeps_two_complete_snapshots_and_unknown_files(tmp_path):
    w = writer(tmp_path)
    (tmp_path / "notes.pkl").write_bytes(b"unrelated")
    (tmp_path / "training.log").write_text("loss=1")
    for n in range(1, 20):
        save(w, n)
    assert len(list(tmp_path.glob("[0-9]*.pkl"))) == 2
    assert (tmp_path / "notes.pkl").read_bytes() == b"unrelated"
    assert (tmp_path / "training.log").read_text() == "loss=1"


def test_verification_pin_survives_rotation_and_writer_restart(tmp_path):
    w = writer(tmp_path)
    save(w, 1)
    pinned = pin_latest_checkpoint(tmp_path)
    # Pin remains even when verification or the agent crashes afterwards.
    w = writer(tmp_path)
    for n in range(2, 7):
        save(w, n)
    assert pinned.is_file()
    assert states(tmp_path) == [1, 5, 6]


def test_failed_new_checkpoint_does_not_prune_previous(tmp_path, monkeypatch):
    w = writer(tmp_path)
    save(w, 1)
    save(w, 2)
    monkeypatch.setattr(pickle, "dump", lambda *a, **kw: (_ for _ in ()).throw(OSError("full")))
    with pytest.raises(OSError):
        save(w, 3)
    assert states(tmp_path) == [1, 2]
    assert not list(tmp_path.glob("*.tmp-*"))


def test_rotation_skips_lock_contention_and_recovers_next_save(tmp_path):
    w = writer(tmp_path)
    save(w, 1)
    with (tmp_path / ".retention.lock").open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for n in range(2, 6):
            save(w, n)
        assert states(tmp_path) == [1, 2, 3, 4, 5]
    save(w, 6)
    assert states(tmp_path) == [5, 6]


def test_pin_and_rotation_share_cross_process_lock(tmp_path):
    w = writer(tmp_path)
    save(w, 1)
    result = []
    with (tmp_path / ".retention.lock").open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        t = threading.Thread(target=lambda: result.append(pin_latest_checkpoint(tmp_path)))
        t.start()
        t.join(0.05)
        assert t.is_alive() and not result
    t.join(2)
    assert not t.is_alive() and len(result) == 1
    for n in range(2, 5):
        save(w, n)
    assert result[0].exists()


def test_unsafe_pins_fail_closed_without_breaking_checkpoint_save(tmp_path):
    w = writer(tmp_path)
    (tmp_path / ".retention-pins").symlink_to(tmp_path / "missing")
    for n in range(1, 5):
        save(w, n)
    assert states(tmp_path) == [1, 2, 3, 4]
    with pytest.raises((ValueError, OSError)):
        pin_latest_checkpoint(tmp_path)


def test_rotation_does_not_unlink_hardlinks_or_symlinks(tmp_path):
    w = writer(tmp_path)
    save(w, 1)
    first = next(tmp_path.glob("*.pkl"))
    (tmp_path / "hardlink").hardlink_to(first)
    link = tmp_path / "00000000-aaaaaaaaaaaa.pkl"
    link.symlink_to(first)
    for n in range(2, 5):
        save(w, n)
    assert first.exists() and link.is_symlink()


def test_configurable_keep_and_numeric_sequence_order(tmp_path):
    w = writer(tmp_path, keep=3)
    for n in range(99999998, 100000003):
        save(w, n)
    assert states(tmp_path) == [100000000, 100000001, 100000002]
    assert pickle.loads(pin_latest_checkpoint(tmp_path).read_bytes())["step"] == 100000002  # noqa: S301
    with pytest.raises(ValueError):
        rotate_checkpoints(tmp_path, keep=1)


def test_checkpoint_manager_releases_historical_payloads():
    class Payload:
        pass

    manager = CheckpointManager(lambda record: "saved")
    first = Payload()
    ref = weakref.ref(first)
    manager.save(first)
    del first
    for _ in range(10):
        manager.save(Payload())
    gc.collect()
    assert ref() is None
    assert len(manager.records) == 2
    assert [r.sequence for r in manager.records] == [10, 11]
