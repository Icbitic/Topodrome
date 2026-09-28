import hashlib
import os
import pickle
import subprocess
import sys

import pytest

from topodrome.runtime.checkpoint import CheckpointRestore
from topodrome.runtime.context import ExperimentContext


class RestorePID:
    def __reduce__(self):
        return os.getpid, ()


def descriptor(tmp_path, value):
    data = pickle.dumps(value)
    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(data)
    return CheckpointRestore(
        "jobs/j/checkpoints/c#sha256=" + hashlib.sha256(data).hexdigest(),
        serialized_path=str(path),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def context(restore):
    return ExperimentContext(
        model=None,
        tokenizer=None,
        device=None,
        job_id="j",
        attempt_id="a",
        checkpoint_restore=restore,
    )


def test_restoration_is_lazy_and_cached(tmp_path, monkeypatch):
    restore = descriptor(tmp_path, {"step": 10})
    real = pickle.loads
    calls = []

    def decode(data):
        calls.append(1)
        return real(data)

    monkeypatch.setattr(pickle, "loads", decode)
    c = context(restore)
    assert c.has_checkpoint and not calls
    first = c.restore_checkpoint()
    assert c.restore_checkpoint() is first
    assert first == {"step": 10} and len(calls) == 1


def test_changed_bytes_rejected_before_unpickling(tmp_path, monkeypatch):
    restore = descriptor(tmp_path, {"step": 10})
    (tmp_path / "checkpoint.pkl").write_bytes(b"corrupt")
    monkeypatch.setattr(pickle, "loads", lambda _: pytest.fail("must verify first"))
    with pytest.raises(RuntimeError, match="changed"):
        context(restore).restore_checkpoint()


def test_inert_descriptor_unpickles_payload_only_in_worker_process(tmp_path):
    restore = descriptor(tmp_path, RestorePID())
    # Serializing the descriptor for Ray must not materialize the payload.
    wire = tmp_path / "descriptor.pkl"
    wire.write_bytes(pickle.dumps(restore))
    code = """import os,pickle,sys
from topodrome.runtime.context import ExperimentContext
restore=pickle.load(open(sys.argv[1], 'rb'))
assert restore.value is None
c=ExperimentContext(model=None,tokenizer=None,device=None,job_id='j',attempt_id='a',checkpoint_restore=restore)
assert c.restore_checkpoint()==os.getpid()
print(os.getpid())
"""
    output = subprocess.check_output(  # noqa: S603 - fixed local test program
        [sys.executable, "-c", code, str(wire)], text=True
    )
    assert int(output.strip()) != os.getpid()
