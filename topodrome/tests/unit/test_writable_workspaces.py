from __future__ import annotations

import asyncio
import errno
import hashlib
import re
import threading
from types import SimpleNamespace

import pytest

from tests.unit.test_storage_http_mirror import FakeAuthority, _storage
from topodrome.runtime import ArtifactRegistration, ModelRegistry
from topodrome.site_agent.finalization import FinalizationJournal
from topodrome.site_agent.runtime_backend import RuntimeExecutionBackend, _ExecutionReport
from topodrome.site_agent.workspace import WorkspaceManager
from topodrome.storage import LocalStorage
from topodrome.storage.paths import LogicalPathError

pytestmark = pytest.mark.unit


def assignment():
    return SimpleNamespace(
        job_id="job-retry",
        attempt_id="attempt-new",
        checkpoint_ref=None,
        raw_spec={
            "entrypoint": "run.py", "model": "test/model", "required_outputs": ["result.txt"],
        },
    )


@pytest.mark.parametrize("cached", [False, True])
def test_retry_workspace_does_not_write_into_a_downloaded_job_tree(tmp_path, cached):
    remote = FakeAuthority()
    old_log = "jobs/job-retry/attempts/attempt-old/logs/attempt.log"
    remote.objects[old_log] = b"previous attempt diagnostics"
    storage = _storage(tmp_path, remote)
    view = storage.resolve("jobs/job-retry") if cached else None
    before = list(remote.calls)
    workspaces = WorkspaceManager(storage=storage, scratch_root=tmp_path / "scratch")
    workspace = workspaces.prepare("job-retry", "attempt-new")
    assert remote.calls == before  # Creating an output workspace does not download history.
    assert workspace.job_root == storage.root / "jobs/job-retry"
    if view is not None:
        assert workspace.job_root != view
        assert (view / "attempts/attempt-old/logs/attempt.log").read_bytes() == remote.objects[
            old_log
        ]

    payload = b"complete workload output\n"
    log = workspace.log_dir / "attempt.log"
    log.write_bytes(payload)
    checkpoint = workspace.checkpoint_dir / "00000001-training.pkl"
    checkpoint.write_bytes(b"latest recovery state")
    backend = SimpleNamespace(storage=storage)
    result = RuntimeExecutionBackend._verify_log(backend, assignment(), workspace)
    logical_log = "jobs/job-retry/attempts/attempt-new/logs/attempt.log"
    assert result == {
        "logical_path": logical_log,
        "start_offset": 0,
        "end_offset": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert remote.objects[logical_log] == payload
    reference = RuntimeExecutionBackend._verify_checkpoints(backend, assignment(), workspace)
    logical_checkpoint = "jobs/job-retry/attempts/attempt-new/checkpoints/00000001-training.pkl"
    assert reference.startswith(logical_checkpoint + "#sha256=")
    assert remote.objects[logical_checkpoint] == checkpoint.read_bytes()
    assert remote.objects[old_log] == b"previous attempt diagnostics"
    assert workspaces.prepare("job-retry", "attempt-new").log_dir == workspace.log_dir


@pytest.mark.asyncio
@pytest.mark.parametrize("blobs", [False, True])
async def test_remote_finalization_restart_preserves_canonical_logs_and_outputs(tmp_path, blobs):
    remote = FakeAuthority()
    remote.objects["jobs/job-retry/attempts/attempt-old/logs/attempt.log"] = b"old failure"
    storage = _storage(tmp_path, remote)
    storage.content_addressed = blobs
    read_view = storage.resolve("jobs/job-retry")
    manager = WorkspaceManager(storage=storage, scratch_root=tmp_path / "scratch")
    workspace = manager.prepare("job-retry", "attempt-new")
    (workspace.log_dir / "attempt.log").write_text("workload finished\n")
    output = workspace.scratch_dir / "computed.txt"
    output.write_text("expensive completed result")
    journal = FinalizationJournal(tmp_path / "private/finalization")

    def backend():
        return RuntimeExecutionBackend(
            site_id="remote",
            registry=ModelRegistry(),
            runtime=SimpleNamespace(),  # No actor is available to repeat computation.
            storage=storage,
            use_ray=False,
            finalization_journal=FinalizationJournal(journal.root),
        )

    initial = backend()
    initial._snapshot_completion(
        assignment(),
        workspace,
        _ExecutionReport((ArtifactRegistration(output, "result.txt", True),), None),
    )
    output.unlink()
    restarted = backend()
    workspace = manager.prepare("job-retry", "attempt-new")
    result = await asyncio.wait_for(
        restarted.resume_finalization(assignment(), workspace, next(journal.pending())),
        3,
    )
    assert result.succeeded and result.artifacts_verified
    logical_log = "jobs/job-retry/attempts/attempt-new/logs/attempt.log"
    log = (workspace.log_dir / "attempt.log").read_bytes()
    assert log.startswith(b"workload finished\n")
    assert remote.objects[logical_log] == log
    assert result.metadata["log"]["sha256"] == hashlib.sha256(log).hexdigest()
    assert journal.contains("attempt-new")  # Retired only after the normal terminal ACK.
    assert (read_view / "attempts/attempt-old/logs/attempt.log").read_bytes() == b"old failure"
    assert not (read_view / "attempts/attempt-new").exists()


@pytest.mark.parametrize("kind", ["symlink-parent", "symlink-file", "file-parent", "fifo"])
@pytest.mark.parametrize("adapter", ["native", "mirror"])
def test_writable_paths_reject_aliases_and_special_files(tmp_path, kind, adapter):
    import os

    storage = (
        LocalStorage(tmp_path / "root")
        if adapter == "native"
        else _storage(tmp_path, FakeAuthority())
    )
    root = storage.root
    target = root / "protected"
    target.write_bytes(b"do not change")
    logical = "jobs/output"
    if kind == "symlink-parent":
        (root / "jobs").symlink_to(root, target_is_directory=True)
    elif kind == "file-parent":
        (root / "jobs").write_bytes(b"not a directory")
    else:
        (root / "jobs").mkdir()
        if kind == "symlink-file":
            (root / logical).symlink_to(target)
        else:
            os.mkfifo(root / logical)
    with pytest.raises(LogicalPathError):
        storage.resolve_writable(logical)
    assert target.read_bytes() == b"do not change"


@pytest.mark.parametrize("logical", ["../outside", "/absolute", "jobs/../other", "jobs//file"])
def test_writable_paths_reject_unsafe_logical_names(tmp_path, logical):
    with pytest.raises(LogicalPathError):
        _storage(tmp_path, FakeAuthority()).resolve_writable(logical)


def test_writable_mirror_paths_do_not_expose_internal_cache_namespaces(tmp_path):
    storage = _storage(tmp_path, FakeAuthority())
    for prefix in (".topodrome-mirror-partials", ".topodrome-mirror-uploads"):
        with pytest.raises(ValueError):
            storage.resolve_writable(prefix + "/output")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("local log missing"),
        PermissionError("local access denied"),
        OSError(errno.ENOSPC, "local disk full"),
    ],
)
async def test_local_finalization_errors_are_not_retried_as_network_failures(tmp_path, error):
    calls = []

    async def fail(*_args):
        calls.append(True)
        raise error

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log = log_dir / "attempt.log"
    log.write_text("completed workload\n")
    backend = SimpleNamespace(finalization_journal=object(), _finalize_outputs=fail)
    with pytest.raises(type(error), match=re.escape(str(error))):
        await asyncio.wait_for(
            RuntimeExecutionBackend._retry_finalization(
                backend,
                assignment(),
                SimpleNamespace(log_dir=log_dir),
                None,
                False,
                threading.Event(),
            ),
            0.5,
        )
    assert calls == [True]
    assert log.read_text() == "completed workload\n"
