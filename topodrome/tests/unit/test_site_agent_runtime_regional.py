from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from topodrome.runtime import AsyncRuntimeExecutionService, CheckpointRecord, ModelRegistry
from topodrome.site_agent.admission import LeaseRequest
from topodrome.site_agent.agent import Assignment
from topodrome.site_agent.regional import RegionalStateStore
from topodrome.site_agent.runtime_backend import RuntimeExecutionBackend, _CheckpointWriter
from topodrome.site_agent.workspace import WorkspaceManager
from topodrome.storage import LocalStorage

pytestmark = pytest.mark.unit


def _assignment(
    attempt_id: str,
    *,
    job_id: str = "job-regional",
    checkpoint_ref: str | None = None,
    portability: str = "region-affine",
) -> Assignment:
    raw_spec: dict[str, object] = {
        "entrypoint": "experiment.py",
        "model": {"repo_id": "test/model", "revision": "rev"},
        "portability": portability,
    }
    if portability == "region-affine":
        raw_spec["affinity_region"] = "beijing"
    return Assignment(
        job_id=job_id,
        attempt_id=attempt_id,
        lease=LeaseRequest(
            lease_id=f"lease-{attempt_id}",
            attempt_id=attempt_id,
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 1},
            min_free_vram_bytes=1,
            expires_at=4_000_000_000,
        ),
        model_key="runtime-identity",
        entrypoint="experiment.py",
        raw_spec=raw_spec,
        checkpoint_ref=checkpoint_ref,
    )


def test_region_affine_checkpoint_is_durable_and_restored_from_regional_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("topodrome.site_agent.regional.os.sync", lambda: None)
    storage = LocalStorage(tmp_path / "global")
    regional = RegionalStateStore(tmp_path / "persistent", create=True)
    backend = RuntimeExecutionBackend(
        site_id="autodl-beijing",
        registry=ModelRegistry(),
        runtime=AsyncRuntimeExecutionService(),
        storage=storage,
        use_ray=False,
        regional_state=regional,
    )
    workspaces = WorkspaceManager(storage=storage, scratch_root=tmp_path / "scratch")
    first = _assignment("attempt-1")
    first_workspace = workspaces.prepare(first.job_id, first.attempt_id)
    directory, prefix = backend._checkpoint_target(first, first_workspace)
    reference = _CheckpointWriter(directory, prefix)(
        CheckpointRecord("optimizer", 1, {"step": 17})
    )

    assert reference.startswith("regional://jobs/job-regional/attempts/attempt-1/")
    assert not tuple(first_workspace.checkpoint_dir.glob("*.pkl"))
    assert backend._verify_checkpoints(first, first_workspace) == reference

    retry = _assignment("attempt-2", checkpoint_ref=reference)
    def forbid_agent_unpickle(*args, **kwargs):
        raise AssertionError("site agent must never deserialize checkpoint payloads")

    with monkeypatch.context() as patch:
        patch.setattr("pickle.loads", forbid_agent_unpickle)
        restored = backend._load_checkpoint(retry)
    assert restored is not None
    assert restored.reference == reference
    assert restored.value is None
    assert restored.materialize() == {"step": 17}


def test_regional_checkpoint_restore_rejects_wrong_scope_or_portability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("topodrome.site_agent.regional.os.sync", lambda: None)
    storage = LocalStorage(tmp_path / "global")
    regional = RegionalStateStore(tmp_path / "persistent", create=True)
    backend = RuntimeExecutionBackend(
        site_id="autodl-beijing",
        registry=ModelRegistry(),
        runtime=AsyncRuntimeExecutionService(),
        storage=storage,
        use_ray=False,
        regional_state=regional,
    )
    workspaces = WorkspaceManager(storage=storage, scratch_root=tmp_path / "scratch")
    first = _assignment("attempt-1")
    workspace = workspaces.prepare(first.job_id, first.attempt_id)
    directory, prefix = backend._checkpoint_target(first, workspace)
    reference = _CheckpointWriter(directory, prefix)(CheckpointRecord("state", 1, 9))

    with pytest.raises(ValueError, match="portability"):
        backend._load_checkpoint(
            _assignment("attempt-2", checkpoint_ref=reference, portability="portable")
        )
    with pytest.raises(ValueError, match="prior attempts"):
        backend._load_checkpoint(
            _assignment("attempt-2", job_id="another-job", checkpoint_ref=reference)
        )
    with pytest.raises(ValueError, match="prior attempts"):
        backend._load_checkpoint(replace(first, checkpoint_ref=reference))


def test_checkpoint_restore_requires_and_verifies_reference_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("topodrome.site_agent.regional.os.sync", lambda: None)
    storage = LocalStorage(tmp_path / "global")
    regional = RegionalStateStore(tmp_path / "persistent", create=True)
    backend = RuntimeExecutionBackend(
        site_id="autodl-beijing",
        registry=ModelRegistry(),
        runtime=AsyncRuntimeExecutionService(),
        storage=storage,
        use_ray=False,
        regional_state=regional,
    )
    first = _assignment("attempt-1")
    workspace = WorkspaceManager(
        storage=storage,
        scratch_root=tmp_path / "scratch",
    ).prepare(first.job_id, first.attempt_id)
    directory, prefix = backend._checkpoint_target(first, workspace)
    reference = _CheckpointWriter(directory, prefix)(CheckpointRecord("state", 1, 9))
    retry = _assignment("attempt-2", checkpoint_ref=reference)

    assert "#sha256=" in reference
    without_digest = reference.split("#", 1)[0]
    with pytest.raises(ValueError, match="requires an immutable SHA-256"):
        backend._load_checkpoint(replace(retry, checkpoint_ref=without_digest))

    checkpoint_path = regional.resolve(without_digest)
    checkpoint_path.write_bytes(b"tampered-checkpoint")
    with pytest.raises(RuntimeError, match="digest does not match"):
        backend._load_checkpoint(retry)
