from __future__ import annotations

from pathlib import Path

import pytest

from topodrome.site_agent.ray_cluster import RayClusterConfig, gpu_resource_name
from topodrome.site_agent.workspace import WorkspaceManager

pytestmark = pytest.mark.unit


def test_workspace_uses_global_outputs_and_local_scratch(tmp_path: Path) -> None:
    (tmp_path / "global").mkdir()
    manager = WorkspaceManager(global_root=tmp_path / "global", scratch_root=tmp_path / "scratch")
    workspace = manager.prepare("job-1", "attempt-1")

    assert workspace.staging_dir == (tmp_path / "global/jobs/job-1/attempts/attempt-1/staging")
    assert workspace.committed_dir == tmp_path / "global/jobs/job-1/committed"
    assert workspace.scratch_dir == (tmp_path / "scratch/jobs/job-1/attempts/attempt-1")
    assert all(path.is_dir() for path in (workspace.staging_dir, workspace.scratch_dir))
    assert "TOKEN" not in " ".join(workspace.environment())

    manager.cleanup_scratch(workspace)
    assert not workspace.scratch_dir.exists()


def test_workspace_rejects_path_traversal(tmp_path: Path) -> None:
    (tmp_path / "global").mkdir()
    manager = WorkspaceManager(global_root=tmp_path / "global", scratch_root=tmp_path / "scratch")
    with pytest.raises(ValueError, match="unsafe job_id"):
        manager.prepare("../escape", "attempt")


def test_ray_resource_names_are_exact_uuid_slots() -> None:
    assert gpu_resource_name("GPU-ABC:123") == "topodrome_gpu_gpu_abc_123"
    config = RayClusterConfig(gpu_uuids=("GPU-A", "GPU-B"))
    assert config.custom_resources == {
        "topodrome_gpu_gpu_a": 1,
        "topodrome_gpu_gpu_b": 1,
    }
    assert config.node_ip_address == "127.0.0.1"
