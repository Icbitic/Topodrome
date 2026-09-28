from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from topodrome.site_agent.admission import LeaseRequest
from topodrome.site_agent.agent import (
    Assignment,
    SiteAgent,
    _PendingTerminalState,
)
from topodrome.site_agent.workspace import AttemptWorkspace

pytestmark = pytest.mark.unit


def _assignment() -> Assignment:
    return Assignment(
        job_id="job-1",
        attempt_id="attempt-1",
        lease=LeaseRequest(
            lease_id="lease-1",
            attempt_id="attempt-1",
            gpu_uuids=("GPU-A",),
            fence_epochs={"GPU-A": 1},
            min_free_vram_bytes=0,
            expires_at=4_000_000_000,
        ),
        model_key="model",
        entrypoint="experiment.py",
    )


def _workspace(tmp_path: Path) -> AttemptWorkspace:
    attempt = tmp_path / "jobs" / "job-1" / "attempts" / "attempt-1"
    log_dir = attempt / "logs"
    log_dir.mkdir(parents=True)
    return AttemptWorkspace(
        job_id="job-1",
        attempt_id="attempt-1",
        job_root=attempt.parents[1],
        attempt_root=attempt,
        staging_dir=attempt / "staging",
        checkpoint_dir=attempt / "checkpoints",
        log_dir=log_dir,
        committed_dir=attempt / "committed",
        scratch_dir=tmp_path / "scratch",
    )


@pytest.mark.asyncio
async def test_terminal_log_verification_retries_before_publishing_completion(
    tmp_path: Path,
) -> None:
    class Backend:
        calls = 0

        async def verify_attempt_log(
            self, _assignment: Assignment, _workspace: AttemptWorkspace
        ) -> None:
            self.calls += 1
            if self.calls == 1:
                raise OSError("transient storage outage")
            return None

    agent = object.__new__(SiteAgent)
    backend = Backend()
    agent.backend = cast(Any, backend)
    agent._stop = asyncio.Event()

    result = await asyncio.wait_for(
        agent._complete_terminal_log(
            _assignment(),
            _workspace(tmp_path),
            _PendingTerminalState("failed"),
            None,
        ),
        timeout=1,
    )

    assert result is not None and result.log_complete
    assert result.log is None
    assert backend.calls == 2


@pytest.mark.asyncio
async def test_terminal_with_verified_absence_is_explicitly_complete(
    tmp_path: Path,
) -> None:
    agent = object.__new__(SiteAgent)
    agent.backend = cast(Any, object())
    agent._stop = asyncio.Event()

    result = await agent._complete_terminal_log(
        _assignment(),
        _workspace(tmp_path),
        _PendingTerminalState("canceled"),
        None,
    )

    assert result is not None and result.log_complete
    assert result.log is None
