from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from topodrome.client import cli
from topodrome.client.client import AttemptLogChunk
from topodrome.client.log_follow import follow_job
from topodrome.domain.specs import JobSpec, JobSummary

pytestmark = pytest.mark.unit


def _summary(state: str) -> JobSummary:
    now = datetime(2026, 9, 9, tzinfo=UTC)
    return JobSummary(
        job_id="job-live",
        state=state,
        spec=JobSpec(
            entrypoint="run.py",
            source_ref="sha256:" + "a" * 64,
            model="model",
            resources={"gpu": 1},
        ),
        created_at=now,
        updated_at=now,
    )


class _RetryingFollowClient:
    def __init__(self) -> None:
        self.poll = -1
        self.contents = {"attempt-1": b"one\n", "attempt-2": "tŵo\n".encode()}

    def status(self, job_id: str) -> JobSummary:
        assert job_id == "job-live"
        self.poll += 1
        return _summary(("running", "running", "succeeded")[min(self.poll, 2)])

    def attempts(self, job_id: str) -> list[dict[str, object]]:
        assert job_id == "job-live"
        first_state = "running" if self.poll == 0 else "failed"
        values: list[dict[str, object]] = [
            {
                "attempt_id": "attempt-1",
                "attempt_number": 1,
                "state": first_state,
                "site_id": "lab1",
            }
        ]
        if self.poll >= 1:
            values.append(
                {
                    "attempt_id": "attempt-2",
                    "attempt_number": 2,
                    "state": "running" if self.poll == 1 else "succeeded",
                    "site_id": "remote-1",
                }
            )
        return values

    def attempt_log_chunk(
        self,
        job_id: str,
        attempt_id: str,
        *,
        offset: int = 0,
        limit_bytes: int = 262144,
        anchor_sha256: str | None = None,
    ) -> AttemptLogChunk:
        assert job_id == "job-live"
        content = self.contents[attempt_id]
        # Force single-byte boundaries to exercise incremental UTF-8 decoding.
        end = min(len(content), offset + 1)
        data = content[offset:end]
        anchor = hashlib.sha256(content[max(0, end - 4096) : end]).hexdigest() if end else None
        complete = (attempt_id == "attempt-1" and self.poll >= 1) or (
            attempt_id == "attempt-2" and self.poll >= 2
        )
        return AttemptLogChunk(data, offset, end, len(content), False, anchor, complete)


def test_follow_job_streams_retries_once_and_keeps_status_separate() -> None:
    client = _RetryingFollowClient()
    stdout = io.StringIO()
    stderr = io.StringIO()

    result = follow_job(
        client,
        "job-live",
        log_output=stdout,
        status_output=stderr,
        poll_seconds=0.1,
        sleep=lambda _: None,
    )

    assert result.state.value == "succeeded"
    assert stdout.getvalue() == "one\ntŵo\n"
    assert stderr.getvalue().splitlines() == [
        "[topodrome] attempt 1 running on lab1",
        "[topodrome] attempt 1 failed on lab1",
        "[topodrome] attempt 2 running on remote-1",
        "[topodrome] attempt 2 succeeded on remote-1",
    ]


def test_follow_job_reports_log_replacement_and_does_not_repoll_terminal_attempt() -> None:
    class ReplacingClient:
        poll = -1
        reads = 0

        def status(self, _job_id: str) -> JobSummary:
            self.poll += 1
            return _summary("running" if self.poll < 2 else "failed")

        def attempts(self, _job_id: str) -> list[dict[str, object]]:
            return [
                {
                    "attempt_id": "attempt-1",
                    "attempt_number": 1,
                    "state": "running" if self.poll < 2 else "failed",
                }
            ]

        def attempt_log_chunk(
            self,
            _job_id: str,
            _attempt_id: str,
            *,
            offset: int = 0,
            limit_bytes: int = 262144,
            anchor_sha256: str | None = None,
        ) -> AttemptLogChunk:
            self.reads += 1
            if self.reads == 1:
                return AttemptLogChunk(b"old", 0, 3, 3, False, "a" * 64, False)
            if self.reads == 2:
                return AttemptLogChunk(b"new", 0, 3, 3, True, "b" * 64, False)
            return AttemptLogChunk(
                b"", 3, 3, 3, False, "b" * 64, self.poll >= 2
            )

    client = ReplacingClient()
    stdout = io.StringIO()
    stderr = io.StringIO()
    follow_job(
        client,
        "job-live",
        log_output=stdout,
        status_output=stderr,
        poll_seconds=0.1,
        sleep=lambda _: None,
    )

    assert stdout.getvalue() == "oldnew"
    assert "log was replaced" in stderr.getvalue()
    assert client.reads == 3


def test_follow_job_waits_for_late_authoritative_terminal_bytes() -> None:
    class LateClient:
        reads = 0

        def status(self, _job_id: str) -> JobSummary:
            return _summary("failed")

        def attempts(self, _job_id: str) -> list[dict[str, object]]:
            return [
                {
                    "attempt_id": "attempt-1",
                    "attempt_number": 1,
                    "state": "lost",
                    "site_id": "remote-1",
                }
            ]

        def attempt_log_chunk(
            self,
            _job_id: str,
            _attempt_id: str,
            *,
            offset: int = 0,
            limit_bytes: int = 262144,
            anchor_sha256: str | None = None,
        ) -> AttemptLogChunk:
            self.reads += 1
            if self.reads == 1:
                return AttemptLogChunk(b"", 0, 0, 0, False, None, False)
            value = b"late flush\n"
            end = len(value)
            anchor = hashlib.sha256(value).hexdigest()
            return AttemptLogChunk(value, 0, end, end, False, anchor, True)

    client = LateClient()
    stdout = io.StringIO()
    result = follow_job(
        client,
        "job-live",
        log_output=stdout,
        status_output=io.StringIO(),
        poll_seconds=0.1,
        sleep=lambda _: None,
    )

    assert result.state.value == "failed"
    assert client.reads == 2
    assert stdout.getvalue() == "late flush\n"


def test_follow_job_finalizes_utf8_at_each_completed_attempt_boundary() -> None:
    class BoundaryClient:
        poll = -1

        def status(self, _job_id: str) -> JobSummary:
            self.poll += 1
            return _summary("running" if self.poll == 0 else "failed")

        def attempts(self, _job_id: str) -> list[dict[str, object]]:
            values: list[dict[str, object]] = [
                {
                    "attempt_id": "attempt-1",
                    "attempt_number": 1,
                    "state": "failed",
                }
            ]
            if self.poll > 0:
                values.append(
                    {
                        "attempt_id": "attempt-2",
                        "attempt_number": 2,
                        "state": "failed",
                    }
                )
            return values

        def attempt_log_chunk(
            self,
            _job_id: str,
            attempt_id: str,
            *,
            offset: int = 0,
            limit_bytes: int = 262144,
            anchor_sha256: str | None = None,
        ) -> AttemptLogChunk:
            value = b"\xc3" if attempt_id == "attempt-1" else b"second"
            data = value[offset:]
            end = len(value)
            anchor = hashlib.sha256(value).hexdigest()
            return AttemptLogChunk(data, offset, end, end, False, anchor, True)

    stdout = io.StringIO()
    follow_job(
        BoundaryClient(),
        "job-live",
        log_output=stdout,
        status_output=io.StringIO(),
        poll_seconds=0.1,
        sleep=lambda _: None,
    )

    assert stdout.getvalue() == "�second"


def test_follow_job_bounds_drain_when_a_lost_site_never_seals_its_log() -> None:
    class DeadSiteClient:
        reads = 0

        def status(self, _job_id: str) -> JobSummary:
            return _summary("failed")

        def attempts(self, _job_id: str) -> list[dict[str, object]]:
            return [
                {
                    "attempt_id": "attempt-1",
                    "attempt_number": 1,
                    "state": "lost",
                    "site_id": "offline-site",
                }
            ]

        def attempt_log_chunk(
            self,
            _job_id: str,
            _attempt_id: str,
            *,
            offset: int = 0,
            limit_bytes: int = 262144,
            anchor_sha256: str | None = None,
        ) -> AttemptLogChunk:
            self.reads += 1
            return AttemptLogChunk(b"", 0, 0, 0, False, None, False)

    now = [0.0]

    def sleep(delay: float) -> None:
        now[0] += delay

    client = DeadSiteClient()
    stderr = io.StringIO()
    result = follow_job(
        client,
        "job-live",
        log_output=io.StringIO(),
        status_output=stderr,
        poll_seconds=0.1,
        terminal_log_grace_seconds=0.25,
        sleep=sleep,
        monotonic=lambda: now[0],
    )

    assert result.state.value == "failed"
    assert client.reads == 4
    assert "final log verification did not arrive within 0.25s" in stderr.getvalue()


def test_terminal_log_grace_still_expires_during_api_disconnect() -> None:
    class DisconnectingClient:
        status_calls = 0

        def status(self, _job_id: str) -> JobSummary:
            self.status_calls += 1
            if self.status_calls > 1:
                raise httpx.ConnectError("offline")
            return _summary("failed")

        def attempts(self, _job_id: str) -> list[dict[str, object]]:
            return [
                {
                    "attempt_id": "attempt-1",
                    "attempt_number": 1,
                    "state": "lost",
                }
            ]

        def attempt_log_chunk(
            self,
            _job_id: str,
            _attempt_id: str,
            *,
            offset: int = 0,
            limit_bytes: int = 262144,
            anchor_sha256: str | None = None,
        ) -> AttemptLogChunk:
            return AttemptLogChunk(b"", 0, 0, 0, False, None, False)

    now = [0.0]

    def sleep(delay: float) -> None:
        now[0] += delay

    stderr = io.StringIO()
    result = follow_job(
        DisconnectingClient(),
        "job-live",
        log_output=io.StringIO(),
        status_output=stderr,
        poll_seconds=0.1,
        terminal_log_grace_seconds=0.25,
        sleep=sleep,
        monotonic=lambda: now[0],
    )

    assert result.state.value == "failed"
    assert "API connection interrupted" in stderr.getvalue()
    assert "final log verification did not arrive" in stderr.getvalue()


class _CliHandle:
    job_id = "job-live"

    def status(self) -> JobSummary:
        return _summary("queued")


class _CliClient:
    def __enter__(self) -> _CliClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def submit(self, *_: object, **__: object) -> _CliHandle:
        return _CliHandle()


def test_submit_wait_keeps_workload_stdout_and_final_json_on_separate_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "job.yaml"
    manifest.write_text("placeholder: true\n")
    monkeypatch.setattr(cli, "Client", _CliClient)
    monkeypatch.setattr(cli, "load_job_file", lambda _: (object(), tmp_path))

    def follow(*_: object, **values: Any) -> JobSummary:
        values["log_output"].write("live output\n")
        return _summary("succeeded")

    monkeypatch.setattr(cli, "follow_job", follow)
    result = CliRunner().invoke(cli.app, ["submit", str(manifest), "--wait"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "live output\n"
    assert "submitted job-live" in result.stderr
    assert '"state": "succeeded"' in result.stderr
    assert "live output" not in result.stderr


def test_submit_wait_ctrl_c_leaves_job_running_and_returns_130(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "job.yaml"
    manifest.write_text("placeholder: true\n")
    monkeypatch.setattr(cli, "Client", _CliClient)
    monkeypatch.setattr(cli, "load_job_file", lambda _: (object(), tmp_path))

    def interrupt(*_: object, **__: object) -> JobSummary:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "follow_job", interrupt)
    result = CliRunner().invoke(cli.app, ["submit", str(manifest), "--wait"])

    assert result.exit_code == 130
    assert result.stdout == ""
    assert "job continues" in result.stderr


def test_submit_help_documents_live_wait_and_polling() -> None:
    result = CliRunner().invoke(cli.app, ["submit", "--help"])

    assert result.exit_code == 0, result.output
    # Rich wraps descriptions as option names change the table's column widths.
    help_text = " ".join(result.output.replace("│", " ").split())
    assert "Follow live attempt output" in help_text
    assert "--poll-interval" in result.output
    assert "--log-grace" in result.output
