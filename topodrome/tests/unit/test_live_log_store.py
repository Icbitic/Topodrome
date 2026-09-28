from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from topodrome.domain.logs import LiveLogChunkReport
from topodrome.domain.specs import AttemptLogDescriptor
from topodrome.scheduler.live_logs import LiveLogStore

pytestmark = pytest.mark.unit


def _report(
    data: bytes,
    *,
    offset: int = 0,
    generation: int = 0,
) -> LiveLogChunkReport:
    return LiveLogChunkReport(
        site_id="remote-1",
        job_id="job-1",
        attempt_id="attempt-1",
        fencing_token="fenced-value",  # noqa: S106 - inert test value
        generation=generation,
        offset=offset,
        data_base64=base64.b64encode(data).decode("ascii"),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def test_live_store_appends_overlaps_and_acknowledgement_replays_once(
    tmp_path: Path,
) -> None:
    store = LiveLogStore(tmp_path)

    first = store.append(_report(b"hello"))
    replay = store.append(_report(b"hello"))
    overlap = store.append(_report(b"lo world", offset=3))
    gap = store.append(_report(b"!", offset=99))

    assert (first.accepted, first.next_offset) == (True, 5)
    assert replay.accepted and replay.duplicate and replay.next_offset == 5
    assert overlap.accepted and not overlap.duplicate and overlap.next_offset == 11
    assert not gap.accepted and gap.next_offset == 11
    assert store.latest_path("job-1", "attempt-1").read_bytes() == b"hello world"  # type: ignore[union-attr]


def test_job_tail_orders_retries_and_bounds_combined_bytes(tmp_path: Path) -> None:
    store = LiveLogStore(tmp_path)
    root = tmp_path / "jobs" / "job-1" / "attempts"
    store.append(_report(b"older").model_copy(update={"attempt_id": "attempt-z"}))
    store.append(_report(b"newer").model_copy(update={"attempt_id": "attempt-a"}))
    ids = ("attempt-z", "attempt-a", "attempt-empty")
    assert not root.exists()
    for limit, expected in ((100, b"older\nnewer"), (7, b"r\nnewer"), (5, b"newer"), (2, b"er")):
        result = store.read_tail("job-1", ids, attempts_root=root, limit_bytes=limit)
        assert result == expected
        assert len(result) <= limit
    assert store.read_tail("job-1", (), attempts_root=root, limit_bytes=100) == b""


def test_live_store_conflict_requires_a_new_generation(tmp_path: Path) -> None:
    store = LiveLogStore(tmp_path)
    store.append(_report(b"old"))

    conflict = store.append(_report(b"new", offset=0))
    replacement = store.append(_report(b"new", generation=conflict.generation))
    stale = store.append(_report(b"old", generation=0))

    assert not conflict.accepted and conflict.reset and conflict.generation == 1
    assert replacement.accepted
    assert not stale.accepted and stale.reset and stale.generation == 2
    assert store.latest_path("job-1", "attempt-1").read_bytes() == b"new"  # type: ignore[union-attr]


def test_range_cursor_crosses_from_live_to_authoritative_and_detects_rewrite(
    tmp_path: Path,
) -> None:
    store = LiveLogStore(tmp_path)
    store.append(_report(b"hello "))
    attempts_root = tmp_path / "jobs" / "job-1" / "attempts"

    live = store.read_range(
        "job-1",
        "attempt-1",
        attempts_root=attempts_root,
        offset=0,
        limit_bytes=100,
        anchor_sha256=None,
    )
    assert live.data == b"hello "

    authoritative = attempts_root / "attempt-1" / "logs" / "attempt.log"
    authoritative.parent.mkdir(parents=True)
    authoritative.write_bytes(b"hello world")
    assert store.register_authoritative(
        "job-1",
        "attempt-1",
        attempts_root=attempts_root,
        descriptor=_descriptor(b"hello world"),
    )
    final = store.read_range(
        "job-1",
        "attempt-1",
        attempts_root=attempts_root,
        offset=live.next_offset,
        limit_bytes=100,
        anchor_sha256=live.anchor_sha256,
    )
    assert not final.reset
    assert final.data == b"world"

    authoritative.write_bytes(b"replacement")
    replaced = store.read_range(
        "job-1",
        "attempt-1",
        attempts_root=attempts_root,
        offset=final.next_offset,
        limit_bytes=100,
        anchor_sha256=final.anchor_sha256,
    )
    assert replaced.reset
    assert replaced.offset == 0
    assert replaced.data == b"replacement"


def test_advisory_sidecar_cleanup_requires_and_preserves_authoritative_log(
    tmp_path: Path,
) -> None:
    store = LiveLogStore(tmp_path)
    store.append(_report(b"complete output\n"))
    live_path = store.latest_path("job-1", "attempt-1")
    assert live_path is not None
    attempts_root = tmp_path / "jobs" / "job-1" / "attempts"

    assert not store.register_authoritative(
        "job-1",
        "attempt-1",
        attempts_root=attempts_root,
        descriptor=_descriptor(b"complete output\n"),
    )
    assert live_path.exists()

    authoritative = attempts_root / "attempt-1" / "logs" / "attempt.log"
    authoritative.parent.mkdir(parents=True)
    authoritative.write_bytes(b"complete output\n")
    wrong = _descriptor(b"other output\n")
    assert not store.register_authoritative(
        "job-1",
        "attempt-1",
        attempts_root=attempts_root,
        descriptor=wrong,
    )
    assert live_path.exists()

    assert store.register_authoritative(
        "job-1",
        "attempt-1",
        attempts_root=attempts_root,
        descriptor=_descriptor(b"complete output\n"),
    )
    assert not live_path.exists()
    assert authoritative.read_bytes() == b"complete output\n"
    assert store.is_complete("job-1", "attempt-1")
    late = store.append(_report(b"late"))
    assert not late.accepted and late.stop
    assert store.latest_path("job-1", "attempt-1") is None
    assert authoritative.read_bytes() == b"complete output\n"


def test_live_store_bounds_bytes_and_eventually_prunes_orphans(tmp_path: Path) -> None:
    store = LiveLogStore(tmp_path / "global", max_stream_bytes=64 * 1024, retention_seconds=1)
    first = b"a" * 40_000
    assert store.append(_report(first)).accepted

    over_quota = store.append(_report(b"b" * 30_000, offset=len(first)))
    assert not over_quota.accepted and over_quota.stop
    live_path = store.latest_path("job-1", "attempt-1")
    assert live_path is not None and live_path.stat().st_size == len(first)

    os.utime(live_path, (0, 0))
    assert (
        store.prune_expired(
            now=2,
            retain_attempts=frozenset({("job-1", "attempt-1")}),
        )
        == 0
    )
    assert live_path.exists()
    assert store.prune_expired(now=2) == 1
    assert store.latest_path("job-1", "attempt-1") is None


def test_verified_absent_log_has_an_explicit_empty_completion_range(
    tmp_path: Path,
) -> None:
    store = LiveLogStore(tmp_path / "global")
    attempts_root = tmp_path / "global" / "jobs" / "job-1" / "attempts"

    assert store.register_authoritative(
        "job-1",
        "attempt-1",
        attempts_root=attempts_root,
        descriptor=None,
    )
    value = store.read_range(
        "job-1",
        "attempt-1",
        attempts_root=attempts_root,
        offset=0,
        limit_bytes=100,
        anchor_sha256=None,
    )

    assert value.complete
    assert value.data == b""
    assert value.size_bytes == 0


def test_append_rejects_live_root_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    global_root = tmp_path / "global"
    global_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    original_mode = outside.stat().st_mode
    (global_root / ".topodrome-live-logs").symlink_to(
        outside, target_is_directory=True
    )

    with pytest.raises(OSError):
        LiveLogStore(global_root).append(_report(b"escape"))

    assert list(outside.iterdir()) == []
    assert outside.stat().st_mode == original_mode


def test_append_rejects_intermediate_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    global_root = tmp_path / "global"
    live_root = global_root / ".topodrome-live-logs"
    live_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (live_root / "job-1").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        LiveLogStore(global_root).append(_report(b"escape"))

    assert list(outside.iterdir()) == []


def test_append_rejects_generation_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    global_root = tmp_path / "global"
    attempt_root = global_root / ".topodrome-live-logs" / "job-1" / "attempt-1"
    attempt_root.mkdir(parents=True)
    outside = tmp_path / "outside.log"
    outside.write_bytes(b"preserve")
    original_mode = outside.stat().st_mode
    (attempt_root / "0.log").symlink_to(outside)

    with pytest.raises(OSError):
        LiveLogStore(global_root).append(_report(b"escape"))

    assert outside.read_bytes() == b"preserve"
    assert outside.stat().st_mode == original_mode


def test_chunk_validation_hides_payload_and_token() -> None:
    secret_output = b"do-not-render"
    token = "also-do-not-render"  # noqa: S105 - inert test value

    with pytest.raises(ValidationError) as captured:
        LiveLogChunkReport(
            site_id="remote-1",
            job_id="job-1",
            attempt_id="attempt-1",
            fencing_token=token,
            offset=0,
            data_base64=base64.b64encode(secret_output).decode("ascii"),
            sha256="0" * 64,
        )

    # The service catches this validation error and substitutes a generic
    # fenced protocol error. Fields are also repr-hidden for safe diagnostics.
    safe_report = LiveLogChunkReport(
        site_id="remote-1",
        job_id="job-1",
        attempt_id="attempt-1",
        fencing_token=token,
        offset=0,
        data_base64=base64.b64encode(secret_output).decode("ascii"),
        sha256=hashlib.sha256(secret_output).hexdigest(),
    )
    assert token not in repr(safe_report)
    assert secret_output.decode() not in repr(safe_report)
    assert "digest does not match" in str(captured.value)


def _descriptor(data: bytes) -> AttemptLogDescriptor:
    return AttemptLogDescriptor(
        logical_path="jobs/job-1/attempts/attempt-1/logs/attempt.log",
        end_offset=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
