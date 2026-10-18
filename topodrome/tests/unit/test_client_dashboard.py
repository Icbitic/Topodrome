from __future__ import annotations

import io
import os
import re
from datetime import UTC, datetime
from typing import Any

import pytest

from topodrome.client.dashboard import (
    DashboardRefreshError,
    render_dashboard,
    watch_dashboard,
)


def _snapshot() -> dict[str, Any]:
    return {
        "generated_at": "2026-09-08T00:00:00Z",
        "sites": [
            {
                "site_id": "lab1",
                "kind": "lab",
                "state": "ready",
                "heartbeat_at": "2026-09-07T23:59:58Z",
                "storage_available": True,
                "queue_depth": 1,
                "region": "local",
            }
        ],
        "gpus": [
            {
                "site_id": "lab1",
                "index": 0,
                "name": "NVIDIA A800",
                "total_memory_bytes": 80 * 1024**3,
                "free_memory_bytes": 20 * 1024**3,
                "utilization_percent": 75,
                "temperature_c": 42,
                "healthy": True,
                "active_lease": True,
                "foreign_process_count": 0,
            },
            {
                "site_id": "lab1",
                "index": 1,
                "name": "NVIDIA A800",
                "total_memory_bytes": 80 * 1024**3,
                "free_memory_bytes": 79 * 1024**3,
                "utilization_percent": 0,
                "healthy": True,
                "foreign_process_count": 1,
            },
        ],
        "jobs": [
            {
                "job_id": "job_123",
                "state": "running",
                "site_id": "lab1",
                "gpu_count": 2,
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            }
        ],
        "kernels": [
            {
                "session_id": "kernel_123",
                "state": "running",
                "mode": "service",
                "site_id": "lab1",
            }
        ],
        "autodl_instances": [
            {
                "name": "topodrome-worker",
                "state": "stopped",
                "region": "bj-B2",
                "gpu_spec": "5090-p",
                "gpu_count": 2,
                "observed_gpu_count": 1,
                "shape_mismatch": True,
            }
        ],
    }


def test_dashboard_renders_consistent_system_snapshot_without_ansi() -> None:
    rendered = render_dashboard(
        _snapshot(),
        now=datetime(2026, 9, 8, 0, 0, 3, tzinfo=UTC),
        color=False,
        width=140,
    )
    assert "snapshot 3s ago" in rendered
    assert "jobs running=1" in rendered
    assert "lab1" in rendered
    assert "60.0G/80.0G" in rendered
    assert "42C" in rendered
    assert "leased" in rendered
    assert "foreign" in rendered
    assert "meta-llama/Meta-Llama-3-8B-Instruct" in rendered
    assert "topodrome-worker" in rendered
    assert "2/1!" in rendered
    assert "\x1b" not in rendered


def test_gpu_owner_and_utilization_colors_preserve_plain_alignment() -> None:
    snapshot = _snapshot()
    snapshot["gpus"][1]["utilization_percent"] = 95
    colored = render_dashboard(snapshot, color=True, width=140)
    plain = render_dashboard(snapshot, color=False, width=140)
    assert re.sub(r"\x1b\[[0-9;]*m", "", colored) == plain
    assert "\x1b[36mleased\x1b[0m" in colored
    assert "\x1b[33mforeign\x1b[0m" in colored
    assert re.search(r"\x1b\[33m *75% *\x1b\[0m", colored)
    assert re.search(r"\x1b\[31m *95% *\x1b\[0m", colored)


def test_dashboard_header_reports_active_policy_generation() -> None:
    snapshot = _snapshot()
    snapshot["policy"] = {
        "status": "active",
        "generation": 4,
        "pending_generation": None,
        "blocker_count": 0,
    }

    first_line = render_dashboard(snapshot, color=False, width=140).splitlines()[0]

    assert "policy g4 blockers=0" in first_line
    assert "->" not in first_line


def test_dashboard_header_reports_pending_policy_and_drain_blockers() -> None:
    snapshot = _snapshot()
    snapshot["policy"] = {
        "status": "draining",
        "generation": 4,
        "pending_generation": 5,
        "blocker_count": 2,
    }

    first_line = render_dashboard(snapshot, color=False, width=140).splitlines()[0]

    assert "policy g4->g5 blockers=2" in first_line


def test_dashboard_policy_indicator_respects_width_and_color_alignment() -> None:
    snapshot = _snapshot()
    snapshot["policy"] = {
        "status": "draining",
        "generation": 4,
        "pending_generation": 5,
        "blocker_count": 2,
    }

    plain = render_dashboard(snapshot, color=False, width=52, height=12)
    colored = render_dashboard(snapshot, color=True, width=52, height=12)

    assert re.sub(r"\x1b\[[0-9;]*m", "", colored) == plain
    assert len(plain.splitlines()) <= 12
    assert all(len(line) <= 52 for line in plain.splitlines())


def test_dashboard_labels_external_gpu_reservations() -> None:
    snapshot = _snapshot()
    gpus = [dict(row) for row in snapshot["gpus"]]
    gpus[1]["externally_reserved"] = True
    snapshot["gpus"] = gpus

    rendered = render_dashboard(snapshot, color=False, width=140)

    reserved = next(line for line in rendered.splitlines() if " 1   NVIDIA" in line)
    assert reserved.endswith("reserved")


def test_dashboard_marks_gpu_telemetry_from_inactive_site_as_stale() -> None:
    snapshot = _snapshot()
    site = dict(snapshot["sites"][0])
    site["state"] = "stopped"
    snapshot["sites"] = [site]
    gpus = [dict(row) for row in snapshot["gpus"]]
    gpus[0]["warm_actor"] = True
    snapshot["gpus"] = gpus

    rendered = render_dashboard(snapshot, color=False, width=140)

    gpu_row = next(line for line in rendered.splitlines() if "60.0G/80.0G" in line)
    header = next(line for line in rendered.splitlines() if "USED/TOTAL" in line)
    assert gpu_row.index("stale") == header.index("HEALTH")
    assert gpu_row.endswith("inactive")
    assert " ok " not in gpu_row
    assert "warm" not in gpu_row


@pytest.mark.parametrize("width", [80, 140])
def test_dashboard_color_does_not_change_visible_alignment(width: int) -> None:
    now = datetime(2026, 9, 8, 0, 0, 3, tzinfo=UTC)
    plain = render_dashboard(_snapshot(), color=False, now=now, width=width)
    colored = render_dashboard(_snapshot(), color=True, now=now, width=width)

    assert re.sub(r"\x1b\[[0-9;]*m", "", colored) == plain


def test_dashboard_shows_job_name() -> None:
    snapshot = _snapshot()
    snapshot["jobs"][0]["name"] = "lens seed 42"
    text = render_dashboard(snapshot, color=False, width=140)
    assert "NAME" in text and "lens seed 42" in text and "  123 " in text
    assert "job_123" not in text


def test_dashboard_active_kernels_and_unfinished_jobs_first() -> None:
    snapshot = _snapshot()
    snapshot["kernels"] = [
        {"session_id": f"kernel-{state}", "state": state}
        for state in ("stopped", "lost", "requested", "starting", "running", "idle", "stopping")
    ]
    snapshot["jobs"] = [
        {"job_id": f"job-{state}", "name": f"job-{state}", "state": state}
        for state in ("succeeded", "queued", "retry_wait", "running")
    ]
    text = render_dashboard(snapshot, color=False, width=140)
    assert "kernels 5" in text
    assert "kernel-stopped" not in text and "kernel-lost" not in text
    assert "kernel-idle" in text
    assert text.index("job-running") < text.index("job-queued") < text.index("job-succeeded")
    assert text.index("job-retry_wait") < text.index("job-succeeded")


def test_freed_kernel_rows_go_to_jobs_and_jobs_are_not_capped_at_twelve() -> None:
    snapshot = _snapshot()
    snapshot["jobs"] = [
        {"job_id": f"job-extra-{i}", "name": f"job-extra-{i}", "state": "queued"} for i in range(30)
    ]
    snapshot["kernels"] = [{"session_id": f"kernel-{i}", "state": "running"} for i in range(8)]
    with_kernels = render_dashboard(snapshot, color=False, width=140, height=50)
    for row in snapshot["kernels"]:
        row["state"] = "stopped"
    without_kernels = render_dashboard(snapshot, color=False, width=140, height=50)
    assert without_kernels.count("job-extra-") > with_kernels.count("job-extra-")
    assert without_kernels.count("job-extra-") > 12
    assert len(without_kernels.splitlines()) <= 50


def test_dashboard_headers_share_column_starts_with_values() -> None:
    lines = render_dashboard(
        _snapshot(),
        color=False,
        width=140,
        now=datetime(2026, 9, 8, 0, 0, 3, tzinfo=UTC),
    ).splitlines()
    site_header = next(line for line in lines if "HEARTBEAT" in line)
    site_row = next(line for line in lines if line.startswith("  lab1") and "ready" in line)
    gpu_header = next(line for line in lines if "USED/TOTAL" in line)
    gpu_row = next(line for line in lines if "60.0G/80.0G" in line)
    job_header = next(line for line in lines if line.startswith("  JOB"))
    job_row = next(line for line in lines if line.startswith("  123 "))

    assert site_header.index("HEARTBEAT") == site_row.index("5s")
    assert site_header.index("REGION") == site_row.index("local")
    assert site_header.index("QUEUE") == site_row.index("1", site_header.index("QUEUE"))
    assert gpu_header.index("IDX") == gpu_row.index("0")
    assert gpu_header.index("VRAM") == gpu_row.index("[")
    assert gpu_header.index("USED/TOTAL") == gpu_row.index("60.0G/80.0G")
    assert gpu_header.index("UTIL") == gpu_row.index("75%")
    assert gpu_header.index("TEMP") == gpu_row.index("42C")
    assert gpu_header.index("HEALTH") == gpu_row.index("ok")
    assert gpu_header.index("OWNER") == gpu_row.index("leased")
    assert job_header.index("GPU") == job_row.index("2", job_header.index("GPU"))
    assert job_header.index("MODEL") == job_row.index("meta-llama")


def test_dashboard_neutralizes_terminal_controls_from_status_labels() -> None:
    snapshot = _snapshot()
    snapshot["sites"] = [
        {
            "site_id": "host\x1b[2J\rspoof\x07",
            "kind": "lab",
            "state": "ready",
            "heartbeat_at": "2026-09-08T00:00:00Z",
            "storage_available": True,
        }
    ]

    rendered = render_dashboard(snapshot, color=False)

    assert "\x1b" not in rendered
    assert "\r" not in rendered
    assert "\x07" not in rendered


@pytest.mark.parametrize("heartbeat", [None, "not-a-timestamp"])
def test_dashboard_renders_unavailable_heartbeat_as_unknown(heartbeat: object) -> None:
    snapshot = _snapshot()
    site = dict(snapshot["sites"][0])
    site["heartbeat_at"] = heartbeat
    snapshot["sites"] = [site]

    rendered = render_dashboard(
        snapshot,
        now=datetime(2026, 9, 8, 0, 0, 3, tzinfo=UTC),
        color=False,
        width=140,
    )
    site_line = next(line for line in rendered.splitlines() if line.strip().startswith("lab1"))

    assert "unknown" in site_line
    assert " 0s " not in site_line


def test_watch_once_fetches_exactly_once() -> None:
    output = io.StringIO()
    calls = 0

    def fetch() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _snapshot()

    watch_dashboard(fetch, once=True, output=output, color=False)
    assert calls == 1
    assert output.getvalue().startswith("TOPODROME")


@pytest.mark.parametrize(("width", "height"), [(20, 12), (52, 18), (123, 24)])
def test_dashboard_fits_one_terminal_page_and_preserves_sections(
    width: int,
    height: int,
) -> None:
    rendered = render_dashboard(
        _snapshot(),
        now=datetime(2026, 9, 8, 0, 0, 3, tzinfo=UTC),
        color=True,
        width=width,
        height=height,
    )
    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered)
    lines = plain.splitlines()

    assert len(lines) <= height
    assert all(len(line) <= width for line in lines)
    assert all(
        heading in lines for heading in ("SITES", "GPUS", "RECENT JOBS", "KERNELS", "AUTODL")
    )


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def _crowded_snapshot() -> dict[str, Any]:
    snapshot = _snapshot()
    snapshot["sites"] = [
        {
            "site_id": site_id,
            "kind": kind,
            "state": state,
            "heartbeat_at": "2026-09-08T00:00:00Z",
            "storage_available": True,
        }
        for site_id, kind, state in (
            ("lab1", "lab", "ready"),
            ("remote-jupyter-1", "jupyter", "ready"),
            ("autodl", "autodl", "stopped"),
        )
    ]
    snapshot["gpus"] = [
        {
            "site_id": site_id,
            "index": index,
            "name": "NVIDIA A800",
            "total_memory_bytes": 80 * 1024**3,
            "free_memory_bytes": (20 + index) * 1024**3,
            "utilization_percent": index,
            "healthy": True,
        }
        for site_id in ("lab1", "remote-jupyter-1")
        for index in range(8)
    ]
    snapshot["gpus"].append(
        {
            "site_id": "autodl",
            "index": 0,
            "name": "RTX 5090",
            "total_memory_bytes": 32 * 1024**3,
            "free_memory_bytes": 31 * 1024**3,
            "utilization_percent": 0,
            "healthy": True,
        }
    )
    snapshot["jobs"] = [
        {
            "job_id": f"job_{index}",
            "state": "succeeded",
            "site_id": "lab1",
            "gpu_count": 1,
            "model": "test/model",
        }
        for index in range(12)
    ]
    snapshot["kernels"] = [
        {
            "session_id": f"kernel_{index}",
            "state": "stopped",
            "mode": "raw-model",
            "site_id": "lab1",
        }
        for index in range(6)
    ]
    snapshot["autodl_instances"] = [
        {
            "name": f"instance-{index}",
            "state": "released",
            "region": "bj",
            "gpu_spec": "5090",
            "gpu_count": 1,
        }
        for index in range(3)
    ]
    return snapshot


def test_dashboard_prioritizes_complete_gpu_inventory_over_history() -> None:
    rendered = render_dashboard(
        _crowded_snapshot(),
        color=False,
        width=123,
        height=42,
    )
    lines = rendered.splitlines()

    assert len(lines) <= 42
    assert sum("NVIDIA A800" in line or "RTX 5090" in line for line in lines) == 17
    assert any("remote-jupyter-1  7" in line for line in lines)
    assert any("more" in line for line in lines[lines.index("RECENT JOBS") :])


def test_active_autodl_instances_keep_rows_when_jobs_fill_page() -> None:
    snapshot = _crowded_snapshot()
    snapshot["jobs"] = [
        {"job_id": f"job-pending-{i}", "name": f"job-pending-{i}", "state": "queued"}
        for i in range(50)
    ]
    snapshot["autodl_instances"] = [
        {"instance_id": "autodl-starting", "state": "starting", "gpu_count": 1},
        {"instance_id": "autodl-running", "state": "running", "gpu_count": 1},
    ]
    text = render_dashboard(snapshot, color=False, width=123, height=42)
    assert "autodl-starting" in text and "autodl-running" in text
    assert "job-pending-0" in text
    assert len(text.splitlines()) <= 42


def test_small_dashboard_samples_each_ready_gpu_site() -> None:
    lines = render_dashboard(
        _crowded_snapshot(),
        color=False,
        width=123,
        height=20,
    ).splitlines()
    gpu_lines = lines[lines.index("GPUS") : lines.index("RECENT JOBS")]

    assert len(lines) <= 20
    assert any(line.strip().startswith("lab1") for line in gpu_lines)
    assert any(line.strip().startswith("remote-jupyter-1") for line in gpu_lines)


def test_watch_uses_alternate_screen_repaints_and_tracks_terminal_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _TTYBuffer()
    calls = 0
    sizes = iter((os.terminal_size((52, 12)), os.terminal_size((96, 18))))
    monkeypatch.setattr(
        "topodrome.client.dashboard.shutil.get_terminal_size",
        lambda _fallback: next(sizes),
    )

    def fetch() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _snapshot()

    def stop_after_two_frames(_interval: float) -> None:
        if calls == 2:
            raise KeyboardInterrupt

    watch_dashboard(
        fetch,
        output=output,
        color=False,
        sleep=stop_after_two_frames,
    )

    value = output.getvalue()
    enter = "\x1b[?1049h\x1b[?25l"
    clear = "\x1b[H\x1b[2J"
    restore = "\x1b[0m\x1b[?25h\x1b[?1049l"
    assert calls == 2
    assert value.startswith(enter + clear)
    assert value.endswith(restore)
    assert value.count(enter) == 1
    assert value.count(clear) == 2
    assert value.count(restore) == 1

    first, second = value.removeprefix(enter).removesuffix(restore).split(clear)[1:]
    assert len(first.splitlines()) <= 12
    assert all(len(line) <= 52 for line in first.splitlines())
    assert len(second.splitlines()) <= 18
    assert all(len(line) <= 96 for line in second.splitlines())


def test_watch_once_on_tty_keeps_complete_snapshot_without_alternate_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _TTYBuffer()
    monkeypatch.setattr(
        "topodrome.client.dashboard.shutil.get_terminal_size",
        lambda _fallback: os.terminal_size((52, 12)),
    )

    watch_dashboard(lambda: _snapshot(), once=True, output=output, color=False)

    rendered = output.getvalue()
    assert rendered.startswith("TOPODROME")
    assert len(rendered.splitlines()) > 12
    assert "\x1b[?1049h" not in rendered
    assert "\x1b[?1049l" not in rendered


def test_watch_redacts_refresh_exception_details() -> None:
    output = io.StringIO()

    def fail() -> dict[str, object]:
        raise RuntimeError("https://host.invalid/?token=do-not-render")

    with pytest.raises(DashboardRefreshError):
        watch_dashboard(fail, once=True, output=output, color=False)
    assert output.getvalue() == "TOPODROME  status refresh failed (RuntimeError)\n"


def test_dashboard_distinguishes_pending_operator_reservation_from_free_gpu() -> None:
    snapshot = _snapshot()
    snapshot["gpus"][0]["operator_reserved"] = True
    snapshot["gpus"][0]["active_lease"] = True

    rendered = render_dashboard(snapshot, width=160, height=40, color=False)

    gpu_row = next(line for line in rendered.splitlines() if " 0   NVIDIA" in line)
    assert gpu_row.endswith("leased+reserved")


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_watch_rejects_invalid_intervals(interval: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        watch_dashboard(lambda: {}, interval_seconds=interval, once=True)
