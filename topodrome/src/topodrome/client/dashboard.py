"""Dependency-free, read-only terminal dashboard for a Topodrome scheduler."""

from __future__ import annotations

import math
import re
import shutil
import sys
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, TextIO

Snapshot = Mapping[str, Any]


class DashboardRefreshError(RuntimeError):
    """A one-shot status refresh failed after a safe diagnostic was rendered."""


def render_dashboard(
    snapshot: Snapshot,
    *,
    now: datetime | None = None,
    color: bool = False,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Render one bounded status snapshot without mutating scheduler state.

    When ``height`` is supplied, the renderer keeps every system section visible
    and uses compact, explicitly truncated sections to fit the terminal page.
    """

    current = now or datetime.now(UTC)
    columns = max(
        1,
        width if width is not None else shutil.get_terminal_size((120, 30)).columns,
    )
    sites = _rows(snapshot.get("sites"))
    gpus = _rows(snapshot.get("gpus"))
    # Also protect clients talking to an older, history-first scheduler.
    jobs = sorted(
        _rows(snapshot.get("jobs")),
        key=lambda row: (
            0
            if row.get("state")
            in {"running", "assigning", "canceling", "finalizing", "downloading"}
            else 1
            if row.get("state") in {"queued", "retry_wait"}
            else 2
        ),
    )
    kernels = [
        row
        for row in _rows(snapshot.get("kernels"))
        if row.get("state") in {"requested", "starting", "running", "idle", "stopping"}
    ]
    providers = _rows(snapshot.get("autodl_instances"))

    counts: dict[str, int] = {}
    for job in jobs:
        state = _text(job.get("state"), "unknown")
        counts[state] = counts.get(state, 0) + 1
    generated = _timestamp(snapshot.get("generated_at"), current)
    age = _age(generated, current)
    title = _paint("TOPODROME", "cyan", color=color)
    policy = _policy_indicator(snapshot.get("policy"), color=color)
    summary = (
        f"{title}  snapshot {age} ago  "
        f"{policy + '  ' if policy else ''}"
        f"sites {len(sites)}  GPUs {len(gpus)}  "
        f"jobs {_state_counts(counts)}  kernels {len(kernels)}"
    )
    header = [_limit_line(summary, columns), "=" * min(columns, 120)]
    sections: list[list[str]] = []

    sections.append(_site_lines(sites, current=current, color=color, width=columns))
    site_states = {
        _text(site.get("site_id")): _text(site.get("state"), "unknown") for site in sites
    }
    sections.append(
        _gpu_lines(
            gpus,
            site_states=site_states,
            color=color,
            width=columns,
        )
    )
    sections.append(_job_lines(jobs, color=color, width=columns))
    sections.append(_kernel_lines(kernels, color=color, width=columns))
    sections.append(_provider_lines(providers, color=color, width=columns))
    lines = _fit_page(
        header,
        sections,
        height=max(1, height) if height is not None else None,
        width=columns,
    )
    return "\n".join(lines).rstrip() + "\n"


def watch_dashboard(
    fetch: Callable[[], Snapshot],
    *,
    interval_seconds: float = 2.0,
    once: bool = False,
    output: TextIO = sys.stdout,
    sleep: Callable[[float], None] = time.sleep,
    color: bool | None = None,
) -> None:
    """Refresh a dashboard until interrupted; the fetch operation is read-only."""

    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("dashboard refresh interval must be positive and finite")
    interactive = bool(output.isatty())
    live_terminal = interactive and not once
    use_color = interactive if color is None else color
    alternate_screen_entered = False
    try:
        if live_terminal:
            output.write("\x1b[?1049h\x1b[?25l")
            output.flush()
            alternate_screen_entered = True
        while True:
            refresh_failed = False
            terminal_size = shutil.get_terminal_size((120, 30))
            columns = max(1, terminal_size.columns)
            rows = max(1, terminal_size.lines)
            try:
                snapshot = fetch()
                frame = render_dashboard(
                    snapshot,
                    color=use_color,
                    width=columns,
                    height=rows if live_terminal else None,
                )
            except KeyboardInterrupt:
                return
            except Exception as exc:
                # Provider and transport exceptions can contain credentials or
                # authenticated URLs.  The class name is sufficient for operators.
                frame = (
                    _limit_line(
                        f"TOPODROME  status refresh failed ({type(exc).__name__})",
                        columns,
                    )
                    + "\n"
                )
                refresh_failed = True
            if live_terminal:
                # Repaint the alternate buffer from its origin.  Avoid a trailing
                # newline so a frame ending on the last terminal row cannot scroll.
                output.write("\x1b[H\x1b[2J")
                output.write(frame.rstrip("\n"))
            else:
                output.write(frame)
            output.flush()
            if once:
                if refresh_failed:
                    raise DashboardRefreshError from None
                return
            try:
                sleep(interval_seconds)
            except KeyboardInterrupt:
                return
    finally:
        if alternate_screen_entered:
            output.write("\x1b[0m\x1b[?25h\x1b[?1049l")
            output.flush()


def _fit_page(
    header: Sequence[str],
    sections: Sequence[Sequence[str]],
    *,
    height: int | None,
    width: int,
) -> list[str]:
    """Fit status sections into one page while preserving their headings."""

    bounded_header = [_limit_line(line, width) for line in header]
    bounded_sections = [[_limit_line(line, width) for line in section] for section in sections]
    complete = [
        *bounded_header,
        *(line for section in bounded_sections for line in section),
    ]
    if height is None or len(complete) <= height:
        return complete

    # A section begins with a visual spacer.  Under height pressure the spacer
    # is expendable, but its heading is retained before any detail rows.
    compact_sections = [
        list(section[1:]) if section and not section[0] else list(section)
        for section in bounded_sections
    ]
    # The separator is decorative.  Once compaction is needed, spend that row
    # on live data instead.
    page_header = bounded_header[:1]
    minimum_with_all_headings = len(page_header) + len(compact_sections)
    if height < minimum_with_all_headings:
        headings = [section[0] for section in compact_sections if section]
        return [*page_header, *headings][:height]

    budgets = [1 if section else 0 for section in compact_sections]
    remaining = height - len(page_header) - sum(budgets)

    # Live site/GPU telemetry is authoritative; the other sections are bounded
    # history.  First make the two inventory tables intelligible, then allocate
    # enough GPU rows to represent each site before spending space on history.
    for index in (0, 1):
        if remaining <= 0:
            break
        if index < len(compact_sections) and budgets[index] < len(compact_sections[index]):
            budgets[index] += 1
            remaining -= 1

    critical_targets = [0 for _ in compact_sections]
    if compact_sections:
        critical_targets[0] = len(compact_sections[0])
    if len(compact_sections) > 1:
        gpu_groups = _display_group_count(compact_sections[1][2:])
        # A truncated table needs a heading, column header, one row per site,
        # and an omission marker.  ``min`` naturally removes the marker when
        # the entire table already fits.
        critical_targets[1] = min(
            len(compact_sections[1]),
            3 + gpu_groups,
        )
    while remaining > 0:
        advanced = False
        for index in (0, 1):
            if index < len(compact_sections) and budgets[index] < critical_targets[index]:
                budgets[index] += 1
                remaining -= 1
                advanced = True
                if remaining <= 0:
                    break
        if not advanced:
            break

    # Keep every remaining section named even in a very short window, then add
    # a summary/header line to each when the hardware view leaves room.
    for index in (3, 4, 2):
        if remaining <= 0:
            break
        if index < len(compact_sections) and budgets[index] < len(compact_sections[index]):
            budgets[index] += 1
            remaining -= 1

    # At ordinary terminal heights all physical GPUs should win over old job
    # and stopped-kernel rows.  This avoids hiding the free tail of a remote
    # eight-GPU host while displaying historical records.
    for index in (0, 1):
        while (
            remaining > 0
            and index < len(compact_sections)
            and budgets[index] < len(compact_sections[index])
        ):
            budgets[index] += 1
            remaining -= 1

    # Provider rows are active capacity, not history (the snapshot excludes
    # released/stopped instances). Keep them and live kernels visible before
    # spending the remaining page on jobs.
    for index in (3, 4, 2):
        while (
            remaining > 0
            and index < len(compact_sections)
            and budgets[index] < len(compact_sections[index])
        ):
            budgets[index] += 1
            remaining -= 1

    rendered_sections = [
        _truncate_section(
            section,
            budget=budget,
            width=width,
            balance_groups=index == 1,
        )
        for index, (section, budget) in enumerate(zip(compact_sections, budgets, strict=True))
    ]

    # If all useful content fits but not every spacer does, use any spare rows
    # to restore visual separation in document order.
    spacers = min(remaining, len(rendered_sections))
    lines = list(page_header)
    for index, section in enumerate(rendered_sections):
        if index < spacers:
            lines.append("")
        lines.extend(section)
    return lines[:height]


def _truncate_section(
    section: Sequence[str],
    *,
    budget: int,
    width: int,
    balance_groups: bool = False,
) -> list[str]:
    if not section or budget <= 0:
        return []
    if len(section) <= budget:
        return list(section)
    if budget == 1:
        return [section[0]]
    visible_details = max(0, budget - 2)
    omitted = len(section) - 1 - visible_details
    marker = _limit_line(f"  … {omitted} more", width)
    details = list(section[1 : 1 + visible_details])
    if balance_groups and visible_details >= 2:
        details = [section[1], *_balanced_group_rows(section[2:], visible_details - 1)]
    return [section[0], *details, marker]


def _balanced_group_rows(rows: Sequence[str], limit: int) -> list[str]:
    """Sample grouped GPU rows while keeping every visible site represented."""

    if limit <= 0 or not rows:
        return []
    if len(rows) <= limit:
        return list(rows)
    groups: dict[str, list[tuple[int, str]]] = {}
    for index, row in enumerate(rows):
        site = _display_group(row) or f"row-{index}"
        groups.setdefault(site, []).append((index, row))

    allocations = {site: 0 for site in groups}
    remaining = limit
    while remaining > 0:
        advanced = False
        for site, members in groups.items():
            if remaining <= 0:
                break
            if allocations[site] < len(members):
                allocations[site] += 1
                remaining -= 1
                advanced = True
        if not advanced:
            break

    selected: list[tuple[int, str]] = []
    for site, members in groups.items():
        count = allocations[site]
        if count == 1:
            selected.append(members[0])
        elif count > 1:
            # Include both ends of a site's index range, then distribute any
            # additional samples evenly between them.
            positions = {
                round(offset * (len(members) - 1) / (count - 1)) for offset in range(count)
            }
            selected.extend(members[position] for position in sorted(positions))
    return [row for _, row in sorted(selected)]


def _display_group_count(rows: Sequence[str]) -> int:
    return len({_display_group(row) for row in rows if _display_group(row)})


def _display_group(row: str) -> str:
    plain = _ANSI_SEQUENCE.sub("", row).strip()
    return plain.split(maxsplit=1)[0] if plain else ""


def _site_lines(
    rows: Sequence[Mapping[str, Any]],
    *,
    current: datetime,
    color: bool,
    width: int,
) -> list[str]:
    lines = ["", _paint("SITES", "bold", color=color)]
    if not rows:
        return [*lines, "  no sites reported"]
    lines.append(
        f"  {_cell('SITE', 20)} {_cell('KIND', 10)} {_cell('STATE', 11)} "
        f"{_cell('HEARTBEAT', 9)} {_cell('REGION', 11)} {_cell('QUEUE', 5)} STORAGE"
    )
    for row in sorted(rows, key=lambda item: _text(item.get("site_id"))):
        heartbeat = _optional_timestamp(row.get("heartbeat_at"))
        heartbeat_age = "unknown" if heartbeat is None else _age(heartbeat, current)
        state = _text(row.get("state"), "unknown")
        line = (
            f"  {_cell(_text(row.get('site_id')), 20)} "
            f"{_cell(_text(row.get('kind')), 10)} "
            f"{_paint(_cell(state, 11), _state_color(state), color=color)} "
            f"{_cell(heartbeat_age, 9)} "
            f"{_cell(_text(row.get('region'), '-'), 11)} "
            f"{_cell(str(_integer(row.get('queue_depth'))), 5)} "
            f"{'ok' if bool(row.get('storage_available')) else 'down'}"
        )
        lines.append(_limit_line(line, width))
    return lines


def _gpu_lines(
    rows: Sequence[Mapping[str, Any]],
    *,
    site_states: Mapping[str, str],
    color: bool,
    width: int,
) -> list[str]:
    lines = ["", _paint("GPUS", "bold", color=color)]
    if not rows:
        return [*lines, "  no GPUs reported"]
    lines.append(
        f"  {_cell('SITE', 17)} {_cell('IDX', 3)} {_cell('GPU', 20)} "
        f"{_cell('VRAM', 14)} {_cell('USED/TOTAL', 11)} "
        f"{_cell('UTIL', 4)} {_cell('TEMP', 4)} "
        f"{_cell('HEALTH', 6)} OWNER"
    )
    ordered = sorted(
        rows,
        key=lambda item: (
            site_states.get(_text(item.get("site_id")), "unknown").casefold() != "ready",
            _text(item.get("site_id")),
            _integer(item.get("index")),
        ),
    )
    for row in ordered:
        site_id = _text(row.get("site_id"))
        telemetry_current = site_states.get(site_id, "unknown").casefold() == "ready"
        total = _integer(row.get("total_memory_bytes"))
        free = min(total, _integer(row.get("free_memory_bytes"))) if total else 0
        used = max(0, total - free)
        ratio = used / total if total else 0.0
        bar = _bar(ratio, 12)
        utilization = _number(row.get("utilization_percent"))
        healthy = bool(row.get("healthy"))
        health_label = "ok" if healthy else "bad"
        health_color = "green" if healthy else "red"
        if not telemetry_current:
            health_label = "stale"
            health_color = "yellow"
        health = _paint(
            _cell(health_label, 6),
            health_color,
            color=color,
        )
        owner = _gpu_owner(row) if telemetry_current else "inactive"
        owner_color = _gpu_owner_color(owner)
        temperature = row.get("temperature_c")
        temperature_text = "-" if temperature is None else f"{_number(temperature):.0f}C"
        memory_text = f"{_gib(used)}/{_gib(total)}"
        utilization_text = f"{utilization:.0f}%"
        utilization_cell = _paint(
            _cell(utilization_text, 4),
            _gpu_util_color(utilization) if telemetry_current else None,
            color=color,
        )
        line = (
            f"  {_cell(site_id, 17)} "
            f"{_cell(str(_integer(row.get('index'))), 3)} "
            f"{_cell(_text(row.get('name')), 20)} "
            f"{_cell(bar, 14)} {_cell(memory_text, 11)} "
            f"{utilization_cell} {_cell(temperature_text, 4)} "
            f"{health} "
            f"{_paint(owner, owner_color, color=color)}"
        )
        lines.append(_limit_line(line, width))
    return lines


def _job_lines(
    rows: Sequence[Mapping[str, Any]],
    *,
    color: bool,
    width: int,
) -> list[str]:
    lines = ["", _paint("RECENT JOBS", "bold", color=color)]
    if not rows:
        return [*lines, "  no jobs reported"]
    if width >= 110:
        columns = [
            ("JOB", 4),
            ("NAME", 22),
            ("STATE", 14),
            ("SITE", 17),
            ("GPU", 3),
            ("PROGRESS", max(30, width - 103)),
            ("MODEL", 35),
        ]
    elif width >= 80:
        columns = [
            ("JOB", 4),
            ("NAME", 16),
            ("STATE", 10),
            ("SITE", 12),
            ("GPU", 3),
            ("PROGRESS", width - 52),
        ]
    elif width >= 60:
        columns = [("JOB", 4), ("STATE", 10), ("PROGRESS", width - 29), ("SITE", 10)]
    else:
        state_width = 10 if width >= 24 else 3
        columns = [
            ("JOB", 4),
            ("STATE", state_width),
            ("PROGRESS", max(1, width - 8 - state_width)),
        ]
    progress_width = dict(columns)["PROGRESS"]
    lines.append("  " + " ".join(_cell(label, size) for label, size in columns))
    for row in rows:
        state = _text(row.get("state"), "unknown")
        values = {
            "JOB": _text(row.get("job_id") or row.get("id")).removeprefix("job_")[:4],
            "NAME": _text(row.get("name"), "-"),
            "STATE": state,
            "SITE": _text(row.get("site_id"), "-"),
            "GPU": str(_integer(row.get("gpu_count") or row.get("gpu"))),
            "PROGRESS": _progress_text(row.get("progress"), width=progress_width),
            "MODEL": _text(row.get("model"), "-"),
        }
        cells = [
            _paint(_cell(values[label], size), _state_color(state), color=color)
            if label == "STATE"
            else _cell(values[label], size)
            for label, size in columns
        ]
        line = "  " + " ".join(cells)
        lines.append(_limit_line(line, width))
    return lines


def _progress_text(progress: Any, *, width: int = 30) -> str:
    if not isinstance(progress, Mapping):
        return "-"
    current = _number(progress.get("current", 0))
    total = _number(progress.get("total"))
    if progress.get("unit") == "bytes":
        stage = _text(progress.get("stage"), "finalizing")
        prefix = "~ " if progress.get("stale") else ""
        percent = f" {100 * current / total:.0f}%" if total > 0 else ""
        counts = f" {_transfer_bytes(current)}/{_transfer_bytes(total)}" if total > 0 else ""
        return _clip(prefix + stage + percent + counts, width)
    count = f"{current:g}"
    percent = ""
    if total > 0:
        percent = f"{100 * (current / total):.0f}%"
        count += f"/{total:g}"
    stale = bool(progress.get("stale"))
    prefix = "~ " if stale else ""
    eta = progress.get("eta_seconds")
    eta_text = ""
    if not stale and isinstance(eta, (int, float)) and math.isfinite(eta) and eta >= 0:
        eta_text = f"ETA {_duration(eta)}"
    core = " ".join(part for part in (percent, count, eta_text) if part)
    # A verbose counter must not push percentage/ETA out either.
    if percent and _display_width(prefix + core) > width:
        core = " ".join(part for part in (percent, eta_text) if part)
    result = prefix + core
    stage = _text(progress.get("stage"))
    message = _text(progress.get("message"))
    remaining = width - _display_width(result) - 1
    if stage and remaining > 0:
        result += " " + _clip(stage, min(12, remaining) if message else remaining)
    remaining = width - _display_width(result) - 1
    if message and remaining > 0:
        result += " " + _clip(message, remaining)
    return _clip(result, width)


def _transfer_bytes(value: float) -> str:
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or suffix == "TiB":
            return f"{value:.1f}{suffix}" if suffix != "B" else f"{value:.0f}B"
        value /= 1024
    return "0B"


def _kernel_lines(
    rows: Sequence[Mapping[str, Any]],
    *,
    color: bool,
    width: int,
) -> list[str]:
    lines = ["", _paint("KERNELS", "bold", color=color)]
    if not rows:
        return lines
    lines.append(f"  {_cell('SESSION', 22)} {_cell('STATE', 12)} {_cell('MODE', 13)} SITE")
    for row in rows:
        state = _text(row.get("state"), "unknown")
        line = (
            f"  {_cell(_text(row.get('session_id') or row.get('id')), 22)} "
            f"{_paint(_cell(state, 12), _state_color(state), color=color)} "
            f"{_cell(_text(row.get('mode')), 13)} "
            f"{_clip(_text(row.get('site_id'), '-'), 20)}"
        )
        lines.append(_limit_line(line, width))
    return lines


def _provider_lines(
    rows: Sequence[Mapping[str, Any]],
    *,
    color: bool,
    width: int,
) -> list[str]:
    lines = ["", _paint("AUTODL", "bold", color=color)]
    if not rows:
        return [*lines, "  no managed instances"]
    lines.append(
        f"  {_cell('INSTANCE', 22)} {_cell('STATE', 12)} {_cell('REGION', 12)} "
        f"{_cell('GPU', 8)} {_cell('COUNT', 5)} IDLE"
    )
    for row in rows:
        state = _text(row.get("state"), "unknown")
        idle = row.get("idle_seconds")
        idle_text = "-" if idle is None else _duration(_number(idle))
        requested = _integer(row.get("gpu_count"))
        observed = _integer(row.get("observed_gpu_count"))
        shape = (
            f"{requested}/{observed}!"
            if bool(row.get("shape_mismatch")) and observed
            else str(requested)
        )
        line = (
            f"  {_cell(_text(row.get('name') or row.get('instance_id') or row.get('id')), 22)} "
            f"{_paint(_cell(state, 12), _state_color(state), color=color)} "
            f"{_cell(_text(row.get('region'), '-'), 12)} "
            f"{_cell(_text(row.get('gpu_spec'), '-'), 8)} "
            f"{_cell(shape, 5)} {idle_text}"
        )
        lines.append(_limit_line(line, width))
    return lines


def _rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _gpu_owner(row: Mapping[str, Any]) -> str:
    explicit = row.get("owner")
    if isinstance(explicit, str) and explicit:
        return explicit
    reserved = bool(row.get("externally_reserved")) or bool(row.get("operator_reserved"))
    foreign = bool(_integer(row.get("foreign_process_count")))
    if bool(row.get("active_lease")):
        parts = ["leased"]
        if reserved:
            parts.append("reserved")
        if foreign:
            parts.append("foreign")
        return "+".join(parts)
    if bool(row.get("warm_replica")) or bool(row.get("warm_actor")):
        parts = ["warm"]
        if reserved:
            parts.append("reserved")
        if foreign:
            parts.append("foreign")
        return "+".join(parts)
    if reserved:
        return "reserved"
    if foreign:
        return "foreign"
    return "free"


def _gpu_owner_color(owner: str) -> str | None:
    parts = set(owner.split("+"))
    if parts & {"reserved", "foreign"}:
        return "yellow"
    if parts & {"leased", "warm"}:
        return "cyan"
    return "green" if owner == "free" else None


def _gpu_util_color(utilization: float) -> str | None:
    if not 0 <= utilization <= 100:
        return None
    return "green" if utilization < 30 else "yellow" if utilization < 80 else "red"


def _timestamp(value: object, default: datetime) -> datetime:
    return _optional_timestamp(value) or default


def _optional_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _age(then: datetime, now: datetime) -> str:
    return _duration(max(0.0, (now - then).total_seconds()))


def _duration(seconds: float) -> str:
    whole = max(0, int(seconds))
    if whole < 60:
        return f"{whole}s"
    minutes, second = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m{second:02d}s"
    hours, minute = divmod(minutes, 60)
    return f"{hours}h{minute:02d}m"


def _gib(value: int) -> str:
    return f"{value / 1024**3:.1f}G"


def _bar(ratio: float, length: int) -> str:
    filled = min(length, max(0, round(ratio * length)))
    return "[" + "#" * filled + "-" * (length - filled) + "]"


def _state_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "0"
    preferred = (
        "running",
        "finalizing",
        "queued",
        "assigning",
        "retry_wait",
        "failed",
        "succeeded",
    )
    labels = [f"{name}={counts[name]}" for name in preferred if counts.get(name)]
    labels.extend(
        f"{name}={count}"
        for name, count in sorted(counts.items())
        if name not in preferred and count
    )
    return ",".join(labels)


def _policy_indicator(value: object, *, color: bool) -> str:
    """Render the active and, while draining, staged scheduling generation."""

    if not isinstance(value, Mapping):
        return ""
    generation = _integer(value.get("generation"))
    if generation < 1:
        return ""
    pending_generation = _integer(value.get("pending_generation"))
    blockers = max(0, _integer(value.get("blocker_count")))
    if pending_generation > 0:
        label = f"policy g{generation}->g{pending_generation} blockers={blockers}"
        return _paint(label, "yellow", color=color)
    return _paint(f"policy g{generation} blockers={blockers}", "green", color=color)


def _state_color(state: str) -> str:
    normalized = state.casefold()
    if normalized in {"ready", "running", "finalizing", "succeeded", "stopped"}:
        return "green"
    if normalized in {"failed", "lost", "error", "released"}:
        return "red"
    if normalized in {"queued", "assigning", "starting", "stopping", "draining"}:
        return "yellow"
    return "cyan"


_ANSI = {
    "bold": "1",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
}

_ANSI_SEQUENCE = re.compile(r"\x1b\[[0-9;]*m")


def _paint(value: str, style: str | None, *, color: bool) -> str:
    if not color or style is None:
        return value
    return f"\x1b[{_ANSI[style]}m{value}\x1b[0m"


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = "".join(
        " " if unicodedata.category(character) in {"Cc", "Cf"} else character
        for character in str(value)
    )
    return text or default


def _clip(value: str, limit: int) -> str:
    if _display_width(value) <= limit:
        return value
    target = max(0, limit - 1)
    clipped: list[str] = []
    width = 0
    for character in value:
        character_width = _character_width(character)
        if width + character_width > target:
            break
        clipped.append(character)
        width += character_width
    return "".join(clipped) + "…"


def _cell(value: str, width: int, *, align: str = "left") -> str:
    clipped = _clip(value, width)
    padding = " " * max(0, width - _display_width(clipped))
    return padding + clipped if align == "right" else clipped + padding


def _limit_line(value: str, width: int) -> str:
    """Clip a colored line by terminal cells without counting ANSI controls."""

    if _display_width(_ANSI_SEQUENCE.sub("", value)) <= width:
        return value
    output: list[str] = []
    visible = 0
    active_style = False
    position = 0
    while position < len(value):
        sequence = _ANSI_SEQUENCE.match(value, position)
        if sequence is not None:
            code = sequence.group(0)
            output.append(code)
            active_style = code != "\x1b[0m"
            position = sequence.end()
            continue
        character = value[position]
        character_width = _character_width(character)
        if visible + character_width > width:
            break
        output.append(character)
        visible += character_width
        position += 1
    if active_style:
        output.append("\x1b[0m")
    return "".join(output)


def _display_width(value: str) -> int:
    return sum(_character_width(character) for character in value)


def _character_width(character: str) -> int:
    if unicodedata.combining(character):
        return 0
    return 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1


def _integer(value: object) -> int:
    if not isinstance(value, str | int | float | bool):
        return 0
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _number(value: object) -> float:
    if not isinstance(value, str | int | float | bool):
        return 0.0
    try:
        result = float(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return result if math.isfinite(result) else 0.0


__all__ = ["DashboardRefreshError", "render_dashboard", "watch_dashboard"]
