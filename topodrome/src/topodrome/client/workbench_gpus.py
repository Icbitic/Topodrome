"""Read-only process presentation shared by GPU rows and their detail page."""

from datetime import UTC, datetime

from .dashboard import _gpu_owner


def gpu_key(row):
    return row.get("_row_key") or f"{row['site_id']}/{row['uuid']}"


def process_status(row):
    state = row.get("processes_status", "unavailable")
    if row.get("processes") is None:
        return "unavailable"
    if state == "current":
        try:
            received = datetime.fromisoformat(
                str(row["processes_received_at"]).replace("Z", "+00:00")
            )
            age = (datetime.now(UTC) - received).total_seconds()
            if age > row.get("processes_ttl_seconds", 30):
                return "stale"
        except (ValueError, KeyError, TypeError):
            return "unavailable"
    return state


def process_memory(process):
    size = process.get("used_memory_bytes")
    if not isinstance(size, (int, float)) or size <= 0:
        return "Not reported"
    return f"{size / 1024**3:.1f} GiB" if size >= 1024**3 else f"{size / 1024**2:.0f} MiB"


def process_name(process):
    name = process.get("process_name") or "Unknown process"
    script = process.get("script_name")
    return f"{name} · {script}" if script and script != name else name


def process_kind(process):
    return "idle worker" if process.get("kind") == "cooperative" else "foreign"


def process_entries(row):
    state = process_status(row)
    if state == "unavailable":
        return [{"info": "Process details unavailable; waiting for a current heartbeat."}]
    result = []
    if state != "current":
        result.append({"info": "Stale process sample; these PIDs may have exited."})
    result.extend(
        sorted(row.get("processes") or [], key=lambda p: (-p.get("used_memory_bytes", 0), p["pid"]))
    )
    if not row.get("processes"):
        result.append({"info": "No foreign processes in this sample."})
    if row.get("processes_truncated"):
        result.append(
            {
                "info": f"Showing {len(row['processes'])} of "
                f"{row.get('processes_total', '?')} processes."
            }
        )
    return result


def gpu_tree(gpus, expanded, query=""):
    rows = []
    for gpu in gpus:
        if query and query not in str(gpu).casefold():
            continue
        identity = gpu_key(gpu)
        rows.append(gpu)
        if identity in expanded:
            for index, process in enumerate(process_entries(gpu)):
                suffix = process.get("pid", f"info-{index}")
                rows.append(
                    {
                        **gpu,
                        "_row_key": f"{identity}/process/{suffix}",
                        "_gpu_parent": identity,
                        "_process": process,
                    }
                )
    return rows


def gpu_cells(row):
    process = row.get("_process")
    if process is not None:
        if "info" in process:
            return ("  ↳", "", process["info"], "", "", "", "")
        return (
            f"  PID {process['pid']}",
            "",
            process_name(process),
            process_memory(process),
            "—",
            process.get("username") or "Not reported",
            process_kind(process),
        )
    total, free = row.get("total_memory_bytes", 0), row.get("free_memory_bytes", 0)
    return (
        row["site_id"],
        row.get("index"),
        row.get("name"),
        f"{(total - free) / 1024**3:.1f} / {total / 1024**3:.1f}",
        f"{row.get('utilization_percent', 0):g}%",
        _gpu_owner(row),
        "ok" if row.get("healthy") else "stale",
    )


def process_details(page, row):
    page.section("Foreign processes · read-only")
    state = process_status(row)
    page.field(
        "Telemetry",
        {"current": "Live heartbeat sample", "stale": "Stale"}.get(state, "Unavailable"),
    )
    if row.get("processes_received_at"):
        page.field("Received", row["processes_received_at"])
    for process in process_entries(row):
        if "info" in process:
            page.item(process["info"])
        else:
            page.field(f"PID {process['pid']}", process_name(process))
            page.field(
                "",
                f"User: {process.get('username') or 'Not reported'} · "
                f"VRAM: {process_memory(process)} · {process_kind(process)}",
            )
    page.item("GPU utilization is device-wide, not a per-process measurement. No process controls.")
