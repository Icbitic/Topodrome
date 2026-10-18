"""Curated, terminal-safe detail pages; no API object dumps."""

import unicodedata
from datetime import datetime

from rich.text import Text

from .dashboard import _gpu_owner, _progress_text
from .workbench_gpus import process_details


def clean(value):
    if value is None:
        return "Not reported"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (dict, list, tuple)):
        return "See entries below"
    return "".join(c for c in str(value)[:4000] if not unicodedata.category(c).startswith("C"))


def date(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime(
            "%d %b %Y %H:%M %Z"
        )
    except ValueError:
        return clean(value)


class Page:
    def __init__(self):
        self.text = Text()

    def section(self, title):
        if self.text.plain:
            self.text.append("\n")
        self.text.append(clean(title) + "\n", style="bold")

    def field(self, label, value, style=""):
        self.text.append(f"{label:<17} ", style="dim")
        self.text.append(clean(value) + "\n", style=style)

    def item(self, value):
        self.text.append("  • " + clean(value) + "\n")


def duration(value):
    if not isinstance(value, (int, float)) or value < 0:
        return "—"
    seconds = int(value)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def timing_cell(row):
    parts = (row.get("timing") or {}).get("seconds", {})
    if not parts:
        return "—"
    wait = sum(parts.get(key, 0) for key in ("queued", "held", "array_wait", "retry_wait"))
    io = sum(parts.get(key, 0) for key in ("inputs", "finalizing"))
    return f"{duration(wait)} / {duration(parts.get('compute', 0))} / {duration(io)}"


def timing_page(page, timing):
    if not timing:
        return
    page.section("Timing · scheduler observed, all attempts")
    page.field("Total elapsed", duration(timing.get("total_seconds")))
    labels = {
        "queued": "Queue",
        "held": "Held",
        "array_wait": "Array allowance",
        "inputs": "Input staging",
        "assigning": "Assignment",
        "starting": "Model / startup",
        "compute": "Computation",
        "checkpointing": "Checkpointing",
        "finalizing": "Finalization",
        "retry_wait": "Retry backoff",
        "canceling": "Cancellation",
        "untracked": "Untracked history",
    }
    for key, label in labels.items():
        value = timing.get("seconds", {}).get(key)
        if value is not None:
            page.field(label, duration(value))
    if not timing.get("complete"):
        page.field("Current phase", timing.get("phase"))


def detail_page(view, row, job=None):
    page = Page()
    if view == "system":
        page.section(row["name"])
        page.field("Value", row["value"])
        page.section("About this metric")
        page.item(row["note"])
    elif view == "jobs" and row.get("is_array"):
        summary = row["array_summary"]
        page.section(summary.get("name") or "Job array")
        page.field("State", summary.get("state"))
        page.field("Owner", summary.get("owner_username"))
        page.field("Submitted", date(summary.get("created_at")))
        page.section("Members")
        page.field("Finished", f"{summary['terminal']} / {summary['total']}")
        for state in (
            "queued",
            "assigning",
            "running",
            "retry_wait",
            "canceling",
            "succeeded",
            "failed",
            "canceled",
        ):
            count = summary.get("counts", {}).get(state)
            if count is not None:
                page.field(state.replace("_", " ").capitalize(), count)
        page.item("Finished includes succeeded, failed and canceled jobs.")
        page.section("Scheduling")
        page.field("Active compute", summary.get("active_compute"))
        page.field("Parallel limit", summary.get("max_parallel"))
        page.field("Held", summary.get("held", False))
        page.field("Cancel requested", summary.get("cancel_requested", False))
        page.item("Member jobs have their own attempts, logs and timing breakdowns.")
        page.section("Identity")
        page.field("Array ID", summary["array_id"])
        page.item("Esc returns to Jobs. Space or the disclosure arrow shows members inline.")
    elif view == "jobs":
        spec = (job or {}).get("spec", {})
        page.section(spec.get("name") or row.get("name") or "Job overview")
        page.field("State", row.get("state"))
        page.field("Site", row.get("site_id") or "Not assigned")
        model = spec.get("model") or row.get("model")
        page.field(
            "Model", model.get("repo_id") if isinstance(model, dict) else model or "None (GPU job)"
        )
        if isinstance(model, dict):
            page.field("Revision", model.get("revision"))
            if model.get("subfolder") is not None:
                page.field("Subfolder", model["subfolder"])
        if model is None:
            page.field("Execution", spec.get("execution") or row.get("execution") or "single")
        page.field("GPUs", row.get("gpu_count") or spec.get("resources", {}).get("gpu"))
        page.field("Progress", _progress_text(row.get("progress"), width=160) or "Not reported")
        if spec.get("description"):
            page.section("Description")
            page.item(spec["description"])
        page.section("Scheduling")
        page.field("Priority", spec.get("priority", 0))
        estimate = spec.get("estimated_runtime_seconds")
        page.field(
            "Runtime estimate", f"{estimate:g} s" if isinstance(estimate, (int, float)) else None
        )
        page.field("Attempts", (job or {}).get("attempts"))
        page.field(
            "Waiting reason",
            (job or {}).get("blocked_reason") or row.get("blocked_reason") or "None",
        )
        if (job or row).get("array_id"):
            page.field("Array", (job or row).get("array_id"))
            page.field("Array index", (job or row).get("array_index"))
        timing_page(page, (job or {}).get("timing") or row.get("timing"))
        outputs = spec.get("required_outputs", [])
        if outputs:
            page.section("Required outputs")
            for output in outputs[:100]:
                page.item(output)
        page.section("Identity")
        page.field("Job ID", row.get("job_id"))
    elif view == "gpus":
        page.section(f"GPU {row.get('index', '?')} · {row.get('site_id', '')}")
        page.field("Model", row.get("name"))
        page.field("Health", "Healthy" if row.get("healthy") else "Stale / unhealthy")
        page.field("Ownership", _gpu_owner(row).replace("+", " + "))
        page.section("Current resources")
        total, free = row.get("total_memory_bytes"), row.get("free_memory_bytes")
        if isinstance(total, (int, float)) and isinstance(free, (int, float)):
            page.field("Memory used", f"{(total - free) / 1024**3:.1f} / {total / 1024**3:.1f} GiB")
            page.field("Memory free", f"{free / 1024**3:.1f} GiB")
        util = row.get("utilization_percent")
        page.field("Utilization", f"{util:g}%" if isinstance(util, (int, float)) else None)
        page.field("Foreign processes", row.get("foreign_process_count", 0))
        page.section("Topodrome allocation")
        page.field("Active lease", bool(row.get("active_lease")))
        page.field("Warm actor", bool(row.get("warm_replica") or row.get("warm_actor")))
        page.field("Reserved", bool(row.get("operator_reserved") or row.get("externally_reserved")))
        process_details(page, row)
        page.section("Identity")
        page.field("GPU UUID", row.get("uuid"))
    elif view in {"sites", "autodl"}:
        page.section(row.get("site_id", "Site"))
        page.field("State", row.get("state"))
        page.field("Type", row.get("kind"))
        page.field("Region", row.get("region") or "Not specified")
        if view == "sites":
            page.section("Placement preference")
            rank = row.get("preference_rank")
            page.field("Order", f"#{rank}" if rank else "Automatic fallback")
            page.item("Preference applies to eligible new work only; running jobs stay untouched.")
        if view == "autodl":
            page.field("Agent", row.get("site_state") or row.get("state"))
            if row.get("state") == "starting":
                page.item(
                    "Registration accepted; installation and authenticated heartbeat are pending."
                )
        page.section("Health & capacity")
        page.field("Last heartbeat", date(row.get("heartbeat_at")))
        page.field("Queued work", row.get("queue_depth"))
        page.field("Storage ready", row.get("storage_available"))
        page.field("GPUs", row.get("gpu_count"))
    elif view == "reservations":
        page.section("GPU reservation")
        page.field("Site", row.get("site_id"))
        page.field("GPU index", row.get("gpu_index"))
        page.field("State", row.get("state"))
        page.field("Expires", date(row.get("expires_at")))
        page.field("Reason", row.get("reason"))
        page.field("Forced eviction", row.get("force", False))
        page.section("Identity")
        page.field("Reservation ID", row.get("reservation_id"))
        page.field("GPU UUID", row.get("gpu_uuid"))
    elif view == "assets":
        page.section(f"{row.get('name', 'Asset')} · {row.get('version', '')}")
        page.field("Visibility", row.get("visibility"))
        page.field("Owner", row.get("owner_username"))
        page.field("Tags", ", ".join(row.get("tags", [])) or "None")
        size = row.get("total_size_bytes", 0)
        page.field("Size", f"{size / 1024**3:.3f} GiB")
        page.field("Published", date(row.get("published_at")))
        page.section("Files")
        for file in row.get("files", [])[:100]:
            if isinstance(file, dict):
                page.item(
                    f"{clean(file.get('relative_path'))}  ({file.get('size_bytes', 0):,} bytes)"
                )
            else:
                page.item(file)
        if not row.get("files"):
            page.item("No file entries in this snapshot")
        elif len(row["files"]) > 100:
            page.item("Showing the first 100 files")
        page.section("Reference")
        page.item(row.get("ref"))
    return page.text


def attempts_page(attempts):
    page = Page()
    if not attempts:
        page.section("No attempts yet")
        page.item("Waiting for the scheduler to assign this job.")
    for index, attempt in enumerate(attempts, 1):
        page.section(
            f"Attempt {attempt.get('attempt_number') or index} · {clean(attempt.get('state'))}"
        )
        page.field("Site", attempt.get("site_id"))
        page.field("Started", date(attempt.get("started_at")))
        if attempt.get("completed_at"):
            page.field("Finished", date(attempt["completed_at"]))
        if attempt.get("reason_code"):
            page.field("Outcome reason", attempt["reason_code"])
        page.field("Recovery snapshot", "Available" if attempt.get("checkpoint_ref") else "None")
        timing = attempt.get("timing", {})
        if timing:
            page.field("Startup", duration(timing.get("startup_seconds")))
            page.field("Compute", duration(timing.get("compute_seconds")))
            page.field("Finalize", duration(timing.get("finalization_seconds")))
        page.field("Attempt ID", attempt.get("attempt_id"))
        for gpu in attempt.get("gpu_uuids") or []:
            page.field("GPU UUID", gpu)
    return page.text


def models_page(models):
    page = Page()
    for model in models:
        page.section(model.get("repo_id", "Model"))
        for label, key in (
            ("Site", "site_id"),
            ("Availability", "availability"),
            ("Revision", "revision"),
            ("Subfolder", "subfolder"),
            ("Backend", "backend"),
            ("Precision", "dtype"),
        ):
            page.field(label, model.get(key))
    if not models:
        page.section("No models reported")
    return page.text
