"""Human-readable, read-only system telemetry rows."""

from .workbench_details import date, duration


def size(value):
    if value is None:
        return "—"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024 or unit == "PiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def system_rows(report, error=None):
    rows = []

    def add(key, name, value, note):
        rows.append({"metric_id": key, "name": name, "value": value, "note": note})

    if not report:
        add(
            "status",
            "System diagnostics",
            "Unavailable" if error else "Loading…",
            "Requires operator access and an upgraded scheduler."
            if error
            else "Storage is sampled in the background; no cleanup is triggered.",
        )
        return rows
    storage = report.get("storage") or {}
    upgrades = report.get("upgrades") or []
    if upgrades:
        upgrade = next((r for r in upgrades if r.get("state") not in
                        {"succeeded", "rolled_back", "failed", "canceled"}), upgrades[-1])
        add("upgrade", "Upgrade", upgrade.get("state", "unknown"),
            str(upgrade.get("name", "")) + " · " + str(upgrade.get("upgrade_id", "")))
        scope = "scheduler / cluster" if upgrade.get("level") in {"scheduler", "cluster"} else \
                ", ".join(upgrade.get("sites") or [])
        add("upgrade-scope", "Upgrade scope", scope,
            "Dispatch pause: " + str(upgrade.get("pause", "unknown")) +
            "; running work is never preempted.")
        blockers = {k: v for k, v in (upgrade.get("blockers") or {}).items() if v}
        add("upgrade-blockers", "Upgrade blockers",
            ", ".join(f"{key.replace('_', ' ')}: {value}" for key, value in blockers.items())
            if blockers else "None reported", "Fresh service inactivity and durable state checks.")
    policy = report.get("retention") or {}
    enabled = policy.get("enabled", False)
    ready = storage.get("estimate_state") == "ready"
    state = "stale · API unavailable" if error else report.get("storage_state", "unavailable")
    add(
        "scope",
        "Storage scope",
        "Active central store",
        "Excludes remote caches, external models and archives outside the active root. "
        "Allocated regular-file bytes; hardlinks counted once; symlinks and submounts excluded.",
    )
    add(
        "sample",
        "Storage sample",
        state,
        "Sampled "
        + date(storage.get("sampled_at"))
        + f" · refreshed at most every {report.get('storage_refresh_seconds', 300)}s; "
        "previous sample retained on error.",
    )
    add(
        "used",
        "Topodrome used",
        size(storage.get("allocated_bytes")),
        "Physical allocation of unique regular files; "
        "excludes directory metadata and external databases.",
    )
    add(
        "free",
        "Filesystem free",
        size(storage.get("filesystem_free_bytes")),
        "Shared host filesystem; total "
        + size(storage.get("filesystem_total_bytes"))
        + ". Not Topodrome-only.",
    )
    for category, value in sorted(storage.get("categories", {}).items()):
        add(
            "category:" + category,
            "  " + category,
            size(value),
            "Part of Topodrome used; shared hardlinks attributed once.",
        )
    add(
        "eligible-day",
        "Reclaimable within 1d",
        size(storage.get("eligible_24h_bytes")) if ready else "—",
        "Estimate including already eligible data, with current references unchanged. "
        "Assets, live jobs, recovery and accepted input pins remain protected. "
        "Not a guaranteed free-space increase: "
        "cleanup budgets, new references and workload can delay deletion."
        + (" Automatic cleanup is OFF." if not enabled else ""),
    )
    add(
        "eligible-now",
        "  Eligible now",
        size(storage.get("eligible_now_bytes")) if ready else "—",
        "Reference-aware estimate using cleanup eligibility rules; no deletion from this view.",
    )
    add(
        "eligible-new",
        "  Newly eligible in 1d",
        size(storage.get("newly_eligible_24h_bytes")) if ready else "—",
        "Additional data aging out within 24 hours, excluding already eligible data. "
        "Unknown is never shown as zero.",
    )
    add(
        "retention",
        "Automatic cleanup",
        "Enabled" if enabled else "Disabled",
        "Advisory; actual cleanup rechecks references under publication/submission locks.",
    )
    add(
        "gc-interval",
        "Cleanup interval",
        duration(policy.get("interval_seconds")),
        f"Per pass: up to {policy.get('max_files_per_pass', '—')} files / "
        + size(policy.get("max_bytes_per_pass")),
    )
    add(
        "artifact-retention",
        "Artifact retention",
        duration(policy.get("artifact_seconds")),
        "Default after terminal completion; overrides and references may retain data longer. "
        "Published assets persist until deleted.",
    )
    add(
        "checkpoint-retention",
        "Failed-job checkpoints",
        duration(policy.get("failed_checkpoint_seconds")),
        "After failed/canceled jobs; active and restoring checkpoints stay protected.",
    )
    add(
        "uptime",
        "Scheduler uptime",
        duration(report.get("uptime_seconds")),
        "Since this scheduler service started.",
    )
    add(
        "database",
        "Database",
        "Ready" if report.get("database_ready") else "Not ready",
        "Connectivity and expected schema: " + str(report.get("schema_revision", "—")),
    )
    add(
        "memory",
        "Scheduler memory",
        size(report.get("scheduler_rss_bytes")),
        "Current scheduler process RSS, not total host memory.",
    )
    load = report.get("host_load_average")
    add(
        "load",
        "Host load · 1 / 5 / 15m",
        " / ".join(f"{v:.2f}" for v in load) if load else "—",
        "Host-wide load averages, not GPU utilization or Topodrome-only activity.",
    )
    add(
        "agents",
        "Connected site agents",
        str(report.get("connected_agents", "—")),
        "Currently connected control channels; see Sites for heartbeat/readiness.",
    )
    scheduling = report.get("policy") or {}
    add(
        "policy",
        "Scheduling policy",
        str(scheduling.get("status", "—")),
        f"Generation {scheduling.get('generation', '—')}; "
        f"pending drain blockers: {scheduling.get('blocker_count', 0)}. Site preference: "
        + (" → ".join(scheduling.get("preferred_site_ids", [])) or "Automatic"),
    )
    return rows
