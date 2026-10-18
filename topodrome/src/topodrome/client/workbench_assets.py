"""Latest asset rows with independently expandable, immutable version history."""

from datetime import datetime


def asset_group(row):
    return row["visibility"], row["name"]


def _published(row):
    try:
        return datetime.fromisoformat(
            str(row.get("published_at")).replace("Z", "+00:00")
        ).timestamp()
    except (ValueError, TypeError):
        return 0


def asset_tree(latest, history, expanded, *, query="", state="loading", limited=False):
    groups = {}
    for row in history:
        groups.setdefault(asset_group(row), {})[row["ref"]] = row
    rows = []
    for parent in latest:
        group = asset_group(parent)
        members = sorted(
            (row for ref, row in groups.get(group, {}).items() if ref != parent["ref"]),
            key=lambda row: (_published(row), row["ref"]),
            reverse=True,
        )
        matches = not query or query in str(parent).casefold()
        members = [row for row in members if matches or query in str(row).casefold()]
        if not matches and not (group in expanded and members):
            continue
        rows.append(parent)
        if group not in expanded:
            continue
        rows.extend({**row, "_asset_parent": parent["ref"]} for row in members)
        message = (
            "History unavailable; r retries (cached versions may be stale)"
            if state == "error"
            else "Loading older versions…"
            if state == "loading"
            else "History limited to the latest 1,000 versions across assets"
            if limited
            else "No older versions"
            if not members and matches
            else None
        )
        if message:
            rows.append(
                {
                    **parent,
                    "ref": parent["ref"] + "/history-status",
                    "_asset_parent": parent["ref"],
                    "_asset_info": message,
                }
            )
    return rows


def asset_cells(row):
    if row.get("_asset_info"):
        return ("  ↳ " + row["_asset_info"], "", "", "", "", "")
    return (
        ("  ↳ " if row.get("_asset_parent") else "") + row["name"],
        row["version"],
        row["visibility"],
        f"{row['total_size_bytes'] / 1024**3:.3f}",
        len(row["files"]),
        ", ".join(row.get("tags", [])),
    )
