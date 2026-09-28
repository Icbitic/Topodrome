"""Stable, inline array groups for the jobs table (no nested navigation)."""

TERMINAL = {"succeeded", "failed", "canceled"}


def array_row(summary, expanded, cache, errors):
    identity = summary["array_id"]
    counts = ", ".join(f"{k} {v}" for k, v in summary["counts"].items())
    if identity in expanded:
        if identity in errors:
            counts += " · refresh unavailable; r retries"
        elif identity not in cache:
            counts += " · loading elements…"
    return {
        "job_id": identity,
        "is_array": True,
        "name": f"{summary.get('name') or 'Array'} ({summary['total']})",
        "state": summary["state"],
        "array_summary": summary,
        "progress": {
            "stage": "array",
            "current": summary["terminal"],
            "total": summary["total"],
            "message": counts,
        },
    }


def job_tree(snapshot, expanded, cache, errors, *, history=False, sort="queue", query=""):
    groups = {row["array_id"]: row for row in snapshot.get("arrays", [])}
    known = {row["job_id"]: row for detail in cache.values() for row in detail.get("members", [])}
    known.update((row["job_id"], row) for row in snapshot.get("jobs", []))
    children = {}
    for row in known.values():
        children.setdefault(row.get("array_id"), []).append(row)
    roots = []
    seen = set()
    for row in snapshot.get("jobs", []):
        parent = row.get("array_id")
        identity = parent if parent in groups else row["job_id"]
        if identity not in seen:
            roots.append(groups[parent] if parent in groups else row)
            seen.add(identity)
    roots.extend(row for identity, row in groups.items() if identity not in seen)
    if sort in {"name", "site"}:
        key = "name" if sort == "name" else "site_id"
        roots.sort(key=lambda row: str(row.get(key) or "").casefold())

    def matches(row):
        return not query or query in str(row).casefold()

    rows = []
    for root in roots:
        if not history and root.get("state") in TERMINAL:
            continue
        if "job_id" in root:
            if matches(root):
                rows.append(root)
            continue
        identity = root["array_id"]
        members = sorted(children.get(identity, []), key=lambda row: row["array_index"])
        members = [r for r in members if history or r.get("state") not in TERMINAL]
        match = matches(root)
        members = [r for r in members if match or matches(r)]
        if not match and not members:
            continue
        rows.append(array_row(root, expanded, cache, errors))
        if identity in expanded:
            rows.extend(members)
    return rows, known
