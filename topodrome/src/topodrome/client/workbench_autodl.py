"""Curated manual registrations, including instances awaiting their first agent."""


def autodl_rows(sites, instances):
    rows = {
        row["site_id"]: dict(row)
        for row in sites
        if row.get("kind") == "autodl" and row.get("state") not in {"stopped", "released"}
    }
    for instance in instances:
        identity = instance.get("site_id")
        if not identity:
            continue
        if instance.get("state") in {"stopped", "released"}:
            rows.pop(identity, None)
            continue
        site = rows.get(identity, {})
        rows[identity] = {
            **site,
            "site_id": identity,
            "kind": "autodl",
            "state": instance.get("state", "starting"),
            "site_state": site.get("state", "awaiting heartbeat"),
            "region": instance.get("region") or site.get("region"),
            "gpu_count": instance.get("gpu_count"),
            "registered_at": instance.get("updated_at"),
        }
    return list(rows.values())
