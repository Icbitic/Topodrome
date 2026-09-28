"""Advisory forced-reservation preview using existing owner-scoped read APIs."""

import asyncio

from rich.text import Text

from .workbench_details import clean

LIVE_JOBS = {"assigning", "starting", "running", "canceling", "cancelling"}
LIVE_ATTEMPTS = {
    "offered",
    "accepted",
    "starting",
    "running",
    "preempt_requested",
    "checkpointing",
    "committing",
}


async def reservation_impact(client, target):
    snapshot = await client.system_status()
    site, uuid = target["site_id"], target["uuid"]
    gpu = next(
        (g for g in snapshot.get("gpus", []) if g.get("site_id") == site and g.get("uuid") == uuid),
        None,
    )
    candidates = [
        j
        for j in snapshot.get("jobs", [])
        if j.get("site_id") == site and j.get("state") in LIVE_JOBS and j.get("gpu_count", 1) != 0
    ]
    semaphore = asyncio.Semaphore(4)

    async def inspect(job):
        async with semaphore:
            try:
                attempts = await client.attempts(job["job_id"])
                attempt_id = job.get("attempt_id")
                if attempt_id:
                    attempt = next((a for a in attempts if a.get("attempt_id") == attempt_id), None)
                else:
                    attempt = max(
                        attempts, key=lambda a: a.get("attempt_number") or 0, default=None
                    )
                if not attempt or attempt.get("state") not in LIVE_ATTEMPTS:
                    return job, None, True
                if attempt.get("site_id") != site:
                    return job, None, True
                uuids = attempt.get("gpu_uuids")
                if not uuids:
                    return job, None, True
                return job, attempt if uuid in uuids else None, False
            except Exception:
                # Preserve the visibility boundary; neither response bodies nor
                # errors should leak credentials or inaccessible job metadata.
                return job, None, True

    matches = await asyncio.gather(*(inspect(job) for job in candidates))
    affected = [(job, attempt) for job, attempt, _ in matches if attempt]
    uncertain = any(unknown for _, _, unknown in matches)
    text = Text()
    text.append("\nJobs currently using this GPU\n", style="bold")
    for job, attempt in affected:
        text.append(clean(job.get("name") or "Unnamed job") + "\n", style="yellow")
        text.append(f"  {clean(job['job_id'])}\n", style="dim")
        text.append(f"  {clean(job['state'])} · {clean(job.get('model'))}\n", style="")
        checkpoint = attempt.get("checkpoint_ref") or attempt.get("restore_checkpoint_ref")
        text.append(
            "  Checkpoint: " + ("reported" if checkpoint else "not reported") + "\n", style=""
        )
        if len(attempt.get("gpu_uuids", [])) > 1:
            text.append(
                "  Multi-GPU job: the whole job is preempted, not just this GPU.\n", style="yellow"
            )
    if not affected:
        if gpu and gpu.get("active_lease"):
            text.append("Active GPU lease, but its job could not be identified.\n", style="yellow")
        else:
            text.append(
                "No matching jobs in the visible snapshot. This is not an idle guarantee.\n",
                style="",
            )
    if gpu is None or uncertain:
        text.append(
            "Some GPU/job details are unavailable or changed during inspection.\n", style="yellow"
        )
    site_state = next(
        (s.get("state") for s in snapshot.get("sites", []) if s.get("site_id") == site), None
    )
    if site_state in {"offline", "stopped"} or (gpu and not gpu.get("healthy", False)):
        text.append("Site/GPU status is stale; the job list may be incomplete.\n", style="yellow")
    if gpu and gpu.get("foreign_process_count", 0):
        text.append("Foreign processes are not stopped by this reservation.\n", style="dim")
    if any(k.get("site_id") == site for k in snapshot.get("kernels", [])):
        text.append(
            "This site also has interactive kernels; their exact GPU mapping is unavailable.\n",
            style="yellow",
        )
    text.append(
        "This bounded, permission-scoped snapshot is advisory. "
        "Assignments may change before the reservation is applied.\n",
        style="dim",
    )
    return text
