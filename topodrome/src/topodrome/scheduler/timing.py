"""Non-overlapping scheduler-observed durations, not worker-clock predictions."""

from datetime import UTC

from topodrome.domain.specs import utc_now

TERMINAL = {"succeeded", "failed", "canceled"}


def aware(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def set_phase(job, phase, *, now=None):
    now = now or utc_now()
    if job.phase == phase and job.phase_started_at is not None:
        return
    totals = dict(job.phase_seconds or {})
    start = job.phase_started_at or job.created_at
    if start is not None and job.phase not in TERMINAL:
        previous = job.phase or "untracked"
        totals[previous] = totals.get(previous, 0.0) + max(
            0, (aware(now) - aware(start)).total_seconds()
        )
    job.phase_seconds = totals
    job.phase = phase
    job.phase_started_at = now


def job_timing(job, *, now=None):
    current = aware(now or utc_now())
    complete = job.state.value in TERMINAL
    end = aware(job.updated_at) if complete else current
    totals = dict(job.phase_seconds or {})
    start = aware(job.phase_started_at or job.created_at)
    elapsed = max(0, (end - start).total_seconds())
    if job.phase not in TERMINAL:
        totals[job.phase or "untracked"] = totals.get(job.phase or "untracked", 0) + elapsed
    total = max(0, (end - aware(job.created_at)).total_seconds())
    # Older rows have no historical phase data. Never label it as GPU compute.
    unknown = max(0, total - sum(totals.values()))
    if unknown > 0.001:
        totals["untracked"] = totals.get("untracked", 0) + unknown
    return {
        "phase": job.phase if not complete else job.state.value,
        "phase_elapsed_seconds": 0 if complete else elapsed,
        "seconds": {key: round(value, 3) for key, value in totals.items()},
        "total_seconds": round(total, 3),
        "observed_at": current.isoformat(),
        "complete": complete,
    }


def attempt_timing(attempt, *, now=None):
    end = aware(attempt.completed_at or now or utc_now())
    created = aware(attempt.created_at)
    started = aware(attempt.started_at) if attempt.started_at else None
    finished = aware(attempt.compute_finished_at) if attempt.compute_finished_at else None
    return {
        "startup_seconds": max(0, ((started or end) - created).total_seconds()),
        "compute_seconds": max(0, ((finished or end) - started).total_seconds()) if started else 0,
        "finalization_seconds": max(0, (end - finished).total_seconds()) if finished else 0,
        "total_seconds": max(0, (end - created).total_seconds()),
        "complete": attempt.completed_at is not None,
    }
