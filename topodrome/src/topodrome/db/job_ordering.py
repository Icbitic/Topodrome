"""One queue order for dispatch, pagination and status.

After a bounded eligible wait, FIFO takes precedence over priority and runtime.
This prevents a stream of newer, shorter or higher-priority work from passing an
old job forever; it neither preempts work nor promises unavailable resources.
"""

from datetime import timedelta

from sqlalchemy import and_, case, func, or_

from topodrome.db.models import JobRow


def runtime_estimate_expression():
    # JSON numeric extraction works on PostgreSQL and SQLite. Missing legacy
    # fields and explicit JSON null both become SQL NULL, ordered last.
    return JobRow.spec["estimated_runtime_seconds"].as_float()


DEFAULT_STARVATION_SECONDS = 1800


def queue_expressions(now, starvation_seconds=DEFAULT_STARVATION_SECONDS):
    eligible = func.coalesce(JobRow.eligible_since, JobRow.retry_not_before, JobRow.created_at)
    aged = eligible <= now - timedelta(seconds=starvation_seconds)
    return (
        case((aged, 1001), else_=JobRow.priority),
        case((aged, None), else_=runtime_estimate_expression()),
        case((aged, eligible), else_=JobRow.created_at),
    )


def queue_cursor(job, now, starvation_seconds=DEFAULT_STARVATION_SECONDS):
    from topodrome.scheduler.timing import aware

    eligible = job.eligible_since or job.retry_not_before or job.created_at
    aged = aware(eligible) <= aware(now) - timedelta(seconds=starvation_seconds)
    return (
        1001 if aged else job.priority,
        None if aged else job.spec.get("estimated_runtime_seconds"),
        eligible if aged else job.created_at,
        job.id,
    )


def after_cursor(expressions, cursor):
    rank, estimate, submitted = expressions
    priority, runtime, created_at, job_id = cursor
    later_fifo = or_(submitted > created_at, and_(submitted == created_at, JobRow.id > job_id))
    later_estimate = (
        and_(estimate.is_(None), later_fifo)
        if runtime is None
        else or_(estimate.is_(None), estimate > runtime, and_(estimate == runtime, later_fifo))
    )
    return or_(rank < priority, and_(rank == priority, later_estimate))
