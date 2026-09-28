"""Transactional input placement, cancellation and progress without GPU leases."""

from datetime import UTC, timedelta

from sqlalchemy import func, select

from topodrome.db.models import AttemptRow, InputStageRow
from topodrome.domain.events import SchedulerEnvelope
from topodrome.domain.specs import InputStageMetrics, JobSpec, new_id, utc_now


def aware(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


async def prepare_inputs(session, job, site_id, *, now=None, capacity=2):
    """Return readiness + a command to send only after transaction commit."""
    now = now or utc_now()
    row = await session.get(InputStageRow, job.id, with_for_update=True)
    if row is not None and row.site_id == site_id and aware(row.expires_at) > now:
        if row.state == "ready":
            from .timing import set_phase

            set_phase(job, "queued", now=now)
            return True, None
        if row.state == "failed" and (now - aware(row.updated_at)).total_seconds() < 30:
            return False, None
        if row.sent_at and (now - aware(row.sent_at)).total_seconds() < 15:
            return False, None
        if row.state in {"failed", "canceled", "consumed"}:
            row.expires_at = now
            return await prepare_inputs(session, job, site_id, now=now, capacity=capacity)
    else:
        count = await session.scalar(
            select(func.count())
            .select_from(InputStageRow)
            .where(
                InputStageRow.site_id == site_id,
                InputStageRow.state.in_(["downloading", "ready"]),
                InputStageRow.expires_at > now,
                InputStageRow.job_id != job.id,
            )
        )
        if count >= capacity:
            return False, None
        if row is None:
            row = InputStageRow(job_id=job.id)
            session.add(row)
        row.stage_id = new_id("stage")
        row.site_id = site_id
        row.state = "downloading"
        row.progress = {}
        row.placement_data = {
            k: v for k, v in (row.placement_data or {}).items() if k == "last_switch_at"
        }
    row.sent_at = now
    from .timing import set_phase

    set_phase(job, "inputs", now=now)
    row.expires_at = now + timedelta(seconds=120)
    # Immutable pinned inputs only, not experiment kwargs, source contents or credentials.
    spec = JobSpec.model_validate(job.spec)
    checkpoint_ref = await session.scalar(
        select(AttemptRow.checkpoint_ref)
        .where(AttemptRow.job_id == job.id, AttemptRow.checkpoint_ref.is_not(None))
        .order_by(AttemptRow.number.desc())
        .limit(1)
    )
    command = SchedulerEnvelope(
        site_id=site_id,
        kind="stage_inputs",
        payload={
            "job_id": job.id,
            "stage_id": row.stage_id,
            "assets": {k: a.model_dump(mode="json") for k, a in spec.assets.items()},
            "source_ref": spec.source_ref,
            "checkpoint_ref": checkpoint_ref,
            "expires_at": row.expires_at.isoformat(),
            "ttl_seconds": 120,
        },
    )
    await session.flush()  # The next candidate must see this staging-capacity claim.
    return False, command


async def retarget_inputs(session, job, row, site_id, *, now=None):
    """Atomically replace the transfer generation, preserving job/attempt state."""
    now = now or utc_now()
    old = SchedulerEnvelope(
        site_id=row.site_id,
        kind="cancel_input_stage",
        payload={
            "job_id": job.id,
            "stage_id": row.stage_id,
        },
    )
    _, command = await prepare_inputs(session, job, site_id, now=now)
    if command is None:
        return None  # Target staging capacity is full; keep the existing intent.
    row.placement_data = {**row.placement_data, "last_switch_at": now.timestamp()}
    return old, command


async def receive_stage(session, site_id, payload):
    row = await session.get(InputStageRow, str(payload.get("job_id", "")), with_for_update=True)
    if row is None or row.site_id != site_id or row.stage_id != payload.get("stage_id"):
        return
    if row.state in {"consumed", "canceled"} or aware(row.expires_at) <= utc_now():
        return
    state = payload.get("state")
    if state not in {"downloading", "ready", "failed", "canceled"}:
        raise ValueError("invalid input staging state")
    if row.state in {"ready", "failed"} and state == "downloading":
        return
    changed = row.state != state
    row.state = state
    # Only bounded, data-only progress is accepted; no remote paths/errors/credentials.
    from topodrome.domain.progress import JobProgress

    row.progress = JobProgress.model_validate(payload["progress"]).model_dump(mode="json")
    if payload.get("metrics") is not None:
        metrics = InputStageMetrics.model_validate(payload["metrics"])
        row.placement_data = {
            **(row.placement_data or {}),
            "metrics": metrics.model_dump(),
            "metrics_at": utc_now().timestamp(),
        }
        if (
            changed
            and state == "ready"
            and metrics.downloaded_bytes >= 256 * 1024
            and metrics.download_seconds >= 1
        ):
            from .estimates import observe

            await observe(
                session,
                site_id,
                "download_bps",
                "",
                metrics.downloaded_bytes / metrics.download_seconds,
            )
    if changed:
        row.updated_at = utc_now()


async def cancel_stage(session, job_id):
    row = await session.get(InputStageRow, job_id, with_for_update=True)
    if row is None or row.state in {"consumed", "canceled"}:
        return None
    row.state = "canceled"
    row.updated_at = utc_now()
    return SchedulerEnvelope(
        site_id=row.site_id,
        kind="cancel_input_stage",
        payload={
            "job_id": row.job_id,
            "stage_id": row.stage_id,
        },
    )
