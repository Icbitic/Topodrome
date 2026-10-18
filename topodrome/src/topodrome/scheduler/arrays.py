"""Durable array bookkeeping, sharing the ordinary job/attempt/lease protocol."""

import asyncio
import hashlib
import json
from collections import Counter

from sqlalchemy import case, func, or_, select

from topodrome.db.models import AttemptRow, JobArrayRow, JobRow, LeaseGPUClaimRow, LeaseRow
from topodrome.db.repositories import _scoped_job_idempotency_key
from topodrome.domain.enums import JobState, LeaseState
from topodrome.domain.errors import AdmissionRejectedError, ConflictError, NotFoundError
from topodrome.domain.specs import JobSpec, new_id, utc_now

TERMINAL = (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED)


async def get_array(session, array_id, *, owner_username=None, lock=False):
    query = select(JobArrayRow).where(JobArrayRow.id == array_id)
    if owner_username is not None:
        query = query.where(JobArrayRow.owner_username == owner_username)
    if lock:
        query = query.with_for_update()
    row = await session.scalar(query)
    if row is None:
        raise NotFoundError(f"array {array_id} does not exist")
    return row


def active_array_jobs(array_id):
    active_claim = (
        select(LeaseGPUClaimRow.lease_id)
        .where(LeaseGPUClaimRow.lease_id == LeaseRow.id, LeaseGPUClaimRow.active.is_(True))
        .exists()
    )
    return (
        select(func.count(func.distinct(JobRow.id)))
        .join(AttemptRow, AttemptRow.job_id == JobRow.id)
        .join(LeaseRow, LeaseRow.attempt_id == AttemptRow.id)
        .where(
            JobRow.array_id == array_id,
            LeaseRow.state.in_(
                [LeaseState.OFFERED, LeaseState.ACCEPTED, LeaseState.ACTIVE, LeaseState.RELEASING]
            ),
            or_(LeaseRow.state != LeaseState.RELEASING, active_claim),
        )
    )


async def array_gate(session, job):
    """Lock parent before count; held until the same transaction creates the lease."""
    if job.array_id is None:
        return None
    parent = await session.scalar(
        select(JobArrayRow)
        .where(JobArrayRow.id == job.array_id)
        .with_for_update(skip_locked=True)
        .execution_options(populate_existing=True)
    )
    if parent is None:
        # Scans may encounter several arrays in opposite orders. Never wait on
        # another dispatcher's parent lock while holding a different parent.
        return "array_admission_busy"
    if parent.cancel_requested:
        return "array_canceled"
    if parent.held:
        return "array_held"
    if (await session.scalar(active_array_jobs(parent.id))) >= parent.max_parallel:
        return "array_concurrency_limit"
    return None


async def submit_array(service, request, *, owner_username, is_admin, idempotency_key):
    serialized = request.model_dump(mode="json")
    fingerprint = hashlib.sha256(
        json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    stored_key = (
        _scoped_job_idempotency_key(owner_username, idempotency_key)
        if idempotency_key is not None
        else None
    )
    # Caller holds the cross-process storage/publication lock. Replay precedes
    # pin resolution so expired inputs cannot invalidate an accepted submission.
    if stored_key is not None:
        async with service.database.session() as session:
            existing = await session.scalar(
                select(JobArrayRow).where(JobArrayRow.idempotency_key == stored_key)
            )
            if existing is not None:
                if existing.request_fingerprint != fingerprint:
                    raise ConflictError("idempotency key was used for a different array")
                return await array_summary(session, existing)
    base = await service.prepare_submission(
        request.job, owner_username=owner_username, is_admin=is_admin
    )
    prepared = request.model_copy(update={"job": base})
    # Source/assets are resolved once, but every model override must independently
    # resolve against the registered catalogue before any child is committed.
    specs = []
    for element in request.elements:
        spec = prepared.element_spec(element)
        if (
            "model" in element.model_fields_set
            and spec.model is not None
            and request.job.resources.min_vram_gb == 0
            and (
                element.resources is None or "min_vram_gb" not in element.resources.model_fields_set
            )
        ):
            # An automatically selected base footprint is not a user-requested
            # override for a different model. Resolve that child's own default.
            spec = spec.model_copy(
                update={"resources": spec.resources.model_copy(update={"min_vram_gb": 0})}
            )
        specs.append((element.index, service._resolve_model(spec)))
    if sum(len(spec.model_dump_json().encode()) for _, spec in specs) > 32 * 1024**2:
        raise AdmissionRejectedError("expanded array metadata exceeds 32 MiB; use smaller arrays")
    now = utc_now()
    async with service.database.transaction() as session:
        parent = JobArrayRow(
            id=new_id("array"),
            owner_username=owner_username,
            name=base.name,
            max_parallel=request.max_parallel,
            held=False,
            cancel_requested=False,
            idempotency_key=stored_key,
            request_fingerprint=fingerprint,
            created_at=now,
        )
        session.add(parent)
        await session.flush()
        # All children and their immutable pins commit together or not at all.
        for index, spec in specs:
            job = await service.repository.submit_job(session, spec, owner_username=owner_username)
            job.array_id, job.array_index = parent.id, index
        await session.flush()
        return await array_summary(session, parent)


async def next_array_model(session, job):
    """One advisory, exact checkpoint; never reserves or reorders a child."""
    if job.array_id is None:
        return None
    current = JobSpec.model_validate(job.spec)
    if current.model is None or current.execution_mode not in {"single", "model_sharded"}:
        return None
    from topodrome.db.job_ordering import queue_expressions

    now = utc_now()
    rank, estimate, submitted = queue_expressions(now)
    rows = await session.scalars(
        select(JobRow)
        .where(
            JobRow.array_id == job.array_id,
            JobRow.owner_username == job.owner_username,
            JobRow.id != job.id,
            JobRow.state.in_((JobState.QUEUED, JobState.RETRY_WAIT)),
            or_(JobRow.retry_not_before.is_(None), JobRow.retry_not_before <= now),
        )
        .order_by(rank.desc(), estimate.asc().nulls_last(), submitted, JobRow.id)
        .limit(32)
    )
    for row in rows:
        candidate = JobSpec.model_validate(row.spec)
        if (
            candidate.model is not None
            and candidate.model.revision is not None
            and candidate.model != current.model
            and candidate.runtime_fingerprint == current.runtime_fingerprint
        ):
            return candidate.model
    return None


async def array_summary(session, row, *, counts=None, active=None):
    if counts is None:
        counts = dict(
            (
                await session.execute(
                    select(JobRow.state, func.count())
                    .where(JobRow.array_id == row.id)
                    .group_by(JobRow.state)
                )
            ).all()
        )
    counts = Counter(counts)
    total = sum(counts.values())
    terminal = sum(counts[state] for state in TERMINAL)
    if active is None:
        active = await session.scalar(active_array_jobs(row.id))
    if terminal == total:
        state = (
            "failed"
            if counts[JobState.FAILED]
            else ("canceled" if counts[JobState.CANCELED] else "succeeded")
        )
    elif row.cancel_requested:
        state = "canceling"
    elif active:
        state = "running"
    elif counts[JobState.RUNNING]:
        state = "finalizing"
    elif row.held:
        state = "held"
    else:
        state = "queued"
    return {
        "array_id": row.id,
        "name": row.name,
        "owner_username": row.owner_username,
        "state": state,
        "counts": {s.value: n for s, n in counts.items()},
        "total": total,
        "terminal": terminal,
        "active_compute": active,
        "max_parallel": row.max_parallel,
        "held": row.held,
        "cancel_requested": row.cancel_requested,
        "created_at": row.created_at,
    }


async def array_detail(service, array_id, *, owner_username=None, after=-1, limit=100):
    from .status import job_status_rows

    async with service.database.session() as session:
        row = await get_array(session, array_id, owner_username=owner_username)
        result = await array_summary(session, row)
        members = list(
            await session.scalars(
                select(JobRow)
                .where(JobRow.array_id == row.id, JobRow.array_index > after)
                .order_by(JobRow.array_index)
                .limit(limit + 1)
            )
        )
        result["members"] = [
            {**row, "index": row["array_index"]}
            for row in await job_status_rows(session, members[:limit])
        ]
        result["next_after"] = members[limit - 1].array_index if len(members) > limit else None

    def add_progress():
        for member in result["members"]:
            if member["attempt_id"] and member["state"] in {
                "running",
                "canceling",
                "finalizing",
                "assigning",
            }:
                member["progress"] = service.live_logs.read_progress(
                    member["job_id"], member["attempt_id"]
                )

    await asyncio.to_thread(add_progress)
    return result


async def update_array(service, array_id, update, *, owner_username=None):
    async with service._dispatch_reload_lock, service.database.transaction() as session:
        row = await get_array(session, array_id, owner_username=owner_username, lock=True)
        if row.cancel_requested:
            raise ConflictError("a canceled array cannot be resumed or changed")
        if update.max_parallel is not None:
            row.max_parallel = update.max_parallel
        if update.held is not None:
            row.held = update.held
        # Changing the ceiling/hold never cancels already admitted work.
        return await array_summary(session, row)


async def cancel_array(service, array_id, *, owner_username=None):
    async with service._dispatch_reload_lock, service.database.transaction() as session:
        row = await get_array(session, array_id, owner_username=owner_username, lock=True)
        row.cancel_requested = True
        result = await array_summary(session, row)
    await reconcile_cancellations(service)
    return result


async def reconcile_cancellations(service):
    # Durable parent intent survives a disconnected caller or scheduler crash.
    async with service.database.session() as session:
        ids = list(
            await session.scalars(
                select(JobRow.id)
                .join(JobArrayRow, JobArrayRow.id == JobRow.array_id)
                .where(
                    JobArrayRow.cancel_requested.is_(True),
                    JobRow.state.not_in(TERMINAL),
                    JobRow.cancel_requested.is_(False),
                )
                .order_by(JobRow.id)
                .limit(100)
            )
        )
    for job_id in ids:
        await service.cancel(job_id)


async def list_arrays(session, *, owner_username=None, limit=50):
    query = select(JobArrayRow)
    if owner_username is not None:
        query = query.where(JobArrayRow.owner_username == owner_username)
    unfinished = (
        select(JobRow.id)
        .where(JobRow.array_id == JobArrayRow.id, JobRow.state.not_in(TERMINAL))
        .exists()
    )
    rows = list(
        await session.scalars(
            query.order_by(case((unfinished, 0), else_=1), JobArrayRow.created_at.desc()).limit(
                limit
            )
        )
    )
    ids = [row.id for row in rows]
    if not ids:
        return []
    counts = {array_id: {} for array_id in ids}
    for array_id, state, count in await session.execute(
        select(JobRow.array_id, JobRow.state, func.count())
        .where(JobRow.array_id.in_(ids))
        .group_by(JobRow.array_id, JobRow.state)
    ):
        counts[array_id][state] = count
    active = dict(
        (
            await session.execute(
                active_array_jobs(JobRow.array_id)
                .with_only_columns(JobRow.array_id, func.count(func.distinct(JobRow.id)))
                .where(JobRow.array_id.in_(ids))
                .group_by(JobRow.array_id)
            )
        ).all()
    )
    return [
        await array_summary(session, row, counts=counts[row.id], active=active.get(row.id, 0))
        for row in rows
    ]
