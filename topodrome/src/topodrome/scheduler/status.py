"""Bounded, read-only scheduler snapshots for operators and dashboards."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, case, func, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from topodrome.db.job_ordering import DEFAULT_STARVATION_SECONDS, queue_expressions
from topodrome.db.models import (
    AttemptRow,
    GPUReservationRow,
    GPURow,
    InputStageRow,
    JobRow,
    KernelSessionRow,
    LeaseGPUClaimRow,
    LeaseRow,
    ModelCapabilityRow,
    ProviderInstanceRow,
    SiteRow,
)
from topodrome.domain.enums import (
    AttemptState,
    GPUReservationState,
    JobState,
    KernelState,
    LeaseState,
    ModelAvailability,
    ProviderInstanceState,
    SiteKind,
    SiteState,
)
from topodrome.domain.specs import JobSpec, utc_now
from topodrome.scheduler.timing import job_timing

_OWNING_LEASE_STATES = (
    LeaseState.OFFERED,
    LeaseState.ACCEPTED,
    LeaseState.ACTIVE,
    LeaseState.RELEASING,
)


async def system_status_snapshot(
    session: AsyncSession,
    *,
    owner_username: str | None = None,
    job_limit: int = 50,
    kernel_limit: int = 50,
    site_limit: int = 100,
    gpu_limit: int = 1024,
    heartbeat_ttl_seconds: float = 30,
    starvation_seconds: float = DEFAULT_STARVATION_SECONDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build one sanitized status frame from a single database session.

    The frame deliberately contains no job environments, provider metadata,
    URLs, or credentials. When ``owner_username`` is set, job and kernel rows
    are filtered before limits are applied; shared site/GPU health remains a
    cluster view. It is bounded so a watching terminal cannot turn into an
    unbounded control-plane query.
    """

    if min(job_limit, kernel_limit, site_limit, gpu_limit) < 1:
        raise ValueError("status limits must be positive")
    if heartbeat_ttl_seconds <= 0:
        raise ValueError("heartbeat TTL must be positive")
    generated_at = _aware(now or utc_now())
    heartbeat_cutoff = generated_at - timedelta(seconds=heartbeat_ttl_seconds)
    sites = list(
        await session.scalars(
            select(SiteRow)
            .where(or_(SiteRow.kind != SiteKind.AUTODL, SiteRow.state != SiteState.STOPPED))
            .order_by(SiteRow.id)
            .limit(site_limit)
        )
    )
    site_ids = [site.id for site in sites]
    gpus = (
        list(
            await session.scalars(
                select(GPURow)
                .where(GPURow.site_id.in_(site_ids))
                .order_by(GPURow.site_id, GPURow.index)
                .limit(gpu_limit)
            )
        )
        if site_ids
        else []
    )
    displayed_gpu_keys = [(gpu.site_id, gpu.uuid) for gpu in gpus]
    jobs_statement = select(JobRow)
    if owner_username is not None:
        jobs_statement = jobs_statement.where(JobRow.owner_username == owner_username)
    job_band = case(
        (JobRow.state.in_((JobState.RUNNING, JobState.ASSIGNING, JobState.CANCELING)), 0),
        (JobRow.state.in_((JobState.QUEUED, JobState.RETRY_WAIT)), 1),
        else_=2,
    )
    rank, estimate, submitted = queue_expressions(generated_at, starvation_seconds)
    jobs = list(
        await session.scalars(
            jobs_statement.order_by(
                job_band,
                case((job_band == 1, rank), (job_band == 0, JobRow.priority), else_=0).desc(),
                case((job_band == 1, estimate)).asc().nulls_last(),
                case((job_band == 1, submitted), (job_band == 0, JobRow.created_at)).asc(),
                JobRow.created_at.desc(),
                JobRow.id,
            ).limit(job_limit)
        )
    )
    # Filter before LIMIT: terminal history must not hide older live sessions.
    kernels_statement = select(KernelSessionRow).where(
        KernelSessionRow.state.not_in((KernelState.STOPPED, KernelState.LOST))
    )
    if owner_username is not None:
        kernels_statement = kernels_statement.where(
            KernelSessionRow.owner_username == owner_username
        )
    kernels = list(
        await session.scalars(
            kernels_statement.order_by(KernelSessionRow.created_at.desc()).limit(kernel_limit)
        )
    )
    providers = list(
        await session.scalars(
            select(ProviderInstanceRow)
            .where(ProviderInstanceRow.provider == "autodl")
            .where(
                ProviderInstanceRow.state.not_in(
                    [ProviderInstanceState.RELEASED, ProviderInstanceState.STOPPED]
                )
            )
            .order_by(ProviderInstanceRow.created_at.desc())
            .limit(100)
        )
    )
    active_claims = (
        set(
            await session.execute(
                select(LeaseGPUClaimRow.site_id, LeaseGPUClaimRow.gpu_uuid)
                .join(LeaseRow, LeaseGPUClaimRow.lease_id == LeaseRow.id)
                .where(
                    tuple_(
                        LeaseGPUClaimRow.site_id,
                        LeaseGPUClaimRow.gpu_uuid,
                    ).in_(displayed_gpu_keys),
                    LeaseGPUClaimRow.active.is_(True),
                    LeaseRow.state.in_(_OWNING_LEASE_STATES),
                )
            )
        )
        if displayed_gpu_keys
        else set()
    )
    operator_reserved = set(
        (
            await session.execute(
                select(GPUReservationRow.site_id, GPUReservationRow.gpu_uuid)
                .where(
                    GPUReservationRow.state.in_(
                        [GPUReservationState.PENDING, GPUReservationState.READY]
                    ),
                    GPUReservationRow.expires_at > generated_at,
                )
                .distinct()
            )
        ).all()
    )
    warm_gpu_claims = (
        set(
            await session.execute(
                select(ModelCapabilityRow.site_id, ModelCapabilityRow.gpu_uuid)
                .distinct()
                .where(
                    tuple_(
                        ModelCapabilityRow.site_id,
                        ModelCapabilityRow.gpu_uuid,
                    ).in_(displayed_gpu_keys),
                    ModelCapabilityRow.availability == ModelAvailability.WARM,
                    ModelCapabilityRow.gpu_uuid != "",
                )
            )
        )
        if displayed_gpu_keys
        else set()
    )
    gpus_by_site: dict[str, list[GPURow]] = {}
    for gpu in gpus:
        gpus_by_site.setdefault(gpu.site_id, []).append(gpu)

    site_rows: list[dict[str, Any]] = []
    gpu_rows: list[dict[str, Any]] = []
    for site in sites:
        effective_state = site.state
        if effective_state is SiteState.READY and (
            site.heartbeat_at is None or _aware(site.heartbeat_at) < heartbeat_cutoff
        ):
            # READY is an agent-reported state, not a permanent health claim.
            # Derive the operator view with the same freshness boundary used
            # by placement so a stale site cannot look schedulable in `top`.
            effective_state = SiteState.OFFLINE
        site_rows.append(
            {
                "site_id": site.id,
                "display_name": site.display_name,
                "kind": site.kind.value,
                "state": effective_state.value,
                "heartbeat_at": site.heartbeat_at,
                "storage_available": site.storage_available,
                "queue_depth": site.queue_depth,
                "region": site.region,
                "paid": site.paid,
            }
        )
        warm_gpu_uuids = {gpu_uuid for site_id, gpu_uuid in warm_gpu_claims if site_id == site.id}
        for gpu in gpus_by_site.get(site.id, []):
            claim_key = (site.id, gpu.uuid)
            gpu_rows.append(
                {
                    "site_id": site.id,
                    "uuid": gpu.uuid,
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_memory_bytes": gpu.total_memory_bytes,
                    "free_memory_bytes": gpu.free_memory_bytes,
                    "utilization_percent": gpu.utilization_percent,
                    "temperature_c": gpu.temperature_c,
                    "foreign_process_count": gpu.foreign_process_count,
                    "foreign_memory_bytes": gpu.foreign_memory_bytes,
                    "externally_reserved": gpu.externally_reserved,
                    "operator_reserved": claim_key in operator_reserved,
                    "healthy": gpu.healthy,
                    "observed_at": gpu.observed_at,
                    "active_lease": claim_key in active_claims,
                    "warm_replica": gpu.uuid in warm_gpu_uuids,
                }
            )

    job_rows = await job_status_rows(session, jobs, now=generated_at)

    kernel_rows = [
        {
            "session_id": kernel.id,
            "state": kernel.state.value,
            "mode": kernel.mode.value,
            "site_id": kernel.site_id,
            "created_at": kernel.created_at,
            "last_activity_at": kernel.last_activity_at,
            "expires_at": kernel.expires_at,
        }
        for kernel in kernels
    ]
    provider_rows = [
        {
            "instance_id": row.id,
            "state": row.state.value,
            "region": row.region,
            "gpu_spec": row.gpu_spec,
            "gpu_count": _requested_gpu_count(row),
            "observed_gpu_count": _positive_int(row.metadata_json.get("observed_gpu_count")),
            "shape_mismatch": row.metadata_json.get("shape_mismatch") is True,
            "idle_seconds": _idle_seconds(row.idle_since, generated_at),
            "updated_at": row.updated_at,
        }
        for row in providers
    ]
    from .arrays import list_arrays

    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "arrays": await list_arrays(session, owner_username=owner_username),
        "sites": site_rows,
        "gpus": gpu_rows,
        "jobs": job_rows,
        "kernels": kernel_rows,
        "autodl_instances": provider_rows,
    }


async def job_status_rows(session, jobs, *, now=None):
    """One bounded, sanitized projection for overview and array drill-down."""
    now = _aware(now or utc_now())
    job_ids = [job.id for job in jobs]
    if not job_ids:
        return []
    stages = {
        row.job_id: row
        for row in await session.scalars(
            select(InputStageRow).where(
                InputStageRow.job_id.in_(job_ids),
                InputStageRow.expires_at > now,
                InputStageRow.state.in_(["downloading", "ready", "failed"]),
            )
        )
    }
    latest_numbers = (
        select(AttemptRow.job_id.label("job_id"), func.max(AttemptRow.number).label("number"))
        .where(AttemptRow.job_id.in_(job_ids))
        .group_by(AttemptRow.job_id)
        .subquery()
    )
    attempts = await session.scalars(
        select(AttemptRow)
        .join(
            latest_numbers,
            and_(
                AttemptRow.job_id == latest_numbers.c.job_id,
                AttemptRow.number == latest_numbers.c.number,
            ),
        )
        .options(selectinload(AttemptRow.lease).selectinload(LeaseRow.claims))
    )
    latest = {attempt.job_id: attempt for attempt in attempts}
    result = []
    for job in jobs:
        spec = JobSpec.model_validate(job.spec)
        attempt = latest.get(job.id)
        stage = stages.get(job.id) if job.state in {JobState.QUEUED, JobState.RETRY_WAIT} else None
        lease = attempt.lease if attempt else None
        released = lease is not None and lease.state is LeaseState.RELEASING
        finalizing = (
            job.state is JobState.RUNNING
            and attempt is not None
            and (attempt.state is AttemptState.COMMITTING and released)
        )
        actual_gpu_count = sum(claim.active for claim in lease.claims) if lease else 0
        result.append(
            {
                "job_id": job.id,
                "array_id": job.array_id,
                "array_index": job.array_index,
                "blocked_reason": job.blocked_reason,
                "timing": job_timing(job, now=now),
                "attempt_id": attempt.id if attempt else None,
                "name": spec.name,
                "state": (
                    "downloading"
                    if stage is not None and stage.state != "ready"
                    else "finalizing"
                    if finalizing
                    else job.state.value
                ),
                "site_id": stage.site_id if stage else attempt.site_id if attempt else None,
                **({"progress": stage.progress} if stage is not None else {}),
                "gpu_count": 0
                if stage is not None or released
                else actual_gpu_count or spec.resources.gpu_count,
                "model": spec.model.alias,
                "workload_kind": spec.workload_kind.value,
                "estimated_runtime_seconds": spec.estimated_runtime_seconds,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
            }
        )
    return result


def _requested_gpu_count(row: ProviderInstanceRow) -> int:
    return _positive_int(row.metadata_json.get("requested_gpu_count")) or 1


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _idle_seconds(value: datetime | None, now: datetime) -> float | None:
    if value is None:
        return None
    return max(0.0, (now - _aware(value)).total_seconds())


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["system_status_snapshot"]
