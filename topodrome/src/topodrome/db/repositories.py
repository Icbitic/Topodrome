from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from topodrome.db.job_ordering import (
    DEFAULT_STARVATION_SECONDS,
    after_cursor,
    queue_expressions,
)
from topodrome.db.models import (
    AgentEventAckRow,
    ArtifactRow,
    AttemptRow,
    GPUFenceCounterRow,
    GPUReservationRow,
    GPURow,
    JobRow,
    JobStateEventRow,
    KernelChannelRequestRow,
    KernelSessionRow,
    LeaseGPUClaimRow,
    LeaseRow,
    ModelActorRow,
    ModelCapabilityRow,
    ProviderInstanceRow,
    SiteRow,
)
from topodrome.domain.enums import (
    ActorState,
    ArtifactState,
    AttemptState,
    GPUReservationState,
    JobState,
    KernelMode,
    KernelState,
    LeaseState,
    ModelAvailability,
    SiteKind,
    SiteState,
)
from topodrome.domain.errors import ConflictError, DurabilityError, NotFoundError, StaleFenceError
from topodrome.domain.events import AgentEnvelope
from topodrome.domain.specs import (
    ArtifactDescriptor,
    AssignmentOffer,
    FenceClaim,
    GPUReservationExtensionRequest,
    GPUReservationRequest,
    GPUReservationSummary,
    JobSpec,
    JobSummary,
    LeaseRenewalGrant,
    ModelCapability,
    SiteHeartbeat,
    new_id,
    utc_now,
)
from topodrome.domain.transitions import (
    ATTEMPT_TRANSITIONS,
    JOB_TRANSITIONS,
    KERNEL_TRANSITIONS,
    LEASE_TRANSITIONS,
    require_transition,
)
from topodrome.protocol.fencing import FencingAuthority
from topodrome.scheduler.placement import GPUCandidate, SiteCandidate


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _owner_username(value: str) -> str:
    """Validate the durable principal label used for resource ownership."""

    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 64:
        raise ValueError("owner username is invalid")
    return value


def _scoped_job_idempotency_key(owner_username: str, key: str) -> str:
    """Fit per-owner idempotency into the existing globally unique column.

    Keeping the current uniqueness constraint avoids a locking table rewrite on
    a live scheduler database. Length-prefixing both values makes the hash input
    unambiguous without persisting another user's raw idempotency key.
    """

    if not isinstance(key, str) or not 1 <= len(key) <= 200:
        raise ValueError("idempotency key must contain between 1 and 200 characters")
    owner = _owner_username(owner_username).encode("utf-8")
    raw_key = key.encode("utf-8")
    payload = len(owner).to_bytes(4, "big") + owner + len(raw_key).to_bytes(4, "big") + raw_key
    return f"owner-v1:{hashlib.sha256(payload).hexdigest()}"


class SchedulerRepository:
    async def _find_idempotent_job(
        self,
        session: AsyncSession,
        *,
        stored_idempotency_key: str,
    ) -> JobRow | None:
        return await session.scalar(
            select(JobRow)
            .where(JobRow.idempotency_key == stored_idempotency_key)
            .options(selectinload(JobRow.attempts))
        )

    async def submit_job(
        self,
        session: AsyncSession,
        spec: JobSpec,
        *,
        idempotency_key: str | None = None,
        owner_username: str = "@operator",
    ) -> JobRow:
        owner = _owner_username(owner_username)
        stored_idempotency_key = (
            None if idempotency_key is None else _scoped_job_idempotency_key(owner, idempotency_key)
        )
        serialized_spec = spec.model_dump(mode="json")
        if stored_idempotency_key is not None:
            existing = await self._find_idempotent_job(
                session,
                stored_idempotency_key=stored_idempotency_key,
            )
            if existing:
                if {
                    "name": None,
                    "description": None,
                    "estimated_runtime_seconds": None,
                    "artifact_retention_seconds": None,
                    **existing.spec,
                } != serialized_spec:
                    raise ConflictError("idempotency key was already used for a different job")
                return existing
        now = utc_now()
        job = JobRow(
            id=new_id("job"),
            owner_username=owner,
            state=JobState.QUEUED,
            spec=serialized_spec,
            idempotency_key=stored_idempotency_key,
            priority=spec.priority,
            attempts=[],
            created_at=now,
            updated_at=now,
            eligible_since=now,
            phase="queued",
            phase_started_at=now,
            phase_seconds={},
        )
        try:
            # A savepoint contains a concurrent unique-key loss so PostgreSQL
            # does not leave the caller's transaction aborted. The competing
            # insert must commit before PostgreSQL reports the violation, after
            # which READ COMMITTED can retrieve and validate the winner.
            async with session.begin_nested():
                session.add(job)
                session.add(
                    JobStateEventRow(
                        job_id=job.id,
                        from_state=None,
                        to_state=JobState.QUEUED.value,
                        created_at=now,
                    )
                )
                await session.flush()
        except IntegrityError:
            if stored_idempotency_key is None:
                raise
            existing = await self._find_idempotent_job(
                session,
                stored_idempotency_key=stored_idempotency_key,
            )
            if existing is None:
                raise
            if {
                "name": None,
                "description": None,
                "estimated_runtime_seconds": None,
                "artifact_retention_seconds": None,
                **existing.spec,
            } != serialized_spec:
                raise ConflictError(
                    "idempotency key was already used for a different job"
                ) from None
            return existing
        return job

    async def get_job(
        self,
        session: AsyncSession,
        job_id: str,
        *,
        lock: bool = False,
        owner_username: str | None = None,
    ) -> JobRow:
        statement = (
            select(JobRow)
            .where(JobRow.id == job_id)
            .options(
                selectinload(JobRow.attempts)
                .selectinload(AttemptRow.lease)
                .selectinload(LeaseRow.claims)
            )
        )
        if owner_username is not None:
            statement = statement.where(JobRow.owner_username == _owner_username(owner_username))
        if lock:
            statement = statement.with_for_update()
        job = await session.scalar(statement)
        if job is None:
            raise NotFoundError(f"job {job_id} does not exist")
        return job

    async def committed_artifacts_for_job(
        self,
        session: AsyncSession,
        job_id: str,
        *,
        owner_username: str | None = None,
    ) -> tuple[ArtifactRow, ...]:
        """Return the immutable committed output set for one visible job.

        The owner-scoped job lookup deliberately happens before the artifact
        query so an inaccessible producer is indistinguishable from a missing
        producer.  A completed job is the publication boundary: callers must
        never construct reusable inputs from an in-progress attempt's staging
        or partially committed files.
        """

        job = await self.get_job(
            session,
            job_id,
            owner_username=owner_username,
        )
        if job.state is not JobState.SUCCEEDED:
            raise ConflictError("job outputs are unavailable until the job succeeds")
        rows = await session.scalars(
            select(ArtifactRow)
            .where(
                ArtifactRow.job_id == job.id,
                ArtifactRow.state == ArtifactState.COMMITTED,
            )
            .order_by(ArtifactRow.logical_path, ArtifactRow.attempt_id)
        )
        return tuple(rows)

    async def claim_next_job(
        self,
        session: AsyncSession,
        *,
        now: datetime | None = None,
    ) -> JobRow | None:
        candidates = await self.lock_job_candidates(session, limit=1, now=now)
        if not candidates:
            return None
        return await self.claim_locked_job(session, candidates[0])

    async def lock_job_candidates(
        self,
        session: AsyncSession,
        *,
        limit: int,
        now: datetime | None = None,
        after: tuple[int, float | None, datetime, str] | None = None,
        starvation_seconds: float = DEFAULT_STARVATION_SECONDS,
    ) -> list[JobRow]:
        """Lock aged jobs first, then priority, shortest estimate and FIFO.

        Callers may inspect the locked rows and claim the first one that can be
        placed.  PostgreSQL's ``SKIP LOCKED`` keeps concurrent scheduler
        generations from selecting the same job while allowing either
        generation to make progress on other queued work.
        """

        if limit < 1:
            raise ValueError("job candidate limit must be positive")
        current = now or utc_now()
        statement = select(JobRow).where(
            JobRow.cancel_requested.is_(False),
            or_(
                JobRow.state == JobState.QUEUED,
                (
                    (JobRow.state == JobState.RETRY_WAIT)
                    & (JobRow.retry_not_before.is_not(None))
                    & (JobRow.retry_not_before <= current)
                ),
            ),
        )
        rank, estimate, submitted = queue_expressions(current, starvation_seconds)
        if after is not None:
            statement = statement.where(after_cursor((rank, estimate, submitted), after))
        statement = (
            statement.order_by(rank.desc(), estimate.asc().nulls_last(), submitted, JobRow.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return list(await session.scalars(statement))

    async def claim_locked_job(self, session: AsyncSession, job: JobRow) -> JobRow:
        """Transition one row returned by :meth:`lock_job_candidates`."""

        from topodrome.scheduler.arrays import array_gate

        if await array_gate(session, job):
            raise ConflictError("array is not currently eligible for another allocation")

        if job.state is JobState.RETRY_WAIT:
            await self._transition_job(session, job, JobState.QUEUED)
        elif job.state is not JobState.QUEUED:
            raise ConflictError("job is no longer eligible for dispatch")
        job.retry_not_before = None
        await self._transition_job(session, job, JobState.ASSIGNING)
        return job

    async def release_claimed_job(self, session: AsyncSession, job: JobRow) -> None:
        if job.state is JobState.ASSIGNING:
            await self._transition_job(session, job, JobState.QUEUED)

    async def list_sites(self, session: AsyncSession) -> list[SiteRow]:
        return list(
            await session.scalars(
                select(SiteRow)
                .options(selectinload(SiteRow.gpus), selectinload(SiteRow.models))
                .where(or_(SiteRow.kind != SiteKind.AUTODL, SiteRow.state != SiteState.STOPPED))
                .order_by(SiteRow.id)
            )
        )

    async def create_gpu_reservation(
        self,
        session: AsyncSession,
        request: GPUReservationRequest,
        *,
        requested_by: str = "operator",
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> GPUReservationRow:
        """Persist one exact reservation after resolving its live GPU identity.

        Idempotency is checked before consulting mutable inventory, so replaying
        a committed request remains safe even if the site is temporarily
        offline or its index mapping later changes.
        """

        current = _aware(now or utc_now())
        requested_by = requested_by.strip()
        if not requested_by or len(requested_by) > 200:
            raise ValueError("requested_by must contain at most 200 characters")
        if idempotency_key is not None and not 1 <= len(idempotency_key) <= 200:
            raise ValueError("idempotency key must contain at most 200 characters")
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    **request.model_dump(
                        mode="json", exclude={"force"} if not request.force else set()
                    ),
                    "requested_by": requested_by,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if idempotency_key is not None:
            existing = await session.scalar(
                select(GPUReservationRow)
                .where(GPUReservationRow.idempotency_key == idempotency_key)
                .with_for_update()
            )
            if existing is not None:
                if existing.request_fingerprint != fingerprint:
                    raise ConflictError(
                        "idempotency key was already used for a different GPU reservation"
                    )
                return existing

        await self.expire_gpu_reservations(session, now=current)
        site = await session.get(SiteRow, request.site_id)
        if site is None:
            raise NotFoundError(f"site {request.site_id} does not exist")
        gpu_statement = select(GPURow).where(GPURow.site_id == request.site_id)
        if request.gpu_uuid is not None:
            gpu_statement = gpu_statement.where(GPURow.uuid == request.gpu_uuid)
        else:
            gpu_statement = gpu_statement.where(GPURow.index == request.gpu_index)
        gpu = await session.scalar(gpu_statement.with_for_update())
        if gpu is None:
            selector = (
                f"UUID {request.gpu_uuid}"
                if request.gpu_uuid is not None
                else f"index {request.gpu_index}"
            )
            raise NotFoundError(f"GPU {selector} does not exist at site {request.site_id}")

        # A concurrent creator with the same key may have committed while this
        # transaction waited for the exact GPU row. Re-read under that lock so
        # the loser returns the original record instead of a false target
        # conflict.
        if idempotency_key is not None:
            committed_replay: GPUReservationRow | None = await session.scalar(
                select(GPUReservationRow)
                .where(GPUReservationRow.idempotency_key == idempotency_key)
                .with_for_update()
            )
            if committed_replay is not None:
                if committed_replay.request_fingerprint != fingerprint:
                    raise ConflictError(
                        "idempotency key was already used for a different GPU reservation"
                    )
                return committed_replay

        active = await session.scalar(
            select(GPUReservationRow)
            .where(
                GPUReservationRow.site_id == request.site_id,
                GPUReservationRow.gpu_uuid == gpu.uuid,
                GPUReservationRow.state.in_(
                    [GPUReservationState.PENDING, GPUReservationState.READY]
                ),
                GPUReservationRow.expires_at > current,
            )
            .with_for_update()
        )
        if active is not None:
            raise ConflictError(
                f"GPU {gpu.uuid} at site {request.site_id} is already operator-reserved"
            )

        has_claim = (
            await session.scalar(
                select(LeaseGPUClaimRow.lease_id)
                .where(
                    LeaseGPUClaimRow.site_id == request.site_id,
                    LeaseGPUClaimRow.gpu_uuid == gpu.uuid,
                    LeaseGPUClaimRow.active.is_(True),
                )
                .limit(1)
            )
            is not None
        )
        has_actor = (
            await session.scalar(
                select(ModelActorRow.id)
                .where(
                    ModelActorRow.site_id == request.site_id,
                    ModelActorRow.gpu_uuid == gpu.uuid,
                    ModelActorRow.active_allocation.is_(True),
                )
                .limit(1)
            )
            is not None
        )
        row = GPUReservationRow(
            id=new_id("reservation"),
            site_id=request.site_id,
            gpu_uuid=gpu.uuid,
            gpu_index=gpu.index,
            requested_by=requested_by,
            reason=request.reason,
            force=request.force,
            duration_seconds=request.duration_seconds,
            request_fingerprint=fingerprint,
            idempotency_key=idempotency_key,
            state=(
                GPUReservationState.PENDING if has_claim or has_actor else GPUReservationState.READY
            ),
            created_at=current,
            expires_at=current + timedelta(seconds=request.duration_seconds),
            updated_at=current,
        )
        try:
            # Same-target creators serialize on the GPU row above. This
            # savepoint additionally handles the cross-target/same-key race:
            # losing the unique idempotency constraint must return the winner
            # (or a semantic conflict), not abort the outer API transaction.
            async with session.begin_nested():
                session.add(row)
                await session.flush()
        except IntegrityError:
            if idempotency_key is None:
                raise
            committed_replay = await session.scalar(
                select(GPUReservationRow).where(
                    GPUReservationRow.idempotency_key == idempotency_key
                )
            )
            if committed_replay is None:
                raise
            if committed_replay.request_fingerprint != fingerprint:
                raise ConflictError(
                    "idempotency key was already used for a different GPU reservation"
                ) from None
            return committed_replay
        return row

    async def get_gpu_reservation(
        self,
        session: AsyncSession,
        reservation_id: str,
        *,
        lock: bool = False,
    ) -> GPUReservationRow:
        statement = select(GPUReservationRow).where(GPUReservationRow.id == reservation_id)
        if lock:
            statement = statement.with_for_update()
        row = await session.scalar(statement)
        if row is None:
            raise NotFoundError(f"GPU reservation {reservation_id} does not exist")
        return row

    async def list_gpu_reservations(
        self,
        session: AsyncSession,
        *,
        limit: int = 100,
        active_only: bool = False,
        now: datetime | None = None,
    ) -> list[GPUReservationRow]:
        if limit < 1:
            raise ValueError("reservation limit must be positive")
        current = _aware(now or utc_now())
        statement = select(GPUReservationRow)
        if active_only:
            statement = statement.where(
                GPUReservationRow.state.in_(
                    [GPUReservationState.PENDING, GPUReservationState.READY]
                ),
                GPUReservationRow.expires_at > current,
            )
        return list(
            await session.scalars(
                statement.order_by(GPUReservationRow.created_at.desc()).limit(limit)
            )
        )

    async def cancel_gpu_reservation(
        self,
        session: AsyncSession,
        reservation_id: str,
        *,
        now: datetime | None = None,
    ) -> GPUReservationRow:
        current = _aware(now or utc_now())
        await self.expire_gpu_reservations(session, now=current)
        row = await self.get_gpu_reservation(session, reservation_id, lock=True)
        if row.state in {GPUReservationState.CANCELED, GPUReservationState.EXPIRED}:
            return row
        row.state = GPUReservationState.CANCELED
        row.canceled_at = current
        row.updated_at = current
        await session.flush()
        return row

    async def extend_gpu_reservation(
        self,
        session: AsyncSession,
        reservation_id: str,
        request: GPUReservationExtensionRequest,
        *,
        now: datetime | None = None,
    ) -> GPUReservationRow:
        """Extend under the reservation lock without clearing its admission fence."""

        row = await self.get_gpu_reservation(session, reservation_id, lock=True)
        # Check time after acquiring the lock, not before potentially waiting.
        current = _aware(now or utc_now())
        if (
            row.state not in {GPUReservationState.PENDING, GPUReservationState.READY}
            or _aware(row.expires_at) <= current
        ):
            raise ConflictError("only an active, unexpired GPU reservation can be extended")
        expected = _aware(request.expected_expires_at)
        try:
            target = expected + timedelta(seconds=request.additional_seconds)
        except OverflowError:
            raise ConflictError("reservation expiry is out of range") from None
        if _aware(row.expires_at) == target:
            return row  # An identical committed request is a harmless replay.
        if _aware(row.expires_at) != expected:
            raise ConflictError("reservation expiry changed; refresh before extending")
        if target > current + timedelta(days=365):
            raise ConflictError("reservation cannot extend more than one year into the future")
        row.expires_at = target
        row.duration_seconds += request.additional_seconds
        row.updated_at = current
        # Preserve identity, state, force, reason and eviction_requested_at.
        await session.flush()
        return row

    async def expire_gpu_reservations(
        self,
        session: AsyncSession,
        *,
        now: datetime | None = None,
    ) -> int:
        current = _aware(now or utc_now())
        rows = list(
            await session.scalars(
                select(GPUReservationRow)
                .where(
                    GPUReservationRow.state.in_(
                        [GPUReservationState.PENDING, GPUReservationState.READY]
                    ),
                    GPUReservationRow.expires_at <= current,
                )
                .with_for_update(skip_locked=True)
            )
        )
        for row in rows:
            row.state = GPUReservationState.EXPIRED
            row.updated_at = current
        if rows:
            await session.flush()
        return len(rows)

    async def active_gpu_reservation_keys(
        self,
        session: AsyncSession,
        *,
        now: datetime | None = None,
    ) -> set[tuple[str, str]]:
        current = _aware(now or utc_now())
        rows = await session.execute(
            select(GPUReservationRow.site_id, GPUReservationRow.gpu_uuid)
            .where(
                GPUReservationRow.state.in_(
                    [GPUReservationState.PENDING, GPUReservationState.READY]
                ),
                GPUReservationRow.expires_at > current,
            )
            .distinct()
        )
        return {(str(site_id), str(gpu_uuid)) for site_id, gpu_uuid in rows}

    async def lock_active_gpu_reservations(
        self,
        session: AsyncSession,
        *,
        now: datetime | None = None,
    ) -> list[GPUReservationRow]:
        """Lock every live reservation for one bounded scheduler reconciliation."""

        current = _aware(now or utc_now())
        await self.expire_gpu_reservations(session, now=current)
        return list(
            await session.scalars(
                select(GPUReservationRow)
                .where(
                    GPUReservationRow.state.in_(
                        [GPUReservationState.PENDING, GPUReservationState.READY]
                    ),
                    GPUReservationRow.expires_at > current,
                )
                .order_by(GPUReservationRow.created_at, GPUReservationRow.id)
                .with_for_update(skip_locked=True)
            )
        )

    async def list_attempts(
        self,
        session: AsyncSession,
        job_id: str,
        *,
        owner_username: str | None = None,
    ) -> list[AttemptRow]:
        await self.get_job(session, job_id, owner_username=owner_username)
        return list(
            await session.scalars(
                select(AttemptRow)
                .where(AttemptRow.job_id == job_id)
                .options(selectinload(AttemptRow.lease).selectinload(LeaseRow.claims))
                .order_by(AttemptRow.number)
            )
        )

    async def get_kernel(
        self,
        session: AsyncSession,
        session_id: str,
        *,
        lock: bool = False,
        owner_username: str | None = None,
    ) -> KernelSessionRow:
        statement = select(KernelSessionRow).where(KernelSessionRow.id == session_id)
        if owner_username is not None:
            statement = statement.where(
                KernelSessionRow.owner_username == _owner_username(owner_username)
            )
        if lock:
            statement = statement.with_for_update()
        kernel = await session.scalar(statement)
        if kernel is None:
            raise NotFoundError(f"kernel session {session_id} does not exist")
        return kernel

    async def rotate_kernel_access(
        self,
        session: AsyncSession,
        session_id: str,
        *,
        now: datetime,
        owner_username: str | None = None,
    ) -> KernelSessionRow:
        """Touch a live session and revoke every previously issued URL."""

        row = await self.touch_kernel(
            session,
            session_id,
            now=now,
            owner_username=owner_username,
        )
        row.access_generation += 1
        await session.flush()
        return row

    async def reserve_kernel_channel_request(
        self,
        session: AsyncSession,
        *,
        session_id: str,
        request_id: str,
        request_digest: bytes,
    ) -> tuple[KernelChannelRequestRow, bool]:
        """Create or recover one durable notebook request by client message ID."""

        existing = await session.scalar(
            select(KernelChannelRequestRow)
            .where(
                KernelChannelRequestRow.session_id == session_id,
                KernelChannelRequestRow.request_id == request_id,
            )
            .with_for_update()
        )
        if existing is not None:
            if existing.request_digest != request_digest:
                raise ConflictError("kernel request_id was reused with different operation content")
            return existing, False
        count = await session.scalar(
            select(func.max(KernelChannelRequestRow.execution_count)).where(
                KernelChannelRequestRow.session_id == session_id
            )
        )
        row = KernelChannelRequestRow(
            id=new_id("kernel_op"),
            session_id=session_id,
            request_id=request_id,
            request_digest=request_digest,
            state="reserved",
            execution_count=int(count or 0) + 1,
            agent_session_id=None,
            agent_connection_generation=None,
            result_digest=None,
            error_code=None,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        session.add(row)
        await session.flush()
        return row, True

    async def mark_kernel_channel_request_dispatched(
        self,
        session: AsyncSession,
        *,
        operation_id: str,
        agent_session_id: str,
        agent_connection_generation: str,
    ) -> KernelChannelRequestRow:
        row = await session.get(KernelChannelRequestRow, operation_id)
        if row is None:
            raise NotFoundError("kernel request reservation does not exist")
        if row.state in {"completed", "failed"}:
            return row
        if row.agent_session_id not in {None, agent_session_id}:
            raise StaleFenceError("kernel request belongs to a replaced site agent")
        if row.agent_connection_generation not in {
            None,
            agent_connection_generation,
        }:
            raise StaleFenceError("kernel request belongs to a replaced site connection")
        row.state = "dispatched"
        row.agent_session_id = agent_session_id
        row.agent_connection_generation = agent_connection_generation
        row.updated_at = utc_now()
        await session.flush()
        return row

    async def complete_kernel_channel_request(
        self,
        session: AsyncSession,
        *,
        operation_id: str,
        session_id: str,
        succeeded: bool,
        result_digest: bytes,
        error_code: str | None,
    ) -> KernelChannelRequestRow:
        row = await session.scalar(
            select(KernelChannelRequestRow)
            .where(KernelChannelRequestRow.id == operation_id)
            .with_for_update()
        )
        if row is None or row.session_id != session_id:
            raise StaleFenceError("kernel result has no current request reservation")
        target = "completed" if succeeded else "failed"
        if row.state not in {"reserved", "dispatched"}:
            if (
                row.state != target
                or row.result_digest != result_digest
                or row.error_code != error_code
            ):
                raise ConflictError("kernel operation replay changed its terminal result")
            return row
        row.state = target
        row.result_digest = result_digest
        row.error_code = error_code
        row.updated_at = utc_now()
        await session.flush()
        return row

    async def latest_pending_kernel_request(
        self,
        session: AsyncSession,
        session_id: str,
    ) -> KernelChannelRequestRow | None:
        row: KernelChannelRequestRow | None = await session.scalar(
            select(KernelChannelRequestRow)
            .where(
                KernelChannelRequestRow.session_id == session_id,
                KernelChannelRequestRow.state.in_(["reserved", "dispatched"]),
            )
            .order_by(KernelChannelRequestRow.created_at.desc())
            .limit(1)
        )
        return row

    async def find_shareable_kernel(
        self,
        session: AsyncSession,
        *,
        request: dict[str, object],
    ) -> KernelSessionRow | None:
        """Return a live service host with the exact requested model/resources."""

        candidates = list(
            await session.scalars(
                select(KernelSessionRow)
                .join(AttemptRow, KernelSessionRow.attempt_id == AttemptRow.id)
                .where(
                    KernelSessionRow.mode == KernelMode.SERVICE,
                    KernelSessionRow.state.in_([KernelState.RUNNING, KernelState.IDLE]),
                    KernelSessionRow.site_id.is_not(None),
                    KernelSessionRow.attempt_id.is_not(None),
                    KernelSessionRow.actor_id.is_not(None),
                    KernelSessionRow.runtime_id.is_not(None),
                    AttemptRow.state.in_(
                        [
                            AttemptState.ACCEPTED,
                            AttemptState.STARTING,
                            AttemptState.RUNNING,
                        ]
                    ),
                )
                .order_by(KernelSessionRow.last_activity_at.desc())
                .with_for_update(skip_locked=True)
            )
        )
        wanted_model = request.get("model")
        wanted_resources = request.get("resources")
        for candidate in candidates:
            if (
                candidate.request.get("model") == wanted_model
                and candidate.request.get("resources") == wanted_resources
            ):
                return candidate
        return None

    async def bind_kernel_attempt(
        self,
        session: AsyncSession,
        *,
        job_id: str,
        attempt_id: str,
        site_id: str,
    ) -> tuple[KernelSessionRow, ...]:
        rows = list(
            await session.scalars(
                select(KernelSessionRow)
                .where(
                    KernelSessionRow.job_id == job_id,
                    KernelSessionRow.attempt_id.is_(None),
                    KernelSessionRow.state == KernelState.REQUESTED,
                )
                .with_for_update()
            )
        )
        for row in rows:
            row.attempt_id = attempt_id
            row.site_id = site_id
            self._transition_kernel(row, KernelState.STARTING)
        return tuple(rows)

    async def update_kernel_lifecycle(
        self,
        session: AsyncSession,
        *,
        session_id: str,
        attempt_id: str,
        target: KernelState,
        site_id: str,
        relay_id: str | None = None,
        runtime_id: str | None = None,
        actor_id: str | None = None,
        now: datetime | None = None,
    ) -> KernelSessionRow:
        current = now or utc_now()
        row = await self.get_kernel(session, session_id, lock=True)
        if row.attempt_id != attempt_id or row.site_id != site_id:
            raise StaleFenceError("kernel session is not bound to this attempt and site")
        if relay_id is not None and row.relay_id not in {None, relay_id}:
            raise StaleFenceError("kernel relay identity is no longer current")
        if target is row.state:
            return row
        if row.state in {KernelState.STOPPED, KernelState.LOST}:
            raise StaleFenceError("kernel session is already terminal")
        self._transition_kernel(row, target)
        if target in {KernelState.RUNNING, KernelState.IDLE}:
            if not relay_id or not runtime_id or not actor_id:
                raise ConflictError(
                    "a running kernel requires relay, runtime, and actor identities"
                )
            row.relay_id = relay_id
            row.runtime_id = runtime_id
            row.actor_id = actor_id
            row.last_activity_at = current
        if target in {KernelState.STOPPED, KernelState.LOST}:
            row.runtime_id = None
            row.actor_id = None
            row.expires_at = None
        return row

    async def touch_kernel(
        self,
        session: AsyncSession,
        session_id: str,
        *,
        now: datetime,
        owner_username: str | None = None,
    ) -> KernelSessionRow:
        row = await self.get_kernel(
            session,
            session_id,
            lock=True,
            owner_username=owner_username,
        )
        if row.state not in {KernelState.RUNNING, KernelState.IDLE}:
            raise ConflictError("kernel is not ready")
        if row.expires_at is not None and _aware(row.expires_at) <= now:
            raise ConflictError("kernel session has expired")
        request = row.request
        idle_timeout = int(request["idle_timeout_seconds"])
        row.state = KernelState.RUNNING
        row.last_activity_at = now
        row.expires_at = now + timedelta(seconds=idle_timeout)
        return row

    async def refresh_connected_kernel(
        self,
        session: AsyncSession,
        session_id: str,
        *,
        now: datetime,
    ) -> KernelSessionRow:
        """Extend a live kernel deadline for a verified IDE connection.

        The managed-Jupyter controller is the only caller. It proves that the
        exact desired child process is alive and currently has a stock-Jupyter
        connection before reaching this method. Unlike an ordinary user touch,
        the refresh may recover a deadline that elapsed while the scheduler was
        restarting; terminal and stopping sessions remain fail closed.
        """

        row = await self.get_kernel(session, session_id, lock=True)
        if row.state not in {KernelState.RUNNING, KernelState.IDLE}:
            raise ConflictError("kernel is not ready")
        idle_timeout = int(row.request["idle_timeout_seconds"])
        row.last_activity_at = now
        row.expires_at = now + timedelta(seconds=idle_timeout)
        return row

    async def request_kernel_stop(
        self,
        session: AsyncSession,
        session_id: str,
        *,
        owner_username: str | None = None,
    ) -> KernelSessionRow:
        row = await self.get_kernel(
            session,
            session_id,
            lock=True,
            owner_username=owner_username,
        )
        if row.state not in {KernelState.STOPPED, KernelState.LOST, KernelState.STOPPING}:
            self._transition_kernel(row, KernelState.STOPPING)
            row.access_generation += 1
        return row

    async def claim_expired_kernels(
        self,
        session: AsyncSession,
        *,
        now: datetime,
    ) -> tuple[KernelSessionRow, ...]:
        rows = list(
            await session.scalars(
                select(KernelSessionRow)
                .where(
                    KernelSessionRow.state.in_([KernelState.RUNNING, KernelState.IDLE]),
                    KernelSessionRow.expires_at.is_not(None),
                    KernelSessionRow.expires_at <= now,
                )
                .with_for_update(skip_locked=True)
            )
        )
        for row in rows:
            self._transition_kernel(row, KernelState.STOPPING)
        return tuple(rows)

    async def stopping_kernels(self, session: AsyncSession) -> tuple[KernelSessionRow, ...]:
        # Database sessions deliberately disable implicit autoflush.  Reaping
        # changes expired rows to STOPPING immediately before this query, so
        # persist those transitions before selecting commands to deliver.
        await session.flush()
        return tuple(
            await session.scalars(
                select(KernelSessionRow)
                .where(KernelSessionRow.state == KernelState.STOPPING)
                .order_by(KernelSessionRow.created_at)
            )
        )

    async def reconcile_kernel_attempts(self, session: AsyncSession) -> int:
        # The caller commonly records a terminal attempt event and reconciles
        # its kernel sessions in the same transaction.  Make that new attempt
        # state visible to the join below when autoflush is disabled.
        await session.flush()
        rows = list(
            await session.scalars(
                select(KernelSessionRow)
                .join(AttemptRow, KernelSessionRow.attempt_id == AttemptRow.id)
                .where(
                    KernelSessionRow.state.not_in([KernelState.STOPPED, KernelState.LOST]),
                    AttemptRow.state.in_(list(_TERMINAL_ATTEMPT_STATES)),
                )
                .options(selectinload(KernelSessionRow.attempt))
                .with_for_update(skip_locked=True)
            )
        )
        for row in rows:
            attempt = row.attempt
            if attempt is None:  # pragma: no cover - protected by the inner join
                continue
            target = (
                KernelState.STOPPED
                if attempt.state in {AttemptState.SUCCEEDED, AttemptState.CANCELED}
                else KernelState.LOST
            )
            if target is KernelState.STOPPED and row.state in {
                KernelState.RUNNING,
                KernelState.IDLE,
            }:
                self._transition_kernel(row, KernelState.STOPPING)
            self._transition_kernel(row, target)
            row.runtime_id = None
            row.actor_id = None
            row.expires_at = None
        return len(rows)

    async def upsert_heartbeat(
        self,
        session: AsyncSession,
        heartbeat: SiteHeartbeat,
        *,
        connection_generation: str | None = None,
        received_at: datetime | None = None,
    ) -> bool:
        site = await session.get(SiteRow, heartbeat.site_id, with_for_update=True)
        if site is None:
            site = SiteRow(
                id=heartbeat.site_id,
                display_name=heartbeat.site_id,
                kind=heartbeat.site_kind,
                state=heartbeat.state,
                paid=heartbeat.site_kind is SiteKind.AUTODL,
            )
            session.add(site)
            await session.flush()
        elif (
            site.agent_session_id == heartbeat.agent_session_id
            and heartbeat.sequence <= site.last_sequence
        ):
            return False

        site.kind = heartbeat.site_kind
        site.state = heartbeat.state
        site.agent_session_id = heartbeat.agent_session_id
        if connection_generation is not None:
            site.agent_connection_generation = connection_generation
        site.last_sequence = heartbeat.sequence
        # Liveness is a scheduler observation, not an assertion made by the
        # remote host.  Site clocks can be skewed (terminal-only Jupyter hosts
        # commonly are), so production ingestion supplies the durable server
        # receipt timestamp.  Direct repository callers may omit it when
        # deliberately seeding historical snapshots in tests or recovery tools.
        site.heartbeat_at = received_at or heartbeat.sent_at
        site.storage_available = heartbeat.storage_available
        site.storage_authoritative = heartbeat.storage_authoritative
        site.supported_workloads = sorted(kind.value for kind in heartbeat.supported_workloads)
        site.cached_assets = sorted(heartbeat.cached_assets)
        site.queue_depth = heartbeat.queue_depth
        site.reliability = heartbeat.reliability
        site.gpu_safety_margin_bytes = heartbeat.gpu_safety_margin_bytes
        site.total_cpu = heartbeat.total_cpu
        site.available_cpu = heartbeat.available_cpu
        site.total_memory_bytes = heartbeat.total_memory_bytes
        site.available_memory_bytes = heartbeat.available_memory_bytes
        site.provider_instance_id = heartbeat.provider_instance_id
        site.region = heartbeat.region
        site.job_running = heartbeat.job_running
        site.assignment_queued_locally = heartbeat.assignment_queued_locally
        site.interactive_kernel_active = heartbeat.interactive_kernel_active
        site.artifact_flush_pending = heartbeat.artifact_flush_pending
        site.staging_operation_active = heartbeat.staging_operation_active
        site.updated_at = utc_now()

        await session.execute(delete(GPURow).where(GPURow.site_id == site.id))
        await session.execute(
            delete(ModelCapabilityRow).where(ModelCapabilityRow.site_id == site.id)
        )
        active_actors = list(
            await session.scalars(
                select(ModelActorRow)
                .where(
                    ModelActorRow.site_id == site.id,
                    ModelActorRow.active_allocation.is_(True),
                )
                .with_for_update()
            )
        )
        actor_by_gpu = {actor.gpu_uuid: actor for actor in active_actors}
        warm_gpu_uuids: set[str] = set()
        for gpu in heartbeat.gpus:
            session.add(
                GPURow(
                    site_id=site.id,
                    uuid=gpu.uuid,
                    index=gpu.index,
                    name=gpu.name,
                    total_memory_bytes=gpu.total_memory_bytes,
                    free_memory_bytes=gpu.free_memory_bytes,
                    utilization_percent=gpu.utilization_percent,
                    temperature_c=gpu.temperature_c,
                    foreign_process_count=len(gpu.blocking_foreign_processes),
                    foreign_memory_bytes=sum(
                        item.used_memory_bytes for item in gpu.blocking_foreign_processes
                    ),
                    externally_reserved=gpu.externally_reserved,
                    healthy=gpu.healthy,
                    observed_at=gpu.observed_at,
                )
            )
        for model in heartbeat.models:
            session.add(
                ModelCapabilityRow(
                    site_id=site.id,
                    alias=model.alias,
                    revision=model.revision,
                    backend=model.backend,
                    backend_version=model.backend_version,
                    dtype=model.dtype,
                    quantization=model.quantization or "",
                    adapter=model.adapter or "",
                    availability=model.availability,
                    gpu_uuid=model.gpu_uuid or "",
                    runtime_fingerprint=model.runtime_fingerprint or "",
                    replica_id=model.replica_id or "",
                    replica_gpu_uuids=list(model.replica_gpu_uuids),
                    replica_key_hash=model.replica_key_hash or "",
                    verified_at=model.verified_at,
                )
            )
            if model.availability is ModelAvailability.WARM and model.gpu_uuid:
                warm_gpu_uuids.add(model.gpu_uuid)
                key_data: dict[str, object] = {
                    "alias": model.alias,
                    "revision": model.revision,
                    "backend": model.backend,
                    "backend_version": model.backend_version,
                    "dtype": model.dtype,
                    "quantization": model.quantization,
                    "adapter": model.adapter,
                    "site_id": site.id,
                    "gpu_uuid": model.gpu_uuid,
                    "runtime_fingerprint": model.runtime_fingerprint,
                    "replica_id": model.replica_id,
                    "replica_gpu_uuids": list(model.replica_gpu_uuids or (model.gpu_uuid,)),
                    "replica_key_hash": model.replica_key_hash,
                    "idle_since": (
                        None if model.idle_since is None else model.idle_since.isoformat()
                    ),
                }
                key_hash = hashlib.sha256(
                    json.dumps(key_data, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                actor = actor_by_gpu.get(model.gpu_uuid)
                if actor is None:
                    actor = ModelActorRow(
                        id=new_id("actor"),
                        site_id=site.id,
                        gpu_uuid=model.gpu_uuid,
                        key_hash=key_hash,
                        key_data=key_data,
                        state=ActorState.IDLE,
                        active_allocation=True,
                        last_used_at=model.idle_since or received_at or heartbeat.sent_at,
                    )
                    session.add(actor)
                    actor_by_gpu[model.gpu_uuid] = actor
                else:
                    actor.key_hash = key_hash
                    actor.key_data = key_data
                    if actor.owner_attempt_id is None:
                        actor.state = ActorState.IDLE
                        if model.idle_since is not None:
                            actor.last_used_at = model.idle_since
        for gpu_uuid, actor in actor_by_gpu.items():
            if gpu_uuid not in warm_gpu_uuids:
                actor.active_allocation = False
                actor.state = ActorState.STOPPED
                actor.owner_attempt_id = None
        await session.flush()
        return True

    async def placement_inventory(self, session: AsyncSession) -> list[SiteCandidate]:
        releasing_sites = {
            row.site_id
            for row in await session.scalars(
                select(ProviderInstanceRow).where(ProviderInstanceRow.provider == "autodl")
            )
            if row.metadata_json.get("release_requested_at")
        }
        sites = list(
            await session.scalars(
                select(SiteRow).options(selectinload(SiteRow.gpus), selectinload(SiteRow.models))
            )
        )
        sites = [site for site in sites if site.id not in releasing_sites]
        # A local admission rejection proves that the GPU sample used for the
        # offer was stale by the time the site re-probed it.  Quarantine only
        # the exact rejected UUIDs, and only until that site's next heartbeat.
        # This is durable across scheduler restarts but cannot become a sticky
        # reservation: a later heartbeat receipt bounds and clears it.
        admission_blocked = {
            (site_id, gpu_uuid)
            for site_id, gpu_uuid in (
                await session.execute(
                    select(
                        LeaseGPUClaimRow.site_id,
                        LeaseGPUClaimRow.gpu_uuid,
                    )
                    .join(LeaseRow, LeaseRow.id == LeaseGPUClaimRow.lease_id)
                    .join(AttemptRow, AttemptRow.id == LeaseRow.attempt_id)
                    .join(SiteRow, SiteRow.id == LeaseGPUClaimRow.site_id)
                    .where(
                        AttemptRow.state == AttemptState.REJECTED,
                        AttemptRow.reason_code == "local_admission_rejected",
                        AttemptRow.completed_at.is_not(None),
                        SiteRow.heartbeat_at.is_not(None),
                        AttemptRow.completed_at >= SiteRow.heartbeat_at,
                    )
                    .distinct()
                )
            ).all()
        }
        active_claims = {
            (row.site_id, row.gpu_uuid)
            for row in await session.scalars(
                select(LeaseGPUClaimRow).where(LeaseGPUClaimRow.active.is_(True))
            )
        }
        actors = {
            (row.site_id, row.gpu_uuid): row
            for row in await session.scalars(
                select(ModelActorRow).where(ModelActorRow.active_allocation.is_(True))
            )
        }
        operator_reserved = await self.active_gpu_reservation_keys(session)
        active_leases = list(
            await session.scalars(
                select(LeaseRow)
                .where(
                    LeaseRow.state.in_([LeaseState.OFFERED, LeaseState.ACCEPTED, LeaseState.ACTIVE])
                )
                .options(selectinload(LeaseRow.attempt).selectinload(AttemptRow.job))
            )
        )
        reserved_host_resources: dict[str, tuple[float, int]] = {}
        for lease in active_leases:
            spec = JobSpec.model_validate(lease.attempt.job.spec)
            reserved_cpu, reserved_memory = reserved_host_resources.get(
                lease.site_id,
                (0.0, 0),
            )
            reserved_host_resources[lease.site_id] = (
                reserved_cpu + spec.resources.cpu,
                reserved_memory + round(spec.resources.memory_gb * 1024**3),
            )
        snapshots: list[SiteCandidate] = []
        for site in sites:
            models = tuple(
                ModelCapability(
                    alias=model.alias,
                    revision=model.revision,
                    backend=model.backend,
                    backend_version=model.backend_version,
                    dtype=model.dtype,
                    quantization=model.quantization or None,
                    adapter=model.adapter or None,
                    availability=model.availability,
                    gpu_uuid=model.gpu_uuid or None,
                    runtime_fingerprint=model.runtime_fingerprint or None,
                    replica_id=model.replica_id or None,
                    replica_gpu_uuids=tuple(model.replica_gpu_uuids or ()),
                    replica_key_hash=model.replica_key_hash or None,
                    verified_at=model.verified_at,
                )
                for model in site.models
            )
            gpu_candidates: list[GPUCandidate] = []
            for gpu in site.gpus:
                actor = actors.get((site.id, gpu.uuid))
                gpu_candidates.append(
                    GPUCandidate(
                        uuid=gpu.uuid,
                        index=gpu.index,
                        total_memory_bytes=gpu.total_memory_bytes,
                        free_memory_bytes=gpu.free_memory_bytes,
                        foreign_process_count=gpu.foreign_process_count,
                        foreign_memory_bytes=gpu.foreign_memory_bytes,
                        # Placement has one fail-closed reservation bit.  The
                        # operator/status APIs retain the distinction between
                        # site-reported external locks and scheduler-owned
                        # timed reservations.
                        externally_reserved=(
                            gpu.externally_reserved or (site.id, gpu.uuid) in operator_reserved
                        ),
                        healthy=gpu.healthy,
                        name=gpu.name,
                        utilization_percent=gpu.utilization_percent,
                        active_lease=(site.id, gpu.uuid) in active_claims,
                        admission_blocked=(site.id, gpu.uuid) in admission_blocked,
                        actor_key_hash=actor.key_hash if actor else None,
                        actor_key_data=actor.key_data if actor else None,
                        actor_state=actor.state if actor else None,
                    )
                )
            reserved_cpu, reserved_memory = reserved_host_resources.get(
                site.id,
                (0.0, 0),
            )
            available_cpu = (
                None
                if site.total_cpu is None or site.available_cpu is None
                else max(0.0, min(site.available_cpu, site.total_cpu - reserved_cpu))
            )
            available_memory = (
                None
                if site.total_memory_bytes is None or site.available_memory_bytes is None
                else max(
                    0,
                    min(
                        site.available_memory_bytes,
                        site.total_memory_bytes - reserved_memory,
                    ),
                )
            )
            snapshots.append(
                SiteCandidate(
                    site_id=site.id,
                    kind=site.kind,
                    state=site.state,
                    heartbeat_at=_aware(site.heartbeat_at or datetime.min.replace(tzinfo=UTC)),
                    storage_available=site.storage_available,
                    storage_authoritative=site.storage_authoritative,
                    supported_workloads=frozenset(site.supported_workloads),
                    queue_depth=site.queue_depth,
                    reliability=site.reliability,
                    paid=site.paid,
                    region=site.region,
                    gpus=tuple(gpu_candidates),
                    models=models,
                    cached_assets=frozenset(site.cached_assets),
                    safety_margin_bytes=site.gpu_safety_margin_bytes,
                    agent_session_id=site.agent_session_id,
                    agent_connection_generation=site.agent_connection_generation,
                    total_cpu=site.total_cpu,
                    available_cpu=available_cpu,
                    total_memory_bytes=site.total_memory_bytes,
                    available_memory_bytes=available_memory,
                )
            )
        return snapshots

    async def create_offer(
        self,
        session: AsyncSession,
        *,
        job: JobRow,
        site_id: str,
        gpu_uuids: Sequence[str],
        authority: FencingAuthority,
        offer_ttl: timedelta,
        token_ttl: timedelta | None = None,
        lease_ttl: timedelta = timedelta(seconds=90),
        replace_idle_actors: bool = False,
        now: datetime | None = None,
    ) -> AssignmentOffer:
        current = now or utc_now()
        if job.state is not JobState.ASSIGNING:
            raise ConflictError("job is not owned by the dispatcher")
        spec = JobSpec.model_validate(job.spec)
        if len(set(gpu_uuids)) != len(gpu_uuids):
            raise ConflictError("assignment GPUs must be unique")
        if len(gpu_uuids) != spec.resources.gpu_count:
            raise ConflictError("assignment GPU count does not match the job's requested GPU count")
        # Lock the same durable inventory rows used by operator reservation
        # creation.  This closes the stale-placement race across scheduler
        # processes: whichever transaction obtains the physical target first
        # wins, and the loser rechecks the committed claim/reservation below.
        locked_gpu_uuids = tuple(
            await session.scalars(
                select(GPURow.uuid)
                .where(
                    GPURow.site_id == site_id,
                    GPURow.uuid.in_(gpu_uuids),
                )
                .order_by(GPURow.uuid)
                .with_for_update()
            )
        )
        if set(locked_gpu_uuids) != set(gpu_uuids):
            raise ConflictError("one or more selected GPUs disappeared from site inventory")
        active = list(
            await session.scalars(
                select(LeaseGPUClaimRow)
                .where(
                    LeaseGPUClaimRow.site_id == site_id,
                    LeaseGPUClaimRow.gpu_uuid.in_(gpu_uuids),
                    LeaseGPUClaimRow.active.is_(True),
                )
                .with_for_update()
            )
        )
        if active:
            raise ConflictError("one or more GPUs acquired another active lease")
        operator_reserved = await session.scalar(
            select(GPUReservationRow.id)
            .where(
                GPUReservationRow.site_id == site_id,
                GPUReservationRow.gpu_uuid.in_(gpu_uuids),
                GPUReservationRow.state.in_(
                    [GPUReservationState.PENDING, GPUReservationState.READY]
                ),
                GPUReservationRow.expires_at > _aware(current),
            )
            .limit(1)
        )
        if operator_reserved is not None:
            raise ConflictError("one or more GPUs became operator-reserved")

        count = await session.scalar(
            select(func.count(AttemptRow.id)).where(AttemptRow.job_id == job.id)
        )
        attempt_number = int(count or 0) + 1
        restore_checkpoint_ref = await session.scalar(
            select(AttemptRow.checkpoint_ref)
            .where(
                AttemptRow.job_id == job.id,
                AttemptRow.checkpoint_ref.is_not(None),
            )
            .order_by(AttemptRow.number.desc())
            .limit(1)
        )
        attempt_id = new_id("attempt")
        lease_id = new_id("lease")
        claims: list[FenceClaim] = []
        for gpu_uuid in gpu_uuids:
            counter = await session.get(
                GPUFenceCounterRow,
                {"site_id": site_id, "gpu_uuid": gpu_uuid},
                with_for_update=True,
            )
            if counter is None:
                counter = GPUFenceCounterRow(site_id=site_id, gpu_uuid=gpu_uuid, epoch=0)
                session.add(counter)
                await session.flush()
            counter.epoch += 1
            claims.append(FenceClaim(gpu_uuid=gpu_uuid, epoch=counter.epoch))

        expires_at = current + offer_ttl
        token, _ = authority.issue(
            attempt_id=attempt_id,
            site_id=site_id,
            claims=tuple(claims),
            ttl=token_ttl or offer_ttl,
            now=current,
        )
        digest = authority.digest(token)
        attempt = AttemptRow(
            id=attempt_id,
            job_id=job.id,
            number=attempt_number,
            state=AttemptState.OFFERED,
            site_id=site_id,
            offer_expires_at=expires_at,
            fencing_token_digest=digest,
            restore_checkpoint_ref=restore_checkpoint_ref,
            created_at=current,
            updated_at=current,
        )
        lease = LeaseRow(
            id=lease_id,
            attempt=attempt,
            site_id=site_id,
            state=LeaseState.OFFERED,
            fencing_token_digest=digest,
            expires_at=expires_at,
            created_at=current,
            updated_at=current,
        )
        session.add(attempt)
        session.add(lease)
        for claim in claims:
            session.add(
                LeaseGPUClaimRow(
                    lease=lease,
                    site_id=site_id,
                    gpu_uuid=claim.gpu_uuid,
                    epoch=claim.epoch,
                    active=True,
                )
            )
        await session.flush()
        return AssignmentOffer(
            job_id=job.id,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            site_id=site_id,
            spec=spec,
            claims=tuple(claims),
            fencing_token=token,
            expires_at=expires_at,
            lease_ttl_seconds=lease_ttl.total_seconds(),
            checkpoint_ref=restore_checkpoint_ref,
            replace_idle_actors=replace_idle_actors,
        )

    async def accept_offer(
        self,
        session: AsyncSession,
        *,
        attempt_id: str,
        site_id: str,
        token: str,
        authority: FencingAuthority,
        lease_ttl: timedelta,
        now: datetime | None = None,
    ) -> AttemptRow:
        current = now or utc_now()
        payload = authority.verify(token, attempt_id=attempt_id, site_id=site_id, now=current)
        attempt = await self._get_attempt(session, attempt_id, lock=True)
        lease = attempt.lease
        if lease is None or lease.fencing_token_digest != authority.digest(token):
            raise StaleFenceError("fencing token is no longer current")
        if attempt.state is AttemptState.ACCEPTED:
            return attempt
        if attempt.offer_expires_at is None or _aware(attempt.offer_expires_at) <= current:
            raise StaleFenceError("assignment offer expired before acceptance")
        require_transition(attempt.state, AttemptState.ACCEPTED, ATTEMPT_TRANSITIONS)
        require_transition(lease.state, LeaseState.ACCEPTED, LEASE_TRANSITIONS)
        expected_claims = {(claim.gpu_uuid, claim.epoch) for claim in lease.claims}
        if expected_claims != {(claim.gpu_uuid, claim.epoch) for claim in payload.claims}:
            raise StaleFenceError("fencing claims do not match the lease")
        attempt.state = AttemptState.ACCEPTED
        attempt.accepted_at = current
        lease.state = LeaseState.ACCEPTED
        lease.expires_at = current + lease_ttl
        job = await self.get_job(session, attempt.job_id, lock=True)
        await self._transition_job(session, job, JobState.RUNNING, attempt_id=attempt.id)
        return attempt

    async def reject_offer(
        self,
        session: AsyncSession,
        *,
        attempt_id: str,
        site_id: str,
        token: str,
        authority: FencingAuthority,
        reason_code: str,
        message: str | None = None,
    ) -> AttemptRow:
        authority.verify(token, attempt_id=attempt_id, site_id=site_id)
        attempt = await self._get_attempt(session, attempt_id, lock=True)
        if attempt.fencing_token_digest != authority.digest(token):
            raise StaleFenceError("fencing token is no longer current")
        if attempt.state is AttemptState.REJECTED:
            return attempt
        require_transition(attempt.state, AttemptState.REJECTED, ATTEMPT_TRANSITIONS)
        attempt.state = AttemptState.REJECTED
        attempt.reason_code = reason_code
        attempt.message = message
        attempt.completed_at = utc_now()
        if attempt.lease:
            require_transition(attempt.lease.state, LeaseState.REJECTED, LEASE_TRANSITIONS)
            attempt.lease.state = LeaseState.REJECTED
            for claim in attempt.lease.claims:
                claim.active = False
        job = await self.get_job(session, attempt.job_id, lock=True)
        await self._transition_job(
            session,
            job,
            JobState.QUEUED,
            attempt_id=attempt.id,
            reason_code=reason_code,
            message=message,
        )
        return attempt

    async def renew_lease(
        self,
        session: AsyncSession,
        *,
        attempt_id: str,
        site_id: str,
        token: str,
        authority: FencingAuthority,
        lease_ttl: timedelta,
        token_ttl: timedelta,
        renewal_id: str,
        renewal_issued_at: datetime,
        replayed: bool = False,
        now: datetime | None = None,
    ) -> LeaseRenewalGrant:
        current = now or utc_now()
        payload = (
            authority.authenticate(token, attempt_id=attempt_id, site_id=site_id)
            if replayed
            else authority.verify(token, attempt_id=attempt_id, site_id=site_id, now=current)
        )
        attempt = await self._get_attempt(session, attempt_id, lock=True)
        lease = attempt.lease
        if (
            lease is None
            or attempt.site_id != site_id
            or lease.site_id != site_id
            or lease.state not in {LeaseState.ACCEPTED, LeaseState.ACTIVE, LeaseState.RELEASING}
        ):
            raise StaleFenceError("lease is no longer renewable")
        if _aware(lease.expires_at) <= current:
            raise StaleFenceError("lease expired before renewal")

        expected_claims = {(claim.gpu_uuid, claim.epoch) for claim in lease.claims}
        token_claims = {(claim.gpu_uuid, claim.epoch) for claim in payload.claims}
        finalizing = (
            lease.state is LeaseState.RELEASING and attempt.state is AttemptState.COMMITTING
        )
        if (
            len(payload.claims) != len(lease.claims)
            or expected_claims != token_claims
            or any(claim.active == finalizing for claim in lease.claims)
            or (lease.state is LeaseState.RELEASING and not finalizing)
        ):
            raise StaleFenceError("fencing claims do not match the active gang lease")

        issued_at = _aware(renewal_issued_at)
        rotated_token, rotated_payload = authority.issue(
            attempt_id=attempt_id,
            site_id=site_id,
            claims=payload.claims,
            ttl=token_ttl,
            now=issued_at,
            token_id=hashlib.sha256(f"lease-renewal:{renewal_id}".encode()).hexdigest(),
        )
        if rotated_payload.expires_at <= current:
            raise StaleFenceError("rotated fencing token expired before acknowledgement")

        request_digest = authority.digest(token)
        rotated_digest = authority.digest(rotated_token)
        if replayed:
            # The event journal has already established that this is the exact
            # request whose transaction committed.  Reconstruct its token but
            # neither extend the lease a second time nor accept any other stale
            # token after the digest rotation.
            if (
                attempt.fencing_token_digest != rotated_digest
                or lease.fencing_token_digest != rotated_digest
            ):
                raise StaleFenceError("renewal replay is no longer current")
        else:
            if (
                attempt.fencing_token_digest != request_digest
                or lease.fencing_token_digest != request_digest
            ):
                raise StaleFenceError("fencing token is no longer current")
            attempt.fencing_token_digest = rotated_digest
            attempt.updated_at = current
            lease.fencing_token_digest = rotated_digest
            lease.expires_at = current + lease_ttl
            lease.updated_at = current

        return LeaseRenewalGrant(
            attempt_id=attempt_id,
            fencing_token=rotated_token,
            fencing_token_expires_at=rotated_payload.expires_at,
            lease_expires_at=_aware(lease.expires_at),
        )

    async def update_attempt(
        self,
        session: AsyncSession,
        *,
        attempt_id: str,
        site_id: str,
        token: str,
        authority: FencingAuthority,
        target: AttemptState,
        reason_code: str | None = None,
        message: str | None = None,
        checkpoint_ref: str | None = None,
        artifacts: Sequence[ArtifactDescriptor] = (),
        gpu_released: bool = False,
        now: datetime | None = None,
    ) -> AttemptRow:
        current = now or utc_now()
        authority.verify(token, attempt_id=attempt_id, site_id=site_id, now=current)
        attempt = await self._get_attempt(session, attempt_id, lock=True)
        if attempt.fencing_token_digest != authority.digest(token):
            raise StaleFenceError("fencing token is no longer current")
        if gpu_released:
            if target is not AttemptState.COMMITTING:
                raise ValueError("GPU release is only valid at compute completion")
            if attempt.state not in {AttemptState.RUNNING, AttemptState.COMMITTING}:
                raise StaleFenceError("attempt is not eligible for finalization")
            lease = attempt.lease
            if lease is None or lease.state not in {LeaseState.ACTIVE, LeaseState.RELEASING}:
                raise StaleFenceError("finalization lease is unavailable")
            if _aware(lease.expires_at) <= current:
                raise StaleFenceError("lease expired before compute completion")
            # Retain renewable attempt authority, but relinquish physical GPU
            # claims. Historical epochs cannot authorize later GPU operations.
            lease.state = LeaseState.RELEASING
            for claim in lease.claims:
                claim.active = False
        if attempt.state is target:
            if checkpoint_ref is not None and checkpoint_ref != attempt.checkpoint_ref:
                attempt.checkpoint_ref = checkpoint_ref
                attempt.updated_at = current
            return attempt
        require_transition(attempt.state, target, ATTEMPT_TRANSITIONS)
        if target is AttemptState.SUCCEEDED:
            if not all(artifact.durable for artifact in artifacts):
                raise DurabilityError("attempt cannot succeed before all artifacts are durable")
            spec = JobSpec.model_validate(attempt.job.spec)
            delivered = {artifact.logical_path for artifact in artifacts}
            missing = set(spec.required_outputs) - delivered
            if missing:
                raise DurabilityError(
                    f"required outputs were not committed: {', '.join(sorted(missing))}"
                )

        attempt.state = target
        attempt.reason_code = reason_code
        attempt.message = message
        attempt.checkpoint_ref = checkpoint_ref or attempt.checkpoint_ref
        attempt.updated_at = current
        if target is AttemptState.STARTING and attempt.lease:
            attempt.lease.state = LeaseState.ACTIVE
        if target is AttemptState.RUNNING:
            attempt.started_at = current
        if target is AttemptState.COMMITTING:
            attempt.compute_finished_at = current
        from topodrome.scheduler.timing import set_phase

        phases = {
            AttemptState.STARTING: "starting",
            AttemptState.RUNNING: "compute",
            AttemptState.COMMITTING: "finalizing",
            AttemptState.PREEMPT_REQUESTED: "checkpointing",
            AttemptState.CHECKPOINTING: "checkpointing",
        }
        if target in phases:
            set_phase(attempt.job, phases[target], now=current)
        if target in _TERMINAL_ATTEMPT_STATES:
            attempt.completed_at = current
            await self._release_claims(attempt)
        if target is AttemptState.SUCCEEDED:
            for artifact in artifacts:
                session.add(
                    ArtifactRow(
                        job_id=attempt.job_id,
                        attempt_id=attempt.id,
                        logical_path=artifact.logical_path,
                        size_bytes=artifact.size_bytes,
                        sha256=artifact.sha256,
                        state=ArtifactState.COMMITTED,
                        committed_at=current,
                    )
                )
            job = await self.get_job(session, attempt.job_id, lock=True)
            await self._transition_job(session, job, JobState.SUCCEEDED, attempt_id=attempt.id)
        elif target in {AttemptState.FAILED, AttemptState.LOST, AttemptState.PREEMPTED}:
            await self._handle_retry(
                session,
                attempt,
                target=target,
                reason_code=reason_code,
                message=message,
                now=current,
            )
        elif target is AttemptState.CANCELED:
            job = await self.get_job(session, attempt.job_id, lock=True)
            if job.state is not JobState.CANCELING:
                await self._transition_job(session, job, JobState.CANCELING, attempt_id=attempt.id)
            await self._transition_job(session, job, JobState.CANCELED, attempt_id=attempt.id)
        return attempt

    async def terminal_attempt_for_fenced_message(
        self,
        session: AsyncSession,
        *,
        attempt_id: str,
        site_id: str,
        token: str,
        authority: FencingAuthority,
        job_id: str | None = None,
    ) -> AttemptRow | None:
        """Fence and classify an agent message that arrived after lease expiry.

        An agent can durably spool a report immediately before its lease is reaped.
        Replaying that message must not revive or rewrite the now-terminal
        attempt, but the scheduler still needs enough proof to acknowledge and
        remove it from the ordered spool.  A valid historical token is sufficient
        for that no-op only when it names this exact attempt and site and the
        optional report job matches the attempt's immutable job. Active attempts
        continue down the normal time-window and current-digest checks.
        """

        authority.authenticate(token, attempt_id=attempt_id, site_id=site_id)
        attempt = await self._get_attempt(session, attempt_id, lock=True)
        if attempt.site_id != site_id or (job_id is not None and attempt.job_id != job_id):
            raise StaleFenceError("agent message does not match its fenced assignment")
        if attempt.state in _TERMINAL_ATTEMPT_STATES:
            return attempt
        return None

    async def authenticate_active_attempt_message(
        self,
        session: AsyncSession,
        *,
        attempt_id: str,
        job_id: str,
        site_id: str,
        token: str,
        authority: FencingAuthority,
        now: datetime | None = None,
    ) -> AttemptRow:
        """Authorize an ephemeral message against the current gang lease.

        Unlike durable attempt events, live log chunks are intentionally not
        journaled. Every chunk must therefore prove, while the attempt row is
        locked, that both token digests, the lease time window, and every active
        GPU claim still match. A rotated, expired, released, or partially
        replaced gang is stale and cannot append bytes.
        """

        current = now or utc_now()
        payload = authority.verify(
            token,
            attempt_id=attempt_id,
            site_id=site_id,
            now=current,
        )
        attempt = await self._get_attempt(session, attempt_id, lock=True)
        lease = attempt.lease
        digest = authority.digest(token)
        active_states = {
            AttemptState.ACCEPTED,
            AttemptState.STARTING,
            AttemptState.RUNNING,
            AttemptState.PREEMPT_REQUESTED,
            AttemptState.CHECKPOINTING,
            AttemptState.COMMITTING,
        }
        if (
            attempt.job_id != job_id
            or attempt.site_id != site_id
            or attempt.state not in active_states
            or lease is None
            or lease.site_id != site_id
            or lease.state not in {LeaseState.ACCEPTED, LeaseState.ACTIVE, LeaseState.RELEASING}
            or _aware(lease.expires_at) <= current
            or attempt.fencing_token_digest is None
            or not hmac.compare_digest(attempt.fencing_token_digest, digest)
            or not hmac.compare_digest(lease.fencing_token_digest, digest)
        ):
            raise StaleFenceError("live message is not from the active lease")

        finalizing = (
            lease.state is LeaseState.RELEASING and attempt.state is AttemptState.COMMITTING
        )
        expected_claims = {
            (claim.gpu_uuid, claim.epoch)
            for claim in lease.claims
            if claim.active != finalizing and claim.site_id == site_id
        }
        token_claims = {(claim.gpu_uuid, claim.epoch) for claim in payload.claims}
        if (
            len(expected_claims) != len(lease.claims)
            or len(token_claims) != len(payload.claims)
            or expected_claims != token_claims
            or (lease.state is LeaseState.RELEASING and not finalizing)
        ):
            raise StaleFenceError("live message does not match the active gang lease")
        return attempt

    async def terminal_kernel_for_fenced_message(
        self,
        session: AsyncSession,
        *,
        session_id: str,
        attempt_id: str,
        site_id: str,
        token: str,
        authority: FencingAuthority,
    ) -> KernelSessionRow | None:
        """Authenticate and classify a late event for an exact kernel lease.

        Interactive lifecycle messages share the site's ordered durable spool
        with heartbeats. A kernel can become terminal centrally while its
        final ``running``/``stopping``/``stopped`` reports are still in flight.
        Once either the bound host attempt or the kernel itself is terminal,
        an authentic historical event is safe only as a no-op: it must be
        acknowledged so later heartbeats can drain, but it must never revive
        the session or complete an unrelated operation.
        """

        authority.authenticate(token, attempt_id=attempt_id, site_id=site_id)
        attempt = await self._get_attempt(session, attempt_id, lock=True)
        if attempt.site_id != site_id:
            raise StaleFenceError("kernel event does not match its fenced assignment")
        row = await self.get_kernel(session, session_id, lock=True)
        if row.attempt_id != attempt_id or row.site_id != site_id:
            raise StaleFenceError("kernel session is not bound to this attempt and site")
        if attempt.state in _TERMINAL_ATTEMPT_STATES or row.state in {
            KernelState.STOPPED,
            KernelState.LOST,
        }:
            return row
        return None

    async def cancel_job(
        self,
        session: AsyncSession,
        job_id: str,
        *,
        owner_username: str | None = None,
    ) -> JobRow:
        job = await self.get_job(
            session,
            job_id,
            lock=True,
            owner_username=owner_username,
        )
        if job.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED}:
            return job
        job.cancel_requested = True
        await self._transition_job(session, job, JobState.CANCELING)
        active_attempt = next(
            (
                attempt
                for attempt in reversed(job.attempts)
                if attempt.state not in _TERMINAL_ATTEMPT_STATES
            ),
            None,
        )
        if active_attempt is None:
            await self._transition_job(session, job, JobState.CANCELED)
        return job

    async def reap_expired_offers(
        self, session: AsyncSession, *, now: datetime | None = None
    ) -> int:
        current = now or utc_now()
        attempts = list(
            await session.scalars(
                select(AttemptRow)
                .where(
                    AttemptRow.state == AttemptState.OFFERED,
                    AttemptRow.offer_expires_at <= current,
                )
                .options(selectinload(AttemptRow.lease).selectinload(LeaseRow.claims))
                .with_for_update(skip_locked=True)
            )
        )
        for attempt in attempts:
            attempt.state = AttemptState.REJECTED
            attempt.reason_code = "offer_expired"
            attempt.completed_at = current
            if attempt.lease:
                attempt.lease.state = LeaseState.EXPIRED
                for claim in attempt.lease.claims:
                    claim.active = False
            job = await self.get_job(session, attempt.job_id, lock=True)
            if job.state is JobState.ASSIGNING:
                await self._transition_job(session, job, JobState.QUEUED, attempt_id=attempt.id)
            elif job.state is JobState.CANCELING:
                await self._transition_job(session, job, JobState.CANCELED, attempt_id=attempt.id)
        return len(attempts)

    async def reap_expired_leases(
        self,
        session: AsyncSession,
        *,
        now: datetime | None = None,
        finalization_grace: timedelta = timedelta(0),
    ) -> int:
        current = now or utc_now()
        leases = list(
            await session.scalars(
                select(LeaseRow)
                .where(
                    LeaseRow.state.in_(
                        [LeaseState.ACCEPTED, LeaseState.ACTIVE, LeaseState.RELEASING]
                    ),
                    LeaseRow.expires_at <= current,
                )
                .options(
                    selectinload(LeaseRow.claims),
                    selectinload(LeaseRow.attempt).selectinload(AttemptRow.job),
                )
                .with_for_update(skip_locked=True)
            )
        )
        for lease in leases:
            if (
                lease.state is LeaseState.RELEASING
                and lease.attempt.state is AttemptState.COMMITTING
                and lease.attempt.job.state is JobState.RUNNING
                and not any(claim.active for claim in lease.claims)
                and _aware(lease.expires_at) + finalization_grace > current
            ):
                continue
            lease.state = LeaseState.LOST
            lease.updated_at = current
            for claim in lease.claims:
                claim.active = False
            attempt = lease.attempt
            if attempt.state not in _TERMINAL_ATTEMPT_STATES:
                attempt.state = AttemptState.LOST
                attempt.reason_code = "lease_expired"
                attempt.message = "site stopped renewing the GPU lease"
                attempt.completed_at = current
                await self._handle_retry(
                    session,
                    attempt,
                    target=AttemptState.LOST,
                    reason_code=attempt.reason_code,
                    message=attempt.message,
                    now=current,
                )
        return len(leases)

    async def record_agent_event(
        self,
        session: AsyncSession,
        envelope: AgentEnvelope,
        *,
        now: datetime | None = None,
    ) -> tuple[bool, AgentEventAckRow]:
        raw = json.dumps(envelope.payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).digest()
        existing = await session.get(AgentEventAckRow, envelope.message_id)
        if existing:
            if existing.payload_digest != digest:
                raise ConflictError("agent reused message_id with a different payload")
            if (
                existing.site_id != envelope.site_id
                or existing.agent_session_id != envelope.agent_session_id
                or existing.sequence != envelope.sequence
            ):
                raise ConflictError("agent reused message_id with different stream coordinates")
            return False, existing
        receipt = AgentEventAckRow(
            message_id=envelope.message_id,
            site_id=envelope.site_id,
            agent_session_id=envelope.agent_session_id,
            sequence=envelope.sequence,
            payload_digest=digest,
            acknowledged_at=now or utc_now(),
        )
        session.add(receipt)
        await session.flush()
        return True, receipt

    @staticmethod
    def gpu_reservation_summary(row: GPUReservationRow) -> GPUReservationSummary:
        return GPUReservationSummary(
            reservation_id=row.id,
            site_id=row.site_id,
            gpu_uuid=row.gpu_uuid,
            gpu_index=row.gpu_index,
            requested_by=row.requested_by,
            reason=row.reason,
            force=row.force,
            duration_seconds=row.duration_seconds,
            state=row.state,
            created_at=_aware(row.created_at),
            expires_at=_aware(row.expires_at),
            canceled_at=(None if row.canceled_at is None else _aware(row.canceled_at)),
            eviction_requested_at=(
                None if row.eviction_requested_at is None else _aware(row.eviction_requested_at)
            ),
        )

    @staticmethod
    def summary(job: JobRow) -> JobSummary:
        from topodrome.scheduler.timing import job_timing

        return JobSummary(
            job_id=job.id,
            owner_username=job.owner_username,
            state=job.state,
            spec=JobSpec.model_validate(job.spec),
            attempts=len(job.attempts),
            retry_not_before=(
                None if job.retry_not_before is None else _aware(job.retry_not_before)
            ),
            created_at=_aware(job.created_at),
            updated_at=_aware(job.updated_at),
            array_id=job.array_id,
            array_index=job.array_index,
            blocked_reason=job.blocked_reason,
            timing=job_timing(job),
        )

    @staticmethod
    def _transition_kernel(row: KernelSessionRow, target: KernelState) -> None:
        if row.state is target:
            return
        require_transition(row.state, target, KERNEL_TRANSITIONS)
        row.state = target

    async def _get_attempt(
        self, session: AsyncSession, attempt_id: str, *, lock: bool
    ) -> AttemptRow:
        statement = (
            select(AttemptRow)
            .where(AttemptRow.id == attempt_id)
            .options(
                selectinload(AttemptRow.lease).selectinload(LeaseRow.claims),
                selectinload(AttemptRow.job),
            )
        )
        if lock:
            statement = statement.with_for_update()
        attempt = await session.scalar(statement)
        if attempt is None:
            raise NotFoundError(f"attempt {attempt_id} does not exist")
        return attempt

    async def _transition_job(
        self,
        session: AsyncSession,
        job: JobRow,
        target: JobState,
        *,
        attempt_id: str | None = None,
        reason_code: str | None = None,
        message: str | None = None,
    ) -> None:
        if job.state is target:
            return
        previous = job.state
        require_transition(previous, target, JOB_TRANSITIONS)
        job.state = target
        job.updated_at = utc_now()
        from topodrome.scheduler.timing import set_phase

        set_phase(
            job, "starting" if target is JobState.RUNNING else target.value, now=job.updated_at
        )
        if target is JobState.QUEUED and job.eligible_since is None:
            job.eligible_since = utc_now()
        elif target in {JobState.RUNNING, JobState.RETRY_WAIT}:
            job.eligible_since = None
        if target not in {JobState.QUEUED, JobState.RETRY_WAIT}:
            job.blocked_reason = None
        session.add(
            JobStateEventRow(
                job_id=job.id,
                from_state=previous.value,
                to_state=target.value,
                attempt_id=attempt_id,
                reason_code=reason_code,
                message=message,
            )
        )

    async def _release_claims(self, attempt: AttemptRow) -> None:
        if not attempt.lease:
            return
        if attempt.lease.state in {LeaseState.ACCEPTED, LeaseState.ACTIVE}:
            attempt.lease.state = LeaseState.RELEASING
        for claim in attempt.lease.claims:
            claim.active = False
        if attempt.lease.state is LeaseState.RELEASING:
            attempt.lease.state = LeaseState.RELEASED

    async def _handle_retry(
        self,
        session: AsyncSession,
        attempt: AttemptRow,
        *,
        target: AttemptState,
        reason_code: str | None,
        message: str | None,
        now: datetime,
    ) -> None:
        job = await self.get_job(session, attempt.job_id, lock=True)
        spec = JobSpec.model_validate(job.spec)
        if job.cancel_requested:
            if job.state is not JobState.CANCELING:
                await self._transition_job(session, job, JobState.CANCELING, attempt_id=attempt.id)
            await self._transition_job(session, job, JobState.CANCELED, attempt_id=attempt.id)
            return
        # Database sessions deliberately disable implicit autoflush.  Persist
        # the terminal state before counting the retry budget so this attempt
        # is included and max_attempts cannot be exceeded by one.
        await session.flush()
        completed_attempts = int(
            await session.scalar(
                select(func.count(AttemptRow.id)).where(
                    AttemptRow.job_id == job.id,
                    AttemptRow.state.in_(
                        [AttemptState.FAILED, AttemptState.LOST, AttemptState.PREEMPTED]
                    ),
                    or_(
                        AttemptRow.forced_reservation_id.is_(None),
                        AttemptRow.state != AttemptState.PREEMPTED,
                    ),
                )
            )
            or 0
        )
        categories = _retry_categories(target, reason_code)
        forced_retry = (
            attempt.forced_reservation_id is not None and target is AttemptState.PREEMPTED
        )
        retry_allowed = bool(categories & spec.retry.retry_on)
        if forced_retry or (retry_allowed and completed_attempts < spec.retry.max_attempts):
            job.retry_not_before = now + timedelta(seconds=spec.retry.backoff_seconds)
            await self._transition_job(
                session,
                job,
                JobState.RETRY_WAIT,
                attempt_id=attempt.id,
                reason_code=reason_code,
                message=message,
            )
        else:
            job.retry_not_before = None
            await self._transition_job(
                session,
                job,
                JobState.FAILED,
                attempt_id=attempt.id,
                reason_code=reason_code,
                message=message,
            )


def _retry_categories(target: AttemptState, reason_code: str | None) -> set[str]:
    categories = {target.value}
    if reason_code:
        categories.add(reason_code)
        if reason_code.startswith("retryable_"):
            categories.add(reason_code.removeprefix("retryable_"))
    return categories


_TERMINAL_ATTEMPT_STATES = {
    AttemptState.SUCCEEDED,
    AttemptState.REJECTED,
    AttemptState.PREEMPTED,
    AttemptState.FAILED,
    AttemptState.CANCELED,
    AttemptState.LOST,
}
