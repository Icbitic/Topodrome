from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from topodrome.db.models import ProviderOperationRow
from topodrome.db.session import Database
from topodrome.domain.enums import ProviderAction, ProviderOperationState
from topodrome.domain.errors import ConflictError, NotFoundError
from topodrome.providers.autodl.lifecycle import (
    OperationClaim,
    OperationKind,
    OperationState,
    ProviderOperationIntent,
    ProviderOperationRecord,
)


class SQLProviderOperationStore:
    """PostgreSQL-backed provider intent journal.

    Every external mutation is claimed and committed here before the provider
    client is called. The lifecycle controller reconciles nonterminal records
    instead of blindly replaying them after a scheduler restart.
    """

    def __init__(self, database: Database, *, provider: str = "autodl") -> None:
        self.database = database
        self.provider = provider

    async def claim(self, intent: ProviderOperationIntent) -> OperationClaim:
        async with self.database.transaction() as session:
            existing = await session.scalar(
                select(ProviderOperationRow)
                .where(ProviderOperationRow.idempotency_key == intent.idempotency_key)
                .with_for_update()
            )
            if existing is not None:
                return OperationClaim(record=_to_record(existing), created=False)
            row = ProviderOperationRow(
                id=intent.operation_id,
                provider=self.provider,
                action=ProviderAction(intent.kind.value),
                state=ProviderOperationState.INTENT,
                idempotency_key=intent.idempotency_key,
                request_redacted=dict(intent.request),
                external_instance_id=intent.instance_uuid,
                created_at=intent.recorded_at,
                updated_at=intent.recorded_at,
            )
            session.add(row)
            await session.flush()
            return OperationClaim(record=_to_record(row), created=True)

    async def mark_in_flight(self, operation_id: str) -> None:
        await self._set_state(operation_id, ProviderOperationState.CALLING)

    async def claim_intent_for_call(self, operation_id: str) -> bool:
        """Atomically own an API call that is provably not yet submitted.

        ``INTENT`` is committed before this transition.  Once ``CALLING`` is
        committed, recovery is observation-only because the provider call may
        have happened.  This compare-and-transition prevents two scheduler
        workers from replaying the same safe pre-call intent.
        """

        async with self.database.transaction() as session:
            result = await session.execute(
                update(ProviderOperationRow)
                .where(
                    ProviderOperationRow.id == operation_id,
                    ProviderOperationRow.state == ProviderOperationState.INTENT,
                )
                .values(state=ProviderOperationState.CALLING)
            )
            return int(getattr(result, "rowcount", 0)) == 1

    async def mark_succeeded(
        self,
        operation_id: str,
        *,
        provider_request_id: str | None,
        result_instance_uuid: str | None,
    ) -> None:
        async with self.database.transaction() as session:
            row = await _locked_operation(session, operation_id)
            row.state = ProviderOperationState.APPLIED
            row.external_request_id = provider_request_id
            row.result_external_instance_id = result_instance_uuid

    async def mark_ambiguous(self, operation_id: str, *, failure_code: str) -> None:
        async with self.database.transaction() as session:
            row = await _locked_operation(session, operation_id)
            row.state = ProviderOperationState.RECONCILING
            row.error_code = failure_code

    async def mark_failed(self, operation_id: str, *, failure_code: str) -> None:
        async with self.database.transaction() as session:
            row = await _locked_operation(session, operation_id)
            row.state = ProviderOperationState.FAILED
            row.error_code = failure_code

    async def unresolved(self, *, limit: int = 100) -> tuple[ProviderOperationRecord, ...]:
        """Return journal entries whose provider outcome still needs observation."""

        states = (
            ProviderOperationState.INTENT,
            ProviderOperationState.CALLING,
            ProviderOperationState.RECONCILING,
        )
        async with self.database.session() as session:
            rows = await session.scalars(
                select(ProviderOperationRow)
                .where(
                    ProviderOperationRow.provider == self.provider,
                    ProviderOperationRow.state.in_(states),
                )
                .order_by(ProviderOperationRow.created_at)
                .limit(limit)
            )
            return tuple(_to_record(row) for row in rows)

    async def records(self) -> tuple[ProviderOperationRecord, ...]:
        """Return the complete provider history for exact-run cleanup audits.

        Cleanup must include terminal CREATE rows: a process can die after the
        provider UUID is committed as ``APPLIED`` but before its caller stores
        that UUID in memory.  This deliberately unpaginated view is used by the
        isolated paid-test SQLite journal, whose lifetime and size are bounded
        by the acceptance runs.
        """

        async with self.database.session() as session:
            rows = await session.scalars(
                select(ProviderOperationRow)
                .where(ProviderOperationRow.provider == self.provider)
                .order_by(ProviderOperationRow.created_at)
            )
            return tuple(_to_record(row) for row in rows)

    async def _set_state(self, operation_id: str, state: ProviderOperationState) -> None:
        async with self.database.transaction() as session:
            row = await _locked_operation(session, operation_id)
            row.state = state


async def _locked_operation(session: AsyncSession, operation_id: str) -> ProviderOperationRow:
    row = await session.scalar(
        select(ProviderOperationRow)
        .where(ProviderOperationRow.id == operation_id)
        .with_for_update()
    )
    if row is None:
        raise NotFoundError(f"provider operation {operation_id} does not exist")
    return row


def _to_record(row: ProviderOperationRow) -> ProviderOperationRecord:
    kind = OperationKind(row.action.value)
    request: Mapping[str, object] = row.request_redacted
    instance_name_value = request.get("instance_name")
    instance_name = str(instance_name_value) if instance_name_value is not None else None
    intent = ProviderOperationIntent(
        operation_id=row.id,
        idempotency_key=row.idempotency_key,
        kind=kind,
        request=request,
        instance_uuid=row.external_instance_id,
        instance_name=instance_name,
        recorded_at=row.created_at,
    )
    state_map = {
        ProviderOperationState.INTENT: OperationState.INTENT_RECORDED,
        ProviderOperationState.CALLING: OperationState.IN_FLIGHT,
        ProviderOperationState.APPLIED: OperationState.SUCCEEDED,
        ProviderOperationState.RECONCILING: OperationState.AMBIGUOUS,
        ProviderOperationState.FAILED: OperationState.FAILED,
    }
    try:
        state = state_map[row.state]
    except KeyError as exc:
        raise ConflictError(f"unknown provider operation state: {row.state}") from exc
    return ProviderOperationRecord(
        intent=intent,
        state=state,
        provider_request_id=row.external_request_id,
        result_instance_uuid=row.result_external_instance_id,
        failure_code=row.error_code,
    )
