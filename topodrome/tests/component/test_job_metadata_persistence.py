import pytest

from topodrome.db.repositories import SchedulerRepository
from topodrome.domain.errors import ConflictError
from topodrome.domain.specs import JobSpec


@pytest.mark.asyncio
async def test_legacy_idempotency_and_metadata_conflict(database):
    repository = SchedulerRepository()
    spec = JobSpec(entrypoint="run.py", model="llama")
    async with database.session() as session:
        job = await repository.submit_job(session, spec, idempotency_key="legacy")
        job.spec = {
            k: v
            for k, v in job.spec.items()
            if k not in {"name", "description", "estimated_runtime_seconds"}
        }
        await session.commit()
        job_id = job.id
    async with database.session() as session:
        repeated = await repository.submit_job(session, spec, idempotency_key="legacy")
        assert repeated.id == job_id
        with pytest.raises(ConflictError):
            await repository.submit_job(
                session, spec.model_copy(update={"name": "changed"}), idempotency_key="legacy"
            )
        with pytest.raises(ConflictError):
            await repository.submit_job(
                session,
                spec.model_copy(update={"estimated_runtime_seconds": 10}),
                idempotency_key="legacy",
            )
