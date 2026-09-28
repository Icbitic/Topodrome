from __future__ import annotations

import httpx
import pytest

from topodrome.config import SchedulerSettings
from topodrome.db.models import AttemptRow, JobRow, SiteRow
from topodrome.domain.enums import AttemptState, JobState, SiteKind, SiteState
from topodrome.domain.specs import JobSpec
from topodrome.scheduler.app import create_app
from topodrome.storage.blobs import blob_transaction

pytestmark = pytest.mark.component


@pytest.mark.asyncio
async def test_claim_requires_own_live_attempt_and_site_auth(database, tmp_path):
    app = create_app(
        SchedulerSettings(
            global_root=tmp_path,
            storage_layout="blobs-v1",
            fence_signing_key="f" * 32,
            agent_tokens={"site": "test-site", "other": "test-other"},
        ),
        database=database,
        run_dispatcher=False,
    )
    async with database.transaction() as s:
        s.add(SiteRow(id="site", display_name="site", kind=SiteKind.LAB, state=SiteState.READY))
        s.add(
            JobRow(
                id="job",
                state=JobState.RUNNING,
                spec=JobSpec(entrypoint="run.py", model="model").model_dump(mode="json"),
            )
        )
        s.add(
            AttemptRow(
                id="attempt", job_id="job", site_id="site", number=1, state=AttemptState.COMMITTING
            )
        )
    body = dict(site_id="site", job_id="job", attempt_id="attempt", sha256="a" * 64, size_bytes=42)
    url = "/api/v1/agents/storage/blobs/claim"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        assert (await c.post(url, json=body)).status_code == 403
        auth = {"Authorization": "Bearer test-site"}
        assert (await c.post(url, json=body, headers=auth)).status_code == 204
        assert (await c.post(url, json=body, headers=auth)).status_code == 204
        with blob_transaction(tmp_path) as db:
            assert db.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
        assert (
            await c.post(
                url,
                json={**body, "site_id": "other"},
                headers={"Authorization": "Bearer test-other"},
            )
        ).status_code == 409
        async with database.transaction() as s:
            job = await s.get(JobRow, "job")
            job.state = JobState.CANCELED
        assert (await c.post(url, json=body, headers=auth)).status_code == 409
