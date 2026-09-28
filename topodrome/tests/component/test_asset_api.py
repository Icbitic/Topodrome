from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from topodrome.client.async_client import AsyncClient
from topodrome.config import SchedulerSettings
from topodrome.db.models import ArtifactRow, AttemptRow, JobRow
from topodrome.db.session import Database
from topodrome.domain.enums import ArtifactState, AttemptState, JobState
from topodrome.domain.specs import JobSpec
from topodrome.scheduler.app import create_app
from topodrome.storage import LocalStorage
from topodrome.storage.artifacts import AttemptArtifactStore

pytestmark = pytest.mark.component


@pytest.mark.asyncio
async def test_named_asset_publish_and_consume_through_api(
    database: Database,
    tmp_path: Path,
) -> None:
    global_root = tmp_path / "global"
    producer_job_id = "job-producer"
    producer_attempt_id = "attempt-success"
    store = AttemptArtifactStore(
        LocalStorage(global_root),
        job_id=producer_job_id,
        attempt_id=producer_attempt_id,
        clock=lambda: datetime(2026, 9, 13, tzinfo=UTC),
    )
    store.stage_bytes("lens/config.json", b"{}\n")
    store.stage_bytes("lens/params.pt", b"weights")
    manifest = store.commit(("lens/config.json", "lens/params.pt"))

    producer_spec = JobSpec(
        entrypoint="producer.py",
        source_ref="sha256:" + "0" * 64,
        model="model-a",
    )
    async with database.transaction() as session:
        session.add(
            JobRow(
                id=producer_job_id,
                owner_username="@operator",
                state=JobState.SUCCEEDED,
                spec=producer_spec.model_dump(mode="json"),
                priority=0,
            )
        )
        session.add(
            AttemptRow(
                id=producer_attempt_id,
                job_id=producer_job_id,
                number=1,
                state=AttemptState.SUCCEEDED,
                completed_at=datetime(2026, 9, 13, tzinfo=UTC),
            )
        )
        session.add_all(
            [
                ArtifactRow(
                    job_id=producer_job_id,
                    attempt_id=producer_attempt_id,
                    logical_path=item.logical_path,
                    size_bytes=item.size_bytes,
                    sha256=item.sha256,
                    state=ArtifactState.COMMITTED,
                    committed_at=manifest.committed_at,
                )
                for item in manifest.artifacts
            ]
        )

    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=global_root,
        fence_signing_key="f" * 32,
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    transport = httpx.ASGITransport(app=app)
    source = tmp_path / "consumer"
    source.mkdir()
    (source / "consumer.py").write_text("def run(context):\n    return context\n")

    async with AsyncClient("http://test", transport=transport) as client:
        published = await client.publish_asset(
            name="tuned-lens",
            version="v1",
            job_id=producer_job_id,
            path="lens/",
            visibility="shared",
        )
        assert published.ref == "asset://shared/tuned-lens@v1"
        assert published.kind == "directory"
        assert [item.relative_path for item in published.files] == [
            "config.json",
            "params.pt",
        ]
        assert await client.assets(visibility="shared") == [published]
        assert await client.asset("tuned-lens", "v1", visibility="shared") == published
        downloaded = await client.download_asset(
            "tuned-lens", "v1", tmp_path / "downloaded-lens", visibility="shared"
        )
        assert (downloaded / "config.json").read_bytes() == b"{}\n"
        assert (downloaded / "params.pt").read_bytes() == b"weights"

        consumer = await client.submit(
            entrypoint="consumer.py",
            source_dir=source,
            model="model-a",
            assets={"lens": published.ref},
        )

        tagged = await client.update_asset_tags(
            "tuned-lens", "v1", add=("Project:Interp", "kind:lens"), visibility="shared"
        )
        assert tagged.tags == ("kind:lens", "project:interp")
        assert tagged.digest == published.digest
        assert tagged.manifest_sha256 == published.manifest_sha256
        assert await client.assets(tags=("project:interp", "kind:lens")) == [tagged]
        assert await client.assets(tags=("project:missing",)) == []
        deleted = await client.delete_asset("tuned-lens", "v1", visibility="shared")
        assert not deleted.already_deleted
        assert not deleted.payloads_deleted
        assert await client.assets() == []
        with pytest.raises(RuntimeError, match="404"):
            await client.asset("tuned-lens", "v1", visibility="shared")
        with pytest.raises(RuntimeError, match="404"):
            await client.update_asset_tags("tuned-lens", "v1", add=("new",), visibility="shared")
        with pytest.raises(RuntimeError, match="404"):
            await client.submit(
                entrypoint="consumer.py",
                source_dir=source,
                model="model-a",
                assets={"lens": published.ref},
            )
        with pytest.raises(RuntimeError, match="409"):
            await client.publish_asset(
                name="tuned-lens",
                version="v1",
                job_id=producer_job_id,
                path="lens/",
                visibility="shared",
            )
        repeated = await client.delete_asset("tuned-lens", "v1", visibility="shared")
        assert repeated.already_deleted
        assert repeated.deleted_at == deleted.deleted_at
        assert (await client.status(consumer.job_id)).spec.assets == consumer.spec.assets

    pinned = consumer.spec.assets["lens"]
    assert pinned.pinned
    assert pinned.kind == "directory"
    assert pinned.digest == published.digest
    assert pinned.total_bytes == len(b"{}\n") + len(b"weights")
    assert tuple(member.relative_path for member in pinned.members) == (
        "config.json",
        "params.pt",
    )
