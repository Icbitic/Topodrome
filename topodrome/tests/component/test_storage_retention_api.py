from __future__ import annotations

import asyncio
import os
import threading
from datetime import timedelta

import httpx
import pytest

from topodrome.config import SchedulerSettings
from topodrome.db.models import ArtifactRow, AttemptRow, JobRow
from topodrome.domain.enums import ArtifactState, AttemptState, JobState
from topodrome.domain.specs import JobSpec, utc_now
from topodrome.scheduler.app import create_app
from topodrome.scheduler.source_store import SourceBundle
from topodrome.storage.artifacts import AttemptArtifactStore
from topodrome.storage.local import LocalStorage

pytestmark = pytest.mark.component


async def setup(database, tmp_path):
    settings = SchedulerSettings(global_root=tmp_path / "global", fence_signing_key="f" * 32)
    app = create_app(settings, database=database, run_dispatcher=False)
    root = settings.global_root
    old = utc_now() - timedelta(days=10)
    store = AttemptArtifactStore(LocalStorage(root), job_id="job-one", attempt_id="attempt-one")
    for name in ("keep.bin", "expire.bin"):
        store.stage_bytes(name, b"payload")
    manifest = store.commit(("keep.bin", "expire.bin"))
    for name in ("keep.bin", "expire.bin"):
        path = root / store.committed_prefix / name
        os.utime(path, (old.timestamp(), old.timestamp()))
    spec = JobSpec(entrypoint="run.py", model="test/test-model")
    async with database.transaction() as session:
        session.add(
            JobRow(
                id="job-one",
                state=JobState.SUCCEEDED,
                spec=spec.model_dump(mode="json"),
                owner_username="@operator",
                created_at=old,
                updated_at=old,
            )
        )
        session.add(
            AttemptRow(
                id="attempt-one",
                job_id="job-one",
                number=1,
                state=AttemptState.SUCCEEDED,
                completed_at=old,
                created_at=old,
                updated_at=old,
            )
        )
        for item in manifest.artifacts:
            session.add(
                ArtifactRow(
                    job_id="job-one",
                    attempt_id="attempt-one",
                    logical_path=item.logical_path,
                    size_bytes=item.size_bytes,
                    sha256=item.sha256,
                    state=ArtifactState.COMMITTED,
                )
            )
    import base64

    source = app.state.service.sources.put(
        SourceBundle(
            files={
                "run.py": base64.b64encode(b"def run(context): pass").decode(),
            }
        )
    )
    return app, root, source.source_ref


@pytest.mark.asyncio
async def test_gc_preserves_published_asset_and_rejects_expired_input(database, tmp_path):
    app, root, source = await setup(database, tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        published = await c.post(
            "/api/v1/assets",
            json={
                "name": "permanent",
                "version": "v1",
                "job_id": "job-one",
                "path": "keep.bin",
            },
        )
        assert published.status_code == 201, published.text
        dry = await c.post("/api/v1/storage/gc")
        assert dry.json()["candidate_files"] == 1
        assert dry.json()["deleted_files"] == 0
        applied = await c.post("/api/v1/storage/gc?apply=true")
        assert applied.json()["deleted_files"] == 1
        assert (root / "jobs/job-one/attempts/attempt-one/committed/keep.bin").is_file()
        for reference in ("asset://private/permanent@v1", "job://job-one/keep.bin"):
            submitted = await c.post(
                "/api/v1/jobs",
                json={
                    "entrypoint": "run.py",
                    "model": "test/test-model",
                    "source_ref": source,
                    "assets": {"data": reference},
                },
            )
            assert submitted.status_code == 202, submitted.text
        submitted = await c.post(
            "/api/v1/jobs",
            json={
                "entrypoint": "run.py",
                "model": "test/test-model",
                "source_ref": source,
                "assets": {"data": "job://job-one/expire.bin"},
            },
        )
        assert submitted.status_code == 400 and "expired" in submitted.text
        publish_expired = await c.post(
            "/api/v1/assets",
            json={
                "name": "too-late",
                "version": "v1",
                "job_id": "job-one",
                "path": "expire.bin",
            },
        )
        assert publish_expired.status_code == 400


@pytest.mark.asyncio
async def test_queued_consumer_protects_expired_artifact(database, tmp_path):
    app, root, source = await setup(database, tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post(
            "/api/v1/jobs",
            json={
                "entrypoint": "run.py",
                "model": "test/test-model",
                "source_ref": source,
                "assets": {"data": "job://job-one/expire.bin"},
            },
        )
        assert response.status_code == 202
        result = await c.post("/api/v1/storage/gc?apply=true")
        assert result.json()["deleted_files"] == 1
        assert (root / "jobs/job-one/attempts/attempt-one/committed/expire.bin").is_file()


@pytest.mark.asyncio
async def test_collection_and_publication_are_serialized(database, tmp_path, monkeypatch):
    app, _, _ = await setup(database, tmp_path)
    import topodrome.scheduler.service as service_module

    original = service_module.collect_retained_storage
    entered, resume = threading.Event(), threading.Event()

    def delayed(*args, **kwargs):
        entered.set()
        assert resume.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(service_module, "collect_retained_storage", delayed)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        gc = asyncio.create_task(c.post("/api/v1/storage/gc?apply=true"))
        assert await asyncio.to_thread(entered.wait, 10)
        publish = asyncio.create_task(
            c.post(
                "/api/v1/assets",
                json={
                    "name": "racing",
                    "version": "v1",
                    "job_id": "job-one",
                    "path": "expire.bin",
                },
            )
        )
        await asyncio.sleep(0.02)
        assert not publish.done()
        resume.set()
        assert (await gc).status_code == 200
        assert (await publish).status_code == 400


@pytest.mark.asyncio
async def test_storage_gc_requires_operator(database, tmp_path):
    credentials = tmp_path / "credentials"
    credentials.mkdir(mode=0o700)
    settings = SchedulerSettings(
        global_root=tmp_path / "global",
        api_token="test-operator",  # noqa: S106
        fence_signing_key="f" * 32,
        user_access={
            "enabled": True,
            "credential_store_path": credentials / "users.json",
            "require_local_account": False,
        },
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        granted = await c.post(
            "/api/v1/access/users/alice", headers={"Authorization": "Bearer test-operator"}
        )
        token = granted.json()["token"]
        response = await c.get("/api/v1/system", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 403
        assert app.state.service.system_metrics.task is None
        for url in ("/api/v1/storage/gc", "/api/v1/storage/gc?apply=true"):
            response = await c.post(url, headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == 403


@pytest.mark.asyncio
async def test_system_metrics_is_cached_non_mutating_and_no_store(database, tmp_path):
    from topodrome.client.async_client import AsyncClient

    app, root, _ = await setup(database, tmp_path)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    metrics = app.state.service.system_metrics
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            response = await c.get("/api/v1/system")
            assert response.status_code == 200
            assert response.headers["Cache-Control"] == "no-store"
            await metrics.task
        async with AsyncClient("http://test", transport=httpx.ASGITransport(app=app)) as client:
            report = await client.system_metrics()
            assert report["storage_state"] == "ready"
            assert report["storage"]["eligible_24h_bytes"] > 0
            assert report["retention"]["enabled"] is False
            assert report["scheduler_rss_bytes"] > 0
        after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        assert after == before
    finally:
        await metrics.close()
