from __future__ import annotations

import os
from datetime import timedelta

import httpx
import pytest
from sqlalchemy import func, select

from topodrome.client import AsyncClient
from topodrome.config import SchedulerSettings
from topodrome.db.models import ArtifactRow, AttemptRow, JobRow, LeaseRow
from topodrome.domain.enums import ArtifactState, AttemptState, JobState
from topodrome.domain.specs import JobSpec, utc_now
from topodrome.scheduler.app import create_app
from topodrome.storage.artifacts import AttemptArtifactStore
from topodrome.storage.blobs import blob_transaction
from topodrome.storage.download_pins import DownloadPins
from topodrome.storage.local import LocalStorage

pytestmark = pytest.mark.component
AUTH = {"Authorization": "Bearer operator-secret"}


async def setup_downloads(database, tmp_path, *, blobs=False):
    credentials = tmp_path / "credentials"
    credentials.mkdir(mode=0o700)
    settings = SchedulerSettings(
        global_root=tmp_path / "global",
        storage_layout="blobs-v1" if blobs else "attempt-v1",
        api_token="operator-secret",  # noqa: S106
        fence_signing_key="f" * 32,
        user_access={
            "enabled": True,
            "credential_store_path": credentials / "users.json",
            "require_local_account": False,
        },
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    storage = LocalStorage(settings.global_root)
    storage.content_addressed = blobs
    store = AttemptArtifactStore(storage, job_id="job-producer", attempt_id="attempt-one")
    files = {"results/a.txt": b"payload", "results/empty": b"", "results/sub/b.bin": b"other"}
    for name, body in files.items():
        store.stage_bytes(name, body)
    manifest = store.commit(tuple(files))
    old = utc_now() - timedelta(days=10)
    for artifact in manifest.artifacts:
        path = settings.global_root / (
            artifact.storage_path or f"{store.committed_prefix}/{artifact.logical_path}"
        )
        os.utime(path, (old.timestamp(), old.timestamp()))
    if blobs:
        with blob_transaction(settings.global_root) as db:
            db.execute("UPDATE objects SET touched=?", (old.timestamp(),))
    async with database.transaction() as session:
        session.add(
            JobRow(
                id="job-producer",
                state=JobState.SUCCEEDED,
                owner_username="alice",
                spec=JobSpec(entrypoint="run.py", model="test/test").model_dump(mode="json"),
                created_at=old,
                updated_at=old,
            )
        )
        session.add(
            AttemptRow(
                id="attempt-one",
                job_id="job-producer",
                number=1,
                state=AttemptState.SUCCEEDED,
                completed_at=old,
                created_at=old,
                updated_at=old,
            )
        )
        for artifact in manifest.artifacts:
            session.add(
                ArtifactRow(
                    job_id="job-producer",
                    attempt_id="attempt-one",
                    logical_path=artifact.logical_path,
                    size_bytes=artifact.size_bytes,
                    sha256=artifact.sha256,
                    state=ArtifactState.COMMITTED,
                )
            )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        tokens = {}
        for user in ("alice", "bob"):
            response = await c.post(f"/api/v1/access/users/{user}", headers=AUTH)
            assert response.status_code == 200, response.text
            tokens[user] = response.json()["token"]
    return app, settings, tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("blobs", [False, True])
async def test_download_job_file_tree_and_asset_without_creating_jobs_or_assets(
    database, tmp_path, blobs
):
    app, settings, tokens = await setup_downloads(database, tmp_path, blobs=blobs)
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient("http://test", token=tokens["alice"], transport=transport) as c:
        assert (
            await c.download("job://job-producer/results/a.txt", tmp_path / "one.txt")
        ).read_bytes() == b"payload"
        tree = await c.download("job://job-producer/results/", tmp_path / "results")
        assert (tree / "sub/b.bin").read_bytes() == b"other"
        assert (tree / "empty").read_bytes() == b""
        assert await c.assets() == []
        published = await c.publish_asset(
            name="named", version="v1", job_id="job-producer", path="results/"
        )
        named = await c.download(published.ref, tmp_path / "named")
        assert (named / "a.txt").read_bytes() == b"payload"
    async with database.session() as session:
        for model, expected in ((JobRow, 1), (AttemptRow, 1), (LeaseRow, 0)):
            assert await session.scalar(select(func.count()).select_from(model)) == expected
    assert not list(DownloadPins(settings.global_root).records())


@pytest.mark.asyncio
async def test_download_authorization_does_not_disclose_foreign_job_or_session(database, tmp_path):
    app, _, tokens = await setup_downloads(database, tmp_path)
    alice = {"Authorization": f"Bearer {tokens['alice']}"}
    bob = {"Authorization": f"Bearer {tokens['bob']}"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post(
            "/api/v1/downloads", json={"reference": "job://job-producer/results/"}, headers=alice
        )
        assert response.status_code == 201, response.text
        descriptor = response.json()
        assert "logical_path" not in str(descriptor)
        assert "private, no-store" == response.headers["Cache-Control"]
        foreign = await c.post(
            "/api/v1/downloads", json={"reference": "job://job-producer/results/"}, headers=bob
        )
        missing = await c.post(
            "/api/v1/downloads", json={"reference": "job://missing/results/"}, headers=bob
        )
        assert foreign.status_code == missing.status_code == 404
        assert foreign.json() == missing.json()
        identity = descriptor["download_id"]
        for method, path in (
            ("POST", f"/api/v1/downloads/{identity}/renew"),
            ("GET", f"/api/v1/downloads/{identity}/files/0"),
            ("DELETE", f"/api/v1/downloads/{identity}"),
        ):
            assert (await c.request(method, path, headers=bob)).status_code == 404
        assert (
            await c.get(f"/api/v1/downloads/{identity}/files/0", headers=alice)
        ).status_code == 200
        assert (
            await c.post(
                "/api/v1/downloads", json={"reference": "job://job-producer/results/a.txt"}
            )
        ).status_code == 401
        for visibility in ("private", "shared"):
            response = await c.post(
                "/api/v1/assets",
                json={
                    "name": visibility,
                    "version": "v1",
                    "job_id": "job-producer",
                    "path": "results/",
                    "visibility": visibility,
                },
                headers=alice,
            )
            assert response.status_code == 201
            response = await c.post(
                "/api/v1/downloads",
                json={"reference": f"asset://{visibility}/{visibility}@v1"},
                headers=bob,
            )
            assert response.status_code == (201 if visibility == "shared" else 404)


@pytest.mark.asyncio
async def test_range_etag_empty_file_and_session_release(database, tmp_path):
    app, _, tokens = await setup_downloads(database, tmp_path)
    headers = {"Authorization": f"Bearer {tokens['alice']}"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
    ) as c:
        response = await c.post(
            "/api/v1/downloads", json={"reference": "job://job-producer/results/"}
        )
        identity = response.json()["download_id"]
        item = response.json()["files"][0]
        url = f"/api/v1/downloads/{identity}/files/0"
        etag = f'"{item["sha256"]}"'
        response = await c.get(url, headers={"Range": "bytes=3-", "If-Match": etag})
        assert response.status_code == 206 and response.content == b"load"
        assert response.headers["Content-Range"] == "bytes 3-6/7"
        assert response.headers["Content-Length"] == "4"
        assert response.headers["ETag"] == etag
        assert response.headers["Accept-Ranges"] == "bytes"
        assert response.headers["Cache-Control"] == "private, no-store"
        for range_value in (
            "bytes=7-",
            "bytes=-2",
            "bytes=0-3",
            "bytes=0-,3-",
            "bytes=" + "9" * 5000 + "-",
        ):
            response = await c.get(url, headers={"Range": range_value})
            assert response.status_code == 416 and not response.content
            assert response.headers["Content-Range"] == "bytes */7"
        assert (await c.get(url, headers={"If-Match": '"different"'})).status_code == 412
        assert (await c.get(f"/api/v1/downloads/{identity}/files/1")).content == b""
        for index in (-1, 3):
            assert (await c.get(f"/api/v1/downloads/{identity}/files/{index}")).status_code == 404
        renewal = await c.post(f"/api/v1/downloads/{identity}/renew")
        assert set(renewal.json()) == {"download_id", "expires_at"}
        assert (await c.delete(f"/api/v1/downloads/{identity}")).status_code == 204
        assert (await c.delete(f"/api/v1/downloads/{identity}")).status_code == 204
        assert (await c.get(url)).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("blobs", [False, True])
@pytest.mark.parametrize("end", ["close", "expire"])
async def test_active_download_protects_data_from_gc_until_close_or_expiry(
    database, tmp_path, monkeypatch, blobs, end
):
    app, settings, tokens = await setup_downloads(database, tmp_path, blobs=blobs)
    headers = {"Authorization": f"Bearer {tokens['alice']}"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post(
            "/api/v1/downloads",
            json={"reference": "job://job-producer/results/a.txt"},
            headers=headers,
        )
        identity = response.json()["download_id"]
        pin = DownloadPins(settings.global_root).get(identity, "alice")
        path = settings.global_root / pin.members[0].logical_path
        protected = await c.post("/api/v1/storage/gc?apply=true", headers=AUTH)
        assert protected.status_code == 200, protected.text
        assert path.is_file()
        # A reconstructed app reads durable pins, rather than an in-memory registry.
        restarted = create_app(settings, database=database, run_dispatcher=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restarted), base_url="http://test", headers=headers
        ) as other:
            assert (await other.get(f"/api/v1/downloads/{identity}/files/0")).content == b"payload"
        if end == "close":
            assert (
                await c.delete(f"/api/v1/downloads/{identity}", headers=headers)
            ).status_code == 204
        else:
            import topodrome.storage.download_pins as pins

            future = utc_now() + timedelta(hours=1)
            monkeypatch.setattr(pins, "utc_now", lambda: future)
            assert (
                await c.get(f"/api/v1/downloads/{identity}/files/0", headers=headers)
            ).status_code == 404
        freed = await c.post("/api/v1/storage/gc?apply=true", headers=AUTH)
        assert freed.status_code == 200, freed.text
        assert not path.exists()
        assert (
            await c.post(
                "/api/v1/downloads",
                json={"reference": "job://job-producer/results/a.txt"},
                headers=headers,
            )
        ).status_code == 400


@pytest.mark.asyncio
async def test_only_committed_success_outputs_and_safe_references_are_downloadable(
    database, tmp_path
):
    app, _, tokens = await setup_downloads(database, tmp_path)
    headers = {"Authorization": f"Bearer {tokens['alice']}"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
    ) as c:
        for ref in (
            "file:///etc/passwd",
            "job://job-producer/../other",
            "job://job-producer/results/?query=x",
            "asset://private/missing@v1/../../other",
        ):
            assert (await c.post("/api/v1/downloads", json={"reference": ref})).status_code in {
                400,
                404,
            }
        async with database.transaction() as session:
            job = await session.get(JobRow, "job-producer")
            job.state = JobState.RUNNING
        response = await c.post(
            "/api/v1/downloads", json={"reference": "job://job-producer/results/"}
        )
        assert response.status_code == 409


@pytest.mark.asyncio
async def test_asset_deletion_does_not_break_an_already_accepted_download(database, tmp_path):
    app, settings, tokens = await setup_downloads(database, tmp_path, blobs=True)
    headers = {"Authorization": f"Bearer {tokens['alice']}"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
    ) as c:
        published = await c.post(
            "/api/v1/assets",
            json={"name": "named", "version": "v1", "job_id": "job-producer", "path": "results/"},
        )
        assert published.status_code == 201, published.text
        response = await c.post("/api/v1/downloads", json={"reference": "asset://private/named@v1"})
        identity = response.json()["download_id"]
        assert (await c.delete("/api/v1/assets/named/v1")).status_code == 200
        assert (await c.post("/api/v1/storage/gc?apply=true", headers=AUTH)).status_code == 200
        assert (await c.get(f"/api/v1/downloads/{identity}/files/0")).content == b"payload"
        assert (
            await c.post("/api/v1/downloads", json={"reference": "asset://private/named@v1"})
        ).status_code == 404
        paths = [
            settings.global_root / m.logical_path
            for m in DownloadPins(settings.global_root).get(identity, "alice").members
        ]
        assert (await c.delete(f"/api/v1/downloads/{identity}")).status_code == 204
        assert (await c.post("/api/v1/storage/gc?apply=true", headers=AUTH)).status_code == 200
        assert all(not p.exists() for p in paths)
