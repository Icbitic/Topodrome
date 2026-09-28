from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from topodrome.asset_upload import CONTENT_TYPE
from topodrome.client import AsyncClient
from topodrome.config import SchedulerSettings
from topodrome.db.models import JobRow
from topodrome.db.session import Database
from topodrome.scheduler.app import create_app
from topodrome.storage import LocalStorage
from topodrome.storage.assets import AssetResolver

pytestmark = pytest.mark.component


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["attempt-v1", "blobs-v1"])
async def test_upload_authorized_without_jobs_and_pin_consumer(
    database: Database,
    tmp_path: Path,
    layout: str,
) -> None:
    credentials = tmp_path / "credentials"
    credentials.mkdir(mode=0o700)
    app = create_app(
        SchedulerSettings(
            storage_layout=layout,
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path / "global",
            api_token="test-operator",  # noqa: S106 - synthetic test credential
            fence_signing_key="f" * 32,
            user_access={
                "enabled": True,
                "credential_store_path": credentials / "users.json",
                "require_local_account": False,
            },
        ),
        database=database,
        run_dispatcher=False,
    )
    transport = httpx.ASGITransport(app=app)
    local = tmp_path / "input.txt"
    local.write_text("locally edited asset\n")
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        denied = await raw.post("/api/v1/asset-uploads/input/v1", content=b"not parsed")
        assert denied.status_code == 401
        tokens = {}
        for user in ("alice", "bob"):
            result = await raw.post(
                f"/api/v1/access/users/{user}", headers={"Authorization": "Bearer test-operator"}
            )
            assert result.status_code == 200
            tokens[user] = result.json()["token"]
    async with AsyncClient("http://test", token=tokens["alice"], transport=transport) as alice:
        asset = await alice.upload_asset(local, name="input", version="v1")
        assert asset.producer_job_id is None
        assert asset.owner_username == "alice"
        assert asset.ref == "asset://private/input@v1"
        assert await alice.asset("input", "v1") == asset
        copied = await alice.download_asset("input", "v1", tmp_path / "copied-input")
        assert copied.read_bytes() == local.read_bytes()
        async with database.session() as session:
            assert await session.scalar(select(func.count()).select_from(JobRow)) == 0
        source = tmp_path / "code"
        source.mkdir()
        (source / "run.py").write_text("def run(context): return {}\n")
        consumer = await alice.submit(
            source_dir=source, entrypoint="run.py", model="model-a", assets={"input": asset.ref}
        )
        pin = consumer.spec.assets["input"]
        assert pin.pinned
        assert pin.members[0].logical_path.startswith(
            "blobs/sha256/" if layout == "blobs-v1" else "artifacts/uploads/"
        )
        assert pin.digest == asset.digest
        resolved = AssetResolver(
            LocalStorage(tmp_path / "global"), cache_root=tmp_path / "cache"
        ).resolve(pin)
        assert resolved.local_path.read_bytes() == local.read_bytes()
        shared = await alice.upload_asset(local, name="input", version="v1", visibility="shared")
        if layout == "blobs-v1":
            blobs = [p for p in (tmp_path / "global/blobs").rglob("*") if p.is_file()]
            assert len(blobs) == 1
            assert blobs[0].stat().st_nlink == 1
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {tokens['bob']}"},
    ) as bob:
        assert (await bob.get("/api/v1/assets/input/v1")).status_code == 404
        assert (await bob.get("/api/v1/assets/input/v1/files/0")).status_code == 404
        shared_file = await bob.get("/api/v1/assets/input/v1/files/0?visibility=shared")
        assert shared_file.status_code == 200
        assert shared_file.content == local.read_bytes()
        assert shared_file.headers["cache-control"] == "no-store"
        assert (
            await bob.get("/api/v1/assets/input/v1/files/99?visibility=shared")
        ).status_code == 404
        assert (await bob.get("/api/v1/assets/input/v1?visibility=shared")).json()[
            "digest"
        ] == shared.digest
        bad = await bob.post(
            "/api/v1/asset-uploads/bad/v1",
            content=b"broken",
            headers={"Content-Type": CONTENT_TYPE},
        )
        assert 400 <= bad.status_code < 500
        assert (await bob.get("/api/v1/assets/bad/v1")).status_code == 404
        for visibility in ("shared", "private"):
            url = f"/api/v1/assets/input/v1?visibility={visibility}"
            assert (await bob.delete(url)).status_code == 404
            tagged = await bob.patch(
                f"/api/v1/assets/input/v1/tags?visibility={visibility}", json={"add": ["mine"]}
            )
            assert tagged.status_code == 404

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {tokens['alice']}"},
    ) as alice:
        assert (await alice.get("/api/v1/assets?tag=bad%20tag")).status_code == 422
        for name in ("bad%20name", "%2Ebad"):
            assert (await alice.delete(f"/api/v1/assets/{name}/v1")).status_code == 404
            assert (await alice.patch(f"/api/v1/assets/{name}/v1/tags", json={})).status_code == 404
        too_many = await alice.patch(
            "/api/v1/assets/input/v1/tags",
            json={
                "add": [f"tag{i}" for i in range(33)],
            },
        )
        assert too_many.status_code == 422
        tagged = await alice.patch("/api/v1/assets/input/v1/tags", json={"add": ["Project:Test"]})
        assert tagged.status_code == 200
        assert tagged.json()["tags"] == ["project:test"]
        assert tagged.headers["cache-control"] == "no-store"
        assert (await alice.delete("/api/v1/assets/input/v1")).status_code == 200
        assert (await alice.get("/api/v1/assets/input/v1/files/0")).status_code == 404
        # Already accepted consumers remain usable after their catalog ref disappears.
        resolved = AssetResolver(
            LocalStorage(tmp_path / "global"), cache_root=tmp_path / "new-cache"
        ).resolve(pin)
        assert resolved.local_path.read_bytes() == local.read_bytes()

    async with AsyncClient(
        "http://test",
        token="test-operator",  # noqa: S106 - synthetic credential
        transport=transport,
    ) as operator:
        updated = await operator.update_asset_tags(
            "input", "v1", add=("reviewed",), visibility="shared"
        )
        assert updated.owner_username == "alice"
        assert updated.tags == ("reviewed",)
        assert not (await operator.delete_asset("input", "v1", visibility="shared")).already_deleted
