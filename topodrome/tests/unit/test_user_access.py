from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from topodrome.client.async_client import AsyncClient
from topodrome.config import SchedulerSettings
from topodrome.db.session import Database
from topodrome.protocol.user_access import UserCredentialStore
from topodrome.scheduler.app import create_app

pytestmark = pytest.mark.unit


def test_grant_authenticate_rotate_and_revoke(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    store = UserCredentialStore(root / "users.json", require_local_account=False)

    first = store.grant("alice")
    assert first.username == "alice"
    assert store.list_users() == ("alice",)
    assert store.is_whitelisted("alice")
    assert not store.is_whitelisted("../alice")
    first_revision = store.authorization_revision("alice")
    assert first_revision is not None
    assert store.authenticate(first.token).username == "alice"  # type: ignore[union-attr]
    assert first.token not in store.path.read_text(encoding="utf-8")

    rotated = store.grant("alice")
    assert rotated.token != first.token
    assert store.authorization_revision("alice") != first_revision
    assert store.authenticate(first.token) is None
    assert store.authenticate(rotated.token).username == "alice"  # type: ignore[union-attr]

    assert store.revoke("alice")
    assert not store.is_whitelisted("alice")
    assert not store.revoke("alice")
    assert store.authenticate(rotated.token) is None


def test_store_rejects_invalid_users_and_unsafe_storage(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    store = UserCredentialStore(private / "users.json", require_local_account=False)
    for username in ("", "../root", "two users", ".hidden"):
        with pytest.raises(ValueError, match="username"):
            store.grant(username)

    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="directory"):
        UserCredentialStore(public / "users.json", require_local_account=False).grant("alice")


def test_store_rejects_symlink_and_malformed_or_overpermissive_file(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    target = private / "target.json"
    target.write_text(
        json.dumps({"schema_version": 1, "users": {}}),
        encoding="utf-8",
    )
    target.chmod(0o600)
    linked = private / "users.json"
    linked.symlink_to(target)
    with pytest.raises(OSError):
        UserCredentialStore(linked, require_local_account=False).list_users()

    linked.unlink()
    linked.write_text("not-json", encoding="utf-8")
    linked.chmod(0o600)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        UserCredentialStore(linked, require_local_account=False).list_users()

    linked.write_text(
        json.dumps({"schema_version": 1, "users": {}}),
        encoding="utf-8",
    )
    os.chmod(linked, 0o644)
    with pytest.raises(PermissionError, match="owner-only"):
        UserCredentialStore(linked, require_local_account=False).list_users()


def test_grant_requires_real_unix_account_by_default(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    store = UserCredentialStore(private / "users.json")
    with pytest.raises(ValueError, match="existing local account"):
        store.grant("topodrome_user_that_does_not_exist")


@pytest.mark.asyncio
async def test_whitelisted_user_auth_and_operator_only_administration(
    database: Database,
    tmp_path: Path,
) -> None:
    credential_root = tmp_path / "credentials"
    credential_root.mkdir(mode=0o700)
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        api_token="operator-secret",  # noqa: S106 - synthetic test credential
        fence_signing_key="f" * 32,
        user_access={
            "enabled": True,
            "credential_store_path": credential_root / "users.json",
            "require_local_account": False,
        },
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/api/v1/gpus")).status_code == 401
        operator = {"Authorization": "Bearer operator-secret"}
        granted = await client.post("/api/v1/access/users/alice", headers=operator)
        assert granted.status_code == 200
        assert granted.headers["cache-control"] == "no-store"
        token = granted.json()["token"]
        assert token and token != "operator-secret"  # noqa: S105 - synthetic canary

        user = {"Authorization": f"Bearer {token}"}
        assert (await client.get("/api/v1/gpus", headers=user)).status_code == 200
        assert (await client.get("/api/v1/access/users", headers=user)).status_code == 403
        assert (await client.post("/api/v1/policy/reload", headers=user)).status_code == 403
        assert (
            await client.get("/api/v1/providers/autodl/control", headers=user)
        ).status_code == 403
        assert (
            await client.post(
                "/api/v1/providers/autodl/registrations/autodl-manual-" + "a" * 24 + "/release",
                headers=user,
                json={"force": True},
            )
        ).status_code == 403
        assert (
            await client.post(
                "/api/v1/gpu-reservations",
                headers=user,
                json={
                    "site_id": "lab1",
                    "gpu_index": 0,
                    "duration_seconds": 60,
                    "reason": "must be rejected before target lookup",
                    "force": True,
                },
            )
        ).status_code == 403
        assert (await client.get("/api/v1/gpu-reservations", headers=user)).status_code == 403
        assert (
            await client.post(
                "/api/v1/gpu-reservations/missing/extend",
                headers=user,
                json={"expected_expires_at": "2026-09-28T12:00:00+00:00", "additional_seconds": 60},
            )
        ).status_code == 403
        assert (
            await client.post(
                "/api/v1/gpu-reservations/missing/cancel",
                headers=user,
            )
        ).status_code == 403
        assert (
            await client.get("/api/v1/providers/autodl/instances", headers=user)
        ).status_code == 403

        users = await client.get("/api/v1/access/users", headers=operator)
        assert users.json() == {"users": ["alice"]}
        revoked = await client.delete("/api/v1/access/users/alice", headers=operator)
        assert revoked.json() == {"username": "alice", "revoked": True}
        assert (await client.get("/api/v1/gpus", headers=user)).status_code == 401

    async with AsyncClient(
        "http://test",
        token="operator-secret",  # noqa: S106 - synthetic test credential
        transport=httpx.ASGITransport(app=app),
    ) as sdk:
        granted = await sdk.grant_access_user("bob")
        assert granted["username"] == "bob"
        assert await sdk.access_users() == ["bob"]
        assert await sdk.revoke_access_user("bob") == {
            "username": "bob",
            "revoked": True,
        }
