from __future__ import annotations

import base64
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from topodrome.config import SchedulerSettings
from topodrome.db.models import JobRow, KernelSessionRow
from topodrome.db.session import Database
from topodrome.domain.enums import JobState, KernelState
from topodrome.domain.specs import utc_now
from topodrome.scheduler.app import create_app

pytestmark = pytest.mark.component


async def _grant(
    client: httpx.AsyncClient,
    username: str,
    *,
    operator: dict[str, str],
) -> dict[str, str]:
    response = await client.post(
        f"/api/v1/access/users/{username}",
        headers=operator,
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


async def _source_ref(client: httpx.AsyncClient, headers: dict[str, str]) -> str:
    response = await client.post(
        "/api/v1/sources",
        headers=headers,
        json={
            "files": {
                "experiment.py": base64.b64encode(
                    b"def run(context):\n    return {'ok': True}\n"
                ).decode()
            }
        },
    )
    assert response.status_code == 201
    return str(response.json()["source_ref"])


@pytest.mark.asyncio
async def test_job_idempotency_and_direct_resources_are_owner_scoped(
    database: Database,
    tmp_path: Path,
) -> None:
    credential_root = tmp_path / "credentials"
    credential_root.mkdir(mode=0o700)
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        api_token="operator-secret",  # noqa: S106 - synthetic credential
        fence_signing_key="f" * 32,
        user_access={
            "enabled": True,
            "credential_store_path": credential_root / "users.json",
            "require_local_account": False,
        },
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    operator = {"Authorization": "Bearer operator-secret"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        alice = await _grant(client, "alice", operator=operator)
        bob = await _grant(client, "bob", operator=operator)
        source_ref = await _source_ref(client, alice)
        spec = {
            "entrypoint": "experiment.py",
            "source_ref": source_ref,
            "model": "model",
            "resources": {"gpu": 1, "min_vram_gb": 1},
        }
        idempotency = {"Idempotency-Key": "same-client-key"}
        alice_first = await client.post(
            "/api/v1/jobs",
            headers={**alice, **idempotency},
            json=spec,
        )
        alice_replay = await client.post(
            "/api/v1/jobs",
            headers={**alice, **idempotency},
            json=spec,
        )
        bob_job = await client.post(
            "/api/v1/jobs",
            headers={**bob, **idempotency},
            json=spec,
        )
        assert alice_first.status_code == alice_replay.status_code == 202
        assert bob_job.status_code == 202
        alice_id = alice_first.json()["job_id"]
        bob_id = bob_job.json()["job_id"]
        assert alice_replay.json()["job_id"] == alice_id
        assert bob_id != alice_id
        assert alice_first.json()["owner_username"] == "alice"
        assert bob_job.json()["owner_username"] == "bob"

        alice_status = await client.get("/api/v1/status", headers=alice)
        bob_status = await client.get("/api/v1/status", headers=bob)
        operator_status = await client.get("/api/v1/status", headers=operator)
        assert [row["job_id"] for row in alice_status.json()["jobs"]] == [alice_id]
        assert [row["job_id"] for row in bob_status.json()["jobs"]] == [bob_id]
        assert {row["job_id"] for row in operator_status.json()["jobs"]} == {
            alice_id,
            bob_id,
        }

        conflict = await client.post(
            "/api/v1/jobs",
            headers={**alice, **idempotency},
            json={**spec, "kwargs": {"changed": True}},
        )
        assert conflict.status_code == 409

        # A denied cancel must not even perturb scheduler-local placement
        # hints for another owner's job.

        foreign_requests = (
            ("GET", f"/api/v1/jobs/{alice_id}"),
            ("GET", f"/api/v1/jobs/{alice_id}/attempts"),
            ("GET", f"/api/v1/jobs/{alice_id}/logs"),
            ("GET", f"/api/v1/jobs/{alice_id}/attempts/guessed/logs"),
            ("POST", f"/api/v1/jobs/{alice_id}/cancel"),
        )
        for method, path in foreign_requests:
            response = await client.request(method, path, headers=bob)
            assert response.status_code == 404, (method, path, response.text)

        own = await client.get(f"/api/v1/jobs/{alice_id}", headers=alice)
        operator_view = await client.get(
            f"/api/v1/jobs/{alice_id}",
            headers=operator,
        )
        assert own.status_code == operator_view.status_code == 200
        assert own.json()["owner_username"] == "alice"
        assert (
            await client.get(f"/api/v1/jobs/{bob_id}", headers=bob)
        ).status_code == 200

    async with database.session() as session:
        alice_row = await session.get(JobRow, alice_id)
        bob_row = await session.get(JobRow, bob_id)
        assert alice_row is not None and bob_row is not None
        assert alice_row.state is JobState.QUEUED
        assert alice_row.owner_username == "alice"
        assert bob_row.owner_username == "bob"
        assert alice_row.idempotency_key != bob_row.idempotency_key
        assert alice_row.idempotency_key != "same-client-key"


@pytest.mark.asyncio
async def test_kernel_control_and_ide_credentials_are_owner_scoped(
    database: Database,
    tmp_path: Path,
) -> None:
    credential_root = tmp_path / "credentials"
    credential_root.mkdir(mode=0o700)
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        api_token="operator-secret",  # noqa: S106 - synthetic credential
        fence_signing_key="f" * 32,
        user_access={
            "enabled": True,
            "credential_store_path": credential_root / "users.json",
            "require_local_account": False,
        },
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    operator = {"Authorization": "Bearer operator-secret"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        alice = await _grant(client, "alice", operator=operator)
        bob = await _grant(client, "bob", operator=operator)
        created = await client.post(
            "/api/v1/kernels",
            headers=alice,
            json={"model": "model", "resources": {"gpu": 1}},
        )
        assert created.status_code == 202
        assert created.json()["owner_username"] == "alice"
        session_id = created.json()["session_id"]

        alice_status = await client.get("/api/v1/status", headers=alice)
        bob_status = await client.get("/api/v1/status", headers=bob)
        operator_status = await client.get("/api/v1/status", headers=operator)
        assert [row["session_id"] for row in alice_status.json()["kernels"]] == [
            session_id
        ]
        assert bob_status.json()["kernels"] == []
        assert [row["session_id"] for row in operator_status.json()["kernels"]] == [
            session_id
        ]

        foreign_requests = (
            ("GET", f"/api/v1/kernels/{session_id}"),
            ("DELETE", f"/api/v1/kernels/{session_id}"),
            ("GET", f"/api/v1/kernels/{session_id}/connect"),
            ("POST", f"/api/v1/kernels/{session_id}/notebook-server"),
            ("GET", f"/api/v1/kernels/{session_id}/notebook-server"),
            ("DELETE", f"/api/v1/kernels/{session_id}/notebook-server"),
        )
        for method, path in foreign_requests:
            response = await client.request(method, path, headers=bob)
            assert response.status_code == 404, (method, path, response.text)

        assert (
            await client.get(f"/api/v1/kernels/{session_id}", headers=alice)
        ).status_code == 200
        operator_view = await client.get(
            f"/api/v1/kernels/{session_id}",
            headers=operator,
        )
        assert operator_view.status_code == 200
        assert operator_view.json()["owner_username"] == "alice"

        # Make the synthetic session connectable without dispatching work. The
        # authorization assertion is specifically that only its owner can mint
        # the short-lived relay bearer.
        async with database.transaction() as session:
            row = await session.get(KernelSessionRow, session_id)
            assert row is not None
            row.state = KernelState.RUNNING
            row.expires_at = utc_now() + timedelta(minutes=10)
        connect = await client.get(
            f"/api/v1/kernels/{session_id}/connect",
            headers=alice,
        )
        assert connect.status_code == 200
        assert "token=" in connect.json()["url"]

        stopped = await client.delete(
            f"/api/v1/kernels/{session_id}",
            headers=alice,
        )
        assert stopped.status_code == 200
        assert stopped.json()["state"] == "stopped"


@pytest.mark.asyncio
async def test_rotation_and_revocation_invalidate_capabilities_without_stopping_kernel(
    database: Database,
    tmp_path: Path,
) -> None:
    credential_root = tmp_path / "credentials"
    credential_root.mkdir(mode=0o700)
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        public_url="http://127.0.0.1:8042",
        api_token="operator-secret",  # noqa: S106 - synthetic credential
        fence_signing_key="f" * 32,
        user_access={
            "enabled": True,
            "credential_store_path": credential_root / "users.json",
            "require_local_account": False,
        },
    )
    app = create_app(settings, database=database, run_dispatcher=False)

    class _ManagedController:
        def __init__(self) -> None:
            self.revoked: list[str] = []

        async def revoke_owner(self, username: str) -> int:
            self.revoked.append(username)
            return 1

        def verify_bridge_credential(self, _session_id: str, _credential: str) -> bool:
            return False

    managed = _ManagedController()
    app.state.service.managed_jupyter_controller = managed
    operator = {"Authorization": "Bearer operator-secret"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        alice = await _grant(client, "alice", operator=operator)
        created = await client.post(
            "/api/v1/kernels",
            headers=alice,
            json={"model": "model", "resources": {"gpu": 1}},
        )
        session_id = created.json()["session_id"]
        async with database.transaction() as session:
            row = await session.get(KernelSessionRow, session_id)
            assert row is not None
            row.state = KernelState.RUNNING
            row.relay_id = "relay-alice"
            row.expires_at = utc_now() + timedelta(minutes=10)

        first_connect = await client.get(
            f"/api/v1/kernels/{session_id}/connect",
            headers=alice,
        )
        assert first_connect.status_code == 200
        first_access = first_connect.json()["url"].split("token=", 1)[1]

        rotated = await client.post(
            "/api/v1/access/users/alice",
            headers=operator,
        )
        assert rotated.status_code == 200
        rotated_headers = {
            "Authorization": f"Bearer {rotated.json()['token']}"
        }
        assert (await client.get("/api/v1/status", headers=alice)).status_code == 401
        assert (
            await client.get("/api/v1/status", headers=rotated_headers)
        ).status_code == 200
        stale_after_rotation = await client.get(
            f"/api/v1/kernel-relay/{session_id}",
            params={"token": first_access},
        )
        assert stale_after_rotation.status_code == 403

        second_connect = await client.get(
            f"/api/v1/kernels/{session_id}/connect",
            headers=rotated_headers,
        )
        assert second_connect.status_code == 200
        second_access = second_connect.json()["url"].split("token=", 1)[1]
        revoked = await client.delete(
            "/api/v1/access/users/alice",
            headers=operator,
        )
        assert revoked.json() == {"username": "alice", "revoked": True}
        assert (
            await client.get("/api/v1/status", headers=rotated_headers)
        ).status_code == 401
        stale_after_revoke = await client.get(
            f"/api/v1/kernel-relay/{session_id}",
            params={"token": second_access},
        )
        assert stale_after_revoke.status_code == 403

    assert managed.revoked == ["alice", "alice", "alice"]
    async with database.session() as session:
        row = await session.get(KernelSessionRow, session_id)
        assert row is not None
        assert row.state is KernelState.RUNNING
