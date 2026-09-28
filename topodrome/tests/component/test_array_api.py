import httpx
import pytest

from tests.component.test_multi_user_isolation import _grant, _source_ref
from topodrome.client.async_client import AsyncClient
from topodrome.config import SchedulerSettings
from topodrome.scheduler.app import create_app

pytestmark = pytest.mark.component


async def test_array_api_sdk_owner_scope_pagination_and_controls(database, tmp_path):
    credentials = tmp_path / "credentials"
    credentials.mkdir(mode=0o700)
    settings = SchedulerSettings(
        global_root=tmp_path / "global",
        api_token="synthetic-operator",  # noqa: S106
        fence_signing_key="f" * 32,
        user_access={
            "enabled": True,
            "credential_store_path": credentials / "users.json",
            "require_local_account": False,
        },
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        operator = {"Authorization": "Bearer synthetic-operator"}
        alice = await _grant(raw, "alice", operator=operator)
        bob = await _grant(raw, "bob", operator=operator)
        source = await _source_ref(raw, alice)
        request = {
            "job": {
                "entrypoint": "experiment.py",
                "model": "model",
                "source_ref": source,
                "name": "Sweep",
                "kwargs": {"seed": 7},
                "priority": 10,
            },
            "elements": [{"index": 0}, {"index": 4, "kwargs": {"seed": 9}}],
            "max_parallel": 2,
        }
        async with AsyncClient(
            "http://test", token=alice["Authorization"].split()[1], transport=transport
        ) as client:
            first = await client.submit_array(request, idempotency_key="sweep-1")
            array_id = first["array_id"]
            repeat = await client.submit_array(request, idempotency_key="sweep-1")
            assert repeat["array_id"] == array_id
            assert len(await client.arrays()) == 1
            detail = await client.array(array_id, limit=1)
            assert detail["members"][0]["index"] == 0 and detail["next_after"] == 0
            rest = await client.array(array_id, after=detail["next_after"])
            assert rest["members"][0]["index"] == 4 and rest["next_after"] is None
            child = await client.status(rest["members"][0]["job_id"])
            assert child.array_id == array_id and child.array_index == 4
            assert child.spec.kwargs == {"seed": 9, "array_index": 4}
            assert child.spec.priority == 10
            assert child.timing["phase"] == "queued"
            assert (await client.update_array(array_id, held=True))["held"]
            resumed = await client.update_array(array_id, held=False, max_parallel=1)
            assert not resumed["held"] and resumed["max_parallel"] == 1
            await client.cancel_array(array_id)
            assert (await client.array(array_id))["counts"] == {"canceled": 2}
        for method, suffix, body in [
            ("GET", "", None),
            ("PATCH", "", {"held": True}),
            ("POST", "/cancel", None),
        ]:
            response = await raw.request(
                method, f"/api/v1/arrays/{array_id}{suffix}", headers=bob, json=body
            )
            assert response.status_code == 404
        assert (await raw.get("/api/v1/arrays", headers=bob)).json() == []
        frame = (await raw.get("/api/v1/status", headers=bob)).json()
        assert frame["arrays"] == [] and frame["jobs"] == []
        conflict = await raw.post(
            "/api/v1/arrays",
            headers={**alice, "Idempotency-Key": "sweep-1"},
            json={**request, "max_parallel": 3},
        )
        assert conflict.status_code == 409
        invalid = await raw.get(f"/api/v1/arrays/{array_id}?limit=1001", headers=alice)
        assert invalid.status_code == 422
        invalid = await raw.post(
            "/api/v1/arrays",
            headers=alice,
            json={**request, "elements": [{"index": 1}, {"index": 1}]},
        )
        assert invalid.status_code == 422
