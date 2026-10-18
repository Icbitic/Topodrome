from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta

import httpx
import pytest
import yaml
from pydantic import ValidationError

from tests.unit.test_placement import capability, site, spec
from tests.unit.test_policy_reload import _write_models, _write_scheduler
from topodrome.config import ModelCatalog, SchedulerSettings, SchedulingPolicySettings
from topodrome.db.models import AttemptRow, JobRow, LeaseRow, SiteRow
from topodrome.domain.enums import (
    ActorState,
    AttemptState,
    JobState,
    LeaseState,
    SiteKind,
    SiteState,
)
from topodrome.domain.errors import DurabilityError
from topodrome.domain.scheduling import SitePreferenceRequest
from topodrome.domain.specs import JobSpec, utc_now
from topodrome.scheduler import policy as policy_module
from topodrome.scheduler.app import create_app
from topodrome.scheduler.placement import choose_placement
from topodrome.scheduler.service import SchedulerService

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "order", [("x", "x"), ("../x",), ("",), ("x\n",), tuple(map(str, range(65)))]
)
def test_site_order_validation(order):
    with pytest.raises(ValidationError):
        SitePreferenceRequest(preferred_site_ids=order, expected_generation=1)
    with pytest.raises(ValidationError):
        SchedulingPolicySettings(preferred_site_ids=order)


def choose(sites, **kwargs):
    return choose_placement(
        spec(),
        sites,
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        preferred_site_ids=("sqz-128",),
        prefer_autodl_over_foreign_processes=True,
        **kwargs,
    )


def test_preference_beats_readiness_cost_but_empty_order_remains_automatic():
    candidates = [site("lab1", free_gib=70), site("sqz-128", free_gib=70)]
    costs = {"lab1": 1.0, "sqz-128": 120.0}
    assert choose(candidates, remaining_seconds_by_site=costs).site_id == "sqz-128"
    assert (
        choose_placement(
            spec(),
            candidates,
            heartbeat_cutoff=utc_now() - timedelta(seconds=30),
            remaining_seconds_by_site=costs,
        ).site_id
        == "lab1"
    )


@pytest.mark.parametrize(
    "blocked", ["offline", "stale", "foreign", "reserved", "lease", "vram", "model", "cpu"]
)
def test_preferred_site_never_bypasses_eligibility_and_falls_back(blocked):
    preferred = site("sqz-128", free_gib=70)
    if blocked == "offline":
        preferred = replace(preferred, state=SiteState.OFFLINE)
    elif blocked == "stale":
        preferred = replace(preferred, heartbeat_at=utc_now() - timedelta(minutes=5))
    elif blocked == "foreign":
        preferred = replace(preferred, gpus=(replace(preferred.gpus[0], foreign_process_count=1),))
    elif blocked == "reserved":
        preferred = replace(preferred, gpus=(replace(preferred.gpus[0], externally_reserved=True),))
    elif blocked == "lease":
        preferred = replace(preferred, gpus=(replace(preferred.gpus[0], active_lease=True),))
    elif blocked == "vram":
        preferred = site("sqz-128", free_gib=1)
    elif blocked == "model":
        preferred = replace(preferred, models=(capability(repo_id="test/different"),))
    elif blocked == "cpu":
        preferred = replace(preferred, available_cpu=0)
    assert choose([preferred, site("lab1", free_gib=70)]).site_id == "lab1"


def test_site_order_cannot_promote_paid_or_foreign_sharing_above_clean_institutional():
    for preferred in (
        site("sqz-128", free_gib=70, paid=True),
        site(
            "sqz-128",
            free_gib=70,
            foreign_process_count=1,
        ),
    ):
        assert (
            choose(
                [preferred, site("lab1", free_gib=70)],
                foreign_capacity_sharing_site_ids=("sqz-128",),
            ).site_id
            == "lab1"
        )


@pytest.mark.parametrize("busy", [False, True])
def test_preference_can_replace_idle_model_but_never_active_actor(busy):
    key = {
        "repo_id": "test/another-model",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "gpu_uuid": "sqz-128-gpu",
        "replica_id": "old-model",
        "replica_gpu_uuids": ["sqz-128-gpu"],
    }
    preferred = site("sqz-128", free_gib=2, actor_key=key)
    if busy:
        preferred = replace(
            preferred,
            gpus=(
                replace(
                    preferred.gpus[0],
                    actor_state=ActorState.BUSY,
                ),
            ),
        )
    chosen = choose([preferred, site("lab1", free_gib=70)], allow_idle_actor_replacement=True)
    assert chosen.site_id == ("lab1" if busy else "sqz-128")


async def make_app(database, tmp_path, **options):
    model = tmp_path / "models.yaml"
    path = tmp_path / "scheduler.yaml"
    _write_models(model, startup_seconds=2)
    _write_scheduler(path, root=tmp_path / "global", model_config=model, queue_delay=30)
    settings_data = SchedulerSettings.from_yaml(path).model_dump(by_alias=True)
    settings_data.update(options)
    settings = SchedulerSettings.model_validate(settings_data)
    app = create_app(
        settings,
        database=database,
        run_dispatcher=False,
        settings_path=path,
    )
    async with database.transaction() as session:
        for site_id, state in (
            ("lab1", SiteState.READY),
            ("sqz-128", SiteState.READY),
            ("offline", SiteState.OFFLINE),
            ("retired", SiteState.RETIRED),
        ):
            session.add(SiteRow(id=site_id, display_name=site_id, kind=SiteKind.LAB, state=state))
    return app, path


def request(order=("sqz-128",), generation=1):
    return SitePreferenceRequest(preferred_site_ids=order, expected_generation=generation)


async def test_live_change_keeps_active_work_persists_and_survives_unrelated_reload(
    database, tmp_path
):
    app, path = await make_app(database, tmp_path)
    service = app.state.service
    async with database.transaction() as session:
        session.add(
            JobRow(
                id="job-live",
                state=JobState.RUNNING,
                spec=JobSpec(entrypoint="test.py", model="test/model").model_dump(mode="json"),
            )
        )
        session.add(
            AttemptRow(
                id="attempt-live",
                job_id="job-live",
                number=1,
                state=AttemptState.RUNNING,
                site_id="lab1",
            )
        )
        session.add(
            LeaseRow(
                id="lease-live",
                attempt_id="attempt-live",
                site_id="lab1",
                state=LeaseState.ACTIVE,
                fencing_token_digest="a" * 64,
                expires_at=utc_now() + timedelta(hours=1),
            )
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.put(
            "/api/v1/policy/site-preference", json=request().model_dump(mode="json")
        )
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["status"] == "active"
        assert response.json()["preferred_site_ids"] == ["sqz-128"]
        assert response.json()["generation"] == 2
        stale = await c.put(
            "/api/v1/policy/site-preference", json=request(("lab1",)).model_dump(mode="json")
        )
        assert stale.status_code == 409
        status = (await c.get("/api/v1/status")).json()["policy"]
        assert status["generation"] == 2 and status["preferred_site_ids"] == ["sqz-128"]
    async with database.session() as session:
        assert (await session.get(JobRow, "job-live")).state is JobState.RUNNING
        assert (await session.get(AttemptRow, "attempt-live")).state is AttemptState.RUNNING
        assert (await session.get(LeaseRow, "lease-live")).state is LeaseState.ACTIVE
    restored = SchedulerService(
        database=database,
        settings=SchedulerSettings.from_yaml(path),
        model_catalog=ModelCatalog.from_yaml(tmp_path / "models.yaml"),
        settings_path=path,
    )
    assert restored.scheduling_policy_snapshot().settings.preferred_site_ids == ("sqz-128",)
    assert (await service.reload_scheduling_policy())["generation"] == 2
    config = yaml.safe_load(path.read_text())
    config["scheduling"]["preferred_site_ids"] = []
    path.write_text(yaml.safe_dump(config))
    await service.reload_scheduling_policy()
    await service.dispatch_once()
    assert service.scheduling_policy_snapshot().settings.preferred_site_ids == ()


async def test_concurrent_edits_are_optimistic_and_offline_preferences_are_allowed(
    database, tmp_path
):
    app, _ = await make_app(database, tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        results = await asyncio.gather(
            *(
                c.put("/api/v1/policy/site-preference", json=request((s,)).model_dump(mode="json"))
                for s in ("lab1", "sqz-128")
            )
        )
        assert sorted(r.status_code for r in results) == [200, 409]
        for value in ("missing", "retired"):
            r = await c.put(
                "/api/v1/policy/site-preference", json=request((value,), 2).model_dump(mode="json")
            )
            assert r.status_code == 400
        r = await c.put(
            "/api/v1/policy/site-preference", json=request(("offline",), 2).model_dump(mode="json")
        )
        assert r.status_code == 200
        r = await c.put(
            "/api/v1/policy/site-preference", json=request((), 3).model_dump(mode="json")
        )
        assert r.status_code == 200 and r.json()["preferred_site_ids"] == []


@pytest.mark.parametrize("after_replace", [False, True])
async def test_preference_commit_failure_never_enables_unproven_order(
    database,
    tmp_path,
    monkeypatch,
    after_replace,
):
    app, _ = await make_app(database, tmp_path)
    service = app.state.service
    store = service._policy_store
    original = store.save
    fsync = policy_module._fsync_directory

    def fail(*args):
        raise OSError("synthetic failure")

    if after_replace:
        monkeypatch.setattr(policy_module, "_fsync_directory", fail)
    else:
        monkeypatch.setattr(store, "save", fail)
    with pytest.raises(DurabilityError):
        await service.set_site_preference(request())
    assert service.scheduling_policy_snapshot().generation == 1
    assert service.scheduling_policy_snapshot().settings.preferred_site_ids == ()
    assert (service._pending_policy_snapshot is not None) is after_replace
    monkeypatch.setattr(store, "save", original)
    monkeypatch.setattr(policy_module, "_fsync_directory", fsync)
    if after_replace:
        await service.dispatch_once()
        assert service.scheduling_policy_snapshot().generation == 2


async def test_site_preference_operator_authorization(database, tmp_path):
    credentials = tmp_path / "credentials"
    credentials.mkdir(mode=0o700)
    app, _ = await make_app(
        database,
        tmp_path,
        api_token="operator-test",  # noqa: S106 -- synthetic credential
        user_access={
            "enabled": True,
            "credential_store_path": credentials / "users.json",
            "require_local_account": False,
        },
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        body = request().model_dump(mode="json")
        assert (await c.put("/api/v1/policy/site-preference", json=body)).status_code == 401
        grant = await c.post(
            "/api/v1/access/users/alice", headers={"Authorization": "Bearer operator-test"}
        )
        assert grant.status_code == 200
        r = await c.put(
            "/api/v1/policy/site-preference",
            json=body,
            headers={"Authorization": "Bearer " + grant.json()["token"]},
        )
        assert r.status_code == 403
        assert app.state.service.scheduling_policy_snapshot().generation == 1


def test_preference_only_is_live_but_mixed_settings_require_drain():
    old = policy_module.initial_policy_snapshot(SchedulerSettings(fence_signing_key="f" * 64), None)
    new = replace(
        old, settings=old.settings.model_copy(update={"preferred_site_ids": ("sqz-128",)})
    )
    assert policy_module.is_live_priority_change(old, new)
    mixed = replace(new, settings=new.settings.model_copy(update={"starvation_seconds": 90}))
    assert not policy_module.is_live_priority_change(old, mixed)
