from __future__ import annotations

import asyncio
import base64
import sys
from datetime import timedelta

import httpx
import pytest
import yaml
from sqlalchemy import select

from topodrome.config import SchedulerSettings
from topodrome.db.models import AttemptRow, LeaseGPUClaimRow, LeaseRow, SiteRow
from topodrome.domain.enums import AttemptState, JobState, LeaseState, SiteKind
from topodrome.domain.errors import ConflictError
from topodrome.domain.events import AgentEnvelope
from topodrome.domain.specs import GPUInventory, JobSpec, ModelCapability, SiteHeartbeat, utc_now
from topodrome.domain.upgrades import UpgradeClaim, UpgradePauseRequest, UpgradeRequest
from topodrome.scheduler.app import create_app
from topodrome.scheduler.control import AgentConnection
from topodrome.scheduler.service import SchedulerService
from topodrome.scheduler.source_store import SourceBundle

pytestmark = pytest.mark.component


async def heartbeat(service, site="sqz", agent_session="agent-1", sequence=1, **activity):
    async def send(_message):
        pass

    await service.connections.register(AgentConnection(site, agent_session, send))
    sample = SiteHeartbeat(
        site_id=site,
        site_kind=SiteKind.LAB,
        agent_session_id=agent_session,
        sequence=sequence,
        gpus=(
            GPUInventory(
                uuid=f"GPU-{site}",
                index=0,
                name="A800",
                total_memory_bytes=80 * 1024**3,
                free_memory_bytes=70 * 1024**3,
                utilization_percent=0,
            ),
        ),
        **activity,
    )
    await service.handle_agent_envelope(
        AgentEnvelope(
            site_id=site,
            agent_session_id=agent_session,
            sequence=sequence,
            kind="heartbeat",
            payload=sample.model_dump(mode="json"),
        )
    )


def service_for(database, tmp_path):
    return SchedulerService(
        database=database,
        settings=SchedulerSettings(
            database_dsn="sqlite+aiosqlite://",
            global_root=tmp_path / "global",
            fence_signing_key="idle-upgrade-fencing-test-00000000",
        ),
    )


async def queue(service, tmp_path, *, pause="targets", scheduler=False):
    path = tmp_path / "release.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "upgrade test",
                "sites": ["sqz"],
                "scheduler": scheduler,
                "apply": [{"argv": [sys.executable, "-c", "pass"]}],
                "verify": [{"argv": [sys.executable, "-c", "pass"]}],
            }
        )
    )
    path.chmod(0o600)
    summary = await service.upgrades.schedule(
        UpgradeRequest(plan_path=path, pause=pause, idempotency_key="test1")
    )
    service.upgrades.store.change(summary["upgrade_id"], expected={"queued"}, state="waiting")
    return summary["upgrade_id"]


@pytest.mark.parametrize(
    "flag",
    [
        "job_running",
        "assignment_queued_locally",
        "interactive_kernel_active",
        "artifact_flush_pending",
        "staging_operation_active",
    ],
)
async def test_free_gpu_is_not_enough_for_idle_upgrade(database, tmp_path, flag):
    service = service_for(database, tmp_path)
    await heartbeat(service, **{flag: True})
    upgrade = await queue(service, tmp_path)
    result = await service.upgrades.claim(upgrade, UpgradeClaim(runner_id="runner"))
    assert not result["claimed"]
    assert result["blockers"]["site_activity"] == 1
    assert service.upgrades.store.current().state == "waiting"


async def test_disconnected_or_stale_site_never_qualifies(database, tmp_path):
    service = service_for(database, tmp_path)
    upgrade = await queue(service, tmp_path)
    assert not (await service.upgrades.claim(upgrade, UpgradeClaim(runner_id="runner")))["claimed"]
    await heartbeat(service)
    async with database.transaction() as session:
        site = await session.get(SiteRow, "sqz")
        site.heartbeat_at = utc_now() - timedelta(hours=1)
    assert not (await service.upgrades.claim(upgrade, UpgradeClaim(runner_id="runner")))["claimed"]


async def test_claim_and_completion_require_new_fresh_agent(database, tmp_path):
    service = service_for(database, tmp_path)
    await heartbeat(service)
    upgrade = await queue(service, tmp_path, pause="none")
    assert not service.upgrades.store.blocked("sqz")
    claim = UpgradeClaim(runner_id="runner")
    assert (await service.upgrades.claim(upgrade, claim))["claimed"]
    assert service.upgrades.store.blocked("sqz")
    service.upgrades.store.change(upgrade, expected={"applying"}, state="verifying")
    assert not (await service.upgrades.complete(upgrade, claim))["completed"]
    await heartbeat(service, agent_session="agent-2")
    assert (await service.upgrades.complete(upgrade, claim))["completed"]
    assert not service.upgrades.store.blocked("sqz")
    assert (await service.upgrades.complete(upgrade, claim))["completed"]


async def test_live_lease_blocks_upgrade_without_any_rewrite(database, tmp_path):
    service = service_for(database, tmp_path)
    await heartbeat(service)
    async with database.transaction() as session:
        job = await service.repository.submit_job(
            session,
            JobSpec(
                entrypoint="experiment.py",
                model={"repo_id": "test/model", "revision": "rev"},
                resources={"gpu": 1, "min_vram_gb": 15},
            ),
        )
        locked = await service.repository.claim_next_job(session)
        offer = await service.repository.create_offer(
            session,
            job=locked,
            site_id="sqz",
            gpu_uuids=("GPU-sqz",),
            authority=service.fencing,
            offer_ttl=timedelta(seconds=180),
            token_ttl=timedelta(hours=3),
            lease_ttl=timedelta(seconds=300),
        )
    upgrade = await queue(service, tmp_path)
    assert not (await service.upgrades.claim(upgrade, UpgradeClaim(runner_id="runner")))["claimed"]
    async with database.session() as session:
        attempt = await session.get(AttemptRow, offer.attempt_id)
        lease = await session.scalar(select(LeaseRow).where(LeaseRow.attempt_id == attempt.id))
        assert attempt.job_id == job.id
        assert lease.state is LeaseState.OFFERED
        claims = list(
            await session.scalars(
                select(LeaseGPUClaimRow).where(
                    LeaseGPUClaimRow.lease_id == lease.id, LeaseGPUClaimRow.active.is_(True)
                )
            )
        )
        assert [claim.gpu_uuid for claim in claims] == ["GPU-sqz"]


async def test_cancel_only_before_apply_and_journal_survives_scheduler_restart(database, tmp_path):
    first = service_for(database, tmp_path)
    await heartbeat(first)
    upgrade = await queue(first, tmp_path)
    second = service_for(database, tmp_path)
    assert second.upgrades.store.blocked("sqz")
    assert not second.upgrades.store.blocked("lab1")
    assert (await second.upgrades.cancel(upgrade))["state"] == "canceled"
    assert not second.upgrades.store.blocked("sqz")
    # Requeue as a distinct operator intent after safe cancellation.
    new = await second.upgrades.schedule(
        UpgradeRequest(plan_path=tmp_path / "release.yaml", idempotency_key="test2")
    )
    second.upgrades.store.change(new["upgrade_id"], expected={"queued"}, state="waiting")
    await heartbeat(second, agent_session="agent-2")
    await second.upgrades.claim(new["upgrade_id"], UpgradeClaim(runner_id="runner"))
    with pytest.raises(ConflictError):
        await second.upgrades.cancel(new["upgrade_id"])


async def test_idle_claim_is_serialized_with_admission(database, tmp_path):
    service = service_for(database, tmp_path)
    await heartbeat(service)
    upgrade = await queue(service, tmp_path, pause="none")
    async with service._dispatch_reload_lock:
        task = asyncio.create_task(
            service.upgrades.claim(upgrade, UpgradeClaim(runner_id="runner"))
        )
        await asyncio.sleep(0)
        assert not task.done()
        assert not service.upgrades.store.blocked("sqz")
    assert (await task)["claimed"]
    assert service.upgrades.store.blocked("sqz")


async def test_pause_can_change_while_waiting_but_never_remove_live_fence(database, tmp_path):
    service = service_for(database, tmp_path)
    await heartbeat(service)
    upgrade = await queue(service, tmp_path, pause="none")
    changed = await service.upgrades.pause(
        upgrade, UpgradePauseRequest(pause="targets", expected_pause="none")
    )
    assert changed["pause"] == "targets"
    assert service.upgrades.store.blocked("sqz")
    with pytest.raises(ConflictError):
        await service.upgrades.pause(
            upgrade, UpgradePauseRequest(pause="all", expected_pause="none")
        )
    await service.upgrades.pause(
        upgrade, UpgradePauseRequest(pause="none", expected_pause="targets")
    )
    assert not service.upgrades.store.blocked("sqz")
    await service.upgrades.claim(upgrade, UpgradeClaim(runner_id="runner"))
    with pytest.raises(ConflictError):
        await service.upgrades.pause(
            upgrade, UpgradePauseRequest(pause="none", expected_pause="none")
        )
    assert service.upgrades.store.blocked("sqz")


@pytest.mark.usefixtures("pre_staged_inputs")
@pytest.mark.parametrize("pause,dispatches", [("targets", True), ("all", False)])
async def test_site_pause_keeps_unaffected_sites_schedulable(database, tmp_path, pause, dispatches):
    service = service_for(database, tmp_path)
    capability = (
        ModelCapability(
            repo_id="test/model",
            revision="rev",
            availability="cached",
            backend="transformers",
            backend_version="configured",
            dtype="bfloat16",
        ),
    )
    await heartbeat(service, models=capability)
    await heartbeat(service, site="lab1", models=capability)
    source = service.sources.put(
        SourceBundle(
            files={"experiment.py": base64.b64encode(b"def run(context): pass\n").decode()}
        )
    ).source_ref
    async with database.transaction() as session:
        job = await service.repository.submit_job(
            session,
            JobSpec(
                entrypoint="experiment.py",
                source_ref=source,
                model={"repo_id": "test/model", "revision": "rev"},
                resources={"gpu": 1, "min_vram_gb": 15},
            ),
        )
    upgrade = await queue(service, tmp_path, pause=pause)
    assert await service.dispatch_once() == dispatches
    if dispatches:
        attempts = await service.attempts(job.id)
        assert len(attempts) == 1
        assert attempts[0]["site_id"] == "lab1"
        # Unaffected-site work does not block this site's safe upgrade.
        assert (await service.upgrades.claim(upgrade, UpgradeClaim(runner_id="runner")))["claimed"]


async def test_old_idle_sample_cannot_hide_new_failure_log_outbox(database, tmp_path):
    service = service_for(database, tmp_path)
    await heartbeat(service)
    async with database.transaction() as session:
        job = await service.repository.submit_job(
            session,
            JobSpec(
                entrypoint="experiment.py",
                model={"repo_id": "test/model", "revision": "rev"},
                resources={"gpu": 1, "min_vram_gb": 15},
            ),
        )
        locked = await service.repository.claim_next_job(session)
        offer = await service.repository.create_offer(
            session,
            job=locked,
            site_id="sqz",
            gpu_uuids=("GPU-sqz",),
            authority=service.fencing,
            offer_ttl=timedelta(seconds=180),
            token_ttl=timedelta(hours=3),
            lease_ttl=timedelta(seconds=300),
        )
        attempt = await session.get(AttemptRow, offer.attempt_id)
        lease = await session.scalar(select(LeaseRow).where(LeaseRow.attempt_id == attempt.id))
        attempt.state = AttemptState.FAILED
        job.state = JobState.FAILED
        lease.state = LeaseState.RELEASED
        claims = list(
            await session.scalars(
                select(LeaseGPUClaimRow).where(LeaseGPUClaimRow.lease_id == lease.id)
            )
        )
        for claim in claims:
            claim.active = False
    upgrade = await queue(service, tmp_path)
    result = await service.upgrades.claim(upgrade, UpgradeClaim(runner_id="runner"))
    assert not result["claimed"]
    assert result["blockers"]["heartbeat_before_last_work"] == 1
    await heartbeat(service, sequence=2, artifact_flush_pending=True)
    assert not (await service.upgrades.claim(upgrade, UpgradeClaim(runner_id="runner")))["claimed"]
    await heartbeat(service, sequence=3)
    assert (await service.upgrades.claim(upgrade, UpgradeClaim(runner_id="runner")))["claimed"]


async def test_scheduler_completion_waits_for_all_previously_connected_agents(database, tmp_path):
    first = service_for(database, tmp_path)
    await heartbeat(first)
    await heartbeat(first, site="lab1")
    upgrade = await queue(first, tmp_path, scheduler=True)
    request = UpgradeClaim(runner_id="runner")
    assert (await first.upgrades.claim(upgrade, request))["claimed"]
    first.upgrades.store.change(upgrade, expected={"applying"}, state="verifying")
    second = service_for(database, tmp_path)
    await heartbeat(second, agent_session="agent-2")
    assert not (await second.upgrades.complete(upgrade, request))["completed"]
    await heartbeat(second, site="lab1", agent_session="agent-2")
    assert (await second.upgrades.complete(upgrade, request))["completed"]


async def test_operator_upgrade_api_and_status_are_sanitized(database, tmp_path):
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path / "global",
        fence_signing_key="f" * 32,
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.ASGITransport(app)
    ) as client:
        path = tmp_path / "release.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": "API release",
                    "scheduler": True,
                    "apply": [{"argv": [sys.executable, "-c", "SYNTHETIC_SECRET"]}],
                    "verify": [{"argv": [sys.executable, "-c", "pass"]}],
                }
            )
        )
        path.chmod(0o600)
        response = await client.post(
            "/api/v1/upgrades",
            json={"plan_path": str(path), "idempotency_key": "api1", "pause": "all"},
        )
        assert response.status_code == 201
        result = response.json()
        listed = await client.get("/api/v1/upgrades")
        assert listed.status_code == 200
        assert "SYNTHETIC_SECRET" not in listed.text
        assert "argv" not in listed.text
        canceled = await client.post(f"/api/v1/upgrades/{result['upgrade_id']}/cancel")
        assert canceled.json()["state"] == "canceled"
