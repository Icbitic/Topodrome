from __future__ import annotations

import asyncio
import base64
import hashlib
import stat
import threading
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from topodrome.config import ModelCatalog, SchedulerSettings
from topodrome.db.models import (
    AttemptRow,
    JobRow,
    KernelSessionRow,
    LeaseRow,
    ProviderInstanceRow,
    SiteRow,
)
from topodrome.db.repositories import SchedulerRepository
from topodrome.db.session import Database
from topodrome.domain.enums import (
    AttemptState,
    JobState,
    KernelMode,
    KernelState,
    LeaseState,
    ProviderInstanceState,
    SiteKind,
    SiteState,
)
from topodrome.domain.errors import DurabilityError
from topodrome.domain.specs import JobSpec, utc_now
from topodrome.scheduler import policy as policy_module
from topodrome.scheduler.app import create_app
from topodrome.scheduler.placement import SiteCandidate
from topodrome.scheduler.service import SchedulerService


@pytest.mark.parametrize("extra_change", [None, "settings", "startup"])
def test_live_priority_exception_rejects_mixed_changes(extra_change: str | None) -> None:
    original = policy_module.initial_policy_snapshot(
        SchedulerSettings(fence_signing_key="f" * 64), None
    )
    changes: dict[str, object] = {"prefer_autodl_over_foreign_processes": True}
    if extra_change == "settings":
        changes["queue_delay_seconds_per_job"] = original.settings.queue_delay_seconds_per_job + 1
    candidate = replace(original, settings=original.settings.model_copy(update=changes))
    if extra_change == "startup":
        candidate = replace(candidate, startup_seconds_by_model={"model": {"lab1": 2.0}})
    assert policy_module.is_live_priority_change(original, candidate) == (
        extra_change != "settings"
    )
    assert not policy_module.is_live_priority_change(original, original)


@pytest.mark.parametrize(
    "sites,live", [((), True), (("lab1",), True), (("lab1", "new-site"), False)]
)
@pytest.mark.parametrize("mixed", [False, True])
def test_live_sharing_restriction(sites: tuple[str, ...], live: bool, mixed: bool) -> None:
    original = policy_module.initial_policy_snapshot(
        SchedulerSettings(fence_signing_key="f" * 64), None
    )
    original = replace(
        original,
        settings=original.settings.model_copy(
            update={
                "foreign_capacity_sharing_site_ids": ("lab1", "remote-jupyter-1"),
            }
        ),
    )
    changes = {"foreign_capacity_sharing_site_ids": sites}
    if mixed:
        changes["queue_delay_seconds_per_job"] = 999
    candidate = replace(original, settings=original.settings.model_copy(update=changes))
    assert policy_module.is_live_priority_change(original, candidate) == (live and not mixed)


class _BlockingInventoryRepository(SchedulerRepository):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def placement_inventory(self, session: AsyncSession) -> list[SiteCandidate]:
        del session
        self.entered.set()
        await self.release.wait()
        return []


def _write_scheduler(
    path: Path,
    *,
    root: Path,
    model_config: Path,
    queue_delay: float,
    lease_ttl: int = 90,
    light_utilization: float = 10,
    source_limits: tuple[int, int] | None = None,
) -> None:
    scheduling: dict[str, object] = {
        "queue_delay_seconds_per_job": queue_delay,
        "light_contention_utilization_percent": light_utilization,
    }
    if source_limits is not None:
        scheduling["source_limits"] = {
            "max_file_bytes": source_limits[0],
            "max_total_bytes": source_limits[1],
        }
    path.write_text(
        yaml.safe_dump(
            {
                "database_dsn": "sqlite+aiosqlite://",
                "global_root": str(root),
                "fence_signing_key": "f" * 64,
                "lease_ttl_seconds": lease_ttl,
                "model_config": str(model_config),
                "scheduling": scheduling,
            }
        ),
        encoding="utf-8",
    )


def _write_models(path: Path, *, startup_seconds: float, model_path: str = "/models/model") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "model": {
                        "revision": "rev",
                        "backend": "transformers",
                        "backend_version": "configured",
                        "dtype": "bfloat16",
                        "min_vram_gb": 15,
                        "sites": {
                            "lab1": {
                                "path": model_path,
                                "estimated_startup_seconds": startup_seconds,
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )


async def test_policy_reload_is_durably_staged_idempotent_and_visible_in_status(
    database: Database,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models.yaml"
    settings_path = tmp_path / "scheduler.yaml"
    _write_models(model_path, startup_seconds=2)
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=30,
    )
    settings = SchedulerSettings.from_yaml(settings_path)
    app = create_app(
        settings,
        database=database,
        run_dispatcher=False,
        settings_path=settings_path,
    )
    service = app.state.service
    original = service.scheduling_policy_snapshot()
    assert original.generation == 1
    assert original.startup_seconds("model") == {"lab1": 2.0}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        initial_status = (await client.get("/api/v1/status")).json()["policy"]
        assert initial_status["status"] == "active"
        assert initial_status["generation"] == 1

        _write_models(model_path, startup_seconds=120)
        _write_scheduler(
            settings_path,
            root=tmp_path / "global",
            model_config=model_path,
            queue_delay=12.5,
        )
        response = await client.post("/api/v1/policy/reload")
        assert response.status_code == 200
        staged = response.json()
        assert staged["status"] == "draining"
        assert staged["generation"] == 1
        assert staged["pending_generation"] == 2
        assert staged["blocker_count"] == 0
        assert response.headers["cache-control"] == "no-store"

        # Repeating the same candidate is idempotent while drain is pending.
        repeated = (await client.post("/api/v1/policy/reload")).json()
        assert repeated == staged
        assert service.scheduling_policy_snapshot() is original

        status = (await client.get("/api/v1/status")).json()["policy"]
        assert status == staged

        # A competing staged candidate is rejected instead of replacing the
        # generation whose durable drain is already in progress.
        _write_scheduler(
            settings_path,
            root=tmp_path / "global",
            model_config=model_path,
            queue_delay=13,
        )
        conflict = await client.post("/api/v1/policy/reload")
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "conflict"

        # The dispatcher owns activation. With an empty barrier this happens
        # on the next tick, before that same tick considers queued work.
        _write_scheduler(
            settings_path,
            root=tmp_path / "global",
            model_config=model_path,
            queue_delay=12.5,
        )
        assert not await service.dispatch_once()
        current = service.scheduling_policy_snapshot()
        assert current.generation == 2
        assert current.settings.queue_delay_seconds_per_job == 12.5
        assert current.startup_seconds("model") == {"lab1": 120.0}
        assert original.settings.queue_delay_seconds_per_job == 30
        assert original.startup_seconds("model") == {"lab1": 2.0}
        # The immutable model catalog used to resolve job semantics is not
        # replaced; only its copied startup estimate enters the new policy.
        assert service.model_catalog.models["model"].sites["lab1"].estimated_startup_seconds == 2
        status = (await client.get("/api/v1/status")).json()["policy"]
        assert status["status"] == "active"
        assert status["generation"] == 2
        assert status["pending_generation"] is None
        assert status["loaded_at"] == staged["pending_loaded_at"]

        state_directory = tmp_path / "global" / ".scheduler-state"
        state_file = state_directory / "scheduling-policy.json"
        assert stat.S_IMODE(state_directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(state_file.stat().st_mode) == 0o600

        _write_scheduler(
            settings_path,
            root=tmp_path / "global",
            model_config=model_path,
            queue_delay=12.5,
            light_utilization=101,
        )
        rejected = await client.post("/api/v1/policy/reload")
        assert rejected.status_code == 400
        assert rejected.json()["code"] == "admission_rejected"
        assert service.scheduling_policy_snapshot() is current

        # A non-policy scheduler change rejects the whole candidate and keeps
        # generation two active.
        _write_scheduler(
            settings_path,
            root=tmp_path / "global",
            model_config=model_path,
            queue_delay=12.5,
            lease_ttl=91,
        )
        rejected = await client.post("/api/v1/policy/reload")
        assert rejected.status_code == 400
        assert rejected.json()["code"] == "admission_rejected"
        assert service.scheduling_policy_snapshot() is current

        # Model paths/backends/revisions are runtime semantics, not hot policy.
        _write_scheduler(
            settings_path,
            root=tmp_path / "global",
            model_config=model_path,
            queue_delay=12.5,
        )
        _write_models(model_path, startup_seconds=120, model_path="/changed")
        rejected = await client.post("/api/v1/policy/reload")
        assert rejected.status_code == 400
        assert service.scheduling_policy_snapshot() is current


async def test_source_limits_follow_the_atomic_policy_generation(
    database: Database,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models.yaml"
    settings_path = tmp_path / "scheduler.yaml"
    _write_models(model_path, startup_seconds=2)
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=30,
        source_limits=(3, 4),
    )
    settings = SchedulerSettings.from_yaml(settings_path)
    app = create_app(
        settings,
        database=database,
        run_dispatcher=False,
        settings_path=settings_path,
    )
    service = app.state.service

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        initial = await client.get("/api/v1/source-limits")
        assert initial.status_code == 200
        assert initial.headers["cache-control"] == "no-store"
        assert initial.json()["max_file_bytes"] == 3
        bundle = {"files": {"source.bin": base64.b64encode(b"1234").decode()}}
        rejected = await client.post("/api/v1/sources", json=bundle)
        assert rejected.status_code == 413

        _write_scheduler(
            settings_path,
            root=tmp_path / "global",
            model_config=model_path,
            queue_delay=30,
            source_limits=(4, 5),
        )
        staged = await client.post("/api/v1/policy/reload")
        assert staged.status_code == 200
        assert staged.json()["status"] == "draining"
        assert service.source_limits().max_file_bytes == 3

        assert not await service.dispatch_once()
        current = await client.get("/api/v1/source-limits")
        assert current.json()["max_file_bytes"] == 4
        assert current.json()["max_total_bytes"] == 5
        accepted = await client.post("/api/v1/sources", json=bundle)
        assert accepted.status_code == 201
        assert service.scheduling_policy_snapshot().generation == 2


@pytest.mark.parametrize("priority_only", [False, True, "restriction", "idle_override"])
async def test_policy_reload_pauses_dispatch_until_interactive_host_lease_finishes(
    database: Database,
    tmp_path: Path,
    priority_only: bool | str,
) -> None:
    model_path = tmp_path / "models.yaml"
    settings_path = tmp_path / "scheduler.yaml"
    _write_models(model_path, startup_seconds=2)
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=30,
    )
    if priority_only == "restriction":
        config = yaml.safe_load(settings_path.read_text())
        config["scheduling"]["foreign_capacity_sharing_site_ids"] = ["lab1"]
        settings_path.write_text(yaml.safe_dump(config))
    settings = SchedulerSettings.from_yaml(settings_path)
    app = create_app(
        settings,
        database=database,
        run_dispatcher=False,
        settings_path=settings_path,
    )
    service = app.state.service
    now = utc_now()
    async with database.transaction() as session:
        session.add(
            SiteRow(
                id="lab1",
                display_name="lab1",
                kind=SiteKind.LAB,
                state=SiteState.READY,
            )
        )
        session.add(
            JobRow(
                id="job-interactive-host",
                state=JobState.RUNNING,
                spec=JobSpec(entrypoint="experiment.py", model="model").model_dump(mode="json"),
            )
        )
        session.add(
            AttemptRow(
                id="attempt-interactive-host",
                job_id="job-interactive-host",
                number=1,
                state=AttemptState.RUNNING,
                site_id="lab1",
                started_at=now,
            )
        )
        session.add(
            LeaseRow(
                id="lease-interactive-host",
                attempt_id="attempt-interactive-host",
                site_id="lab1",
                state=LeaseState.ACTIVE,
                fencing_token_digest=hashlib.sha256(b"test-fence").hexdigest(),
                expires_at=now + timedelta(minutes=5),
            )
        )
        session.add(
            KernelSessionRow(
                id="kernel-interactive-host",
                state=KernelState.IDLE,
                mode=KernelMode.SERVICE,
                request={},
                job_id="job-interactive-host",
                attempt_id="attempt-interactive-host",
                site_id="lab1",
            )
        )

    _write_models(model_path, startup_seconds=2 if priority_only else 120)
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=30 if priority_only else 12,
    )
    if priority_only is True:
        config = yaml.safe_load(settings_path.read_text())
        config["scheduling"]["prefer_autodl_over_foreign_processes"] = True
        settings_path.write_text(yaml.safe_dump(config))
    if priority_only == "idle_override":
        config = yaml.safe_load(settings_path.read_text())
        config["scheduling"]["institutional_warm_actor_idle_timeout_seconds_by_site"] = {
            "lab1": 300,
            "remote-jupyter-1": 300,
        }
        settings_path.write_text(yaml.safe_dump(config))
    staged = await service.reload_scheduling_policy()
    assert staged["status"] == "draining"
    assert staged["blocker_count"] == 1
    assert not await service.dispatch_once()
    if priority_only:
        assert service.scheduling_policy_snapshot().generation == 2
        assert (await service.policy_status())["status"] == "active"
        async with database.session() as session:
            attempt = await session.get(AttemptRow, "attempt-interactive-host")
            lease = await session.get(LeaseRow, "lease-interactive-host")
            job = await session.get(JobRow, "job-interactive-host")
            kernel = await session.get(KernelSessionRow, "kernel-interactive-host")
            assert attempt is not None and attempt.state == AttemptState.RUNNING
            assert lease is not None and lease.state == LeaseState.ACTIVE
            assert lease.fencing_token_digest == hashlib.sha256(b"test-fence").hexdigest()
            assert lease.attempt_id == "attempt-interactive-host"
            assert lease.site_id == "lab1"
            assert job is not None and job.state == JobState.RUNNING
            assert kernel is not None and kernel.state == KernelState.IDLE
        restored = create_app(
            SchedulerSettings.from_yaml(settings_path),
            database=database,
            run_dispatcher=False,
            settings_path=settings_path,
        ).state.service
        assert (await restored.policy_status())["status"] == "active"
        assert restored.scheduling_policy_snapshot().generation == 2
        return
    assert service.scheduling_policy_snapshot().generation == 1

    # Interactive sessions use the same durable attempt/lease barrier as
    # batch work. They finish naturally; reload never stops or moves them.
    async with database.transaction() as session:
        attempt = await session.get(AttemptRow, "attempt-interactive-host")
        lease = await session.get(LeaseRow, "lease-interactive-host")
        job = await session.get(JobRow, "job-interactive-host")
        kernel = await session.get(KernelSessionRow, "kernel-interactive-host")
        assert attempt is not None and lease is not None and job is not None
        assert kernel is not None
        attempt.state = AttemptState.SUCCEEDED
        attempt.completed_at = utc_now()
        lease.state = LeaseState.RELEASED
        job.state = JobState.SUCCEEDED
        kernel.state = KernelState.STOPPED

    assert not await service.dispatch_once()
    assert service.scheduling_policy_snapshot().generation == 2
    assert (await service.policy_status())["status"] == "active"


async def test_policy_reload_pending_state_survives_scheduler_restart(
    database: Database,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models.yaml"
    settings_path = tmp_path / "scheduler.yaml"
    _write_models(model_path, startup_seconds=2)
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=30,
    )
    settings = SchedulerSettings.from_yaml(settings_path)
    first = create_app(
        settings,
        database=database,
        run_dispatcher=False,
        settings_path=settings_path,
    ).state.service
    _write_models(model_path, startup_seconds=120)
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=12,
    )
    assert (await first.reload_scheduling_policy())["pending_generation"] == 2

    # Startup uses the durable active+pending pair, not the edited YAML's
    # pending policy as if it had already become active.
    restarted = create_app(
        settings,
        database=database,
        run_dispatcher=False,
        settings_path=settings_path,
    ).state.service
    restored = await restarted.policy_status()
    assert restored["status"] == "draining"
    assert restored["generation"] == 1
    assert restored["pending_generation"] == 2
    assert restarted.scheduling_policy_snapshot().settings.queue_delay_seconds_per_job == 30

    assert not await restarted.dispatch_once()
    assert restarted.scheduling_policy_snapshot().generation == 2
    unchanged = await restarted.reload_scheduling_policy()
    assert unchanged["status"] == "active"
    assert unchanged["generation"] == 2


async def test_policy_staging_waits_behind_inflight_placement_transaction(
    database: Database,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models.yaml"
    settings_path = tmp_path / "scheduler.yaml"
    _write_models(model_path, startup_seconds=2)
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=30,
    )
    settings = SchedulerSettings.from_yaml(settings_path)
    repository = _BlockingInventoryRepository()
    service = SchedulerService(
        database=database,
        settings=settings,
        repository=repository,
        model_catalog=ModelCatalog.from_yaml(model_path),
        settings_path=settings_path,
    )
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=12,
    )

    dispatch = asyncio.create_task(service.dispatch_once())
    await repository.entered.wait()
    reload_task = asyncio.create_task(service.reload_scheduling_policy())
    await asyncio.sleep(0)
    assert not reload_task.done()
    repository.release.set()
    assert not await dispatch
    staged = await reload_task
    assert staged["status"] == "draining"
    assert staged["pending_generation"] == 2


async def test_policy_stage_cancellation_finishes_commit_and_keeps_dispatch_paused(
    database: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "models.yaml"
    settings_path = tmp_path / "scheduler.yaml"
    _write_models(model_path, startup_seconds=2)
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=30,
    )
    settings = SchedulerSettings.from_yaml(settings_path)
    service = create_app(
        settings,
        database=database,
        run_dispatcher=False,
        settings_path=settings_path,
    ).state.service
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=12,
    )
    assert service._policy_store is not None
    original_save = service._policy_store.save
    entered = threading.Event()
    release = threading.Event()

    def blocked_save(state: object) -> None:
        entered.set()
        assert release.wait(timeout=5)
        original_save(state)  # type: ignore[arg-type]

    monkeypatch.setattr(service._policy_store, "save", blocked_save)
    reload_task = asyncio.create_task(service.reload_scheduling_policy())
    assert await asyncio.to_thread(entered.wait, 5)
    reload_task.cancel()
    await asyncio.sleep(0)
    assert not reload_task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await reload_task

    status = await service.policy_status()
    assert status["status"] == "draining"
    assert status["pending_generation"] == 2
    monkeypatch.setattr(service._policy_store, "save", original_save)
    assert not await service.dispatch_once()
    assert service.scheduling_policy_snapshot().generation == 2


async def test_post_replace_fsync_failure_fails_closed_with_pending_generation(
    database: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "models.yaml"
    settings_path = tmp_path / "scheduler.yaml"
    _write_models(model_path, startup_seconds=2)
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=30,
    )
    settings = SchedulerSettings.from_yaml(settings_path)
    service = create_app(
        settings,
        database=database,
        run_dispatcher=False,
        settings_path=settings_path,
    ).state.service
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=12,
    )

    original_fsync = policy_module._fsync_directory
    monkeypatch.setattr(
        policy_module,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("injected fsync failure")),
    )
    with pytest.raises(DurabilityError, match="dispatch is paused"):
        await service.reload_scheduling_policy()
    status = await service.policy_status()
    assert status["status"] == "draining"
    assert status["pending_generation"] == 2

    monkeypatch.setattr(policy_module, "_fsync_directory", original_fsync)
    assert not await service.dispatch_once()
    assert service.scheduling_policy_snapshot().generation == 2


async def test_policy_drain_waits_for_durable_provider_startup(
    database: Database,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models.yaml"
    settings_path = tmp_path / "scheduler.yaml"
    _write_models(model_path, startup_seconds=2)
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=30,
    )
    settings = SchedulerSettings.from_yaml(settings_path)
    service = SchedulerService(
        database=database,
        settings=settings,
        model_catalog=ModelCatalog.from_yaml(model_path),
        provider_controller=object(),  # type: ignore[arg-type]
        settings_path=settings_path,
    )
    async with database.transaction() as session:
        session.add(
            ProviderInstanceRow(
                id="provider-starting",
                provider="autodl",
                external_id="pro-starting",
                site_id="autodl",
                state=ProviderInstanceState.STARTING,
                region="westDC3",
                gpu_spec="5090-p",
                image_uuid="image",
                metadata_json={
                    "requested_gpu_count": 1,
                    "startup_heartbeat_seen": False,
                },
            )
        )
    _write_scheduler(
        settings_path,
        root=tmp_path / "global",
        model_config=model_path,
        queue_delay=12,
    )
    staged = await service.reload_scheduling_policy()
    assert staged["blocker_count"] == 1
    assert not await service.dispatch_once()
    assert service.scheduling_policy_snapshot().generation == 1

    async with database.transaction() as session:
        row = await session.get(ProviderInstanceRow, "provider-starting")
        assert row is not None
        row.state = ProviderInstanceState.RUNNING
        row.metadata_json = {**row.metadata_json, "startup_heartbeat_seen": True}
    assert not await service.dispatch_once()
    assert service.scheduling_policy_snapshot().generation == 2


async def test_policy_reload_endpoint_uses_normal_api_authentication(
    database: Database,
    tmp_path: Path,
) -> None:
    credential = "policy-reload-test-credential"
    settings = SchedulerSettings(
        database_dsn="sqlite+aiosqlite://",
        global_root=tmp_path,
        public_url="https://scheduler.example",
        insecure_development_mode=False,
        api_token=credential,
        fence_signing_key="f" * 32,
    )
    app = create_app(settings, database=database, run_dispatcher=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://scheduler.example",
    ) as client:
        assert (await client.post("/api/v1/policy/reload")).status_code == 401
        authenticated = await client.post(
            "/api/v1/policy/reload",
            headers={"Authorization": f"Bearer {credential}"},
        )
        assert authenticated.status_code == 409
        assert authenticated.json()["code"] == "conflict"
