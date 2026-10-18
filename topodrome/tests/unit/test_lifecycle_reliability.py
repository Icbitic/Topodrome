import asyncio
import hashlib
import os
import subprocess
import sys
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from tests.unit.test_scheduler_repository import heartbeat, job_spec
from tests.unit.test_site_agent_preemption import _build_agent, _CooperativeBackend
from topodrome.config import SchedulerSettings
from topodrome.db.models import InputStageRow, LeaseGPUClaimRow
from topodrome.db.repositories import SchedulerRepository
from topodrome.domain.enums import AttemptState, JobState, LeaseState
from topodrome.domain.specs import AssetMember, AssetRef, utc_now
from topodrome.protocol.fencing import FencingAuthority
from topodrome.runtime import ArtifactRegistration, ModelRegistry
from topodrome.scheduler.finalization import expired_upload_message, recover
from topodrome.scheduler.input_staging import cancel_stage, prepare_inputs, receive_stage
from topodrome.scheduler.status import system_status_snapshot
from topodrome.site_agent.agent import Assignment, _PendingTerminalState
from topodrome.site_agent.finalization import FinalizationJournal
from topodrome.site_agent.input_staging import InputStager
from topodrome.site_agent.runtime_backend import RuntimeExecutionBackend, _ExecutionReport
from topodrome.site_agent.workspace import WorkspaceManager
from topodrome.storage import AssetResolver, LocalStorage, asset_manifest_digest
from topodrome.storage.base import StorageUnavailableError
from topodrome.storage.progress import (
    TransferCancelled,
    cancellable_transfers,
    check_transfer_cancelled,
    transfer_lock,
)
from topodrome.storage.transfer_process import run_transfer

pytestmark = pytest.mark.unit


def test_cancel_transfer_reaps_exact_child_and_retains_partial(tmp_path):
    partial = tmp_path / "partial"
    event = threading.Event()
    pid = tmp_path / "pid"
    script = (
        "import os,time,pathlib; "
        f"pathlib.Path({str(pid)!r}).write_text(str(os.getpid())); "
        "print('partial', flush=True); time.sleep(60)"
    )

    def cancel_when_started():
        while not pid.exists():
            time.sleep(0.01)
        event.set()

    thread = threading.Thread(target=cancel_when_started)
    thread.start()
    started = time.monotonic()
    with partial.open("wb") as output, cancellable_transfers(event):
        with pytest.raises(TransferCancelled):
            run_transfer(
                [sys.executable, "-c", script], timeout=30, stdout=output, stderr=subprocess.DEVNULL
            )
    thread.join(timeout=2)
    assert time.monotonic() - started < 2
    assert partial.read_bytes() == b"partial\n"
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


def test_lock_wait_cancels_without_unlocking_another_owner():
    lock = threading.Lock()
    lock.acquire()
    event = threading.Event()
    event.set()
    with cancellable_transfers(threading.Event()):
        with pytest.raises(TransferCancelled), cancellable_transfers(event), transfer_lock(lock):
            raise AssertionError("entered occupied lock")
    assert lock.locked()
    lock.release()


@pytest.mark.asyncio
async def test_input_staging_has_no_claims_progress_and_fences_stale_reports(database):
    repo = SchedulerRepository()
    async with database.transaction() as session:
        await repo.upsert_heartbeat(session, heartbeat())
        job = await repo.submit_job(session, job_spec())
        row = await repo.get_job(session, job.id)
        ready, command = await prepare_inputs(session, row, "lab1")
        assert not ready and command.kind == "stage_inputs"
        stage_id = command.payload["stage_id"]
        assert not await session.scalar(select(func.count()).select_from(LeaseGPUClaimRow))
        assert row.state is JobState.QUEUED
        await receive_stage(
            session,
            "other-site",
            {
                **command.payload,
                "state": "ready",
                "progress": {"current": 0},
            },
        )
        assert (await session.get(InputStageRow, row.id)).state == "downloading"
        snapshot = await system_status_snapshot(session)
        assert snapshot["jobs"][0]["state"] == "downloading"
        assert snapshot["jobs"][0]["gpu_count"] == 0
        await receive_stage(
            session,
            "lab1",
            {
                **command.payload,
                "state": "ready",
                "progress": {"current": 0},
            },
        )
        ready, command = await prepare_inputs(session, row, "lab1")
        assert ready and command is None
        cancellation = await cancel_stage(session, row.id)
        assert cancellation.payload["stage_id"] == stage_id
        await receive_stage(
            session,
            "lab1",
            {
                "job_id": row.id,
                "stage_id": stage_id,
                "state": "ready",
                "progress": {"current": 0},
            },
        )
        assert (await session.get(InputStageRow, row.id)).state == "canceled"


@pytest.mark.asyncio
async def test_stager_cancellation_waits_for_owned_thread_and_unpins():
    started = threading.Event()
    stopped = threading.Event()
    pins = set()

    def resolve(_ref):
        started.set()
        try:
            while True:
                check_transfer_cancelled()
                time.sleep(0.01)
        finally:
            stopped.set()

    resolver = SimpleNamespace(
        pin=lambda owner, refs: pins.add(owner),
        unpin=lambda owner: pins.discard(owner),
        resolve=resolve,
    )

    async def publish(*args):
        pass

    member = AssetMember(
        logical_path="assets/a",
        relative_path="a",
        size_bytes=1,
        sha256=hashlib.sha256(b"a").hexdigest(),
    )
    ref = AssetRef(
        source_ref="asset://x/a@v1",
        kind="file",
        members=(member,),
        digest=asset_manifest_digest((member,)),
        total_bytes=1,
    )
    stager = InputStager(resolver, SimpleNamespace(publish=publish))
    payload = {
        "job_id": "job-1",
        "stage_id": "stage-1",
        "assets": {"a": ref.model_dump()},
        "expires_at": (utc_now() + timedelta(seconds=120)).isoformat(),
    }
    await stager.accept(payload)
    await asyncio.wait_for(asyncio.to_thread(started.wait), 1)
    await stager.accept(payload)  # Duplicate delivery cannot start another transfer.
    task = stager.stages["job-1"].task
    stager.cancel(payload)
    await asyncio.wait_for(task, 1)
    assert stopped.is_set() and not pins
    await stager.tick()
    assert not stager.stages


@pytest.mark.asyncio
async def test_restart_resumes_only_upload_and_retry_preserves_computed_output(tmp_path):
    class FlakyStorage(LocalStorage):
        failures = 1

        def flush_and_verify(self, path, **kwargs):
            if self.failures:
                self.failures -= 1
                raise StorageUnavailableError("synthetic transport outage")
            return super().flush_and_verify(path, **kwargs)

    storage = FlakyStorage(tmp_path / "global")
    journal = FinalizationJournal(tmp_path / "private" / "finalization")
    spec = job_spec().model_copy(update={"required_outputs": ("result.txt",)})
    assignment = SimpleNamespace(
        job_id="job-1",
        attempt_id="attempt-1",
        raw_spec=spec.model_dump(mode="json"),
        checkpoint_ref=None,
    )
    workspace = WorkspaceManager(storage=storage, scratch_root=tmp_path / "scratch").prepare(
        assignment.job_id, assignment.attempt_id
    )
    output = workspace.scratch_dir / "computed.txt"
    output.write_text("expensive completed computation")
    backend = RuntimeExecutionBackend(
        site_id="lab1",
        registry=ModelRegistry(),
        runtime=SimpleNamespace(),
        storage=storage,
        use_ray=False,
        finalization_journal=journal,
    )
    report = _ExecutionReport((ArtifactRegistration(output, "result.txt", True),), None)
    backend._snapshot_completion(assignment, workspace, report)
    assert journal.contains(assignment.attempt_id)
    output.unlink()  # Recovery cannot depend on scratch or an actor handle.
    restarted = RuntimeExecutionBackend(
        site_id="lab1",
        registry=ModelRegistry(),
        runtime=SimpleNamespace(),
        storage=storage,
        use_ray=False,
        finalization_journal=FinalizationJournal(journal.root),
    )
    result = await asyncio.wait_for(
        restarted.resume_finalization(
            assignment, workspace, next(restarted.finalization_journal.pending())
        ),
        4,
    )
    assert result.succeeded and result.artifacts_verified
    assert journal.contains(assignment.attempt_id)  # Not removed before terminal ACK.
    assert "fencing_token" not in next(journal.root.glob("*.json")).read_text()
    assert (journal.root.stat().st_mode & 0o777) == 0o700


@pytest.mark.asyncio
async def test_failed_or_canceled_job_does_not_wait_for_remote_log_upload(tmp_path):
    agent, assignment, _, fences, _ = _build_agent(tmp_path, _CooperativeBackend(""))
    agent.terminal_log_journal = FinalizationJournal(tmp_path / "terminal-logs")
    workspace = agent.workspaces.prepare(assignment.job_id, assignment.attempt_id)
    (workspace.log_dir / "attempt.log").write_text("before death")

    async def unavailable(*args):
        raise AssertionError("must not attempt remote I/O before releasing compute")

    agent.backend.verify_attempt_log = unavailable
    result = await agent._complete_terminal_log(
        assignment, workspace, _PendingTerminalState("canceled"), None
    )
    assert result.state == "canceled" and not result.log_complete
    assert agent.terminal_log_journal.contains(assignment.attempt_id)
    assert (await agent._activity_payload())["artifact_flush_pending"]
    fences.close()


@pytest.mark.asyncio
async def test_recovery_is_idempotent_upload_only_and_never_revives_cancelled_job(database):
    repo = SchedulerRepository()
    authority = FencingAuthority(b"f" * 32)
    settings = SchedulerSettings(fence_signing_key="f" * 64)
    async with database.transaction() as session:
        await repo.upsert_heartbeat(session, heartbeat())
        await repo.submit_job(session, job_spec())
        job = await repo.claim_next_job(session)
        offer = await repo.create_offer(
            session,
            job=job,
            site_id="lab1",
            gpu_uuids=("gpu-0",),
            authority=authority,
            offer_ttl=timedelta(seconds=30),
        )
        await repo.accept_offer(
            session,
            attempt_id=offer.attempt_id,
            site_id="lab1",
            token=offer.fencing_token,
            authority=authority,
            lease_ttl=timedelta(seconds=90),
        )
        for state in (AttemptState.STARTING, AttemptState.RUNNING, AttemptState.COMMITTING):
            await repo.update_attempt(
                session,
                attempt_id=offer.attempt_id,
                site_id="lab1",
                token=offer.fencing_token,
                authority=authority,
                target=state,
                gpu_released=state is AttemptState.COMMITTING,
            )
        attempt = await repo._get_attempt(session, offer.attempt_id, lock=True)
        attempt.lease.expires_at = utc_now() - timedelta(seconds=5)
        await session.flush()
        await repo.reap_expired_leases(session, finalization_grace=timedelta(hours=24))
        assert attempt.state is AttemptState.COMMITTING
        assert await expired_upload_message(
            session,
            repo,
            authority,
            "lab1",
            {
                "attempt_id": attempt.id,
                "fencing_token": offer.fencing_token,
            },
        )
        receipt = SimpleNamespace(acknowledged_at=utc_now(), message_id="recover-1")
        payload = {"job_id": job.id, "attempt_id": attempt.id}
        result = await recover(session, repo, authority, "lab1", payload, settings, receipt)
        assert result["accepted"]
        assignment = Assignment.from_mapping(result["assignment"])
        assert assignment.attempt_id == attempt.id
        assert attempt.lease.state is LeaseState.RELEASING
        assert not any(c.active for c in attempt.lease.claims)
        replay = await recover(
            session, repo, authority, "lab1", payload, settings, receipt, replayed=True
        )
        assert result == replay
        attempt.job.state = JobState.CANCELING
        assert not (await recover(session, repo, authority, "lab1", payload, settings, receipt))[
            "accepted"
        ]


def test_cache_pins_protect_shared_payload_and_never_evict_authoritative_storage(tmp_path):
    local = LocalStorage(tmp_path / "local")
    path = local.resolve("assets/a")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"a")
    member = AssetMember(
        logical_path="assets/a",
        relative_path="a",
        size_bytes=1,
        sha256=hashlib.sha256(b"a").hexdigest(),
    )
    ref = AssetRef(
        source_ref="asset://x/a@v1",
        kind="file",
        members=(member,),
        digest=asset_manifest_digest((member,)),
        total_bytes=1,
    )
    resolver = AssetResolver(local)
    resolver.resolve(ref)
    assert resolver.collect(max_age_seconds=0, max_bytes=0) == 0
    assert path.exists()
    # A mirror validates the remote copy before allowing exact replica unlink.
    local.replica_eviction_candidate = lambda member: path.stat()

    def evict(member, candidate):
        assert candidate == path.stat()
        path.unlink()
        return 1

    local.evict_verified_replica = evict
    resolver.pin("active-job", (ref,))
    assert resolver.collect(max_age_seconds=0, max_bytes=0) == 0
    resolver.unpin("active-job")
    assert resolver.collect(max_age_seconds=0, max_bytes=0) == 1
    assert not path.exists() and not resolver.cached_assets()


def test_upload_retry_verifies_existing_remote_bytes_instead_of_copying_again(tmp_path):
    from tests.unit.test_storage_http_mirror import FakeAuthority, _storage
    from topodrome.storage.progress import reuse_verified_outputs

    remote = FakeAuthority()
    storage = _storage(tmp_path, remote)
    target = storage.resolve("jobs/a/attempts/b/committed/result")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"completed result")
    storage.flush_and_verify("jobs/a/attempts/b/committed/result")
    commands = len(remote.calls)
    with reuse_verified_outputs():
        verified = storage.flush_and_verify("jobs/a/attempts/b/committed/result")
    assert verified.digest.sha256 == hashlib.sha256(b"completed result").hexdigest()
    assert not any(command[0] == "write" for command in remote.calls[commands:])


@pytest.mark.asyncio
async def test_dispatch_requires_real_staging_ack_before_any_gpu_lease(database, tmp_path):
    from tests.unit.test_scheduler_dispatch import _heartbeat, _source_ref
    from topodrome.domain.enums import SiteKind
    from topodrome.scheduler.control import AgentConnection, AgentConnectionRegistry
    from topodrome.scheduler.service import SchedulerService

    connections = AgentConnectionRegistry()
    messages = []

    async def send(message):
        messages.append(message)

    connection = AgentConnection("lab1", "lab1-session", send)
    await connections.register(connection)
    service = SchedulerService(
        database=database,
        connections=connections,
        settings=SchedulerSettings(global_root=tmp_path, fence_signing_key="f" * 64),
    )
    async with database.transaction() as session:
        await service.repository.upsert_heartbeat(
            session, _heartbeat("lab1", SiteKind.LAB), connection_generation=connection.generation
        )
    source_ref = _source_ref(service)
    job = await service.submit(job_spec().model_copy(update={"source_ref": source_ref}))
    assert not await service.dispatch_once()
    stage_command = messages[-1]["payload"]
    assert stage_command["job_id"] == job.job_id
    async with database.session() as session:
        assert not await session.scalar(select(func.count()).select_from(LeaseGPUClaimRow))
    from topodrome.site_agent.runtime_backend import source_resolver_for_storage

    storage = LocalStorage(tmp_path)

    async def publish(kind, payload):
        assert kind == "input_stage"
        async with database.transaction() as session:
            await receive_stage(session, "lab1", payload)

    stager = InputStager(
        AssetResolver(storage),
        SimpleNamespace(publish=publish),
        source_resolver=source_resolver_for_storage(storage),
    )
    await stager.accept(stage_command)
    await stager.stages[job.job_id].task
    await stager.tick()
    assert await service.dispatch_once()
    async with database.session() as session:
        assert await session.scalar(select(func.count()).select_from(LeaseGPUClaimRow)) == 1
    await stager.close()
