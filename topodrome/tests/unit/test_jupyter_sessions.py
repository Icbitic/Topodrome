from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from topodrome.domain.enums import KernelMode, KernelState
from topodrome.domain.specs import KernelRequest, ResourceRequirements
from topodrome.jupyter.session import (
    KernelSession,
    SessionLease,
    SessionLeaseConflict,
    SessionLeaseRegistry,
    SessionStateError,
)

pytestmark = pytest.mark.unit


def _request(mode: KernelMode, *, gpu: int = 1) -> KernelRequest:
    return KernelRequest(
        model="test/llama",
        mode=mode,
        resources=ResourceRequirements(gpu=gpu),
        idle_timeout_seconds=60,
    )


def test_service_sessions_share_actor_without_owning_physical_gpu() -> None:
    registry = SessionLeaseRegistry()
    first = SessionLease(workload_lease_id="lease-1", actor_id="shared-actor")
    second = SessionLease(workload_lease_id="lease-2", actor_id="shared-actor")

    registry.reserve("session-1", KernelMode.SERVICE, first, requested_gpus=1)
    registry.reserve("session-2", KernelMode.SERVICE, second, requested_gpus=1)

    with pytest.raises(ValueError, match="shared actor"):
        registry.reserve(
            "bad-service",
            KernelMode.SERVICE,
            SessionLease(
                workload_lease_id="lease-3",
                actor_id="shared-actor",
                gpu_uuids=("GPU-1",),
            ),
            requested_gpus=1,
        )


def test_raw_model_session_is_private_and_gpu_exclusive() -> None:
    registry = SessionLeaseRegistry()
    raw = SessionLease(
        workload_lease_id="lease-raw",
        actor_id="private-actor",
        gpu_uuids=("GPU-1",),
        exclusive=True,
        private_actor=True,
    )
    registry.reserve("raw-1", KernelMode.RAW_MODEL, raw, requested_gpus=1)

    with pytest.raises(SessionLeaseConflict, match=r"GPU|actor"):
        registry.reserve(
            "raw-2",
            KernelMode.RAW_MODEL,
            SessionLease(
                workload_lease_id="lease-other",
                actor_id="other-actor",
                gpu_uuids=("GPU-1",),
                exclusive=True,
                private_actor=True,
            ),
            requested_gpus=1,
        )
    with pytest.raises(SessionLeaseConflict, match="private"):
        registry.reserve(
            "service",
            KernelMode.SERVICE,
            SessionLease(workload_lease_id="lease-service", actor_id="private-actor"),
            requested_gpus=1,
        )

    registry.release("raw-1")
    registry.reserve(
        "raw-2",
        KernelMode.RAW_MODEL,
        SessionLease(
            workload_lease_id="lease-other",
            actor_id="other-actor",
            gpu_uuids=("GPU-1",),
            exclusive=True,
            private_actor=True,
        ),
        requested_gpus=1,
    )


def test_session_state_and_sliding_idle_expiry() -> None:
    created = datetime(2026, 9, 7, tzinfo=UTC)
    session = KernelSession(
        session_id="kernel-1",
        request=_request(KernelMode.SERVICE),
        created_at=created,
        last_activity_at=created,
    )
    session.begin_start()
    session.mark_running(
        lease=SessionLease(workload_lease_id="lease", actor_id="actor"),
        runtime_id="runtime",
        relay_id="relay",
        connect_url="https://gateway.invalid/kernels/1?token=session-secret",
        site_id="lab1",
        now=created,
    )
    session.mark_idle(created + timedelta(seconds=10))

    assert not session.is_idle_expired(created + timedelta(seconds=69))
    assert session.is_idle_expired(created + timedelta(seconds=70))
    session.touch(created + timedelta(seconds=20))
    assert session.state is KernelState.RUNNING
    assert session.expires_at == created + timedelta(seconds=80)

    session.request_stop()
    session.mark_stopped()
    assert session.summary().connect_url is None
    with pytest.raises(SessionStateError):
        session.begin_start()


def test_raw_request_cannot_claim_zero_gpus() -> None:
    now = datetime(2026, 9, 7, tzinfo=UTC)
    with pytest.raises(ValueError, match="at least one GPU"):
        KernelSession(
            session_id="kernel-1",
            request=_request(KernelMode.RAW_MODEL, gpu=0),
            created_at=now,
            last_activity_at=now,
        )
