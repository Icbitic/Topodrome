"""Interactive session state and service-versus-raw lease invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from topodrome.domain.enums import KernelMode, KernelState
from topodrome.domain.specs import KernelRequest, KernelSummary


class SessionStateError(RuntimeError):
    """A session transition is invalid for its current lifecycle state."""


class SessionLeaseConflict(RuntimeError):
    """A physical GPU, private actor, or workload lease is already owned."""


@dataclass(frozen=True, slots=True)
class SessionLease:
    """Resources allocated to a lightweight service or exclusive raw kernel."""

    workload_lease_id: str
    actor_id: str
    gpu_uuids: tuple[str, ...] = ()
    exclusive: bool = False
    private_actor: bool = False

    def __post_init__(self) -> None:
        if not self.workload_lease_id or not self.actor_id:
            raise ValueError("session leases require workload and actor identities")
        if len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("session lease contains duplicate GPU UUIDs")

    def validate_for(self, mode: KernelMode, *, requested_gpus: int) -> None:
        if mode is KernelMode.RAW_MODEL:
            required = max(1, requested_gpus)
            if not self.exclusive or not self.private_actor or len(self.gpu_uuids) < required:
                raise ValueError(
                    "raw-model sessions require an exclusive private actor and GPU lease"
                )
            return
        if self.exclusive or self.private_actor or self.gpu_uuids:
            raise ValueError(
                "service sessions use a shared actor and do not own its physical GPU lease"
            )


_ALLOWED_TRANSITIONS: dict[KernelState, frozenset[KernelState]] = {
    KernelState.REQUESTED: frozenset(
        {KernelState.STARTING, KernelState.STOPPING, KernelState.LOST}
    ),
    KernelState.STARTING: frozenset({KernelState.RUNNING, KernelState.STOPPING, KernelState.LOST}),
    KernelState.RUNNING: frozenset({KernelState.IDLE, KernelState.STOPPING, KernelState.LOST}),
    KernelState.IDLE: frozenset({KernelState.RUNNING, KernelState.STOPPING, KernelState.LOST}),
    KernelState.STOPPING: frozenset({KernelState.STOPPED, KernelState.LOST}),
    KernelState.STOPPED: frozenset(),
    KernelState.LOST: frozenset(),
}


@dataclass(slots=True)
class KernelSession:
    session_id: str
    request: KernelRequest
    created_at: datetime
    last_activity_at: datetime
    state: KernelState = KernelState.REQUESTED
    site_id: str | None = None
    connect_url: str | None = field(default=None, repr=False)
    lease: SessionLease | None = field(default=None, repr=False)
    runtime_id: str | None = None
    relay_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.created_at.tzinfo is None or self.last_activity_at.tzinfo is None:
            raise ValueError("session timestamps must be timezone-aware")
        if self.request.mode is KernelMode.RAW_MODEL and self.request.resources.gpu < 1:
            raise ValueError("raw-model sessions require at least one GPU")

    @property
    def expires_at(self) -> datetime:
        return self.last_activity_at + timedelta(seconds=self.request.idle_timeout_seconds)

    @property
    def terminal(self) -> bool:
        return self.state in {KernelState.STOPPED, KernelState.LOST}

    def transition(self, target: KernelState) -> None:
        if target is self.state:
            return
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise SessionStateError(f"cannot transition kernel from {self.state} to {target}")
        self.state = target

    def begin_start(self) -> None:
        self.transition(KernelState.STARTING)

    def mark_running(
        self,
        *,
        lease: SessionLease,
        runtime_id: str,
        relay_id: str,
        connect_url: str,
        site_id: str,
        now: datetime,
    ) -> None:
        lease.validate_for(self.request.mode, requested_gpus=self.request.resources.gpu)
        if not runtime_id or not relay_id or not connect_url or not site_id:
            raise ValueError("a running session requires runtime, relay, URL, and site identities")
        self.transition(KernelState.RUNNING)
        self.lease = lease
        self.runtime_id = runtime_id
        self.relay_id = relay_id
        self.connect_url = connect_url
        self.site_id = site_id
        self.touch(now)

    def mark_idle(self, now: datetime) -> None:
        self._validate_now(now)
        self.transition(KernelState.IDLE)
        self.last_activity_at = now

    def touch(self, now: datetime) -> None:
        self._validate_now(now)
        if self.state is KernelState.IDLE:
            self.transition(KernelState.RUNNING)
        elif self.state is not KernelState.RUNNING:
            raise SessionStateError("only running or idle sessions accept activity")
        self.last_activity_at = now

    def request_stop(self) -> None:
        if not self.terminal:
            self.transition(KernelState.STOPPING)

    def mark_stopped(self) -> None:
        if self.state is not KernelState.STOPPED:
            self.transition(KernelState.STOPPED)
        self._clear_connection()

    def mark_lost(self) -> None:
        if self.state is not KernelState.LOST:
            self.transition(KernelState.LOST)
        self._clear_connection()

    def is_idle_expired(self, now: datetime) -> bool:
        self._validate_now(now)
        return self.state in {KernelState.RUNNING, KernelState.IDLE} and now >= self.expires_at

    def summary(self) -> KernelSummary:
        return KernelSummary(
            session_id=self.session_id,
            state=self.state,
            mode=self.request.mode,
            model=self.request.model,
            site_id=self.site_id,
            connect_url=self.connect_url,
            created_at=self.created_at,
            expires_at=None if self.terminal else self.expires_at,
        )

    def _validate_now(self, now: datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("session clocks must be timezone-aware")
        if now < self.created_at:
            raise ValueError("session activity cannot predate creation")

    def _clear_connection(self) -> None:
        self.connect_url = None
        self.lease = None
        self.runtime_id = None
        self.relay_id = None


class SessionLeaseRegistry:
    """In-process guard for actor and physical-GPU sharing rules.

    The durable scheduler lease remains authoritative. This registry is the
    site-local final check that prevents raw-model state from being shared.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionLease] = {}
        self._workload_owners: dict[str, str] = {}
        self._gpu_owners: dict[str, str] = {}
        self._private_actor_owners: dict[str, str] = {}
        self._service_actor_sessions: dict[str, set[str]] = {}

    def reserve(
        self,
        session_id: str,
        mode: KernelMode,
        lease: SessionLease,
        *,
        requested_gpus: int,
    ) -> None:
        if session_id in self._sessions:
            if self._sessions[session_id] == lease:
                return
            raise SessionLeaseConflict("session already owns a different lease")
        lease.validate_for(mode, requested_gpus=requested_gpus)
        owner = self._workload_owners.get(lease.workload_lease_id)
        if owner is not None:
            raise SessionLeaseConflict("workload lease is already owned")

        if mode is KernelMode.RAW_MODEL:
            if lease.actor_id in self._private_actor_owners:
                raise SessionLeaseConflict("private actor is already owned")
            if self._service_actor_sessions.get(lease.actor_id):
                raise SessionLeaseConflict("actor is already shared by service sessions")
            for gpu_uuid in lease.gpu_uuids:
                if gpu_uuid in self._gpu_owners:
                    raise SessionLeaseConflict("physical GPU is already owned")
        elif lease.actor_id in self._private_actor_owners:
            raise SessionLeaseConflict("private raw-model actor cannot be shared")

        self._sessions[session_id] = lease
        self._workload_owners[lease.workload_lease_id] = session_id
        if mode is KernelMode.RAW_MODEL:
            self._private_actor_owners[lease.actor_id] = session_id
            for gpu_uuid in lease.gpu_uuids:
                self._gpu_owners[gpu_uuid] = session_id
        else:
            self._service_actor_sessions.setdefault(lease.actor_id, set()).add(session_id)

    def release(self, session_id: str) -> None:
        lease = self._sessions.pop(session_id, None)
        if lease is None:
            return
        self._workload_owners.pop(lease.workload_lease_id, None)
        if self._private_actor_owners.get(lease.actor_id) == session_id:
            self._private_actor_owners.pop(lease.actor_id, None)
        for gpu_uuid in lease.gpu_uuids:
            if self._gpu_owners.get(gpu_uuid) == session_id:
                self._gpu_owners.pop(gpu_uuid, None)
        service_sessions = self._service_actor_sessions.get(lease.actor_id)
        if service_sessions is not None:
            service_sessions.discard(session_id)
            if not service_sessions:
                self._service_actor_sessions.pop(lease.actor_id, None)


__all__ = [
    "KernelSession",
    "SessionLease",
    "SessionLeaseConflict",
    "SessionLeaseRegistry",
    "SessionStateError",
]
