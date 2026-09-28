"""Bounded advisory GPU process samples; never placement or process authority."""

import unicodedata
from collections import OrderedDict
from datetime import UTC, datetime

from topodrome.domain.specs import SiteHeartbeat


def _text(value: str | None, limit: int = 160) -> str | None:
    if not value:
        return None
    return "".join(c for c in value if not unicodedata.category(c).startswith("C"))[:limit]


def _time(value) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class ProcessTelemetry:
    """Only the latest committed heartbeat; lost on scheduler restart on purpose.

    Bound both sites and entries per sample. Never persist command lines or
    use this diagnostic cache for admission, ownership decisions or signals.
    """

    MAX_SITES = 100
    MAX_GPUS = 32
    MAX_PROCESSES = 64

    def __init__(self):
        self._sites = OrderedDict()

    def record(self, heartbeat: SiteHeartbeat, *, generation: str | None, received_at: datetime):
        gpus = {}
        for gpu in heartbeat.gpus[: self.MAX_GPUS]:
            processes = sorted(gpu.foreign_processes, key=lambda p: (-p.used_memory_bytes, p.pid))
            gpus[gpu.uuid] = {
                "observed_at": _time(gpu.observed_at),
                "processes_total": len(processes),
                "processes_truncated": len(processes) > self.MAX_PROCESSES,
                "processes": [
                    {
                        "pid": process.pid,
                        "used_memory_bytes": process.used_memory_bytes,
                        "process_name": _text(process.process_name),
                        "username": _text(process.username, 64),
                        "script_name": _text(process.script_name),
                        "kind": "cooperative" if process.reclaimable else "foreign",
                    }
                    for process in processes[: self.MAX_PROCESSES]
                ],
            }
        self._sites[heartbeat.site_id] = (
            (heartbeat.agent_session_id, generation),
            _time(received_at),
            gpus,
        )
        self._sites.move_to_end(heartbeat.site_id)
        while len(self._sites) > self.MAX_SITES:
            self._sites.popitem(last=False)

    def decorate(self, rows, *, identities, now: datetime, ttl_seconds: float):
        for row in rows:
            row.update(processes=None, processes_status="unavailable", processes_received_at=None)
            cached = self._sites.get(row["site_id"])
            if cached is None:
                continue
            identity, received_at, gpus = cached
            # Replaced/disconnected agents and mismatched inventory must never
            # inherit another generation's PIDs, even for the same GPU UUID.
            if identity != identities.get(row["site_id"]):
                continue
            gpu = gpus.get(row["uuid"])
            if gpu is None or gpu["observed_at"] != _time(row.get("observed_at")):
                continue
            row.update(
                processes=[dict(process) for process in gpu["processes"]],
                processes_total=gpu["processes_total"],
                processes_truncated=gpu["processes_truncated"],
                processes_received_at=received_at,
                processes_status=(
                    "current"
                    if 0 <= (_time(now) - received_at).total_seconds() <= ttl_seconds
                    else "stale"
                ),
                processes_ttl_seconds=ttl_seconds,
            )
