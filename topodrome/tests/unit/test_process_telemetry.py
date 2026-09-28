from datetime import UTC, datetime, timedelta

import pytest

from topodrome.domain.specs import GPUInventory, GPUProcess, SiteHeartbeat
from topodrome.scheduler.process_telemetry import ProcessTelemetry

pytestmark = pytest.mark.unit
NOW = datetime(2026, 9, 28, tzinfo=UTC)


def heartbeat(*, sequence=1, processes=None, site="lab1"):
    return SiteHeartbeat(
        site_id=site,
        site_kind="lab",
        agent_session_id="session",
        sequence=sequence,
        gpus=(
            GPUInventory(
                uuid="GPU-0",
                index=0,
                name="A800",
                total_memory_bytes=80 * 1024**3,
                free_memory_bytes=60 * 1024**3,
                utilization_percent=50,
                observed_at=NOW,
                processes=processes
                or (
                    GPUProcess(
                        pid=10,
                        used_memory_bytes=100,
                        process_name="python\n\x1b",
                        username="alice",
                        script_name="train.py",
                    ),
                    GPUProcess(
                        pid=20, used_memory_bytes=200, process_name="python", reclaimable=True
                    ),
                    GPUProcess(pid=30, used_memory_bytes=300, topodrome_owned=True),
                ),
            ),
        ),
    )


def decorated(store, *, identity=("session", "generation"), now=NOW, observed=NOW):
    row = {"site_id": "lab1", "uuid": "GPU-0", "observed_at": observed}
    store.decorate([row], identities={"lab1": identity}, now=now, ttl_seconds=30)
    return row


def test_process_telemetry_is_sanitized_bounded_and_advisory():
    store = ProcessTelemetry()
    assert decorated(store)["processes_status"] == "unavailable"
    store.record(heartbeat(), generation="generation", received_at=NOW)
    row = decorated(store)
    assert row["processes_status"] == "current"
    assert [p["pid"] for p in row["processes"]] == [20, 10]
    assert row["processes"][0]["kind"] == "cooperative"
    assert row["processes"][1]["process_name"] == "python"
    assert row["processes"][1]["username"] == "alice"
    assert all("command_line" not in process for process in row["processes"])
    row["processes"][0]["pid"] = 999
    assert decorated(store)["processes"][0]["pid"] == 20

    many = tuple(GPUProcess(pid=i + 1) for i in range(store.MAX_PROCESSES + 5))
    store.record(heartbeat(processes=many), generation="generation", received_at=NOW)
    row = decorated(store)
    assert row["processes_truncated"]
    assert row["processes_total"] == len(many)
    assert len(row["processes"]) == store.MAX_PROCESSES
    for i in range(store.MAX_SITES):
        store.record(heartbeat(site=f"other-{i}"), generation="generation", received_at=NOW)
    assert len(store._sites) == store.MAX_SITES
    assert decorated(store)["processes_status"] == "unavailable"


@pytest.mark.parametrize(
    "identity", [None, ("session", "replacement"), ("new-session", "generation")]
)
def test_replaced_or_disconnected_agents_cannot_lend_their_pids(identity):
    store = ProcessTelemetry()
    store.record(heartbeat(), generation="generation", received_at=NOW)
    row = decorated(store, identity=identity)
    assert row["processes"] is None
    assert row["processes_status"] == "unavailable"


def test_process_samples_expire_and_must_match_the_inventory_sample():
    store = ProcessTelemetry()
    store.record(heartbeat(), generation="generation", received_at=NOW)
    assert decorated(store, now=NOW + timedelta(seconds=31))["processes_status"] == "stale"
    assert decorated(store, observed=NOW + timedelta(seconds=1))["processes"] is None
    assert decorated(store, observed=NOW.isoformat())["processes_status"] == "current"


def test_old_agent_payload_is_accepted_without_user_or_script():
    process = GPUProcess(pid=123, process_name="python")
    store = ProcessTelemetry()
    store.record(heartbeat(processes=(process,)), generation="generation", received_at=NOW)
    process = decorated(store)["processes"][0]
    assert process["username"] is None and process["script_name"] is None
