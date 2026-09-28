from __future__ import annotations

import asyncio
import json
import secrets
import sqlite3
import stat
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from topodrome.domain.specs import SiteHeartbeat
from topodrome.site_agent.outbound import (
    EventAcknowledgement,
    OutboundClient,
    WebSocketTransport,
)
from topodrome.site_agent.spool import EventSpool, SpoolEvent

pytestmark = pytest.mark.unit


def _seed_legacy_spool(
    path: Path,
    rows: list[tuple[int, str, str, str, float, float | None, float | None]],
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                acked_at REAL,
                abandoned_at REAL
            );
            """
        )
        connection.execute("INSERT INTO metadata(key, value) VALUES ('stream_id', 'legacy-stream')")
        connection.executemany(
            """
            INSERT INTO events(
                sequence, event_id, kind, payload_json, created_at, acked_at, abandoned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def test_heartbeat_embeds_its_durable_stream_coordinates(tmp_path: Path) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    event = spool.append_heartbeat(
        {
            "site_id": "site-1",
            "site_kind": "lab",
            "state": "ready",
            "gpus": [],
            "models": [],
        }
    )

    heartbeat = SiteHeartbeat.model_validate(event.payload)
    assert heartbeat.agent_session_id == spool.stream_id
    assert heartbeat.sequence == event.sequence
    spool.close()


def test_initialization_migrates_only_pending_legacy_heartbeats(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.sqlite3"
    pending_payload = json.dumps(
        {
            "agent_session_id": "legacy-stream",
            "cached_datasets": ["asset-a"],
            "sequence": 11,
        },
        indent=2,
    )
    acked_payload = '{"cached_datasets":["audit-acked"],"sequence":12}'
    abandoned_payload = '{"cached_datasets":["audit-abandoned"],"sequence":13}'
    nonheartbeat_payload = '{"cached_datasets":["not-a-heartbeat"]}'
    _seed_legacy_spool(
        path,
        [
            (11, "event-pending", "heartbeat", pending_payload, 1.0, None, None),
            (12, "event-acked", "heartbeat", acked_payload, 2.0, 20.0, None),
            (
                13,
                "event-abandoned",
                "heartbeat",
                abandoned_payload,
                3.0,
                None,
                30.0,
            ),
            (14, "event-other", "attempt_event", nonheartbeat_payload, 4.0, None, None),
        ],
    )

    spool = EventSpool(path)
    pending = spool.pending()
    assert [(event.sequence, event.event_id) for event in pending] == [
        (11, "event-pending"),
        (14, "event-other"),
    ]
    assert pending[0].payload == {
        "agent_session_id": "legacy-stream",
        "cached_assets": ["asset-a"],
        "sequence": 11,
    }
    assert pending[1].payload == {"cached_datasets": ["not-a-heartbeat"]}
    spool.close()

    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            """
            SELECT sequence, event_id, payload_json, created_at, acked_at, abandoned_at
              FROM events
             ORDER BY sequence
            """
        ).fetchall()
        marker = connection.execute(
            """
            SELECT value FROM metadata
             WHERE key = 'migration.pending-heartbeat-cached-assets.v1'
            """
        ).fetchone()
    finally:
        connection.close()

    assert [(row[0], row[1]) for row in rows] == [
        (11, "event-pending"),
        (12, "event-acked"),
        (13, "event-abandoned"),
        (14, "event-other"),
    ]
    assert json.loads(rows[0][2]) == {
        "agent_session_id": "legacy-stream",
        "cached_assets": ["asset-a"],
        "sequence": 11,
    }
    assert rows[1] == (12, "event-acked", acked_payload, 2.0, 20.0, None)
    assert rows[2] == (
        13,
        "event-abandoned",
        abandoned_payload,
        3.0,
        None,
        30.0,
    )
    assert rows[3] == (14, "event-other", nonheartbeat_payload, 4.0, None, None)
    assert marker == ("1",)


def test_pending_heartbeat_migration_rolls_back_payloads_and_marker_together(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.sqlite3"
    first_payload = '{"cached_datasets":["asset-a"],"sequence":1}'
    second_payload = '{"cached_datasets":["asset-b"],"sequence":2}'
    _seed_legacy_spool(
        path,
        [
            (1, "event-1", "heartbeat", first_payload, 1.0, None, None),
            (2, "event-2", "heartbeat", second_payload, 2.0, None, None),
        ],
    )
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TRIGGER reject_second_heartbeat_migration
            BEFORE UPDATE OF payload_json ON events
            WHEN OLD.event_id = 'event-2'
            BEGIN
                SELECT RAISE(ABORT, 'injected migration failure');
            END;
            """
        )
    finally:
        connection.close()

    with pytest.raises(sqlite3.IntegrityError, match="injected migration failure"):
        EventSpool(path)

    connection = sqlite3.connect(path)
    try:
        payloads = connection.execute(
            "SELECT payload_json FROM events ORDER BY sequence"
        ).fetchall()
        marker = connection.execute(
            """
            SELECT value FROM metadata
             WHERE key = 'migration.pending-heartbeat-cached-assets.v1'
            """
        ).fetchone()
    finally:
        connection.close()
    assert payloads == [(first_payload,), (second_payload,)]
    assert marker is None


class RecordingTransport:
    def __init__(self) -> None:
        self.connected = False
        self.sent: list[SpoolEvent] = []
        self.live: list[tuple[str, dict[str, Any]]] = []

    async def connect(self) -> None:
        self.connected = True

    async def send_event(self, event: SpoolEvent) -> int:
        assert self.connected
        self.sent.append(event)
        return event.sequence

    async def send_live(self, kind: str, payload: Mapping[str, Any]) -> None:
        assert self.connected
        self.live.append((kind, dict(payload)))

    async def commands(self) -> AsyncIterator[Mapping[str, Any]]:
        if False:
            yield {}

    async def close(self) -> None:
        self.connected = False


class DisconnectBeforeAckTransport(RecordingTransport):
    async def send_event(self, event: SpoolEvent) -> int:
        assert self.connected
        self.sent.append(event)
        raise ConnectionError("scheduler disconnected before its durable ACK")


@pytest.mark.asyncio
async def test_websocket_ack_correlates_scheduler_message_id() -> None:
    class Socket:
        def __init__(self) -> None:
            self._messages = iter(['{"kind":"ack","payload":{"message_id":"event-1"}}'])

        def __aiter__(self) -> Socket:
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._messages)
            except StopIteration:
                raise StopAsyncIteration from None

    transport = WebSocketTransport(
        "ws://scheduler.invalid",
        site_id="site-1",
        bearer_token=secrets.token_urlsafe(),
        agent_session_id="session-1",
    )
    transport._socket = Socket()  # type: ignore[assignment]
    waiter = asyncio.get_running_loop().create_future()
    transport._ack_waiters["event-1"] = (7, waiter)

    await transport._read_messages()

    assert await waiter == EventAcknowledgement(sequence=7, payload={"message_id": "event-1"})


@pytest.mark.asyncio
async def test_spool_replays_unacknowledged_events_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite3"
    first = EventSpool(path)
    event_one = first.append("heartbeat", {"free": 1})
    event_two = first.append("attempt_event", {"state": "running"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    first.close()

    reopened = EventSpool(path)
    stream_id = reopened.stream_id
    assert [event.sequence for event in reopened.pending()] == [
        event_one.sequence,
        event_two.sequence,
    ]
    transport = RecordingTransport()
    client = OutboundClient(reopened, lambda: transport)

    assert await client.flush() == 2
    assert [event.event_id for event in transport.sent] == [
        event_one.event_id,
        event_two.event_id,
    ]
    assert reopened.pending_count == 0
    reopened.close()

    final = EventSpool(path)
    assert final.stream_id == stream_id
    assert final.pending() == ()
    final.close()


@pytest.mark.asyncio
async def test_notebook_live_results_never_enter_the_durable_spool(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite3"
    spool = EventSpool(path)
    transport = RecordingTransport()
    client = OutboundClient(spool, lambda: transport)
    payload = {
        "operation_id": "operation-1",
        "result": {"stdout": "private notebook output"},
    }

    assert await client.publish_live("kernel_result", payload)
    assert transport.live == [("kernel_result", payload)]
    assert spool.pending_count == 0
    spool.close()

    reopened = EventSpool(path)
    assert reopened.pending() == ()
    reopened.close()


@pytest.mark.asyncio
async def test_confirmed_event_is_abandoned_after_disconnect_before_ack(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite3"
    spool = EventSpool(path)
    disconnected = DisconnectBeforeAckTransport()
    client = OutboundClient(spool, lambda: disconnected)

    with pytest.raises(ConnectionError, match="before its durable ACK"):
        await client.publish_confirmed("assignment_accepted", {"attempt_id": "attempt-1"})

    assert [event.kind for event in disconnected.sent] == ["assignment_accepted"]
    assert spool.pending_count == 0
    spool.close()

    # A reconnect cannot replay an acceptance whose caller chose not to execute.
    reopened = EventSpool(path)
    recording = RecordingTransport()
    replacement = OutboundClient(reopened, lambda: recording)
    assert await replacement.flush() == 0
    assert recording.sent == []
    reopened.close()


@pytest.mark.asyncio
async def test_correlated_request_replays_same_coordinates_after_lost_ack(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.sqlite3"
    spool = EventSpool(path)
    disconnected = DisconnectBeforeAckTransport()

    class GrantTransport(RecordingTransport):
        async def send_event(self, event: SpoolEvent) -> EventAcknowledgement:
            assert self.connected
            self.sent.append(event)
            return EventAcknowledgement(
                sequence=event.sequence,
                payload={
                    "message_id": event.event_id,
                    "duplicate": True,
                    "lease_renewal": {"fencing_token": "rotated"},
                },
            )

    recovered = GrantTransport()
    transports = iter((disconnected, recovered))
    client = OutboundClient(spool, lambda: next(transports))
    payload = {"attempt_id": "attempt-1", "fencing_token": "old"}

    with pytest.raises(ConnectionError, match="before its durable ACK"):
        await client.publish_request("lease-renewal:attempt-1", "lease_renewal", payload)
    assert spool.pending_count == 1

    ack = await client.publish_request("lease-renewal:attempt-1", "lease_renewal", payload)

    assert ack["duplicate"] is True
    assert ack["lease_renewal"] == {"fencing_token": "rotated"}
    assert recovered.sent[0].event_id == disconnected.sent[0].event_id
    assert recovered.sent[0].sequence == disconnected.sent[0].sequence
    assert spool.pending_count == 0
    spool.close()


def test_duplicate_event_id_is_idempotent_but_content_is_immutable(tmp_path: Path) -> None:
    spool = EventSpool(tmp_path / "events.sqlite3")
    first = spool.append("heartbeat", {"sequence": 1}, event_id="evt-1")
    duplicate = spool.append("heartbeat", {"sequence": 1}, event_id="evt-1")
    assert duplicate == first

    with pytest.raises(ValueError, match="different content"):
        spool.append("heartbeat", {"sequence": 2}, event_id="evt-1")
    spool.close()
