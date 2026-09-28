"""Durable, ordered buffering of site-agent status events."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PENDING_HEARTBEAT_ASSET_MIGRATION_KEY = "migration.pending-heartbeat-cached-assets.v1"


@dataclass(frozen=True, slots=True)
class SpoolEvent:
    sequence: int
    event_id: str
    kind: str
    payload: Mapping[str, Any]
    created_at: float

    def as_message(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "created_at": self.created_at,
        }


class EventSpool:
    """An append-only SQLite queue with cumulative acknowledgements.

    Events are committed with ``synchronous=FULL`` before callers may report
    success.  Acknowledged events stay available for a bounded audit period and
    can be pruned explicitly.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                acked_at REAL,
                abandoned_at REAL
            );
            CREATE INDEX IF NOT EXISTS ix_events_pending
                ON events(sequence) WHERE acked_at IS NULL;
            """
        )
        # Existing agent spools predate abandoned events. Keep this local
        # migration idempotent so an upgrade never requires deleting the
        # persist-before-send audit trail.
        columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(events)").fetchall()
        }
        if "abandoned_at" not in columns:
            self._connection.execute("ALTER TABLE events ADD COLUMN abandoned_at REAL")
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_events_deliverable
                ON events(sequence)
             WHERE acked_at IS NULL AND abandoned_at IS NULL
            """
        )
        _migrate_pending_heartbeat_cached_assets(self._connection)
        self._connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES ('stream_id', ?)",
            (str(uuid.uuid4()),),
        )
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def append(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        event_id: str | None = None,
        created_at: float | None = None,
    ) -> SpoolEvent:
        if not kind:
            raise ValueError("event kind is required")
        identifier = event_id or str(uuid.uuid4())
        timestamp = time.time() if created_at is None else created_at
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._lock:
            try:
                cursor = self._connection.execute(
                    """
                    INSERT INTO events(event_id, kind, payload_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (identifier, kind, encoded, timestamp),
                )
            except sqlite3.IntegrityError:
                row = self._connection.execute(
                    "SELECT * FROM events WHERE event_id = ?", (identifier,)
                ).fetchone()
                if row is None:
                    raise
                if row["kind"] != kind or row["payload_json"] != encoded:
                    raise ValueError(
                        f"event id {identifier} was reused with different content"
                    ) from None
                return _event_from_row(row)
            if cursor.lastrowid is None:  # pragma: no cover - SQLite invariant
                raise RuntimeError("SQLite did not assign an event sequence")
            sequence = int(cursor.lastrowid)
        return SpoolEvent(sequence, identifier, kind, dict(payload), timestamp)

    def append_heartbeat(
        self,
        payload: Mapping[str, Any],
        *,
        created_at: float | None = None,
    ) -> SpoolEvent:
        """Append a heartbeat whose embedded stream coordinates are atomic.

        ``SiteHeartbeat`` deliberately repeats the durable envelope's session
        and sequence.  Both values are derived inside the same SQLite write
        transaction so a crash can never leave a replayable heartbeat with a
        mismatched inner sequence.
        """

        identifier = str(uuid.uuid4())
        timestamp = time.time() if created_at is None else created_at
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO events(event_id, kind, payload_json, created_at)
                    VALUES (?, 'heartbeat', '{}', ?)
                    """,
                    (identifier, timestamp),
                )
                if cursor.lastrowid is None:  # pragma: no cover - SQLite invariant
                    raise RuntimeError("SQLite did not assign an event sequence")
                sequence = int(cursor.lastrowid)
                stream_row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'stream_id'"
                ).fetchone()
                if stream_row is None:  # pragma: no cover - schema invariant
                    raise RuntimeError("event spool has no stream identity")
                enriched = {
                    **dict(payload),
                    "agent_session_id": str(stream_row[0]),
                    "sequence": sequence,
                }
                encoded = json.dumps(enriched, sort_keys=True, separators=(",", ":"))
                connection.execute(
                    "UPDATE events SET payload_json = ? WHERE sequence = ?",
                    (encoded, sequence),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return SpoolEvent(sequence, identifier, "heartbeat", enriched, timestamp)

    def pending(self, *, limit: int = 1000) -> tuple[SpoolEvent, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT sequence, event_id, kind, payload_json, created_at
                  FROM events
                 WHERE acked_at IS NULL AND abandoned_at IS NULL
                 ORDER BY sequence
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def acknowledge_through(self, sequence: int, *, acknowledged_at: float | None = None) -> int:
        if sequence < 0:
            raise ValueError("sequence cannot be negative")
        timestamp = time.time() if acknowledged_at is None else acknowledged_at
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE events SET acked_at = ?
                 WHERE sequence <= ? AND acked_at IS NULL AND abandoned_at IS NULL
                """,
                (timestamp, sequence),
            )
        return int(cursor.rowcount)

    def abandon(self, sequence: int, *, abandoned_at: float | None = None) -> bool:
        """Permanently suppress one event that never received a durable ACK.

        This is intentionally not cumulative: earlier status remains replayable.
        It is used for offer acceptance when an ambiguous disconnect means the
        site must choose safety and refrain from executing that assignment.
        """

        if sequence < 1:
            raise ValueError("sequence must be positive")
        timestamp = time.time() if abandoned_at is None else abandoned_at
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE events SET abandoned_at = ?
                 WHERE sequence = ? AND acked_at IS NULL AND abandoned_at IS NULL
                """,
                (timestamp, sequence),
            )
        return bool(cursor.rowcount)

    def prune_acknowledged(self, *, before: float) -> int:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM events WHERE acked_at IS NOT NULL AND acked_at < ?",
                (before,),
            )
        return int(cursor.rowcount)

    @property
    def pending_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) FROM events
                 WHERE acked_at IS NULL AND abandoned_at IS NULL
                """
            ).fetchone()
        return int(row[0])

    @property
    def stream_id(self) -> str:
        """Stable identity for this replayable event sequence."""

        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'stream_id'"
            ).fetchone()
        if row is None:  # pragma: no cover - protected by schema initialization
            raise RuntimeError("event spool has no stream identity")
        return str(row[0])


def _migrate_pending_heartbeat_cached_assets(connection: sqlite3.Connection) -> None:
    """Rewrite the one legacy heartbeat field while events are still offline.

    Only deliverable heartbeats need to satisfy the current wire schema.
    Acknowledged and deliberately abandoned rows are immutable audit history,
    so initialization never rewrites them.  The transaction also records the
    migration marker, making the scan both atomic and one-time.
    """

    connection.execute("BEGIN IMMEDIATE")
    try:
        completed = connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (_PENDING_HEARTBEAT_ASSET_MIGRATION_KEY,),
        ).fetchone()
        if completed is not None:
            connection.execute("COMMIT")
            return

        rows = connection.execute(
            """
            SELECT sequence, payload_json
              FROM events
             WHERE kind = 'heartbeat'
               AND acked_at IS NULL
               AND abandoned_at IS NULL
             ORDER BY sequence
            """
        ).fetchall()
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            if not isinstance(payload, dict):
                raise ValueError("pending heartbeat payload must be a JSON object")
            if "cached_datasets" not in payload:
                continue
            legacy_assets = payload.pop("cached_datasets")
            # A current field, if present, is authoritative. This can happen
            # only in a partially upgraded producer and avoids overwriting a
            # value already written under the current protocol.
            payload.setdefault("cached_assets", legacy_assets)
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            cursor = connection.execute(
                """
                UPDATE events
                   SET payload_json = ?
                 WHERE sequence = ?
                   AND kind = 'heartbeat'
                   AND acked_at IS NULL
                   AND abandoned_at IS NULL
                """,
                (encoded, int(row["sequence"])),
            )
            if cursor.rowcount != 1:  # pragma: no cover - held write lock invariant
                raise RuntimeError("pending heartbeat changed during spool migration")

        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, '1')",
            (_PENDING_HEARTBEAT_ASSET_MIGRATION_KEY,),
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def _event_from_row(row: sqlite3.Row) -> SpoolEvent:
    return SpoolEvent(
        sequence=int(row["sequence"]),
        event_id=str(row["event_id"]),
        kind=str(row["kind"]),
        payload=json.loads(row["payload_json"]),
        created_at=float(row["created_at"]),
    )


__all__ = ["EventSpool", "SpoolEvent"]
