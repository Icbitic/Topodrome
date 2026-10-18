"""Outbound-only scheduler control transport.

The orchestration layer depends on :class:`AgentTransport`, not on a concrete
WebSocket implementation.  This makes status replay deterministic and keeps
the optional networking dependency out of local tests.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from .spool import EventSpool, SpoolEvent


@dataclass(frozen=True, slots=True)
class EventAcknowledgement:
    sequence: int
    payload: Mapping[str, Any]


class AgentTransport(Protocol):
    async def connect(self) -> None: ...

    async def send_event(self, event: SpoolEvent) -> EventAcknowledgement | int: ...

    async def send_live(
        self, kind: str, payload: Mapping[str, Any]
    ) -> EventAcknowledgement | Mapping[str, Any] | None: ...

    def commands(self) -> AsyncIterator[Mapping[str, Any]]: ...

    async def close(self) -> None: ...


TransportFactory = Callable[[], AgentTransport]


class OutboundClient:
    """Persist-before-send event publisher with ordered cumulative ACKs."""

    def __init__(
        self,
        spool: EventSpool,
        transport_factory: TransportFactory,
        *,
        heartbeat_ack: Callable[[SpoolEvent, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self._spool = spool
        self._transport_factory = transport_factory
        self._transport: AgentTransport | None = None
        self._flush_lock = asyncio.Lock()
        self._request_events: dict[str, SpoolEvent] = {}
        self._request_results: dict[str, Mapping[str, Any]] = {}
        self.heartbeat_ack = heartbeat_ack

    def enqueue(self, kind: str, payload: Mapping[str, Any]) -> SpoolEvent:
        return self._spool.append(kind, payload)

    @property
    def agent_session_id(self) -> str:
        return self._spool.stream_id

    async def connect(self) -> None:
        if self._transport is None:
            transport = self._transport_factory()
            await transport.connect()
            self._transport = transport

    async def flush(self, *, batch_size: int = 256) -> int:
        """Replay pending events in sequence and return the number acknowledged."""

        async with self._flush_lock:
            await self.connect()
            assert self._transport is not None
            acknowledged = 0
            while True:
                batch = self._spool.pending(limit=batch_size)
                if not batch:
                    return acknowledged
                for event in batch:
                    ack = _normalise_ack(await self._transport.send_event(event))
                    ack_sequence = ack.sequence
                    if ack_sequence != event.sequence:
                        raise RuntimeError(
                            f"scheduler acknowledged {ack_sequence} for sent event "
                            f"{event.sequence}; the ordered stream is inconsistent"
                        )
                    self._remember_request_result(event, ack.payload)
                    acknowledged += self._spool.acknowledge_through(ack_sequence)

    async def confirm(self, event: SpoolEvent, *, batch_size: int = 256) -> None:
        """Deliver through ``event`` and require its correlated scheduler ACK.

        Unlike :meth:`flush`, this stops immediately after the target is
        acknowledged. A later event cannot turn a committed acceptance into an
        apparent failure.
        """

        async with self._flush_lock:
            await self.connect()
            assert self._transport is not None
            while True:
                pending = tuple(
                    item
                    for item in self._spool.pending(limit=batch_size)
                    if item.sequence <= event.sequence
                )
                if not pending:
                    return
                for item in pending:
                    ack = _normalise_ack(await self._transport.send_event(item))
                    ack_sequence = ack.sequence
                    if ack_sequence != item.sequence:
                        raise RuntimeError(
                            f"scheduler acknowledged {ack_sequence} for sent event "
                            f"{item.sequence}; the ordered stream is inconsistent"
                        )
                    self._remember_request_result(item, ack.payload)
                    self._spool.acknowledge_through(ack_sequence)
                    if item.sequence == event.sequence:
                        return

    async def publish(self, kind: str, payload: Mapping[str, Any]) -> SpoolEvent:
        event = self.enqueue(kind, payload)
        await self._flush_best_effort()
        return event

    async def publish_request(
        self,
        request_key: str,
        kind: str,
        payload: Mapping[str, Any],
        *,
        batch_size: int = 256,
    ) -> Mapping[str, Any]:
        """Durably retry one correlated request until its ACK payload is known.

        Unlike assignment acceptance, an ambiguous renewal must be replayed
        with the same message id: the scheduler may already have rotated its
        digest.  The pending spool record and its stable stream coordinates
        make that replay deterministic.  Earlier status is delivered first so
        no old-token event can remain queued behind a successful rotation.
        """

        event = self._request_events.get(request_key)
        if event is None:
            event = self.enqueue(kind, payload)
            self._request_events[request_key] = event
        elif event.kind != kind or dict(event.payload) != dict(payload):
            raise RuntimeError("correlated request payload changed before acknowledgement")

        try:
            async with self._flush_lock:
                await self.connect()
                assert self._transport is not None
                while event.event_id not in self._request_results:
                    pending = tuple(
                        item
                        for item in self._spool.pending(limit=batch_size)
                        if item.sequence <= event.sequence
                    )
                    if not pending:
                        break
                    for item in pending:
                        ack = _normalise_ack(await self._transport.send_event(item))
                        if ack.sequence != item.sequence:
                            raise RuntimeError(
                                f"scheduler acknowledged {ack.sequence} for sent event "
                                f"{item.sequence}; the ordered stream is inconsistent"
                            )
                        self._remember_request_result(item, ack.payload)
                        self._spool.acknowledge_through(ack.sequence)
                        if item.sequence == event.sequence:
                            break
                result = self._request_results.pop(event.event_id, None)
                if result is None:
                    raise RuntimeError("scheduler ACK omitted the correlated response payload")
                self._request_events.pop(request_key, None)
                return result
        except BaseException:
            # Preserve both the request coordinates and the unacknowledged row.
            # A retry after an ACK-lost disconnect must reconstruct the exact
            # same fencing grant rather than submit a stale new renewal.
            with suppress(Exception):
                await self.disconnect()
            raise

    async def publish_confirmed(self, kind: str, payload: Mapping[str, Any]) -> SpoolEvent:
        """Publish one safety-critical event only if the scheduler durably ACKs it.

        A failed or ambiguous send is retained in the audit history but removed
        from the replay queue. Replaying acceptance after its handler has given
        up could otherwise make the scheduler believe an unexecuted lease is
        active.
        """

        event = self.enqueue(kind, payload)
        try:
            await self.confirm(event)
        except BaseException:
            self._spool.abandon(event.sequence)
            with suppress(Exception):
                await self.disconnect()
            raise
        return event

    async def publish_live(self, kind: str, payload: Mapping[str, Any]) -> bool:
        """Send non-replayable interactive traffic without writing the spool."""

        return await self.publish_live_request(kind, payload) is not None

    async def publish_live_request(
        self, kind: str, payload: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        """Send an offset-idempotent live event and return its correlated ACK.

        The event itself is deliberately absent from the durable SQLite spool.
        Callers that need retry semantics retain their byte offset and resend
        that same range until this method returns an accepting response.
        """

        try:
            await self.connect()
            assert self._transport is not None
            acknowledgement = await self._transport.send_live(kind, payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A cell result is intentionally ephemeral.  Persisting and
            # replaying notebook traffic could duplicate mutable execution.
            with suppress(Exception):
                await self.disconnect()
            return None
        if acknowledgement is None:
            # Compatibility with injected transports written before live ACK
            # payloads existed. A successful send is still a successful live
            # event, but it has no protocol-specific coordinates.
            return {}
        if isinstance(acknowledgement, EventAcknowledgement):
            return dict(acknowledgement.payload)
        return dict(acknowledgement)

    async def publish_heartbeat(self, payload: Mapping[str, Any]) -> SpoolEvent:
        """Persist a heartbeat with coordinates matching its outer envelope."""

        event = self._spool.append_heartbeat(payload)
        await self._flush_best_effort()
        return event

    async def _flush_best_effort(self) -> None:
        try:
            await self.flush()
        except asyncio.CancelledError:
            raise
        except Exception:
            # The event is already durable. A broken socket, malformed ACK, or
            # transient protocol deployment mismatch must not kill heartbeat
            # and lease-renewal production; the control loop will reconnect.
            with suppress(Exception):
                await self.disconnect()

    async def commands(self) -> AsyncIterator[Mapping[str, Any]]:
        await self.connect()
        assert self._transport is not None
        async for command in self._transport.commands():
            yield command

    async def disconnect(self) -> None:
        transport, self._transport = self._transport, None
        if transport is not None:
            await transport.close()

    def _remember_request_result(
        self, event: SpoolEvent, payload: Mapping[str, Any]
    ) -> None:
        if event.kind == "heartbeat" and self.heartbeat_ack is not None:
            self.heartbeat_ack(event, payload)
        if any(item.event_id == event.event_id for item in self._request_events.values()):
            self._request_results[event.event_id] = dict(payload)


class ReconnectingControlLoop:
    """Run an outbound command channel with bounded exponential backoff."""

    def __init__(
        self,
        client: OutboundClient,
        handler: Callable[[Mapping[str, Any]], Awaitable[None]],
        *,
        initial_backoff: float = 0.5,
        maximum_backoff: float = 30.0,
    ) -> None:
        self.client = client
        self.handler = handler
        self.initial_backoff = initial_backoff
        self.maximum_backoff = maximum_backoff

    async def run(self, stop: asyncio.Event) -> None:
        delay = self.initial_backoff
        while not stop.is_set():
            try:
                await self.client.flush()
                async for command in self.client.commands():
                    await self.handler(command)
                    if stop.is_set():
                        break
                delay = self.initial_backoff
            except asyncio.CancelledError:
                raise
            except Exception:
                await self.client.disconnect()
                jittered = delay * secrets.SystemRandom().uniform(0.8, 1.2)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=jittered)
                except TimeoutError:
                    pass
                delay = min(self.maximum_backoff, max(self.initial_backoff, delay * 2))
        await self.client.disconnect()


class WebSocketTransport:
    """Minimal JSON WebSocket transport, imported only on an enrolled site."""

    def __init__(
        self,
        url: str,
        *,
        site_id: str,
        bearer_token: str,
        agent_session_id: str,
        open_timeout: float = 15.0,
        acknowledgement_timeout: float = 30.0,
    ) -> None:
        self.url = url
        self.site_id = site_id
        self.agent_session_id = agent_session_id
        self._bearer_token = bearer_token
        self.open_timeout = open_timeout
        self.acknowledgement_timeout = acknowledgement_timeout
        self._socket: Any = None
        self._reader_task: asyncio.Task[None] | None = None
        self._ack_waiters: dict[
            str, tuple[int, asyncio.Future[EventAcknowledgement]]
        ] = {}
        self._commands: asyncio.Queue[Mapping[str, Any] | None] = asyncio.Queue()
        self._send_lock = asyncio.Lock()
        self._reader_error: BaseException | None = None

    async def connect(self) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "WebSocket control requires the optional 'websockets' package"
            ) from exc
        self._socket = await connect(
            self.url,
            additional_headers={
                "Authorization": f"Bearer {self._bearer_token}",
                "X-Topodrome-Site": self.site_id,
            },
            open_timeout=self.open_timeout,
        )
        self._reader_error = None
        self._reader_task = asyncio.create_task(
            self._read_messages(), name=f"topodrome-ws-reader-{self.site_id}"
        )

    async def send_event(self, event: SpoolEvent) -> EventAcknowledgement:
        if self._socket is None:
            raise ConnectionError("transport is not connected")
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._ack_waiters[event.event_id] = (event.sequence, waiter)
        message = {
            "message_id": event.event_id,
            "site_id": self.site_id,
            "agent_session_id": self.agent_session_id,
            "sequence": event.sequence,
            "kind": event.kind,
            "payload": dict(event.payload),
            "sent_at": datetime.fromtimestamp(event.created_at, tz=UTC).isoformat(),
        }
        try:
            async with self._send_lock:
                await self._socket.send(json.dumps(message, separators=(",", ":")))
            return await asyncio.wait_for(waiter, timeout=self.acknowledgement_timeout)
        except BaseException:
            self._ack_waiters.pop(event.event_id, None)
            if not waiter.done():
                waiter.cancel()
            raise

    async def send_live(
        self, kind: str, payload: Mapping[str, Any]
    ) -> EventAcknowledgement:
        if self._socket is None:
            raise ConnectionError("transport is not connected")
        event_id = f"live_{secrets.token_urlsafe(18)}"
        waiter: asyncio.Future[EventAcknowledgement] = (
            asyncio.get_running_loop().create_future()
        )
        self._ack_waiters[event_id] = (0, waiter)
        message = {
            "message_id": event_id,
            "site_id": self.site_id,
            "agent_session_id": self.agent_session_id,
            "sequence": 0,
            "kind": kind,
            "payload": dict(payload),
            "sent_at": datetime.now(UTC).isoformat(),
        }
        try:
            async with self._send_lock:
                await self._socket.send(json.dumps(message, separators=(",", ":")))
            return await asyncio.wait_for(waiter, timeout=self.acknowledgement_timeout)
        except BaseException:
            self._ack_waiters.pop(event_id, None)
            if not waiter.done():
                waiter.cancel()
            raise

    async def commands(self) -> AsyncIterator[Mapping[str, Any]]:
        if self._socket is None:
            raise ConnectionError("transport is not connected")
        while True:
            message = await self._commands.get()
            if message is None:
                error = self._reader_error
                if error is not None:
                    raise ConnectionError("scheduler control channel closed") from error
                return
            yield message

    async def _read_messages(self) -> None:
        assert self._socket is not None
        try:
            async for raw_message in self._socket:
                message = json.loads(raw_message)
                kind = message.get("kind") or message.get("type")
                if kind == "ack":
                    payload = message.get("payload")
                    if isinstance(payload, Mapping) and payload.get("message_id") is not None:
                        event_id = str(payload["message_id"])
                        pending = self._ack_waiters.pop(event_id, None)
                    else:
                        # Backward-compatible support for sequence-based ACKs.
                        raw_sequence = (
                            payload.get("sequence")
                            if isinstance(payload, Mapping)
                            else message.get("sequence")
                        )
                        if raw_sequence is None:
                            raise ValueError("scheduler ACK has no message_id or sequence")
                        sequence = int(raw_sequence)
                        event_id, pending = next(
                            (
                                (identifier, value)
                                for identifier, value in self._ack_waiters.items()
                                if value[0] == sequence
                            ),
                            ("", None),
                        )
                        if pending is not None:
                            self._ack_waiters.pop(event_id, None)
                    if pending is not None:
                        ack_sequence, waiter = pending
                        if not waiter.done():
                            waiter.set_result(
                                EventAcknowledgement(
                                    sequence=ack_sequence,
                                    payload=dict(payload) if isinstance(payload, Mapping) else {},
                                )
                            )
                else:
                    await self._commands.put(message)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._reader_error = exc
        finally:
            error = self._reader_error or ConnectionError("scheduler control channel closed")
            for _, waiter in self._ack_waiters.values():
                if not waiter.done():
                    waiter.set_exception(error)
            self._ack_waiters.clear()
            await self._commands.put(None)

    async def close(self) -> None:
        if self._socket is not None:
            socket, self._socket = self._socket, None
            await socket.close()
        if self._reader_task is not None:
            task, self._reader_task = self._reader_task, None
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


__all__ = [
    "AgentTransport",
    "EventAcknowledgement",
    "OutboundClient",
    "ReconnectingControlLoop",
    "TransportFactory",
    "WebSocketTransport",
]


def _normalise_ack(value: EventAcknowledgement | int) -> EventAcknowledgement:
    if isinstance(value, EventAcknowledgement):
        return value
    return EventAcknowledgement(sequence=int(value), payload={})
