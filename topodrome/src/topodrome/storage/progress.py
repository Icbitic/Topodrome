"""Best-effort, thread/task-scoped storage telemetry; never a durability signal."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from threading import Event

from .base import StorageUnavailableError


class TransferCancelled(RuntimeError):
    """An explicit stop request, never a retryable transport failure."""


class TransferAuthorityExpired(StorageUnavailableError):
    """Pause I/O until upload-only authority has been renewed."""


TransferObserver = Callable[[str, int, int], None]
_observer: ContextVar[TransferObserver | None] = ContextVar("transfer_observer", default=None)
_cancellation: ContextVar[Event | None] = ContextVar("transfer_cancellation", default=None)
_authority: ContextVar[Callable[[], bool] | None] = ContextVar("transfer_authority", default=None)
_reuse_verified: ContextVar[bool] = ContextVar("reuse_verified_outputs", default=False)


@contextmanager
def reuse_verified_outputs() -> Iterator[None]:
    token = _reuse_verified.set(True)
    try:
        yield
    finally:
        _reuse_verified.reset(token)


def reusing_verified_outputs() -> bool:
    return _reuse_verified.get()


@contextmanager
def transfer_authority(check: Callable[[], bool]) -> Iterator[None]:
    token = _authority.set(check)
    try:
        yield
    finally:
        _authority.reset(token)


@contextmanager
def cancellable_transfers(event: Event) -> Iterator[None]:
    token = _cancellation.set(event)
    try:
        check_transfer_cancelled()
        yield
    finally:
        _cancellation.reset(token)


def check_transfer_cancelled() -> None:
    event = _cancellation.get()
    if event is not None and event.is_set():
        raise TransferCancelled("transfer canceled")
    authority = _authority.get()
    if authority is not None and not authority():
        raise TransferAuthorityExpired("waiting for finalization authority")


@contextmanager
def transfer_lock(lock) -> Iterator[None]:
    """A canceled waiter must not wait for another job's multi-GiB transfer."""
    while not lock.acquire(timeout=0.2):
        check_transfer_cancelled()
    try:
        check_transfer_cancelled()
        yield
    finally:
        lock.release()


@contextmanager
def observe_transfers(callback: TransferObserver) -> Iterator[None]:
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


def observing_transfers() -> bool:
    return _observer.get() is not None


def transfer_progress(stage: str, current: int, total: int) -> None:
    check_transfer_cancelled()
    callback = _observer.get()
    if callback is not None:
        # A broken UI/log sink must not turn a successful upload into a retry.
        with suppress(Exception):
            callback(stage, max(0, min(current, total)), max(0, total))
