"""Deterministic cleanup primitives for experiment-scoped mutations."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class CleanupFailure:
    callback: str
    error: BaseException


class CleanupError(RuntimeError):
    """Raised after all cleanup callbacks ran and at least one failed."""

    def __init__(self, failures: tuple[CleanupFailure, ...]) -> None:
        self.failures = failures
        names = ", ".join(failure.callback for failure in failures)
        super().__init__(f"{len(failures)} cleanup callback(s) failed: {names}")


@dataclass(slots=True)
class _Callback:
    function: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    active: bool = True


class CleanupHandle:
    """Allows a registered cleanup to be disarmed after manual restoration."""

    __slots__ = ("_callback", "_lock")

    def __init__(self, callback: _Callback, lock: RLock) -> None:
        self._callback = callback
        self._lock = lock

    def cancel(self) -> None:
        with self._lock:
            self._callback.active = False


def _callback_name(callback: Callable[..., Any]) -> str:
    return getattr(callback, "__qualname__", getattr(callback, "__name__", type(callback).__name__))


async def _await_cleanup(value: Awaitable[Any]) -> None:
    await value


def _wait_sync(value: Any) -> None:
    if not inspect.isawaitable(value):
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_await_cleanup(value))
        return
    close = getattr(value, "close", None)
    if close is not None:
        close()
    raise RuntimeError(
        "async cleanup cannot be awaited by close() inside a running event loop; use await aclose()"
    )


class CleanupStack:
    """Small LIFO cleanup stack that always attempts every callback."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._callbacks: list[_Callback] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def push(self, callback: Callable[..., Any], /, *args: Any, **kwargs: Any) -> CleanupHandle:
        if not callable(callback):
            raise TypeError("cleanup callback must be callable")
        item = _Callback(callback, args, dict(kwargs))
        with self._lock:
            if self._closed:
                raise RuntimeError("cleanup stack is already closed")
            self._callbacks.append(item)
        return CleanupHandle(item, self._lock)

    def _take_callbacks(self) -> list[_Callback]:
        with self._lock:
            if self._closed:
                return []
            self._closed = True
            callbacks = list(reversed(self._callbacks))
            self._callbacks.clear()
            return callbacks

    def close(self) -> None:
        failures: list[CleanupFailure] = []
        for item in self._take_callbacks():
            if not item.active:
                continue
            try:
                _wait_sync(item.function(*item.args, **item.kwargs))
            except BaseException as exc:  # all callbacks must still be attempted
                failures.append(CleanupFailure(_callback_name(item.function), exc))
        if failures:
            raise CleanupError(tuple(failures))

    async def aclose(self) -> None:
        failures: list[CleanupFailure] = []
        for item in self._take_callbacks():
            if not item.active:
                continue
            try:
                result = item.function(*item.args, **item.kwargs)
                if isinstance(result, Awaitable) or inspect.isawaitable(result):
                    await result
            except BaseException as exc:  # all callbacks must still be attempted
                failures.append(CleanupFailure(_callback_name(item.function), exc))
        if failures:
            raise CleanupError(tuple(failures))
