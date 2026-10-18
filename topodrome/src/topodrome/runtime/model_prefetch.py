"""Best-effort bounded read-ahead of already registered, local checkpoint files.

There are no downloads, CUDA calls, persistent copies, or readiness claims here.
The kernel manages the reclaimable file cache. Application buffers stay bounded
to 8 MiB per reader; readers are capped per site, not per queued child.
"""

from __future__ import annotations

import os
import stat
import threading
import time

from .model_payload import verify_transformers_payload


class ModelPrefetcher:
    def __init__(
        self,
        *,
        max_bytes=16 * 1024**3,
        concurrency=1,
        memory_floor_bytes=16 * 1024**3,
        memory_available=None,
        deadline_seconds=120,
    ):
        if not 0 <= max_bytes <= 64 * 1024**3 or not 1 <= concurrency <= 4:
            raise ValueError("invalid checkpoint read-ahead bounds")
        if memory_floor_bytes < 0 or deadline_seconds <= 0:
            raise ValueError("invalid checkpoint read-ahead memory/deadline")
        self.max_bytes = max_bytes
        self.memory_floor_bytes = memory_floor_bytes
        self.memory_available = memory_available or (lambda: 0)
        self.deadline_seconds = deadline_seconds
        self._slots = threading.BoundedSemaphore(concurrency)
        self._lock = threading.Lock()
        self._active = {}
        self._recent = {}
        self._closed = False
        self.bytes_read = 0
        self.failures = 0

    def schedule(self, registration):
        if (
            not self.max_bytes
            or registration.local_path is None
            or not registration.payload_identity
        ):
            return False
        key = registration.key
        with self._lock:
            now = time.monotonic()
            self._recent = {name: stamp for name, stamp in self._recent.items() if now - stamp < 60}
            if self._closed or key in self._active or key in self._recent:
                return False
            if not self._slots.acquire(blocking=False):
                return False
            canceled = threading.Event()
            self._active[key] = canceled
        threading.Thread(
            target=self._read,
            args=(registration, canceled),
            name="topodrome-checkpoint-read-ahead",
            daemon=True,
        ).start()
        return True

    def _read(self, registration, canceled):
        key = registration.key
        read_bytes = 0
        try:
            if self.memory_available() < self.memory_floor_bytes:
                return
            verified = verify_transformers_payload(
                registration.local_path,
                revision=key.revision,
                subfolder=key.subfolder,
            )
            if (
                verified.identity_digest != registration.payload_identity
                or verified.weight_bytes > self.max_bytes
            ):
                return
            root = registration.local_path
            if key.subfolder:
                root = root / key.subfolder
            deadline = time.monotonic() + self.deadline_seconds
            buffer = bytearray(8 * 1024**2)
            for name in verified.weight_files:
                descriptor = os.open(root / name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb", buffering=0) as source:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        return
                    while read_bytes < self.max_bytes:
                        if (
                            canceled.is_set()
                            or time.monotonic() >= deadline
                            or self.memory_available() < self.memory_floor_bytes
                        ):
                            return
                        size = source.readinto(
                            memoryview(buffer)[: min(len(buffer), self.max_bytes - read_bytes)]
                        )
                        if not size:
                            break
                        read_bytes += size
        except Exception:
            # A speculative read never fails useful computation or publishes a
            # cached capability. The ordinary loader retains all verification.
            with self._lock:
                self.failures += 1
        finally:
            with self._lock:
                self.bytes_read += read_bytes
                self._active.pop(key, None)
                if read_bytes:
                    self._recent[key] = time.monotonic()
            self._slots.release()

    def close(self):
        with self._lock:
            self._closed = True
            for canceled in self._active.values():
                canceled.set()

    @property
    def active(self):
        with self._lock:
            return len(self._active)
