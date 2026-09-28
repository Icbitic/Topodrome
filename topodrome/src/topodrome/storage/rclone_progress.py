"""Drain bounded rclone JSON statistics without forwarding paths or diagnostics."""

import json
import math
import os
import selectors
import subprocess
import time

from topodrome.storage.progress import check_transfer_cancelled, transfer_progress
from topodrome.storage.rclone_environment import rclone_process_environment


def run_upload(command: list[str], *, timeout: float, size: int) -> subprocess.CompletedProcess:
    args = [*command, "--use-json-log", "--stats", "1s", "--stats-log-level", "NOTICE"]
    deadline = time.monotonic() + timeout
    with subprocess.Popen(  # noqa: S603 - executable and argv are built by the storage adapter
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=rclone_process_environment(),
    ) as process:
        assert process.stderr is not None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stderr, selectors.EVENT_READ)
                pending = b""
                discarding = False
                while selector.get_map():
                    check_transfer_cancelled()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    for key, _ in selector.select(min(remaining, 1)):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if not discarding and len(line) <= 65536:
                                _stat(line, size)
                            discarding = False
                        if len(pending) > 65536:
                            pending = b""
                            discarding = True
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    return subprocess.CompletedProcess(command, code)


def _stat(line: bytes, size: int) -> None:
    try:
        value = json.loads(line)
        current = value["stats"]["bytes"]
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            return
        if not math.isfinite(current) or current < 0:
            return
        transfer_progress("upload", min(int(current), size), size)
    except (ValueError, KeyError, TypeError, OverflowError):
        pass
