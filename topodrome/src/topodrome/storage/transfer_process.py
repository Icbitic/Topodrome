"""Cancelable subprocess.run equivalent for owned, bounded storage commands."""

import subprocess
import time
from collections.abc import Callable

from .progress import check_transfer_cancelled


def run_transfer(
    args,
    *,
    timeout: float,
    check: bool = False,
    progress: Callable[[], None] | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    check_transfer_cancelled()
    deadline = time.monotonic() + timeout
    # Retain the Popen handle until wait(): its PID cannot be reused while
    # unreaped. Never signal a name, process group, or an unrelated PID.
    with subprocess.Popen(args, **kwargs) as child:  # noqa: S603
        try:
            while True:
                check_transfer_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, timeout)
                try:
                    stdout, stderr = child.communicate(timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if progress is not None:
                        progress()
            check_transfer_cancelled()
            if progress is not None:
                progress()
            result = subprocess.CompletedProcess(args, child.returncode, stdout, stderr)
            if check:
                result.check_returncode()
            return result
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
