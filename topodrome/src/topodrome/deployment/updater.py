"""Independent, single-writer update runner; never runs inside the scheduler.

Only operator-frozen commands run. Live mutations require the scheduler's atomic
idle claim. A crash in apply keeps admission fenced and requires explicit recovery.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import httpx

from topodrome.config import SchedulerSettings, _read_yaml
from topodrome.deployment.upgrade_store import UpgradeStore, _private_directory
from topodrome.domain.errors import ConflictError
from topodrome.domain.upgrades import UpgradeRecord, UpgradeStep


class UpgradeCommandFailed(RuntimeError):
    pass


class UpgradeRunner:
    def __init__(
        self,
        store: UpgradeStore,
        http: httpx.Client,
        execute: Callable[[UpgradeStep, Path], None] | None = None,
    ):
        self.store, self.http = store, http
        self.runner_id = uuid4().hex
        self.execute = execute or self._execute

    @staticmethod
    def _execute(step: UpgradeStep, release: Path) -> None:
        temporary = release / "tmp"
        _private_directory(temporary)
        cache = release.parent.parent / "uv-cache"
        _private_directory(cache)
        environment = {
            key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL") if key in os.environ
        }
        environment.update({"TMPDIR": str(temporary), "UV_CACHE_DIR": str(cache)})
        argv = [arg.replace("{release}", str(release)) for arg in step.argv]
        # No shell, credentials in argv, raw command output, or inherited
        # scheduler secrets. Scripts load their protected configuration directly.
        process = subprocess.Popen(  # noqa: S603 -- private frozen operator release, no shell
            argv,
            cwd=release,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=step.timeout_seconds)
        except BaseException:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            raise
        if code:
            raise UpgradeCommandFailed("upgrade step exited unsuccessfully")

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        _private_directory(self.store.root)
        fd = os.open(self.store.root / "runner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise PermissionError("unsafe update runner lock")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            os.close(fd)

    def _steps(self, record: UpgradeRecord, steps: tuple[UpgradeStep, ...]) -> None:
        self.store.validate_release(record)
        for step in steps:
            self.execute(step, self.store.release_path(record))

    def _post(self, record: UpgradeRecord, action: str) -> dict[str, Any]:
        response = self.http.post(
            f"/api/v1/upgrades/{record.upgrade_id}/{action}", json={"runner_id": self.runner_id}
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def _verify(self, record: UpgradeRecord) -> None:
        rollback = record.state == "verifying_rollback"
        self._steps(record, record.plan.verify_rollback if rollback else record.plan.verify)
        deadline = time.monotonic() + record.plan.health_timeout_seconds
        while True:
            try:
                if self._post(record, "complete").get("completed"):
                    return
            except (httpx.TransportError, httpx.HTTPStatusError):
                pass  # The scheduler may still be starting; this is not apply replay.
            if time.monotonic() >= deadline:
                raise TimeoutError("upgraded services did not become healthy")
            time.sleep(2)

    def run_once(self) -> bool:
        record = self.store.current()
        if record is None or record.state == "needs_attention":
            return False
        try:
            if record.state == "queued":
                record = self.store.change(
                    record.upgrade_id,
                    expected={"queued"},
                    state="preparing",
                    runner_id=self.runner_id,
                )
                self._steps(record, record.plan.prepare)
                record = self.store.change(
                    record.upgrade_id, expected={"preparing"}, state="waiting"
                )
            if record.state == "preparing":
                # Preparation is restricted to isolated release files. Do not
                # silently replay even this phase after losing its owner.
                self.store.change(
                    record.upgrade_id,
                    expected={"preparing"},
                    state="failed",
                    error_type="InterruptedPreparation",
                )
                return False
            if record.state == "waiting":
                if not self._post(record, "claim").get("claimed"):
                    return False
                record = self.store.get(record.upgrade_id)
            if record.state == "applying":
                if record.runner_id != self.runner_id:
                    self.store.change(
                        record.upgrade_id,
                        expected={"applying"},
                        state="needs_attention",
                        error_type="InterruptedApply",
                    )
                    return False
                self._steps(record, record.plan.apply)
                record = self.store.change(
                    record.upgrade_id, expected={"applying"}, state="verifying"
                )
            if record.state == "rolling_back":
                if record.runner_id not in {None, self.runner_id}:
                    self.store.change(
                        record.upgrade_id,
                        expected={"rolling_back"},
                        state="needs_attention",
                        error_type="InterruptedRollback",
                    )
                    return False
                record = self.store.change(
                    record.upgrade_id, expected={"rolling_back"}, runner_id=self.runner_id
                )
                self._steps(record, record.plan.rollback)
                record = self.store.change(
                    record.upgrade_id, expected={"rolling_back"}, state="verifying_rollback"
                )
            if record.state in {"verifying", "verifying_rollback"}:
                record = self.store.change(
                    record.upgrade_id, expected={record.state}, runner_id=self.runner_id
                )
                self._verify(record)
                return True
        except (httpx.TransportError, httpx.HTTPStatusError):
            # An unacknowledged idle claim is recovered by its runner ID. Never
            # interpret API unavailability as permission to deploy.
            return False
        except ConflictError:
            return False  # A safe pre-apply cancellation won the journal CAS.
        except Exception as exc:
            current = self.store.get(record.upgrade_id)
            if current.state in {"succeeded", "rolled_back", "canceled", "failed"}:
                return False
            if (
                current.mutation_started
                and current.plan.rollback
                and current.state in {"applying", "verifying"}
            ):
                self.store.change(
                    current.upgrade_id,
                    expected={current.state},
                    state="rolling_back",
                    error_type=type(exc).__name__,
                )
            else:
                self.store.change(
                    current.upgrade_id,
                    expected={current.state},
                    state="needs_attention" if current.mutation_started else "failed",
                    error_type=type(exc).__name__,
                )
            return False
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run queued idle-gated Topodrome upgrades")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    settings = SchedulerSettings(_env_file=args.env_file, **_read_yaml(args.config))
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    headers = (
        {}
        if settings.api_token is None
        else {"Authorization": "Bearer " + settings.api_token.get_secret_value()}
    )
    with httpx.Client(
        base_url=f"http://127.0.0.1:{settings.api_port}",
        headers=headers, timeout=30, trust_env=False
    ) as http:
        runner = UpgradeRunner(UpgradeStore(settings.global_root), http)
        with runner.exclusive():
            while not stop.is_set():
                runner.run_once()
                if args.once:
                    return
                stop.wait(5)


if __name__ == "__main__":
    main()
