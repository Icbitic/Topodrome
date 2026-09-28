"""Publish finalization progress through the existing fenced live-log channel."""

import time
from contextlib import suppress
from pathlib import Path

from topodrome.domain.progress import PROGRESS_PREFIX, JobProgress


class TransferLogReporter:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.started = time.monotonic()
        self.last_at = float("-inf")
        self.last_stage = ""

    def __call__(self, stage: str, current: int = 0, total: int = 0) -> None:
        now = time.monotonic()
        if stage == self.last_stage and now - self.last_at < 2:
            return
        self.last_at, self.last_stage = now, stage
        record = JobProgress(
            current=current,
            total=total or None,
            stage=stage,
            unit="bytes",
            elapsed_seconds=max(0, now - self.started),
        )
        with suppress(Exception):
            with self.log_path.open("a", encoding="utf-8") as output:
                output.write("\n" + PROGRESS_PREFIX + record.model_dump_json() + "\n")
