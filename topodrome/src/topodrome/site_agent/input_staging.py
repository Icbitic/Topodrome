"""Bounded site-local input preparation. No Ray actor or GPU lease is involved."""

import asyncio
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime

from topodrome.domain.progress import JobProgress
from topodrome.domain.specs import AssetRef, InputStageMetrics
from topodrome.storage.progress import TransferCancelled, cancellable_transfers, observe_transfers


@dataclass
class InputStage:
    job_id: str
    stage_id: str
    assets: dict[str, AssetRef]
    expires_at: float
    stop: threading.Event = field(default_factory=threading.Event)
    state: str = "downloading"
    progress: dict = field(
        default_factory=lambda: JobProgress(
            current=0, stage="downloading", unit="bytes"
        ).model_dump(mode="json")
    )
    task: asyncio.Task | None = None
    source_ref: str | None = None
    checkpoint_ref: str | None = None
    metrics: dict = field(default_factory=lambda: InputStageMetrics().model_dump())


class InputStager:
    def __init__(
        self, resolver, outbound, *, source_resolver=None, checkpoint_resolver=None, capacity=2
    ):
        self.resolver = resolver
        self.outbound = outbound
        self.capacity = capacity
        self.source_resolver = source_resolver
        self.checkpoint_resolver = checkpoint_resolver
        self.stages: dict[str, InputStage] = {}

    async def accept(self, payload):
        job_id, stage_id = str(payload["job_id"]), str(payload["stage_id"])
        assets = {str(k): AssetRef.model_validate(v) for k, v in payload["assets"].items()}
        if len(assets) > 128 or any(not a.pinned for a in assets.values()):
            raise ValueError("invalid staging input pins")
        # Relative TTL avoids comparing clocks on independent hosts. The central
        # row's own deadline still fences all late reports and dispatch.
        ttl = payload.get("ttl_seconds")
        if ttl is not None and (not isinstance(ttl, int) or not 1 <= ttl <= 300):
            raise ValueError("invalid staging TTL")
        expiry = (
            time.time() + ttl
            if ttl is not None
            else datetime.fromisoformat(payload["expires_at"]).timestamp()
        )
        if not time.time() < expiry <= time.time() + 300:
            raise ValueError("invalid staging deadline")
        previous = self.stages.get(job_id)
        source_ref = payload.get("source_ref")
        checkpoint_ref = payload.get("checkpoint_ref")
        if source_ref is not None and not isinstance(source_ref, str):
            raise ValueError("invalid staging source reference")
        if previous is not None:
            if previous.stage_id == stage_id and not previous.stop.is_set():
                if (
                    previous.assets != assets
                    or previous.source_ref != source_ref
                    or previous.checkpoint_ref != checkpoint_ref
                ):
                    raise ValueError("staging identity changed")
                previous.expires_at = expiry
                return
            if previous.task is not None and not previous.task.done():
                previous.stop.set()
                return  # Retry delivery once its exact transfer has exited.
            self.resolver.unpin(previous.stage_id)
            self.stages.pop(job_id)
        if len(self.stages) >= self.capacity:
            raise RuntimeError("input staging capacity is full")
        stage = InputStage(
            job_id, stage_id, assets, expiry, source_ref=source_ref, checkpoint_ref=checkpoint_ref
        )
        stage.metrics["remaining_bytes"] = sum(a.total_bytes or 0 for a in assets.values())
        await asyncio.to_thread(self.resolver.pin, stage_id, assets.values())
        self.stages[job_id] = stage
        stage.task = asyncio.create_task(self._stage(stage))

    async def _stage(self, stage):
        remaining = sum(a.total_bytes or 0 for a in stage.assets.values())
        ref_active = False
        member_ready = 0
        downloaded = 0
        download_seconds = 0.0
        download_start = None
        download_offset = 0

        def progress(phase, current, total):
            nonlocal member_ready, downloaded, download_seconds, download_start, download_offset
            now = time.monotonic()
            live_bytes, live_seconds = 0, 0.0
            if phase == "input-view-start":
                member_ready = 0
            if phase in {"downloading", "download-complete"}:
                if download_start is None:
                    download_start, download_offset = now, current
                live_bytes = max(0, current - download_offset)
                live_seconds = max(0, now - download_start)
                if phase == "download-complete":
                    downloaded += live_bytes
                    download_seconds += live_seconds
                    download_start = None
                    live_bytes, live_seconds = 0, 0.0
            if phase == "input-member-ready":
                member_ready += current
                current = 0
            active_file = (
                current
                if phase
                in {
                    "downloading",
                    "download-complete",
                    "verifying-input",
                }
                else 0
            )
            missing = (
                max(0, remaining - member_ready - active_file)
                if ref_active
                else remaining + max(0, total - current)
            )
            stage.metrics = InputStageMetrics(
                remaining_bytes=missing,
                downloaded_bytes=downloaded + live_bytes,
                download_seconds=download_seconds + live_seconds,
            ).model_dump()
            stage.progress = JobProgress(
                current=current, total=total or None, stage=phase, unit="bytes"
            ).model_dump(mode="json")

        def transfer():
            nonlocal remaining, ref_active, member_ready
            with cancellable_transfers(stage.stop), observe_transfers(progress):
                for ref in stage.assets.values():
                    ref_active, member_ready = True, 0
                    self.resolver.resolve(ref)
                    remaining -= ref.total_bytes or 0
                    member_ready = 0
                ref_active = False
                if stage.source_ref is not None:
                    if self.source_resolver is None:
                        raise RuntimeError("source resolver unavailable")
                    self.source_resolver(stage.source_ref)
                if stage.checkpoint_ref is not None:
                    if self.checkpoint_resolver is None:
                        raise RuntimeError("checkpoint resolver unavailable")
                    self.checkpoint_resolver(stage.checkpoint_ref)

        operation = asyncio.create_task(asyncio.to_thread(transfer))
        try:
            await asyncio.shield(operation)
            stage.state = "ready"
            progress("inputs-ready", 0, 0)
        except TransferCancelled:
            stage.state = "canceled"
            progress("stopping-download", 0, 0)
        except asyncio.CancelledError:
            stage.stop.set()
            with suppress(Exception):
                await operation
            stage.state = "canceled"
            raise
        except Exception as exc:
            stage.state = "failed"
            # No arbitrary diagnostics (paths/credentials) cross the protocol.
            progress("input-error", 0, 0)
            stage.progress["message"] = type(exc).__name__
            stage.stop.set()
        finally:
            if stage.state != "ready":
                self.resolver.unpin(stage.stage_id)

    def cancel(self, payload):
        stage = self.stages.get(str(payload.get("job_id")))
        if stage and stage.stage_id == payload.get("stage_id"):
            stage.stop.set()

    def consumed(self, job_id):
        stage = self.stages.get(job_id)
        if stage and stage.task.done():
            self.resolver.unpin(stage.stage_id)
            self.stages.pop(job_id)

    def ready(self, job_id, spec, checkpoint_ref=None):
        stage = self.stages.get(job_id)
        return bool(
            stage
            and stage.state == "ready"
            and not stage.stop.is_set()
            and stage.expires_at > time.time()
            and stage.assets == spec.assets
            and stage.source_ref == spec.source_ref
            and stage.checkpoint_ref == checkpoint_ref
        )

    async def tick(self):
        for job_id, stage in tuple(self.stages.items()):
            if stage.expires_at <= time.time():
                stage.stop.set()
            if stage.stop.is_set() and stage.task.done():
                self.resolver.unpin(stage.stage_id)
                if stage.state == "ready":
                    stage.state = "canceled"
            await self.outbound.publish(
                "input_stage",
                {
                    "job_id": stage.job_id,
                    "stage_id": stage.stage_id,
                    "state": stage.state,
                    "progress": stage.progress,
                    "metrics": stage.metrics,
                },
            )
            if stage.stop.is_set() and stage.task.done():
                self.stages.pop(job_id, None)

    async def close(self):
        for stage in self.stages.values():
            stage.stop.set()
        await asyncio.gather(
            *(stage.task for stage in self.stages.values()), return_exceptions=True
        )
        for stage in self.stages.values():
            self.resolver.unpin(stage.stage_id)
        self.stages.clear()
