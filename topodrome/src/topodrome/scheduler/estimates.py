"""Measured, advisory costs. Never confer GPU or artifact authority."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import timedelta

from sqlalchemy import select

from topodrome.db.models import SchedulingMeasurementRow
from topodrome.domain.specs import utc_now

from .input_staging import aware


def model_context(spec):
    return _digest(spec.model.model_dump(mode="json"))


def workload_context(spec):
    # Exact comparable workload only; never persist user code/kwargs in telemetry.
    return _digest(
        {
            k: v
            for k, v in spec.model_dump(mode="json").items()
            if k
            in {"entrypoint", "source_ref", "model", "assets", "resources", "kwargs", "runtime"}
        }
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def observe(session, site_id, metric, context_key, value, *, now=None):
    if not math.isfinite(value) or value <= 0:
        return
    now = now or utc_now()
    # Insert-if-absent plus row lock also handles two different attempts reporting
    # their first observation concurrently. Replayed events are filtered upstream.
    if session.bind.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    await session.execute(
        insert(SchedulingMeasurementRow)
        .values(
            site_id=site_id,
            metric=metric,
            context_key=context_key,
            value=value,
            samples=0,
            updated_at=now,
        )
        .on_conflict_do_nothing()
    )
    row = await session.get(
        SchedulingMeasurementRow,
        (site_id, metric, context_key),
        with_for_update=True,
        populate_existing=True,
    )
    if row.samples and aware(row.updated_at) > now - timedelta(days=7):
        # Clip isolated spikes; sustained changes still move the EWMA.
        sample = max(row.value / 4, min(value, row.value * 4))
        row.value = 0.7 * row.value + 0.3 * sample
        row.samples = min(row.samples + 1, 1000000)
    else:
        row.value, row.samples = value, 1
    row.updated_at = now
    await session.flush()


async def history(session, spec, site_ids, settings, *, now=None):
    now = now or utc_now()
    rows = await session.scalars(
        select(SchedulingMeasurementRow).where(
            SchedulingMeasurementRow.site_id.in_(site_ids),
            SchedulingMeasurementRow.context_key.in_(
                ("", model_context(spec), workload_context(spec))
            ),
            SchedulingMeasurementRow.updated_at
            > now - timedelta(seconds=settings.history_max_age_seconds),
        )
    )
    return {(r.site_id, r.metric, r.context_key): r.value for r in rows}


def startup_estimates(spec, sites, configured, observations):
    estimates = dict(configured)
    for site in sites:
        measured = observations.get((site.site_id, "startup", model_context(spec)))
        if measured is not None:
            estimates[site.site_id] = measured
    return estimates


def remaining_seconds(spec, site, placement, observations, settings, staged=None, *, now=None):
    """Estimate only remaining cost, ignoring bytes already spent on another site.

    Unknown compute duration cancels between sites. Return I/O is learned only
    for the exact same workload; without it, no speculative output size is invented.
    """
    now = now or utc_now()
    missing = sum(
        a.total_bytes or 0 for a in spec.assets.values() if a.cache_key not in site.cached_assets
    )
    data = (staged.placement_data or {}) if staged is not None else {}
    fresh_metrics = (
        staged is not None
        and staged.site_id == site.site_id
        and data.get("metrics_at", 0) >= now.timestamp() - 30
    )
    metrics = data.get("metrics", {}) if fresh_metrics else {}
    download_rate = observations.get(
        (site.site_id, "download_bps", ""), settings.default_download_mib_per_second * 1024**2
    )
    if metrics.get("downloaded_bytes", 0) >= 256 * 1024 and metrics.get("download_seconds", 0) >= 1:
        # A fresh in-progress sample takes precedence over older link history.
        download_rate = max(1024, metrics["downloaded_bytes"] / metrics["download_seconds"])
    if fresh_metrics:
        missing = metrics["remaining_bytes"]
    overhead = settings.preparation_overhead_seconds if spec.source_ref or spec.assets else 0
    input_seconds = missing / (settings.local_read_mib_per_second * 1024**2)
    if not site.storage_authoritative:
        input_seconds += missing / max(1024, download_rate)
    if staged is not None and staged.site_id == site.site_id and staged.state == "ready":
        input_seconds, overhead = 0, 0
    work_key = workload_context(spec)
    compute = observations.get((site.site_id, "compute", work_key))
    finalization = observations.get((site.site_id, "finalization", work_key), 0)
    comparable_compute = [
        v for (_, metric, key), v in observations.items() if metric == "compute" and key == work_key
    ]
    comparable_finalization = [
        v
        for (_, metric, key), v in observations.items()
        if metric == "finalization" and key == work_key
    ]
    # The public runtime hint includes output I/O: do not add that I/O twice.
    work = (
        compute + finalization
        if compute is not None
        else spec.estimated_runtime_seconds
        or max(comparable_compute, default=0) + max(comparable_finalization, default=0)
    )
    return input_seconds + overhead + placement.estimated_ready_seconds + work


def worthwhile_switch(current, alternative, staged, settings, *, now=None):
    now = now or utc_now()
    data = staged.placement_data or {}
    last_switch = data.get("last_switch_at")
    if last_switch is not None and now.timestamp() - last_switch < settings.cooldown_seconds:
        return False
    if current is None:
        return True  # No eligible GPU remains here; an available target can make progress.
    saving = current - alternative
    return (
        saving >= settings.min_saving_seconds and saving >= current * settings.min_saving_fraction
    )
