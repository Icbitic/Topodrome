"""Soft future-gang protection for aged jobs. Never grants a physical lease."""

from dataclasses import replace

from topodrome.domain.enums import ModelAvailability
from topodrome.scheduler.placement import choose_placement


def future_gang(spec, sites, options, *, preferred_site=None):
    future = []
    for site in sites:
        gpus = []
        required = int(spec.resources.min_vram_gb * 1024**3) + max(
            site.safety_margin_bytes, options.get("safety_margin_bytes", 0)
        )
        for gpu in site.gpus:
            # Only Topodrome-owned memory/claims may be predicted as releasable.
            if gpu.active_lease or gpu.actor_key_hash is not None:
                gpu = replace(
                    gpu,
                    active_lease=False,
                    actor_key_hash=None,
                    actor_key_data=None,
                    actor_state=None,
                    # Prefer a gang needing fewer releases, not busy cards
                    # whose hypothetical empty-memory score looks better.
                    free_memory_bytes=min(
                        required, max(0, gpu.total_memory_bytes - gpu.foreign_memory_bytes)
                    ),
                    utilization_percent=(
                        gpu.utilization_percent if gpu.foreign_process_count else 0
                    ),
                )
            gpus.append(gpu)
        future.append(
            replace(
                site,
                gpus=tuple(gpus),
                available_cpu=site.total_cpu,
                available_memory_bytes=site.total_memory_bytes,
                models=tuple(
                    model.model_copy(update={"availability": ModelAvailability.CACHED})
                    if model.availability is ModelAvailability.WARM
                    else model
                    for model in site.models
                ),
            )
        )
    if preferred_site:
        preferred = choose_placement(
            spec, [s for s in future if s.site_id == preferred_site], **options
        )
        if preferred:
            return preferred
    return choose_placement(spec, future, **options)


def protect_gang(sites, protected):
    if protected is None:
        return sites
    return [
        replace(
            site,
            gpus=tuple(
                replace(gpu, externally_reserved=True)
                if site.site_id == protected.site_id and gpu.uuid in protected.gpu_uuids
                else gpu
                for gpu in site.gpus
            ),
        )
        for site in sites
    ]
