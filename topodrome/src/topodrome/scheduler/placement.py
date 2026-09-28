from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Literal

from topodrome.domain.enums import (
    ActorState,
    KernelMode,
    ModelAvailability,
    SiteKind,
    SiteState,
    WorkloadKind,
)
from topodrome.domain.specs import JobSpec, ModelCapability

_DEFAULT_CACHED_STARTUP_SECONDS = 60.0
_DEFAULT_OBTAINABLE_STARTUP_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class GPUCandidate:
    uuid: str
    index: int
    total_memory_bytes: int
    free_memory_bytes: int
    foreign_process_count: int
    foreign_memory_bytes: int
    healthy: bool
    externally_reserved: bool = False
    name: str = ""
    utilization_percent: float = 0
    active_lease: bool = False
    # A local admission rejection means the scheduler's last persisted GPU
    # sample was stale.  The repository sets this only until a newer heartbeat
    # is received; it is distinct from both an active lease and an operator
    # reservation.
    admission_blocked: bool = False
    actor_key_hash: str | None = None
    actor_key_data: dict[str, object] | None = None
    actor_state: ActorState | None = None


@dataclass(frozen=True, slots=True)
class SiteCandidate:
    site_id: str
    kind: SiteKind
    state: SiteState
    heartbeat_at: datetime
    storage_available: bool
    supported_workloads: frozenset[str]
    queue_depth: int
    reliability: float
    paid: bool
    region: str | None
    gpus: tuple[GPUCandidate, ...]
    models: tuple[ModelCapability, ...]
    cached_assets: frozenset[str] = field(default_factory=frozenset)
    storage_authoritative: bool = False
    safety_margin_bytes: int = 0
    agent_session_id: str | None = None
    agent_connection_generation: str | None = None
    total_cpu: float | None = None
    available_cpu: float | None = None
    total_memory_bytes: int | None = None
    available_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class Placement:
    site_id: str
    site_kind: SiteKind
    gpu_uuids: tuple[str, ...]
    score: float
    model_availability: ModelAvailability
    uses_warm_actor: bool
    # Coarse physical-safety signals rank before model warmth within one cost
    # tier.  The headroom class is capped so tiny telemetry differences do not
    # defeat warm reuse: 0 means less than one requested VRAM footprint free,
    # 1 means one, and 2 means at least two.
    vram_headroom_class: int
    shares_with_foreign_processes: bool
    estimated_ready_seconds: float
    max_utilization_percent: float
    fast_gpu_class: bool
    contention_band: Literal["clean", "light", "heavy"]
    # Number of resident actor GPU allocations that the exact assignment must
    # replace before local admission.  This is zero for ordinary reuse/free
    # capacity and counts complete source replicas. Partial replica eviction is
    # never considered a valid target placement.
    idle_actor_replacement_gpu_count: int = 0


@dataclass(frozen=True, slots=True)
class _IdleActorGroup:
    actor_id: str
    gpu_uuids: tuple[str, ...]


def model_matches(capability: ModelCapability, spec: JobSpec) -> bool:
    wanted = spec.model
    return (
        capability.alias == wanted.alias
        and (wanted.revision == "configured" or capability.revision == wanted.revision)
        and capability.backend == wanted.backend
        and (wanted.backend_version is None or capability.backend_version == wanted.backend_version)
        and capability.dtype == wanted.dtype
        and capability.quantization == wanted.quantization
        and capability.adapter == wanted.adapter
    )


def choose_placement(
    spec: JobSpec,
    sites: list[SiteCandidate],
    *,
    heartbeat_cutoff: datetime,
    safety_margin_bytes: int = 0,
    fast_gpu_name_patterns: tuple[str, ...] = (),
    foreign_capacity_sharing_site_ids: tuple[str, ...] = (),
    startup_seconds_by_site: Mapping[str, float] | None = None,
    queue_delay_seconds_per_job: float = 30.0,
    light_contention_utilization_percent: float = 10.0,
    allow_idle_actor_replacement: bool = False,
    prefer_autodl_over_foreign_processes: bool = False,
    remaining_seconds_by_site: Mapping[str, float] | None = None,
) -> Placement | None:
    """Choose one atomic placement using strict contention/cost bands.

    With paid preference enabled, clean institutional gangs precede AutoDL,
    then all foreign sharing. Legacy policy puts light sharing before AutoDL.
    Within a band, an
    empty explicit fast-GPU pattern list treats all institutional GPUs as fast.
    """

    try:
        fast_patterns = tuple(re.compile(pattern) for pattern in fast_gpu_name_patterns)
    except re.error as exc:
        raise ValueError(f"invalid fast GPU name pattern: {exc}") from None
    foreign_capacity_sharing_sites = frozenset(foreign_capacity_sharing_site_ids)
    startup_estimates = startup_seconds_by_site or {}

    institutional = [site for site in sites if site.kind in {SiteKind.LAB, SiteKind.JUPYTER}]
    baseline_institutional_placements = _eligible_placements(
        spec,
        institutional,
        heartbeat_cutoff=heartbeat_cutoff,
        safety_margin_bytes=safety_margin_bytes,
        foreign_capacity_sharing_sites=foreign_capacity_sharing_sites,
        startup_seconds_by_site=startup_estimates,
        queue_delay_seconds_per_job=queue_delay_seconds_per_job,
        fast_gpu_patterns=fast_patterns,
        light_contention_utilization_percent=light_contention_utilization_percent,
        allow_idle_actor_replacement=False,
    )
    institutional_placements = [
        placement
        for placement in baseline_institutional_placements
        if placement.idle_actor_replacement_gpu_count == 0
    ]
    replacements_by_target = {
        (placement.site_id, placement.gpu_uuids): placement
        for placement in baseline_institutional_placements
        if placement.idle_actor_replacement_gpu_count > 0
    }
    if allow_idle_actor_replacement:
        replacements_by_target.update(
            {
                (placement.site_id, placement.gpu_uuids): placement
                for placement in _eligible_placements(
                    spec,
                    institutional,
                    heartbeat_cutoff=heartbeat_cutoff,
                    safety_margin_bytes=safety_margin_bytes,
                    foreign_capacity_sharing_sites=foreign_capacity_sharing_sites,
                    startup_seconds_by_site=startup_estimates,
                    queue_delay_seconds_per_job=queue_delay_seconds_per_job,
                    fast_gpu_patterns=fast_patterns,
                    light_contention_utilization_percent=(light_contention_utilization_percent),
                    allow_idle_actor_replacement=True,
                )
                if placement.idle_actor_replacement_gpu_count > 0
            }
        )
    institutional_replacements = list(replacements_by_target.values())
    autodl_placements = _eligible_placements(
        spec,
        [site for site in sites if site.kind is SiteKind.AUTODL],
        heartbeat_cutoff=heartbeat_cutoff,
        safety_margin_bytes=safety_margin_bytes,
        foreign_capacity_sharing_sites=foreign_capacity_sharing_sites,
        startup_seconds_by_site=startup_estimates,
        queue_delay_seconds_per_job=queue_delay_seconds_per_job,
        fast_gpu_patterns=fast_patterns,
        light_contention_utilization_percent=light_contention_utilization_percent,
        allow_idle_actor_replacement=allow_idle_actor_replacement,
    )

    clean = [
        placement for placement in institutional_placements if placement.contention_band == "clean"
    ]
    lightly_contended = [
        placement for placement in institutional_placements if placement.contention_band == "light"
    ]
    heavily_contended = [
        placement for placement in institutional_placements if placement.contention_band == "heavy"
    ]
    clean_replacements = [
        placement
        for placement in institutional_replacements
        if placement.contention_band == "clean"
    ]
    light_replacements = [
        placement
        for placement in institutional_replacements
        if placement.contention_band == "light"
    ]
    heavy_replacements = [
        placement
        for placement in institutional_replacements
        if placement.contention_band == "heavy"
    ]

    # Preserve legacy ordering for existing policy snapshots until reload activates.
    # Paid preference moves every foreign-sharing band behind AutoDL, including
    # idle-actor replacement; clean replacement still avoids paid capacity.
    bands = (
        clean,
        lightly_contended,
        clean_replacements,
        light_replacements,
        autodl_placements,
        heavily_contended,
        heavy_replacements,
    )
    if prefer_autodl_over_foreign_processes:
        bands = (
            clean,
            clean_replacements,
            autodl_placements,
            lightly_contended,
            light_replacements,
            heavily_contended,
            heavy_replacements,
        )
    for band in bands:
        if band:
            return max(
                band,
                key=lambda placement: (
                    placement.fast_gpu_class,
                    -placement.idle_actor_replacement_gpu_count,
                    -(remaining_seconds_by_site or {}).get(
                        placement.site_id, placement.estimated_ready_seconds
                    ),
                    placement.vram_headroom_class,
                    placement.score,
                    placement.site_id,
                ),
            )
    return None


def _eligible_placements(
    spec: JobSpec,
    sites: list[SiteCandidate],
    *,
    heartbeat_cutoff: datetime,
    safety_margin_bytes: int,
    foreign_capacity_sharing_sites: frozenset[str],
    startup_seconds_by_site: Mapping[str, float],
    queue_delay_seconds_per_job: float,
    fast_gpu_patterns: tuple[re.Pattern[str], ...],
    light_contention_utilization_percent: float,
    allow_idle_actor_replacement: bool,
) -> list[Placement]:
    placements: list[Placement] = []

    for site in sites:
        if site.state is not SiteState.READY or site.heartbeat_at < heartbeat_cutoff:
            continue
        if not site.storage_available:
            continue
        if spec.workload_kind.value not in site.supported_workloads:
            continue
        if spec.affinity_site_id and spec.affinity_site_id != site.site_id:
            continue
        if spec.affinity_region and spec.affinity_region != site.region:
            continue
        if site.available_cpu is not None and spec.resources.cpu > site.available_cpu:
            continue
        required_host_memory = round(spec.resources.memory_gb * 1024**3)
        if (
            site.available_memory_bytes is not None
            and required_host_memory > site.available_memory_bytes
        ):
            continue

        site_safety_margin = max(safety_margin_bytes, site.safety_margin_bytes)
        required_memory = int(spec.resources.min_vram_gb * 1024**3) + site_safety_margin
        capacity_sharing_allowed = site.site_id in foreign_capacity_sharing_sites
        policy_eligible_gpus = tuple(
            gpu for gpu in site.gpus if capacity_sharing_allowed or gpu.foreign_process_count == 0
        )

        capabilities = [capability for capability in site.models if model_matches(capability, spec)]
        if not capabilities:
            continue
        parallelism_mode = spec.model.parallelism.mode
        private_actor = (
            parallelism_mode == "distributed"
            or spec.workload_kind is WorkloadKind.ADAPTER_TRAINING
            or (
                spec.workload_kind is WorkloadKind.INTERACTIVE
                and spec.kwargs.get("mode") == KernelMode.RAW_MODEL.value
            )
        )
        availability = min(
            (
                _effective_availability(
                    capability,
                    spec,
                    private_actor=private_actor,
                )
                for capability in capabilities
            ),
            key=lambda state: {
                ModelAvailability.WARM: 0,
                ModelAvailability.CACHED: 1,
                ModelAvailability.OBTAINABLE: 2,
            }[state],
        )

        warm_sharded_groups = (
            _warm_sharded_groups(
                spec,
                policy_eligible_gpus,
                required_memory=required_memory,
                safety_margin_bytes=site_safety_margin,
            )
            if parallelism_mode == "model_sharded" and not private_actor
            else ()
        )
        if (
            parallelism_mode == "model_sharded"
            and not warm_sharded_groups
            and availability is ModelAvailability.WARM
        ):
            # A differently shaped warm process proves that weights are local,
            # but it cannot satisfy this exact replica topology.
            availability = ModelAvailability.CACHED

        usable: list[GPUCandidate] = []
        replaceable_actor = private_actor or parallelism_mode == "model_sharded"
        replacement_allowed = replaceable_actor or allow_idle_actor_replacement
        idle_actor_groups = _complete_idle_actor_groups(policy_eligible_gpus)
        for gpu in sorted(
            policy_eligible_gpus,
            key=lambda item: (
                -_physical_safety_class(
                    item,
                    required_memory,
                    light_contention_utilization_percent,
                ),
                not _gpu_is_fast(item, fast_gpu_patterns),
                not (
                    not replaceable_actor
                    and parallelism_mode == "single"
                    and item.actor_state is ActorState.IDLE
                    and _actor_matches_spec(item.actor_key_data, spec)
                ),
                bool(item.actor_key_hash is not None),
                item.utilization_percent,
                -item.free_memory_bytes,
                item.index,
            ),
        ):
            if (
                not gpu.healthy
                or gpu.active_lease
                or gpu.admission_blocked
                or gpu.externally_reserved
            ):
                continue
            actor_matches = _actor_matches_spec(gpu.actor_key_data, spec)
            if (
                replacement_allowed
                and gpu.actor_key_hash is not None
                and not (not replaceable_actor and parallelism_mode == "single" and actor_matches)
            ):
                # Private work never inherits resident state. An exact idle
                # allocation is still a viable slot. Ordinary model replacement
                # reaches this path only when explicitly authorized by a real
                # queued job. Complete-group selection below prevents a shard
                # from being evicted independently.
                if gpu.uuid not in idle_actor_groups or gpu.total_memory_bytes < required_memory:
                    continue
                usable.append(gpu)
                continue
            # A resident warm actor is a separate physical allocation. Only a
            # compatible idle actor can lend that GPU to this workload.
            if gpu.actor_key_hash is not None and not (
                actor_matches and gpu.actor_state is ActorState.IDLE
            ):
                continue
            if gpu.actor_key_hash is None and gpu.free_memory_bytes < required_memory:
                continue
            if gpu.actor_key_hash is not None and (
                gpu.total_memory_bytes < required_memory
                or gpu.free_memory_bytes < site_safety_margin
            ):
                continue
            usable.append(gpu)
        gang_candidates: list[
            tuple[
                tuple[GPUCandidate, ...],
                bool,
                tuple[_IdleActorGroup, ...],
            ]
        ] = []
        if parallelism_mode == "model_sharded" and not private_actor:
            gang_candidates.extend((gang, True, ()) for gang in warm_sharded_groups)
        for candidate in combinations(usable, spec.resources.gpu_count):
            selected_candidate = tuple(candidate)
            replacement_groups = _replacement_groups_for_selection(
                selected_candidate,
                spec,
                private_actor=private_actor,
                idle_actor_groups=idle_actor_groups,
            )
            if replacement_groups is None:
                continue
            if replacement_groups and not replacement_allowed:
                continue
            uses_candidate_warm_actor = (
                not private_actor
                and parallelism_mode == "single"
                and not replacement_groups
                and any(_actor_matches_spec(gpu.actor_key_data, spec) for gpu in selected_candidate)
            )
            gang_candidates.append(
                (
                    selected_candidate,
                    uses_candidate_warm_actor,
                    replacement_groups,
                )
            )
        if not gang_candidates:
            continue
        selected, uses_warm_actor, replacement_groups = max(
            gang_candidates,
            key=lambda candidate: (
                min(
                    _physical_safety_class(
                        gpu,
                        required_memory,
                        light_contention_utilization_percent,
                        free_memory_bytes=_prospective_free_memory(
                            gpu,
                            candidate[2],
                        ),
                    )
                    for gpu in candidate[0]
                ),
                all(_gpu_is_fast(gpu, fast_gpu_patterns) for gpu in candidate[0]),
                candidate[1],
                -sum(len(group.gpu_uuids) for group in candidate[2]),
                -max(gpu.utilization_percent for gpu in candidate[0]),
                sum(_prospective_free_memory(gpu, candidate[2]) for gpu in candidate[0]),
            ),
        )
        replacement_gpu_count = sum(len(group.gpu_uuids) for group in replacement_groups)
        prospective_free = {
            gpu.uuid: _prospective_free_memory(gpu, replacement_groups) for gpu in selected
        }
        if availability is ModelAvailability.WARM and not uses_warm_actor:
            # A warm capability proves the exact weights are cached at this
            # site, but it is a zero-startup placement only when the selected
            # UUIDs own that compatible actor.
            availability = ModelAvailability.CACHED
        model_score = {
            ModelAvailability.WARM: 1000.0,
            ModelAvailability.CACHED: 650.0,
            ModelAvailability.OBTAINABLE: 250.0,
        }[availability]
        cached_assets = tuple(
            asset
            for asset in spec.assets.values()
            if (cache_key := asset.cache_key) is not None and cache_key in site.cached_assets
        )
        cached_asset_bytes = sum(asset.total_bytes or 0 for asset in cached_assets)
        cached_asset_count = len(cached_assets)
        # A hit avoids both network transfer and a complete content hash. Keep
        # the byte bonus bounded so locality cannot defeat physical safety or
        # the scheduler's explicit institutional/paid contention bands.
        locality_score = (
            cached_asset_count * 30.0
            + min(
                cached_asset_bytes / 1024**3,
                100.0,
            )
            * 5.0
        )
        free_gib = sum(prospective_free[gpu.uuid] for gpu in selected) / 1024**3
        startup_seconds = (
            0.0
            if uses_warm_actor
            else startup_seconds_by_site.get(
                site.site_id,
                (
                    _DEFAULT_CACHED_STARTUP_SECONDS
                    if availability is ModelAvailability.CACHED
                    else _DEFAULT_OBTAINABLE_STARTUP_SECONDS
                ),
            )
        )
        estimated_ready_seconds = startup_seconds + (site.queue_depth * queue_delay_seconds_per_job)
        score = (
            model_score
            + locality_score
            + (180.0 if not site.paid else 0.0)
            + site.reliability * 100.0
            + min(free_gib, 100.0) * 0.2
            - sum(gpu.utilization_percent for gpu in selected) * 0.5
            - site.queue_depth * 20.0
            - sum(gpu.foreign_process_count for gpu in selected) * 3.0
        )
        placements.append(
            Placement(
                site_id=site.site_id,
                site_kind=site.kind,
                gpu_uuids=tuple(gpu.uuid for gpu in selected),
                score=score,
                model_availability=availability,
                uses_warm_actor=uses_warm_actor,
                vram_headroom_class=min(
                    _vram_headroom_class(prospective_free[gpu.uuid], required_memory)
                    for gpu in selected
                ),
                shares_with_foreign_processes=any(
                    gpu.foreign_process_count > 0 for gpu in selected
                ),
                estimated_ready_seconds=estimated_ready_seconds,
                max_utilization_percent=max(gpu.utilization_percent for gpu in selected),
                fast_gpu_class=(all(_gpu_is_fast(gpu, fast_gpu_patterns) for gpu in selected)),
                contention_band=_contention_band(
                    selected,
                    required_memory,
                    light_contention_utilization_percent,
                    free_memory_bytes_by_gpu=prospective_free,
                ),
                idle_actor_replacement_gpu_count=replacement_gpu_count,
            )
        )
    return placements


def _actor_matches_spec(key_data: dict[str, object] | None, spec: JobSpec) -> bool:
    if key_data is None:
        return False
    parallelism = spec.model.parallelism
    replica_members = _replica_members(key_data)
    if replica_members is None:
        # Legacy single-GPU heartbeats did not publish replica membership.
        # Multi-GPU reuse is never inferred from that incomplete topology.
        if parallelism.degree != 1:
            return False
    elif len(replica_members) != parallelism.degree:
        return False
    if parallelism.mode == "model_sharded":
        replica_id = key_data.get("replica_id")
        if not isinstance(replica_id, str) or not replica_id:
            return False
    actor_gpu_uuid = key_data.get("gpu_uuid")
    if (
        replica_members is not None
        and actor_gpu_uuid is not None
        and actor_gpu_uuid not in replica_members
    ):
        return False
    wanted = spec.model
    expected: dict[str, object] = {
        "alias": wanted.alias,
        "backend": wanted.backend,
        "dtype": wanted.dtype,
        "quantization": wanted.quantization,
        "adapter": wanted.adapter,
        "runtime_fingerprint": spec.runtime_fingerprint,
    }
    if wanted.revision != "configured":
        expected["revision"] = wanted.revision
    if wanted.backend_version is not None:
        expected["backend_version"] = wanted.backend_version
    return all(key_data.get(key) == value for key, value in expected.items())


def _complete_idle_actor_groups(
    gpus: tuple[GPUCandidate, ...],
) -> dict[str, _IdleActorGroup]:
    """Index only complete, idle actor replicas by every physical member.

    The actor identity advertised by the site is ``replica_id``; the central
    ``ModelActorRow.id`` is intentionally not part of placement telemetry. A
    legacy single-GPU capability without a replica identity remains locally
    replaceable through its one exact slot, while an unidentified multi-GPU
    group is never inferred.
    """

    by_uuid = {gpu.uuid: gpu for gpu in gpus}
    groups: dict[tuple[str, tuple[str, ...]], _IdleActorGroup] = {}
    for gpu in gpus:
        key_data = gpu.actor_key_data
        if key_data is None or gpu.actor_key_hash is None or gpu.actor_state is not ActorState.IDLE:
            continue
        members = _replica_members(key_data)
        raw_actor_id = key_data.get("replica_id")
        if not members:
            members = (gpu.uuid,)
        if len(set(members)) != len(members) or gpu.uuid not in members:
            continue
        if isinstance(raw_actor_id, str) and raw_actor_id:
            actor_id = raw_actor_id
        elif len(members) == 1:
            actor_id = f"legacy-single:{gpu.uuid}"
        else:
            continue
        groups[(actor_id, members)] = _IdleActorGroup(
            actor_id=actor_id,
            gpu_uuids=members,
        )

    result: dict[str, _IdleActorGroup] = {}
    for (_actor_id, members), group in groups.items():
        peers: list[GPUCandidate] = []
        for member in members:
            peer = by_uuid.get(member)
            peer_key = None if peer is None else peer.actor_key_data
            peer_members = _replica_members(peer_key)
            if (
                peer is None
                or peer.actor_key_hash is None
                or peer.actor_state is not ActorState.IDLE
                or peer.active_lease
                or peer.admission_blocked
                or peer.externally_reserved
                or not peer.healthy
                or (peer_members not in {None, members})
            ):
                peers = []
                break
            peers.append(peer)
        if len(peers) != len(members):
            continue
        first_identity = _actor_model_identity(peers[0].actor_key_data)
        if first_identity is None or any(
            _actor_model_identity(peer.actor_key_data) != first_identity for peer in peers[1:]
        ):
            continue
        for member in members:
            result[member] = group
    return result


def _actor_model_identity(
    key_data: dict[str, object] | None,
) -> tuple[object, ...] | None:
    if key_data is None:
        return None
    return tuple(
        key_data.get(key)
        for key in (
            "alias",
            "revision",
            "backend",
            "backend_version",
            "dtype",
            "quantization",
            "adapter",
            "runtime_fingerprint",
        )
    )


def _replacement_groups_for_selection(
    selected: tuple[GPUCandidate, ...],
    spec: JobSpec,
    *,
    private_actor: bool,
    idle_actor_groups: Mapping[str, _IdleActorGroup],
) -> tuple[_IdleActorGroup, ...] | None:
    selected_uuids = {gpu.uuid for gpu in selected}
    groups: dict[tuple[str, tuple[str, ...]], _IdleActorGroup] = {}
    for gpu in selected:
        if gpu.actor_key_hash is None:
            continue
        reusable_single = (
            not private_actor
            and spec.model.parallelism.mode == "single"
            and _actor_matches_spec(gpu.actor_key_data, spec)
        )
        if reusable_single:
            continue
        group = idle_actor_groups.get(gpu.uuid)
        if group is None or not set(group.gpu_uuids).issubset(selected_uuids):
            return None
        groups[(group.actor_id, group.gpu_uuids)] = group
    return tuple(groups[key] for key in sorted(groups))


def _prospective_free_memory(
    gpu: GPUCandidate,
    replacement_groups: tuple[_IdleActorGroup, ...],
) -> int:
    replacing = any(gpu.uuid in group.gpu_uuids for group in replacement_groups)
    if not replacing:
        return gpu.free_memory_bytes
    # The resident Topodrome actor is reclaimable, while blocking foreign
    # memory is not. This is used only for ranking/classification; the site
    # agent re-probes physical memory after exact actor disposal.
    return max(
        gpu.free_memory_bytes,
        max(0, gpu.total_memory_bytes - gpu.foreign_memory_bytes),
    )


def _vram_headroom_class(free_memory_bytes: int, required_memory_bytes: int) -> int:
    """Return a stable 0/1/2 class for current physical VRAM headroom."""

    if required_memory_bytes <= 0:
        return 2
    return min(2, free_memory_bytes // required_memory_bytes)


def _physical_safety_class(
    gpu: GPUCandidate,
    required_memory_bytes: int,
    light_contention_utilization_percent: float,
    *,
    free_memory_bytes: int | None = None,
) -> int:
    if gpu.foreign_process_count == 0:
        return 3
    if (
        _vram_headroom_class(
            gpu.free_memory_bytes if free_memory_bytes is None else free_memory_bytes,
            required_memory_bytes,
        )
        >= 2
        and gpu.utilization_percent <= light_contention_utilization_percent
    ):
        return 2
    return 1


def _contention_band(
    gpus: tuple[GPUCandidate, ...],
    required_memory_bytes: int,
    light_contention_utilization_percent: float,
    *,
    free_memory_bytes_by_gpu: Mapping[str, int] | None = None,
) -> Literal["clean", "light", "heavy"]:
    if all(gpu.foreign_process_count == 0 for gpu in gpus):
        return "clean"
    if all(
        _vram_headroom_class(
            (
                gpu.free_memory_bytes
                if free_memory_bytes_by_gpu is None
                else free_memory_bytes_by_gpu[gpu.uuid]
            ),
            required_memory_bytes,
        )
        >= 2
        and gpu.utilization_percent <= light_contention_utilization_percent
        for gpu in gpus
    ):
        return "light"
    return "heavy"


def _gpu_is_fast(
    gpu: GPUCandidate,
    fast_gpu_patterns: tuple[re.Pattern[str], ...],
) -> bool:
    return not fast_gpu_patterns or any(pattern.search(gpu.name) for pattern in fast_gpu_patterns)


def _warm_sharded_groups(
    spec: JobSpec,
    gpus: tuple[GPUCandidate, ...],
    *,
    required_memory: int,
    safety_margin_bytes: int,
) -> tuple[tuple[GPUCandidate, ...], ...]:
    """Find complete exact warm replicas without combining unrelated shards."""

    by_uuid = {gpu.uuid: gpu for gpu in gpus}
    groups: set[tuple[str, tuple[str, ...]]] = set()
    for gpu in gpus:
        key_data = gpu.actor_key_data
        if not _actor_matches_spec(key_data, spec) or key_data is None:
            continue
        members = _replica_members(key_data)
        replica_id = key_data.get("replica_id")
        if members is None:
            continue
        if not isinstance(replica_id, str) or not replica_id:
            continue
        if len(members) == spec.model.parallelism.degree and len(set(members)) == len(members):
            groups.add((replica_id, members))
    complete: list[tuple[GPUCandidate, ...]] = []
    for replica_id, members in sorted(groups):
        selected: list[GPUCandidate] = []
        for gpu_uuid in members:
            candidate = by_uuid.get(gpu_uuid)
            if (
                candidate is None
                or not candidate.healthy
                or candidate.active_lease
                or candidate.admission_blocked
                or candidate.externally_reserved
                or candidate.actor_state is not ActorState.IDLE
                or candidate.total_memory_bytes < required_memory
                or candidate.free_memory_bytes < safety_margin_bytes
                or not _actor_matches_spec(candidate.actor_key_data, spec)
                or (candidate.actor_key_data or {}).get("replica_id") != replica_id
                or _replica_members(candidate.actor_key_data) != members
            ):
                break
            selected.append(candidate)
        if len(selected) == len(members):
            complete.append(tuple(selected))
    return tuple(complete)


def _replica_members(key_data: dict[str, object] | None) -> tuple[str, ...] | None:
    if key_data is None:
        return None
    value = key_data.get("replica_gpu_uuids")
    if not isinstance(value, (list, tuple)):
        return None
    return tuple(str(member) for member in value)


def _effective_availability(
    capability: ModelCapability,
    spec: JobSpec,
    *,
    private_actor: bool,
) -> ModelAvailability:
    if capability.availability is not ModelAvailability.WARM:
        return capability.availability
    if private_actor or capability.runtime_fingerprint != spec.runtime_fingerprint:
        # A live actor implies the exact weights are already local, even when
        # its process environment cannot be reused by this job.
        return ModelAvailability.CACHED
    return ModelAvailability.WARM
