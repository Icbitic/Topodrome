from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from topodrome.domain.enums import ActorState, ModelAvailability, SiteKind, SiteState
from topodrome.domain.specs import JobSpec, ModelCapability, utc_now
from topodrome.scheduler.placement import GPUCandidate, SiteCandidate, choose_placement

GIB = 1024**3


def capability(
    availability: ModelAvailability = ModelAvailability.CACHED,
    *,
    repo_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_uuid: str | None = None,
    runtime_fingerprint: str | None = None,
) -> ModelCapability:
    return ModelCapability(
        repo_id=repo_id,
        revision="rev",
        backend="transformers",
        backend_version="4.57.6",
        dtype="bfloat16",
        availability=availability,
        gpu_uuid=gpu_uuid,
        runtime_fingerprint=runtime_fingerprint,
    )


def site(
    site_id: str,
    *,
    free_gib: int,
    paid: bool = False,
    kind: SiteKind | None = None,
    gpu_name: str = "NVIDIA A800-SXM4-80GB",
    models: tuple[ModelCapability, ...] | None = None,
    actor_key: dict[str, object] | None = None,
    safety_margin_gib: int = 0,
    foreign_process_count: int = 0,
    utilization_percent: float = 0,
    cached_assets: frozenset[str] = frozenset(),
) -> SiteCandidate:
    return SiteCandidate(
        site_id=site_id,
        kind=kind or (SiteKind.AUTODL if paid else SiteKind.LAB),
        state=SiteState.READY,
        heartbeat_at=utc_now(),
        storage_available=True,
        supported_workloads=frozenset({"batch", "interactive", "adapter_training"}),
        queue_depth=0,
        reliability=1,
        paid=paid,
        region=None,
        gpus=(
            GPUCandidate(
                uuid=f"{site_id}-gpu",
                index=0,
                total_memory_bytes=80 * GIB,
                free_memory_bytes=free_gib * GIB,
                foreign_process_count=foreign_process_count,
                foreign_memory_bytes=(80 - free_gib) * GIB,
                healthy=True,
                name=gpu_name,
                utilization_percent=utilization_percent,
                actor_key_hash="hash" if actor_key else None,
                actor_key_data=actor_key,
                actor_state=ActorState.IDLE if actor_key else None,
            ),
        ),
        models=models or (capability(),),
        cached_assets=cached_assets,
        safety_margin_bytes=safety_margin_gib * GIB,
    )


def spec() -> JobSpec:
    return JobSpec(
        entrypoint="experiment.py",
        model={
            "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "revision": "rev",
            "backend_version": "4.57.6",
        },
        resources={"gpu": 1, "min_vram_gb": 15},
    )


def test_placement_prefers_site_with_large_required_asset_cached() -> None:
    digest = "b" * 64
    raw_wanted = spec().model_dump(mode="json")
    raw_wanted["assets"] = {
        "lens": {
            "source_ref": "job://job-1/lens/",
            "kind": "directory",
            "digest": digest,
            "total_bytes": 10 * GIB,
            "members": [
                {
                    "logical_path": "jobs/job-1/attempts/attempt-1/committed/params.pt",
                    "relative_path": "params.pt",
                    "size_bytes": 10 * GIB,
                    "sha256": "a" * 64,
                }
            ],
        }
    }
    wanted = JobSpec.model_validate(raw_wanted)
    cache_key = wanted.assets["lens"].cache_key
    assert cache_key is not None
    placement = choose_placement(
        wanted,
        [
            site("cached", free_gib=40, cached_assets=frozenset({cache_key})),
            site("uncached", free_gib=40),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
    )

    assert placement is not None
    assert placement.site_id == "cached"


def test_placement_asset_locality_requires_the_exact_cached_view() -> None:
    digest = "b" * 64
    raw_wanted = spec().model_dump(mode="json")
    raw_wanted["assets"] = {
        "lens": {
            "source_ref": "job://job-1/lens/",
            "kind": "directory",
            "digest": digest,
            "total_bytes": 10 * GIB,
            "members": [
                {
                    "logical_path": "jobs/job-1/attempts/attempt-1/committed/params.pt",
                    "relative_path": "params.pt",
                    "size_bytes": 10 * GIB,
                    "sha256": "a" * 64,
                }
            ],
        }
    }
    wanted = JobSpec.model_validate(raw_wanted)
    exact_key = wanted.assets["lens"].cache_key
    assert exact_key is not None

    raw_other = dict(raw_wanted["assets"]["lens"])
    raw_other["kind"] = "file"
    raw_other["source_ref"] = "job://job-2/params.pt"
    other_raw = spec().model_dump(mode="json")
    other_raw["assets"] = {"lens": raw_other}
    other = JobSpec.model_validate(other_raw)
    other_key = other.assets["lens"].cache_key
    assert other_key is not None
    assert other.assets["lens"].digest == wanted.assets["lens"].digest
    assert other_key != exact_key

    placement = choose_placement(
        wanted,
        [
            site("exact", free_gib=40, cached_assets=frozenset({exact_key})),
            site("different-view", free_gib=41, cached_assets=frozenset({other_key})),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
    )

    assert placement is not None
    assert placement.site_id == "exact"


def test_placement_rejects_blocking_foreign_process_by_default() -> None:
    placement = choose_placement(
        spec(),
        [
            site(
                "remote-jupyter-1",
                free_gib=20,
                kind=SiteKind.JUPYTER,
                foreign_process_count=1,
            ),
            site("autodl", free_gib=20, paid=True),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        safety_margin_bytes=5 * GIB,
    )

    assert placement is not None
    assert placement.site_id == "autodl"


def test_placement_allows_foreign_process_only_for_opted_in_site() -> None:
    placement = choose_placement(
        spec(),
        [site("lab1", free_gib=20, foreign_process_count=1)],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        safety_margin_bytes=5 * GIB,
        foreign_capacity_sharing_site_ids=("lab1",),
    )
    assert placement is not None
    assert placement.site_id == "lab1"


def test_roomier_shared_jupyter_capacity_beats_crowded_warm_lab_actor() -> None:
    requested = spec()
    runtime_fingerprint = requested.runtime_fingerprint
    actor_key = {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "runtime_fingerprint": runtime_fingerprint,
    }
    crowded_warm_lab = site(
        "lab1",
        free_gib=31,
        models=(
            capability(
                ModelAvailability.WARM,
                gpu_uuid="lab1-gpu",
                runtime_fingerprint=runtime_fingerprint,
            ),
        ),
        actor_key=actor_key,
        safety_margin_gib=1,
        foreign_process_count=1,
    )
    roomy_jupyter = site(
        "remote-jupyter-1",
        free_gib=64,
        kind=SiteKind.JUPYTER,
        safety_margin_gib=1,
        foreign_process_count=1,
    )

    placement = choose_placement(
        requested,
        [crowded_warm_lab, roomy_jupyter],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        foreign_capacity_sharing_site_ids=("lab1", "remote-jupyter-1"),
    )

    assert placement is not None
    assert placement.site_id == "remote-jupyter-1"
    assert placement.vram_headroom_class == 2
    assert placement.shares_with_foreign_processes


def test_warm_shared_capacity_remains_eligible_without_roomier_alternative() -> None:
    requested = spec()
    runtime_fingerprint = requested.runtime_fingerprint
    actor_key = {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "runtime_fingerprint": runtime_fingerprint,
    }
    warm_lab = site(
        "lab1",
        free_gib=40,
        models=(
            capability(
                ModelAvailability.WARM,
                gpu_uuid="lab1-gpu",
                runtime_fingerprint=runtime_fingerprint,
            ),
        ),
        actor_key=actor_key,
        safety_margin_gib=1,
        foreign_process_count=1,
    )

    placement = choose_placement(
        requested,
        [warm_lab, site("autodl", free_gib=32, paid=True)],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        foreign_capacity_sharing_site_ids=("lab1",),
    )

    assert placement is not None
    assert placement.site_id == "lab1"
    assert placement.uses_warm_actor
    assert placement.shares_with_foreign_processes
    assert placement.contention_band == "light"


def test_active_autodl_beats_high_utilization_foreign_sharing() -> None:
    placement = choose_placement(
        spec(),
        [
            site(
                "lab1",
                free_gib=64,
                foreign_process_count=1,
                utilization_percent=80,
            ),
            site("autodl", free_gib=32, paid=True),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        foreign_capacity_sharing_site_ids=("lab1",),
    )

    assert placement is not None
    assert placement.site_id == "autodl"


@pytest.mark.parametrize("foreign_count", [0, 1])
@pytest.mark.parametrize("prefer_paid", [False, True])
def test_paid_preference_overrides_light_sharing_but_preserves_clean_capacity(
    foreign_count: int,
    prefer_paid: bool,
) -> None:
    placement = choose_placement(
        spec(),
        [
            site("lab1", free_gib=64, foreign_process_count=foreign_count),
            site("autodl", free_gib=32, paid=True),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        foreign_capacity_sharing_site_ids=("lab1",),
        prefer_autodl_over_foreign_processes=prefer_paid,
    )
    assert placement is not None
    assert placement.site_id == ("autodl" if prefer_paid and foreign_count else "lab1")


@pytest.mark.parametrize(
    ("utilization_percent", "expected_site"),
    [(10, "lab1"), (10.1, "autodl")],
)
def test_light_contention_threshold_is_a_strict_policy_band(
    utilization_percent: float,
    expected_site: str,
) -> None:
    placement = choose_placement(
        spec(),
        [
            site(
                "lab1",
                free_gib=64,
                foreign_process_count=1,
                utilization_percent=utilization_percent,
            ),
            site("autodl", free_gib=32, paid=True),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        foreign_capacity_sharing_site_ids=("lab1",),
        light_contention_utilization_percent=10,
    )

    assert placement is not None
    assert placement.site_id == expected_site


def test_active_autodl_beats_low_headroom_foreign_sharing() -> None:
    placement = choose_placement(
        spec(),
        [
            site(
                "lab1",
                free_gib=20,
                foreign_process_count=1,
                utilization_percent=0,
            ),
            site("autodl", free_gib=32, paid=True),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        foreign_capacity_sharing_site_ids=("lab1",),
    )

    assert placement is not None
    assert placement.site_id == "autodl"


def test_calibrated_fast_start_beats_extra_headroom_on_clean_capacity() -> None:
    placement = choose_placement(
        spec(),
        [
            site("lab1", free_gib=24),
            site("remote-jupyter-1", free_gib=64, kind=SiteKind.JUPYTER),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        startup_seconds_by_site={"lab1": 2, "remote-jupyter-1": 120},
    )

    assert placement is not None
    assert placement.site_id == "lab1"
    assert placement.estimated_ready_seconds == 2


@pytest.mark.parametrize(
    ("lab_queue_depth", "expected_site"),
    [(3, "lab1"), (4, "remote-jupyter-1")],
)
def test_queue_delay_is_compared_with_cross_site_cold_start(
    lab_queue_depth: int,
    expected_site: str,
) -> None:
    lab = replace(site("lab1", free_gib=24), queue_depth=lab_queue_depth)
    jupyter = site("remote-jupyter-1", free_gib=64, kind=SiteKind.JUPYTER)

    placement = choose_placement(
        spec(),
        [lab, jupyter],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        startup_seconds_by_site={"lab1": 2, "remote-jupyter-1": 120},
        queue_delay_seconds_per_job=30,
    )

    assert placement is not None
    assert placement.site_id == expected_site


def test_selected_warm_jupyter_actor_has_zero_cold_start_penalty() -> None:
    requested = spec()
    runtime_fingerprint = requested.runtime_fingerprint
    actor_key = {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "runtime_fingerprint": runtime_fingerprint,
    }
    warm_jupyter = site(
        "remote-jupyter-1",
        free_gib=24,
        kind=SiteKind.JUPYTER,
        models=(
            capability(
                ModelAvailability.WARM,
                gpu_uuid="remote-jupyter-1-gpu",
                runtime_fingerprint=runtime_fingerprint,
            ),
        ),
        actor_key=actor_key,
    )

    placement = choose_placement(
        requested,
        [site("lab1", free_gib=24), warm_jupyter],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        startup_seconds_by_site={"lab1": 2, "remote-jupyter-1": 120},
    )

    assert placement is not None
    assert placement.site_id == "remote-jupyter-1"
    assert placement.uses_warm_actor
    assert placement.estimated_ready_seconds == 0


def test_external_reservation_removes_only_that_gpu_from_scheduling() -> None:
    jupyter = site("jupyter", free_gib=30, kind=SiteKind.JUPYTER)
    jupyter = replace(
        jupyter,
        gpus=(
            replace(
                jupyter.gpus[0],
                uuid="GPU-reserved",
                index=0,
                externally_reserved=True,
            ),
            replace(
                jupyter.gpus[0],
                uuid="GPU-free",
                index=1,
                externally_reserved=False,
            ),
        ),
    )

    placement = choose_placement(
        spec(),
        [jupyter],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        safety_margin_bytes=GIB,
    )

    assert placement is not None
    assert placement.gpu_uuids == ("GPU-free",)


def test_placement_prefers_idle_gpu_over_busier_gpu_with_more_free_memory() -> None:
    candidate = site("lab1", free_gib=20)
    candidate = replace(
        candidate,
        gpus=(
            replace(
                candidate.gpus[0],
                uuid="GPU-busy",
                index=0,
                free_memory_bytes=40 * GIB,
                utilization_percent=100,
            ),
            replace(
                candidate.gpus[0],
                uuid="GPU-idle",
                index=1,
                free_memory_bytes=20 * GIB,
                utilization_percent=0,
            ),
        ),
    )

    placement = choose_placement(
        spec(),
        [candidate],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        safety_margin_bytes=GIB,
    )

    assert placement is not None
    assert placement.gpu_uuids == ("GPU-idle",)


def test_placement_rejects_below_model_plus_margin() -> None:
    assert (
        choose_placement(
            spec(),
            [site("jupyter", free_gib=19)],
            heartbeat_cutoff=utc_now() - timedelta(seconds=30),
            safety_margin_bytes=5 * GIB,
        )
        is None
    )


def test_site_reported_safety_margin_is_applied_centrally() -> None:
    assert (
        choose_placement(
            spec(),
            [site("jupyter", free_gib=19, safety_margin_gib=5)],
            heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        )
        is None
    )
    assert (
        choose_placement(
            spec(),
            [site("jupyter", free_gib=20, safety_margin_gib=5)],
            heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        )
        is not None
    )


def test_institutional_capacity_beats_running_paid_capacity() -> None:
    placement = choose_placement(
        spec(),
        [site("autodl", free_gib=32, paid=True), site("lab1", free_gib=20)],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
    )
    assert placement is not None
    assert placement.site_id == "lab1"


def test_fast_jupyter_gpu_beats_warmer_slow_institutional_gpu() -> None:
    fast = site(
        "aa-fast-jupyter",
        free_gib=20,
        kind=SiteKind.JUPYTER,
        gpu_name="NVIDIA A800-SXM4-80GB",
        models=(capability(ModelAvailability.OBTAINABLE),),
    )
    slow = site(
        "zz-slow-lab",
        free_gib=30,
        gpu_name="NVIDIA Tesla T4",
        models=(capability(ModelAvailability.WARM),),
    )

    placement = choose_placement(
        spec(),
        [slow, fast],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        fast_gpu_name_patterns=(r"^NVIDIA A800-SXM4-80GB$",),
    )

    assert placement is not None
    assert placement.site_id == "aa-fast-jupyter"


def test_other_institutional_gpu_beats_active_autodl_when_no_fast_gpu_exists() -> None:
    placement = choose_placement(
        spec(),
        [
            site("autodl", free_gib=32, paid=True),
            site("lab1", free_gib=20, gpu_name="NVIDIA Tesla T4"),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        fast_gpu_name_patterns=(r"A800",),
    )

    assert placement is not None
    assert placement.site_id == "lab1"


def test_active_autodl_is_used_only_when_no_institutional_placement_exists() -> None:
    placement = choose_placement(
        spec(),
        [
            site("lab1", free_gib=14, gpu_name="NVIDIA A800-SXM4-80GB"),
            site("autodl", free_gib=32, paid=True),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        fast_gpu_name_patterns=(r"A800",),
    )

    assert placement is not None
    assert placement.site_id == "autodl"


def test_autodl_affinity_bypasses_institutional_tiers() -> None:
    requested = spec().model_copy(update={"affinity_site_id": "autodl"})

    placement = choose_placement(
        requested,
        [site("lab1", free_gib=20), site("autodl", free_gib=32, paid=True)],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
    )

    assert placement is not None
    assert placement.site_id == "autodl"


def test_institutional_affinity_never_spills_to_autodl() -> None:
    requested = spec().model_copy(update={"affinity_site_id": "lab1"})

    placement = choose_placement(
        requested,
        [site("lab1", free_gib=14), site("autodl", free_gib=32, paid=True)],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
    )

    assert placement is None


def test_incompatible_paid_capacity_does_not_block_free_lab_fallback() -> None:
    placement = choose_placement(
        spec(),
        [
            site(
                "autodl",
                free_gib=32,
                paid=True,
                models=(capability(repo_id="test/not-the-requested-model"),),
            ),
            site("lab1", free_gib=20),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
    )
    assert placement is not None
    assert placement.site_id == "lab1"


def test_stale_paid_capacity_does_not_preempt_a_fresh_lab_site() -> None:
    cutoff = utc_now() - timedelta(seconds=30)
    stale_autodl = replace(
        site("autodl", free_gib=32, paid=True),
        heartbeat_at=cutoff - timedelta(seconds=1),
    )

    placement = choose_placement(
        spec(),
        [stale_autodl, site("lab1", free_gib=20)],
        heartbeat_cutoff=cutoff,
    )

    assert placement is not None
    assert placement.site_id == "lab1"


def test_warm_compatible_actor_reuses_its_separate_allocation() -> None:
    runtime_fingerprint = spec().runtime_fingerprint
    key = {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "runtime_fingerprint": runtime_fingerprint,
    }
    candidate = site(
        "lab1",
        free_gib=2,
        models=(
            capability(
                ModelAvailability.WARM,
                gpu_uuid="lab1-gpu",
                runtime_fingerprint=runtime_fingerprint,
            ),
        ),
        actor_key=key,
    )
    placement = choose_placement(
        spec(), candidate and [candidate], heartbeat_cutoff=utc_now() - timedelta(seconds=30)
    )
    assert placement is not None
    assert placement.uses_warm_actor


def test_same_site_selector_does_not_hide_warm_actor_behind_emptier_gpu() -> None:
    requested = spec()
    runtime_fingerprint = requested.runtime_fingerprint
    actor_key = {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "runtime_fingerprint": runtime_fingerprint,
    }
    template = site("lab1", free_gib=80).gpus[0]
    cold = replace(template, uuid="GPU-COLD-EMPTY", index=0, free_memory_bytes=80 * GIB)
    warm = replace(
        template,
        uuid="GPU-WARM",
        index=1,
        free_memory_bytes=63 * GIB,
        actor_key_hash="warm",
        actor_key_data=actor_key,
        actor_state=ActorState.IDLE,
    )
    candidate = replace(
        site("lab1", free_gib=80),
        gpus=(cold, warm),
        models=(
            capability(
                ModelAvailability.WARM,
                gpu_uuid="GPU-WARM",
                runtime_fingerprint=runtime_fingerprint,
            ),
        ),
    )

    placement = choose_placement(
        requested,
        [candidate],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        startup_seconds_by_site={"lab1": 2},
    )

    assert placement is not None
    assert placement.gpu_uuids == ("GPU-WARM",)
    assert placement.uses_warm_actor
    assert placement.estimated_ready_seconds == 0


def test_warm_site_capability_does_not_zero_startup_for_another_selected_gpu() -> None:
    requested = spec()
    runtime_fingerprint = requested.runtime_fingerprint
    actor_key = {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "runtime_fingerprint": runtime_fingerprint,
    }
    warm = replace(
        site("lab1", free_gib=24).gpus[0],
        uuid="GPU-WARM-BUSY",
        actor_key_hash="warm",
        actor_key_data=actor_key,
        actor_state=ActorState.IDLE,
        active_lease=True,
    )
    cold = replace(
        site("lab1", free_gib=30).gpus[0],
        uuid="GPU-COLD-FREE",
        index=1,
    )
    candidate = replace(
        site("lab1", free_gib=30),
        gpus=(warm, cold),
        models=(
            capability(
                ModelAvailability.WARM,
                gpu_uuid="GPU-WARM-BUSY",
                runtime_fingerprint=runtime_fingerprint,
            ),
        ),
    )

    placement = choose_placement(
        requested,
        [candidate],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        startup_seconds_by_site={"lab1": 120},
    )

    assert placement is not None
    assert placement.gpu_uuids == ("GPU-COLD-FREE",)
    assert not placement.uses_warm_actor
    assert placement.model_availability is ModelAvailability.CACHED
    assert placement.estimated_ready_seconds == 120


def test_unspecified_revision_can_match_capability_before_catalog_resolution() -> None:
    requested = spec().model_copy(
        update={"model": spec().model.model_copy(update={"revision": None})}
    )
    runtime_fingerprint = requested.runtime_fingerprint
    actor_key = {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "resolved-immutable-revision",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "runtime_fingerprint": runtime_fingerprint,
    }
    candidate = site(
        "lab1",
        free_gib=2,
        models=(
            ModelCapability(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                revision="resolved-immutable-revision",
                backend="transformers",
                backend_version="4.57.6",
                dtype="bfloat16",
                availability=ModelAvailability.WARM,
                gpu_uuid="lab1-gpu",
                runtime_fingerprint=runtime_fingerprint,
            ),
        ),
        actor_key=actor_key,
    )

    placement = choose_placement(
        requested,
        [candidate],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
    )

    assert placement is not None
    assert placement.uses_warm_actor


def test_multi_gpu_group_can_replace_idle_single_gpu_actor_but_never_reuses_it() -> None:
    base = spec()
    payload = base.model_dump(mode="python")
    payload["resources"]["gpu"] = 2
    payload["model"]["parallelism"] = {
        "mode": "distributed",
        "degree": 2,
        "engine": "ray_worker_group",
    }
    requested = JobSpec.model_validate(payload)
    runtime_fingerprint = requested.runtime_fingerprint
    actor_key = {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "runtime_fingerprint": runtime_fingerprint,
    }
    first = site("lab1", free_gib=30).gpus[0]
    second = replace(
        first,
        uuid="lab1-gpu-2",
        index=1,
        free_memory_bytes=2 * GIB,
        actor_key_hash="warm-hash",
        actor_key_data=actor_key,
        actor_state=ActorState.IDLE,
    )
    candidate = replace(
        site("lab1", free_gib=30),
        gpus=(first, second),
        models=(
            capability(
                ModelAvailability.WARM,
                gpu_uuid="lab1-gpu-2",
                runtime_fingerprint=runtime_fingerprint,
            ),
        ),
    )

    placement = choose_placement(
        requested,
        [candidate],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
    )

    assert placement is not None
    assert set(placement.gpu_uuids) == {"lab1-gpu", "lab1-gpu-2"}
    assert not placement.uses_warm_actor
    assert placement.model_availability is ModelAvailability.CACHED


def test_single_gpu_job_never_reuses_a_member_of_sharded_replica() -> None:
    requested = spec()
    actor_key = {
        "repo_id": requested.model.repo_id,
        "revision": requested.model.revision,
        "backend": requested.model.backend,
        "backend_version": requested.model.backend_version,
        "dtype": requested.model.dtype,
        "quantization": requested.model.quantization,
        "adapter": requested.model.adapter,
        "gpu_uuid": "lab1-gpu",
        # Deliberately match the runtime fingerprint to prove topology is an
        # independent admission invariant, not an accidental hash side effect.
        "runtime_fingerprint": requested.runtime_fingerprint,
        "replica_id": "sharded-replica",
        "replica_gpu_uuids": ["lab1-gpu", "lab1-gpu-2"],
    }
    candidate = site(
        "lab1",
        free_gib=2,
        models=(
            capability(
                ModelAvailability.WARM,
                gpu_uuid="lab1-gpu",
                runtime_fingerprint=requested.runtime_fingerprint,
            ),
        ),
        actor_key=actor_key,
    )

    placement = choose_placement(
        requested,
        [candidate],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
    )

    assert placement is None


def test_sharded_job_never_reuses_independent_single_gpu_actors_as_a_replica() -> None:
    base = spec()
    requested = JobSpec.model_validate(
        {
            **base.model_dump(mode="python"),
            "resources": {"gpu": 2, "min_vram_gb": 15},
            "model": {
                **base.model.model_dump(mode="python"),
                "parallelism": {
                    "mode": "model_sharded",
                    "degree": 2,
                    "engine": "transformers_device_map",
                    "strategy": "balanced",
                    "max_memory_gb_per_gpu": 15,
                },
            },
        }
    )
    template = site("lab1", free_gib=2).gpus[0]

    def single_actor(gpu_uuid: str) -> dict[str, object]:
        return {
            "repo_id": requested.model.repo_id,
            "revision": requested.model.revision,
            "backend": requested.model.backend,
            "backend_version": requested.model.backend_version,
            "dtype": requested.model.dtype,
            "quantization": requested.model.quantization,
            "adapter": requested.model.adapter,
            "gpu_uuid": gpu_uuid,
            "runtime_fingerprint": requested.runtime_fingerprint,
            "replica_id": f"single-{gpu_uuid}",
            "replica_gpu_uuids": [gpu_uuid],
        }

    gpus = tuple(
        replace(
            template,
            uuid=gpu_uuid,
            index=index,
            actor_key_hash=f"single-{index}",
            actor_key_data=single_actor(gpu_uuid),
            actor_state=ActorState.IDLE,
        )
        for index, gpu_uuid in enumerate(("GPU-A", "GPU-B"))
    )
    candidate = replace(
        site("lab1", free_gib=2),
        gpus=gpus,
        models=(
            capability(
                ModelAvailability.WARM,
                gpu_uuid="GPU-A",
                runtime_fingerprint=requested.runtime_fingerprint,
            ),
        ),
    )

    placement = choose_placement(
        requested,
        [candidate],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
    )

    assert placement is not None
    assert not placement.uses_warm_actor
    assert placement.model_availability is ModelAvailability.CACHED


def test_fast_institutional_gang_is_selected_atomically_by_gpu_name() -> None:
    base = spec()
    requested = JobSpec.model_validate(
        {
            **base.model_dump(mode="python"),
            "resources": {"gpu": 2, "min_vram_gb": 15},
            "model": {
                **base.model.model_dump(mode="python"),
                "parallelism": {
                    "mode": "distributed",
                    "degree": 2,
                    "engine": "ray_worker_group",
                },
            },
        }
    )
    template = site("lab1", free_gib=40).gpus[0]
    slow = replace(template, uuid="GPU-SLOW", index=0, name="NVIDIA Tesla T4")
    fast_a = replace(
        template,
        uuid="GPU-FAST-A",
        index=1,
        name="NVIDIA A800-SXM4-80GB",
        free_memory_bytes=30 * GIB,
    )
    fast_b = replace(
        template,
        uuid="GPU-FAST-B",
        index=2,
        name="NVIDIA A800-SXM4-80GB",
        free_memory_bytes=25 * GIB,
    )
    institutional = replace(
        site("lab1", free_gib=40),
        gpus=(slow, fast_a, fast_b),
    )
    paid_template = site("autodl", free_gib=32, paid=True).gpus[0]
    paid = replace(
        site("autodl", free_gib=32, paid=True),
        gpus=(
            paid_template,
            replace(paid_template, uuid="autodl-gpu-2", index=1),
        ),
    )

    placement = choose_placement(
        requested,
        [paid, institutional],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        fast_gpu_name_patterns=(r"^NVIDIA A800-SXM4-80GB$",),
    )

    assert placement is not None
    assert placement.site_id == "lab1"
    assert placement.gpu_uuids == ("GPU-FAST-A", "GPU-FAST-B")


def test_model_sharded_placement_reuses_only_one_proven_complete_replica() -> None:
    requested = JobSpec(
        entrypoint="experiment.py",
        model={
            "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "revision": "rev",
            "backend_version": "4.57.6",
            "parallelism": {
                "mode": "model_sharded",
                "degree": 2,
                "max_memory_gb_per_gpu": 10,
            },
        },
        resources={"gpu": 2, "min_vram_gb": 10},
    )
    members = ("GPU-B", "GPU-A")

    def actor_key(gpu_uuid: str, replica_id: str = "replica-live") -> dict[str, object]:
        return {
            "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "revision": "rev",
            "backend": "transformers",
            "backend_version": "4.57.6",
            "dtype": "bfloat16",
            "quantization": None,
            "adapter": None,
            "gpu_uuid": gpu_uuid,
            "runtime_fingerprint": requested.runtime_fingerprint,
            "replica_id": replica_id,
            "replica_gpu_uuids": list(members),
        }

    gpus = (
        GPUCandidate(
            uuid="GPU-A",
            index=0,
            total_memory_bytes=24 * GIB,
            free_memory_bytes=2 * GIB,
            foreign_process_count=0,
            foreign_memory_bytes=0,
            healthy=True,
            actor_key_hash="a",
            actor_key_data=actor_key("GPU-A"),
            actor_state=ActorState.IDLE,
        ),
        GPUCandidate(
            uuid="GPU-B",
            index=1,
            total_memory_bytes=24 * GIB,
            free_memory_bytes=2 * GIB,
            foreign_process_count=0,
            foreign_memory_bytes=0,
            healthy=True,
            actor_key_hash="b",
            actor_key_data=actor_key("GPU-B"),
            actor_state=ActorState.IDLE,
        ),
    )
    models = tuple(
        ModelCapability(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            revision="rev",
            backend="transformers",
            backend_version="4.57.6",
            dtype="bfloat16",
            availability=ModelAvailability.WARM,
            gpu_uuid=gpu_uuid,
            runtime_fingerprint=requested.runtime_fingerprint,
            replica_id="replica-live",
            replica_gpu_uuids=members,
        )
        for gpu_uuid in members
    )
    candidate = replace(site("lab1", free_gib=20), gpus=gpus, models=models)

    placement = choose_placement(
        requested,
        [candidate],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        safety_margin_bytes=GIB,
    )

    assert placement is not None
    assert placement.gpu_uuids == members
    assert placement.uses_warm_actor
    assert placement.model_availability is ModelAvailability.WARM

    mismatched = replace(
        candidate,
        gpus=(gpus[0], replace(gpus[1], actor_key_data=actor_key("test/GPU-B", "other"))),
    )
    fresh = choose_placement(
        requested,
        [mismatched],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        safety_margin_bytes=GIB,
    )
    assert fresh is not None
    assert not fresh.uses_warm_actor
    assert fresh.model_availability is ModelAvailability.CACHED


def test_contended_warm_sharded_replica_does_not_hide_clean_cold_gang() -> None:
    requested = JobSpec(
        entrypoint="experiment.py",
        model={
            "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "revision": "rev",
            "backend_version": "4.57.6",
            "parallelism": {
                "mode": "model_sharded",
                "degree": 2,
                "max_memory_gb_per_gpu": 15,
            },
        },
        resources={"gpu": 2, "min_vram_gb": 15},
    )
    warm_members = ("GPU-WARM-A", "GPU-WARM-B")

    def warm_key(gpu_uuid: str) -> dict[str, object]:
        return {
            "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "revision": "rev",
            "backend": "transformers",
            "backend_version": "4.57.6",
            "dtype": "bfloat16",
            "quantization": None,
            "adapter": None,
            "gpu_uuid": gpu_uuid,
            "runtime_fingerprint": requested.runtime_fingerprint,
            "replica_id": "warm-heavy",
            "replica_gpu_uuids": list(warm_members),
        }

    template = site("lab1", free_gib=30).gpus[0]
    warm_gpus = tuple(
        replace(
            template,
            uuid=gpu_uuid,
            index=index,
            free_memory_bytes=40 * GIB,
            foreign_process_count=1,
            foreign_memory_bytes=20 * GIB,
            utilization_percent=80,
            actor_key_hash=f"warm-{index}",
            actor_key_data=warm_key(gpu_uuid),
            actor_state=ActorState.IDLE,
        )
        for index, gpu_uuid in enumerate(warm_members)
    )
    clean_gpus = (
        replace(template, uuid="GPU-CLEAN-A", index=2, free_memory_bytes=30 * GIB),
        replace(template, uuid="GPU-CLEAN-B", index=3, free_memory_bytes=30 * GIB),
    )
    models = tuple(
        ModelCapability(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            revision="rev",
            backend="transformers",
            backend_version="4.57.6",
            dtype="bfloat16",
            availability=ModelAvailability.WARM,
            gpu_uuid=gpu_uuid,
            runtime_fingerprint=requested.runtime_fingerprint,
            replica_id="warm-heavy",
            replica_gpu_uuids=warm_members,
        )
        for gpu_uuid in warm_members
    )
    candidate = replace(
        site("lab1", free_gib=30),
        gpus=(*warm_gpus, *clean_gpus),
        models=models,
    )

    placement = choose_placement(
        requested,
        [candidate],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        foreign_capacity_sharing_site_ids=("lab1",),
    )

    assert placement is not None
    assert placement.gpu_uuids == ("GPU-CLEAN-A", "GPU-CLEAN-B")
    assert placement.contention_band == "clean"
    assert not placement.uses_warm_actor
    assert placement.model_availability is ModelAvailability.CACHED


@pytest.mark.parametrize(
    ("workload_kind", "kwargs"),
    [
        ("adapter_training", {}),
        ("interactive", {"mode": "raw-model"}),
    ],
)
def test_private_workload_can_replace_exact_idle_actor_on_only_gpu(
    workload_kind: str,
    kwargs: dict[str, str],
) -> None:
    base = spec()
    requested = JobSpec.model_validate(
        {**base.model_dump(mode="python"), "workload_kind": workload_kind, "kwargs": kwargs}
    )
    actor_key = {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "runtime_fingerprint": base.runtime_fingerprint,
    }
    candidate = site(
        "lab1",
        free_gib=2,
        models=(
            capability(
                ModelAvailability.WARM,
                gpu_uuid="lab1-gpu",
                runtime_fingerprint=base.runtime_fingerprint,
            ),
        ),
        actor_key=actor_key,
    )

    placement = choose_placement(
        requested,
        [candidate],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
    )

    assert placement is not None
    assert not placement.uses_warm_actor
    assert placement.model_availability is ModelAvailability.CACHED


def test_placement_filters_reported_available_host_resources() -> None:
    constrained = replace(
        site("lab1", free_gib=30),
        total_cpu=8,
        available_cpu=1.5,
        total_memory_bytes=64 * GIB,
        available_memory_bytes=3 * GIB,
    )
    base = spec()
    requested = base.model_copy(
        update={"resources": base.resources.model_copy(update={"cpu": 2, "memory_gb": 4})}
    )

    assert (
        choose_placement(
            requested,
            [constrained],
            heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        )
        is None
    )


def test_incompatible_warm_actor_blocks_gpu_until_eviction() -> None:
    key = {
        "repo_id": "test/another-model",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
    }
    candidate = site(
        "lab1",
        free_gib=30,
        models=(capability(ModelAvailability.OBTAINABLE),),
        actor_key=key,
    )
    assert (
        choose_placement(spec(), [candidate], heartbeat_cutoff=utc_now() - timedelta(seconds=30))
        is None
    )


def test_queued_ordinary_model_can_replace_complete_idle_actor_before_autodl() -> None:
    key = {
        "repo_id": "test/another-model",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "gpu_uuid": "lab1-gpu",
        "replica_id": "ray-actor-old-model",
        "replica_gpu_uuids": ["lab1-gpu"],
    }
    replacement = choose_placement(
        spec(),
        [
            site(
                "lab1",
                free_gib=2,
                models=(capability(ModelAvailability.CACHED),),
                actor_key=key,
            ),
            site("autodl", free_gib=32, paid=True),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        allow_idle_actor_replacement=True,
    )

    assert replacement is not None
    assert replacement.site_id == "lab1"
    assert replacement.idle_actor_replacement_gpu_count == 1
    assert not replacement.uses_warm_actor


@pytest.mark.parametrize("blocked", [None, "active_lease", "busy", "reservation"])
def test_autodl_queue_driven_idle_replacement(blocked) -> None:
    key = {
        "repo_id": "test/another-model",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "gpu_uuid": "autodl-gpu",
        "replica_id": "old-model",
        "replica_gpu_uuids": ["autodl-gpu"],
    }
    candidate = site("autodl", free_gib=2, paid=True, actor_key=key)
    gpu = candidate.gpus[0]
    if blocked == "active_lease":
        gpu = replace(gpu, active_lease=True)
    elif blocked == "busy":
        gpu = replace(gpu, actor_state=ActorState.BUSY)
    elif blocked == "reservation":
        gpu = replace(gpu, externally_reserved=True)
    candidate = replace(candidate, gpus=(gpu,))
    cutoff = utc_now() - timedelta(seconds=30)
    assert choose_placement(spec(), [candidate], heartbeat_cutoff=cutoff) is None
    result = choose_placement(
        spec(), [candidate], heartbeat_cutoff=cutoff, allow_idle_actor_replacement=True
    )
    if blocked:
        assert result is None
    else:
        assert result is not None
        assert result.site_id == "autodl"
        assert result.idle_actor_replacement_gpu_count == 1
        assert not result.uses_warm_actor


def test_ordinary_clean_capacity_precedes_idle_actor_replacement() -> None:
    key = {
        "repo_id": "test/another-model",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "gpu_uuid": "lab1-gpu",
        "replica_id": "ray-actor-old-model",
        "replica_gpu_uuids": ["lab1-gpu"],
    }
    placement = choose_placement(
        spec(),
        [
            site(
                "lab1",
                free_gib=2,
                models=(capability(ModelAvailability.CACHED),),
                actor_key=key,
            ),
            site(
                "remote-jupyter-1",
                free_gib=32,
                kind=SiteKind.JUPYTER,
            ),
        ],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        allow_idle_actor_replacement=True,
    )

    assert placement is not None
    assert placement.site_id == "remote-jupyter-1"
    assert placement.idle_actor_replacement_gpu_count == 0


def test_replacement_never_selects_only_one_member_of_resident_replica() -> None:
    members = ("lab1-gpu-0", "lab1-gpu-1")

    def resident(gpu_uuid: str, index: int) -> GPUCandidate:
        return GPUCandidate(
            uuid=gpu_uuid,
            index=index,
            total_memory_bytes=80 * GIB,
            free_memory_bytes=2 * GIB,
            foreign_process_count=0,
            foreign_memory_bytes=0,
            healthy=True,
            actor_key_hash=f"hash-{index}",
            actor_key_data={
                "repo_id": "test/another-model",
                "revision": "rev",
                "backend": "transformers",
                "backend_version": "4.57.6",
                "dtype": "bfloat16",
                "quantization": None,
                "adapter": None,
                "gpu_uuid": gpu_uuid,
                "replica_id": "two-gpu-resident",
                "replica_gpu_uuids": list(members),
            },
            actor_state=ActorState.IDLE,
        )

    lab = replace(
        site("lab1", free_gib=2),
        gpus=tuple(resident(gpu_uuid, index) for index, gpu_uuid in enumerate(members)),
    )
    placement = choose_placement(
        spec(),
        [lab, site("autodl", free_gib=32, paid=True)],
        heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        allow_idle_actor_replacement=True,
    )

    assert placement is not None
    assert placement.site_id == "autodl"


def test_incompatible_runtime_fingerprint_never_reuses_warm_actor() -> None:
    requested = spec()
    incompatible_runtime = "b" * 64
    key = {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "rev",
        "backend": "transformers",
        "backend_version": "4.57.6",
        "dtype": "bfloat16",
        "quantization": None,
        "adapter": None,
        "runtime_fingerprint": incompatible_runtime,
    }
    candidate = site(
        "lab1",
        free_gib=2,
        models=(
            capability(
                ModelAvailability.WARM,
                gpu_uuid="lab1-gpu",
                runtime_fingerprint=incompatible_runtime,
            ),
        ),
        actor_key=key,
    )

    assert incompatible_runtime != requested.runtime_fingerprint
    assert (
        choose_placement(
            requested,
            [candidate],
            heartbeat_cutoff=utc_now() - timedelta(seconds=30),
        )
        is None
    )
