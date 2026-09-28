from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from topodrome.config import AutoDLSettings
from topodrome.domain.enums import JobState, ModelAvailability
from topodrome.domain.errors import InvalidTransitionError, StaleFenceError
from topodrome.domain.specs import (
    FenceClaim,
    JobSpec,
    ModelCapability,
    RuntimeEnvironment,
    SiteHeartbeat,
    utc_now,
)
from topodrome.domain.transitions import JOB_TRANSITIONS, require_transition
from topodrome.protocol.fencing import FenceEpochs, FencingAuthority


def test_job_spec_parses_public_shape_and_rejects_traversal() -> None:
    spec = JobSpec(
        entrypoint="experiments/label_heads.py",
        model="meta-llama-3-8b-instruct",
        assets={"evaluation": "asset://shared/mmlu@v1"},
        resources={"gpu": 1, "min_vram_gb": 15},
    )
    assert spec.model.alias == "meta-llama-3-8b-instruct"
    assert spec.assets["evaluation"].source_ref == "asset://shared/mmlu@v1"
    assert not spec.assets["evaluation"].pinned
    assert spec.resources.gpu_count == 1
    with pytest.raises(ValidationError):
        JobSpec(entrypoint="../escape.py", model="x")


def test_job_spec_asset_pins_are_complete_and_aliases_are_safe() -> None:
    member = {
        "logical_path": "jobs/job-1/attempts/attempt-1/committed/params.pt",
        "relative_path": "params.pt",
        "size_bytes": 7,
        "sha256": "a" * 64,
    }
    pinned = JobSpec(
        entrypoint="experiment.py",
        model="model",
        assets={
            "lens": {
                "source_ref": "job://job-1/params.pt",
                "kind": "file",
                "digest": "b" * 64,
                "total_bytes": 7,
                "members": [member],
            }
        },
    )
    assert pinned.assets["lens"].pinned
    assert pinned.assets["lens"].kind == "file"

    with pytest.raises(ValidationError, match="kind, digest, total_bytes, and members"):
        JobSpec(
            entrypoint="experiment.py",
            model="model",
            assets={"lens": {"source_ref": "job://job-1/params.pt", "digest": "b" * 64}},
        )
    with pytest.raises(ValidationError, match="safe single path components"):
        JobSpec(
            entrypoint="experiment.py",
            model="model",
            assets={"../lens": "job://job-1/params.pt"},
        )


def test_runtime_environment_rejects_credentials() -> None:
    with pytest.raises(ValidationError, match="credentials"):
        RuntimeEnvironment(env={"HF_TOKEN": "not-allowed"})
    with pytest.raises(ValidationError, match="credentials"):
        RuntimeEnvironment(pip=("private @ https://user:password@example.invalid/pkg",))
    with pytest.raises(ValidationError, match="credentials"):
        RuntimeEnvironment(working_dir_ref="https://example.invalid/code.zip?token=not-allowed")
    with pytest.raises(ValidationError, match="reserved"):
        RuntimeEnvironment(env={"CUDA_VISIBLE_DEVICES": "7"})


def test_runtime_environment_requires_content_addressed_dependencies() -> None:
    with pytest.raises(ValidationError, match=r"exact .* pins"):
        RuntimeEnvironment(pip=("einops==0.8.0",))
    with pytest.raises(ValidationError, match=r"exact .* pins"):
        RuntimeEnvironment(
            pip=(f"einops>=0.8.0 --hash=sha256:{'b' * 64}",),
        )
    with pytest.raises(ValidationError, match="sha256 reference"):
        RuntimeEnvironment(working_dir_ref="https://public.invalid/source.zip")

    runtime = RuntimeEnvironment(
        pip=(f"einops==0.8.0 --hash=sha256:{'b' * 64}",),
        working_dir_ref=f"sha256:{'a' * 64}",
    )
    assert runtime.working_dir_ref == f"sha256:{'a' * 64}"


def test_runtime_fingerprint_is_stable_and_covers_environment_and_resources() -> None:
    base = JobSpec(
        entrypoint="experiment.py",
        model="model",
        resources={"gpu": 1, "cpu": 2, "memory_gb": 4},
        runtime={
            "pip": [f"einops==0.8.0 --hash=sha256:{'b' * 64}"],
            "env": {"SECOND": "2", "FIRST": "1"},
            "working_dir_ref": "sha256:" + "a" * 64,
        },
    )
    reordered = JobSpec(
        entrypoint="experiment.py",
        model="model",
        resources={"gpu": 1, "cpu": 2, "memory_gb": 4},
        runtime={
            "pip": [f"einops==0.8.0 --hash=sha256:{'b' * 64}"],
            "env": {"FIRST": "1", "SECOND": "2"},
            "working_dir_ref": "sha256:" + "a" * 64,
        },
    )
    changed = reordered.model_copy(
        update={"resources": reordered.resources.model_copy(update={"cpu": 3})}
    )

    assert len(base.runtime_fingerprint) == 64
    assert base.runtime_fingerprint == reordered.runtime_fingerprint
    assert changed.runtime_fingerprint != base.runtime_fingerprint


def test_multi_gpu_intent_is_explicit_and_part_of_runtime_identity() -> None:
    with pytest.raises(ValidationError, match="multi-GPU intent must be explicit"):
        JobSpec(
            entrypoint="experiment.py",
            model="model",
            resources={"gpu": 2, "min_vram_gb": 10},
        )

    sharded = JobSpec(
        entrypoint="experiment.py",
        model={
            "alias": "model",
            "parallelism": {
                "mode": "model_sharded",
                "degree": 2,
                "max_memory_gb_per_gpu": 10,
            },
        },
        resources={"gpu": 2, "min_vram_gb": 10},
    )
    distributed = JobSpec(
        entrypoint="experiment.py",
        model={
            "alias": "model",
            "parallelism": {"mode": "distributed", "degree": 2},
        },
        resources={"gpu": 2, "min_vram_gb": 10},
    )

    assert sharded.model.parallelism.engine == "transformers_device_map"
    assert distributed.model.parallelism.engine == "ray_worker_group"
    assert sharded.runtime_fingerprint != distributed.runtime_fingerprint


def test_model_sharding_fails_closed_on_unsafe_memory_or_replica_metadata() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        JobSpec(
            entrypoint="experiment.py",
            model={
                "alias": "model",
                "parallelism": {
                    "mode": "model_sharded",
                    "degree": 2,
                    "max_memory_gb_per_gpu": 12,
                },
            },
            resources={"gpu": 2, "min_vram_gb": 10},
        )
    with pytest.raises(ValidationError, match="must belong"):
        ModelCapability(
            alias="model",
            revision="rev",
            backend="transformers",
            backend_version="1",
            dtype="bfloat16",
            availability=ModelAvailability.WARM,
            gpu_uuid="GPU-X",
            runtime_fingerprint="a" * 64,
            replica_gpu_uuids=("GPU-A", "GPU-B"),
        )


def test_host_telemetry_is_complete_and_bounded() -> None:
    with pytest.raises(ValidationError, match="complete set"):
        SiteHeartbeat(
            site_id="lab1",
            site_kind="lab",
            agent_session_id="session",
            sequence=1,
            total_cpu=8,
        )
    with pytest.raises(ValidationError, match="cannot exceed"):
        SiteHeartbeat(
            site_id="lab1",
            site_kind="lab",
            agent_session_id="session",
            sequence=1,
            total_cpu=8,
            available_cpu=9,
            total_memory_bytes=16,
            available_memory_bytes=8,
        )


def test_job_kwargs_reject_nested_credentials_without_rejecting_tokenizer_options() -> None:
    unsafe_values = (
        {"client": {"api_key": "not-allowed"}},
        {"sources": ["https://user:password@provider.invalid/object"]},
        {"source": "https://provider.invalid/object?access_token=not-allowed"},
        {"source": "https://provider.invalid/object?access%5Ftoken=not-allowed"},
        {"header": "Authorization: Bearer not-allowed"},
        {"material": "-----BEGIN PRIVATE KEY-----\nnot-allowed"},
    )
    for kwargs in unsafe_values:
        with pytest.raises(ValidationError, match="credentials"):
            JobSpec(
                entrypoint="experiment.py",
                model="model",
                kwargs=kwargs,
            )

    spec = JobSpec(
        entrypoint="experiment.py",
        model="model",
        kwargs={
            "tokenizer_name": "local-tokenizer",
            "max_new_tokens": 8,
            "public_source": "https://assets.invalid/data?tokens=8",
        },
    )
    assert spec.kwargs["max_new_tokens"] == 8


def test_autodl_idle_shutdown_policy_is_configurable_and_bounded() -> None:
    assert AutoDLSettings().idle_timeout_seconds == 300
    assert AutoDLSettings(idle_timeout_seconds=300).idle_timeout_seconds == 300
    for invalid in (0, 59, 86401):
        with pytest.raises(ValidationError):
            AutoDLSettings(idle_timeout_seconds=invalid)


def test_autodl_rental_gpu_count_setting_is_removed() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        AutoDLSettings(max_gpu_count=4)


def test_transition_guard() -> None:
    require_transition(JobState.QUEUED, JobState.ASSIGNING, JOB_TRANSITIONS)
    with pytest.raises(InvalidTransitionError):
        require_transition(JobState.QUEUED, JobState.SUCCEEDED, JOB_TRANSITIONS)


def test_fencing_is_scoped_expiring_and_tamper_evident() -> None:
    authority = FencingAuthority(b"x" * 32)
    now = utc_now()
    token, payload = authority.issue(
        attempt_id="attempt-1",
        site_id="lab1",
        claims=(FenceClaim(gpu_uuid="gpu-0", epoch=3),),
        ttl=timedelta(seconds=10),
        now=now,
    )
    assert authority.verify(token, attempt_id="attempt-1", site_id="lab1", now=now) == payload
    with pytest.raises(StaleFenceError):
        authority.verify(token + "x", now=now)
    with pytest.raises(StaleFenceError, match="expired"):
        authority.verify(token, now=now + timedelta(seconds=11))


def test_agent_persists_monotonic_fence_epochs() -> None:
    epochs = FenceEpochs()
    epochs.accept((FenceClaim(gpu_uuid="gpu-0", epoch=2),))
    with pytest.raises(StaleFenceError):
        epochs.accept((FenceClaim(gpu_uuid="gpu-0", epoch=2),))
    epochs.accept((FenceClaim(gpu_uuid="gpu-0", epoch=3),))
