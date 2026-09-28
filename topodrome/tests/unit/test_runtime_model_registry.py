from __future__ import annotations

from pathlib import Path

import pytest

from topodrome.runtime import (
    ActorKey,
    AmbiguousModelError,
    ModelAvailability,
    ModelKey,
    ModelRegistration,
    ModelRegistry,
)
from topodrome.runtime.ray_actor import ray_gpu_resource_name


def model_key(*, revision: str = "0123456789abcdef") -> ModelKey:
    return ModelKey(
        alias="meta-llama-3-8b-instruct",
        revision=revision,
        backend="transformers",
        backend_version="4.48.0",
        dtype="bfloat16",
    )


def test_model_key_is_exact_stable_and_path_free() -> None:
    first = model_key()
    same = ModelKey.from_mapping(first.as_dict())
    changed = model_key(revision="fedcba9876543210")

    assert first == same
    assert first.fingerprint == same.fingerprint
    assert first != changed
    assert first.fingerprint != changed.fingerprint
    assert "path" not in first.as_dict()
    assert len(first.short_name) <= 57


def test_actor_key_separates_physical_allocation_from_model_identity() -> None:
    key = model_key()
    first = ActorKey("lab1", "GPU-aaaa", key)
    second = ActorKey("lab1", "GPU-bbbb", key)

    assert first.model == second.model
    assert first.slot != second.slot
    assert ActorKey.from_mapping(first.as_dict()) == first


def test_registry_never_advertises_private_model_path(tmp_path: Path) -> None:
    private_path = tmp_path / "private-model-cache"
    registration = ModelRegistration(
        key=model_key(),
        origin="meta-llama/Meta-Llama-3-8B-Instruct",
        local_path=private_path,
        payload_identity="a" * 64,
        loader_options={"internal_setting": "must-not-leak"},
    )
    registry = ModelRegistry([registration])

    public = registry.public_capabilities()[0]
    encoded = repr(public.as_dict())

    assert public.availability is ModelAvailability.CACHED
    assert str(private_path) not in encoded
    assert "must-not-leak" not in encoded
    assert str(private_path) not in repr(registration)
    assert registry.get(model_key()).local_path == private_path


def test_unverified_local_path_is_not_advertised_as_cached(tmp_path: Path) -> None:
    registry = ModelRegistry(
        [
            ModelRegistration(
                key=model_key(),
                origin="meta-llama/Meta-Llama-3-8B-Instruct",
                local_path=tmp_path / "partial",
            )
        ]
    )

    assert registry.public_capabilities()[0].availability is ModelAvailability.OBTAINABLE


@pytest.mark.parametrize(
    "origin",
    [
        "/ssd/models/Meta-Llama-3-8B-Instruct",
        "file:///ssd/models/model",
        "https://token@example.invalid/model",
        "https://example.invalid/model?token=secret",
    ],
)
def test_registry_rejects_origins_that_could_leak_site_paths_or_secrets(
    origin: str,
) -> None:
    with pytest.raises(ValueError):
        ModelRegistration(key=model_key(), origin=origin)


def test_registry_requires_exact_selection_when_alias_is_ambiguous() -> None:
    registry = ModelRegistry(
        [
            ModelRegistration(key=model_key(revision="one"), origin="owner/model"),
            ModelRegistration(key=model_key(revision="two"), origin="owner/model"),
        ]
    )

    with pytest.raises(AmbiguousModelError):
        registry.resolve("meta-llama-3-8b-instruct")
    assert registry.resolve("meta-llama-3-8b-instruct", revision="two").key.revision == "two"


def test_ray_resource_name_matches_site_agent_convention() -> None:
    assert ray_gpu_resource_name("GPU-ABCD:1234") == "topodrome_gpu_gpu_abcd_1234"
    with pytest.raises(ValueError):
        ray_gpu_resource_name("---")
