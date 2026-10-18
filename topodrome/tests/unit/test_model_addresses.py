from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from topodrome.config import ModelCatalog, ModelDefinition
from topodrome.domain.errors import AdmissionRejectedError
from topodrome.domain.specs import JobSpec, KernelRequest, ModelCapability, ModelRequirement
from topodrome.runtime.model_key import ModelKey
from topodrome.runtime.model_payload import (
    ModelPayloadVerificationError,
    verify_transformers_payload,
)
from topodrome.runtime.model_registry import ModelRegistration, ModelRegistry, UnknownModelError
from topodrome.scheduler.placement import model_matches
from topodrome.scheduler.service import SchedulerService

pytestmark = pytest.mark.unit
REPO = "research/training-run-a"
FIRST = "a" * 40
SECOND = "b" * 40


def catalog():
    return ModelCatalog(
        models={
            REPO: ModelDefinition(
                repo_id=REPO,
                revision=FIRST,
                refs=("main", "step20000"),
                min_vram_gb=2,
                sites={"lab1": {"path": "/ssd/models/run/first"}},
                snapshots=(
                    {
                        "revision": SECOND,
                        "refs": ["step40000"],
                        "subfolder": "checkpoint/model",
                        "sites": {"lab1": {"path": "/ssd/models/run/second"}},
                    },
                ),
            )
        }
    )


def resolve(model):
    service = SchedulerService.__new__(SchedulerService)
    service.model_catalog = catalog()
    return service._resolve_model(
        JobSpec(
            entrypoint="run.py",
            model=model,
            resources={"gpu": 1, "min_vram_gb": 2},
        )
    )


@pytest.mark.parametrize("value", [REPO, {"repo_id": REPO}])
def test_repo_address_and_shorthand_pin_registered_default(value):
    spec = resolve(value)
    assert spec.model.repo_id == REPO
    assert spec.model.revision == FIRST
    assert spec.model.subfolder is None
    assert spec.resources.min_vram_gb == 2


@pytest.mark.parametrize("revision", [SECOND, "step40000"])
def test_explicit_checkpoint_and_subfolder_are_not_overwritten(revision):
    spec = resolve({"repo_id": REPO, "revision": revision, "subfolder": "checkpoint/model"})
    assert spec.model.revision == SECOND
    assert spec.model.subfolder == "checkpoint/model"
    assert spec.model.repo_id == REPO


@pytest.mark.parametrize(
    "selection",
    [
        {"repo_id": "other/not-registered"},
        {"repo_id": REPO, "revision": "unknown"},
        {"repo_id": REPO, "revision": SECOND},
        {"repo_id": REPO, "subfolder": "wrong"},
    ],
)
def test_unknown_repository_revision_or_subfolder_rejects_without_substitution(selection):
    with pytest.raises(AdmissionRejectedError, match="unregistered model"):
        resolve(selection)


def test_accepted_snapshot_does_not_follow_later_default_changes():
    spec = resolve({"repo_id": REPO, "revision": "main"})
    changed = catalog()
    changed.models[REPO].revision = SECOND
    assert spec.model.revision == FIRST
    assert resolve(None).model is None


@pytest.mark.parametrize(
    "bad",
    [
        "meta-llama-3-8b-instruct",
        "none",
        "owner",
        "/owner/repo",
        "owner/../repo",
        "owner/repo/extra",
        "owner/repo..",
        "owner/repo.git",
        "owner/repo\\bad",
        "owner/repo\n",
        " owner/repo",
        "/ssd/models/model",
        "owner/re--po",
        {"alias": "meta-llama-3-8b-instruct"},
        {"repo_id": REPO, "alias": "old"},
        {"repo_id": REPO, "revision": "configured"},
    ],
)
def test_old_alias_interface_and_unsafe_names_are_rejected(bad):
    with pytest.raises(ValidationError):
        JobSpec(entrypoint="run.py", model=bad)


@pytest.mark.parametrize("subfolder", ["", "/weights", "../weights", "a/../b", "a//b", "a\\b"])
def test_subfolder_is_canonical_and_cannot_escape_snapshot(subfolder):
    with pytest.raises(ValidationError, match="subfolder"):
        ModelRequirement(repo_id=REPO, subfolder=subfolder)


@pytest.mark.parametrize("host", ["huggingface.co", "hf-mirror.com"])
def test_model_page_urls_normalize_without_network(host):
    request = KernelRequest(model=f"https://{host}/{REPO}/tree/step40000/checkpoint/model")
    assert request.model.repo_id == REPO
    assert request.model.revision == "step40000"
    assert request.model.subfolder == "checkpoint/model"
    assert ModelRequirement.model_validate(f"https://{host}/{REPO}").revision is None


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/research/model",
        "https://other.invalid/research/model",
        "https://token@huggingface.co/research/model",
        "https://huggingface.co/research/model?token=x",
        "https://huggingface.co/research/model#readme",
        "https://huggingface.co/research/model/resolve/main",
        "https://huggingface.co/research/model/tree/main/../bad",
        "https://huggingface.co/research/model/tree/main/%2e%2e/bad",
    ],
)
def test_url_convenience_is_not_arbitrary_url_loading(url):
    with pytest.raises(ValidationError):
        ModelRequirement.model_validate(url)


def test_conflicting_url_revision_rejects():
    with pytest.raises(ValidationError, match="disagree"):
        ModelRequirement(repo_id=f"https://huggingface.co/{REPO}/tree/main", revision=SECOND)


def test_registered_labels_must_not_be_ambiguous():
    with pytest.raises(ValidationError, match="select one snapshot"):
        ModelDefinition(
            repo_id=REPO,
            revision=FIRST,
            refs=("main",),
            sites={},
            snapshots=({"revision": SECOND, "refs": ["main"], "sites": {}},),
        )


def test_remote_origin_requires_pinned_commit_not_moving_label():
    with pytest.raises(ValidationError, match="full commit"):
        ModelDefinition(repo_id=REPO, revision="main", sites={"lab1": {"origin": REPO}})


def test_different_snapshots_cannot_overwrite_one_cache_path():
    first = catalog().models[REPO]
    first.snapshots[0].sites["lab1"].path = first.sites["lab1"].path
    with pytest.raises(ValidationError, match="share a writable cache path"):
        ModelCatalog(models={REPO: first})


def test_subfolder_separates_registry_actor_keys_and_placement():
    first = ModelKey(REPO, FIRST, "transformers", "configured", "bfloat16")
    other = replace(first, subfolder="checkpoint/model")
    assert first.fingerprint != other.fingerprint
    assert ModelKey.from_mapping(other.as_dict()) == other
    registry = ModelRegistry([ModelRegistration(first), ModelRegistration(other)])
    assert registry.resolve(REPO, revision=FIRST).key == first
    assert registry.resolve(REPO, revision=FIRST, subfolder=other.subfolder).key == other
    with pytest.raises(UnknownModelError):
        registry.resolve(REPO, revision=FIRST, subfolder="missing")
    capability = ModelCapability(**first.as_dict(), availability="cached")
    wanted = JobSpec(entrypoint="run.py", model=other.as_dict())
    assert not model_matches(capability, wanted)
    assert model_matches(capability, JobSpec(entrypoint="run.py", model=first.as_dict()))
    assert (
        capability.identity_tuple()
        != capability.model_copy(update={"subfolder": other.subfolder}).identity_tuple()
    )


def test_payload_verification_uses_snapshot_root_manifest_and_selected_subfolder(tmp_path: Path):
    content = {"config.json": b"{}", "tokenizer.json": b"{}", "model.safetensors": b"weights"}
    folder = tmp_path / "checkpoint/model"
    folder.mkdir(parents=True)
    entries = {}
    for name, data in content.items():
        (folder / name).write_bytes(data)
        entries[f"checkpoint/model/{name}"] = {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "lfs_sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest = tmp_path / ".cache/huggingface/trees" / f"{FIRST}.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"format_version": 1, "files": entries}))
    payload = verify_transformers_payload(
        tmp_path, revision=FIRST, subfolder="checkpoint/model", verify_lfs_sha256=True
    )
    assert payload.weight_files == ("model.safetensors",)
    with pytest.raises(ModelPayloadVerificationError):
        verify_transformers_payload(tmp_path, revision=FIRST)
