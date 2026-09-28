from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from topodrome.runtime import (
    HuggingFaceModelAcquirer,
    ModelKey,
    ModelLoadError,
    ModelRegistration,
    TransformersLoader,
)

EXACT_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"


def model_key() -> ModelKey:
    return ModelKey(
        alias="meta-llama-3-8b-instruct",
        revision="exact-commit",
        backend="transformers",
        backend_version="4.48.0",
        dtype="bfloat16",
    )


class _Model:
    def __init__(self) -> None:
        self.device = None
        self.evaluating = False

    def to(self, device: str) -> _Model:
        self.device = device
        return self

    def eval(self) -> None:
        self.evaluating = True


class _Factory:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, source: str, **kwargs: Any) -> Any:
        self.calls.append((source, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def install_fake_backends(monkeypatch: pytest.MonkeyPatch) -> tuple[_Factory, _Factory, list[str]]:
    empty_cache_calls: list[str] = []
    torch = ModuleType("torch")
    torch.bfloat16 = object()
    torch.float16 = object()
    torch.float32 = object()
    torch.cuda = SimpleNamespace(empty_cache=lambda: empty_cache_calls.append("empty"))
    model_factory = _Factory(_Model())
    tokenizer_factory = _Factory(object())
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = model_factory
    transformers.AutoTokenizer = tokenizer_factory
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return model_factory, tokenizer_factory, empty_cache_calls


def test_transformers_loader_is_lazy_local_and_disposable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_factory, tokenizer_factory, empty_cache_calls = install_fake_backends(monkeypatch)
    registration = ModelRegistration(key=model_key(), local_path=tmp_path)
    loader = TransformersLoader(registration)

    loaded = loader(model_key())

    model_source, model_options = model_factory.calls[0]
    tokenizer_source, tokenizer_options = tokenizer_factory.calls[0]
    assert model_source == tokenizer_source == str(tmp_path)
    assert model_options["local_files_only"] is True
    assert "revision" not in model_options
    assert "token" not in model_options
    assert tokenizer_options["local_files_only"] is True
    assert loaded.model.device == "cuda:0"
    assert loaded.model.evaluating
    assert loaded.close is not None
    loaded.close()
    assert empty_cache_calls == ["empty"]


def test_transformers_loader_shards_across_exact_gpu_only_memory_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_factory, _tokenizer_factory, _empty_cache_calls = install_fake_backends(monkeypatch)
    model_factory.result.hf_device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": "cuda:0",
        "model.layers.1": 1,
        "lm_head": "cuda:1",
    }
    loader = TransformersLoader(
        ModelRegistration(key=model_key(), local_path=tmp_path)
    ).for_model_sharding(
        degree=2,
        strategy="balanced",
        max_memory_bytes_per_gpu=10 * 1024**3,
    )

    loaded = loader(model_key())

    options = model_factory.calls[0][1]
    assert options["device_map"] == "balanced"
    assert options["max_memory"] == {0: 10 * 1024**3, 1: 10 * 1024**3}
    assert "cpu" not in options["max_memory"]
    assert "disk" not in options["max_memory"]
    assert options["low_cpu_mem_usage"] is True
    assert loaded.model.device is None  # ``.to`` must not collapse the shards.


@pytest.mark.parametrize(
    "device_map",
    [
        {"model": 0},
        {"model": 0, "lm_head": "cpu"},
        {"model": 0, "lm_head": "disk"},
        {"model": 0, "lm_head": 2},
    ],
)
def test_transformers_loader_rejects_incomplete_or_offloaded_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_map: dict[str, object],
) -> None:
    model_factory, _tokenizer_factory, _empty_cache_calls = install_fake_backends(monkeypatch)
    model_factory.result.hf_device_map = device_map
    loader = TransformersLoader(
        ModelRegistration(key=model_key(), local_path=tmp_path)
    ).for_model_sharding(
        degree=2,
        strategy="balanced",
        max_memory_bytes_per_gpu=10 * 1024**3,
    )

    with pytest.raises(ModelLoadError, match="device map"):
        loader(model_key())


def test_loader_configuration_cannot_override_topodrome_device_map(tmp_path: Path) -> None:
    registration = ModelRegistration(
        key=model_key(),
        local_path=tmp_path,
        loader_options={"device_map": "auto"},
    )
    with pytest.raises(ValueError, match="explicit model parallelism"):
        TransformersLoader(registration)


def test_loader_error_redacts_physical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_backends(monkeypatch)
    transformers = sys.modules["transformers"]
    transformers.AutoModelForCausalLM = _Factory(RuntimeError(f"failure at {tmp_path}"))
    loader = TransformersLoader(ModelRegistration(key=model_key(), local_path=tmp_path))

    with pytest.raises(ModelLoadError) as caught:
        loader(model_key())

    assert str(tmp_path) not in str(caught.value)


def test_hf_acquirer_consumes_secret_only_inside_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "model"
    calls: list[dict[str, Any]] = []
    credential = uuid4().hex
    hub = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        destination.mkdir()
        (destination / "config.json").write_text("{}")
        (destination / "tokenizer.json").write_text("{}")
        (destination / "model.safetensors").write_bytes(b"weights")
        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
        return str(destination)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    acquirer = HuggingFaceModelAcquirer(
        destination=destination,
        token_provider=lambda: credential,
    )
    registration = ModelRegistration(key=model_key(), origin="meta-llama/Meta-Llama-3-8B-Instruct")

    acquired = acquirer.acquire(registration)

    assert calls[0]["token"] == credential
    assert calls[0]["endpoint"] == "https://hf-mirror.com"
    assert calls[0]["revision"] == "exact-commit"
    assert acquired.local_path == destination
    assert acquired.origin == registration.origin
    assert credential not in repr(acquirer)
    assert "HF_ENDPOINT" not in os.environ


def test_hf_acquirer_records_exact_revision_for_offline_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "model"
    hub = ModuleType("huggingface_hub")

    def snapshot_download(**_kwargs: Any) -> str:
        files = {
            "config.json": b"{}",
            "tokenizer.json": b"{}",
            "model.safetensors": b"weights",
        }
        metadata_root = destination / ".cache" / "huggingface" / "download"
        for name, content in files.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            metadata = metadata_root / f"{name}.metadata"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            object_id = hashlib.sha256(content).hexdigest()
            metadata.write_text(
                f"{EXACT_REVISION}\n{object_id}\n1234.5\n",
                encoding="utf-8",
            )
        return str(destination)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    key = ModelKey(
        alias="model",
        revision=EXACT_REVISION,
        backend="transformers",
        backend_version="4.57.6",
        dtype="bfloat16",
    )

    acquired = HuggingFaceModelAcquirer(destination=destination).acquire(
        ModelRegistration(key=key, origin="owner/model")
    )

    assert acquired.payload_identity is not None
    assert (
        destination / ".cache" / "huggingface" / "trees" / f"{EXACT_REVISION}.json"
    ).is_file()


@pytest.mark.parametrize("field", ["token", "api-key", "db_password", "secret"])
def test_loader_options_cannot_smuggle_credentials_to_an_actor(field: str) -> None:
    with pytest.raises(ValueError):
        ModelRegistration(key=model_key(), loader_options={field: "credential"})
