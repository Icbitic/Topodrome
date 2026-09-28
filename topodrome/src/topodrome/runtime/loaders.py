"""Lazy model acquisition and Transformers loading at the site boundary."""

from __future__ import annotations

import gc
import importlib
import os
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from .actor import LoadedModel
from .model_key import ModelKey
from .model_payload import (
    ModelPayloadVerificationError,
    record_huggingface_local_dir_manifest,
    verify_transformers_payload,
)
from .model_registry import ModelRegistration


class ModelAcquisitionError(RuntimeError):
    pass


class ModelLoadError(RuntimeError):
    pass


SecretProvider = Callable[[], Any]
_ENVIRONMENT_LOCK = RLock()


def _secret_value(provider: SecretProvider | None) -> str | None:
    if provider is None:
        return None
    value = provider()
    if value is None:
        return None
    unwrap = getattr(value, "get_secret_value", None)
    if callable(unwrap):
        value = unwrap()
    if not isinstance(value, str) or not value:
        raise ModelAcquisitionError("model credential provider returned an invalid value")
    return value


@contextmanager
def _hf_endpoint(endpoint: str | None) -> Iterator[None]:
    """Temporarily scope HF_ENDPOINT to an acquisition/load operation."""

    if endpoint is None:
        yield
        return
    parsed = urlsplit(endpoint)
    sensitive_query = {
        "access_token",
        "api_key",
        "auth",
        "key",
        "secret",
        "signature",
        "token",
    }
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or {key.casefold() for key, _ in parse_qsl(parsed.query)} & sensitive_query
    ):
        raise ValueError("Hugging Face endpoint must be an HTTP(S) URL")
    with _ENVIRONMENT_LOCK:
        previous = os.environ.get("HF_ENDPOINT")
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = previous


def _credential_free_options(name: str, options: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(options)
    sensitive_suffixes = ("password", "secret", "token", "api_key")
    if any(str(key).casefold().replace("-", "_").endswith(sensitive_suffixes) for key in values):
        raise ValueError(f"{name} must not contain credentials; use token_provider")
    return values


@dataclass(slots=True)
class HuggingFaceModelAcquirer:
    """Stage an exact Hub revision before a credential-free actor starts.

    The token provider is invoked only inside :meth:`acquire`, is excluded from
    repr, and is never copied into the returned registration.  Call this from
    the site model service, then pass a :class:`TransformersLoader` built from
    the returned local registration to Ray.
    """

    destination: Path = field(repr=False)
    endpoint: str = "https://hf-mirror.com"
    token_provider: SecretProvider | None = field(default=None, repr=False)
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()
    verify_payload: bool = True

    def __post_init__(self) -> None:
        destination = Path(self.destination).expanduser()
        if not destination.is_absolute():
            raise ValueError("model acquisition destination must be absolute")
        self.destination = destination

    @property
    def contains_credentials(self) -> bool:
        return self.token_provider is not None

    def acquire(self, registration: ModelRegistration) -> ModelRegistration:
        registered_path = None if registration.local_path is None else Path(registration.local_path)
        if registered_path is not None:
            try:
                verified = verify_transformers_payload(
                    registered_path,
                    revision=registration.key.revision,
                    verify_lfs_sha256=bool(
                        re.fullmatch(r"[0-9a-f]{40,64}", registration.key.revision)
                    ),
                )
            except ModelPayloadVerificationError:
                pass
            else:
                return ModelRegistration(
                    key=registration.key,
                    origin=registration.origin,
                    local_path=registered_path,
                    payload_identity=verified.identity_digest,
                    loader_options=registration.loader_options,
                )
        if registration.origin is None:
            raise ModelAcquisitionError(
                f"model {registration.key.alias!r} has no obtainable origin"
            )
        token = _secret_value(self.token_provider)
        try:
            with _hf_endpoint(self.endpoint):
                hub = importlib.import_module("huggingface_hub")
                snapshot_download = hub.snapshot_download
                arguments: dict[str, Any] = {
                    "repo_id": registration.origin,
                    "revision": registration.key.revision,
                    "local_dir": str(self.destination),
                    "token": token,
                    "endpoint": self.endpoint.rstrip("/"),
                }
                if self.allow_patterns:
                    arguments["allow_patterns"] = list(self.allow_patterns)
                if self.ignore_patterns:
                    arguments["ignore_patterns"] = list(self.ignore_patterns)
                downloaded = Path(snapshot_download(**arguments))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # Hub exceptions frequently include URLs or local paths.  Keep
            # those details out of scheduler-visible error strings.
            raise ModelAcquisitionError(
                f"failed to acquire model {registration.key.alias!r} at the configured revision"
            ) from None
        finally:
            token = None

        local_path = downloaded if downloaded.is_absolute() else self.destination
        payload_identity: str | None = None
        if self.verify_payload:
            try:
                if re.fullmatch(r"[0-9a-f]{40,64}", registration.key.revision):
                    record_huggingface_local_dir_manifest(
                        local_path,
                        revision=registration.key.revision,
                    )
                verified = verify_transformers_payload(
                    local_path,
                    revision=registration.key.revision,
                    verify_lfs_sha256=bool(
                        re.fullmatch(r"[0-9a-f]{40,64}", registration.key.revision)
                    ),
                )
            except ModelPayloadVerificationError:
                raise ModelAcquisitionError(
                    f"acquired model {registration.key.alias!r} failed payload verification"
                ) from None
            payload_identity = verified.identity_digest
        return ModelRegistration(
            key=registration.key,
            origin=registration.origin,
            local_path=local_path,
            payload_identity=payload_identity,
            loader_options=registration.loader_options,
        )


_DTYPE_ATTRIBUTES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
}


@dataclass(slots=True)
class TransformersLoader:
    """Load a registered model without importing Torch at package import time."""

    registration: ModelRegistration
    device: str = "cuda:0"
    local_files_only: bool | None = None
    trust_remote_code: bool = False
    endpoint: str | None = None
    token_provider: SecretProvider | None = field(default=None, repr=False)
    model_class: str = "AutoModelForCausalLM"
    tokenizer_class: str = "AutoTokenizer"
    model_kwargs: Mapping[str, Any] = field(default_factory=dict, repr=False)
    tokenizer_kwargs: Mapping[str, Any] = field(default_factory=dict, repr=False)
    sharding_degree: int | None = None
    sharding_strategy: str | None = None
    max_memory_bytes_per_gpu: int | None = None

    def __post_init__(self) -> None:
        if self.registration.key.backend.casefold() != "transformers":
            raise ValueError("TransformersLoader requires the transformers backend")
        if not self.device:
            raise ValueError("device must not be empty")
        if self.local_files_only is None:
            self.local_files_only = self.registration.local_path is not None
        if self.local_files_only and self.registration.local_path is None:
            raise ValueError("local_files_only requires a registered local_path")
        if self.registration.local_path is None and self.registration.origin is None:
            raise ValueError("model registration has neither local_path nor origin")
        self.model_kwargs = _credential_free_options("model_kwargs", self.model_kwargs)
        self.tokenizer_kwargs = _credential_free_options("tokenizer_kwargs", self.tokenizer_kwargs)
        controlled_options = {"device_map", "max_memory"}
        if controlled_options & {
            str(key) for key in (*self.registration.loader_options, *self.model_kwargs)
        }:
            raise ValueError(
                "device_map and max_memory are controlled by the explicit "
                "model parallelism contract"
            )
        if self.sharding_degree is None:
            if self.sharding_strategy is not None or self.max_memory_bytes_per_gpu is not None:
                raise ValueError("incomplete model-sharding loader configuration")
        else:
            if self.sharding_degree < 2:
                raise ValueError("model sharding requires at least two GPUs")
            if self.sharding_strategy not in {
                "balanced",
                "balanced_low_0",
                "sequential",
            }:
                raise ValueError("unsupported Transformers device-map strategy")
            if self.max_memory_bytes_per_gpu is None or self.max_memory_bytes_per_gpu <= 0:
                raise ValueError("model sharding requires a positive per-GPU memory limit")

    @property
    def contains_credentials(self) -> bool:
        return self.token_provider is not None

    def for_model_sharding(
        self,
        *,
        degree: int,
        strategy: str,
        max_memory_bytes_per_gpu: int,
    ) -> TransformersLoader:
        """Return a credential-equivalent loader with a GPU-only device map.

        Supplying only integer CUDA keys to Transformers/Accelerate prevents an
        oversized model from silently spilling layers into CPU RAM or disk.
        """

        return replace(
            self,
            sharding_degree=degree,
            sharding_strategy=strategy,
            max_memory_bytes_per_gpu=max_memory_bytes_per_gpu,
        )

    def _dtype(self, torch: Any, key: ModelKey) -> Any:
        if key.dtype.casefold() == "auto":
            return "auto"
        attribute = _DTYPE_ATTRIBUTES.get(key.dtype.casefold())
        if attribute is None:
            raise ModelLoadError(f"unsupported dtype {key.dtype!r} for model {key.alias!r}")
        return getattr(torch, attribute)

    def __call__(self, key: ModelKey) -> LoadedModel:
        if key != self.registration.key:
            raise ModelLoadError("loader was invoked with an incompatible logical model key")
        local_path = (
            None if self.registration.local_path is None else Path(self.registration.local_path)
        )
        if local_path is not None and not local_path.exists():
            raise ModelLoadError(f"local files for model {key.alias!r} are unavailable")
        if key.quantization and "quantization_config" not in self.model_kwargs:
            raise ModelLoadError(f"model {key.alias!r} requires an explicit quantization_config")
        source = str(local_path) if local_path is not None else self.registration.origin
        assert source is not None
        token = _secret_value(self.token_provider)
        try:
            with _hf_endpoint(self.endpoint):
                torch = importlib.import_module("torch")
                transformers = importlib.import_module("transformers")
                model_type = getattr(transformers, self.model_class)
                tokenizer_type = getattr(transformers, self.tokenizer_class)
                shared: dict[str, Any] = {
                    "local_files_only": bool(self.local_files_only),
                    "trust_remote_code": self.trust_remote_code,
                }
                if local_path is None:
                    shared["revision"] = key.revision
                if token is not None:
                    shared["token"] = token

                tokenizer_arguments = {**shared, **self.tokenizer_kwargs}
                model_arguments = {
                    **shared,
                    "torch_dtype": self._dtype(torch, key),
                    **dict(self.registration.loader_options),
                    **self.model_kwargs,
                }
                if self.sharding_degree is not None:
                    assert self.sharding_strategy is not None
                    assert self.max_memory_bytes_per_gpu is not None
                    model_arguments.update(
                        {
                            "device_map": self.sharding_strategy,
                            "max_memory": {
                                index: self.max_memory_bytes_per_gpu
                                for index in range(self.sharding_degree)
                            },
                            "low_cpu_mem_usage": True,
                        }
                    )
                tokenizer = tokenizer_type.from_pretrained(source, **tokenizer_arguments)
                model = model_type.from_pretrained(source, **model_arguments)
                if self.sharding_degree is None:
                    model = model.to(self.device)
                else:
                    _verify_exact_gpu_device_map(model, degree=self.sharding_degree)
                evaluate = getattr(model, "eval", None)
                if callable(evaluate):
                    evaluate()
        except (KeyboardInterrupt, SystemExit):
            raise
        except ModelLoadError:
            raise
        except Exception:
            # Do not propagate backend messages containing physical paths or
            # authenticated Hub URLs into attempt events.
            raise ModelLoadError(
                f"failed to load model {key.alias!r} at the configured revision"
            ) from None
        finally:
            token = None

        def close() -> None:
            nonlocal model, tokenizer
            model = None
            tokenizer = None
            gc.collect()
            cuda = getattr(torch, "cuda", None)
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()

        return LoadedModel(model=model, tokenizer=tokenizer, device=self.device, close=close)


def _verify_exact_gpu_device_map(model: Any, *, degree: int) -> None:
    """Fail closed unless the loaded model actually occupies every leased GPU."""

    raw = getattr(model, "hf_device_map", None)
    if not isinstance(raw, Mapping) or not raw:
        raise ModelLoadError("model backend did not expose a verified sharded device map")
    devices: set[int] = set()
    for raw_device in raw.values():
        if isinstance(raw_device, int):
            devices.add(raw_device)
            continue
        value = str(raw_device).casefold()
        if value.isdigit():
            devices.add(int(value))
            continue
        match = re.fullmatch(r"cuda:(\d+)", value)
        if match is not None:
            devices.add(int(match.group(1)))
            continue
        # Explicitly reject CPU/disk/meta offload and unknown accelerators.
        raise ModelLoadError("model device map escaped the leased CUDA GPU set")
    if devices != set(range(degree)):
        raise ModelLoadError("model device map did not use every leased CUDA GPU")


__all__ = [
    "HuggingFaceModelAcquirer",
    "ModelAcquisitionError",
    "ModelLoadError",
    "SecretProvider",
    "TransformersLoader",
]
