"""Conservative, offline weight replacement; never changes physical ownership.

All tensor names/shapes and the complete functional model configuration are
checked before the first write. A write failure is fatal to the actor: callers
must dispose it rather than expose a partly replaced checkpoint.
"""

from __future__ import annotations

import copy
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_payload import verify_transformers_payload

_VOLATILE_CONFIG = {"_name_or_path", "_commit_hash", "transformers_version"}
_CACHE_ATTRIBUTES = ("max_seq_len_cached", "original_max_seq_len", "_seq_len_cached")
_BUFFER_LIMIT = 64 * 1024**2
_HOOK_MAPS = (
    "_forward_hooks",
    "_forward_pre_hooks",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks_always_called",
)


def _module_state(model):
    return tuple(
        (
            name,
            type(module),
            id(getattr(module.forward, "__func__", module.forward)),
            id(getattr(module, "_compiled_call_impl", None)),
            tuple(
                (
                    attribute,
                    tuple(
                        (key, id(value)) for key, value in getattr(module, attribute, {}).items()
                    ),
                )
                for attribute in _HOOK_MAPS
            ),
        )
        for name, module in model.named_modules()
    )


def _config(value):
    return {key: item for key, item in value.items() if key not in _VOLATILE_CONFIG}


def _root(loader):
    registration = loader.registration
    if registration.local_path is None or loader.contains_credentials or loader.trust_remote_code:
        return None
    root = Path(registration.local_path)
    if registration.key.subfolder:
        root = root / registration.key.subfolder
    return root


def _read_config(root):
    value = json.loads((root / "config.json").read_text())
    if not isinstance(value, dict):
        raise ValueError("model configuration must be an object")
    return _config(value)


def _loader_options(loader):
    # Loader-level semantics must not change under an existing model/device map.
    return (
        loader.model_class,
        loader.tokenizer_class,
        loader.device,
        loader.sharding_degree,
        loader.sharding_strategy,
        loader.max_memory_bytes_per_gpu,
        dict(loader.registration.loader_options),
        dict(loader.model_kwargs),
        dict(loader.tokenizer_kwargs),
    )


@dataclass(slots=True)
class SwitchState:
    disk_config: dict[str, Any]
    live_config: dict[str, Any]
    options: Any
    buffers: dict[str, Any] = field(repr=False)
    cache_attributes: tuple[Any, ...] = field(repr=False)
    module_state: tuple[Any, ...] = field(repr=False)


def capture_switch_state(loader, model):
    """Keep only small initial buffers, never a second copy of model weights."""
    root = _root(loader)
    if root is None or not hasattr(model, "named_buffers") or not hasattr(model, "config"):
        return None
    if getattr(model, "is_quantized", False) or getattr(model, "_is_stateful", False):
        return None
    try:
        buffers = dict(model.named_buffers())
        if sum(value.numel() * value.element_size() for value in buffers.values()) > _BUFFER_LIMIT:
            return None
        initial = {name: value.detach().cpu().clone() for name, value in buffers.items()}
        attributes = tuple(
            (name, attribute, copy.deepcopy(getattr(module, attribute)))
            for name, module in model.named_modules()
            for attribute in _CACHE_ATTRIBUTES
            if isinstance(getattr(module, attribute, None), (int, float))
        )
        return SwitchState(
            _read_config(root),
            _config(model.config.to_dict()),
            _loader_options(loader),
            initial,
            attributes,
            _module_state(model),
        )
    except (AttributeError, OSError, ValueError, TypeError):
        return None


def _storage_key(tensor):
    return (
        str(tensor.device),
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
    )


def replace_weights(loader, loaded, previous_key):
    """Return False *without mutation* for an unsupported checkpoint layout."""
    state = loaded.switch_state
    key = loader.registration.key
    root = _root(loader)
    if (
        not isinstance(state, SwitchState)
        or root is None
        or previous_key is None
        or previous_key.dtype != key.dtype
        or previous_key.backend != key.backend
        or previous_key.backend_version != key.backend_version
        or key.quantization
        or key.adapter
        or previous_key.quantization
        or previous_key.adapter
        or _loader_options(loader) != state.options
        or _read_config(root) != state.disk_config
        or _config(loaded.model.config.to_dict()) != state.live_config
        or _module_state(loaded.model) != state.module_state
    ):
        return False
    verified = verify_transformers_payload(
        loader.registration.local_path,
        revision=key.revision,
        subfolder=key.subfolder,
    )
    if loader.registration.payload_identity and (
        verified.identity_digest != loader.registration.payload_identity
    ):
        raise ValueError("checkpoint no longer matches its registered payload")
    if any(not name.endswith(".safetensors") for name in verified.weight_files):
        return False  # Keep native Transformers handling for pickle/quantized/custom loaders.

    torch = importlib.import_module("torch")
    safe_open = importlib.import_module("safetensors").safe_open
    model = loaded.model
    tensors = model.state_dict(keep_vars=True)
    buffers = dict(model.named_buffers())
    if set(buffers) != set(state.buffers) or any(
        tuple(value.shape) != tuple(state.buffers[name].shape) for name, value in buffers.items()
    ):
        return False
    modules = dict(model.named_modules())
    if any(name not in modules for name, _, _ in state.cache_attributes):
        return False
    aliases = {}
    for name, tensor in tensors.items():
        if tensor.layout != torch.strided or str(tensor.device) == "meta":
            return False
        aliases.setdefault(_storage_key(tensor), set()).add(name)

    # Opening safetensors headers does not materialize a second CPU/GPU model.
    files = {}
    identities = {}
    for name in verified.weight_files:
        path = root / name
        stat = path.stat()
        identities[name] = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            for tensor_name in shard.keys():
                if tensor_name in files or tensor_name not in tensors:
                    return False
                if tuple(shard.get_slice(tensor_name).get_shape()) != tuple(
                    tensors[tensor_name].shape
                ):
                    return False
                files[tensor_name] = name
    if any(len(group.intersection(files)) != 1 for group in aliases.values()):
        return False

    # Prepare tokenizer/config before mutation too. A different tokenizer is
    # legitimate (e.g. instruction tuning), but is never reused by assumption.
    transformers = importlib.import_module("transformers")
    tokenizer = getattr(transformers, loader.tokenizer_class).from_pretrained(
        str(root),
        local_files_only=True,
        trust_remote_code=False,
        **dict(loader.tokenizer_kwargs),
    )
    generation_type = getattr(transformers, "GenerationConfig", None)
    generation = None
    if generation_type is not None and hasattr(model, "generation_config"):
        path = root / "generation_config.json"
        generation = (
            generation_type.from_dict(json.loads(path.read_text()))
            if path.is_file()
            else generation_type.from_model_config(model.config)
        )

    # From here any error requires actor disposal, not rollback to old identity.
    copied = set()
    # A returned experiment may have issued work on non-default CUDA streams.
    # Complete that old-generation work before touching any parameter storage.
    for device in loaded.devices or (loaded.device,):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
    with torch.no_grad():
        for name in verified.weight_files:
            path = root / name
            stat = path.stat()
            if identities[name] != (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ):
                raise ValueError("checkpoint changed during weight replacement")
            with safe_open(str(path), framework="pt", device="cpu") as shard:
                for tensor_name in shard.keys():
                    tensor = tensors[tensor_name]
                    identity = _storage_key(tensor)
                    if identity not in copied:
                        tensor.copy_(shard.get_tensor(tensor_name))
                        copied.add(identity)
            stat = path.stat()
            if identities[name] != (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ):
                raise ValueError("checkpoint changed during weight replacement")
        for name, initial in state.buffers.items():
            if name not in files:
                buffers[name].copy_(initial)
        for name, attribute, initial in state.cache_attributes:
            setattr(modules[name], attribute, initial)
        # GenerationMixin may retain an inference KV cache on the model object.
        if hasattr(model, "_cache"):
            delattr(model, "_cache")
        if generation is not None:
            model.generation_config = generation
        model.zero_grad(set_to_none=True)
        model.eval()
        for device in loaded.devices or (loaded.device,):
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
    loaded.tokenizer = tokenizer
    # Match a fresh local load's public metadata as well as its tensor bytes.
    # These fields are deliberately excluded from the functional-config guard.
    model.name_or_path = str(root)
    model.config._name_or_path = str(root)
    model.config._commit_hash = key.revision
    return True
