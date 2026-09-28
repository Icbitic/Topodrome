"""Stable, path-free identities for loaded models and model actors."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_ACTOR_COMPONENT = re.compile(r"[^a-zA-Z0-9_-]+")


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain NUL bytes")
    return value


def _optional_text(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _required_text(name, value)


@dataclass(frozen=True, slots=True, order=True)
class ModelKey:
    """Exact logical identity of a loadable model.

    A ``ModelKey`` deliberately contains no filesystem location.  This makes it
    safe to put in scheduler messages, logs, and Ray actor names.  The site-local
    :class:`~topodrome.runtime.model_registry.ModelRegistry` maps the key to a
    private path when it is time to load the model.
    """

    alias: str
    revision: str
    backend: str
    backend_version: str
    dtype: str = "auto"
    quantization: str | None = None
    adapter: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "alias", _required_text("alias", self.alias))
        object.__setattr__(self, "revision", _required_text("revision", self.revision))
        object.__setattr__(self, "backend", _required_text("backend", self.backend))
        object.__setattr__(
            self,
            "backend_version",
            _required_text("backend_version", self.backend_version),
        )
        object.__setattr__(self, "dtype", _required_text("dtype", self.dtype))
        object.__setattr__(
            self,
            "quantization",
            _optional_text("quantization", self.quantization),
        )
        object.__setattr__(self, "adapter", _optional_text("adapter", self.adapter))

    def as_dict(self) -> dict[str, str | None]:
        """Return the complete logical identity in a JSON-safe form."""

        return {
            "alias": self.alias,
            "revision": self.revision,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "adapter": self.adapter,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelKey:
        """Build a key from protocol/config data with strict field names."""

        return cls(
            alias=value["alias"],
            revision=value["revision"],
            backend=value["backend"],
            backend_version=value["backend_version"],
            dtype=value.get("dtype", "auto"),
            quantization=value.get("quantization"),
            adapter=value.get("adapter"),
        )

    @property
    def fingerprint(self) -> str:
        """A stable digest suitable for cache and actor identifiers."""

        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def short_name(self) -> str:
        alias = _ACTOR_COMPONENT.sub("_", self.alias).strip("_-").lower() or "model"
        return f"{alias[:40]}-{self.fingerprint[:16]}"


@dataclass(frozen=True, slots=True, order=True)
class ActorKey:
    """Identity of one warm model allocation on an exact physical GPU set.

    ``gpu_uuid`` remains the primary member for backward-compatible single-GPU
    actor slots. ``replica_gpu_uuids`` is ordered because Transformers device
    maps address the selected devices as local CUDA ordinals in that order.
    """

    site_id: str
    gpu_uuid: str
    model: ModelKey
    runtime_fingerprint: str = "legacy-runtime"
    replica_gpu_uuids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_id", _required_text("site_id", self.site_id))
        object.__setattr__(self, "gpu_uuid", _required_text("gpu_uuid", self.gpu_uuid))
        object.__setattr__(
            self,
            "runtime_fingerprint",
            _required_text("runtime_fingerprint", self.runtime_fingerprint),
        )
        members = self.replica_gpu_uuids or (self.gpu_uuid,)
        members = tuple(_required_text("replica_gpu_uuid", value) for value in members)
        if len(set(members)) != len(members):
            raise ValueError("replica_gpu_uuids must be unique")
        if self.gpu_uuid not in members:
            raise ValueError("primary gpu_uuid must belong to replica_gpu_uuids")
        object.__setattr__(self, "replica_gpu_uuids", members)

    @property
    def slot(self) -> tuple[str, str]:
        """Physical warm-allocation slot, distinct from an attempt lease."""

        return (self.site_id, self.gpu_uuid)

    @property
    def replica_slot(self) -> tuple[str, tuple[str, ...]]:
        """The complete ordered physical allocation owned by this process."""

        return self.site_id, self.replica_gpu_uuids

    def as_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "gpu_uuid": self.gpu_uuid,
            "model": self.model.as_dict(),
            "runtime_fingerprint": self.runtime_fingerprint,
            "replica_gpu_uuids": list(self.replica_gpu_uuids),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ActorKey:
        return cls(
            site_id=value["site_id"],
            gpu_uuid=value["gpu_uuid"],
            model=ModelKey.from_mapping(value["model"]),
            runtime_fingerprint=value.get("runtime_fingerprint", "legacy-runtime"),
            replica_gpu_uuids=tuple(value.get("replica_gpu_uuids") or (value["gpu_uuid"],)),
        )
