"""Site-private model configuration and public capability reporting."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from .model_key import ModelKey


class ModelRegistryError(RuntimeError):
    """Base exception for invalid or ambiguous registry operations."""


class UnknownModelError(KeyError, ModelRegistryError):
    pass


class AmbiguousModelError(ModelRegistryError):
    pass


class DuplicateModelError(ModelRegistryError):
    pass


class ModelAvailability(StrEnum):
    WARM = "warm"
    CACHED = "cached"
    OBTAINABLE = "obtainable"


_SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "key",
    "secret",
    "sig",
    "signature",
    "token",
}


def _looks_sensitive(name: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    return normalized in _SENSITIVE_QUERY_KEYS or normalized.endswith(
        ("_password", "_secret", "_token", "_api_key")
    )


def _validate_public_origin(origin: str | None) -> str | None:
    if origin is None:
        return None
    if not isinstance(origin, str) or not origin.strip():
        raise ValueError("origin must be a non-empty string when supplied")
    origin = origin.strip()
    parsed = urlsplit(origin)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("origin must not contain URL credentials")
    query_keys = {key.casefold() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)}
    if query_keys & _SENSITIVE_QUERY_KEYS:
        raise ValueError("origin must not contain credential query parameters")
    if parsed.scheme == "file" or Path(origin).is_absolute():
        raise ValueError("origin must be a public logical origin, not a local path")
    return origin


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """A logical model plus site-local loading information.

    ``local_path``, ``payload_identity`` and ``loader_options`` are
    intentionally excluded from the repr and from all public capability
    methods. Registry instances belong in the site process, never in scheduler
    messages. A local path is advertised as cached only after the site has
    attached a verified payload identity.
    """

    key: ModelKey
    origin: str | None = None
    local_path: Path | str | None = field(default=None, repr=False)
    payload_identity: str | None = field(default=None, repr=False)
    loader_options: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _validate_public_origin(self.origin))
        if self.local_path is not None:
            local_path = Path(self.local_path).expanduser()
            if not local_path.is_absolute():
                raise ValueError("local_path must be absolute")
            object.__setattr__(self, "local_path", local_path)
        if self.payload_identity is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.payload_identity
        ):
            raise ValueError("payload_identity must be a SHA-256 digest")
        loader_options = dict(self.loader_options)
        if any(_looks_sensitive(str(name)) for name in loader_options):
            raise ValueError(
                "loader_options must not contain credentials; use an ephemeral secret provider"
            )
        object.__setattr__(self, "loader_options", MappingProxyType(loader_options))

    def public_capability(self, availability: ModelAvailability) -> ModelCapability:
        return ModelCapability(
            key=self.key,
            origin=self.origin,
            availability=availability,
        )


@dataclass(frozen=True, slots=True)
class ModelCapability:
    """Path-free model information that may be sent to the scheduler."""

    key: ModelKey
    availability: ModelAvailability
    origin: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.key.as_dict(),
            "origin": self.origin,
            "availability": self.availability.value,
        }


class ModelRegistry:
    """Thread-safe registry of exact models available at a physical site."""

    def __init__(self, registrations: Iterable[ModelRegistration] = ()) -> None:
        self._lock = RLock()
        self._by_key: dict[ModelKey, ModelRegistration] = {}
        self._by_alias: dict[str, set[ModelKey]] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: ModelRegistration, *, replace: bool = False) -> None:
        if not isinstance(registration, ModelRegistration):
            raise TypeError("registration must be a ModelRegistration")
        with self._lock:
            if registration.key in self._by_key and not replace:
                raise DuplicateModelError(
                    f"model is already registered: {registration.key.alias!r} "
                    f"at revision {registration.key.revision!r}"
                )
            self._by_key[registration.key] = registration
            self._by_alias.setdefault(registration.key.alias, set()).add(registration.key)

    def unregister(self, key: ModelKey) -> ModelRegistration:
        with self._lock:
            try:
                registration = self._by_key.pop(key)
            except KeyError as exc:
                raise UnknownModelError(key.alias) from exc
            keys = self._by_alias[key.alias]
            keys.discard(key)
            if not keys:
                del self._by_alias[key.alias]
            return registration

    def get(self, key: ModelKey) -> ModelRegistration:
        """Return private loading configuration for trusted site code."""

        with self._lock:
            try:
                return self._by_key[key]
            except KeyError as exc:
                raise UnknownModelError(key.alias) from exc

    def resolve(
        self,
        alias: str,
        *,
        revision: str | None = None,
        backend: str | None = None,
        backend_version: str | None = None,
        dtype: str | None = None,
        quantization: str | None = None,
        adapter: str | None = None,
    ) -> ModelRegistration:
        """Resolve an alias, refusing to guess when multiple exact keys match."""

        with self._lock:
            candidates = list(self._by_alias.get(alias, ()))
            candidates = [
                key
                for key in candidates
                if (revision is None or key.revision == revision)
                and (backend is None or key.backend == backend)
                and (backend_version is None or key.backend_version == backend_version)
                and (dtype is None or key.dtype == dtype)
                and (quantization is None or key.quantization == quantization)
                and (adapter is None or key.adapter == adapter)
            ]
            if not candidates:
                raise UnknownModelError(alias)
            if len(candidates) != 1:
                revisions = sorted({key.revision for key in candidates})
                raise AmbiguousModelError(
                    f"model alias {alias!r} is ambiguous; exact revision/backend "
                    f"is required (matching revisions: {revisions!r})"
                )
            return self._by_key[candidates[0]]

    def public_capabilities(
        self,
        *,
        warm_keys: Iterable[ModelKey] = (),
        cached_keys: Iterable[ModelKey] = (),
    ) -> tuple[ModelCapability, ...]:
        """Return deterministic capability data without site-local paths."""

        warm = set(warm_keys)
        cached = set(cached_keys)
        with self._lock:
            registrations = sorted(
                self._by_key.values(),
                key=lambda item: (
                    item.key.alias,
                    item.key.revision,
                    item.key.fingerprint,
                ),
            )
            capabilities = []
            for registration in registrations:
                if registration.key in warm:
                    availability = ModelAvailability.WARM
                elif registration.key in cached or (
                    registration.local_path is not None
                    and registration.payload_identity is not None
                ):
                    availability = ModelAvailability.CACHED
                else:
                    availability = ModelAvailability.OBTAINABLE
                capabilities.append(registration.public_capability(availability))
            return tuple(capabilities)

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_key)
