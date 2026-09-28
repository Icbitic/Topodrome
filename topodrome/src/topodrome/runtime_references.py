"""Validation for immutable, credential-free Ray runtime inputs."""

from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import PurePath

_PINNED_REQUIREMENT = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,198}[A-Za-z0-9])?"
    r"(?:\[[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*\])?"
    r"==[A-Za-z0-9][A-Za-z0-9._+!-]{0,199}"
    r"(?: --hash=sha256:[0-9a-f]{64})+$"
)
_CONTENT_REFERENCE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAY_PACKAGE_REFERENCE = re.compile(
    r"^gcs://_ray_pkg_[0-9a-f]{16,64}\.(?:zip|whl)$"
)


def validate_pinned_runtime_requirements(values: Collection[str]) -> tuple[str, ...]:
    """Require a bounded set of exact pip pins with allowed-content hashes.

    A version alone is not immutable: an index can replace the file served for
    that version.  Pip's per-requirement ``--hash`` syntax makes both the
    resolved version and every accepted distribution byte-identical.
    """

    requirements = tuple(str(value) for value in values)
    if len(requirements) > 256:
        raise ValueError("runtime pip requirements exceed the 256-item limit")
    if len(set(requirements)) != len(requirements):
        raise ValueError("runtime pip requirements must not contain duplicates")
    for requirement in requirements:
        if len(requirement) > 1000 or not _PINNED_REQUIREMENT.fullmatch(requirement):
            raise ValueError(
                "runtime pip requirements must use exact 'package==version "
                "--hash=sha256:<lowercase-digest>' pins"
            )
    return requirements


def validate_job_working_dir_reference(value: str | None) -> str | None:
    """Accept only a scheduler-owned, content-addressed source identity."""

    if value is not None and not _CONTENT_REFERENCE.fullmatch(value):
        raise ValueError("runtime working_dir_ref must be an exact lowercase sha256 reference")
    return value


def validate_published_ray_working_dir(value: str | None) -> str | None:
    """Accept only a Ray content package or a resolved site-local directory.

    The public job schema contains a SHA-256 identity.  Immediately before Ray
    actor creation, the site agent resolves that identity either to a local
    absolute tree or to Ray's own content-addressed GCS package URI.  Direct
    HTTP/object-store URLs remain mutable and are intentionally rejected.
    """

    if value is None:
        return None
    if _RAY_PACKAGE_REFERENCE.fullmatch(value):
        return value
    path = PurePath(value)
    if path.is_absolute() and ".." not in path.parts and "\x00" not in value:
        return value
    raise ValueError(
        "Ray working_dir must be a resolved absolute path or content-addressed gcs package"
    )


__all__ = [
    "validate_job_working_dir_reference",
    "validate_pinned_runtime_requirements",
    "validate_published_ray_working_dir",
]
