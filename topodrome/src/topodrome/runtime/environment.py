"""Credential-free process environments for site-local Ray execution."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Collection, Mapping, MutableMapping

from topodrome.runtime_references import (
    validate_pinned_runtime_requirements,
    validate_published_ray_working_dir,
)

# Ray is started by the credential-bearing site agent, so it must not inherit
# that process's environment wholesale.  Keep this deliberately explicit: these
# values affect Python, CUDA, collective communication, locale, certificates,
# and local cache placement, but none is an authentication credential.
_RAY_HEAD_ENVIRONMENT_KEYS = frozenset(
    {
        "CUDA_CACHE_DISABLE",
        "CUDA_CACHE_MAXSIZE",
        "CUDA_CACHE_PATH",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "CUDA_DEVICE_ORDER",
        "CUDA_HOME",
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_MODULE_LOADING",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "LOGNAME",
        "MALLOC_ARENA_MAX",
        "MKL_NUM_THREADS",
        "NCCL_ALGO",
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_IB_HCA",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_NSOCKS_PERTHREAD",
        "NCCL_P2P_DISABLE",
        "NCCL_PROTO",
        "NCCL_SHM_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_SOCKET_NTHREADS",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_REQUIRE_CUDA",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
        "PYTHONWARNINGS",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        "REQUESTS_CA_BUNDLE",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TOKENIZERS_PARALLELISM",
        "TORCH_CUDA_ARCH_LIST",
        "TRITON_CACHE_DIR",
        "TZ",
        "USER",
        "VIRTUAL_ENV",
        "XDG_CACHE_HOME",
    }
)

_CREDENTIAL_ENVIRONMENT_KEY = re.compile(
    r"(?:^|_)(?:ACCESS_KEY|API_KEY|AUTHORIZATION|CREDENTIALS?|PASSWORD|PASSWD|"
    r"PRIVATE_KEY|SECRETS?|TOKENS?)(?:$|_)",
    re.IGNORECASE,
)
_RESERVED_WORKER_ENVIRONMENT_KEY = re.compile(
    r"^(?:CUDA_VISIBLE_DEVICES|RAY_.+|TOPODROME_.+)$",
    re.IGNORECASE,
)

# Ray 2.58 accepts an import path for per-task/actor setup hooks, but does not
# upload arbitrary callable objects in a task/actor ``runtime_env``. Keep the
# allowlist manifest in an internal environment variable that the hook consumes
# and removes before model or experiment code starts.
RAY_ALLOWED_JOB_KEYS_ENVIRONMENT = "TOPODROME_RAY_ALLOWED_JOB_KEYS"
RAY_WORKER_ENVIRONMENT_SETUP_HOOK = (
    "topodrome.runtime.environment.sanitize_configured_ray_worker_environment"
)
RAY_DISABLE_FAILURE_SIGNAL_HANDLER_ENVIRONMENT = (
    "RAY_DISABLE_FAILURE_SIGNAL_HANDLER"
)
_EMBEDDED_CREDENTIAL = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://[^\s/@]+@|"
    r"(?:^|[?&;\s])(?:token|secret|password|credentials?|authorization|"
    r"api[_-]?key)\s*=)"
)


def configure_ray_driver_environment(
    environment: MutableMapping[str, str] | None = None,
) -> None:
    """Configure Ray before its driver library is imported.

    Ray's native failure handler treats the service-control ``SIGTERM`` as a
    crash and can abort a long-lived asyncio driver when the local Ray head is
    subsequently drained.  Topodrome owns graceful signal handling at the
    site-agent boundary, so the driver must leave fatal-signal handling to the
    host process.  This variable is intentionally *not* part of the Ray-head
    environment allowlist: the independently managed head retains Ray's own
    process-level handlers.
    """

    target = os.environ if environment is None else environment
    target["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    target[RAY_DISABLE_FAILURE_SIGNAL_HANDLER_ENVIRONMENT] = "1"


def validate_job_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Validate user-requested values before they enter a Ray runtime_env."""

    validated: dict[str, str] = {}
    for raw_key, raw_value in environment.items():
        key = str(raw_key)
        value = str(raw_value)
        if not key or "\x00" in key or "=" in key or "\x00" in value:
            raise ValueError("runtime environment contains an invalid name or value")
        if _CREDENTIAL_ENVIRONMENT_KEY.search(key):
            raise ValueError("runtime environment must not contain credentials")
        if _RESERVED_WORKER_ENVIRONMENT_KEY.match(key):
            raise ValueError(f"runtime environment key is reserved by Topodrome: {key}")
        validated[key] = value
    return validated


def validate_runtime_references(
    pip: Collection[str],
    working_dir: str | None,
) -> tuple[tuple[str, ...], str | None]:
    """Reject credentials embedded in package or working-directory references."""

    requirements = tuple(str(item) for item in pip)
    references = (*requirements, *((working_dir,) if working_dir is not None else ()))
    if any("\x00" in item or _EMBEDDED_CREDENTIAL.search(item) for item in references):
        raise ValueError("Ray runtime references must not contain credentials")
    return (
        validate_pinned_runtime_requirements(requirements),
        validate_published_ray_working_dir(working_dir),
    )


def minimal_ray_head_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the explicit, credential-free environment inherited by Ray.

    The site agent performs authenticated model acquisition before actor
    creation.  Consequently Ray needs runtime and accelerator configuration,
    but never the agent enrollment, fence, Hugging Face, Jupyter, or provider
    credentials present in the parent process.
    """

    values = os.environ if source is None else source
    environment = {
        key: values[key]
        for key in sorted(_RAY_HEAD_ENVIRONMENT_KEYS)
        if key in values and not _CREDENTIAL_ENVIRONMENT_KEY.search(key)
    }
    environment.setdefault("PATH", os.defpath)
    # Ray 2.58+ otherwise detects the ancestor ``uv run`` and tries to build a
    # second, incomplete environment for workers.
    environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    # A first driver otherwise makes Ray eagerly fork one idle Python worker
    # per logical CPU.  Large GPU hosts can have hundreds of CPUs, turning an
    # idle site agent into a multi-gigabyte background workload.  Ray still
    # starts workers on demand with this disabled.
    environment["RAY_prestart_worker_first_driver"] = "0"
    return environment


def sanitize_ray_worker_environment(
    environment: MutableMapping[str, str] | None = None,
    *,
    allowed_job_keys: Collection[str] = (),
) -> None:
    """Apply the Ray allowlist before a worker loads models or user code.

    Ray adds private ``RAY_*`` coordinates after the head starts.  Preserve
    those internal values while rejecting credential-shaped names.  Everything
    inherited from the site-agent side remains constrained to the same explicit
    runtime allowlist as the head process.
    """

    target = os.environ if environment is None else environment
    original = dict(target)
    sanitized = minimal_ray_head_environment(original)
    sanitized.update(
        {
            key: value
            for key, value in original.items()
            if key.startswith("RAY_") and not _CREDENTIAL_ENVIRONMENT_KEY.search(key)
        }
    )
    for key in allowed_job_keys:
        if (
            key in original
            and not _CREDENTIAL_ENVIRONMENT_KEY.search(key)
            and not _RESERVED_WORKER_ENVIRONMENT_KEY.match(key)
        ):
            sanitized[key] = original[key]
    sanitized["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    sanitized["RAY_prestart_worker_first_driver"] = "0"
    target.clear()
    target.update(sanitized)


def encode_ray_worker_allowed_job_keys(keys: Collection[str]) -> str:
    """Encode the non-secret key allowlist consumed by the importable hook."""

    return json.dumps(sorted(set(keys)), separators=(",", ":"))


def sanitize_configured_ray_worker_environment() -> None:
    """No-argument importable Ray hook using a fail-closed key manifest."""

    raw = os.environ.get(RAY_ALLOWED_JOB_KEYS_ENVIRONMENT, "[]")
    allowed: tuple[str, ...] = ()
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, list) and all(isinstance(key, str) for key in decoded):
            allowed = tuple(decoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    sanitize_ray_worker_environment(allowed_job_keys=allowed)


__all__ = [
    "RAY_ALLOWED_JOB_KEYS_ENVIRONMENT",
    "RAY_DISABLE_FAILURE_SIGNAL_HANDLER_ENVIRONMENT",
    "RAY_WORKER_ENVIRONMENT_SETUP_HOOK",
    "configure_ray_driver_environment",
    "encode_ray_worker_allowed_job_keys",
    "minimal_ray_head_environment",
    "sanitize_configured_ray_worker_environment",
    "sanitize_ray_worker_environment",
    "validate_job_environment",
    "validate_runtime_references",
]
