"""Experiment entrypoint resolution and invocation."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, Protocol, cast

from .context import ExperimentContext


class Experiment(Protocol):
    def __call__(self, context: ExperimentContext, **kwargs: Any) -> Any: ...


_SOURCE_IMPORT_LOCK = RLock()


def _split_file_reference(reference: str) -> tuple[str, str] | None:
    path, separator, attribute = reference.rpartition(":")
    if separator and path.endswith(".py"):
        if not attribute or "." in attribute:
            raise ValueError("file entrypoint must name one top-level callable")
        return path, attribute
    if reference.endswith(".py"):
        return reference, "run"
    return None


def _resolve_file_entrypoint(
    reference: str, attribute: str, source_root: Path | str | None
) -> Experiment:
    if source_root is None:
        raise ValueError("a source_root is required for a .py entrypoint")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or ".." in logical.parts or logical.suffix != ".py":
        raise ValueError("file entrypoint must be a safe source-root-relative .py path")
    root = Path(source_root).resolve()
    candidate = (root / Path(*logical.parts)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("file entrypoint resolves outside source_root")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    digest = hashlib.sha256(str(candidate).encode() + b"\0" + candidate.read_bytes()).hexdigest()
    module_name = f"_topodrome_experiment_{digest[:24]}"
    specification = importlib.util.spec_from_file_location(module_name, candidate)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load experiment entrypoint {reference!r}")
    module = importlib.util.module_from_spec(specification)
    # Source roots may contain ordinary sibling modules. Isolate the sys.path
    # mutation and always restore it; the loaded experiment keeps its globals.
    with _SOURCE_IMPORT_LOCK:
        sys.modules[module_name] = module
        sys.path.insert(0, str(root))
        try:
            specification.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        finally:
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass
    function = getattr(module, attribute, None)
    if not callable(function):
        sys.modules.pop(module_name, None)
        raise TypeError(f"entrypoint {reference!r} does not define callable {attribute!r}")
    return cast(Experiment, function)


def resolve_entrypoint(reference: str, *, source_root: Path | str | None = None) -> Experiment:
    """Resolve a relative Python source file or ``package.module:function``.

    Relative files default to a top-level ``run`` callable and are contained to
    ``source_root`` after symlink resolution.
    """

    if not isinstance(reference, str) or not reference:
        raise ValueError("entrypoint must not be empty")
    file_reference = _split_file_reference(reference)
    if file_reference is not None:
        path, attribute = file_reference
        return _resolve_file_entrypoint(path, attribute, source_root)

    if reference.count(":") != 1:
        raise ValueError("entrypoint must use 'package.module:function' syntax")
    module_name, attribute_name = reference.split(":", 1)
    if not module_name or not attribute_name or "." in attribute_name:
        raise ValueError("entrypoint must name one module-level callable")
    function = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(function):
        raise TypeError(f"entrypoint {reference!r} is not callable")
    return cast(Experiment, function)


async def _await_experiment(value: Any) -> Any:
    return await value


def _finish_awaitable(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_experiment(value))
    close = getattr(value, "close", None)
    if close is not None:
        close()
    raise RuntimeError(
        "a synchronous model actor cannot run an async experiment from an active event loop"
    )


def invoke_experiment(
    experiment: Experiment | Any,
    context: ExperimentContext,
    kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Invoke a callable or an object exposing ``run(context, **kwargs)``."""

    target: Callable[..., Any]
    if isinstance(experiment, str):
        target = resolve_entrypoint(experiment, source_root=context.workspace)
    elif callable(experiment):
        target = experiment
    else:
        candidate = getattr(experiment, "run", None)
        if not callable(candidate):
            raise TypeError("experiment must be callable or expose a callable run method")
        target = cast(Callable[..., Any], candidate)
    return _finish_awaitable(target(context, **dict(kwargs or {})))
