"""Attempt-scoped imports without unloading the actor's model dependencies.

Python's import cache and search path are process-global. Local (non-Ray) actors
therefore serialize source scopes; independent Ray actor processes still run in
parallel. This is import hygiene for trusted experiments, not a code sandbox.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock, local
from types import ModuleType
from typing import Any

_LOCK = RLock()
_ACTIVE = local()


def _under_root(value: Any, root: Path) -> bool:
    if not isinstance(value, str) or not value or value in ("built-in", "frozen"):
        return False
    try:
        candidate = Path(value)
        # Most entries are installed libraries. Avoid filesystem lookups for
        # each of their files on every job (notably on network filesystems).
        if candidate.is_absolute() and not candidate.is_relative_to(root):
            return False
        return candidate.resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def _belongs_to_source(module: Any, root: Path) -> bool:
    if not isinstance(module, ModuleType):
        return False
    attributes = vars(module)
    if _under_root(attributes.get("__file__"), root):
        return True
    specification = attributes.get("__spec__")
    if specification is not None and _under_root(specification.origin, root):
        return True
    return any(_under_root(path, root) for path in attributes.get("__path__", ()))


class SourceImports:
    """Temporarily install one source tree and restore shadowed modules exactly.

    ``retain`` is for the public resolver's returned callable: its globals and
    module identities remain consistent between resolution and invocation, but
    are never left installed in the warm actor between jobs.
    """

    def __init__(self, root: Path | str, *, retain: bool = False) -> None:
        self.root = Path(root).resolve()
        self.retain = retain
        self.modules: dict[str, Any] = {}

    def _names(self) -> set[str]:
        names = set()
        if self.root.is_dir():
            for child in self.root.iterdir():
                if child.is_dir() and child.name.isidentifier():
                    # Includes PEP 420 namespace packages (no __init__.py).
                    names.add(child.name)
                elif child.is_file():
                    for suffix in importlib.machinery.all_suffixes():
                        if child.name.endswith(suffix):
                            name = child.name[: -len(suffix)]
                            if name.isidentifier():
                                names.add(name)
                            break
        return names

    @contextmanager
    def activate(self) -> Iterator[None]:
        with _LOCK:
            names = self._names()
            displaced = {
                name: module
                for name, module in tuple(sys.modules.items())
                if name.split(".", 1)[0] in names
                or name in self.modules
                or _belongs_to_source(module, self.root)
            }
            path_object, previous_path = sys.path, sys.path[:]
            previous_scope = getattr(_ACTIVE, "scope", None)
            try:
                for name in displaced:
                    sys.modules.pop(name, None)
                sys.modules.update(self.modules)
                sys.path.insert(0, str(self.root))
                importlib.invalidate_caches()
                _ACTIVE.scope = self
                yield
            finally:
                owned = {
                    name: module
                    for name, module in tuple(sys.modules.items())
                    if name.split(".", 1)[0] in names
                    or name in self.modules
                    or _belongs_to_source(module, self.root)
                }
                for name in owned:
                    sys.modules.pop(name, None)
                sys.modules.update(displaced)
                self.modules = owned if self.retain else {}
                path_object[:] = previous_path
                sys.path = path_object
                _ACTIVE.scope = previous_scope
                # Do not retain finders (including namespace package paths) for
                # expired attempt directories in a long-lived worker.
                for path in tuple(sys.path_importer_cache):
                    if _under_root(path, self.root):
                        sys.path_importer_cache.pop(path, None)
                importlib.invalidate_caches()


@contextmanager
def source_imports(root: Path | str | None) -> Iterator[None]:
    """Keep imports active through invocation, awaiting, and actor cleanup."""
    if root is None:
        yield
        return
    with _LOCK:
        active = getattr(_ACTIVE, "scope", None)
        if active is not None and active.root == Path(root).resolve():
            yield
        else:
            with SourceImports(root).activate():
                yield
