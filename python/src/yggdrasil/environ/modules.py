"""Module path / distribution lookup helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Union

try:
    import importlib.metadata as ilm  # type: ignore
except Exception:  # pragma: no cover
    ilm = None  # type: ignore


__all__ = [
    "module_name_to_project_name",
    "packages_distributions_cached",
    "resolve_local_lib_path",
]


def packages_distributions_cached() -> Dict[str, List[str]]:
    """Memoized :func:`importlib.metadata.packages_distributions`.

    Walks every installed distribution exactly once per process and
    caches the mapping ``{top_level_module: [dist_name, ...]}``. Used
    by the jobs introspection and workspace-PyPI publishing paths,
    both of which call this on every staged task and would otherwise
    re-scan ``site-packages`` for every module they touch.
    """
    cached = getattr(packages_distributions_cached, "_cache", None)
    if cached is None:
        if ilm is None:
            cached = {}
        else:
            try:
                cached = dict(ilm.packages_distributions())
            except Exception:  # noqa: BLE001 — defensive: missing metadata
                cached = {}
        packages_distributions_cached._cache = cached  # type: ignore[attr-defined]
    return cached


MODULE_PROJECT_NAMES_ALIASES = {
    "yggdrasil": "ygg",
    "jwt": "PyJWT",
}


def module_name_to_project_name(module_name: str) -> str:
    """Map module import names to PyPI project names when they differ."""
    return MODULE_PROJECT_NAMES_ALIASES.get(module_name, module_name)


def resolve_local_lib_path(lib: Union[str, ModuleType]) -> Path:
    """
    Resolve a lib spec (path string, module name, or module object)
    into a concrete filesystem path.

    Package-walk rule:
    - If the resolved path is inside a Python package (dir containing __init__.py),
      walk upward and return the *top-most* directory that still contains __init__.py.
    - If not in a package context, return the resolved file/dir.
    """
    if isinstance(lib, ModuleType):
        mod_file = getattr(lib, "__file__", None)
        if not mod_file:
            raise ValueError(f"Module {lib.__name__!r} has no __file__; cannot determine path")
        path = Path(mod_file).resolve()
    else:
        p = Path(lib)
        if p.exists():
            path = p.resolve()
        else:
            try:
                mod = importlib.import_module(lib)
            except ImportError as e:
                raise ModuleNotFoundError(
                    f"'{lib}' is neither an existing path nor an importable module"
                ) from e
            mod_file = getattr(mod, "__file__", None)
            if not mod_file:
                raise ModuleNotFoundError(
                    f"Module {mod.__name__!r} has no __file__; cannot determine path"
                )
            path = Path(mod_file).resolve()

    start_dir = path.parent if path.is_file() else path

    top_pkg_dir: Optional[Path] = None
    current = start_dir

    while True:
        if (current / "__init__.py").exists():
            top_pkg_dir = current
            parent = current.parent
            if parent == current:
                break
            current = parent
        else:
            break

    return top_pkg_dir.resolve() if top_pkg_dir is not None else path
