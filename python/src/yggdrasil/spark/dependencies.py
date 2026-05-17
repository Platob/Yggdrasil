"""Function-dependency scanning for :class:`Dataset`.

Lives in its own module — without a top-level ``pyspark`` import —
so the scan logic stays testable in lean environments (Spark
Connect clients, CI runners without a JVM, …) where
:mod:`yggdrasil.spark.frame` can't be imported as-is.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Callable

__all__ = ["function_top_modules"]


#: Cached union of stdlib + builtin module names. Filled lazily
#: on first call to :func:`_stdlib_modules`. Frozen so test
#: monkeypatches can swap in tighter sets without growing the
#: real cache.
_STDLIB_MODULES: "frozenset[str] | None" = None


def _stdlib_modules() -> frozenset[str]:
    """Return the standard-library / builtin name set."""
    global _STDLIB_MODULES
    if _STDLIB_MODULES is None:
        stdlib = getattr(sys, "stdlib_module_names", None) or set()
        _STDLIB_MODULES = frozenset(stdlib) | frozenset(sys.builtin_module_names)
    return _STDLIB_MODULES


def function_top_modules(fn: Callable[..., Any]) -> set[str]:
    """Return the set of top-level module names *fn* depends on.

    Walks the function's :attr:`__globals__` and looks at every
    module-typed entry, every plain object's ``__module__``, and
    the function's defining module itself. Returns the *top*
    package name (e.g. ``"yggdrasil"`` not ``"yggdrasil.spark"``)
    so the result is shaped for ``pip install`` /
    ``DatabricksEnv.withDependencies`` lookups.

    Standard library and builtin modules are filtered out
    (:data:`sys.stdlib_module_names`); ``__main__`` /
    ``__builtin__`` /  the empty string are dropped because pip
    can't install them and Spark Connect ships the driver's
    bytecode through cloudpickle already.

    Conservative by design: a false positive just means we
    install a lib we didn't strictly need; a false negative
    breaks UDF execution on the cluster.
    """
    stdlib = _stdlib_modules()
    found: set[str] = set()

    def _add(module_name: Any) -> None:
        if not isinstance(module_name, str) or not module_name:
            return
        top = module_name.split(".", 1)[0]
        if not top or top in ("__main__", "__builtin__", "builtins"):
            return
        if top in stdlib:
            return
        if top.startswith("_"):
            # Private modules (``_typeshed`` etc.) are stdlib-adjacent
            # implementation detail — skip.
            return
        found.add(top)

    _add(getattr(fn, "__module__", None))

    globs = getattr(fn, "__globals__", None) or {}
    for value in globs.values():
        if isinstance(value, ModuleType):
            _add(getattr(value, "__name__", None))
        else:
            _add(getattr(value, "__module__", None))

    # Walk closure cells too — captured outer-scope objects might
    # belong to libraries that don't appear in __globals__.
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue  # uninitialized cell
        if isinstance(value, ModuleType):
            _add(getattr(value, "__name__", None))
        elif callable(value):
            _add(getattr(value, "__module__", None))

    return found


# Private aliases re-exported by :mod:`yggdrasil.spark.frame` for
# backward compatibility with the in-line implementation.
_function_top_modules = function_top_modules
