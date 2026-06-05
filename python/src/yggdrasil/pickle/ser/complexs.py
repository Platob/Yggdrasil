"""
Backward-compatible re-export façade for the complex serialization layer.

The implementation has been split into four focused modules:
    libs.py        — shared utilities and base serializer classes
    callables.py   — FunctionSerialized / MethodSerialized + helpers
    dataclasses.py — DataclassSerialized + helpers
    exceptions.py  — BaseExceptionSerialized + traceback helpers

Everything that was previously in this file is re-exported here so that all
existing import paths (``from yggdrasil.pickle.ser.complexs import …``) keep
working without modification.

``import inspect`` is kept at module level so that test code that does::

    import yggdrasil.pickle.ser.complexs as complexs_module
    monkeypatch.setattr(complexs_module.inspect, "getclosurevars", …)

continues to work — patching ``inspect.getclosurevars`` through this module
attribute affects the shared ``inspect`` module object used by callables.py.
"""

from __future__ import annotations

import inspect  # noqa: F401 — kept for test monkeypatching; see module docstring
from types import FunctionType, MethodType, ModuleType

from yggdrasil.pickle.ser.callables import (
    FunctionSerialized,
    MethodSerialized,
)
from yggdrasil.pickle.ser.dataclasses import (
    DataclassSerialized,
)
from yggdrasil.pickle.ser.exceptions import (
    BaseExceptionSerialized,
)
from yggdrasil.pickle.ser.libs import (
    ClassSerialized,
    ComplexSerialized,
    ModuleSerialized,
)

# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

__all__ = [
    "ComplexSerialized",
    "ModuleSerialized",
    "ClassSerialized",
    "FunctionSerialized",
    "MethodSerialized",
    "BaseExceptionSerialized",
    "DataclassSerialized",
]


def _iter_subclasses(cls: type) -> set[type]:
    """
    Return all transitive subclasses of ``cls``.

    Using a visited set avoids duplicate traversal and guards against any
    accidental weirdness from repeated imports / dynamic class creation.
    """
    seen: set[type] = set()
    stack = [cls]

    while stack:
        current = stack.pop()
        for subcls in current.__subclasses__():
            if subcls not in seen:
                seen.add(subcls)
                stack.append(subcls)

    return seen


def _safe_register_class(serializer_cls: type, *, pytype: type | None = None) -> None:
    """
    Register a serializer class with Tags.

    This wrapper keeps registration logic centralized and makes it easier to
    harden behavior if Tags becomes stricter later.
    """
    if pytype is None:
        Tags.register_class(serializer_cls)
    else:
        Tags.register_class(serializer_cls, pytype=pytype)


# ---------------------------------------------------------------------------
# Tags registration
# ---------------------------------------------------------------------------

from yggdrasil.pickle.ser.tags import Tags  # noqa: E402 — must come after class definitions

for _cls in _iter_subclasses(ComplexSerialized):
    _safe_register_class(_cls)

for _pytype, _cls in (
    (ModuleType, ModuleSerialized),
    (FunctionType, FunctionSerialized),
    (MethodType, MethodSerialized),
    (BaseException, BaseExceptionSerialized),
):
    _safe_register_class(_cls, pytype=_pytype)

ModuleSerialized = Tags.get_class(Tags.MODULE) or ModuleSerialized
ClassSerialized = Tags.get_class(Tags.CLASS) or ClassSerialized
FunctionSerialized = Tags.get_class(Tags.FUNCTION) or FunctionSerialized
MethodSerialized = Tags.get_class(Tags.METHOD) or MethodSerialized
BaseExceptionSerialized = Tags.get_class(Tags.BASE_EXCEPTION) or BaseExceptionSerialized
DataclassSerialized = Tags.get_class(Tags.DATACLASS) or DataclassSerialized

del _cls, _pytype