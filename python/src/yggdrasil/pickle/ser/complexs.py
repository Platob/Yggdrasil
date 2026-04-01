"""
Backward-compatible re-export façade for the complex serialization layer.

The implementation has been split into three focused modules:
    libs.py        — shared utilities and base serializer classes
    callables.py   — FunctionSerialized / MethodSerialized + helpers
    dataclasses.py — DataclassSerialized + helpers

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
from yggdrasil.pickle.ser.libs import (
    BaseExceptionSerialized,
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

# ---------------------------------------------------------------------------
# Tags registration
# ---------------------------------------------------------------------------

from yggdrasil.pickle.ser.tags import Tags  # noqa: E402 — must come after class definitions

for _cls in ComplexSerialized.__subclasses__():
    Tags.register_class(_cls)

for _pytype, _cls in (
    (ModuleType, ModuleSerialized),
    (FunctionType, FunctionSerialized),
    (MethodType, MethodSerialized),
    (BaseException, BaseExceptionSerialized),
):
    Tags.register_class(_cls, pytype=_pytype)

del _cls, _pytype

