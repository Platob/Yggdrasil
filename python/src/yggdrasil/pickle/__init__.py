"""Yggdrasil's serde wire format — re-export the canonical surface.

The actual implementation lives under :mod:`yggdrasil.pickle.ser`; this
package re-exposes the four entry points (:func:`dump`, :func:`dumps`,
:func:`load`, :func:`loads`) plus :func:`serialize` and the error /
``Serialized`` / ``Tags`` types at the top level so call sites can write
the canonical ``from yggdrasil.pickle import dumps, loads`` without
reaching into ``.ser``.

Re-exports go through ``__getattr__`` (PEP 562) rather than a top-level
import because ``yggdrasil.pickle.ser.serde`` pulls in
:mod:`yggdrasil.io`, which transitively re-enters :mod:`yggdrasil.data`
during ``data_field.py``'s own load (``data_field.py`` does
``import yggdrasil.pickle.json`` before its classes are bound). An
eager re-export here would race that import and raise
``ImportError: cannot import name 'Field' from partially initialized
module``. Deferring to first attribute access breaks the cycle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.pickle import json as json  # noqa: F401 — re-export

if TYPE_CHECKING:  # pragma: no cover — only for static analysis
    from yggdrasil.pickle.ser import (
        HeaderDecodeError,
        InvalidCodecError,
        MetadataDecodeError,
        SerializationError,
        Serialized,
        Tags,
        dump,
        dumps,
        load,
        loads,
        serialize,
    )

__all__ = [
    "Tags",
    "Serialized",
    "SerializationError",
    "HeaderDecodeError",
    "MetadataDecodeError",
    "InvalidCodecError",
    "dump",
    "dumps",
    "load",
    "loads",
    "serialize",
    "json",
]

# Names re-exported from ``yggdrasil.pickle.ser`` on first attribute access.
_SER_REEXPORTS = frozenset({
    "dump", "dumps", "load", "loads", "serialize",
    "Serialized", "Tags",
    "SerializationError", "HeaderDecodeError",
    "MetadataDecodeError", "InvalidCodecError",
})


def __getattr__(name: str) -> Any:
    if name in _SER_REEXPORTS:
        from yggdrasil.pickle import ser
        value = getattr(ser, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module 'yggdrasil.pickle' has no attribute {name!r}")
