"""Shared coercion and parsing helpers for primitive ``_convert_pyobj``.

Everything in this module is private to :mod:`yggdrasil.data.types.primitive`.
No imports of concrete primitive types here — helpers are pure and take no
``DataType`` dependency so they can be reused from every leaf module without
creating import cycles.
"""
from __future__ import annotations

from typing import Any

import pyarrow as pa

__all__ = [
    "_bytes_to_str",
    "_coerce_str",
    "_BOOL_TRUE",
    "_BOOL_FALSE",
    "_INT_ARROW_SIGNED",
    "_INT_ARROW_UNSIGNED",
    "_INT_DDL_SIGNED",
    "_INT_DDL_UNSIGNED",
]


# ---------------------------------------------------------------------------
# str / bytes coercion
# ---------------------------------------------------------------------------

def _bytes_to_str(value: bytes | bytearray | memoryview) -> str | None:
    """Decode bytes-like to UTF-8 str, returning None on failure."""
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError:
        return None


def _coerce_str(value: Any) -> str | None:
    """Return value as str when it is a str/bytes/bytearray/memoryview."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _bytes_to_str(value)
    return None


# ---------------------------------------------------------------------------
# Boolean parsing tokens
# ---------------------------------------------------------------------------

_BOOL_TRUE = frozenset({"true", "t", "1", "yes", "y", "on"})
_BOOL_FALSE = frozenset({"false", "f", "0", "no", "n", "off", ""})


# ---------------------------------------------------------------------------
# Integer Arrow / DDL lookup tables
# ---------------------------------------------------------------------------

_INT_ARROW_SIGNED = {1: pa.int8, 2: pa.int16, 4: pa.int32, 8: pa.int64}
_INT_ARROW_UNSIGNED = {1: pa.uint8, 2: pa.uint16, 4: pa.uint32, 8: pa.uint64}
_INT_DDL_SIGNED = {1: "BYTE", 2: "SHORT", 4: "INT", 8: "BIGINT"}
# Spark has no native unsigned integers; we widen to preserve range.
_INT_DDL_UNSIGNED = {1: "SHORT", 2: "INT", 4: "BIGINT", 8: "DECIMAL(20, 0)"}
