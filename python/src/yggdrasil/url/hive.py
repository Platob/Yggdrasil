from __future__ import annotations

from typing import Any
from urllib.parse import quote, unquote

import pyarrow as pa

__all__ = ["HIVE_DEFAULT_PARTITION", "hive_encode", "hive_decode", "hive_split", "hive_cast_value"]

# ---------------------------------------------------------------------------
# Hive partition layout — ``<col>=<val>/`` directory encoding
# ---------------------------------------------------------------------------
#
# Used by :class:`yggdrasil.io.nested.folder_path.FolderPath` to lay out
# tabular folders the same way Hive / Spark / Delta do, so the on-disk
# tree round-trips through any partition-aware reader. The helpers
# live here next to the URL-component encoders because the convention
# is fundamentally URL-shaped: each path segment is the URL-quoted
# value joined onto the column name with an ``=``.

#: Sentinel a Hive writer emits for ``None`` partition values — the
#: same string Hive, Spark, and Delta agree on, so an externally-
#: produced tree drops in here without translation.
HIVE_DEFAULT_PARTITION: str = "__HIVE_DEFAULT_PARTITION__"


def hive_encode(value: Any) -> str:
    """Encode *value* as a filesystem-safe Hive partition value.

    ``None`` → :data:`HIVE_DEFAULT_PARTITION` matching the Hive /
    Spark / Delta convention. Everything else is ``str(value)`` URL-
    quoted with the path-separator + ``=`` characters reserved so
    the encoded value can be split back unambiguously on a single
    ``=`` and never collides with a directory boundary.
    """
    if value is None:
        return HIVE_DEFAULT_PARTITION
    return quote(str(value), safe="")


def hive_decode(raw: str) -> Any:
    """Inverse of :func:`hive_encode` — returns the URL-decoded string.

    The caller is responsible for casting the result to the partition
    column's declared dtype (the URL layer doesn't know the schema
    at parse time; see :func:`hive_cast_value` for the dtype-aware
    half).
    """
    if raw == HIVE_DEFAULT_PARTITION:
        return None
    return unquote(raw)


def hive_split(name: str) -> "tuple[str, Any] | None":
    """Parse a Hive-encoded segment into ``(column, value)``.

    Returns ``None`` when *name* doesn't match the ``<col>=<val>``
    convention so the caller can treat the entry as a plain (non-
    Hive) directory.
    """
    if "=" not in name:
        return None
    col, _, raw = name.partition("=")
    if not col:
        return None
    return col, hive_decode(raw)


def hive_cast_value(value: Any, dtype: "pa.DataType | None") -> Any:
    """Cast a :func:`hive_decode`-d value to *dtype*, leaving raw on failure.

    Used when the partition column's declared dtype is in scope —
    int64 ``partition_key`` lands as :class:`int`, a timestamp
    partition as :class:`datetime`. When *dtype* is ``None`` or the
    cast raises (un-castable value), the decoded string passes
    through unchanged — every caller's downstream prune is
    conservative on undecidable shapes so a no-op cast just forces
    the row-level filter to run.

    Fast path: the common partition dtypes (integers, floats, bool,
    string, the typed ints we tag on every cached response's
    ``partition_key``) cast natively with the built-in constructor.
    Allocating a one-element ``pa.array`` and dispatching the cast
    kernel on every prune check shows up at the top of the cache
    hot path — the native path is ~50× faster. Anything outside
    the fast set (timestamps, decimals, lists, …) falls back to
    the pyarrow round-trip which still handles arbitrary types.
    """
    if value is None or dtype is None:
        return value
    fast = _HIVE_FAST_CAST.get(dtype.id)
    if fast is not None:
        try:
            return fast(value)
        except (TypeError, ValueError):
            return value
    try:
        arr = pa.array([value]).cast(dtype, safe=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError, NotImplementedError):
        return value
    return arr[0].as_py()


def _cast_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lo = value.lower()
        if lo in ("true", "1", "t", "yes"):
            return True
        if lo in ("false", "0", "f", "no"):
            return False
        raise ValueError(value)
    return bool(value)


# Map :class:`pa.DataType.id` (a small int) to the native Python cast.
# Built once at import time so the partition prune hot path is a
# dict lookup + a builtin call.
_HIVE_FAST_CAST: "dict[int, Any]" = {
    pa.int8().id: int,
    pa.int16().id: int,
    pa.int32().id: int,
    pa.int64().id: int,
    pa.uint8().id: int,
    pa.uint16().id: int,
    pa.uint32().id: int,
    pa.uint64().id: int,
    pa.float16().id: float,
    pa.float32().id: float,
    pa.float64().id: float,
    pa.bool_().id: _cast_bool,
    pa.string().id: str,
    pa.large_string().id: str,
}
