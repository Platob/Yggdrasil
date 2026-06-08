from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Tuple

__all__ = [
    "dicts_equal",
    "dict_diff"
]


# ``...`` (Ellipsis) is the project-wide "missing key" sentinel — see
# the convention note in ``AGENTS.md`` / ``CLAUDE.md``. Distinguishes
# "key not in dict" from "key present with value None".


def _normalize(obj: Any) -> Any:
    """
    Normalize nested structures so equality is stable:
    - dict: sort keys + normalize values
    - list/tuple: normalize items (keeps order)
    - set: sort normalized items (orderless)
    - float: keep as float (handled separately for tolerance)
    """
    if isinstance(obj, dict):
        return {k: _normalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_normalize(x) for x in obj)
    return obj

def _equal(a: Any, b: Any, float_tol: float = 0.0) -> bool:
    # Float tolerance (optional)
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        try:
            return math.isclose(float(a), float(b), rel_tol=float_tol, abs_tol=float_tol)
        except Exception:
            pass

    # Deep normalize compare for dict/list/set
    return _normalize(a) == _normalize(b)

def dicts_equal(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    keys: Iterable[str] | None = None,
    treat_missing_as_none: bool = True,
    float_tol: float = 0.0,
) -> bool:
    """
    Equality check for two dicts with options:
    - keys: only compare these keys
    - treat_missing_as_none: missing key == None if other side is None
    - float_tol: tolerance for float comparisons
    """
    if keys is None:
        keys = set(a.keys()) | set(b.keys())

    for k in keys:
        av = a.get(k, ...)
        bv = b.get(k, ...)

        if treat_missing_as_none:
            if av is ... and bv is None:
                continue
            if bv is ... and av is None:
                continue
            if av is ... and bv is ...:
                continue

        if not _equal(av, bv, float_tol=float_tol):
            return False

    return True

def dict_diff(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    keys: Iterable[str] | None = None,
    treat_missing_as_none: bool = True,
    float_tol: float = 0.0,
) -> Dict[str, Tuple[Any, Any]]:
    """
    Returns {key: (a_val, b_val)} for all keys that differ.
    """
    if keys is None:
        keys = set(a.keys()) | set(b.keys())

    out: Dict[str, Tuple[Any, Any]] = {}
    for k in keys:
        av = a.get(k, ...)
        bv = b.get(k, ...)

        if treat_missing_as_none:
            if av is ... and bv is None:
                continue
            if bv is ... and av is None:
                continue
            if av is ... and bv is ...:
                continue

        if not _equal(av, bv, float_tol=float_tol):
            out[k] = (None if av is ... else av, None if bv is ... else bv)
    return out
