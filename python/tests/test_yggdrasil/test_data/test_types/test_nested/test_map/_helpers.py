"""Shared helpers for MapType tests.

Different engines surface map-like values differently (pyarrow returns
``list[tuple]``, polars returns ``list[dict]``, etc.); these helpers
collapse all of them into a plain ``dict`` keyed by the parsed scalar
so tests can do straightforward equality assertions.
"""
from __future__ import annotations

import math
from typing import Any


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def normalize_scalar(value: Any) -> Any:
    if _is_nan(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def normalize_map_like(value: Any) -> Any:
    """Normalize pyarrow/polars/pandas representations of a map row."""
    if value is None:
        return None

    if isinstance(value, dict):
        return {normalize_scalar(k): normalize_scalar(v) for k, v in value.items()}

    if isinstance(value, list):
        if all(isinstance(item, tuple) and len(item) == 2 for item in value):
            return {normalize_scalar(k): normalize_scalar(v) for k, v in value}

        if all(
            isinstance(item, dict) and "key" in item and "value" in item
            for item in value
        ):
            return {
                normalize_scalar(item["key"]): normalize_scalar(item["value"])
                for item in value
            }

    return value
