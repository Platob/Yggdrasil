"""Helpers for StructType tests.

``normalize_nested`` flattens the small numeric/NaN differences that
leak out of pyspark/pandas/polars row dicts so equality assertions
can be direct.
"""
from __future__ import annotations

import math
from typing import Any


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _normalize_scalar(value: Any) -> Any:
    if _is_nan(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def normalize_nested(value: Any) -> Any:
    # PySpark hands struct columns back as ``Row`` when the toPandas
    # path bypasses Arrow optimization (Java 17+ on PySpark 3.x);
    # collapse to a plain dict so assertions can stay direct.
    as_dict = getattr(value, "asDict", None)
    if callable(as_dict):
        try:
            value = as_dict(recursive=True)
        except TypeError:
            value = as_dict()
    if isinstance(value, dict):
        return {k: normalize_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_nested(v) for v in value]
    return _normalize_scalar(value)
