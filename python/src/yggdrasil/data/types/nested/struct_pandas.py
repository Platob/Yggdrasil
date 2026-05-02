"""Pandas cast helpers for :class:`StructType` targets.

Pandas struct values are Python dicts in object-dtype columns —
there's no first-class struct type, so the helpers do row-by-row
conversion via :func:`pd.Series.map`.

* :func:`cast_pandas_struct_series` — dict → struct: per-target-child
  extraction, recurse into the child cast, then rebuild row dicts.
* :func:`cast_pandas_list_series` — list/tuple → struct by positional
  index; out-of-bounds is None.
* :func:`cast_pandas_tabular` — DataFrame column rebuild against the
  merged schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.support import get_pandas

if TYPE_CHECKING:
    import pandas as pd
    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.data_field import Field
    from .array import ArrayType
    from .struct import StructType


__all__ = [
    "cast_pandas_struct_series",
    "cast_pandas_list_series",
    "cast_pandas_tabular",
]


def cast_pandas_struct_series(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    pd = get_pandas()

    if not options.need_cast(series):
        return series

    if options.target_field is None:
        return series
    elif options.source_field.dtype.type_id != DataTypeId.STRUCT:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: "Field" = options.source_field
    source_type: "StructType" = source_field.dtype
    target_type: "StructType" = options.target_field.dtype

    def normalize_row(value: Any) -> dict[str, Any] | None:
        if value is None or (pd.isna(value) if not isinstance(value, dict) else False):
            return None
        if isinstance(value, dict):
            return value
        if hasattr(value, "asDict"):
            return value.asDict(recursive=True)
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        raise TypeError(f"Unsupported struct-like pandas value: {type(value)!r}")

    normalized = series.map(normalize_row)
    num_rows = len(series)
    out_cols: dict[str, pd.Series] = {}

    for i, target_child in enumerate(target_type.children_fields):
        source_child = source_type.field_by(
            name=target_child.name,
            index=i,
            raise_error=False,
        )

        if source_child is None:
            out_cols[target_child.name] = target_child.default_pandas_series(size=num_rows)
            continue

        extracted = normalized.map(
            lambda row, key=source_child.name: None if row is None else row.get(key)
        )

        out_cols[target_child.name] = target_child.cast_pandas_series(
            extracted,
            options=options.copy(
                source_field=source_child,
                target_field=target_child,
            ),
        )

    rows: list[dict[str, Any] | None] = []

    for row_idx in range(num_rows):
        src_row = normalized.iloc[row_idx]
        if src_row is None:
            rows.append(None)
            continue

        row: dict[str, Any] = {}
        for target_child in target_type.children_fields:
            row[target_child.name] = out_cols[target_child.name].iloc[row_idx]
        rows.append(row)

    return pd.Series(rows, index=series.index, name=options.target_field.name, dtype="object")


def cast_pandas_list_series(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    pd = get_pandas()

    if not options.need_cast(series):
        return series

    if options.target_field is None:
        return series
    elif options.source_field.dtype.type_id != DataTypeId.ARRAY:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: "Field" = options.source_field
    source_type: "ArrayType" = source_field.dtype
    target_type: "StructType" = options.target_field.dtype

    def normalize_row(value: Any) -> list[Any] | None:
        if value is None:
            return None
        if pd.isna(value) if not isinstance(value, (list, tuple)) else False:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
            out = value.tolist()
            return out if isinstance(out, list) else list(out)
        raise TypeError(f"Unsupported list-like pandas value: {type(value)!r}")

    normalized = series.map(normalize_row)
    num_rows = len(series)
    out_cols: dict[str, pd.Series] = {}

    for i, target_child in enumerate(target_type.children_fields):
        extracted = normalized.map(
            lambda row, idx=i: None if row is None or idx >= len(row) else row[idx]
        )

        out_cols[target_child.name] = target_child.cast_pandas_series(
            extracted,
            options=options.copy(
                source_field=source_type.item_field,
                target_field=target_child,
            ),
        )

    rows: list[dict[str, Any] | None] = []

    for row_idx in range(num_rows):
        src_row = normalized.iloc[row_idx]
        if src_row is None:
            rows.append(None)
            continue

        row: dict[str, Any] = {}
        for target_child in target_type.children_fields:
            row[target_child.name] = out_cols[target_child.name].iloc[row_idx]
        rows.append(row)

    return pd.Series(rows, index=series.index, name=options.target_field.name, dtype="object")


def cast_pandas_tabular(
    data: "pd.DataFrame",
    options: "CastOptions",
) -> "pd.DataFrame":
    pd = get_pandas()

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Unsupported tabular type: {type(data)!r}")

    if not options.need_cast(data, check_names=True):
        return data

    source_schema = options.source_schema
    target_schema = options.merged_schema

    out: dict[str, pd.Series] = {}
    num_rows = len(data)

    for i, target_field in enumerate(target_schema.children_fields):
        source_field = source_schema.field_by(
            name=target_field.name,
            index=i,
            raise_error=False,
        )

        if source_field is None:
            casted = target_field.default_pandas_series(size=num_rows)
        else:
            casted = target_field.cast_pandas_series(
                data[source_field.name],
                options=options.copy(
                    source_field=source_field,
                    target_field=target_field,
                ),
            )

        out[target_field.name] = casted.rename(target_field.name)

    return pd.DataFrame(out, index=data.index)
