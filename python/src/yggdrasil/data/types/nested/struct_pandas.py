"""Pandas cast helpers for :class:`StructType` targets.

Pandas has no first-class struct type, so struct/list values flow
through Arrow (or Polars) under the hood and surface as Python-object
cells in the resulting Series.  The public helpers in this module pick
the fastest path that can complete the cast end-to-end:

1. **PyArrow round-trip.**  Treat the pandas Series / DataFrame as
   Arrow input via ``pa.array(..., from_pandas=True)`` / ``pa.Table.
   from_pandas``, dispatch to the Arrow cast helpers, and surface back
   through ``Array.to_pandas()`` / ``Table.to_pandas()``.  No per-row
   Python loop, no ``to_pylist`` materialisation hop.
2. **Polars round-trip.**  When the Arrow path rejects the source
   shape (mixed-schema dicts, list-of-mixed-dtype, …), fall back to
   ``pl.from_pandas`` → :func:`cast_polars_tabular` / expression cast
   → ``to_pandas``.  Polars accepts a slightly different set of
   object-dtype inputs than Arrow.
3. **Column-by-column.**  Last-resort path that casts each column /
   child through its own engine (Arrow or Polars) and reassembles the
   pandas frame / object Series.  This is the only path that touches
   row-shaped Python values, and it only runs when both vectorised
   paths above fail.

The arrow→polars→columnwise chain mirrors the wider repo rule "no
Python ``for`` over data" — anything reachable from a real caller
should land on (1) or (2).  (3) stays as the documented fallback for
shapes pyarrow and polars both reject.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.lazy_imports import pandas_module, polars_module
from yggdrasil.exceptions import CastError

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


# ---------------------------------------------------------------------------
# Series-shaped fallback chain — pyarrow → polars → column-by-column
# ---------------------------------------------------------------------------


def _series_via_arrow(series: "pd.Series", options: "CastOptions") -> "pd.Series":
    """Round-trip *series* through Arrow using the target field's caster.

    Raises whatever the Arrow path raises — callers wrap this in a
    try/except as the first strategy in the fallback chain.  Output is
    rebuilt via ``Array.to_pandas()`` so the Arrow → pandas hop stays
    inside the C bridge (struct cells surface as dicts, list cells as
    numpy arrays — the standard pyarrow mapping).
    """
    source_arrow_type = options.source.dtype.to_arrow()
    source_array = pa.array(series, type=source_arrow_type, from_pandas=True)

    casted = options.target.dtype.cast_arrow_array(source_array, options=options)
    if isinstance(casted, pa.ChunkedArray):
        casted = casted.combine_chunks()

    result = casted.to_pandas()
    result.index = series.index
    result.name = options.target.name
    return result


def _series_via_polars(series: "pd.Series", options: "CastOptions") -> "pd.Series":
    """Round-trip *series* through Polars using the target field's caster.

    Raises whatever Polars raises — callers wrap this in a try/except
    as the second strategy in the fallback chain.
    """
    pl = polars_module()
    pl_series = pl.from_pandas(series)

    casted_pl = options.target.dtype.cast_polars_series(pl_series, options=options)

    if hasattr(casted_pl, "to_pandas"):
        result = casted_pl.to_pandas()
    else:  # pl.Expr — degenerate; resolve via DataFrame and pull the series back.
        df = pl.DataFrame({series.name: pl_series}).select(casted_pl.alias(series.name))
        result = df.to_pandas()[series.name]

    result.index = series.index
    result.name = options.target.name
    return result


def _run_pandas_series_fallback_chain(
    series: "pd.Series",
    options: "CastOptions",
    *,
    columnwise: "Any",
) -> "pd.Series":
    """Run the pyarrow → polars → column-by-column cast chain for a Series.

    Each strategy is tried in order; failures are collected so that if
    every path rejects, the surfaced :class:`CastError` carries the
    full provenance (source field, target field, and the original
    pyarrow / polars exception that triggered the fallback).  Any path
    that succeeds returns its cast Series directly.
    """
    failures: list[BaseException] = []

    try:
        return _series_via_arrow(series, options)
    except CastError:
        raise
    except Exception as exc:
        failures.append(exc)

    try:
        return _series_via_polars(series, options)
    except CastError:
        raise
    except Exception as exc:
        failures.append(exc)

    try:
        return columnwise(series, options)
    except CastError:
        raise
    except Exception as exc:
        reason = (
            "; ".join(
                f"{path}={type(err).__name__}: {err}"
                for path, err in zip(("pyarrow", "polars", "columnwise"), failures + [exc])
            )
        )
        raise CastError(
            f"all pandas cast strategies failed ({reason})",
            source=options.source,
            target=options.target,
            original=exc,
        ) from exc


# ---------------------------------------------------------------------------
# Struct → struct (series)
# ---------------------------------------------------------------------------


def cast_pandas_struct_series(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    if not options.need_cast(series):
        return series

    if options.target is None:
        return series
    elif options.source.dtype.type_id != DataTypeId.STRUCT:
        raise CastError(
            f"source is {options.source.dtype} — expected struct",
            source=options.source,
            target=options.target,
        )

    return _run_pandas_series_fallback_chain(
        series,
        options,
        columnwise=_struct_series_columnwise,
    )


def _struct_series_columnwise(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    """Per-target-child cast that survives heterogeneous dict shapes.

    Each child column lands as a pandas Series, cast through its own
    ``cast_pandas_series`` (Arrow/Polars under the hood for primitive
    children).  Row reassembly goes through
    :func:`_reassemble_object_series` — a one-shot
    ``pa.StructArray.from_arrays`` plus ``Array.to_pandas()`` — so the
    dict-cell materialisation stays inside the Arrow → pandas C bridge.
    """
    pd = pandas_module()
    source_field: "Field" = options.source
    source_type: "StructType" = source_field.dtype
    target_type: "StructType" = options.target.dtype

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

    for i, target_child in enumerate(target_type.children):
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
                source=source_child,
                target=target_child,
            ),
        )

    return _reassemble_object_series(
        out_cols,
        target_type.children,
        normalized,
        series_index=series.index,
        series_name=options.target.name,
    )


# ---------------------------------------------------------------------------
# Array → struct (series)
# ---------------------------------------------------------------------------


def cast_pandas_list_series(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    if not options.need_cast(series):
        return series

    if options.target is None:
        return series
    elif options.source.dtype.type_id != DataTypeId.ARRAY:
        raise CastError(
            f"source is {options.source.dtype} — expected list",
            source=options.source,
            target=options.target,
        )

    return _run_pandas_series_fallback_chain(
        series,
        options,
        columnwise=_list_series_columnwise,
    )


def _list_series_columnwise(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    """Per-target-child cast that survives heterogeneous list-of-mixed-row shapes.

    Each positional child lands as a pandas Series, cast through its
    own ``cast_pandas_series`` (Arrow / Polars under the hood). Row
    reassembly goes through :func:`_reassemble_object_series` — one
    ``pa.StructArray.from_arrays`` + ``Array.to_pandas()`` pass, no
    Python ``for`` over rows.
    """
    pd = pandas_module()
    source_field: "Field" = options.source
    source_type: "ArrayType" = source_field.dtype
    target_type: "StructType" = options.target.dtype

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
    out_cols: dict[str, pd.Series] = {}

    for i, target_child in enumerate(target_type.children):
        extracted = normalized.map(
            lambda row, idx=i: None if row is None or idx >= len(row) else row[idx]
        )

        out_cols[target_child.name] = target_child.cast_pandas_series(
            extracted,
            options=options.copy(
                source=source_type.item_field,
                target=target_child,
            ),
        )

    return _reassemble_object_series(
        out_cols,
        target_type.children,
        normalized,
        series_index=series.index,
        series_name=options.target.name,
    )


def _reassemble_object_series(
    out_cols: "dict[str, pd.Series]",
    children: "list[Field]",
    null_marker: "pd.Series",
    series_index: "pd.Index",
    series_name: str,
) -> "pd.Series":
    """Pack per-child casted Series into a pandas object Series of dicts.

    Wraps the per-child Series as one ``pa.StructArray`` (each column
    → Arrow array, no per-row Python work) and lets Arrow's
    ``Array.to_pandas()`` produce the Python-dict cells through the
    Arrow → pandas C bridge.  Null parent rows mask to ``None`` via
    ``StructArray.from_arrays(mask=...)`` — no post-processing loop.
    """
    pd = pandas_module()
    num_rows = len(series_index)
    if num_rows == 0:
        return pd.Series([], index=series_index, name=series_name, dtype="object")

    target_names = [c.name for c in children]
    child_arrays: list[pa.Array] = []
    for child in children:
        col = out_cols[child.name]
        try:
            arr = pa.array(col, from_pandas=True)
        except Exception:
            # Object cells that don't round-trip through Arrow (mixed
            # nested Python objects, etc.) — fall back to Arrow's
            # type-free inference path which is what would happen
            # without ``from_pandas=True``.
            arr = pa.array(col)
        child_arrays.append(arr)

    null_mask_arr = pa.array(null_marker.isna(), type=pa.bool_(), from_pandas=True)

    struct_arr = pa.StructArray.from_arrays(
        child_arrays,
        names=target_names,
        mask=null_mask_arr,
    )
    result = struct_arr.to_pandas()
    result.index = series_index
    result.name = series_name
    return result


# ---------------------------------------------------------------------------
# DataFrame-shaped fallback chain
# ---------------------------------------------------------------------------


def cast_pandas_tabular(
    data: "pd.DataFrame",
    options: "CastOptions",
) -> "pd.DataFrame":
    pd = pandas_module()

    if not isinstance(data, pd.DataFrame):
        raise CastError(
            f"unsupported tabular type {type(data).__name__}",
            source=options.source,
            target=options.target,
        )

    if not options.need_cast(data, check_names=True):
        return data

    failures: list[BaseException] = []

    # Strategy 1 — pyarrow Table round-trip.  Single ``Table.from_pandas``
    # → ``cast_arrow_tabular`` → ``to_pandas`` keeps every column inside
    # Arrow for the cast pass.
    try:
        return _tabular_via_arrow(data, options)
    except CastError:
        raise
    except Exception as exc:
        failures.append(exc)

    # Strategy 2 — polars DataFrame round-trip.  Same shape via
    # ``pl.from_pandas`` → :func:`cast_polars_tabular` → ``to_pandas``.
    # Useful when pyarrow rejects an object-dtype column shape Polars
    # happily ingests.
    try:
        return _tabular_via_polars(data, options)
    except CastError:
        raise
    except Exception as exc:
        failures.append(exc)

    # Strategy 3 — column-by-column.  Each column flows through its own
    # ``cast_pandas_series`` (which itself runs the
    # pyarrow→polars→columnwise chain).  ``cast_pandas_series`` already
    # wraps its own failures in ``CastError`` carrying the per-column
    # field provenance, so any failure here propagates with the correct
    # leaf identified — no extra wrap needed.
    try:
        return _tabular_columnwise(data, options)
    except CastError:
        raise
    except Exception as exc:
        reason = (
            "; ".join(
                f"{path}={type(err).__name__}: {err}"
                for path, err in zip(("pyarrow", "polars", "columnwise"), failures + [exc])
            )
        )
        raise CastError(
            f"all pandas tabular cast strategies failed ({reason})",
            source=options.source,
            target=options.target,
            original=exc,
        ) from exc


def _tabular_via_arrow(
    data: "pd.DataFrame",
    options: "CastOptions",
) -> "pd.DataFrame":
    preserve_index = bool(data.index.name)
    arrow_table = pa.Table.from_pandas(data, preserve_index=preserve_index)
    target_dtype = options.target.dtype
    casted_table = target_dtype.cast_arrow_tabular(arrow_table, options=options)
    result = casted_table.to_pandas()
    if preserve_index and data.index.name in result.columns:
        result = result.set_index(data.index.name)
    else:
        result.index = data.index
    return result


def _tabular_via_polars(
    data: "pd.DataFrame",
    options: "CastOptions",
) -> "pd.DataFrame":
    pl = polars_module()
    pl_df = pl.from_pandas(data)
    target_dtype = options.target.dtype
    casted_pl = target_dtype.cast_polars_tabular(pl_df, options=options)
    if hasattr(casted_pl, "collect"):
        casted_pl = casted_pl.collect()
    result = casted_pl.to_pandas()
    result.index = data.index
    return result


def _tabular_columnwise(
    data: "pd.DataFrame",
    options: "CastOptions",
) -> "pd.DataFrame":
    pd = pandas_module()
    source_schema = options.source
    target_schema = options.target.to_struct()

    out: dict[str, pd.Series] = {}
    num_rows = len(data)

    for i, target_field in enumerate(target_schema.children):
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
                    source=source_field,
                    target=target_field,
                ),
            )

        out[target_field.name] = casted.rename(target_field.name)

    return pd.DataFrame(out, index=data.index)
