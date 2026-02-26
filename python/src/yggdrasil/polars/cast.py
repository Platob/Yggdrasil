"""Polars <-> Arrow casting helpers and converters."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import polars as pl
import pyarrow as pa
import pyarrow.types as pat

from yggdrasil.arrow.cast import (
    cast_arrow_tabular,
    is_arrow_type_binary_like,
    is_arrow_type_list_like,
    is_arrow_type_string_like,
    default_arrow_scalar,
)
from yggdrasil.data.cast import CastOptions, register_converter
from yggdrasil.io.path import LocalDataPath, SystemPath
from yggdrasil.pyutils.serde import ObjectSerde

__all__ = [
    "register_converter",
    "cast_polars_array",
    "cast_polars_array_to_temporal",
    "cast_polars_array_to_struct",
    "cast_polars_array_to_list",
    "cast_polars_dataframe",
    "cast_polars_lazyframe",
    "arrow_type_to_polars_type",
    "polars_type_to_arrow_type",
    "arrow_field_to_polars_field",
    "polars_field_to_arrow_field",
    "polars_dataframe_to_arrow_table",
    "arrow_table_to_polars_dataframe",
    "any_to_polars_dataframe",
    "any_polars_to_arrow_field",
]

# ---------------------------------------------------------------------------
# Primitive Arrow -> Polars dtype mapping
# ---------------------------------------------------------------------------

ARROW_TO_POLARS: Dict[pa.DataType, pl.DataType] = {
    pa.null(): pl.Null(),
    pa.bool_(): pl.Boolean(),
    pa.int8(): pl.Int8(),
    pa.int16(): pl.Int16(),
    pa.int32(): pl.Int32(),
    pa.int64(): pl.Int64(),
    pa.uint8(): pl.UInt8(),
    pa.uint16(): pl.UInt16(),
    pa.uint32(): pl.UInt32(),
    pa.uint64(): pl.UInt64(),
    pa.float16(): pl.Float32(),  # best-effort upcast
    pa.float32(): pl.Float32(),
    pa.float64(): pl.Float64(),
    pa.string_view(): pl.Utf8(),
    pa.large_string(): pl.Utf8(),
    pa.string(): pl.Utf8(),
    pa.binary_view(): pl.Binary(),
    pa.large_binary(): pl.Binary(),
    pa.binary(): pl.Binary(),
    pa.date32(): pl.Date(),
}

POLARS_BASE_TO_ARROW: Dict[pl.DataType, pa.DataType] = {v: k for k, v in ARROW_TO_POLARS.items()}
POLARS_TYPE_CLASSES: tuple = tuple(v.__class__ for v in ARROW_TO_POLARS.values())

# Timezone alias groups — members are treated as equivalent when normalising.
TIMEZONE_ALIASES: Dict[str, frozenset] = {
    "UTC": frozenset(("UTC", "Etc/UTC", "+00:00")),
    "CET": frozenset(("CET", "Europe/Paris", "Europe/Zurich")),
}

# Numeric source classes (used repeatedly in temporal dispatch)
_NUMERIC_CLASSES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)

# ---------------------------------------------------------------------------
# Expr/Series evaluation helpers
# ---------------------------------------------------------------------------


def _safe_col_name(*, series_name: str, parent_name: str | None, fallback: str = "__col__") -> str:
    """
    Build a stable column name for Series->Frame evaluation and nested casting.

    Avoids hardcoded '_col_' and reduces collision risk in nested casts.
    """
    series_name = series_name or ""
    parent_name = parent_name or ""

    if series_name and parent_name:
        return f"{parent_name}.{series_name}"
    if series_name:
        return series_name
    if parent_name:
        return parent_name
    return fallback


def _eval_expr_on_series(
    series: pl.Series,
    expr: pl.Expr,
    *,
    col_name: str,
    out_name: str | None = None,
) -> pl.Series:
    """Evaluate an expression against a concrete Series while preserving shape."""
    out = series.to_frame(name=col_name).select(expr).to_series()
    return out.rename(out_name or series.name)


# ---------------------------------------------------------------------------
# Temporal cast helpers
# ---------------------------------------------------------------------------


def _safe_strptime(arr: pl.Expr, dtype: pl.DataType, fmt: str) -> pl.Expr:
    """Attempt a single strptime parse, returning null on failure (strict=False)."""
    return arr.str.strptime(dtype, fmt, strict=False, ambiguous="earliest")


def _coalesce_strptime(arr: pl.Expr, dtype: pl.DataType, formats: list[str]) -> pl.Expr:
    """Try each format in order; first non-null result wins (per row)."""
    return pl.coalesce([_safe_strptime(arr, dtype, fmt) for fmt in formats])


# Formats that embed a UTC offset (%z) — strptime yields Datetime[tu, UTC].
# These must be handled separately: parse to tz-aware, strip tz, then coalesce
# with naive results. Mixing tz-aware and naive in one pl.coalesce() raises
# SchemaError("failed to determine supertype of datetime[μs, UTC] and datetime[μs]").
_DATETIME_FORMATS_WITH_TZ = [
    "%Y-%m-%dT%H:%M:%S%.f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%.f%z",  # space sep, fractional seconds, offset
    "%Y-%m-%d %H:%M:%S%z",  # space sep, no fractional seconds, offset
    "%Y-%m-%d %H:%M%z",  # space sep, no seconds, offset  ← matches "2026-02-17 03:00+01:00"
]

# All remaining formats — strptime yields naive Datetime[tu].
_DATETIME_FORMATS_NAIVE = [
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%Y%m%dT%H%M%S",
    "%Y%m%d",
    "%d %b %Y %H:%M:%S",
    "%d %b %Y",
    "%b %d, %Y",
]

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y%m%d",
    "%d %b %Y",
    "%b %d, %Y",
    "%d.%m.%Y",
]

_TIME_FORMATS = [
    "%H:%M:%S%.f",
    "%H:%M:%S",
    "%H:%M",
    "%I:%M:%S %p",
    "%I:%M %p",
]


def _normalize_utc_offset(arr: pl.Expr) -> pl.Expr:
    """Rewrite '+01:00' / '-05:30' style offsets to '+0100' / '-0530'.

    Polars strptime %z handles +HHMM but not +HH:MM (RFC 3339 style).
    Only rewrites if a colon-offset is actually present — leaves other
    strings untouched.
    """
    return arr.str.replace(
        r"([+-])(\d{2}):(\d{2})$",
        r"${1}${2}${3}",
    )


def _unsafe_string_to_datetime(arr: pl.Expr, dtype: pl.Datetime) -> pl.Expr:
    tu = dtype.time_unit or "us"
    bare = pl.Datetime(tu)
    utc = pl.Datetime(tu, "UTC")

    normalized = _normalize_utc_offset(arr)  # +HH:MM → +HHMM

    tz_parsed = [
        _safe_strptime(normalized, utc, fmt).dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        for fmt in _DATETIME_FORMATS_WITH_TZ
    ]
    naive_parsed = [_safe_strptime(arr, bare, fmt) for fmt in _DATETIME_FORMATS_NAIVE]

    return pl.coalesce(tz_parsed + naive_parsed)


def _unsafe_string_to_date(arr: pl.Expr) -> pl.Expr:
    return _coalesce_strptime(arr, pl.Date(), _DATE_FORMATS)


def _unsafe_string_to_time(arr: pl.Expr) -> pl.Expr:
    return _coalesce_strptime(arr, pl.Time(), _TIME_FORMATS)


def _unsafe_string_to_duration(arr: pl.Expr, dtype: pl.Duration) -> pl.Expr:
    """Interpret integer strings as epoch-offset integers → Duration."""
    tu = dtype.time_unit or "us"
    return arr.cast(pl.Int64, strict=False).cast(pl.Duration(tu), strict=False)


def _apply_tz(expr: pl.Expr, src_tz: str | None, tgt_tz: str | None) -> pl.Expr:
    """
    Transition a naive or tz-aware Datetime expr to *tgt_tz*.

    Decision table
    ──────────────
    src_tz   tgt_tz   action
    ────────────────────────────────────────────────────────────────────
    None     None     no-op                              (both naive)
    None     set      replace_time_zone(tgt)             (stamp as tgt)
    set      None     replace(src) → convert(UTC) → replace(None)
    set      set=src  replace_time_zone(src)             (same zone, no math)
    set      set≠src  replace(src) → convert(tgt)        (proper conversion)
    """
    if src_tz is None and tgt_tz is None:
        return expr

    if src_tz is None:
        return expr.dt.replace_time_zone(tgt_tz, ambiguous="earliest")

    if tgt_tz is None:
        return expr.dt.replace_time_zone(src_tz, ambiguous="earliest").dt.convert_time_zone("UTC").dt.replace_time_zone(
            None
        )

    if src_tz == tgt_tz:
        return expr.dt.replace_time_zone(src_tz, ambiguous="earliest")

    return expr.dt.replace_time_zone(src_tz, ambiguous="earliest").dt.convert_time_zone(tgt_tz)


# ---------------------------------------------------------------------------
# Temporal array cast
# ---------------------------------------------------------------------------


def cast_polars_array_to_temporal(
    array: Union[pl.Series, pl.Expr],
    source: pl.DataType,
    target: pl.datatypes.TemporalType,
    safe: bool,
    source_tz: str | None = None,
    target_tz: str | None = None,
    to_expr: bool = False,
    parent_name: str | None = None,
) -> Union[pl.Series, pl.Expr]:
    """Cast *array* to a temporal Polars dtype with full timezone handling.

    Returns Expr when:
      - input is Expr, or
      - input is Series and to_expr=True

    Returns Series when input is Series and to_expr=False.
    """
    is_expr = isinstance(array, pl.Expr)
    series_name = "" if is_expr else array.name  # type: ignore[union-attr]

    col_name = _safe_col_name(series_name=series_name, parent_name=parent_name)
    working: pl.Expr = array if is_expr else pl.col(col_name)  # type: ignore[assignment]

    src_cls = source.__class__
    tgt_cls = target.__class__

    # Timezone resolution: dtype-embedded wins over explicit override
    _src_tz: str | None = (source.time_zone if src_cls is pl.Datetime else None) or source_tz  # type: ignore[union-attr]
    _tgt_tz: str | None = (target.time_zone if tgt_cls is pl.Datetime else None) or target_tz  # type: ignore[union-attr]

    # ── Datetime ──────────────────────────────────────────────────────────────
    if tgt_cls is pl.Datetime:
        tu = target.time_unit or "us"  # type: ignore[union-attr]

        if src_cls is pl.Datetime:
            working = working.cast(pl.Datetime(tu), strict=safe)
            working = _apply_tz(working, _src_tz, _tgt_tz)

        elif src_cls in (pl.String, pl.Utf8):
            if safe:
                working = working.str.strptime(pl.Datetime(tu), strict=True, ambiguous="earliest")
            else:
                working = _unsafe_string_to_datetime(working, target)
            working = _apply_tz(working, _src_tz, _tgt_tz)

        elif src_cls is pl.Date:
            working = working.cast(pl.Datetime(tu), strict=safe)
            working = _apply_tz(working, None, _tgt_tz)

        elif src_cls is pl.Time:
            epoch = pl.lit(0).cast(pl.Datetime(tu))
            working = epoch + working.cast(pl.Duration(tu), strict=safe)
            working = _apply_tz(working, None, _tgt_tz)

        elif src_cls is pl.Duration:
            epoch = pl.lit(0).cast(pl.Datetime(tu))
            working = epoch + working.cast(pl.Duration(tu), strict=safe)
            working = _apply_tz(working, None, _tgt_tz)

        elif src_cls in _NUMERIC_CLASSES:
            working = working.cast(pl.Int64, strict=safe).cast(pl.Datetime(tu), strict=safe)
            working = _apply_tz(working, _src_tz, _tgt_tz)

        else:
            working = working.cast(target, strict=safe)

    # ── Date ──────────────────────────────────────────────────────────────────
    elif tgt_cls is pl.Date:
        if src_cls is pl.Date:
            pass  # no-op

        elif src_cls in (pl.String, pl.Utf8):
            if safe:
                working = working.str.strptime(pl.Date(), strict=True, ambiguous="earliest")
            else:
                working = _unsafe_string_to_date(working)

        elif src_cls is pl.Datetime:
            if _src_tz:
                working = working.dt.replace_time_zone(_src_tz, ambiguous="earliest")
            if _tgt_tz and _tgt_tz != _src_tz:
                working = working.dt.convert_time_zone(_tgt_tz)
            working = working.dt.date()

        elif src_cls in _NUMERIC_CLASSES:
            # Days-since-epoch integer → Date
            working = working.cast(pl.Int32, strict=safe).cast(pl.Date, strict=safe)

        else:
            working = working.cast(pl.Date, strict=safe)

    # ── Time ──────────────────────────────────────────────────────────────────
    elif tgt_cls is pl.Time:
        if src_cls is pl.Time:
            pass  # no-op

        elif src_cls in (pl.String, pl.Utf8):
            if safe:
                working = working.str.strptime(pl.Time(), strict=True, ambiguous="earliest")
            else:
                working = _unsafe_string_to_time(working)

        elif src_cls is pl.Datetime:
            if _src_tz:
                working = working.dt.replace_time_zone(_src_tz, ambiguous="earliest")
            if _tgt_tz and _tgt_tz != _src_tz:
                working = working.dt.convert_time_zone(_tgt_tz)
            working = working.dt.time()

        elif src_cls is pl.Duration:
            working = working.cast(pl.Int64, strict=safe).cast(pl.Time, strict=safe)

        elif src_cls in _NUMERIC_CLASSES:
            working = working.cast(pl.Int64, strict=safe).cast(pl.Time, strict=safe)

        else:
            working = working.cast(pl.Time, strict=safe)

    # ── Duration ──────────────────────────────────────────────────────────────
    elif tgt_cls is pl.Duration:
        tu = target.time_unit or "us"  # type: ignore[union-attr]

        if src_cls is pl.Duration:
            working = working.cast(target, strict=safe)  # type: ignore[arg-type]

        elif src_cls in (pl.String, pl.Utf8):
            if safe:
                raise NotImplementedError(
                    "Safe string→Duration parsing is not natively supported by Polars. "
                    "Provide integer strings (epoch offset in the target time_unit) "
                    "or use safe=False."
                )
            working = _unsafe_string_to_duration(working, target)  # type: ignore[arg-type]

        elif src_cls in _NUMERIC_CLASSES:
            working = working.cast(pl.Int64, strict=safe).cast(pl.Duration(tu), strict=safe)

        elif src_cls is pl.Datetime:
            if _src_tz:
                working = (
                    working.dt.replace_time_zone(_src_tz, ambiguous="earliest")
                    .dt.convert_time_zone("UTC")
                    .dt.replace_time_zone(None)
                )
            epoch = pl.lit(0).cast(pl.Datetime(tu))
            working = working.cast(pl.Datetime(tu), strict=safe) - epoch

        elif src_cls is pl.Date:
            epoch = pl.lit(0).cast(pl.Date)
            working = (working - epoch).cast(pl.Duration(tu), strict=safe)

        else:
            working = working.cast(target, strict=safe)  # type: ignore[arg-type]

    else:
        raise TypeError(f"Unsupported temporal target type: {target!r}")

    if series_name:
        working = working.alias(series_name)

    if is_expr or to_expr:
        return working

    return _eval_expr_on_series(array, working, col_name=col_name, out_name=series_name)


def cast_polars_array_to_bool(
    array: Union[pl.Series, pl.Expr],
    options: "CastOptions",
    to_expr: bool = False,
    parent_name: str | None = None,
) -> Union[pl.Series, pl.Expr]:
    """Cast a Series / Expr to pl.Boolean with optional Expr return mode."""
    is_expr = isinstance(array, pl.Expr)
    series_name = options.source_arrow_field.name if is_expr else array.name
    spf = options.source_polars_field
    src_cls = spf.dtype.__class__

    col_name = _safe_col_name(series_name=series_name, parent_name=parent_name)
    working: pl.Expr = array if is_expr else pl.col(col_name)

    # ── Already bool ──────────────────────────────────────────────────────────
    if src_cls is pl.Boolean:
        result = working

    # ── Numeric: non-zero → True ──────────────────────────────────────────────
    elif src_cls in _NUMERIC_CLASSES:
        result = working.cast(pl.Int64, strict=options.safe).ne(0)

    # ── String: keyword mapping ───────────────────────────────────────────────
    elif src_cls in (pl.String, pl.Utf8):
        lowered = working.str.to_lowercase()

        true_expr = lowered.is_in(["true", "1", "yes", "on"])
        false_expr = lowered.is_in(["false", "0", "no", "off"])

        if options.safe:
            unrecognised = (~true_expr) & (~false_expr) & lowered.is_not_null()
            result = (
                pl.when(unrecognised)
                .then(
                    pl.lit("Cannot cast string to bool: ")
                    .add(working.cast(pl.String))
                    .cast(pl.Boolean, strict=True)  # intentional failure path
                )
                .when(true_expr)
                .then(pl.lit(True))
                .when(false_expr)
                .then(pl.lit(False))
                .otherwise(pl.lit(None, dtype=pl.Boolean))
            )
        else:
            result = (
                pl.when(true_expr)
                .then(pl.lit(True))
                .when(false_expr)
                .then(pl.lit(False))
                .otherwise(pl.lit(None, dtype=pl.Boolean))
            )

    # ── Null source ───────────────────────────────────────────────────────────
    elif src_cls is pl.Null:
        result = working.cast(pl.Boolean, strict=False)

    # ── Fallback ──────────────────────────────────────────────────────────────
    else:
        result = working.cast(pl.Boolean, strict=options.safe)

    result = result.alias(series_name)

    if is_expr or to_expr:
        return result

    return _eval_expr_on_series(array, result, col_name=col_name, out_name=series_name)


# ---------------------------------------------------------------------------
# Struct array cast
# ---------------------------------------------------------------------------


def _resolve_source_field(name: str, src_fields: list[pl.Field], strict_match_names: bool) -> pl.Field | None:
    """Look up a field by name from *src_fields*, respecting *strict_match_names*.

    Priority: exact match → case-insensitive match (when not strict).
    """
    exact = {f.name: f for f in src_fields}
    if name in exact:
        return exact[name]
    if not strict_match_names:
        folded = {f.name.casefold(): f for f in src_fields}
        return folded.get(name.casefold())
    return None


def cast_polars_array_to_struct(
    array: Union[pl.Series, pl.Expr],
    options: "CastOptions",
    to_expr: bool = False,
    parent_name: str | None = None,
) -> Union[pl.Series, pl.Expr]:
    """Cast *array* to a Struct Polars dtype.

    Returns Expr when:
      - input is Expr, or
      - input is Series and to_expr=True

    Returns Series when input is Series and to_expr=False.
    """
    is_expr = isinstance(array, pl.Expr)
    spf, tpf = options.source_polars_field, options.target_polars_field
    series_name = array.name if not is_expr else options.source_arrow_field.name

    frame_name = _safe_col_name(series_name=series_name, parent_name=parent_name)

    # Non-struct source: JSON decode or direct cast
    if spf.dtype.__class__ is not pl.Struct:
        if spf.dtype.__class__ in (pl.String, pl.Utf8, pl.Object):
            # json_decode on Expr vs Series behaves differently; keep it simple:
            if is_expr:
                result = array.str.json_decode(dtype=tpf.dtype).cast(tpf.dtype, strict=options.safe).alias(series_name)
                return result if (is_expr or to_expr) else array.to_frame(name=frame_name).select(result).to_series()
            array = array.str.json_decode(dtype=tpf.dtype)  # type: ignore[union-attr]
        result = array.cast(tpf.dtype, strict=options.safe).alias(series_name)  # type: ignore[union-attr]
        if is_expr or to_expr:
            return result
        return result if isinstance(result, pl.Series) else _eval_expr_on_series(array, result, col_name=frame_name, out_name=series_name)  # type: ignore[arg-type]

    out: list[pl.Expr] = []

    for i, (tgt_arrow_child, tgt_pl_child) in enumerate(zip(options.target_arrow_field.type, tpf.dtype.fields)):
        src: Optional[pl.Field] = options.source_child_polars_field(
            index=i,
            name=tgt_pl_child.name,
            raise_error=not options.add_missing_columns,
        )

        if src is None:
            out.append(
                default_polars_array(
                    is_expr=True,  # always build Expr for pl.struct()
                    num_rows=0,
                    arrow_field=tgt_arrow_child,
                    dtype=tgt_pl_child.dtype,
                )
            )
        else:
            # Extract the child — always as Expr so pl.struct() can compose them.
            child_src: pl.Expr = (
                array.struct.field(src.name)
                if is_expr
                else pl.col(frame_name).struct.field(src.name)
            )
            out.append(
                cast_polars_array(
                    child_src,
                    options.copy(
                        source_field=polars_field_to_arrow_field(src),
                        target_field=tgt_arrow_child,
                    ),
                    to_expr=True,
                    parent_name=frame_name,
                )
            )

    struct_expr = pl.struct(out).alias(series_name)

    if is_expr or to_expr:
        return struct_expr

    # Evaluate against a DataFrame built from the source series so all
    # .struct.field(...) references resolve correctly.
    return array.to_frame(name=frame_name).select(struct_expr).to_series()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# List / Array cast
# ---------------------------------------------------------------------------


def cast_polars_array_to_list(
    array: Union[pl.Series, pl.Expr],
    options: "CastOptions",
    to_expr: bool = False,
    parent_name: str | None = None,
) -> Union[pl.Series, pl.Expr]:
    """Cast *array* to a List / Array Polars dtype.

    Returns Expr when:
      - input is Expr, or
      - input is Series and to_expr=True

    Returns Series when input is Series and to_expr=False.
    """
    is_expr = isinstance(array, pl.Expr)

    saf, taf = options.source_arrow_field, options.target_arrow_field
    spf, tpf = options.source_polars_field, options.target_polars_field
    series_name = saf.name if saf is not None else ("" if is_expr else array.name)

    col_name = _safe_col_name(series_name=series_name, parent_name=parent_name)
    base: pl.Expr = array if is_expr else pl.col(col_name)

    def _eval(result: pl.Expr) -> pl.Expr | pl.Series:
        if is_expr or to_expr:
            return result
        return _eval_expr_on_series(array, result, col_name=col_name, out_name=series_name)

    def _child_options(src_arrow_inner: pa.Field, tgt_arrow_inner: pa.Field) -> "CastOptions":
        return options.copy(source_field=src_arrow_inner, target_field=tgt_arrow_inner)

    # Resolve inner Arrow fields
    src_arrow_inner: pa.Field = options.source_child_arrow_field(index=0)
    tgt_arrow_inner: pa.Field = options.target_child_arrow_field(index=0)

    parent_path = _safe_col_name(series_name=series_name, parent_name=parent_name)

    # ── Variable-length List ──────────────────────────────────────────────────
    if (
        pa.types.is_list(saf.type)
        or pa.types.is_large_list(saf.type)
        or pa.types.is_list_view(saf.type)
        or pa.types.is_large_list_view(saf.type)
    ):
        if src_arrow_inner.type == tgt_arrow_inner.type:
            return _eval(base.cast(tpf.dtype, strict=options.safe).alias(series_name))

        inner_expr = cast_polars_array(
            pl.element(),
            _child_options(src_arrow_inner, tgt_arrow_inner),
            to_expr=True,
            parent_name=parent_path,
        )

        result = base.list.eval(inner_expr, parallel=True).cast(tpf.dtype, strict=options.safe).alias(series_name)
        return _eval(result)

    # ── Fixed-size List ───────────────────────────────────────────────────────
    if pa.types.is_fixed_size_list(saf.type):
        if src_arrow_inner.type == tgt_arrow_inner.type:
            return _eval(base.cast(tpf.dtype, strict=options.safe).alias(series_name))

        # Cast Array → List first, then cast inner elements via list.eval,
        # then cast back to target (List or Array)
        as_list = base.cast(pl.List(arrow_type_to_polars_type(src_arrow_inner.type)))

        inner_expr = cast_polars_array(
            pl.element(),
            _child_options(src_arrow_inner, tgt_arrow_inner),
            to_expr=True,
            parent_name=parent_path,
        )

        result = as_list.list.eval(inner_expr, parallel=True).cast(tpf.dtype, strict=options.safe).alias(series_name)
        return _eval(result)

    # ── Scalar source — wrap into a 1-element list ────────────────────────────
    scalar_cast = cast_polars_array(
        base,
        _child_options(saf, tgt_arrow_inner),
        to_expr=True,
        parent_name=parent_path,
    )

    result = pl.concat_list([scalar_cast]).cast(tpf.dtype, strict=options.safe).alias(series_name)
    return _eval(result)


# ---------------------------------------------------------------------------
# Core array cast
# ---------------------------------------------------------------------------


@register_converter(pl.Series, pl.Series)
@register_converter(pl.Expr, pl.Expr)
def cast_polars_array(
    array: Union[pl.Series, pl.Expr],
    options: Optional[CastOptions] = None,
    parent_name: Optional[str] = None,
    to_expr: bool = False,
) -> Union[pl.Series, pl.Expr]:
    """Cast a Polars Series / Expr to the dtype described by *options*.

    Returns Expr when:
      - input is Expr, or
      - input is Series and to_expr=True

    Returns Series when input is Series and to_expr=False.
    """
    options = CastOptions.check_arg(options)
    tpf = options.target_polars_field
    need_fill = options.need_nullability_fill(source_obj=array)

    if not options.need_polars_type_cast(source_obj=array):
        out = array
        if need_fill:
            dv = default_arrow_scalar(options.target_field.type, nullable=options.target_field.nullable).as_py()
            out = out.fill_null(dv)

        if isinstance(out, pl.Expr):
            return out.alias(tpf.name)
        return out if out.name == tpf.name else out.alias(tpf.name)

    spf = options.source_polars_field
    spdt, tpdt = spf.dtype, tpf.dtype
    is_series = isinstance(array, pl.Series)
    is_expr = isinstance(array, pl.Expr)

    base_name: str = (array.name if is_series else tpf.name) or ""
    col_name: str = _safe_col_name(series_name=base_name, parent_name=parent_name)

    # ── Null source ───────────────────────────────────────────────────────────
    if spdt.__class__ is pl.Null:
        if is_expr:
            if tpf.nullable:
                out_ne: Union[pl.Series, pl.Expr] = array.cast(tpdt)
            else:
                dv = default_arrow_scalar(options.target_field.type, nullable=tpf.nullable).as_py()
                out_ne = pl.lit(dv, dtype=tpdt)
            return out_ne.alias(tpf.name)

        # Series input
        if to_expr:
            base = pl.col(col_name)
            if tpf.nullable:
                out_ex = base.cast(tpdt, strict=False).alias(tpf.name)
            else:
                dv = default_arrow_scalar(options.target_field.type, nullable=tpf.nullable).as_py()
                out_ex = pl.lit(dv, dtype=tpdt).alias(tpf.name)
            return out_ex

        if tpf.nullable:
            out_s = array.cast(tpdt)
        else:
            dv = default_arrow_scalar(options.target_field.type, nullable=tpf.nullable).as_py()
            out_s = pl.Series(tpf.name, [dv] * len(array), dtype=tpdt)

        return out_s if out_s.name == tpf.name else out_s.alias(tpf.name)

    # ── Temporal target ───────────────────────────────────────────────────────
    elif tpdt.is_temporal():
        array = cast_polars_array_to_temporal(
            array,
            source=spdt,
            target=tpdt,
            safe=options.safe,
            source_tz=getattr(options, "source_tz", None),
            target_tz=getattr(options, "target_tz", None),
            to_expr=to_expr,
            parent_name=col_name,
        )

    # ── Struct target ─────────────────────────────────────────────────────────
    elif tpdt.__class__ is pl.Struct:
        array = cast_polars_array_to_struct(array, options, to_expr=to_expr, parent_name=col_name)

    # ── List / Array target ───────────────────────────────────────────────────
    elif tpdt.__class__ in (pl.List, pl.Array):
        array = cast_polars_array_to_list(array, options, to_expr=to_expr, parent_name=col_name)

    # ── Boolean target ────────────────────────────────────────────────────────
    elif tpdt.__class__ is pl.Boolean:
        array = cast_polars_array_to_bool(array, options, to_expr=to_expr, parent_name=col_name)

    # ── Scalar cast ───────────────────────────────────────────────────────────
    else:
        if is_expr:
            array = array.cast(tpdt, strict=options.safe)
        elif to_expr:
            array = pl.col(col_name).cast(tpdt, strict=options.safe)
        else:
            array = array.cast(tpdt, strict=options.safe)

    # ── Nullability fill ──────────────────────────────────────────────────────
    if need_fill:
        dv = default_arrow_scalar(options.target_field.type, nullable=options.target_field.nullable).as_py()
        array = array.fill_null(dv)

    if isinstance(array, pl.Expr):
        return array.alias(tpf.name)
    return array if array.name == tpf.name else array.alias(tpf.name)


# ---------------------------------------------------------------------------
# DataFrame cast
# ---------------------------------------------------------------------------


@register_converter(pl.DataFrame, pl.DataFrame)
def cast_polars_dataframe(df: pl.DataFrame, options: Optional[CastOptions] = None) -> pl.DataFrame:
    """Cast a Polars DataFrame to a target Arrow schema (expr-first on eager DF)."""
    options = CastOptions.check_arg(options)
    options.check_source(df)
    target_schema = options.target_arrow_schema

    if target_schema is None:
        return df

    # ── Build source lookups ──────────────────────────────────────────────────
    src_pl_fields: list[pl.Field] = [pl.Field(n, d) for n, d in df.schema.items()]
    src_arrow_fields: list[pa.Field] = [polars_field_to_arrow_field(f) for f in src_pl_fields]

    exact_idx: dict[str, int] = {f.name: i for i, f in enumerate(src_pl_fields)}
    folded_idx: dict[str, int] = {f.name.casefold(): i for i, f in enumerate(src_pl_fields)}

    # ── Walk target schema and build Exprs ────────────────────────────────────
    result_exprs: list[pl.Expr] = []
    found_src_indices: set[int] = set()

    for col_pos, tgt_arrow_field in enumerate(target_schema):
        tgt_pl_field = arrow_field_to_polars_field(tgt_arrow_field)

        # Name resolution: exact → case-insensitive → positional
        src_idx: int | None = exact_idx.get(tgt_pl_field.name)
        if src_idx is None and not options.strict_match_names:
            src_idx = folded_idx.get(tgt_pl_field.name.casefold())
        if src_idx is None and not options.strict_match_names and col_pos < len(src_pl_fields):
            src_idx = col_pos

        if src_idx is None:
            # ── Missing column ────────────────────────────────────────────────
            if not options.add_missing_columns:
                raise pa.ArrowInvalid(
                    f"Column '{tgt_pl_field.name}' ({tgt_arrow_field.type}) not found in "
                    f"source DataFrame.  Available: {list(exact_idx)}.\n"
                    f"  Hint: pass add_missing_columns=True to fill with defaults, or "
                    f"set strict_match_names=False to enable case-insensitive / "
                    f"positional matching."
                )

            dv = default_arrow_scalar(tgt_arrow_field.type, nullable=tgt_arrow_field.nullable).as_py()
            col_expr = pl.lit(dv, dtype=tgt_pl_field.dtype).alias(tgt_pl_field.name)

        else:
            # ── Cast source column as Expr (forced) ───────────────────────────
            found_src_indices.add(src_idx)
            src_name = src_pl_fields[src_idx].name

            col_expr = cast_polars_array(
                pl.col(src_name),
                options.copy(
                    source_field=src_arrow_fields[src_idx],
                    target_field=tgt_arrow_field,
                ),
                to_expr=True,  # force expr mode for inner casts
                parent_name=src_name,
            )

            # Rename to target if source name differed
            if src_name != tgt_pl_field.name:
                col_expr = col_expr.alias(tgt_pl_field.name)

        result_exprs.append(col_expr)

    # ── Append extra (unmatched) source columns ───────────────────────────────
    if options.allow_add_columns:
        for i, src_field in enumerate(src_pl_fields):
            if i not in found_src_indices:
                result_exprs.append(pl.col(src_field.name))

    # single pass evaluation
    return df.select(result_exprs)


def default_polars_array(is_expr: bool, num_rows: int, arrow_field: pa.Field, dtype: pl.DataType):
    value = default_arrow_scalar(arrow_field.type, nullable=arrow_field.nullable).as_py()

    if is_expr:
        return pl.lit(value, dtype=dtype).alias(arrow_field.name)

    return pl.Series(
        name=arrow_field.name,
        values=[value] * num_rows,
        dtype=dtype,
    )


@register_converter(pl.LazyFrame, pl.LazyFrame)
def cast_polars_lazyframe(lf: pl.LazyFrame, options: Optional[CastOptions] = None) -> pl.LazyFrame:
    """Cast a Polars LazyFrame to a target Arrow schema — fully lazy.

    Uses ``lf.schema`` (no collect) to build one ``pl.Expr`` per target
    column, then issues a single ``lf.select(exprs)`` so the cast is folded
    into the query plan.

    Column matching strategy (priority order)
    ─────────────────────────────────────────
    1. Exact name match.
    2. Case-insensitive match  (when ``strict_match_names=False``).
    3. Positional fallback     (when ``strict_match_names=False``).

    ``add_missing_columns=True``  — synthesise missing target columns with
                                    ``pl.lit(default)``.
    ``allow_add_columns=True``    — append unmatched source columns at the end.

    Raises
    ------
    pa.ArrowInvalid
        When a required target column is absent and
        ``add_missing_columns=False``.
    """
    options = CastOptions.check_arg(options)
    target_schema = options.target_arrow_schema

    if target_schema is None:
        return lf

    lf_schema: dict[str, pl.DataType] = dict(lf.schema)

    # ── Build source lookups from schema (no collect) ─────────────────────────
    src_pl_fields: list[pl.Field] = [pl.Field(n, d) for n, d in lf_schema.items()]
    src_arrow_fields: list[pa.Field] = [polars_field_to_arrow_field(f) for f in src_pl_fields]

    exact_idx: dict[str, int] = {f.name: i for i, f in enumerate(src_pl_fields)}
    folded_idx: dict[str, int] = {f.name.casefold(): i for i, f in enumerate(src_pl_fields)}

    select_exprs: list[pl.Expr] = []
    found_src_indices: set[int] = set()

    for col_pos, tgt_arrow_field in enumerate(target_schema):
        tgt_pl_field = arrow_field_to_polars_field(tgt_arrow_field)

        # Name resolution: exact → case-insensitive → positional
        src_idx: int | None = exact_idx.get(tgt_pl_field.name)
        if src_idx is None and not options.strict_match_names:
            src_idx = folded_idx.get(tgt_pl_field.name.casefold())
        if src_idx is None and not options.strict_match_names and col_pos < len(src_pl_fields):
            src_idx = col_pos

        if src_idx is None:
            # ── Missing column ────────────────────────────────────────────────
            if not options.add_missing_columns:
                raise pa.ArrowInvalid(
                    f"Column '{tgt_pl_field.name}' ({tgt_arrow_field.type}) not found in "
                    f"source LazyFrame.  Available: {list(exact_idx)}.\n"
                    f"  Hint: pass add_missing_columns=True to fill with defaults, or "
                    f"set strict_match_names=False to enable case-insensitive / "
                    f"positional matching."
                )
            dv = default_arrow_scalar(tgt_arrow_field.type, nullable=tgt_arrow_field.nullable).as_py()
            expr = pl.lit(dv, dtype=tgt_pl_field.dtype).alias(tgt_pl_field.name)
        else:
            # ── Cast source column as Expr — stays in the query plan ──────────
            found_src_indices.add(src_idx)
            src_col_name = src_pl_fields[src_idx].name
            expr = cast_polars_array(
                pl.col(src_col_name),
                options.copy(
                    source_field=src_arrow_fields[src_idx],
                    target_field=tgt_arrow_field,
                ),
                parent_name=src_col_name,
            )
            # Rename when source name differs from target (positional / folded)
            if src_col_name != tgt_pl_field.name:
                expr = expr.alias(tgt_pl_field.name)

        select_exprs.append(expr)

    # ── Append unmatched source columns ───────────────────────────────────────
    if options.allow_add_columns:
        for i, src_field in enumerate(src_pl_fields):
            if i not in found_src_indices:
                select_exprs.append(pl.col(src_field.name))

    return lf.select(select_exprs)


# ---------------------------------------------------------------------------
# Arrow <-> Polars table / dataframe converters
# ---------------------------------------------------------------------------


@register_converter(pl.DataFrame, pa.Table)
@register_converter(pl.LazyFrame, pa.Table)
def polars_dataframe_to_arrow_table(data: pl.DataFrame | pl.LazyFrame, options: Optional[CastOptions] = None) -> pa.Table:
    """Convert a Polars DataFrame to a ``pyarrow.Table``.

    When a target schema is configured, :func:`cast_polars_dataframe` is
    applied first; any residual Arrow-level differences are resolved by a
    subsequent :func:`cast_arrow_tabular` pass.
    """
    options = CastOptions.check_arg(options)

    if options.target_field is not None:
        data = cast_polars_dataframe(data, options)

    compat_level = pl.CompatLevel.newest()
    try:
        table = data.to_arrow(compat_level=compat_level)
    except pa.ArrowInvalid:
        table = data.rechunk().to_arrow(compat_level=compat_level)

    if options.target_field is not None:
        table = cast_arrow_tabular(table, options)

    return table


@register_converter(pa.Table, pl.DataFrame)
def arrow_table_to_polars_dataframe(table: pa.Table, options: Optional[CastOptions] = None) -> pl.DataFrame:
    """Convert a ``pyarrow.Table`` to a Polars DataFrame.

    When a target schema is configured, ``cast_arrow_tabular`` is applied
    before conversion so Arrow-level type differences are resolved first.
    """
    options = CastOptions.check_arg(options)

    if options.target_arrow_schema is not None:
        table = cast_arrow_tabular(table, options)

    return pl.from_arrow(table)


@register_converter(Any, pl.DataFrame)
def any_to_polars_dataframe(obj: Any, options: Optional[CastOptions] = None) -> pl.DataFrame:
    """Convert any supported object to a Polars DataFrame.

    Supported inputs: ``None``, file path, pandas DataFrame, PyArrow
    Table / RecordBatch, PySpark DataFrame, plain dict / list, or anything
    Polars can ingest via ``pl.DataFrame()``.
    """
    if not isinstance(obj, pl.DataFrame):
        options = CastOptions.check_arg(options)

        if obj is None:
            return pl.DataFrame([], schema=options.target_polars_schema)

        if isinstance(obj, (str, SystemPath)):
            return LocalDataPath(obj).read_polars(cast_options=options)

        ns = ObjectSerde.full_namespace(obj)

        if ns.startswith("pandas."):
            obj = pl.from_pandas(obj, nan_to_null=True, include_index=bool(obj.index.name))
        elif ns.startswith("pyarrow."):
            obj = pl.from_arrow(obj)
        elif ns.startswith("pyspark."):
            from ..spark.lib import pyspark_sql

            obj = obj  # type: ignore[no-redef]
            obj = pl.from_arrow(obj.toArrow())
        else:
            obj = pl.DataFrame(obj)

    return cast_polars_dataframe(obj, options)


# ---------------------------------------------------------------------------
# PyArrow DataType <-> Polars DataType
# ---------------------------------------------------------------------------


@register_converter(pa.DataType, pl.DataType)
def arrow_type_to_polars_type(
    arrow_type: Union[pa.DataType, pa.TimestampType, pa.ListType, pa.MapType],
    options: Optional[dict] = None,
) -> pl.DataType:
    """Convert a ``pyarrow.DataType`` to a Polars dtype.

    Raises
    ------
    TypeError
        When *arrow_type* has no known Polars equivalent.
    """
    # Fast path: primitive lookup
    dtype = ARROW_TO_POLARS.get(arrow_type)
    if dtype is not None:
        return dtype

    if pat.is_timestamp(arrow_type):
        unit = arrow_type.unit if arrow_type.unit != "s" else "ms"
        return pl.Datetime(time_unit=unit, time_zone=arrow_type.tz)

    if pat.is_date(arrow_type):
        return pl.Date()

    if pat.is_time(arrow_type):
        return pl.Time()

    if pat.is_duration(arrow_type):
        unit = arrow_type.unit if arrow_type.unit != "s" else "ms"
        return pl.Duration(time_unit=unit)

    if pat.is_decimal(arrow_type):
        return pl.Decimal(precision=arrow_type.precision, scale=arrow_type.scale)

    if is_arrow_type_binary_like(arrow_type):
        return pl.Binary()

    if is_arrow_type_string_like(arrow_type):
        return pl.Utf8()

    if pat.is_dictionary(arrow_type):
        return pl.Categorical()

    if pat.is_map(arrow_type):
        key = arrow_type_to_polars_type(arrow_type.key_type)
        val = arrow_type_to_polars_type(arrow_type.item_type)
        return pl.List(pl.Struct([pl.Field("key", key), pl.Field("value", val)]))

    if is_arrow_type_list_like(arrow_type):
        return pl.List(arrow_type_to_polars_type(arrow_type.value_type))

    if pat.is_struct(arrow_type):
        return pl.Struct([pl.Field(f.name, arrow_type_to_polars_type(f.type)) for f in arrow_type])

    raise TypeError(
        f"No Polars equivalent for Arrow type {arrow_type!r} (id={arrow_type.id}).\n"
        f"  Hint: Register a custom converter via "
        f"register_converter(pa.DataType, pl.DataType), or cast the Arrow column "
        f"to a supported type before conversion."
    )


@register_converter(pa.Field, pl.Field)
def arrow_field_to_polars_field(field: pa.Field, options: Optional[dict] = None) -> pl.Field:
    """Convert a ``pyarrow.Field`` to a ``pl.Field``."""
    built = pl.Field(field.name, arrow_type_to_polars_type(field.type))
    try:
        built.nullable = field.nullable
    except AttributeError:
        pass  # older Polars versions may not support assignment
    return built


@register_converter(pl.DataType, pa.DataType)
def polars_type_to_arrow_type(pl_type: Union[pl.DataType, type], options: Optional[dict] = None) -> pa.DataType:
    """Convert a Polars dtype (class or instance) to a ``pyarrow.DataType``.

    Raises
    ------
    TypeError
        When *pl_type* has no known Arrow equivalent.
    """
    existing = POLARS_BASE_TO_ARROW.get(pl_type) or POLARS_BASE_TO_ARROW.get(type(pl_type))
    if existing is not None:
        return existing

    if pl_type.is_nested():
        if isinstance(pl_type, pl.Array):
            return pa.list_(
                polars_type_to_arrow_type(pl_type.inner),
                list_size=pl_type.shape[0],  # type: ignore
            )

        if isinstance(pl_type, pl.List):
            return pa.list_(polars_type_to_arrow_type(pl_type.inner))

        if isinstance(pl_type, pl.Struct):
            return pa.struct([polars_field_to_arrow_field(f) for f in pl_type.fields])
    else:
        if pl_type.is_temporal():
            if isinstance(pl_type, pl.Datetime):
                return pa.timestamp(pl_type.time_unit, tz=pl_type.time_zone)

            if isinstance(pl_type, pl.Duration):
                return pa.duration(pl_type.time_unit)

            if isinstance(pl_type, pl.Time):
                return pa.time64("ns")

        if pl_type.is_decimal():
            p, s = pl_type.precision, pl_type.scale
            return pa.decimal128(p, s) if (p is None or p <= 38) else pa.decimal256(p, s)

        if isinstance(pl_type, (pl.Categorical, pl.Enum)):
            return pa.dictionary(index_type=pa.int32(), value_type=pa.string())

        if isinstance(pl_type, pl.Object):
            return pa.string()

    raise TypeError(
        f"No Arrow equivalent for Polars dtype {pl_type!r} "
        f"(class={type(pl_type).__name__}).\n"
        f"  Hint: Register a custom converter via "
        f"register_converter(pl.DataType, pa.DataType), or cast the Polars column "
        f"to a supported dtype first (e.g. series.cast(pl.Utf8))."
    )


@register_converter(pl.Field, pa.Field)
def polars_field_to_arrow_field(field: pl.Field, options: Optional[CastOptions] = None) -> pa.Field:
    """Convert a ``pl.Field`` to a ``pyarrow.Field``."""
    return pa.field(
        field.name,
        polars_type_to_arrow_type(field.dtype),
        nullable=getattr(field, "nullable", True),
        metadata=getattr(field, "metadata", None),
    )


@register_converter(pl.Series, pa.Field)
@register_converter(pl.Expr, pa.Field)
def polars_array_to_arrow_field(array: Union[pl.Series, pl.Expr], options: Optional[CastOptions] = None) -> pa.Field:
    """Infer an Arrow field from a Polars Series or Expr."""
    return polars_field_to_arrow_field(polars_array_to_polars_field(array, options))


@register_converter(pl.Series, pl.Field)
@register_converter(pl.Expr, pl.Field)
def polars_array_to_polars_field(array: Union[pl.Series, pl.Expr], options: Optional[CastOptions] = None) -> pl.Field:
    """Infer a Polars field from a Polars Series or Expr."""
    name = getattr(array, "name", "") or ""
    dtype = getattr(array, "dtype", pl.Object())
    null_count = getattr(array, "null_count", lambda: None)()
    nullable = (null_count > 0) if null_count is not None else True
    f = pl.Field(name=name, dtype=dtype)
    f.nullable = nullable
    return f


# ---------------------------------------------------------------------------
# any_polars_to_arrow_field — broad dispatch
# ---------------------------------------------------------------------------


def any_polars_to_arrow_field(obj: Any, options: Optional[CastOptions] = None) -> pa.Field:
    """Convert any Polars object to a ``pyarrow.Field``.

    Handles ``pl.DataFrame``, ``pl.LazyFrame``, ``pl.Series``, ``pl.Expr``,
    ``pl.DataType`` (instance or class), and ``pl.Field``.
    """
    options = CastOptions.check_arg(options)

    if options.target_arrow_field is not None:
        return options.target_arrow_field

    def _meta() -> tuple[str, bool, Any]:
        if options.target_field is not None:
            return options.target_field.name, options.target_field.nullable, options.target_field.metadata

        if options.source_field is not None:
            return options.source_field.name, options.source_field.nullable, options.source_field.metadata

        return "root", True, None

    if isinstance(obj, pl.DataFrame):
        fields = [polars_array_to_arrow_field(obj.to_series(i)) for i in range(obj.shape[1])]
        return pa.field("root", pa.struct(fields), metadata={"__class__": "polars.DataFrame"})

    if isinstance(obj, pl.LazyFrame):
        fields = [polars_field_to_arrow_field(pl.Field(name=n, dtype=d)) for n, d in obj.schema.items()]
        return pa.field("root", pa.struct(fields), metadata={"__class__": "polars.LazyFrame"})

    if isinstance(obj, (pl.Series, pl.Expr)):
        return polars_array_to_arrow_field(obj)

    if isinstance(obj, pl.DataType):
        name, nullable, meta = _meta()
        return pa.field(name, polars_type_to_arrow_type(obj), nullable=nullable, metadata=meta)

    if isinstance(obj, pl.Field):
        name, nullable, meta = _meta()
        return pa.field(
            obj.name,
            polars_type_to_arrow_type(obj.dtype),
            nullable=nullable,
            metadata=meta,
        )

    if isinstance(obj, pl.Schema):
        name, nullable, meta = _meta()
        arrow_type = pa.struct([polars_field_to_arrow_field(pl.Field(name=k, dtype=t)) for k, t in obj.items()])

        return pa.field(
            name=name,
            type=arrow_type,
            nullable=False,
            metadata=meta,
        )

    if isinstance(obj, POLARS_TYPE_CLASSES):
        name, nullable, meta = _meta()
        return pa.field(name, polars_type_to_arrow_type(type(obj)()), nullable=nullable, metadata=meta)

    raise ValueError(
        f"Cannot convert {type(obj)!r} to a pyarrow.Field.\n"
        f"  Accepted types: pl.DataFrame, pl.LazyFrame, pl.Series, pl.Expr, "
        f"pl.DataType (instance or class), pl.Field.\n"
        f"  Received: {obj!r:.120}"
    )