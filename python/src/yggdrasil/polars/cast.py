"""Polars <-> Arrow casting helpers and converters."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import polars as pl
import pyarrow as pa
import pyarrow.types as pat

from ..io.path import LocalDataPath, SystemPath
from ..pyutils.serde import ObjectSerde
from ..types.cast.arrow_cast import (
    cast_arrow_tabular,
    is_arrow_type_binary_like,
    is_arrow_type_list_like,
    is_arrow_type_string_like,
)
from ..types.cast.cast_options import CastOptions
from ..types.cast.registry import register_converter
from ..types.python_defaults import default_arrow_scalar

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

POLARS_BASE_TO_ARROW: Dict[pl.DataType, pa.DataType] = {
    v: k for k, v in ARROW_TO_POLARS.items()
}

POLARS_TYPE_CLASSES: tuple = tuple(v.__class__ for v in ARROW_TO_POLARS.values())

# Timezone alias groups — members are treated as equivalent when normalising.
TIMEZONE_ALIASES: Dict[str, frozenset] = {
    "UTC": frozenset(("UTC", "Etc/UTC", "+00:00")),
    "CET": frozenset(("CET", "Europe/Paris", "Europe/Zurich")),
}

# Numeric source classes (used repeatedly in temporal dispatch)
_NUMERIC_CLASSES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)

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
# with naive results.  Mixing tz-aware and naive in one pl.coalesce() raises
# SchemaError("failed to determine supertype of datetime[μs, UTC] and datetime[μs]").
_DATETIME_FORMATS_WITH_TZ = [
    "%Y-%m-%dT%H:%M:%S%.f%z",
    "%Y-%m-%dT%H:%M:%S%z",
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


def _unsafe_string_to_datetime(arr: pl.Expr, dtype: pl.Datetime) -> pl.Expr:
    """Parse string to Datetime, trying many formats, yielding a naive result.

    tz-aware formats (%z) are parsed to UTC then stripped to naive so that
    the entire coalesce list has a uniform dtype (no supertype resolution
    across tz-aware / naive that would raise SchemaError).

    The caller (:func:`cast_polars_array_to_temporal`) applies timezone
    semantics via :func:`_apply_tz` after this function returns.
    """
    tu = dtype.time_unit or "us"
    bare = pl.Datetime(tu)
    utc  = pl.Datetime(tu, "UTC")

    # Parse tz-aware formats → UTC → strip to naive (uniform bare dtype)
    tz_parsed = [
        _safe_strptime(arr, utc, fmt)
            .dt.convert_time_zone("UTC")
            .dt.replace_time_zone(None)
        for fmt in _DATETIME_FORMATS_WITH_TZ
    ]
    # Parse naive formats directly
    naive_parsed = [_safe_strptime(arr, bare, fmt) for fmt in _DATETIME_FORMATS_NAIVE]

    # All branches now have dtype Datetime[tu] — safe to coalesce
    return pl.coalesce(tz_parsed + naive_parsed)


def _unsafe_string_to_date(arr: pl.Expr) -> pl.Expr:
    return _coalesce_strptime(arr, pl.Date(), _DATE_FORMATS)


def _unsafe_string_to_time(arr: pl.Expr) -> pl.Expr:
    return _coalesce_strptime(arr, pl.Time(), _TIME_FORMATS)


def _unsafe_string_to_duration(arr: pl.Expr, dtype: pl.Duration) -> pl.Expr:
    """Interpret integer strings as epoch-offset integers → Duration."""
    tu = dtype.time_unit or "us"
    return arr.cast(pl.Int64, strict=False).cast(pl.Duration(tu), strict=False)


def _apply_tz(
    expr: pl.Expr,
    src_tz: str | None,
    tgt_tz: str | None,
) -> pl.Expr:
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
        return (
            expr.dt.replace_time_zone(src_tz, ambiguous="earliest")
                .dt.convert_time_zone("UTC")
                .dt.replace_time_zone(None)
        )

    if src_tz == tgt_tz:
        return expr.dt.replace_time_zone(src_tz, ambiguous="earliest")

    return (
        expr.dt.replace_time_zone(src_tz, ambiguous="earliest")
            .dt.convert_time_zone(tgt_tz)
    )


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
) -> Union[pl.Series, pl.Expr]:
    """Cast *array* to a temporal Polars dtype with full timezone handling.

    Parameters
    ----------
    array:
        Source Series or Expr.
    source:
        Source Polars dtype (used for dispatch; avoids a dtype call on Expr).
    target:
        Target temporal dtype instance (``pl.Datetime``, ``pl.Date``,
        ``pl.Time``, or ``pl.Duration``).
    safe:
        When ``True`` strict parsing / casting is used and errors raise.
        When ``False`` coalesce-based multi-format tries are used and
        unparseable rows become null.
    source_tz / target_tz:
        Explicit timezone overrides.  Dtype-embedded ``time_zone`` takes
        precedence; these are used when the dtype carries no tz metadata
        (e.g. integer epoch columns that logically represent UTC).

    Returns
    -------
    Same type as *array*.
    """
    is_expr = isinstance(array, pl.Expr)
    series_name = "" if is_expr else array.name  # type: ignore[union-attr]
    working: pl.Expr = array if is_expr else pl.lit(array)  # type: ignore[arg-type]

    src_cls = source.__class__
    tgt_cls = target.__class__

    # Timezone resolution: dtype-embedded wins over explicit override
    _src_tz: str | None = (
        source.time_zone if src_cls is pl.Datetime else None  # type: ignore[union-attr]
    ) or source_tz

    _tgt_tz: str | None = (
        target.time_zone if tgt_cls is pl.Datetime else None  # type: ignore[union-attr]
    ) or target_tz

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
                working = _unsafe_string_to_datetime(working, target)  # type: ignore[arg-type]
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
            working = working.cast(target, strict=safe)  # type: ignore[arg-type]

    # ── Date ──────────────────────────────────────────────────────────────────
    elif tgt_cls is pl.Date:
        # Date is always naive. For tz-aware Datetime sources we convert to
        # target_tz first so that .date() reflects the correct calendar day.
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
        # Time is always naive. Convert tz-aware Datetime to target_tz before
        # extracting wall-clock time so the hour is locally correct.
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
            # ns-since-midnight physical int → Time
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
            # Normalise to UTC before differencing so the result is physically
            # consistent regardless of source timezone.
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

    if not is_expr:
        return pl.select(working).to_series().alias(series_name)
    return working


# ---------------------------------------------------------------------------
# Struct array cast
# ---------------------------------------------------------------------------

def _resolve_source_field(
    name: str,
    src_fields: list[pl.Field],
    strict_match_names: bool,
) -> pl.Field | None:
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
    source_field: pl.Field,
    target_field: pl.Field,
    options: "CastOptions",
) -> Union[pl.Series, pl.Expr]:
    """
    Cast a Series / Expr to a pl.Struct dtype, field-by-field, staying lazy for Expr.

    Matching:
      - exact name / case-insensitive (via _resolve_source_field)
      - positional fallback when strict_match_names=False

    Missing target fields:
      - raise if add_missing_columns=False
      - else fill with nulls cast to target dtype (Polars-native)

    Extra source fields:
      - preserved when allow_add_columns=True
    """
    try:
        source_dtype, target_dtype = source_field.dtype, target_field.dtype
    except AttributeError:
        source_dtype, target_dtype = source_field, target_field
        source_field = pl.Field("", source_dtype)
        target_field = pl.Field("", target_dtype)

    if not isinstance(target_dtype, pl.Struct):
        raise TypeError(f"target_field.dtype must be pl.Struct, got {target_dtype!r}")

    target: pl.Struct = target_dtype
    is_expr = isinstance(array, pl.Expr)
    series_name = source_field.name if is_expr else array.name  # type: ignore[union-attr]

    # ------------------------------------------------------------
    # Normalize input into a Struct (Expr stays Expr; Series stays Series)
    # ------------------------------------------------------------
    src_cls = source_dtype.__class__

    if src_cls is not pl.Struct:
        # JSON-ish to struct
        if src_cls in (pl.String, pl.Utf8, pl.Object):
            array = array.str.json_decode(dtype=target)  # type: ignore[union-attr]
        else:
            # Wrap scalar into 1-field struct using the first target field
            if not target.fields:
                # empty struct: best we can do is cast directly
                if is_expr:
                    return pl.lit(None, dtype=target).alias(series_name)
                return pl.Series(series_name, [None] * len(array), dtype=target)  # type: ignore[arg-type]

            first_tgt = target.fields[0]
            child = cast_polars_array(array, options.copy(target_field=first_tgt))

            if is_expr:
                array = pl.struct([child.alias(first_tgt.name)]).alias(series_name)  # type: ignore[union-attr]
            else:
                # build struct series via one eager select (minimal)
                child_s = child.rename(first_tgt.name)  # type: ignore[union-attr]
                array = pl.DataFrame({first_tgt.name: child_s}).select(
                    pl.struct(pl.all()).alias(series_name)
                ).to_series()

    # After normalization, treat the source as struct for field mapping.
    # Prefer the actual struct dtype if we can, else fall back to source_field dtype.
    struct_dtype = array.dtype if not is_expr else source_dtype  # type: ignore[union-attr]
    if not isinstance(struct_dtype, pl.Struct):
        # json_decode can still yield Null if everything is null; handle gracefully
        struct_dtype = source_dtype if isinstance(source_dtype, pl.Struct) else pl.Struct([])

    src_fields: list[pl.Field] = struct_dtype.fields  # type: ignore[union-attr]
    tgt_fields: list[pl.Field] = target.fields

    # Base expression used to extract fields
    base = array if is_expr else pl.col("__s__")

    found_src_names: set[str] = set()
    out_field_exprs: list[pl.Expr] = []

    # ------------------------------------------------------------
    # Build expressions for each target field (lazy)
    # ------------------------------------------------------------
    for pos, tgt_f in enumerate(tgt_fields):
        src_f = _resolve_source_field(tgt_f.name, src_fields, options.strict_match_names)

        if src_f is None and not options.strict_match_names and pos < len(src_fields):
            src_f = src_fields[pos]

        if src_f is None:
            if not options.add_missing_columns:
                raise ValueError(
                    f"Struct field '{tgt_f.name}' not found in source struct "
                    f"{[f.name for f in src_fields]}. "
                    f"Hint: set add_missing_columns=True to fill missing fields."
                )
            # Polars-native default: null cast to target dtype
            out_field_exprs.append(pl.lit(None, dtype=tgt_f.dtype).alias(tgt_f.name))
            continue

        found_src_names.add(src_f.name)
        field_expr = base.struct.field(src_f.name)

        # Recurse with per-field options (stay in Polars land; no Arrow conversions)
        child_options = options.copy(source_field=src_f, target_field=tgt_f)
        out_field_exprs.append(cast_polars_array(field_expr, child_options).alias(tgt_f.name))

    # Extra source fields passthrough
    if options.allow_add_columns:
        for src_f in src_fields:
            if src_f.name not in found_src_names:
                out_field_exprs.append(base.struct.field(src_f.name).alias(src_f.name))

    # Assemble output struct expr
    struct_expr = pl.struct(out_field_exprs).alias(series_name)

    # ------------------------------------------------------------
    # Return lazily for Expr; evaluate once for Series
    # ------------------------------------------------------------
    if is_expr:
        return struct_expr

    df = pl.DataFrame({"__s__": array})
    return df.select(struct_expr).to_series()


# ---------------------------------------------------------------------------
# List / Array cast
# ---------------------------------------------------------------------------

import polars as pl
from typing import Union

def cast_polars_array_to_list(
    array: Union[pl.Series, pl.Expr],
    source: pl.DataType,
    target: Union[pl.List, pl.Array],
    options: "CastOptions",
) -> Union[pl.Series, pl.Expr]:
    """
    Cast a Series / Expr to pl.List or pl.Array, casting inner elements recursively.

    - Expr path stays fully lazy (returns Expr).
    - Series path evaluates once at the end.
    - If source is List/Array: cast inner via list.eval(pl.element() -> cast_polars_array(...)).
    - If source is scalar: wrap into 1-element list, then cast inner.
    """
    is_expr = isinstance(array, pl.Expr)
    series_name = "" if is_expr else array.name  # type: ignore[union-attr]

    src_cls = source.__class__
    tgt_cls = target.__class__
    tgt_inner: pl.DataType = target.inner  # type: ignore[union-attr]

    base = array if is_expr else pl.col("__s__")

    # ---------------------------------------------------------------------
    # Source is already a List/Array → cast inner elements lazily
    # ---------------------------------------------------------------------
    if src_cls in (pl.List, pl.Array):
        src_inner: pl.DataType = source.inner  # type: ignore[union-attr]

        # Fast path: identical inner dtype, just cast container type
        if src_inner == tgt_inner:
            expr = base.cast(target, strict=options.safe).alias(series_name)
            if is_expr:
                return expr
            return pl.DataFrame({"__s__": array}).select(expr).to_series()

        # Recursive inner cast using list.eval
        child_options = options.copy(
            source_field=pl.Field("element", src_inner),
            target_field=pl.Field("element", tgt_inner),
        )

        inner_expr = cast_polars_array(pl.element(), child_options)

        # list.eval exists for List; for Array, Polars still exposes list namespace
        # on array-like in most versions. If not, you can cast Array->List first.
        list_casted = base.list.eval(inner_expr, parallel=True)

        expr = list_casted.cast(target, strict=options.safe).alias(series_name)

        if is_expr:
            return expr
        return pl.DataFrame({"__s__": array}).select(expr).to_series()

    # ---------------------------------------------------------------------
    # Scalar source → wrap into list, then cast inner
    # ---------------------------------------------------------------------
    child_options = options.copy(
        source_field=pl.Field("element", source),
        target_field=pl.Field("element", tgt_inner),
    )

    # Cast the scalar itself (recursive)
    scalar_cast = cast_polars_array(base, child_options)

    # Wrap each scalar value into a 1-element list.
    # Use concat_list so this works lazily for Expr and eagerly for Series.
    wrapped = pl.concat_list([scalar_cast]).alias(series_name)

    # If target is Array, enforce width=1 (or let cast fail if width != 1)
    expr = wrapped.cast(target, strict=options.safe).alias(series_name) if tgt_cls is pl.Array else wrapped.cast(target, strict=options.safe).alias(series_name)

    if is_expr:
        return expr
    return pl.DataFrame({"__s__": array}).select(expr).to_series()


# ---------------------------------------------------------------------------
# Core array cast
# ---------------------------------------------------------------------------

@register_converter(pl.Series, pl.Series)
@register_converter(pl.Expr, pl.Expr)
def cast_polars_array(
    array: Union[pl.Series, pl.Expr],
    options: Optional[CastOptions] = None,
) -> Union[pl.Series, pl.Expr]:
    """Cast a Polars Series / Expr to the dtype described by *options*.

    Dispatch order
    ──────────────
    1. No-op when source and target dtypes already match.
    2. Null source → typed null column (any target).
    3. Temporal target → :func:`cast_polars_array_to_temporal`.
    4. Nested ↔ any → Arrow round-trip (handles List / Struct / Map).
    5. Scalar → ``series.cast(target_dtype)``.

    Nullability fill is applied after the cast when required.
    """
    options = CastOptions.check_arg(options)
    tpf = options.target_polars_field
    need_fill = options.need_nullability_fill(source_obj=array)

    if not options.need_polars_type_cast(source_obj=array):
        if need_fill:
            dv = default_arrow_scalar(
                options.target_field.type,
                nullable=options.target_field.nullable,
            ).as_py()
            array = array.fill_null(dv)
        return array if array.name == tpf.name else array.alias(tpf.name)

    spf = options.source_polars_field
    spdt, tpdt = spf.dtype, tpf.dtype
    is_series = isinstance(array, pl.Series)

    # ── Null source ───────────────────────────────────────────────────────────
    if spdt.__class__ is pl.Null:
        if tpf.nullable:
            array = array.cast(tpdt) if is_series else pl.lit(None, dtype=tpdt)
        else:
            dv = default_arrow_scalar(options.target_field.type, nullable=tpf.nullable).as_py()
            array = (
                pl.Series(tpf.name, [dv] * len(array), dtype=tpdt)
                if is_series
                else pl.lit(dv, dtype=tpdt)
            )
        return array if array.name == tpf.name else array.alias(tpf.name)

    # ── Temporal target ───────────────────────────────────────────────────────
    if tpdt.is_temporal():
        array = cast_polars_array_to_temporal(
            array,
            source=spdt,
            target=tpdt,
            safe=options.safe,
            source_tz=getattr(options, "source_tz", None),
            target_tz=getattr(options, "target_tz", None),
        )

    # ── Nested cast via Arrow round-trip ──────────────────────────────────────
    # ── Struct target ────────────────────────────────────────────────────────────────
    elif tpdt.__class__ is pl.Struct:
        array = cast_polars_array_to_struct(array, spf, tpf, options)

    # ── List / Array target ─────────────────────────────────────────────────────────
    elif tpdt.__class__ in (pl.List, pl.Array):
        array = cast_polars_array_to_list(array, spdt, tpdt, options)

    # ── Scalar cast ───────────────────────────────────────────────────────────────────────────
    else:
        array = array.cast(tpdt, strict=options.safe)

    # ── Nullability fill ──────────────────────────────────────────────────────
    if need_fill:
        dv = default_arrow_scalar(
            options.target_field.type,
            nullable=options.target_field.nullable,
        ).as_py()
        array = array.fill_null(dv)

    return array if array.name == tpf.name else array.alias(tpf.name)


# ---------------------------------------------------------------------------
# DataFrame cast
# ---------------------------------------------------------------------------

@register_converter(pl.DataFrame, pl.DataFrame)
def cast_polars_dataframe(
    df: pl.DataFrame,
    options: Optional[CastOptions] = None,
) -> pl.DataFrame:
    """Cast a Polars DataFrame to a target Arrow schema.

    Column matching strategy (priority order)
    ─────────────────────────────────────────
    1. Exact name match.
    2. Case-insensitive match  (when ``strict_match_names=False``).
    3. Positional fallback     (when ``strict_match_names=False``).

    ``add_missing_columns=True``  — synthesise missing target columns with
                                    their default / null value.
    ``allow_add_columns=True``    — append unmatched source columns at the end.

    Raises
    ------
    pa.ArrowInvalid
        When a required target column is absent and
        ``add_missing_columns=False``.
    """
    options = CastOptions.check_arg(options)
    options.check_source(df)
    target_schema = options.target_arrow_schema

    if target_schema is None:
        return df

    n_rows = len(df)

    # ── Build source lookups ──────────────────────────────────────────────────
    src_pl_fields: list[pl.Field] = [pl.Field(n, d) for n, d in df.schema.items()]
    src_arrow_fields: list[pa.Field] = [polars_field_to_arrow_field(f) for f in src_pl_fields]

    exact_idx:  dict[str, int] = {f.name: i            for i, f in enumerate(src_pl_fields)}
    folded_idx: dict[str, int] = {f.name.casefold(): i for i, f in enumerate(src_pl_fields)}

    # ── Walk target schema ────────────────────────────────────────────────────
    result_columns: list[pl.Series] = []
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
            dv = default_arrow_scalar(
                tgt_arrow_field.type, nullable=tgt_arrow_field.nullable
            ).as_py()
            col = pl.Series(
                name=tgt_pl_field.name,
                values=[dv] * n_rows,
                dtype=tgt_pl_field.dtype,
            )
        else:
            # ── Cast source column ────────────────────────────────────────────
            found_src_indices.add(src_idx)
            col = cast_polars_array(
                df.to_series(src_idx),
                options.copy(
                    source_field=src_arrow_fields[src_idx],
                    target_field=tgt_arrow_field,
                ),
            )
            # Rename to target if the source name differed (positional / folded match)
            if col.name != tgt_pl_field.name:
                col = col.alias(tgt_pl_field.name)

        result_columns.append(col)

    # ── Append extra (unmatched) source columns ───────────────────────────────
    if options.allow_add_columns:
        for i, src_series in enumerate(df.get_columns()):
            if i not in found_src_indices:
                result_columns.append(src_series)

    return pl.DataFrame(result_columns)


@register_converter(pl.LazyFrame, pl.LazyFrame)
def cast_polars_lazyframe(
    lf: pl.LazyFrame,
    options: Optional[CastOptions] = None,
) -> pl.LazyFrame:
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

    exact_idx:  dict[str, int] = {f.name: i            for i, f in enumerate(src_pl_fields)}
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
            dv = default_arrow_scalar(
                tgt_arrow_field.type, nullable=tgt_arrow_field.nullable
            ).as_py()
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
def polars_dataframe_to_arrow_table(
    data: pl.DataFrame,
    options: Optional[CastOptions] = None,
) -> pa.Table:
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
def arrow_table_to_polars_dataframe(
    table: pa.Table,
    options: Optional[CastOptions] = None,
) -> pl.DataFrame:
    """Convert a ``pyarrow.Table`` to a Polars DataFrame.

    When a target schema is configured, ``cast_arrow_tabular`` is applied
    before conversion so Arrow-level type differences are resolved first.
    """
    options = CastOptions.check_arg(options)

    if options.target_arrow_schema is not None:
        table = cast_arrow_tabular(table, options)

    return pl.from_arrow(table)


@register_converter(Any, pl.DataFrame)
def any_to_polars_dataframe(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pl.DataFrame:
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
            obj: pyspark_sql.DataFrame = obj
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
        return pl.Struct([
            pl.Field(f.name, arrow_type_to_polars_type(f.type))
            for f in arrow_type
        ])

    raise TypeError(
        f"No Polars equivalent for Arrow type {arrow_type!r} (id={arrow_type.id}).\n"
        f"  Hint: Register a custom converter via "
        f"register_converter(pa.DataType, pl.DataType), or cast the Arrow column "
        f"to a supported type before conversion."
    )


@register_converter(pa.Field, pl.Field)
def arrow_field_to_polars_field(
    field: pa.Field,
    options: Optional[dict] = None,
) -> pl.Field:
    """Convert a ``pyarrow.Field`` to a ``pl.Field``."""
    built = pl.Field(field.name, arrow_type_to_polars_type(field.type))
    try:
        built.nullable = field.nullable
    except AttributeError:
        pass  # older Polars versions may not support assignment
    return built


@register_converter(pl.DataType, pa.DataType)
def polars_type_to_arrow_type(
    pl_type: Union[pl.DataType, type],
    options: Optional[dict] = None,
) -> pa.DataType:
    """Convert a Polars dtype (class or instance) to a ``pyarrow.DataType``.

    Raises
    ------
    TypeError
        When *pl_type* has no known Arrow equivalent.
    """
    existing = POLARS_BASE_TO_ARROW.get(pl_type) or POLARS_BASE_TO_ARROW.get(type(pl_type))
    if existing is not None:
        return existing

    if isinstance(pl_type, pl.Datetime):
        return pa.timestamp(pl_type.time_unit, tz=pl_type.time_zone)

    if isinstance(pl_type, pl.Duration):
        return pa.duration(pl_type.time_unit)

    if isinstance(pl_type, pl.Decimal):
        p, s = pl_type.precision, pl_type.scale
        return pa.decimal128(p, s) if (p is None or p <= 38) else pa.decimal256(p, s)

    if isinstance(pl_type, pl.List):
        return pa.list_(polars_type_to_arrow_type(pl_type.inner))

    if isinstance(pl_type, pl.Struct):
        return pa.struct([polars_field_to_arrow_field(f) for f in pl_type.fields])

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
def polars_field_to_arrow_field(
    field: pl.Field,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Convert a ``pl.Field`` to a ``pyarrow.Field``."""
    return pa.field(
        field.name,
        polars_type_to_arrow_type(field.dtype),
        nullable=getattr(field, "nullable", True),
        metadata=getattr(field, "metadata", None),
    )


@register_converter(pl.Series, pa.Field)
@register_converter(pl.Expr, pa.Field)
def polars_array_to_arrow_field(
    array: Union[pl.Series, pl.Expr],
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Infer an Arrow field from a Polars Series or Expr."""
    return polars_field_to_arrow_field(polars_array_to_polars_field(array, options))


@register_converter(pl.Series, pl.Field)
@register_converter(pl.Expr, pl.Field)
def polars_array_to_polars_field(
    array: Union[pl.Series, pl.Expr],
    options: Optional[CastOptions] = None,
) -> pl.Field:
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

def any_polars_to_arrow_field(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Convert any Polars object to a ``pyarrow.Field``.

    Handles ``pl.DataFrame``, ``pl.LazyFrame``, ``pl.Series``, ``pl.Expr``,
    ``pl.DataType`` (instance or class), and ``pl.Field``.
    """
    options = CastOptions.check_arg(options)

    def _meta() -> tuple[str, bool, Any]:
        tf = options.target_field
        return (
            "root" if tf is None else tf.name,
            True if tf is None else tf.nullable,
            None if tf is None else tf.metadata,
        )

    if isinstance(obj, pl.DataFrame):
        fields = [polars_array_to_arrow_field(obj.to_series(i)) for i in range(obj.shape[1])]
        return pa.field("root", pa.struct(fields), metadata={"__class__": "polars.DataFrame"})

    if isinstance(obj, pl.LazyFrame):
        fields = [
            polars_field_to_arrow_field(pl.Field(name=n, dtype=d))
            for n, d in obj.schema.items()
        ]
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

    if isinstance(obj, POLARS_TYPE_CLASSES):
        name, nullable, meta = _meta()
        return pa.field(name, polars_type_to_arrow_type(type(obj)()), nullable=nullable, metadata=meta)

    raise ValueError(
        f"Cannot convert {type(obj)!r} to a pyarrow.Field.\n"
        f"  Accepted types: pl.DataFrame, pl.LazyFrame, pl.Series, pl.Expr, "
        f"pl.DataType (instance or class), pl.Field.\n"
        f"  Received: {obj!r:.120}"
    )