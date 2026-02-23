"""
Polars DataFrame extension helpers for joins and resampling.

Includes compatibility monkeypatches for older Polars builds:

- pl.Expr.dtype property (get/set) if missing
- pl.Field.nullable: bool property (get/set) if missing
- pl.Field.metadata: Optional[dict[bytes, bytes]] property (get/set) if missing

Also overrides pl.Expr.alias and pl.Expr.cast to preserve injected metadata.
"""

from __future__ import annotations

import datetime
import inspect
import weakref
from typing import Any, Literal, Mapping, Optional, Sequence

from .lib import polars as pl

__all__ = [
    "join_coalesced",
    "resample",
]

# ---------------------------------------------------------------------
# Monkeypatches: pl.Expr.dtype, pl.Field.nullable, pl.Field.metadata
# Prefer setattr/getattr; fallback to id-map if Polars objects disallow attrs.
# ---------------------------------------------------------------------

# Private attribute names stored on Polars objects (when possible)
_YGG_EXPR_DTYPE_ATTR = "_ygg_dtype"
_YGG_EXPR_META_ATTR = "_ygg_expr_meta"  # reserved (optional future use)

_YGG_FIELD_NULLABLE_ATTR = "_ygg_nullable"
_YGG_FIELD_METADATA_ATTR = "_ygg_metadata"

# Fallback stores (used only when setattr/getattr fails)
_EXPR_DTYPE_FALLBACK: dict[int, object] = {}
_FIELD_NULLABLE_FALLBACK: dict[int, bool] = {}
_FIELD_METADATA_FALLBACK: dict[int, Optional[dict[bytes, bytes]]] = {}


def _fallback_set(store: dict, obj: object, value: object) -> None:
    """Store by id(obj) with best-effort cleanup on GC."""
    k = id(obj)
    store[k] = value
    try:
        weakref.finalize(obj, store.pop, k, None)
    except Exception:
        # If finalize can't attach, accept possible stale entries.
        pass


def _try_getattr(obj: object, name: str):
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _try_setattr(obj: object, name: str, value: object) -> bool:
    """Return True if stored on the object; False if impossible."""
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        pass
    try:
        object.__setattr__(obj, name, value)
        return True
    except Exception:
        return False


# ---- pl.Expr.dtype ---------------------------------------------------

def _expr_dtype_get(self: "pl.Expr"):
    v = _try_getattr(self, _YGG_EXPR_DTYPE_ATTR)
    if v is not None:
        return v
    return _EXPR_DTYPE_FALLBACK.get(id(self))


def _expr_dtype_set(self: "pl.Expr", value) -> None:
    if not _try_setattr(self, _YGG_EXPR_DTYPE_ATTR, value):
        _fallback_set(_EXPR_DTYPE_FALLBACK, self, value)


def ensure_polars_expr_dtype_property() -> None:
    """Add pl.Expr.dtype property (getter+setter) ONLY if it doesn't exist."""
    if pl is None:
        return
    if hasattr(pl.Expr, "dtype"):
        return
    pl.Expr.dtype = property(_expr_dtype_get, _expr_dtype_set)  # type: ignore[attr-defined]


# ---- pl.Field.nullable -----------------------------------------------

def _field_nullable_get(self: "pl.Field") -> bool:
    v = _try_getattr(self, _YGG_FIELD_NULLABLE_ATTR)
    if v is not None:
        return bool(v)
    v2 = _FIELD_NULLABLE_FALLBACK.get(id(self))
    return True if v2 is None else bool(v2)  # default True


def _field_nullable_set(self: "pl.Field", value: bool) -> None:
    v = bool(value)
    if not _try_setattr(self, _YGG_FIELD_NULLABLE_ATTR, v):
        _fallback_set(_FIELD_NULLABLE_FALLBACK, self, v)


# ---- pl.Field.metadata -----------------------------------------------

def _field_metadata_get(self: "pl.Field") -> Optional[dict[bytes, bytes]]:
    v = _try_getattr(self, _YGG_FIELD_METADATA_ATTR)
    if v is not None:
        return v  # type: ignore[return-value]
    return _FIELD_METADATA_FALLBACK.get(id(self))


def _field_metadata_set(self: "pl.Field", value: Optional[dict[bytes, bytes]]) -> None:
    if value is not None:
        if not isinstance(value, dict):
            raise TypeError(f"Field.metadata must be dict[bytes, bytes] or None, got {type(value)}")
        out: dict[bytes, bytes] = {}
        for k, v in value.items():
            if not isinstance(k, (bytes, bytearray)) or not isinstance(v, (bytes, bytearray)):
                raise TypeError("Field.metadata must be dict[bytes, bytes] (bytes keys and bytes values)")
            out[bytes(k)] = bytes(v)
        value = out

    if not _try_setattr(self, _YGG_FIELD_METADATA_ATTR, value):
        _fallback_set(_FIELD_METADATA_FALLBACK, self, value)


def ensure_polars_field_nullable_metadata_properties() -> None:
    """Add pl.Field.nullable and pl.Field.metadata properties ONLY if missing."""
    if pl is None:
        return

    if not hasattr(pl.Field, "nullable"):
        pl.Field.nullable = property(_field_nullable_get, _field_nullable_set)  # type: ignore[attr-defined]

    if not hasattr(pl.Field, "metadata"):
        pl.Field.metadata = property(_field_metadata_get, _field_metadata_set)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------
# Preserve injected Expr metadata across alias/cast
# ---------------------------------------------------------------------

def _copy_expr_attrs(src: "pl.Expr", dst: "pl.Expr") -> "pl.Expr":
    """Best-effort copy of our injected Expr attrs from src -> dst."""
    for attr in (_YGG_EXPR_DTYPE_ATTR, _YGG_EXPR_META_ATTR):
        v = _try_getattr(src, attr)
        if v is None:
            continue
        _try_setattr(dst, attr, v) or _fallback_set(_EXPR_DTYPE_FALLBACK, dst, v) if attr == _YGG_EXPR_DTYPE_ATTR else None
    return dst


def patch_expr_alias_cast_preserve_metadata() -> None:
    if pl is None:
        return

    # ---- alias ----
    if hasattr(pl.Expr, "alias") and not getattr(pl.Expr, "_ygg_alias_patched", False):
        _orig_alias = pl.Expr.alias  # type: ignore[attr-defined]

        def _alias(self: "pl.Expr", *args, **kwargs):
            out = _orig_alias(self, *args, **kwargs)
            return _copy_expr_attrs(self, out)

        pl.Expr.alias = _alias  # type: ignore[assignment]
        _try_setattr(pl.Expr, "_ygg_alias_patched", True)

    # ---- cast ----
    if hasattr(pl.Expr, "cast") and not getattr(pl.Expr, "_ygg_cast_patched", False):
        _orig_cast = pl.Expr.cast  # type: ignore[attr-defined]

        def _cast(self: "pl.Expr", *args, **kwargs):
            out = _orig_cast(self, *args, **kwargs)

            # Preserve existing metadata first
            _copy_expr_attrs(self, out)

            # Update dtype metadata to match cast target when provided
            target_dtype = args[0] if args else kwargs.get("dtype")
            if target_dtype is not None:
                if not _try_setattr(out, _YGG_EXPR_DTYPE_ATTR, target_dtype):
                    _fallback_set(_EXPR_DTYPE_FALLBACK, out, target_dtype)

            return out

        pl.Expr.cast = _cast  # type: ignore[assignment]
        _try_setattr(pl.Expr, "_ygg_cast_patched", True)


# Patch on import
ensure_polars_expr_dtype_property()
ensure_polars_field_nullable_metadata_properties()
patch_expr_alias_cast_preserve_metadata()


# ---------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------

AggSpec = Mapping[str, Any] | Sequence["pl.Expr"]


def join_coalesced(
    left: "pl.DataFrame",
    right: "pl.DataFrame",
    on: str | list[str],
    how: str = "left",
    suffix: str = "_right",
) -> "pl.DataFrame":
    """
    Join two DataFrames and merge overlapping columns:
    - prefer values from `left`
    - fallback to `right` where left is null
    """
    on_cols = {on} if isinstance(on, str) else set(on)
    common = (set(left.columns) & set(right.columns)) - on_cols

    joined = left.join(right, on=list(on_cols), how=how, suffix=suffix)

    joined = joined.with_columns(
        [pl.coalesce(pl.col(c), pl.col(f"{c}{suffix}")).alias(c) for c in common]
    ).drop([f"{c}{suffix}" for c in common])

    return joined


def _normalize_group_by(group_by: str | Sequence[str] | None) -> list[str] | None:
    if group_by is None:
        return None
    if isinstance(group_by, str):
        return [group_by]
    return list(group_by)


def _filter_kwargs_for_callable(fn: object, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Polars APIs vary across versions (e.g. upsample(offset=), partition_by(maintain_order=)).
    Only pass kwargs supported by the installed signature; also drop None values.
    """
    sig = inspect.signature(fn)  # type: ignore[arg-type]
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if (k in allowed and v is not None)}


def _expr_from_agg(col: str, agg: Any) -> "pl.Expr":
    base = pl.col(col)

    if isinstance(agg, pl.Expr):
        return agg.alias(col)

    if callable(agg):
        out = agg(base)
        if not isinstance(out, pl.Expr):
            raise TypeError(f"Callable agg for '{col}' must return a polars.Expr, got {type(out)}")
        return out.alias(col)

    if isinstance(agg, str):
        name = agg.lower()
        if not hasattr(base, name):
            raise ValueError(f"Unknown agg '{agg}' for column '{col}'.")
        return getattr(base, name)().alias(col)

    raise TypeError(
        f"Invalid agg for '{col}': {agg!r}. Use pl.Expr, callable, or string name like 'sum'/'mean'/'last'."
    )


def _normalize_aggs(agg: AggSpec) -> list["pl.Expr"]:
    if isinstance(agg, Mapping):
        return [_expr_from_agg(col, spec) for col, spec in agg.items()]

    out = list(agg)
    if not all(isinstance(e, pl.Expr) for e in out):
        bad = [type(e) for e in out if not isinstance(e, pl.Expr)]
        raise TypeError(f"agg sequence must be polars.Expr; got non-Expr types: {bad}")
    return out


def _is_datetime(dtype: object) -> bool:
    # Datetime-only inference (ignore Date).
    return isinstance(dtype, pl.Datetime)


def _infer_time_col(df: "pl.DataFrame") -> str:
    for name, dtype in df.schema.items():
        if _is_datetime(dtype):
            return name
    raise ValueError("resample: time_col not provided and no Datetime column found in DataFrame schema.")


def _ensure_datetime_like(df: "pl.DataFrame", time_col: str) -> "pl.DataFrame":
    dtype = df.schema.get(time_col)
    if dtype is None:
        raise KeyError(f"resample: time_col '{time_col}' not found in DataFrame columns.")

    # Date is allowed but cast up so sub-day resampling works.
    if isinstance(dtype, pl.Date):
        return df.with_columns(pl.col(time_col).cast(pl.Datetime))

    if isinstance(dtype, pl.Datetime):
        return df

    # Convenience: attempt cast for non-temporal column.
    return df.with_columns(pl.col(time_col).cast(pl.Datetime))


def _timedelta_to_polars_duration(td: datetime.timedelta) -> str:
    """Convert python timedelta -> polars duration string (w/d/h/m/s/ms/us)."""
    if td < datetime.timedelta(0):
        raise ValueError(f"Negative timedelta not supported: {td!r}")

    total_us = int(td.total_seconds() * 1_000_000)

    units = [
        (7 * 24 * 3600 * 1_000_000, "w"),
        (24 * 3600 * 1_000_000, "d"),
        (3600 * 1_000_000, "h"),
        (60 * 1_000_000, "m"),
        (1_000_000, "s"),
        (1_000, "ms"),
        (1, "us"),
    ]

    for factor, suffix in units:
        if total_us % factor == 0:
            return f"{total_us // factor}{suffix}"

    return f"{total_us}us"


def _normalize_duration(v: str | datetime.timedelta | None) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, datetime.timedelta):
        return _timedelta_to_polars_duration(v)
    raise TypeError(f"Expected str|timedelta|None for duration, got {type(v)}")


def _upsample_single(
    df: "pl.DataFrame",
    *,
    time_col: str,
    every: str | datetime.timedelta,
    offset: str | datetime.timedelta | None,
    keep_group_order: bool,
) -> "pl.DataFrame":
    df = df.sort(time_col)

    every_n = _normalize_duration(every)
    offset_n = _normalize_duration(offset)

    upsample_kwargs = _filter_kwargs_for_callable(
        pl.DataFrame.upsample,
        {
            "time_column": time_col,
            "every": every_n,
            "offset": offset_n,  # may not exist on older polars; filtered safely
            "maintain_order": keep_group_order,
        },
    )
    return df.upsample(**upsample_kwargs)


def resample(
    df: "pl.DataFrame",
    *,
    time_col: str | None = None,
    every: str | datetime.timedelta,
    group_by: str | Sequence[str] | None = None,
    agg: AggSpec | None = None,
    period: str | datetime.timedelta | None = None,
    offset: str | datetime.timedelta | None = None,
    closed: Literal["left", "right", "both", "none"] = "left",
    label: Literal["left", "right"] = "left",
    start_by: Literal[
        "window",
        "datapoint",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ] = "window",
    fill: Literal["forward", "backward", "zero", "none"] = "none",
    keep_group_order: bool = True,
) -> "pl.DataFrame":
    """
    Pandas-ish resample for Polars.

    - agg is None  -> upsample (insert missing timestamps)
    - agg provided -> group_by_dynamic + aggregations
    """
    gb = _normalize_group_by(group_by)

    if time_col is None:
        time_col = _infer_time_col(df)

    if agg is None and fill not in ("forward", "backward", "zero", "none"):
        raise ValueError(f"Unsupported fill={fill!r}")

    df = _ensure_datetime_like(df, time_col)

    every_n = _normalize_duration(every)
    period_n = _normalize_duration(period)
    offset_n = _normalize_duration(offset)

    # -------------------------
    # UPSAMPLE
    # -------------------------
    if agg is None:
        if gb is None:
            out = _upsample_single(
                df,
                time_col=time_col,
                every=every_n,
                offset=offset_n,
                keep_group_order=keep_group_order,
            ).sort(time_col)
        else:
            part_kwargs = _filter_kwargs_for_callable(
                pl.DataFrame.partition_by,
                {
                    "by": gb,
                    "as_dict": True,
                    "maintain_order": keep_group_order,
                },
            )
            parts = df.partition_by(**part_kwargs)  # type: ignore[arg-type]

            out_parts: list["pl.DataFrame"] = []
            for key, gdf in parts.items():  # type: ignore[union-attr]
                key_vals = key if isinstance(key, tuple) else (key,)

                gdf_up = _upsample_single(
                    gdf,
                    time_col=time_col,
                    every=every_n,
                    offset=offset_n,
                    keep_group_order=keep_group_order,
                )

                # Drop possibly-null group cols produced by upsample, then re-stamp constants.
                drop_cols = [c for c in gb if c in gdf_up.columns]
                if drop_cols:
                    gdf_up = gdf_up.drop(drop_cols)

                gdf_up = gdf_up.with_columns(
                    [pl.lit(v).alias(col) for col, v in zip(gb, key_vals)]
                )

                out_parts.append(gdf_up)

            out = pl.concat(out_parts, how="vertical").sort([*gb, time_col])

        if fill != "none":
            non_fill_cols = {time_col, *(gb or [])}
            fill_cols = [c for c in out.columns if c not in non_fill_cols]

            if fill in ("forward", "backward"):
                out = out.with_columns([pl.col(c).fill_null(strategy=fill) for c in fill_cols])
            elif fill == "zero":
                out = out.with_columns([pl.col(c).fill_null(0) for c in fill_cols])

        return out

    # -------------------------
    # DOWNSAMPLE
    # -------------------------
    aggs = _normalize_aggs(agg)

    gbd_kwargs = _filter_kwargs_for_callable(
        pl.DataFrame.group_by_dynamic,
        {
            "index_column": time_col,
            "every": every_n,
            "period": period_n,
            "offset": offset_n,
            "closed": closed,
            "label": label,
            "start_by": start_by,
            "group_by": gb,
        },
    )

    return df.group_by_dynamic(**gbd_kwargs).agg(aggs).sort([*(gb or []), time_col])


# Attach methods to Polars DataFrame at import time
if pl is not None:
    setattr(pl.DataFrame, "join_coalesced", join_coalesced)
    setattr(pl.DataFrame, "resample", resample)