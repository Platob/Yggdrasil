"""Curated / ``dash_*`` view standardisation helpers.

Curated and business-display tables in this codebase follow the
"source value + standardised equivalent" shape (CLAUDE.md
*Business-display layer (``dash_*``)*): the row carries the raw
value the source emitted (``value`` in ``MWh``, ``°F``, ``EUR``),
the unit it was emitted in (``value_unit``), **and** one or more
equivalent columns in canonical / project-standard units
(``value_kwh``, ``value_c``, ``value_usd``, ``value_chf``) so
downstream BI / ML can aggregate across vendors without re-parsing
strings at the call site.

This module is the matching helper surface. Two parallel
implementations — Polars (used inside scheduled-Job entrypoints
where the worker speaks Polars directly) and Spark (used inside
notebooks / Spark-SQL pipelines / the curated DAGs that already
run on the Spark warehouse). Both share the same
``with_unit_equivalent(s)`` / ``with_currency_equivalents``
signatures so a caller can switch engines with a one-line import
change.

Public surface:

* :func:`with_unit_equivalent` /
  :func:`with_currency_equivalents` — Polars DataFrame /
  LazyFrame helpers that append the standardised columns.
* :func:`spark_with_unit_equivalent` /
  :func:`spark_with_currency_equivalents` — Spark DataFrame
  equivalents (lazy pyspark import).
* :func:`polars_convert_unit` / :func:`spark_convert_unit` —
  raw expression builders for callers that want to splice the
  conversion into a larger ``select`` / ``with_columns`` chain
  without going through the DataFrame helper.
* :func:`dash_dual_value_fields` — schema-field builder that
  emits the canonical ``(name, name_unit, name_<target>)`` /
  ``(name, name_currency, name_eur, name_usd, name_chf)``
  triplet/quintuplet a curated/dash table should ship with.

The unit-conversion math comes from :mod:`yggdrasil.enums.units`;
currency conversion from :class:`yggdrasil.fxrate.FxRate`. The
schema-field builder routes through :func:`yggdrasil.data.field` so
the resulting columns flow through the same DDL / Spark / Arrow
path as every other curated column.
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Union

from yggdrasil.enums.currency import Currency
from yggdrasil.enums.units import Unit, unit_family_for

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl
    import pyspark.sql as ss
    from pyspark.sql import Column as SparkColumn
    from yggdrasil.data import Field
    from yggdrasil.fxrate import FxRate


__all__ = [
    "polars_convert_unit",
    "with_unit_equivalent",
    "with_currency_equivalents",
    "spark_convert_unit",
    "spark_with_unit_equivalent",
    "spark_with_currency_equivalents",
    "dash_dual_value_fields",
    "standardized_column_name",
]


LOGGER = logging.getLogger(__name__)


# Default currency targets for the "EUR / USD / CHF equivalent"
# pattern. Aligned with the FX dash-view convention (CLAUDE.md
# "Business-display layer"); callers can override per call.
DEFAULT_CURRENCY_TARGETS: tuple[Currency, ...] = (
    Currency.EUR, Currency.USD, Currency.CHF,
)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def standardized_column_name(value_col: str, target: Union[Unit, Currency, str]) -> str:
    """Canonical column name for the standardised equivalent of *value_col*.

    Returns ``f"{value_col}_{token}"`` with *token* downcased and
    sanitised for SQL identifier comfort: ``°C`` → ``"c"``, ``m³`` →
    ``"m3"``, ``gal_US`` → ``"gal_us"``, ``EUR`` → ``"eur"``,
    ``kWh`` → ``"kwh"``.
    """
    if isinstance(target, Unit):
        token = target.symbol
    elif isinstance(target, Currency):
        token = target.code
    else:
        token = str(target)
    # Drop the degree glyph + lowercase + replace cube glyph. Keep
    # underscores in-place (``gal_US`` → ``gal_us``).
    sanitised = (
        token.replace("°", "")
             .replace("³", "3")
             .replace("²", "2")
             .lower()
    )
    return f"{value_col}_{sanitised}"


# ---------------------------------------------------------------------------
# Unit-family resolution (shared)
# ---------------------------------------------------------------------------


def _resolve_family(
    target: Union[Unit, str],
    family: Optional[type[Unit]],
) -> tuple[type[Unit], Unit]:
    """Resolve a ``(family, target_member)`` pair from caller input.

    Accepts an explicit *family* (one of :data:`UNIT_FAMILIES`) plus a
    *target* token that the family knows, OR derives the family from
    *target* via :func:`unit_family_for` when *family* is ``None``.
    Raises with a hint when *target* doesn't land in *family*.
    """
    if family is None:
        if isinstance(target, Unit):
            family = type(target)
        else:
            family = unit_family_for(target)
    if isinstance(target, Unit) and not isinstance(target, family):
        raise ValueError(
            f"Unit target {target!r} belongs to {type(target).__name__}, "
            f"not the requested family {family.__name__}. Pass family="
            f"{type(target).__name__} or a target from {family.__name__}."
        )
    target_unit = family.from_(target)
    return family, target_unit


# ===========================================================================
# Polars surface
# ===========================================================================


def polars_convert_unit(
    value: Union["pl.Expr", str],
    source: Union[Unit, str, "pl.Expr"],
    target: Union[Unit, str],
    *,
    family: Optional[type[Unit]] = None,
) -> "pl.Expr":
    """Polars expression converting *value* from *source* unit to *target*.

    *value* is either a column name (resolved as :func:`polars.col`) or
    a pre-built :class:`polars.Expr`. *source* may be:

    * a unit member / symbol / alias (static conversion — single
      multiply+offset);
    * a :class:`polars.Expr` or column name (per-row conversion — the
      source unit varies by row; falls back to ``replace_strict``
      against the family's symbol→factor / symbol→offset maps).

    *target* is always a static unit member or symbol. *family* may be
    omitted when *target* (or *source*, when static) is unambiguous;
    pass it explicitly when the target token is shared across families.
    """
    from yggdrasil.lazy_imports import polars as pl

    family, target_unit = _resolve_family(target, family)
    value_expr: pl.Expr = pl.col(value) if isinstance(value, str) else value

    if isinstance(source, (Unit, str)):
        try:
            source_unit = family.from_(source)
        except (TypeError, ValueError):
            source_unit = None
        if source_unit is not None:
            # Static path: collapses to a constant-folded affine expression.
            if source_unit is target_unit:
                return value_expr
            canonical = value_expr * source_unit.factor + source_unit.offset
            return (canonical - target_unit.offset) / target_unit.factor
        # Fall through — treat ``source`` as a column reference name.
        source_expr: pl.Expr = pl.col(source)  # type: ignore[arg-type]
    elif isinstance(source, pl.Expr):
        source_expr = source
    else:
        raise TypeError(
            f"polars_convert_unit: source must be a Unit, symbol str, "
            f"polars.Expr, or column name; got {type(source).__name__}: {source!r}"
        )

    # Per-row path: look up factor + offset by the source column's
    # symbol via ``replace_strict``. Unknown tokens map to null (the
    # resulting equivalent column is null on that row) rather than
    # raising — curated views typically prefer "carry the unknown
    # through" over "kill the whole batch".
    source_str = source_expr.cast(pl.Utf8)
    factor_expr = source_str.replace_strict(
        family.factor_map(), default=None, return_dtype=pl.Float64,
    )
    offset_expr = source_str.replace_strict(
        family.offset_map(), default=None, return_dtype=pl.Float64,
    )
    canonical = value_expr.cast(pl.Float64) * factor_expr + offset_expr
    return (canonical - target_unit.offset) / target_unit.factor


def with_unit_equivalent(
    df: "pl.DataFrame | pl.LazyFrame",
    *,
    value_col: str,
    unit_col: Union[str, Unit],
    target: Union[Unit, str],
    family: Optional[type[Unit]] = None,
    suffix: Optional[str] = None,
) -> "pl.DataFrame | pl.LazyFrame":
    """Append a standardised-unit equivalent column to *df*.

    Args:
        df: Polars DataFrame or LazyFrame; the same type is returned.
        value_col: Name of the numeric column carrying the raw value.
        unit_col: Either the name of a string column carrying the
            source unit symbol per row (per-row conversion), or a
            :class:`Unit` member / symbol applied uniformly to every
            row (static conversion).
        target: Target unit — member or symbol from the same family.
        family: Optional explicit family; defaults to the one *target*
            (and *unit_col* when static) belongs to.
        suffix: Override the appended column name. Defaults to
            :func:`standardized_column_name` (``f"{value_col}_<target>"``).

    Returns:
        A frame with one extra ``f"{value_col}_<target>"`` column (or
        the explicit *suffix*) in Float64 dtype. Rows whose
        ``unit_col`` doesn't match any known symbol get ``null`` in
        the equivalent column — same fall-through shape every other
        normalisation helper in this codebase uses.
    """
    family, target_unit = _resolve_family(target, family)

    if isinstance(unit_col, (Unit, str)) and not _is_likely_column_name(df, unit_col, family):
        source: Any = unit_col
    else:
        source = unit_col  # column name path

    out_name = suffix if suffix is not None else standardized_column_name(
        value_col, target_unit,
    )
    expr = polars_convert_unit(value_col, source, target_unit, family=family)
    return df.with_columns(expr.alias(out_name))


def _polars_columns(df: "pl.DataFrame | pl.LazyFrame") -> list[str]:
    """Return *df*'s column names, using the lazy-safe path for LazyFrame."""
    from yggdrasil.lazy_imports import polars as pl

    if isinstance(df, pl.LazyFrame):
        return df.collect_schema().names()
    return list(df.columns)


def _is_likely_column_name(
    df: "pl.DataFrame | pl.LazyFrame",
    value: Union[str, Unit],
    family: type[Unit],
) -> bool:
    """``True`` when *value* should be treated as a column reference.

    Ambiguous case: a caller passes ``unit_col="MWh"`` — is that a
    static unit ("every row is in MWh") or the name of a column that
    happens to be called ``"MWh"``? Resolution: if *df* actually has
    a column named *value*, treat it as a column; otherwise treat it
    as a static unit symbol. :class:`Unit` instances are always static.
    """
    if isinstance(value, Unit):
        return False
    # Strings that ``family.is_valid`` already accepts AND that aren't a
    # column on the frame are static units; strings that match a column
    # name win (column-name interpretation, per-row conversion).
    columns = _polars_columns(df)
    return value in columns


def _resolve_currency_col(
    currency_col: Union[str, Currency],
    columns: list[str],
    get_distinct: Callable[[], tuple[Currency, ...]],
    fn_name: str,
) -> tuple[tuple[Currency, ...], Optional[Currency]]:
    """Return ``(sources, static_source)`` for a currency_col argument.

    ``static_source`` is ``None`` when *currency_col* names a per-row column;
    otherwise it is the resolved :class:`Currency` that applies to every row.
    """
    if isinstance(currency_col, Currency):
        return (currency_col,), currency_col
    if isinstance(currency_col, str) and currency_col in columns:
        return get_distinct(), None
    if isinstance(currency_col, str):
        static = Currency.from_(currency_col)
        return (static,), static
    raise TypeError(
        f"{fn_name}: currency_col must be a Currency, ISO code str, or column name; "
        f"got {type(currency_col).__name__}: {currency_col!r}"
    )


def with_currency_equivalents(
    df: "pl.DataFrame | pl.LazyFrame",
    *,
    value_col: str,
    currency_col: Union[str, Currency],
    targets: Sequence[Union[Currency, str]] = DEFAULT_CURRENCY_TARGETS,
    fx: Optional["FxRate"] = None,
    at: Optional[Union[_dt.datetime, _dt.date, str]] = None,
    at_col: Optional[str] = None,
) -> "pl.DataFrame | pl.LazyFrame":
    """Append EUR / USD / CHF equivalents (or any *targets*) to *df*.

    Args:
        df: Polars DataFrame or LazyFrame.
        value_col: Numeric column carrying the source-currency amount.
        currency_col: Either the name of a string column carrying the
            source-currency ISO code per row (per-row conversion), or
            a :class:`Currency` member / ISO code applied uniformly.
        targets: Iterable of target currencies. Defaults to
            ``(EUR, USD, CHF)`` per the CLAUDE.md FX dash convention.
        fx: Pre-built :class:`FxRate` session — useful in tests with
            a stub backend. Defaults to a process-wide
            :class:`FxRate` singleton.
        at: Single point in time at which to look up rates. Defaults
            to "now". Mutually exclusive with *at_col*.
        at_col: Name of a per-row date/timestamp column (the rate for
            each row is taken from the FX history at that point —
            useful for back-filling a historical curated table).
            Mutually exclusive with *at*.

    Returns:
        A frame with one extra ``f"{value_col}_<target>"`` column per
        entry in *targets* (Float64). Rows whose source currency
        doesn't appear in the FX response, or whose ``at_col``
        timestamp doesn't land on a published rate, get ``null`` in
        the matching equivalent column.

    The implementation collects every distinct ``(source, target)``
    pair in *df*, fires a single :meth:`FxRate.latest` (or
    :meth:`FxRate.fetch`) call to materialise the rates, joins them
    back onto *df*, and emits one equivalent column per target. This
    is one round-trip per ``deploy_scheduled_fxrate_job`` cron tick,
    not one per row.
    """
    if at is not None and at_col is not None:
        raise ValueError(
            "with_currency_equivalents: pass at OR at_col, not both. "
            "Use at=<datetime> for a single snapshot; at_col=<col_name> "
            "for historical per-row lookup."
        )

    from yggdrasil.lazy_imports import polars as pl
    from yggdrasil.fxrate import FxRate

    if fx is None:
        fx = FxRate()

    target_currencies = tuple(Currency.from_(t) for t in targets)
    if not target_currencies:
        raise ValueError(
            "with_currency_equivalents(targets=()): pass at least one target "
            "currency (default targets are EUR/USD/CHF)."
        )

    is_lazy = isinstance(df, pl.LazyFrame)
    eager = df.collect() if is_lazy else df

    def _get_distinct_polars() -> tuple[Currency, ...]:
        seen = eager[currency_col].cast(pl.Utf8).drop_nulls().unique().to_list()  # type: ignore[union-attr]
        return tuple(Currency.from_(s) for s in seen if s)

    sources, static_source = _resolve_currency_col(
        currency_col, list(eager.columns), _get_distinct_polars, "with_currency_equivalents"
    )

    pairs: list[tuple[Currency, Currency]] = [
        (src, tgt)
        for src in sources
        for tgt in target_currencies
        if src != tgt
    ]
    rate_lookup = _fetch_rate_lookup(
        fx, pairs=pairs, at=at, at_col=at_col, eager=eager,
    )

    out = eager
    for tgt in target_currencies:
        out_name = standardized_column_name(value_col, tgt)
        if static_source is not None:
            # Static source: one literal rate per target, or 1.0 when same.
            if static_source == tgt:
                rate = 1.0
            else:
                rate = rate_lookup.get((static_source.code, tgt.code, None))
                if rate is None:
                    rate = float("nan")
            out = out.with_columns(
                (pl.col(value_col).cast(pl.Float64) * pl.lit(rate)).alias(out_name)
            )
        else:
            # Per-row source: build a (source, [at_date]) → rate Polars
            # mapping and join. ``at_col``-driven historical lookup uses
            # the per-row date; otherwise the latest snapshot rate.
            rate_col = _build_polars_rate_expr(
                rate_lookup, target=tgt, currency_col=currency_col, at_col=at_col,
            )
            out = out.with_columns(
                (pl.col(value_col).cast(pl.Float64) * rate_col).alias(out_name)
            )
    return out.lazy() if is_lazy else out


def _fetch_rate_lookup(
    fx: "FxRate",
    *,
    pairs: Sequence[tuple[Currency, Currency]],
    at: Optional[Any],
    at_col: Optional[str],
    eager: "pl.DataFrame",
) -> dict[tuple[str, str, Optional[_dt.date]], float]:
    """Pull FX rates from *fx* and return a ``(source, target, date|None) → rate`` map.

    Snapshot mode (*at* set or both None) uses :meth:`FxRate.latest`;
    historical mode (*at_col* set) uses :meth:`FxRate.fetch` over the
    window spanned by the column and groups rates by date.
    """
    from yggdrasil.lazy_imports import polars as pl

    if not pairs:
        return {}

    if at_col is not None:
        if at_col not in eager.columns:
            raise ValueError(
                f"with_currency_equivalents(at_col={at_col!r}): column not "
                f"found in DataFrame. Available: {eager.columns!r}."
            )
        dates_series = eager[at_col].cast(pl.Datetime("us", time_zone="UTC"), strict=False)
        dates = dates_series.drop_nulls().to_list()
        if not dates:
            return {}
        start = min(dates)
        end = max(dates)
        df = fx.fetch(
            pairs=[(s.code, t.code) for s, t in pairs],
            start=start, end=end, sampling="1d",
        )
        out: dict[tuple[str, str, Optional[_dt.date]], float] = {}
        for row in df.iter_rows(named=True):
            ts = row["from_timestamp"]
            out[(row["source"], row["target"], ts.date())] = float(row["value"])
        return out

    df = fx.latest(pairs=[(s.code, t.code) for s, t in pairs])
    return {
        (row["source"], row["target"], None): float(row["value"])
        for row in df.iter_rows(named=True)
    }


def _build_polars_rate_expr(
    rate_lookup: Mapping[tuple[str, str, Optional[_dt.date]], float],
    *,
    target: Currency,
    currency_col: str,
    at_col: Optional[str],
) -> "pl.Expr":
    """Polars expression producing the per-row rate to *target*."""
    from yggdrasil.lazy_imports import polars as pl

    target_code = target.code
    if at_col is None:
        # Snapshot: build a simple {source_code: rate} replace_strict.
        snapshot: dict[str, float] = {
            src: rate
            for (src, tgt, _date), rate in rate_lookup.items()
            if tgt == target_code
        }
        snapshot.setdefault(target_code, 1.0)
        return (
            pl.col(currency_col).cast(pl.Utf8)
              .replace_strict(snapshot, default=None, return_dtype=pl.Float64)
        )

    # Historical per-row date lookup: build a {(src, date_str): rate}
    # composite key encoded as a single Polars string for replace_strict.
    historical: dict[str, float] = {}
    for (src, tgt, date), rate in rate_lookup.items():
        if tgt != target_code or date is None:
            continue
        historical[f"{src}|{date.isoformat()}"] = rate
    # When source == target on the same day, rate is identically 1.0 —
    # don't bother enumerating every date, instead branch in the expression.
    key_expr = (
        pl.col(currency_col).cast(pl.Utf8)
        + pl.lit("|")
        + pl.col(at_col).cast(pl.Date).cast(pl.Utf8)
    )
    lookup_expr = key_expr.replace_strict(historical, default=None, return_dtype=pl.Float64)
    return (
        pl.when(pl.col(currency_col).cast(pl.Utf8) == pl.lit(target_code))
          .then(pl.lit(1.0))
          .otherwise(lookup_expr)
    )


# ===========================================================================
# Spark surface
# ===========================================================================


def spark_convert_unit(
    value: Union["SparkColumn", str],
    source: Union[Unit, str, "SparkColumn"],
    target: Union[Unit, str],
    *,
    family: Optional[type[Unit]] = None,
) -> "SparkColumn":
    """Spark Column expression converting *value* from *source* unit to *target*.

    Mirrors :func:`polars_convert_unit`. *value* is either a column
    name or a :class:`pyspark.sql.Column`. *source* may be a static
    unit or a column reference (per-row conversion).
    """
    from pyspark.sql import Column as SparkColumn
    from pyspark.sql import functions as F

    family, target_unit = _resolve_family(target, family)
    value_col: SparkColumn = F.col(value) if isinstance(value, str) else value

    if isinstance(source, (Unit, str)):
        try:
            source_unit = family.from_(source)
        except (TypeError, ValueError):
            source_unit = None
        if source_unit is not None:
            if source_unit is target_unit:
                return value_col
            canonical = value_col * F.lit(source_unit.factor) + F.lit(source_unit.offset)
            return (canonical - F.lit(target_unit.offset)) / F.lit(target_unit.factor)
        source_expr: SparkColumn = F.col(source)  # type: ignore[arg-type]
    elif isinstance(source, SparkColumn):
        source_expr = source
    else:
        raise TypeError(
            f"spark_convert_unit: source must be a Unit, symbol str, "
            f"pyspark.sql.Column, or column name; got {type(source).__name__}: "
            f"{source!r}"
        )

    # Per-row path: build literal maps for factor + offset, lookup
    # with ``element_at``. Unknown tokens → null (same fall-through
    # as the Polars path).
    factor_map_expr = _spark_literal_map(family.factor_map(), F)
    offset_map_expr = _spark_literal_map(family.offset_map(), F)
    factor = F.element_at(factor_map_expr, source_expr.cast("string"))
    offset = F.element_at(offset_map_expr, source_expr.cast("string"))
    canonical = value_col.cast("double") * factor + offset
    return (canonical - F.lit(target_unit.offset)) / F.lit(target_unit.factor)


def _spark_literal_map(mapping: Mapping[str, float], F: Any) -> "SparkColumn":
    """Build a Spark ``create_map`` Column from a Python dict."""
    pairs: list[Any] = []
    for k, v in mapping.items():
        pairs.append(F.lit(k))
        pairs.append(F.lit(v))
    return F.create_map(*pairs)


def spark_with_unit_equivalent(
    df: "ss.DataFrame",
    *,
    value_col: str,
    unit_col: Union[str, Unit],
    target: Union[Unit, str],
    family: Optional[type[Unit]] = None,
    suffix: Optional[str] = None,
) -> "ss.DataFrame":
    """Spark equivalent of :func:`with_unit_equivalent`."""
    family, target_unit = _resolve_family(target, family)
    if isinstance(unit_col, Unit) or (
        isinstance(unit_col, str) and unit_col not in df.columns
    ):
        source: Any = unit_col
    else:
        source = unit_col
    out_name = suffix if suffix is not None else standardized_column_name(
        value_col, target_unit,
    )
    expr = spark_convert_unit(value_col, source, target_unit, family=family)
    return df.withColumn(out_name, expr)


def spark_with_currency_equivalents(
    df: "ss.DataFrame",
    *,
    value_col: str,
    currency_col: Union[str, Currency],
    targets: Sequence[Union[Currency, str]] = DEFAULT_CURRENCY_TARGETS,
    fx: Optional["FxRate"] = None,
    at: Optional[Union[_dt.datetime, _dt.date, str]] = None,
    at_col: Optional[str] = None,
) -> "ss.DataFrame":
    """Spark equivalent of :func:`with_currency_equivalents`.

    Same shape: collects distinct ``(source, target)`` pairs, fires
    one :meth:`FxRate.latest` or :meth:`FxRate.fetch` call, joins
    rates onto *df*. Snapshot mode emits a static literal map per
    target; historical mode (``at_col`` set) joins against a
    spark-side rates DataFrame keyed by ``(source, target, date)``.
    """
    if at is not None and at_col is not None:
        raise ValueError(
            "spark_with_currency_equivalents: pass at OR at_col, not both."
        )

    from pyspark.sql import functions as F
    from yggdrasil.fxrate import FxRate

    if fx is None:
        fx = FxRate()

    target_currencies = tuple(Currency.from_(t) for t in targets)
    if not target_currencies:
        raise ValueError(
            "spark_with_currency_equivalents(targets=()): pass at least one "
            "target currency."
        )

    sources, static_source = _resolve_currency_col(
        currency_col,
        list(df.columns),
        lambda: tuple(_distinct_currencies_spark(df, currency_col)),  # type: ignore[arg-type]
        "spark_with_currency_equivalents",
    )

    pairs: list[tuple[Currency, Currency]] = [
        (src, tgt)
        for src in sources
        for tgt in target_currencies
        if src != tgt
    ]
    rate_lookup = _fetch_rate_lookup_spark(
        fx, pairs=pairs, at_col=at_col, df=df,
    )

    out = df
    for tgt in target_currencies:
        out_name = standardized_column_name(value_col, tgt)
        if static_source is not None:
            rate = 1.0 if static_source == tgt else rate_lookup.get(
                (static_source.code, tgt.code, None), float("nan"),
            )
            out = out.withColumn(out_name, F.col(value_col).cast("double") * F.lit(rate))
        elif at_col is None:
            snapshot = {
                src: rate
                for (src, t, _d), rate in rate_lookup.items()
                if t == tgt.code
            }
            snapshot.setdefault(tgt.code, 1.0)
            map_expr = _spark_literal_map(snapshot, F)
            rate_col = F.element_at(map_expr, F.col(currency_col).cast("string"))
            out = out.withColumn(
                out_name, F.col(value_col).cast("double") * rate_col,
            )
        else:
            historical = {
                f"{src}|{date.isoformat()}": rate
                for (src, t, date), rate in rate_lookup.items()
                if t == tgt.code and date is not None
            }
            map_expr = _spark_literal_map(historical, F)
            key = F.concat(
                F.col(currency_col).cast("string"),
                F.lit("|"),
                F.col(at_col).cast("date").cast("string"),
            )
            same_target = F.col(currency_col).cast("string") == F.lit(tgt.code)
            rate_col = F.when(same_target, F.lit(1.0)).otherwise(
                F.element_at(map_expr, key)
            )
            out = out.withColumn(
                out_name, F.col(value_col).cast("double") * rate_col,
            )
    return out


def _distinct_currencies_spark(df: "ss.DataFrame", currency_col: str) -> list[Currency]:
    rows = df.select(currency_col).distinct().collect()
    out: list[Currency] = []
    for row in rows:
        token = row[0]
        if token is None:
            continue
        try:
            out.append(Currency.from_(token))
        except (TypeError, ValueError):
            LOGGER.warning(
                "Skipping unparseable currency token %r from column %r",
                token, currency_col,
            )
    return out


def _fetch_rate_lookup_spark(
    fx: "FxRate",
    *,
    pairs: Sequence[tuple[Currency, Currency]],
    at_col: Optional[str],
    df: "ss.DataFrame",
) -> dict[tuple[str, str, Optional[_dt.date]], float]:
    if not pairs:
        return {}
    if at_col is None:
        result = fx.latest(pairs=[(s.code, t.code) for s, t in pairs])
        return {
            (row["source"], row["target"], None): float(row["value"])
            for row in result.iter_rows(named=True)
        }
    from pyspark.sql import functions as F
    bounds = df.agg(
        F.min(F.col(at_col).cast("timestamp")).alias("lo"),
        F.max(F.col(at_col).cast("timestamp")).alias("hi"),
    ).first()
    if bounds is None or bounds["lo"] is None or bounds["hi"] is None:
        return {}
    result = fx.fetch(
        pairs=[(s.code, t.code) for s, t in pairs],
        start=bounds["lo"], end=bounds["hi"], sampling="1d",
    )
    return {
        (row["source"], row["target"], row["from_timestamp"].date()): float(row["value"])
        for row in result.iter_rows(named=True)
    }


# ===========================================================================
# Schema-field builder
# ===========================================================================


def dash_dual_value_fields(
    name: str,
    *,
    target: Union[Unit, Currency, str, Sequence[Union[Currency, str]], None] = None,
    family: Optional[type[Unit]] = None,
    currency: bool = False,
    source_unit_nullable: bool = False,
    nullable: bool = False,
    dtype: str = "decimal(18, 6)",
    equivalent_dtype: Optional[str] = None,
) -> list["Field"]:
    """Build the canonical ``(value, unit, value_<target>…)`` field set.

    For a curated / dash row that needs to carry both the source
    value (whatever unit the upstream feed emits) and a standardised
    equivalent in a canonical / project-wide unit:

    .. code-block:: python

        from yggdrasil.data import Schema
        from yggdrasil.databricks.standardize import dash_dual_value_fields
        from yggdrasil.enums.units import EnergyUnit

        Schema.from_fields([
            *dash_dual_value_fields("price", target=Currency.EUR, currency=True),
            *dash_dual_value_fields("volume", target=EnergyUnit.KWH),
        ])
        # → price, price_currency, price_eur, price_usd, price_chf,
        #   volume, volume_unit, volume_kwh

    Args:
        name: Base column name (``"price"`` / ``"volume"`` / ``"capacity"``).
        target: For unit families, a single target unit (member or
            symbol). For currencies (``currency=True``), either a single
            :class:`Currency` (one equivalent column) or a sequence of
            them (one equivalent column per entry — defaults to
            ``(EUR, USD, CHF)`` when ``True`` is passed).
        family: Explicit unit family — required when *target* is a
            string that's ambiguous across families. Ignored when
            ``currency=True``.
        currency: When ``True`` *target* is interpreted as currencies;
            the second field is named ``<name>_currency`` and the
            equivalents land as ``<name>_<iso>`` (e.g. ``price_eur``).
            When ``False`` *target* is a unit family; the second field
            is named ``<name>_unit`` and the equivalent is
            ``<name>_<symbol>``.
        source_unit_nullable: Nullability of the ``<name>_unit`` /
            ``<name>_currency`` column. Defaults to ``False``: every
            curated row must carry the source unit.
        nullable: Nullability of the source-value column and the
            equivalent columns. Defaults to ``False``.
        dtype: DataType string for the source value column. Defaults
            to ``"decimal(18, 6)"`` — safe for monetary and physical
            measurements alike.
        equivalent_dtype: DataType string for the equivalent columns.
            Defaults to *dtype*; override to use a different
            precision/scale for standardised equivalents (e.g.
            ``"float64"`` for non-monetary equivalents).

    Returns:
        Ordered list of :class:`yggdrasil.data.Field` objects ready to
        feed :func:`yggdrasil.data.Schema.from_fields` or any DDL builder.
    """
    from yggdrasil.data import field
    from yggdrasil.data.types.base import DataType

    eq_dtype = equivalent_dtype if equivalent_dtype is not None else dtype
    source_dtype = DataType.from_str(dtype)
    equivalent_dtype_obj = DataType.from_str(eq_dtype)
    unit_dtype = DataType.from_str("string")

    if currency:
        if target is None:
            currency_targets: Sequence[Currency] = DEFAULT_CURRENCY_TARGETS
        elif isinstance(target, Currency):
            currency_targets = (target,)
        elif isinstance(target, str):
            currency_targets = (Currency.from_(target),)
        elif isinstance(target, (list, tuple)):
            currency_targets = tuple(Currency.from_(t) for t in target)
        else:
            raise TypeError(
                f"dash_dual_value_fields(currency=True, target={target!r}): "
                f"target must be None (defaults to EUR/USD/CHF), a Currency / "
                f"ISO code str (single equivalent), or a sequence of them."
            )
        unit_col_name = f"{name}_currency"
        equivalents = [
            field(
                standardized_column_name(name, c),
                equivalent_dtype_obj,
                nullable=nullable,
                metadata={"standardized_unit": c.code, "standardized_family": "currency"},
            )
            for c in currency_targets
        ]
    else:
        if target is None:
            raise ValueError(
                "dash_dual_value_fields(currency=False, target=None): pass a "
                "target Unit (or symbol) for physical-unit dual columns, or "
                "set currency=True for the EUR/USD/CHF currency pattern."
            )
        if isinstance(target, (list, tuple)):
            raise TypeError(
                "dash_dual_value_fields: physical-unit families take a single "
                "target unit, not a sequence. Pass currency=True for the "
                "multi-target currency pattern."
            )
        family, target_unit = _resolve_family(target, family)
        unit_col_name = f"{name}_unit"
        equivalents = [
            field(
                standardized_column_name(name, target_unit),
                equivalent_dtype_obj,
                nullable=nullable,
                metadata={
                    "standardized_unit": target_unit.symbol,
                    "standardized_family": family.__name__,
                },
            )
        ]

    fields: list[Field] = [
        field(name, source_dtype, nullable=nullable),
        field(unit_col_name, unit_dtype, nullable=source_unit_nullable),
        *equivalents,
    ]
    return fields
