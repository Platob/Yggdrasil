from __future__ import annotations

from typing import Any, Optional, Union

import polars as pl
import pyarrow as pa

from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.options import CastOptions
from yggdrasil.data.cast.registry import register_converter

__all__ = [
    "register_converter",
    "cast_polars_array",
    "cast_polars_dataframe",
    "cast_polars_lazyframe",
    "any_to_polars_dataframe",
    "polars_dataframe_to_arrow_table",
]


@register_converter(pl.Series, pl.Series)
@register_converter(pl.Expr, pl.Expr)
def cast_polars_array(
    array: Union[pl.Series, pl.Expr],
    options: Optional[CastOptions] = None,
) -> Union[pl.Series, pl.Expr]:
    return CastOptions.check(options).cast_polars_series(array)


@register_converter(pl.DataFrame, pl.DataFrame)
def cast_polars_dataframe(
    df: pl.DataFrame,
    options: Optional[CastOptions] = None,
) -> pl.DataFrame:
    return CastOptions.check(options).cast_polars_tabular(df)


@register_converter(pl.LazyFrame, pl.LazyFrame)
def cast_polars_lazyframe(
    df: pl.LazyFrame,
    options: Optional[CastOptions] = None,
) -> pl.LazyFrame:
    return CastOptions.check(options).cast_polars_tabular(df)


@register_converter(Any, pl.DataFrame)
def any_to_polars_dataframe(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pl.DataFrame:
    """Convert *any* supported object to a ``pl.DataFrame``, then cast to the target schema.

    Dispatch order is by isinstance against the well-known engine
    classes first (polars, pyarrow), then by ``type(obj).__module__``
    for the rarer engines that we can't isinstance against without
    importing them (pandas, pyspark). The module-prefix check is
    cheap — ``type(obj).__module__`` is a single attribute read,
    no ``inspect.unwrap`` walk.

    Supported input types:

    * ``pl.DataFrame`` — passed through to :func:`cast_polars_dataframe`.
    * ``pl.LazyFrame`` — ``collect()`` first, then cast.
    * ``pa.Table`` / ``pa.RecordBatch`` / ``pa.RecordBatchReader`` /
      ``pa.dataset.Scanner`` — routed through ``pl.from_arrow``
      (zero-copy where possible).
    * ``pandas.DataFrame`` — routed through ``pl.from_pandas``.
    * ``pyspark.sql.DataFrame`` — materialised via Spark's ``toArrow()`` and
      loaded with ``pl.from_arrow``.
    * ``None`` — an empty frame with the target schema (if any).
    * Everything else (dicts, sequences of dicts, dataclasses, …) —
      constructed directly via ``pl.DataFrame(obj)``.
    """
    opts = CastOptions.check(options)

    if isinstance(obj, pl.DataFrame):
        return cast_polars_dataframe(obj, opts)

    if isinstance(obj, pl.LazyFrame):
        return cast_polars_dataframe(obj.collect(), opts)

    # Arrow shapes — isinstance is faster than the namespace probe and
    # covers the vast majority of cross-engine handoffs.
    if isinstance(obj, pa.Table):
        return cast_polars_dataframe(pl.from_arrow(obj), opts)
    if isinstance(obj, pa.RecordBatch):
        table = pa.Table.from_batches([obj], schema=obj.schema)  # type: ignore[arg-type]
        return cast_polars_dataframe(pl.from_arrow(table), opts)

    if obj is None:
        schema = (
            opts.target.to_schema().to_polars_schema()
            if opts.target is not None
            else None
        )
        return cast_polars_dataframe(pl.DataFrame(schema=schema), opts)

    # ``type(obj).__module__`` is a C-level attribute read; the legacy
    # ``ObjectSerde.full_namespace`` path runs ``inspect.unwrap`` plus
    # a handful of ``getattr`` hops, which dominates this dispatch for
    # small frames.
    module = (type(obj).__module__ or "").partition(".")[0]

    if module == "pyarrow":
        # Less-common pyarrow shapes — Scanner / Dataset / Streamer.
        if hasattr(obj, "to_table"):
            obj = obj.to_table()
        elif hasattr(obj, "read_all"):
            obj = obj.read_all()

        if not isinstance(obj, pa.Table):
            raise TypeError(f"Cannot convert {type(obj).__name__} to polars.DataFrame")

        df = pl.from_arrow(obj)
    elif module == "pandas":
        df = pl.from_pandas(obj)
    elif module == "pyspark":
        import pyspark.sql as pyspark_sql

        if isinstance(obj, pyspark_sql.DataFrame):
            from yggdrasil.spark.cast import spark_dataframe_to_arrow
            df = pl.from_arrow(spark_dataframe_to_arrow(obj))
        else:
            df = pl.DataFrame(obj)
    else:
        df = pl.DataFrame(obj)

    return cast_polars_dataframe(df, opts)


@register_converter(pl.DataFrame, pa.Table)
@register_converter(pl.LazyFrame, pa.Table)
def polars_dataframe_to_arrow_table(
    df: Union[pl.DataFrame, pl.LazyFrame],
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """Convert a Polars frame to a ``pa.Table``, then cast to the target schema.

    ``pl.LazyFrame`` inputs are materialised via ``collect()`` first.
    The polars→arrow bridge runs with ``compat_level=newest()`` so
    string / binary / list columns surface as Arrow view types
    (``string_view`` / ``binary_view`` / ``list_view``) which Polars
    produces zero-copy — a ~6× speedup over the legacy ``to_arrow()``
    default (large_string / large_binary / large_list) on the hot
    name-heavy shape.  The downstream :func:`cast_arrow_tabular`
    pass casts view types to whatever the target schema demands
    (typically ``string`` / ``binary``), which Arrow's compute kernels
    handle cheaply via the view layout.
    """
    opts = CastOptions.check(options)

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"polars_dataframe_to_arrow_table expected polars.DataFrame/LazyFrame, "
            f"got {type(df).__name__}"
        )

    return cast_arrow_tabular(
        df.to_arrow(compat_level=pl.CompatLevel.newest()),
        opts,
    )
