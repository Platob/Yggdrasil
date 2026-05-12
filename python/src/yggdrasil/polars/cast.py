from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Optional, Union

import polars as pl
import pyarrow as pa

from yggdrasil.arrow.cast import cast_arrow_tabular, rechunk_arrow_batches
from yggdrasil.data.options import CastOptions
from yggdrasil.data.cast.registry import register_converter
from yggdrasil.pickle.serde import ObjectSerde

__all__ = [
    "register_converter",
    "cast_polars_array",
    "cast_polars_dataframe",
    "cast_polars_lazyframe",
    "any_to_polars_dataframe",
    "polars_dataframe_to_arrow_table",
    "rechunk_polars_frames",
    "rechunk_polars_frame",
]

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]


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

    Supported input types (detected via namespace inspection):

    * ``pl.DataFrame`` — passed through to :func:`cast_polars_dataframe`.
    * ``pl.LazyFrame`` — ``collect()`` first, then cast.
    * ``pa.Table`` / ``pa.RecordBatch`` / ``pa.RecordBatchReader`` — routed
      through ``pl.from_arrow`` (zero-copy).
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

    if obj is None:
        schema = (
            opts.target_field.to_schema().to_polars_schema()
            if opts.target_field is not None
            else None
        )
        return cast_polars_dataframe(pl.DataFrame(schema=schema), opts)

    namespace = ObjectSerde.full_namespace(obj)

    if namespace.startswith("pyarrow"):
        if isinstance(obj, pa.RecordBatch):
            obj = pa.Table.from_batches([obj], schema=obj.schema)  # type: ignore[arg-type]
        elif hasattr(obj, "to_table"):
            obj = obj.to_table()

        if not isinstance(obj, pa.Table):
            raise TypeError(f"Cannot convert {type(obj).__name__} to polars.DataFrame")

        df = pl.from_arrow(obj)
    elif namespace.startswith("pandas"):
        df = pl.from_pandas(obj)
    elif namespace.startswith("pyspark"):
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
    The resulting Arrow table is routed through :func:`cast_arrow_tabular`
    so ``options.target_field`` is honoured symmetrically with the
    pandas and Spark helpers.
    """
    opts = CastOptions.check(options)

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"polars_dataframe_to_arrow_table expected polars.DataFrame/LazyFrame, "
            f"got {type(df).__name__}"
        )

    return cast_arrow_tabular(df.to_arrow(), opts)


def _polars_frame_to_batches(frame: PolarsFrame) -> Iterator[pa.RecordBatch]:
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    # ``DataFrame.to_arrow`` is zero-copy; the resulting table's batches
    # share buffers with the polars frame.
    yield from frame.to_arrow().to_batches()


def rechunk_polars_frames(
    frames: Iterable[PolarsFrame],
    *,
    byte_size: int | None = None,
    row_size: int | None = None,
    lazy: bool = False,
    memory_pool: pa.MemoryPool | None = None,
) -> Iterator[PolarsFrame]:
    """Stream-coalesce/slice polars frames to ~``byte_size`` / ``row_size`` chunks.

    Thin polars-shaped wrapper over
    :func:`yggdrasil.arrow.cast.rechunk_arrow_batches` — each input frame is
    converted to Arrow record batches (zero-copy via ``DataFrame.to_arrow``),
    fed through the shared rechunker, and yielded back as polars frames.

    Parameters
    ----------
    frames :
        Iterable of ``pl.DataFrame`` / ``pl.LazyFrame``. ``LazyFrame`` inputs
        are ``collect()``-ed on demand; sizing decisions need materialized
        data.
    byte_size, row_size :
        Same semantics as :func:`rechunk_arrow_batches` — neither set is
        passthrough, ``row_size`` slices, ``byte_size`` coalesces, both caps
        the row target.
    lazy :
        Emit ``pl.LazyFrame`` chunks instead of ``pl.DataFrame``.
    memory_pool :
        Forwarded to :func:`rechunk_arrow_batches` for buffered concats.

    See :func:`yggdrasil.arrow.cast.rechunk_arrow_batches` for the underlying
    algorithm.
    """
    def _batches() -> Iterator[pa.RecordBatch]:
        for frame in frames:
            yield from _polars_frame_to_batches(frame)

    rechunked = rechunk_arrow_batches(
        _batches(),
        byte_size=byte_size,
        row_size=row_size,
        memory_pool=memory_pool,
    )
    for batch in rechunked:
        df = pl.from_arrow(batch)
        yield df.lazy() if lazy else df


def rechunk_polars_frame(
    frame: PolarsFrame,
    *,
    byte_size: int | None = None,
    row_size: int | None = None,
    lazy: bool = False,
    memory_pool: pa.MemoryPool | None = None,
) -> Iterator[PolarsFrame]:
    """Re-chunk a single polars frame; convenience for the common one-frame case.

    Equivalent to ``rechunk_polars_frames([frame], ...)``. ``LazyFrame``
    input is ``collect()``-ed before chunking.
    """
    return rechunk_polars_frames(
        [frame],
        byte_size=byte_size,
        row_size=row_size,
        lazy=lazy,
        memory_pool=memory_pool,
    )
