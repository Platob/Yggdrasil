from typing import Optional

import pyarrow as pa

from .arrow import (
    ArrowCastOptions,
    cast_arrow_array,
    cast_arrow_table,
    cast_arrow_record_batch_reader,
)
from .registry import register_converter
from ...libs.pandaslib import pandas, require_pandas

__all__ = [
    "cast_pandas_series",
    "cast_pandas_dataframe",
    "arrow_array_to_pandas_series",
    "arrow_table_to_pandas_dataframe",
    "record_batch_reader_to_pandas_dataframe",
    "pandas_series_to_arrow_array",
    "pandas_dataframe_to_arrow_table",
    "pandas_dataframe_to_record_batch_reader",
]

# ---------------------------------------------------------------------------
# pandas type aliases + decorator wrapper (safe when pandas is missing)
# ---------------------------------------------------------------------------

if pandas is not None:
    require_pandas()

    PandasSeries = pandas.Series
    PandasDataFrame = pandas.DataFrame

    def pandas_converter(*args, **kwargs):
        return register_converter(*args, **kwargs)

else:
    # Dummy types so annotations/decorators don't explode without pandas
    class _PandasDummy:  # pragma: no cover - only used when pandas not installed
        pass

    PandasSeries = _PandasDummy
    PandasDataFrame = _PandasDummy

    def pandas_converter(*_args, **_kwargs):  # pragma: no cover - no-op decorator
        def _decorator(func):
            return func

        return _decorator


# ---------------------------------------------------------------------------
# pandas <-> Arrow via ArrowCastOptions
# ---------------------------------------------------------------------------


@pandas_converter(PandasSeries, PandasSeries)
def cast_pandas_series(
    series: "pandas.Series",
    options: Optional[ArrowCastOptions] = None,
) -> "pandas.Series":
    """
    Cast a pandas Series to a target Arrow type using Arrow casting rules.

    The target type/field should be provided via `options` (e.g. options.target_schema
    or options.target_field, depending on how ArrowCastOptions is defined).

    Arrow does:
      - type cast
      - nullability enforcement
      - default handling (via cast_arrow_array)
    We then convert back to pandas and restore index/name.
    """
    options = ArrowCastOptions.check_arg(options)

    arrow_array = pa.array(series, from_pandas=True)
    casted = cast_arrow_array(arrow_array, options)

    result = casted.to_pandas()
    result.index = series.index
    result.name = series.name
    return result


@pandas_converter(PandasDataFrame, PandasDataFrame)
def cast_pandas_dataframe(
    dataframe: "pandas.DataFrame",
    options: Optional[ArrowCastOptions] = None,
) -> "pandas.DataFrame":
    """
    Cast a pandas DataFrame to a target Arrow schema using Arrow casting rules.

    Behavior is analogous to the Polars version, but we delegate casting to
    `cast_arrow_table` and then adjust columns on the pandas side:

      - options.target_schema: Arrow schema / field used by cast_arrow_table
      - options.allow_add_columns:
          * False: result only has columns from the cast Arrow table
          * True: extra pandas columns (not in the target schema / cast result)
                  are appended unchanged
    """
    options = ArrowCastOptions.check_arg(options)

    original_index = dataframe.index

    arrow_table = pa.Table.from_pandas(dataframe, preserve_index=False)
    casted_table = cast_arrow_table(arrow_table, options)

    result = casted_table.to_pandas()
    result.index = original_index

    if getattr(options, "allow_add_columns", False):
        casted_cols = set(result.columns)
        extra_cols = [col for col in dataframe.columns if col not in casted_cols]

        if extra_cols:
            extra_df = dataframe[extra_cols]
            extra_df.index = result.index
            result = pandas.concat([result, extra_df], axis=1)

    return result


# ---------------------------------------------------------------------------
# Arrow -> pandas
# ---------------------------------------------------------------------------


@pandas_converter(pa.Array, PandasSeries)
@pandas_converter(pa.ChunkedArray, PandasSeries)
def arrow_array_to_pandas_series(
    array: pa.Array,
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pandas.Series":
    """
    Convert a pyarrow.Array (or ChunkedArray) to a pandas Series,
    optionally applying Arrow casting via ArrowCastOptions before conversion.
    """
    opts = ArrowCastOptions.check_arg(cast_options)

    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()

    casted = cast_arrow_array(array, opts)
    return casted.to_pandas()


@pandas_converter(pa.Table, PandasDataFrame)
def arrow_table_to_pandas_dataframe(
    table: pa.Table,
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pandas.DataFrame":
    """
    Convert a pyarrow.Table to a pandas DataFrame, optionally applying Arrow
    casting rules first.
    """
    opts = ArrowCastOptions.check_arg(cast_options)

    if opts.target_schema is not None:
        table = cast_arrow_table(table, opts)

    return table.to_pandas()


@pandas_converter(pa.RecordBatchReader, PandasDataFrame)
def record_batch_reader_to_pandas_dataframe(
    reader: pa.RecordBatchReader,
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pandas.DataFrame":
    """
    Convert a pyarrow.RecordBatchReader to a pandas DataFrame.

    - If cast_options.target_schema is set, we first apply
      `cast_arrow_record_batch_reader` and then collect to a Table and pandas DF.
    """
    opts = ArrowCastOptions.check_arg(cast_options)

    if opts.target_schema is not None:
        reader = cast_arrow_record_batch_reader(reader, opts)

    batches = list(reader)
    if not batches:
        empty_table = pa.Table.from_arrays([], names=[])
        return empty_table.to_pandas()

    table = pa.Table.from_batches(batches)
    return arrow_table_to_pandas_dataframe(table, opts)


# ---------------------------------------------------------------------------
# pandas -> Arrow
# ---------------------------------------------------------------------------


@pandas_converter(PandasSeries, pa.Array)
def pandas_series_to_arrow_array(
    series: "pandas.Series",
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.Array:
    """
    Convert a pandas Series to a pyarrow.Array, optionally applying Arrow
    casting via ArrowCastOptions.
    """
    opts = ArrowCastOptions.check_arg(cast_options)

    array = pa.array(series, from_pandas=True)
    return cast_arrow_array(array, opts)


@pandas_converter(PandasDataFrame, pa.Table)
def pandas_dataframe_to_arrow_table(
    dataframe: "pandas.DataFrame",
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.Table:
    """
    Convert a pandas DataFrame to a pyarrow.Table, optionally applying Arrow
    casting rules via ArrowCastOptions.
    """
    opts = ArrowCastOptions.check_arg(cast_options)

    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    return cast_arrow_table(table, opts)


@pandas_converter(PandasDataFrame, pa.RecordBatchReader)
def pandas_dataframe_to_record_batch_reader(
    dataframe: "pandas.DataFrame",
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatchReader:
    """
    Convert a pandas DataFrame to a pyarrow.RecordBatchReader, optionally
    applying Arrow casting via ArrowCastOptions.
    """
    opts = ArrowCastOptions.check_arg(cast_options)

    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    table = cast_arrow_table(table, opts)

    batches = table.to_batches()
    return pa.RecordBatchReader.from_batches(table.schema, batches)
