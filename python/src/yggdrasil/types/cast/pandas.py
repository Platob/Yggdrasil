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
]


# ---------- pandas <-> Arrow via ArrowCastOptions ----------


@require_pandas
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


@require_pandas
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


# ---------- Arrow -> pandas ----------


@require_pandas
def arrow_array_to_pandas_series(
    array: pa.Array,
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pandas.Series":
    """
    Convert a pyarrow.Array to a pandas Series, optionally applying Arrow casting
    via ArrowCastOptions before conversion.
    """
    opts = ArrowCastOptions.check_arg(cast_options)
    casted = cast_arrow_array(array, opts)
    return casted.to_pandas()


@require_pandas
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


@require_pandas
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


# ---------- pandas -> Arrow ----------


@require_pandas
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


@require_pandas
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


@require_pandas
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


# ---------- register converters (like Polars) ----------


if pandas is not None:
    # Same-type pandas casts using Arrow types
    register_converter(pandas.Series, pandas.Series)(cast_pandas_series)
    register_converter(pandas.DataFrame, pandas.DataFrame)(cast_pandas_dataframe)

    # pandas -> Arrow
    register_converter(pandas.Series, pa.Array)(pandas_series_to_arrow_array)
    register_converter(pandas.DataFrame, pa.Table)(pandas_dataframe_to_arrow_table)
    register_converter(
        pandas.DataFrame,
        pa.RecordBatchReader,
    )(pandas_dataframe_to_record_batch_reader)

    # Arrow -> pandas
    register_converter(pa.Array, pandas.Series)(arrow_array_to_pandas_series)
    register_converter(pa.ChunkedArray, pandas.Series)(arrow_array_to_pandas_series)
    register_converter(pa.Table, pandas.DataFrame)(arrow_table_to_pandas_dataframe)
    register_converter(
        pa.RecordBatchReader,
        pandas.DataFrame,
    )(record_batch_reader_to_pandas_dataframe)
