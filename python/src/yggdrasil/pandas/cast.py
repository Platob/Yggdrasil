"""Pandas <-> Arrow casting helpers and converters."""

from typing import Optional

import pyarrow as pa

from yggdrasil.pandas.lib import pandas
from yggdrasil.types.cast.arrow_cast import cast_arrow_array, cast_arrow_tabular
from yggdrasil.types.cast.cast_options import CastOptions
from yggdrasil.types.cast.registry import register_converter

__all__ = [
    "cast_pandas_series",
    "cast_pandas_dataframe",
    "arrow_array_to_pandas_series",
    "arrow_table_to_pandas_dataframe",
    "pandas_series_to_arrow_array",
    "pandas_dataframe_to_arrow_table",
]



@register_converter(pandas.Series, pandas.Series)
def cast_pandas_series(
    series: pandas.Series,
    options: Optional[CastOptions] = None,
) -> pandas.Series:
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
    options = CastOptions.check_arg(options)

    arrow_array = pa.array(series, from_pandas=True)
    casted = cast_arrow_array(arrow_array, options)

    result = casted.to_pandas()
    result.index = series.index
    result.name = series.name
    return result


@register_converter(pandas.DataFrame, pandas.DataFrame)
def cast_pandas_dataframe(
    dataframe: pandas.DataFrame,
    options: Optional[CastOptions] = None,
) -> pandas.DataFrame:
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
    options = CastOptions.check_arg(options)

    original_index = dataframe.index

    arrow_table = pa.Table.from_pandas(dataframe, preserve_index=False)
    casted_table = cast_arrow_tabular(arrow_table, options)

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


@register_converter(pa.Array, pandas.Series)
@register_converter(pa.ChunkedArray, pandas.Series)
def arrow_array_to_pandas_series(
    array: pa.Array,
    cast_options: Optional[CastOptions] = None,
) -> pandas.Series:
    """
    Convert a pyarrow.Array (or ChunkedArray) to a pandas Series,
    optionally applying Arrow casting via ArrowCastOptions before conversion.
    """
    opts = CastOptions.check_arg(cast_options)

    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()

    casted = cast_arrow_array(array, opts)
    return casted.to_pandas()


@register_converter(pa.Table, pandas.DataFrame)
def arrow_table_to_pandas_dataframe(
    table: pa.Table,
    cast_options: Optional[CastOptions] = None,
) -> pandas.DataFrame:
    """
    Convert a pyarrow.Table to a pandas DataFrame, optionally applying Arrow
    casting rules first.
    """
    opts = CastOptions.check_arg(cast_options)

    if opts.target_arrow_schema is not None:
        table = cast_arrow_tabular(table, opts)

    return table.to_pandas()


# ---------------------------------------------------------------------------
# pandas -> Arrow
# ---------------------------------------------------------------------------


@register_converter(pandas.Series, pa.Array)
def pandas_series_to_arrow_array(
    series: pandas.Series,
    cast_options: Optional[CastOptions] = None,
) -> pa.Array:
    """
    Convert a pandas Series to a pyarrow.Array, optionally applying Arrow
    casting via ArrowCastOptions.
    """
    opts = CastOptions.check_arg(cast_options)

    array = pa.array(series, from_pandas=True)
    return cast_arrow_array(array, opts)


@register_converter(pandas.DataFrame, pa.Table)
def pandas_dataframe_to_arrow_table(
    df: pandas.DataFrame,
    cast_options: Optional[CastOptions] = None,
) -> pa.Table:
    """
    Convert a pandas DataFrame to a pyarrow.Table, optionally applying Arrow
    casting rules via ArrowCastOptions.
    """
    opts = CastOptions.check_arg(cast_options)

    table = pa.Table.from_pandas(df, preserve_index=False)
    return cast_arrow_tabular(table, opts)
