from typing import Optional, Any

import pyarrow as pa

from ...libs.pandaslib import pandas, require_pandas
from .arrow import ArrowCastOptions, cast_arrow_array, cast_arrow_table

__all__ = [
    "cast_pandas_series",
    "cast_pandas_dataframe",
]


@require_pandas
def cast_pandas_series(
    series: "pandas.Series",
    options: Optional[ArrowCastOptions] = None,
    default_value: Any = None,
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

    # Convert to Arrow, cast via Arrow, then back to pandas.
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
    default_value: Any = None,
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

    # Keep original index to restore after Arrow roundtrip
    original_index = dataframe.index

    # Convert to Arrow (no index column), cast with Arrow semantics
    arrow_table = pa.Table.from_pandas(dataframe, preserve_index=False)
    casted_table = cast_arrow_table(arrow_table, options)

    # Back to pandas
    result = casted_table.to_pandas()
    result.index = original_index

    # If we allow extra columns, bring over any columns that weren't part of the cast
    if getattr(options, "allow_add_columns", False):
        casted_cols = set(result.columns)
        extra_cols = [col for col in dataframe.columns if col not in casted_cols]

        if extra_cols:
            # Preserve original values for extra columns
            extra_df = dataframe[extra_cols]
            extra_df.index = result.index  # keep index aligned
            result = pandas.concat([result, extra_df], axis=1)

    return result
