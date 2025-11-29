from typing import Optional

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
    arrow_type: pa.DataType,
    options: Optional[ArrowCastOptions] = None,
) -> "pandas.Series":
    """Cast a pandas Series to a target Arrow type using Arrow casting rules."""

    options = options or ArrowCastOptions()
    arrow_array = pa.array(series, from_pandas=True)
    casted = cast_arrow_array(arrow_array, arrow_type, options)

    result = casted.to_pandas()
    result.name = series.name
    return result


@require_pandas
def cast_pandas_dataframe(
    dataframe: "pandas.DataFrame",
    arrow_schema: pa.Schema,
    options: Optional[ArrowCastOptions] = None,
) -> "pandas.DataFrame":
    """Cast a pandas DataFrame to a target Arrow schema using Arrow casting rules."""

    options = options or ArrowCastOptions()
    arrow_table = pa.Table.from_pandas(dataframe, preserve_index=False)
    casted = cast_arrow_table(arrow_table, arrow_schema, options)

    return casted.to_pandas()
