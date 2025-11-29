from typing import Optional

import pyarrow as pa

from ...libs.polarslib import polars, require_polars
from .arrow import ArrowCastOptions, cast_arrow_array, cast_arrow_table

__all__ = [
    "cast_polars_series",
    "cast_polars_dataframe",
]


@require_polars
def cast_polars_series(
    series: "polars.Series",
    arrow_type: pa.DataType,
    options: Optional[ArrowCastOptions] = None,
) -> "polars.Series":
    """Cast a Polars Series to a target Arrow type using Arrow casting rules."""

    options = options or ArrowCastOptions()
    compat_level = polars.CompatLevel.newest()
    arrow_array = series.to_arrow(compat_level=compat_level)
    casted = cast_arrow_array(arrow_array, arrow_type, options)

    # Preserve the original series name where possible.
    return polars.Series(name=series.name, values=casted)


@require_polars
def cast_polars_dataframe(
    dataframe: "polars.DataFrame",
    arrow_schema: pa.Schema,
    options: Optional[ArrowCastOptions] = None,
) -> "polars.DataFrame":
    """Cast a Polars DataFrame to a target Arrow schema using Arrow casting rules."""

    options = options or ArrowCastOptions()
    compat_level = polars.CompatLevel.newest()
    arrow_table = dataframe.to_arrow(compat_level=compat_level)
    casted = cast_arrow_table(arrow_table, arrow_schema, options)

    return polars.from_arrow(casted)
