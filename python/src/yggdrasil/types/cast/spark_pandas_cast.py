from typing import Optional

from .arrow_cast import ArrowCastOptions
from .registry import register_converter

# Reuse existing Spark <-> Arrow helpers
from .spark_cast import (
    spark_dataframe_to_arrow_table,
    arrow_table_to_spark_dataframe,
)

# Reuse existing pandas <-> Arrow helpers
from .pandas_cast import (
    pandas_dataframe_to_arrow_table,
    arrow_table_to_pandas_dataframe,
)

from ...libs.sparklib import pyspark
from ...libs.pandaslib import pandas, require_pandas

__all__ = [
    "spark_dataframe_to_pandas_dataframe",
    "pandas_dataframe_to_spark_dataframe",
]

# ---------------------------------------------------------------------------
# Type aliases + decorator wrapper (safe when one side is missing)
# ---------------------------------------------------------------------------

if pyspark is not None and pandas is not None:
    require_pandas()

    SparkDataFrame = pyspark.sql.DataFrame
    PandasDataFrame = pandas.DataFrame

    def spark_pandas_converter(*args, **kwargs):
        return register_converter(*args, **kwargs)

else:
    # Dummy stand-ins so decorators/annotations don't explode if one lib is absent
    class _Dummy:  # pragma: no cover - only used when Spark or pandas not installed
        pass

    SparkDataFrame = _Dummy
    PandasDataFrame = _Dummy

    def spark_pandas_converter(*_args, **_kwargs):  # pragma: no cover - no-op decorator
        def _decorator(func):
            return func

        return _decorator


# ---------------------------------------------------------------------------
# Spark DataFrame <-> pandas DataFrame via Arrow
# ---------------------------------------------------------------------------


@spark_pandas_converter(SparkDataFrame, PandasDataFrame)
def spark_dataframe_to_pandas_dataframe(
    dataframe: "pyspark.sql.DataFrame",
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pandas.DataFrame":
    """
    Convert a Spark DataFrame to a pandas DataFrame using Arrow as the bridge.

    Flow:
      Spark DataFrame
        -> (spark_dataframe_to_arrow_table) pyarrow.Table
        -> (arrow_table_to_pandas_dataframe) pandas.DataFrame

    Casting behavior:
      - ArrowCastOptions are applied on the Spark -> Arrow side via
        spark_dataframe_to_arrow_table.
      - The resulting Arrow table is already in the target schema, so we pass
        `None` to arrow_table_to_pandas_dataframe to avoid double-casting.
    """
    if pyspark is None or pandas is None:
        raise RuntimeError("Both pyspark and pandas are required for this conversion")

    opts = ArrowCastOptions.check_arg(cast_options)

    # Spark -> Arrow (includes Arrow-side casting if target_schema is set)
    table = spark_dataframe_to_arrow_table(dataframe, opts)

    # Arrow -> pandas (no extra casting; table already conforms to target schema)
    return arrow_table_to_pandas_dataframe(table, None)


@spark_pandas_converter(PandasDataFrame, SparkDataFrame)
def pandas_dataframe_to_spark_dataframe(
    dataframe: "pandas.DataFrame",
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pyspark.sql.DataFrame":
    """
    Convert a pandas DataFrame to a Spark DataFrame using Arrow as the bridge.

    Flow:
      pandas.DataFrame
        -> (pandas_dataframe_to_arrow_table) pyarrow.Table
        -> (arrow_table_to_spark_dataframe) Spark DataFrame

    Casting behavior:
      - ArrowCastOptions are applied on the pandas -> Arrow side via
        pandas_dataframe_to_arrow_table.
      - The resulting Arrow table is already in the target schema, so we pass
        `None` to arrow_table_to_spark_dataframe to avoid double-casting.
    """
    if pyspark is None or pandas is None:
        raise RuntimeError("Both pyspark and pandas are required for this conversion")

    opts = ArrowCastOptions.check_arg(cast_options)

    # pandas -> Arrow (includes Arrow-side casting if target_schema is set)
    table = pandas_dataframe_to_arrow_table(dataframe, opts)

    # Arrow -> Spark (no extra casting; table already conforms to target schema)
    return arrow_table_to_spark_dataframe(table, None)
