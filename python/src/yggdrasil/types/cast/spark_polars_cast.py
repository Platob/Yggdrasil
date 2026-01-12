"""Spark <-> Polars conversion helpers via Arrow."""

from typing import Optional

import pyarrow as pa

from .cast_options import CastOptions
from .polars_cast import *
from .registry import register_converter
from .spark_cast import *
from ...libs.polarslib import polars
from ...libs.sparklib import pyspark

__all__ = [
    "spark_dataframe_to_polars_dataframe",
    "polars_dataframe_to_spark_dataframe",
    "spark_dtype_to_polars_dtype",
    "polars_dtype_to_spark_dtype",
    "spark_field_to_polars_field",
    "polars_field_to_spark_field",
]

# ---------------------------------------------------------------------------
# Type aliases + decorator wrapper (safe when Spark/Polars are missing)
# ---------------------------------------------------------------------------
if pyspark is not None and polars is not None:
    def spark_polars_converter(*args, **kwargs):
        """Return a register_converter wrapper when deps are available."""
        return register_converter(*args, **kwargs)
else:
    def spark_polars_converter(*_args, **_kwargs):  # pragma: no cover - no-op decorator
        """Return a no-op decorator when deps are missing."""
        def _decorator(func):
            """Return the function unchanged."""
            return func

        return _decorator


# ---------------------------------------------------------------------------
# Spark DataFrame <-> Polars DataFrame via Arrow
# ---------------------------------------------------------------------------
@spark_polars_converter(SparkDataFrame, PolarsDataFrame)
def spark_dataframe_to_polars_dataframe(
    dataframe: SparkDataFrame,
    options: Optional[CastOptions] = None,
) -> PolarsDataFrame:
    """
    Convert a Spark DataFrame to a Polars DataFrame using Arrow as the bridge.

    Flow:
      Spark DataFrame
        -> (spark_dataframe_to_arrow_table) pyarrow.Table
        -> (arrow_table_to_polars_dataframe) Polars DataFrame

    Casting behavior:
      - ArrowCastOptions are applied on the Spark -> Arrow side via
        spark_dataframe_to_arrow_table.
      - The resulting Arrow table is already in the target schema, so we pass
        `None` to arrow_table_to_polars_dataframe to avoid double-casting.
    """
    if pyspark is None or polars is None:
        raise RuntimeError("Both pyspark and polars are required for this conversion")

    # Spark -> Arrow (includes Arrow-side casting if target_schema is set)
    table = spark_dataframe_to_arrow_table(dataframe, options)

    # Arrow -> Polars (no extra casting; table already conforms to target schema)
    return arrow_table_to_polars_dataframe(table, options)


@spark_polars_converter(PolarsDataFrame, SparkDataFrame)
def polars_dataframe_to_spark_dataframe(
    dataframe: PolarsDataFrame,
    options: Optional[CastOptions] = None,
) -> SparkDataFrame:
    """
    Convert a Polars DataFrame to a Spark DataFrame using Arrow as the bridge.

    Flow:
      Polars DataFrame
        -> (polars_dataframe_to_arrow_table) pyarrow.Table
        -> (arrow_table_to_spark_dataframe) Spark DataFrame

    Casting behavior:
      - ArrowCastOptions are applied on the Polars -> Arrow side via
        polars_dataframe_to_arrow_table.
      - The resulting Arrow table is already in the target schema, so we pass
        `None` to arrow_table_to_spark_dataframe to avoid double-casting.
    """
    if pyspark is None or polars is None:
        raise RuntimeError("Both pyspark and polars are required for this conversion")

    options = CastOptions.check_arg(options)

    # Polars -> Arrow (includes Arrow-side casting if target_schema is set)
    table = polars_dataframe_to_arrow_table(dataframe, options)

    # Arrow -> Spark (no extra casting; table already conforms to target schema)
    return arrow_table_to_spark_dataframe(table, options)


# ---------------------------------------------------------------------------
# Spark DataType <-> Polars DataType via Arrow
# ---------------------------------------------------------------------------
@spark_polars_converter(SparkDataType, PolarsDataType)
def spark_dtype_to_polars_dtype(
    dtype: "pyspark.sql.types.DataType",
    options: Optional[CastOptions] = None,
) -> "polars.datatypes.DataType":
    """
    Convert a Spark DataType to a Polars DataType via Arrow.

    Flow:
      Spark DataType
        -> (wrap in StructField) Spark StructField
        -> (spark_field_to_arrow_field) pyarrow.Field
        -> Arrow DataType
        -> (arrow_type_to_polars_type) Polars DataType
    """
    if pyspark is None or polars is None:
        raise RuntimeError("Both pyspark and polars are required for this conversion")

    options = CastOptions.check_arg(options)

    # Wrap Spark DataType into a StructField so we can reuse existing helper
    sf = pyspark.sql.types.StructField("value", dtype, nullable=True)
    arrow_field = spark_field_to_arrow_field(sf, options)
    arrow_type = arrow_field.type

    return arrow_type_to_polars_type(arrow_type, options)


@spark_polars_converter(PolarsDataType, SparkDataType)
def polars_dtype_to_spark_dtype(
    dtype: "polars.datatypes.DataType",
    options: Optional[CastOptions] = None,
) -> "pyspark.sql.types.DataType":
    """
    Convert a Polars DataType to a Spark DataType via Arrow.

    Flow:
      Polars DataType
        -> (dummy Series) polars.Series(dtype)
        -> Arrow Array
        -> Arrow DataType
        -> pyarrow.Field
        -> (arrow_field_to_spark_field) Spark StructField
        -> Spark DataType
    """
    if pyspark is None or polars is None:
        raise RuntimeError("Both pyspark and polars are required for this conversion")

    options = CastOptions.check_arg(options)

    # Build an empty Series just to obtain the Arrow dtype
    s = polars.Series("value", [], dtype=dtype)
    arr = s.to_arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    arrow_type = arr.type

    arrow_field = pa.field("value", arrow_type, nullable=True)
    spark_field = arrow_field_to_spark_field(arrow_field, options)
    return spark_field.dataType


# ---------------------------------------------------------------------------
# Spark StructField <-> Polars Field via Arrow
# ---------------------------------------------------------------------------
@spark_polars_converter(SparkStructField, PolarsField)
def spark_field_to_polars_field(
    field: SparkStructField,
    options: Optional[CastOptions] = None,
) -> PolarsField:
    """
    Convert a Spark StructField to a Polars Field via Arrow.

    Flow:
      Spark StructField
        -> (spark_field_to_arrow_field) pyarrow.Field
        -> Arrow DataType
        -> (arrow_type_to_polars_type) Polars DataType
        -> Polars Field(name, dtype)
    """
    if pyspark is None or polars is None:
        raise RuntimeError("Both pyspark and polars are required for this conversion")

    options = CastOptions.check_arg(options)

    arrow_field = spark_field_to_arrow_field(field, options)
    pl_dtype = arrow_type_to_polars_type(arrow_field.type, options)

    # Polars Field does not encode nullability explicitly; dtype will be nullable by default
    return PolarsField(arrow_field.name, pl_dtype)


@spark_polars_converter(PolarsField, SparkStructField)
def polars_field_to_spark_field(
    field: PolarsField,
    options: Optional[CastOptions] = None,
) -> SparkStructField:
    """
    Convert a Polars Field to a Spark StructField via Arrow.

    Flow:
      Polars Field(name, dtype)
        -> (dummy Series) polars.Series(dtype)
        -> Arrow Array
        -> Arrow DataType
        -> pyarrow.Field
        -> (arrow_field_to_spark_field) Spark StructField
    """
    if pyspark is None or polars is None:
        raise RuntimeError("Both pyspark and polars are required for this conversion")

    options = CastOptions.check_arg(options)

    # field.dtype is a Polars DataType
    pl_dtype = field.dtype

    s = polars.Series(field.name, [], dtype=pl_dtype)
    arr = s.to_arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    arrow_type = arr.type

    # We default nullable=True; if you want strict nullability you can extend this
    arrow_field = pa.field(field.name, arrow_type, nullable=True)
    spark_field = arrow_field_to_spark_field(arrow_field, options)
    return spark_field
