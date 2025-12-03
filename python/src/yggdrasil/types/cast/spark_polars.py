from typing import Optional

import pyarrow as pa

from .arrow import ArrowCastOptions
from .registry import register_converter

# Spark <-> Arrow helpers
from .spark import (
    spark_dataframe_to_arrow_table,
    arrow_table_to_spark_dataframe,
)

# Polars <-> Arrow helpers
from .polars import (
    polars_dataframe_to_arrow_table,
    arrow_table_to_polars_dataframe,
)

from ...libs.sparklib import (
    pyspark,
    arrow_field_to_spark_field,
    spark_field_to_arrow_field,
)
from ...libs.polarslib import (
    polars,
    require_polars,
    arrow_type_to_polars_type,
)

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
    require_polars()

    from pyspark.sql.types import DataType as SparkDataTypeCls, StructField as SparkStructFieldCls
    from polars.datatypes import DataType as PolarsDataTypeCls, Field as PolarsFieldCls

    SparkDataFrame = pyspark.sql.DataFrame
    SparkDataType = SparkDataTypeCls
    SparkStructField = SparkStructFieldCls

    PolarsDataFrame = polars.DataFrame
    PolarsDataType = PolarsDataTypeCls
    PolarsField = PolarsFieldCls

    def spark_polars_converter(*args, **kwargs):
        return register_converter(*args, **kwargs)

else:
    # Dummy stand-ins so decorators/annotations don't explode if one side is missing
    class _Dummy:  # pragma: no cover - only used when Spark or Polars not installed
        pass

    SparkDataFrame = _Dummy
    SparkDataType = _Dummy
    SparkStructField = _Dummy

    PolarsDataFrame = _Dummy
    PolarsDataType = _Dummy
    PolarsField = _Dummy

    def spark_polars_converter(*_args, **_kwargs):  # pragma: no cover - no-op decorator
        def _decorator(func):
            return func

        return _decorator


# ---------------------------------------------------------------------------
# Spark DataFrame <-> Polars DataFrame via Arrow
# ---------------------------------------------------------------------------


@spark_polars_converter(SparkDataFrame, PolarsDataFrame)
def spark_dataframe_to_polars_dataframe(
    dataframe: "pyspark.sql.DataFrame",
    cast_options: Optional[ArrowCastOptions] = None,
) -> "polars.DataFrame":
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

    opts = ArrowCastOptions.check_arg(cast_options)

    # Spark -> Arrow (includes Arrow-side casting if target_schema is set)
    table = spark_dataframe_to_arrow_table(dataframe, opts)

    # Arrow -> Polars (no extra casting; table already conforms to target schema)
    return arrow_table_to_polars_dataframe(table, None)


@spark_polars_converter(PolarsDataFrame, SparkDataFrame)
def polars_dataframe_to_spark_dataframe(
    dataframe: "polars.DataFrame",
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pyspark.sql.DataFrame":
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

    opts = ArrowCastOptions.check_arg(cast_options)

    # Polars -> Arrow (includes Arrow-side casting if target_schema is set)
    table = polars_dataframe_to_arrow_table(dataframe, opts)

    # Arrow -> Spark (no extra casting; table already conforms to target schema)
    return arrow_table_to_spark_dataframe(table, None)


# ---------------------------------------------------------------------------
# Spark DataType <-> Polars DataType via Arrow
# ---------------------------------------------------------------------------


@spark_polars_converter(SparkDataType, PolarsDataType)
def spark_dtype_to_polars_dtype(
    dtype: "pyspark.sql.types.DataType",
    cast_options: Optional[ArrowCastOptions] = None,
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

    opts = ArrowCastOptions.check_arg(cast_options)

    # Wrap Spark DataType into a StructField so we can reuse existing helper
    sf = pyspark.sql.types.StructField("value", dtype, nullable=True)
    arrow_field = spark_field_to_arrow_field(sf, opts)
    arrow_type = arrow_field.type

    return arrow_type_to_polars_type(arrow_type, opts)


@spark_polars_converter(PolarsDataType, SparkDataType)
def polars_dtype_to_spark_dtype(
    dtype: "polars.datatypes.DataType",
    cast_options: Optional[ArrowCastOptions] = None,
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

    opts = ArrowCastOptions.check_arg(cast_options)

    # Build an empty Series just to obtain the Arrow dtype
    s = polars.Series("value", [], dtype=dtype)
    arr = s.to_arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    arrow_type = arr.type

    arrow_field = pa.field("value", arrow_type, nullable=True)
    spark_field = arrow_field_to_spark_field(arrow_field, opts)
    return spark_field.dataType


# ---------------------------------------------------------------------------
# Spark StructField <-> Polars Field via Arrow
# ---------------------------------------------------------------------------


@spark_polars_converter(SparkStructField, PolarsField)
def spark_field_to_polars_field(
    field: "pyspark.sql.types.StructField",
    cast_options: Optional[ArrowCastOptions] = None,
) -> "polars.datatypes.Field":
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

    opts = ArrowCastOptions.check_arg(cast_options)

    arrow_field = spark_field_to_arrow_field(field, opts)
    pl_dtype = arrow_type_to_polars_type(arrow_field.type, opts)

    # Polars Field does not encode nullability explicitly; dtype will be nullable by default
    return PolarsField(arrow_field.name, pl_dtype)


@spark_polars_converter(PolarsField, SparkStructField)
def polars_field_to_spark_field(
    field: "polars.datatypes.Field",
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pyspark.sql.types.StructField":
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

    opts = ArrowCastOptions.check_arg(cast_options)

    # field.dtype is a Polars DataType
    pl_dtype = field.dtype

    s = polars.Series(field.name, [], dtype=pl_dtype)
    arr = s.to_arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    arrow_type = arr.type

    # We default nullable=True; if you want strict nullability you can extend this
    arrow_field = pa.field(field.name, arrow_type, nullable=True)
    spark_field = arrow_field_to_spark_field(arrow_field, opts)
    return spark_field
