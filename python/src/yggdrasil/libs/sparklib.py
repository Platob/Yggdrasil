from typing import Any

import pyarrow as pa

try:
    import pyspark  # type: ignore
    from pyspark.sql import SparkSession, DataFrame, Column
    import pyspark.sql.types as T

    pyspark = pyspark
    SparkSession = SparkSession
    SparkDataFrame = DataFrame
    SparkColumn = Column
    SparkDataType = T.DataType

    # Primitive Arrow -> Spark mappings
    ARROW_TO_SPARK = {
        pa.null(): T.NullType(),
        pa.bool_(): T.BooleanType(),

        pa.int8(): T.ByteType(),
        pa.int16(): T.ShortType(),
        pa.int32(): T.IntegerType(),
        pa.int64(): T.LongType(),

        # Spark has no unsigned; best effort widen
        pa.uint8(): T.ShortType(),
        pa.uint16(): T.IntegerType(),
        pa.uint32(): T.LongType(),
        pa.uint64(): T.LongType(),  # could also be DecimalType, but this is simpler

        pa.float16(): T.FloatType(),  # best-effort
        pa.float32(): T.FloatType(),
        pa.float64(): T.DoubleType(),

        pa.string(): T.StringType(),
        pa.string_view(): T.StringType(),
        pa.large_string(): T.StringType(),

        pa.binary(): T.BinaryType(),
        pa.binary_view(): T.BinaryType(),
        pa.large_binary(): T.BinaryType(),

        pa.date32(): T.DateType(),
        pa.date64(): T.DateType(),  # drop time-of-day

        # Timestamp with any unit → TimestampType (Spark is microsecond-resolution)
        pa.timestamp("us", "UTC"): T.TimestampType(),
    }
except ImportError:  # pragma: no cover - Spark not available
    pyspark = None
    SparkSession = None
    SparkDataFrame = None
    SparkColumn = None
    SparkDataType = None
    ARROW_TO_SPARK = {}


# Primitive Spark -> Arrow mapping (only for the types in ARROW_TO_SPARK)
SPARK_TO_ARROW = {v: k for k, v in ARROW_TO_SPARK.items()}


__all__ = [
    "pyspark",
    "require_pyspark",
    "SparkSession",
    "SparkDataFrame",
    "SparkColumn",
    "SparkDataType",
    "ARROW_TO_SPARK",
    "SPARK_TO_ARROW",
    "arrow_type_to_spark_type",
    "arrow_field_to_spark_field",
    "spark_type_to_arrow_type",
    "spark_field_to_arrow_field",
]


def require_pyspark(active_session: bool = False):
    """
    Optionally enforce that pyspark (and an active SparkSession) exists.
    """
    if pyspark is None:
        raise ImportError(
            "pyspark is required to use this function. "
            "Install it or run inside a Spark/Databricks environment."
        )

    if active_session:
        if SparkSession is None:
            raise ImportError(
                "pyspark.sql.SparkSession is required to check for an active session."
            )
        if SparkSession.getActiveSession() is None:
            raise RuntimeError(
                "An active SparkSession is required to use this function. "
                "Create one with SparkSession.builder.getOrCreate()."
            )


def arrow_type_to_spark_type(
    arrow_type: pa.DataType,
    cast_options: Any = None,
) -> "T.DataType":
    """
    Convert a pyarrow.DataType to a pyspark.sql.types.DataType.
    """
    require_pyspark()

    import pyarrow.types as pat

    # Fast path: exact mapping hit
    spark_type = ARROW_TO_SPARK.get(arrow_type)
    if spark_type is not None:
        return spark_type

    # Decimal
    if pat.is_decimal(arrow_type):
        return T.DecimalType(precision=arrow_type.precision, scale=arrow_type.scale)

    # Timestamp
    if pat.is_timestamp(arrow_type):
        tz = getattr(arrow_type, "tz", None)
        if tz:
            return T.TimestampType()
        return T.TimestampNTZType()

    # List / LargeList
    if pat.is_list(arrow_type) or pat.is_large_list(arrow_type):
        element_arrow = arrow_type.value_type
        element_spark = arrow_type_to_spark_type(element_arrow, cast_options)
        return T.ArrayType(elementType=element_spark, containsNull=True)

    # Fixed-size list -> treat as an array in Spark
    if pat.is_fixed_size_list(arrow_type):
        element_arrow = arrow_type.value_type
        element_spark = arrow_type_to_spark_type(element_arrow, cast_options)
        return T.ArrayType(elementType=element_spark, containsNull=True)

    # Struct
    if pat.is_struct(arrow_type):
        fields = [arrow_field_to_spark_field(f, cast_options) for f in arrow_type]
        return T.StructType(fields)

    # Map -> MapType(keyType, valueType)
    if pat.is_map(arrow_type):
        key_arrow = arrow_type.key_type
        item_arrow = arrow_type.item_type
        key_spark = arrow_type_to_spark_type(key_arrow, cast_options)
        value_spark = arrow_type_to_spark_type(item_arrow, cast_options)
        return T.MapType(
            keyType=key_spark,
            valueType=value_spark,
            valueContainsNull=True,
        )

    # Duration -> best-effort: store as LongType
    if pat.is_duration(arrow_type):
        return T.LongType()

    # Fallback numeric: widen to Long/Double
    if pat.is_integer(arrow_type):
        return T.LongType()
    if pat.is_floating(arrow_type):
        return T.DoubleType()

    # Binary / String families
    if pat.is_binary(arrow_type) or pat.is_large_binary(arrow_type):
        return T.BinaryType()
    if pat.is_string(arrow_type) or pat.is_large_string(arrow_type):
        return T.StringType()

    raise TypeError(f"Unsupported or unknown Arrow type for Spark conversion: {arrow_type!r}")


def arrow_field_to_spark_field(
    field: pa.Field,
    cast_options: Any = None,
) -> "T.StructField":
    """
    Convert a pyarrow.Field to a pyspark StructField.
    """
    spark_type = arrow_type_to_spark_type(field.type, cast_options)

    return T.StructField(
        name=field.name,
        dataType=spark_type,
        nullable=field.nullable,
        metadata={},
    )


def spark_type_to_arrow_type(
    spark_type: "T.DataType",
    cast_options: Any = None,
) -> pa.DataType:
    """
    Convert a pyspark.sql.types.DataType to a pyarrow.DataType.
    """
    require_pyspark()
    from pyspark.sql.types import (
        BooleanType,
        ByteType,
        ShortType,
        IntegerType,
        LongType,
        FloatType,
        DoubleType,
        StringType,
        BinaryType,
        DateType,
        TimestampType,
        TimestampNTZType,
        DecimalType,
        ArrayType,
        MapType,
        StructType,
    )

    # Primitive types
    if isinstance(spark_type, BooleanType):
        return pa.bool_()
    if isinstance(spark_type, ByteType):
        return pa.int8()
    if isinstance(spark_type, ShortType):
        return pa.int16()
    if isinstance(spark_type, IntegerType):
        return pa.int32()
    if isinstance(spark_type, LongType):
        return pa.int64()
    if isinstance(spark_type, FloatType):
        return pa.float32()
    if isinstance(spark_type, DoubleType):
        return pa.float64()
    if isinstance(spark_type, StringType):
        return pa.string()
    if isinstance(spark_type, BinaryType):
        return pa.binary()
    if isinstance(spark_type, DateType):
        return pa.date32()
    if isinstance(spark_type, TimestampType):
        return pa.timestamp("us", "UTC")
    if isinstance(spark_type, TimestampNTZType):
        return pa.timestamp("us")

    # DecimalType
    if isinstance(spark_type, DecimalType):
        return pa.decimal128(spark_type.precision, spark_type.scale)

    # ArrayType
    if isinstance(spark_type, ArrayType):
        element_arrow = spark_type_to_arrow_type(spark_type.elementType, cast_options)
        return pa.list_(element_arrow)

    # MapType
    if isinstance(spark_type, MapType):
        key_arrow = spark_type_to_arrow_type(spark_type.keyType, cast_options)
        value_arrow = spark_type_to_arrow_type(spark_type.valueType, cast_options)
        return pa.map_(key_arrow, value_arrow)

    # StructType
    if isinstance(spark_type, StructType):
        arrow_fields = [
            spark_field_to_arrow_field(f, cast_options)
            for f in spark_type.fields
        ]
        return pa.struct(arrow_fields)

    raise TypeError(f"Unsupported or unknown Spark type for Arrow conversion: {spark_type!r}")


def spark_field_to_arrow_field(
    field: "T.StructField",
    cast_options: Any = None,
) -> pa.Field:
    """
    Convert a pyspark StructField to a pyarrow.Field.
    """
    arrow_type = spark_type_to_arrow_type(field.dataType, cast_options)
    return pa.field(
        name=field.name,
        type=arrow_type,
        nullable=field.nullable,
    )
