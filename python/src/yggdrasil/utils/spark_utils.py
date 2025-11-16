from __future__ import annotations

from typing import Any

import polars as pl
import pyarrow as pa
import pandas as pd

import pyspark as spark
import pyspark.sql as spark_sql
import pyspark.sql.functions as spark_functions
import pyspark.sql.types as spark_types

from yggdrasil.utils.arrow_utils import safe_arrow_tabular

ARROW_TYPE_TO_SPARK_TYPE = {
    pa.bool_(): spark_types.BooleanType(),
    pa.utf8(): spark_types.StringType(),
    pa.large_string(): spark_types.StringType(),
    pa.binary(): spark_types.BinaryType(),
    pa.large_binary(): spark_types.BinaryType(),
    pa.int8(): spark_types.ByteType(),
    pa.int16(): spark_types.IntegerType(),
    pa.int32(): spark_types.IntegerType(),
    pa.int64(): spark_types.LongType(),
    pa.float32(): spark_types.FloatType(),
    pa.float64(): spark_types.DoubleType(),
    pa.date32(): spark_types.DateType(),
    pa.date64(): spark_types.TimestampType(),
}

SPARK_TO_ARROW_TYPE_MAP = {
    spark_types.BooleanType: lambda _: pa.bool_(),
    spark_types.ByteType: lambda _: pa.int8(),
    spark_types.ShortType: lambda _: pa.int16(),
    spark_types.IntegerType: lambda _: pa.int32(),
    spark_types.LongType: lambda _: pa.int64(),
    spark_types.FloatType: lambda _: pa.float32(),
    spark_types.DoubleType: lambda _: pa.float64(),
    spark_types.StringType: lambda _: pa.utf8(),
    spark_types.BinaryType: lambda _: pa.binary(),
    spark_types.DateType: lambda _: pa.date32(),
    spark_types.TimestampType: lambda _: pa.timestamp('us', "UTC"),
    spark_types.TimestampNTZType: lambda _: pa.timestamp('us'),
    spark_types.DecimalType: lambda t: pa.decimal128(t.precision, t.scale),

    # Complex types with recursive conversion
    spark_types.ArrayType: lambda t: pa.list_(spark_to_arrow_type(t.elementType)),
    spark_types.MapType: lambda t: pa.map_(
        spark_to_arrow_type(t.keyType),
        spark_to_arrow_type(t.valueType),
    ),
    spark_types.StructType: lambda t: pa.struct([
        pa.field(
            field.name,
            spark_to_arrow_type(field.dataType),
            field.nullable,
            metadata=field.metadata
        )
        for field in t.fields
    ])
}

__all__ = [
    "ARROW_TYPE_TO_SPARK_TYPE",
    "spark", "spark_sql", "spark_types", "spark_functions",
    "spark_to_arrow_type",
    "cast_nested_spark_field",
    "safe_spark_dataframe"
]


# Monkey patch
def toPolars(self: spark_sql.DataFrame):
    arrow_table: pa.Table = self.toArrow()
    return pl.from_arrow(arrow_table)

setattr(spark_sql.DataFrame, "toPolars", toPolars)


def safe_spark_dataframe(
    obj: Any,
    spark_session: spark_sql.SparkSession | None = None
):
    if isinstance(obj, spark_sql.DataFrame):
        return obj

    from ..types import DataField

    spark_session = spark_session or spark_sql.SparkSession.getActiveSession()

    if not spark_session:
        raise RuntimeError(
            f"Cannot build spark dataframe from {type(obj)} without active spark session, create one"
        )

    if not isinstance(obj, (pa.RecordBatch, pa.Table)):
        obj = safe_arrow_tabular(obj)

    if isinstance(obj, pa.RecordBatch):
        obj = pa.Table.from_batches([obj])

    schema = DataField.from_arrow_schema(obj.schema)
    return spark_session._create_from_arrow_table(
        obj,
        schema=schema.to_spark_field().dataType,
        timezone="UTC"
    )


def spark_to_arrow_type(spark_type: spark_types.DataType):
    """Convert a Spark type to a PyArrow type.

    Args:
        spark_type: PySpark DataType

    Returns:
        A PyArrow DataType

    Raises:
        ImportError: If PySpark is not installed
        TypeError: If the Spark type cannot be converted to a PyArrow type
    """
    # Find the correct type converter using the type of spark_type
    for spark_class, converter in SPARK_TO_ARROW_TYPE_MAP.items():
        if isinstance(spark_type, spark_class):
            return converter(spark_type)

    # If we get here, no matching type was found
    raise TypeError(f"Cannot convert Spark type {spark_type} to PyArrow type")


def cast_nested_spark_field(
    column: spark_sql.Column,
    source_field: spark_types.StructField,
    target_field: spark_types.StructField,
) -> spark_sql.Column:
    """
    Recursively cast a Spark column from source_field to target_field,
    handling structs / arrays / maps in a schema-aware way.
    """
    src_type: spark_types.DataType = source_field.dataType
    tgt_type: spark_types.DataType = target_field.dataType
    casted = None

    # === STRUCT ===
    if isinstance(tgt_type, spark_types.StructType):
        if isinstance(src_type, spark_types.StructType):
            src_fields_by_name = {f.name: f for f in src_type.fields}
            field_exprs = []

            for tgt_child in tgt_type.fields:
                src_child = src_fields_by_name.get(tgt_child.name)

                if src_child is None:
                    # Missing in source → null of target field type
                    field_exprs.append(
                        spark_functions.lit(None).cast(tgt_child.dataType).alias(tgt_child.name)
                    )
                    continue

                child_col = column.getField(src_child.name)
                cast_child = cast_nested_spark_field(
                    child_col,
                    source_field=src_child,
                    target_field=tgt_child,
                ).alias(tgt_child.name)

                field_exprs.append(cast_child)

            casted = spark_functions.struct(*field_exprs)

    # === ARRAY ===
    elif isinstance(tgt_type, spark_types.ArrayType):
        if isinstance(src_type, spark_types.ArrayType):
            casted = column.cast(tgt_type)

    # === MAP ===
    elif isinstance(tgt_type, spark_types.MapType):
        if isinstance(src_type, spark_types.MapType):
            src_key_type = src_type.keyType
            src_val_type = src_type.valueType
            tgt_key_type = tgt_type.keyType
            tgt_val_type = tgt_type.valueType

            src_key_field = spark_types.StructField("key", src_key_type, nullable=False, metadata={})
            src_val_field = spark_types.StructField("value", src_val_type, nullable=True, metadata={})
            tgt_key_field = spark_types.StructField("key", tgt_key_type, nullable=False, metadata={})
            tgt_val_field = spark_types.StructField("value", tgt_val_type, nullable=True, metadata={})

            casted = spark_functions.map_from_entries(
                spark_functions.transform(
                    spark_functions.map_entries(column),
                    lambda entry: spark_functions.struct(
                        cast_nested_spark_field(
                            entry["key"],
                            source_field=src_key_field,
                            target_field=tgt_key_field,
                        ).alias("key"),
                        cast_nested_spark_field(
                            entry["value"],
                            source_field=src_val_field,
                            target_field=tgt_val_field,
                        ).alias("value"),
                    ),
                )
            )

    if not casted:
        casted = column.cast(tgt_type)

    # === PRIMITIVE / OTHER ===
    # For scalar-ish types we just let Spark handle the cast.
    return casted.alias(target_field.name)
