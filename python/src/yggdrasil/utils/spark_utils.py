from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from .fake_module import make_fake_module

if TYPE_CHECKING:
    # These are only imported for type-checkers / IDEs.
    import pyspark.sql as spark_sql
    import pyspark.sql.types as spark_types

# Conditionally import pyspark at runtime
try:
    import pyspark as spark
    import pyspark.sql as spark_sql
    import pyspark.sql.types as spark_types
    import pyspark.sql.functions as spark_functions

    HAVE_SPARK = True

    # Re-export commonly used types at module level
    StructType = spark_types.StructType
    StructField = spark_types.StructField
    ArrayType = spark_types.ArrayType
    MapType = spark_types.MapType
    ARROW_TYPE_TO_SPARK_TYPE = {
        pa.utf8(): spark_types.StringType(),
        pa.binary(): spark_types.BinaryType(),
        pa.int8(): spark_types.ByteType(),
        pa.int16(): spark_types.IntegerType(),
        pa.int32(): spark_types.IntegerType(),
        pa.int64(): spark_types.LongType(),
        pa.float32(): spark_types.FloatType(),
        pa.float64(): spark_types.DoubleType(),
        pa.date32(): spark_types.DateType(),
        pa.date64(): spark_types.TimestampType(),
        pa.decimal128(38,18): spark_types.DecimalType(38,18),
        pa.timestamp("ns"): spark_types.TimestampNTZType(),
    }

except ImportError:
    spark = make_fake_module(module_name="spark")
    spark_sql = make_fake_module(module_name="spark_sql")
    spark_types = make_fake_module(module_name="spark_types")
    spark_functions = make_fake_module(module_name="spark_functions")

    HAVE_SPARK = False

    # Keep names defined so annotations don’t explode
    StructType = None
    StructField = None
    ArrayType = None
    MapType = None
    ARROW_TYPE_TO_SPARK_TYPE = {}

__all__ = [
    "HAVE_SPARK",
    "ARROW_TYPE_TO_SPARK_TYPE",
    "spark", "spark_sql", "spark_types", "spark_functions",
    "spark_to_arrow_type",
    "cast_nested_spark_field",
]


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
    if not HAVE_SPARK:
        raise ImportError("PySpark is required for _spark_to_arrow_type. Install it with 'pip install pyspark'.")

    # Type mapping from Spark to PyArrow for atomic types
    type_mapping = {
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
        spark_types.TimestampType: lambda _: pa.timestamp('us'),
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

    # Find the correct type converter using the type of spark_type
    for spark_class, converter in type_mapping.items():
        if isinstance(spark_type, spark_class):
            return converter(spark_type)

    # If we get here, no matching type was found
    raise TypeError(f"Cannot convert Spark type {spark_type} to PyArrow type")


def cast_nested_spark_field(
    column: spark_sql.Column,
    source_field: StructField,
    target_field: StructField,
) -> spark_sql.Column:
    """
    Recursively cast a Spark column from source_field to target_field,
    handling structs / arrays / maps in a schema-aware way.
    """
    src_type: spark_types.DataType = source_field.dataType
    tgt_type: spark_types.DataType = target_field.dataType
    casted = None

    # === STRUCT ===
    if isinstance(tgt_type, StructType):
        if isinstance(src_type, StructType):
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
    elif isinstance(tgt_type, ArrayType):
        if isinstance(src_type, ArrayType):
            casted = column.cast(tgt_type)

    # === MAP ===
    elif isinstance(tgt_type, MapType):
        if isinstance(src_type, MapType):
            src_key_type = src_type.keyType
            src_val_type = src_type.valueType
            tgt_key_type = tgt_type.keyType
            tgt_val_type = tgt_type.valueType

            src_key_field = StructField("key", src_key_type, nullable=False, metadata={})
            src_val_field = StructField("value", src_val_type, nullable=True, metadata={})
            tgt_key_field = StructField("key", tgt_key_type, nullable=False, metadata={})
            tgt_val_field = StructField("value", tgt_val_type, nullable=True, metadata={})

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
