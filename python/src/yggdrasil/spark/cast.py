"""Spark <-> Arrow casting helpers and converters.

This module provides bidirectional type-mapping and data-casting between
Apache Spark (PySpark) and Apache Arrow (PyArrow).  Every public function
accepts an optional ``CastOptions`` argument that carries the target schema
**and** a ``safe`` flag:

- ``safe=True``  (default)  – best-effort: invalid casts become ``null`` /
  empty values; missing columns are filled with type-appropriate defaults.
- ``safe=False`` – strict: any type mismatch or missing column raises
  immediately.

The module is structured in three layers:

1. **Type converters** – stateless, pure functions that map Arrow ↔ Spark
   type objects without touching any data.
2. **Field / schema converters** – wrap type converters to also carry name,
   nullability, and metadata.
3. **Data converters** – operate on live Spark DataFrames / Columns using the
   type information from layers 1-2.

JSON decoding for compound targets
-----------------------------------
When the *source* column is ``StringType`` or ``BinaryType`` and the *target*
type is a compound type (struct, array, or map), the casters automatically
attempt to parse the column as JSON using ``from_json`` **before** applying
field-level / element-level casting.  This handles the common commodity-data
pattern where nested structures are stored as JSON strings in a flat column
(e.g. a ``STRING`` column containing ``'{"bid":99.5,"ask":100.0}'``).

- ``BinaryType`` sources are decoded to UTF-8 string first via ``CAST(... AS STRING)``.
- Spark's ``from_json`` silently returns ``null`` for rows whose JSON is
  malformed or structurally incompatible — consistent with ``safe=True``
  semantics.  In ``safe=False`` mode callers can enforce non-nullability on
  the result via the standard null-fill machinery.
"""

from __future__ import annotations

from typing import Any, Optional

import pyarrow as pa
import pyspark.sql as pyspark_sql
import pyspark.sql.types as T
from yggdrasil.data import Field, Schema
from yggdrasil.data.cast import CastOptions, register_converter
from yggdrasil.environ import PyEnv
from yggdrasil.pickle.serde import ObjectSerde

__all__ = [
    "any_to_spark_field",
    "any_to_spark_schema",
    "any_to_spark_dataframe",
    "cast_spark_column",
    "cast_spark_dataframe",
]


@register_converter(Any, T.StructField)
def any_to_spark_field(
    field: pa.Field,
    options: Optional[CastOptions] = None,
) -> T.StructField:
    return Field.from_(field).to_pyspark_field()


@register_converter(Any, T.StructType)
def any_to_spark_schema(
    schema: pa.Schema,
    options: Optional[CastOptions] = None,
) -> T.StructType:
    return Schema.from_any(schema).to_spark_schema()


@register_converter(pyspark_sql.Column, pyspark_sql.Column)
def cast_spark_column(
    column: pyspark_sql.Column,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.Column:
    return CastOptions.check(options).cast_spark_column(column)


@register_converter(pyspark_sql.DataFrame, pyspark_sql.DataFrame)
def cast_spark_dataframe(
    dataframe: pyspark_sql.DataFrame,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.DataFrame:
    return CastOptions.check(options).cast_spark_tabular(dataframe)


@register_converter(Any, pyspark_sql.DataFrame)
def any_to_spark_dataframe(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.DataFrame:
    spark = PyEnv.spark_session(
        create=True,
        import_error=True,
        install_spark=False,
    )

    opts = CastOptions.check(options)

    if isinstance(obj, pyspark_sql.DataFrame):
        return opts.cast_spark_tabular(obj)

    if obj is None:
        return spark.createDataFrame([], schema=opts.target_field.to_schema().to_spark_schema())

    namespace = ObjectSerde.full_namespace(obj)

    if namespace.startswith("pyarrow"):
        if isinstance(obj, pa.RecordBatch):
            obj = pa.Table.from_batches([obj], schema=obj.schema) # type: ignore
        elif hasattr(obj, "to_table"):
            obj = obj.to_table()

        if isinstance(obj, pa.Table):
            spark_schema = any_to_spark_schema(obj.schema, None)
            df = spark.createDataFrame(obj, schema=spark_schema)
        else:
            raise ValueError(
                f"Cannot convert {type(obj)} to pyspark.sql.DataFrame"
            )
    elif namespace.startswith("yggdrasil."):
        from yggdrasil.spark.frame import DynamicFrame

        if isinstance(obj, DynamicFrame):
            schema = opts.target_schema

            df = obj.df if schema is None else obj.cast(schema=schema)
        else:
            raise ValueError(
                f"Cannot create spark dataframe from {type(obj)}"
            )
    else:
        # Route through Polars as the intermediate representation for arbitrary inputs.
        from yggdrasil.polars.cast import any_to_polars_dataframe, polars_dataframe_to_arrow_table

        arrow_table = polars_dataframe_to_arrow_table(
            any_to_polars_dataframe(obj, opts), opts
        )
        spark_schema = any_to_spark_schema(arrow_table.schema, None)
        df = spark.createDataFrame(arrow_table, schema=spark_schema)

    return opts.cast_spark_tabular(df)
