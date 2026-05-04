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

from yggdrasil.arrow.cast import (
    any_to_arrow_table,
    rechunk_arrow_batches_by_byte_size,
)
from yggdrasil.data.cast import register_converter
from yggdrasil.data.options import CastOptions
from yggdrasil.environ import PyEnv

__all__ = [
    "any_to_spark_field",
    "any_to_spark_schema",
    "any_to_spark_dataframe",
    "cast_spark_column",
    "cast_spark_dataframe",
]

#: Cap on per-batch Arrow size handed to ``createDataFrame``. 128 MiB matches
#: Spark's preferred Arrow batch size and keeps a single oversized chunk from
#: pinning a partition's working memory.
_SPARK_ARROW_BATCH_BYTE_LIMIT: int = 128 * 1024 * 1024


@register_converter(Any, T.StructField)
def any_to_spark_field(
    field: pa.Field,
    options: Optional[CastOptions] = None,
) -> T.StructField:
    return options.check_source(field).merged_field.to_pyspark_field()


@register_converter(Any, T.StructType)
def any_to_spark_schema(
    schema: pa.Schema,
    options: Optional[CastOptions] = None,
) -> T.StructType:
    return options.check_source(schema).merged_schema.to_spark_schema()


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
    opts = CastOptions.check(options)

    if isinstance(obj, pyspark_sql.DataFrame):
        return opts.cast_spark_tabular(obj)

    spark = PyEnv.spark_session(
        create=True,
        import_error=True,
        install_spark=False,
    )

    if obj is None:
        return spark.createDataFrame([], schema=opts.merged_schema.to_spark_schema())

    arrow_table = any_to_arrow_table(obj, options=opts)
    rechunked = list(rechunk_arrow_batches_by_byte_size(
        arrow_table.to_batches(),
        byte_size=_SPARK_ARROW_BATCH_BYTE_LIMIT,
        memory_pool=opts.arrow_memory_pool,
    ))
    arrow_table = pa.Table.from_batches(rechunked, schema=arrow_table.schema)
    df = spark.createDataFrame(arrow_table, schema=opts.merged_schema.to_spark_schema())
    return opts.cast_spark(df)
