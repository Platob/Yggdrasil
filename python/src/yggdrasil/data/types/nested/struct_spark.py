"""Spark cast helpers for :class:`StructType` targets.

Spark casts lower to :class:`pyspark.sql.Column` expressions —
projection-time, no row-by-row work. The cast becomes part of the
physical plan, fused with any pushdown filters and the final
``select(*cols)``.

* :func:`cast_spark_struct_column` — struct → struct via
  ``F.struct(...)``, with a null-preservation guard
  (``F.when(col.isNull(), F.lit(None)).otherwise(...)``) because
  ``F.struct`` always returns a non-null row.
* :func:`cast_spark_map_column` — map → struct via
  ``F.element_at(col, F.lit(name))`` per child.
* :func:`cast_spark_list_column` — list → struct via
  ``F.get(col, F.lit(i))`` (null-on-OOB; plain ``col[i]`` raises
  ``ArrayIndexOutOfBoundsException`` in Spark 4.x).
* :func:`cast_spark_tabular` — DataFrame select with one cast Column
  per target field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.support import get_spark_sql

if TYPE_CHECKING:
    import pyspark.sql as psql
    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.data_field import Field
    from .map import MapType
    from .struct import StructType


__all__ = [
    "cast_spark_struct_column",
    "cast_spark_map_column",
    "cast_spark_list_column",
    "cast_spark_tabular",
]


def cast_spark_struct_column(
    column: Any,
    options: "CastOptions",
) -> Any:
    spark = get_spark_sql()
    F = spark.functions

    if not options.need_cast(column):
        return column

    if options.source.dtype.type_id != DataTypeId.STRUCT:
        raise TypeError(f"Cannot cast {options.source} to {options.target}")

    source_field: "Field" = options.source
    source_type: "StructType" = source_field.dtype
    target_type: "StructType" = options.target.dtype

    child_columns: list[Any] = []

    for i, target_child in enumerate(target_type.children):
        source_child = source_type.field_by(
            name=target_child.name,
            index=i,
            raise_error=False,
        )

        if source_child is None:
            child = target_child.default_spark_column(alias=target_child.name)
        else:
            child = target_child.cast_spark_column(
                column[source_child.name],
                options=options.copy(
                    source=source_child,
                    target=target_child,
                ),
            ).alias(target_child.name)

        child_columns.append(child)

    # Preserve null source rows: F.struct always returns a non-null struct,
    # so without this guard a null row becomes {child: None, ...}.
    return F.when(column.isNull(), F.lit(None)).otherwise(F.struct(*child_columns))


def cast_spark_map_column(
    column: Any,
    options: "CastOptions",
) -> Any:
    spark = get_spark_sql()
    F = spark.functions

    if not options.need_cast(column):
        return column

    if options.source.dtype.type_id != DataTypeId.MAP:
        raise TypeError(f"Cannot cast {options.source} to {options.target}")

    source_field: "Field" = options.source
    source_type: "MapType" = source_field.dtype
    target_type: "StructType" = options.target.dtype

    child_columns: list[Any] = []

    for target_child in target_type.children:
        extracted = F.element_at(column, F.lit(target_child.name))

        casted = target_child.cast_spark_column(
            extracted,
            options=options.copy(
                source=source_type.value_field,
                target=target_child,
            ),
        ).alias(target_child.name)

        child_columns.append(casted)

    return F.when(column.isNull(), F.lit(None)).otherwise(F.struct(*child_columns))


def cast_spark_list_column(
    column: Any,
    options: "CastOptions",
) -> Any:
    spark = get_spark_sql()
    F = spark.functions

    if not options.need_cast(column):
        return column

    if options.source.dtype.type_id != DataTypeId.ARRAY:
        raise TypeError(f"Cannot cast {options.source} to {options.target}")

    source_field: "Field" = options.source
    target_type: "StructType" = options.target.dtype

    # ArrayType import is lazy via TYPE_CHECKING; we read .item_field
    # off the runtime instance below.
    source_type = source_field.dtype

    child_columns: list[Any] = []

    for i, target_child in enumerate(target_type.children):
        # F.get is the null-on-out-of-bounds accessor; plain column[i] /
        # column.getItem(i) raises an ArrayIndexOutOfBoundsException in
        # Spark 4.x when the source list is shorter than the target struct.
        extracted = F.get(column, F.lit(i))

        casted = target_child.cast_spark_column(
            extracted,
            options=options.copy(
                source=source_type.item_field,
                target=target_child,
            ),
        ).alias(target_child.name)

        child_columns.append(casted)

    return F.when(column.isNull(), F.lit(None)).otherwise(F.struct(*child_columns))


def cast_spark_tabular(
    data: "psql.DataFrame",
    options: "CastOptions",
) -> "psql.DataFrame":
    spark = get_spark_sql()

    if not isinstance(data, spark.DataFrame):
        raise TypeError(f"Unsupported tabular type: {type(data)!r}")

    if not options.need_cast(data, check_names=True):
        return data

    source_schema = options.source
    target_schema = options.merged_schema

    # Engine-level fast bypass — Field/DataType detail (semantic
    # subclass, metadata) doesn't surface in the underlying Spark
    # ``StructType``. When the source frame's Spark schema already
    # equals the target engine schema, ``DataFrame.select`` would just
    # rebuild the plan with the same columns; skip it.
    target_spark_schema = target_schema.to_spark_schema()
    if data.schema == target_spark_schema:
        return data

    cols: list[Any] = []

    for i, target_field in enumerate(target_schema.children):
        source_field = source_schema.field_by(
            name=target_field.name,
            index=i,
            raise_error=False,
        )

        if source_field is None:
            col = target_field.default_spark_column(alias=target_field.name)
        else:
            col = target_field.cast_spark_column(
                spark.functions.col(source_field.name),
                options=options.copy(
                    source=source_field,
                    target=target_field,
                ),
            ).alias(target_field.name)

        cols.append(col)

    return data.select(*cols)
