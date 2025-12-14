from typing import Optional, Tuple, List

import pyarrow as pa

from .arrow_cast import (
    cast_arrow_tabular,
    record_batch_reader_to_table,
    record_batch_to_table,
    arrow_field_to_schema,
)
from .cast_options import (
    CastOptions,
)
from .registry import register_converter
from ..python_defaults import default_arrow_scalar, default_python_scalar
from ...libs.sparklib import (
    pyspark,
    arrow_field_to_spark_field,
    spark_field_to_arrow_field,
    arrow_type_to_spark_type,
    spark_type_to_arrow_type,
)

__all__ = [
    "cast_spark_dataframe",
    "cast_spark_column",
    "spark_dataframe_to_arrow_table",
    "spark_dataframe_to_record_batch_reader",
    "arrow_table_to_spark_dataframe",
    "arrow_record_batch_to_spark_dataframe",
    "arrow_record_batch_reader_to_spark_dataframe",
    "spark_schema_to_arrow_schema",
    "arrow_schema_to_spark_schema",
    "arrow_field_to_spark_field",
    "spark_field_to_arrow_field",
    "arrow_type_to_spark_type",
    "spark_type_to_arrow_type",
    "SparkDataFrame",
    "SparkColumn",
    "SparkSchema",
    "SparkDataType",
    "SparkStructField",
]

# ---------------------------------------------------------------------------
# Spark type aliases + decorator wrapper (safe when pyspark is missing)
# ---------------------------------------------------------------------------

if pyspark is not None:
    import pyspark.sql.types as T  # type: ignore[import]
    from pyspark.sql import functions as F  # type: ignore[import]

    SparkDataFrame = pyspark.sql.DataFrame
    SparkColumn = pyspark.sql.Column
    SparkSchema = T.StructType
    SparkDataType = T.DataType
    SparkStructField = T.StructField

    def spark_converter(*args, **kwargs):
        return register_converter(*args, **kwargs)

else:  # pyspark missing -> dummies + no-op decorator
    class _SparkDummy:  # pragma: no cover
        pass

    SparkDataFrame = _SparkDummy
    SparkColumn = _SparkDummy
    SparkSchema = _SparkDummy
    SparkDataType = _SparkDummy
    SparkStructField = _SparkDummy

    def spark_converter(*_args, **_kwargs):  # pragma: no cover
        def _decorator(func):
            return func

        return _decorator


# ---------------------------------------------------------------------------
# Spark DF / Column <-> Arrow using ArrowCastOptions
# ---------------------------------------------------------------------------

@spark_converter(SparkDataFrame, SparkDataFrame)
def cast_spark_dataframe(
    dataframe: "pyspark.sql.DataFrame",
    options: Optional[CastOptions] = None,
) -> "pyspark.sql.DataFrame":
    """
    Cast a Spark DataFrame using Arrow *types* but without collecting to Arrow.

    - `options` is normalized via ArrowCastOptions.check_arg.
      It can be:
        * ArrowCastOptions
        * dict (if ArrowCastOptions.from_dict exists)
        * pa.Schema / pa.Field / pa.DataType
        * None  -> no-op
    - Only schema / type info is used; values stay in Spark.
    - For non-nullable target fields, nulls are filled with a default:
        * `default_value` if passed
        * otherwise `default_from_arrow_hint(field.type)`
    """
    options = CastOptions.check_arg(options)
    sub_target_arrow_schema = options.target_arrow_schema

    # No target -> nothing to do
    if sub_target_arrow_schema is None:
        return dataframe

    source_spark_fields = dataframe.schema
    source_arrow_fields = [spark_field_to_arrow_field(f) for f in source_spark_fields]

    target_arrow_fields: list[pa.Field] = list(sub_target_arrow_schema)
    child_target_spark_fields = [arrow_field_to_spark_field(f) for f in target_arrow_fields]
    target_spark_schema = arrow_schema_to_spark_schema(sub_target_arrow_schema)

    source_name_to_index = {
        field.name: idx for idx, field in enumerate(source_arrow_fields)
    }

    if not options.strict_match_names:
        source_name_to_index.update({
            field.name.casefold(): idx for idx, field in enumerate(source_arrow_fields)
        })

    casted_columns: List[Tuple[SparkStructField, SparkColumn]] = []
    found_source_names = set()

    for sub_target_index, child_target_spark_field in enumerate(child_target_spark_fields):
        child_target_arrow_field = target_arrow_fields[sub_target_index]

        find_name = child_target_spark_field.name if options.strict_match_names else child_target_spark_field.name.casefold()
        source_index = source_name_to_index.get(find_name)

        if source_index is None:
            if not options.add_missing_columns:
                raise ValueError(f"Missing column {child_target_spark_field!r} in source data {child_target_spark_fields!r}")

            dv = default_arrow_scalar(dtype=child_target_arrow_field.type, nullable=child_target_arrow_field.nullable)

            casted_column = F.lit(dv.as_py()).cast(child_target_spark_field.dataType)
        else:
            child_source_arrow_field = source_arrow_fields[sub_target_index]
            child_source_spark_field = source_spark_fields[sub_target_index]
            found_source_names.add(child_source_spark_field.name)

            casted_column = cast_spark_column(
                dataframe[source_index],
                options=options.copy(
                    source_arrow_field=child_source_arrow_field,
                    target_arrow_field=child_target_arrow_field
                )
            )

        casted_columns.append(
            (child_target_spark_field, casted_column)
        )

    if options.allow_add_columns:
        extra_columns = [
            f.name for f in source_spark_fields
            if f.name not in found_source_names
        ]

        if extra_columns:
            for extra_column_name in extra_columns:
                casted_columns.append(
                    (source_spark_fields[extra_column_name], dataframe[extra_column_name])
                )

    result = dataframe.select(*[c for _, c in casted_columns])

    return result.sparkSession.createDataFrame(result.rdd, schema=target_spark_schema)


@spark_converter(SparkColumn, SparkColumn)
def cast_spark_column(
    column: "pyspark.sql.Column",
    options: Optional[CastOptions] = None,
) -> "pyspark.sql.Column":
    """
    Cast a single Spark Column using an Arrow target *type*.

    `options` is interpreted via ArrowCastOptions.check_arg, and only the
    target_field is used. Supports:
      - pa.DataType
      - pa.Field
      - pa.Schema (we use the first field)
      - ArrowCastOptions (with target_field set)

    This is a pure Spark cast: no collection or Arrow arrays involved.
    """
    options = CastOptions.check_arg(options)
    target_spark_field = options.target_spark_field()

    if target_spark_field is None:
        # Nothing to cast to, return the column as-is
        return column

    target_spark_type = target_spark_field.dataType
    source_field = options.source_spark_field()

    if isinstance(target_spark_type, T.StructType):
        casted = cast_spark_column_to_struct(column, options=options)
    elif isinstance(target_spark_type, T.ArrayType):
        casted = cast_spark_column_to_list(column, options=options)
    elif isinstance(target_spark_type, T.MapType):
        casted = cast_spark_column_to_map(column, options=options)
    else:
        casted = column.cast(target_spark_type)

    return (
        check_column_nullability(
            casted,
            source_field=source_field,
            target_field=target_spark_field,
            mask=column.isNull()
        )
        .alias(target_spark_field.name)
    )


def check_column_nullability(
    column: "pyspark.sql.Column",
    source_field: "T.StructField",
    target_field: "T.StructField",
    mask: "pyspark.sql.Column"
) -> "pyspark.sql.Column":
    source_nullable = True if source_field is None else source_field.nullable
    target_nullable = True if target_field is None else target_field.nullable

    if source_nullable and not target_nullable:
        dv = default_python_scalar(target_field).as_py()

        column = F.when(mask, F.lit(dv)).otherwise(column)

    return column


def cast_spark_column_to_list(
    column: "pyspark.sql.Column",
    options: Optional[CastOptions] = None,
) -> "pyspark.sql.Column":
    """
    Cast a Spark Column to an ArrayType using Arrow field type info.

    This is a pure Spark cast: no collection or Arrow arrays involved.
    """
    options = CastOptions.check_arg(options)

    target_arrow_field = options.target_field
    target_spark_field = options.target_spark_field()

    if target_arrow_field is None:
        # No target type info, just pass through
        return column

    target_spark_type: T.ArrayType = target_spark_field.dataType

    source_spark_field = options.source_spark_field() or T.StructField(
        name=column.getAlias() or "col",
        dataType=column.dtype,
        nullable=True,
    )
    source_spark_type = source_spark_field.dataType

    if not isinstance(source_spark_type, T.ArrayType):
        raise ValueError(f"Cannot cast {source_spark_field} to {target_spark_field}")

    # ---- Arrow element fields ----
    target_list_type = target_arrow_field.type

    # Handle list / large_list / fixed_size_list
    if pa.types.is_list(target_list_type) or pa.types.is_large_list(target_list_type):
        target_element_arrow_field: pa.Field = target_list_type.value_field
    elif pa.types.is_fixed_size_list(target_list_type):
        target_element_arrow_field: pa.Field = target_list_type.value_field
    else:
        raise ValueError(
            f"Expected list-like Arrow type for {target_arrow_field}, "
            f"got {target_list_type}"
        )

    # Build a synthetic "element field" for the source array
    source_element_spark_field = T.StructField(
        name=source_spark_field.name,
        dataType=source_spark_type.elementType,
        nullable=source_spark_type.containsNull,
    )
    source_element_arrow_field = spark_field_to_arrow_field(
        source_element_spark_field
    )

    # Options for casting individual elements
    element_cast_options = options.copy(
        source_arrow_field=source_element_arrow_field,
        target_arrow_field=target_element_arrow_field,
    )

    # Cast each element using the same Arrow-aware machinery
    casted = F.transform(
        column,
        lambda x: cast_spark_column(x, options=element_cast_options),
    )

    # Final cast to enforce the exact Spark ArrayType (element type + containsNull)
    casted = casted.cast(target_spark_type)

    return casted


def cast_spark_column_to_struct(
    column: "pyspark.sql.Column",
    options: Optional[CastOptions] = None,
) -> "pyspark.sql.Column":
    """
    Cast a Spark Column to a StructType using Arrow field type info.

    This is a pure Spark cast: no collection or Arrow arrays involved.
    """
    options = CastOptions.check_arg(options)

    target_arrow_field = options.target_field
    target_spark_field = options.target_spark_field()

    if target_arrow_field is None:
        return column

    target_spark_type: T.StructType = target_spark_field.dataType

    source_spark_field = options.source_spark_field() or T.StructField(
        name=column.getAlias() or "col", dataType=column.dtype, nullable=True
    )
    source_spark_type: T.StructType = source_spark_field.dataType

    if not isinstance(source_spark_type, T.StructType):
        raise ValueError(f"Cannot cast {source_spark_field} to {target_spark_field}")

    source_spark_fields = list(source_spark_type.fields)
    source_arrow_fields = [spark_field_to_arrow_field(f) for f in source_spark_fields]

    target_arrow_fields: list[pa.Field] = list(options.target_field.type)
    target_spark_fields = list(target_spark_type.fields)

    name_to_index = {f.name: idx for idx, f in enumerate(source_spark_fields)}
    if not options.strict_match_names:
        name_to_index.update({
            f.name.casefold(): idx for idx, f in enumerate(source_spark_fields)
        })

    children = []
    found_source_names = set()

    for child_target_index, child_target_spark_field in enumerate(target_spark_fields):
        child_target_arrow_field: pa.Field = target_arrow_fields[child_target_index]

        find_name = child_target_spark_field.name if options.strict_match_names else child_target_spark_field.name.casefold()
        source_index = name_to_index.get(find_name)

        if source_index is None:
            if not options.add_missing_columns:
                raise ValueError(f"Missing column {child_target_arrow_field!r} from {target_arrow_fields}")

            dv = default_arrow_scalar(dtype=child_target_arrow_field.type, nullable=child_target_arrow_field.nullable)

            casted_column = F.lit(dv.as_py()).cast(child_target_spark_field.dataType)
        else:
            child_source_arrow_field = source_arrow_fields[child_target_index]
            child_source_spark_field = source_spark_fields[child_target_index]
            found_source_names.add(child_source_spark_field.name)

            casted_column = cast_spark_column(
                column.getField(child_source_arrow_field.name),
                options=options.copy(
                    source_arrow_field=child_source_arrow_field,
                    target_arrow_field=child_target_arrow_field
                )
            )

        children.append(casted_column.alias(child_target_spark_field.name))

    return F.struct(*children)


def cast_spark_column_to_map(
    column: "pyspark.sql.Column",
    options: Optional[CastOptions] = None,
) -> "pyspark.sql.Column":
    """
    Cast a Spark Column to a MapType using Arrow field type info.

    This is a pure Spark cast: no collection or Arrow arrays involved.
    """
    options = CastOptions.check_arg(options)

    target_arrow_field = options.target_field
    target_spark_field = options.target_spark_field()

    if target_arrow_field is None:
        return column

    target_spark_type: T.MapType = target_spark_field.dataType

    source_spark_field = options.source_spark_field() or T.StructField(
        name=column.getAlias() or "col",
        dataType=column.dtype,
        nullable=True,
    )
    source_spark_type = source_spark_field.dataType

    if not isinstance(source_spark_type, T.MapType):
        raise ValueError(f"Cannot cast {source_spark_field} to {target_spark_field}")

    # ---------- Arrow key/value fields ----------
    target_map_type = target_arrow_field.type
    if not pa.types.is_map(target_map_type):
        raise ValueError(
            f"Expected Arrow map type for {target_arrow_field}, got {target_map_type}"
        )

    target_key_arrow_field: pa.Field = target_map_type.key_field
    target_value_arrow_field: pa.Field = target_map_type.item_field

    # ---------- Spark key/value fields ----------
    source_key_spark_field = T.StructField(
        name=f"{source_spark_field.name}_key",
        dataType=source_spark_type.keyType,
        nullable=False,  # Spark map keys are non-null
    )
    source_value_spark_field = T.StructField(
        name=f"{source_spark_field.name}_value",
        dataType=source_spark_type.valueType,
        nullable=source_spark_type.valueContainsNull,
    )

    source_key_arrow_field = spark_field_to_arrow_field(source_key_spark_field)
    source_value_arrow_field = spark_field_to_arrow_field(source_value_spark_field)

    # ---------- Cast options for key/value ----------
    key_cast_options = options.copy(
        source_arrow_field=source_key_arrow_field,
        target_arrow_field=target_key_arrow_field,
    )
    value_cast_options = options.copy(
        source_arrow_field=source_value_arrow_field,
        target_arrow_field=target_value_arrow_field,
    )

    # ---------- Transform entries ----------
    entries = F.map_entries(column)  # array<struct<key, value>>

    casted_entries = F.transform(
        entries,
        lambda entry: F.struct(
            cast_spark_column(entry["key"], options=key_cast_options).alias("key"),
            cast_spark_column(entry["value"], options=value_cast_options).alias("value"),
        ),
    )

    casted_map = F.map_from_entries(casted_entries)

    # Enforce exact target MapType (keyType, valueType, valueContainsNull)
    casted_map = casted_map.cast(target_spark_type)

    return casted_map


# ---------------------------------------------------------------------------
# Spark DataFrame <-> Arrow Table / RecordBatchReader
# ---------------------------------------------------------------------------

@spark_converter(SparkDataFrame, pa.Table)
def spark_dataframe_to_arrow_table(
    dataframe: "pyspark.sql.DataFrame",
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """Convert a Spark DataFrame to a pyarrow.Table.

    If ``options.target_schema`` is provided, the DataFrame is first cast
    using :func:`cast_spark_dataframe` before conversion. The resulting Arrow
    schema is derived from the cast target schema when available; otherwise it
    is inferred from the Spark schema via :func:`spark_field_to_arrow_field`.
    """
    if pyspark is None:
        raise RuntimeError("pyspark is required to convert Spark to Arrow")

    opts = CastOptions.check_arg(options)

    if opts.target_arrow_schema is not None:
        dataframe = cast_spark_dataframe(dataframe, opts)
        arrow_schema = opts.target_arrow_schema
    else:
        arrow_schema = pa.schema(
            [spark_field_to_arrow_field(f, options) for f in dataframe.schema]
        )

    return cast_arrow_tabular(dataframe.toArrow(), CastOptions.check_arg(arrow_schema))


@spark_converter(SparkDataFrame, pa.RecordBatchReader)
def spark_dataframe_to_record_batch_reader(
    dataframe: "pyspark.sql.DataFrame",
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """Convert a Spark DataFrame to a pyarrow.RecordBatchReader."""
    table = spark_dataframe_to_arrow_table(dataframe, options)
    batches = table.to_batches()
    return pa.RecordBatchReader.from_batches(table.schema, batches) # type: ignore[attr-defined]


@spark_converter(pa.Table, SparkDataFrame)
def arrow_table_to_spark_dataframe(
    table: pa.Table,
    options: Optional[CastOptions] = None,
) -> "pyspark.sql.DataFrame":
    """Convert a pyarrow.Table to a Spark DataFrame.

    If a target schema is supplied, :func:`cast_arrow_table` is applied before
    creating the Spark DataFrame. Column types are derived from the Arrow
    schema using :func:`arrow_field_to_spark_field` to preserve nullability and
    metadata-driven mappings.
    """
    if pyspark is None:
        raise RuntimeError("pyspark is required to convert Arrow to Spark")

    opts = CastOptions.check_arg(options)

    if opts.target_arrow_schema is not None:
        table = cast_arrow_tabular(table, opts)

    spark = pyspark.sql.SparkSession.getActiveSession()  # type: ignore[union-attr]
    if spark is None:
        raise RuntimeError(
            "An active SparkSession is required to convert Arrow data to Spark"
        )

    spark_schema = arrow_schema_to_spark_schema(table.schema)

    return spark.createDataFrame(table, schema=spark_schema)


@spark_converter(pa.RecordBatch, SparkDataFrame)
def arrow_record_batch_to_spark_dataframe(
    batch: pa.RecordBatch,
    options: Optional[CastOptions] = None,
) -> "pyspark.sql.DataFrame":
    """Convert a pyarrow.RecordBatch to a Spark DataFrame."""
    table = record_batch_to_table(batch, options)
    return arrow_table_to_spark_dataframe(table, options)


@spark_converter(pa.RecordBatchReader, SparkDataFrame)
def arrow_record_batch_reader_to_spark_dataframe(
    reader: pa.RecordBatchReader,
    options: Optional[CastOptions] = None,
) -> "pyspark.sql.DataFrame":
    """Convert a pyarrow.RecordBatchReader to a Spark DataFrame."""
    table = record_batch_reader_to_table(reader, options)
    return arrow_table_to_spark_dataframe(table, options)


# ---------------------------------------------------------------------------
# Arrow <-> Spark type / field / schema converters (hooked into registry)
# ---------------------------------------------------------------------------
@spark_converter(SparkDataFrame, SparkDataType)
def spark_dataframe_to_spark_type(
    df: SparkDataFrame,
    options: Optional[CastOptions] = None,
) -> pa.DataType:
    return df.schema


@spark_converter(SparkDataFrame, SparkStructField)
def spark_dataframe_to_spark_field(
    df: SparkDataFrame,
    options: Optional[CastOptions] = None,
) -> pa.DataType:
    return SparkStructField(
        df.getAlias() or "root",
        df.schema,
        nullable=False,
    )


@spark_converter(SparkDataFrame, pa.Field)
def spark_dataframe_to_arrow_field(
    df: SparkDataFrame,
    options: Optional[CastOptions] = None,
) -> pa.DataType:
    return spark_field_to_arrow_field(
        spark_dataframe_to_spark_field(df, options),
        options
    )


@spark_converter(SparkDataFrame, pa.Schema)
def spark_dataframe_to_arrow_schema(
    df: SparkDataFrame,
    options: Optional[CastOptions] = None,
) -> pa.DataType:
    return arrow_field_to_schema(
        spark_field_to_arrow_field(
            spark_dataframe_to_spark_field(df, options),
            options
        ),
        options
    )


@spark_converter(pa.DataType, SparkDataType)
def _arrow_type_to_spark_type_for_registry(
    dtype: pa.DataType,
    options: Optional[CastOptions] = None,
) -> "T.DataType":  # type: ignore[name-defined]
    """
    Registry wrapper: pyarrow.DataType -> pyspark.sql.types.DataType
    """
    return arrow_type_to_spark_type(dtype, options)


@spark_converter(pa.Field, SparkStructField)
def _arrow_field_to_spark_field_for_registry(
    field: pa.Field,
    options: Optional[CastOptions] = None,
) -> "T.StructField":  # type: ignore[name-defined]
    """
    Registry wrapper: pyarrow.Field -> pyspark.sql.types.StructField
    """
    return arrow_field_to_spark_field(field, options)


@spark_converter(SparkDataType, pa.DataType)
def _spark_type_to_arrow_type_for_registry(
    dtype: "T.DataType",  # type: ignore[name-defined]
    options: Optional[CastOptions] = None,
) -> pa.DataType:
    """
    Registry wrapper: pyspark.sql.types.DataType -> pyarrow.DataType
    """
    return spark_type_to_arrow_type(dtype, options)


@spark_converter(SparkStructField, pa.Field)
def _spark_field_to_arrow_field_for_registry(
    field: "T.StructField",  # type: ignore[name-defined]
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """
    Registry wrapper: pyspark.sql.types.StructField -> pyarrow.Field
    """
    return spark_field_to_arrow_field(field, options)


@spark_converter(SparkSchema, pa.Schema)
def spark_schema_to_arrow_schema(
    schema: "T.StructType",  # type: ignore[name-defined]
    options: Optional[CastOptions] = None,
) -> pa.Schema:
    """
    Convert a pyspark StructType to a pyarrow.Schema.
    """
    opts = CastOptions.check_arg(options)
    arrow_fields = [
        spark_field_to_arrow_field(field, opts)
        for field in schema.fields
    ]
    return pa.schema(arrow_fields)


@spark_converter(pa.Schema, SparkSchema)
def arrow_schema_to_spark_schema(
    schema: pa.Schema,
    options: Optional[CastOptions] = None,
) -> "T.StructType":  # type: ignore[name-defined]
    """
    Convert a pyarrow.Schema to a pyspark StructType.
    """
    opts = CastOptions.check_arg(options)
    spark_fields = [
        arrow_field_to_spark_field(field, opts)
        for field in schema
    ]
    return T.StructType(spark_fields)
