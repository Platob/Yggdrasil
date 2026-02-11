"""Spark <-> Arrow casting helpers and converters."""

from typing import Optional, Tuple, List, Any, Union

import pyarrow as pa
import pyspark.sql as SparkSQL
import pyspark.sql.functions as F
import pyspark.sql.types as T

from ..types.cast.arrow_cast import cast_arrow_tabular, arrow_field_to_schema
from ..types.cast.cast_options import CastOptions, CastOptionsArg
from ..types.cast.registry import register_converter
from ..types.python_defaults import default_arrow_scalar, default_python_scalar

__all__ = [
    "cast_spark_dataframe",
    "cast_spark_column",
    "spark_dataframe_to_arrow_table",
    "arrow_table_to_spark_dataframe",
    "spark_schema_to_arrow_schema",
    "arrow_schema_to_spark_schema",
    "arrow_field_to_spark_field",
    "spark_field_to_arrow_field",
    "arrow_type_to_spark_type",
    "spark_type_to_arrow_type",
    "any_to_spark_dataframe",
]

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
    getattr(pa, "string_view", pa.string)(): T.StringType(),
    getattr(pa, "large_string", pa.string)(): T.StringType(),

    pa.binary(): T.BinaryType(),
    getattr(pa, "binary_view", pa.binary)(): T.BinaryType(),
    getattr(pa, "large_binary", pa.binary)(): T.BinaryType(),

    pa.date32(): T.DateType(),
    pa.date64(): T.DateType(),  # drop time-of-day

    # Timestamp with any unit → TimestampType (Spark is microsecond-resolution)
    pa.timestamp("us", "UTC"): T.TimestampType(),
}


# Primitive Spark -> Arrow mapping (only for the types in ARROW_TO_SPARK)
SPARK_TO_ARROW = {v: k for k, v in ARROW_TO_SPARK.items()}

def arrow_type_to_spark_type(
    arrow_type: Union[pa.DataType, pa.Decimal128Type, pa.ListType, pa.MapType],
    cast_options: Optional[CastOptionsArg] = None,
) -> T.DataType:
    """
    Convert a pyarrow.DataType to a pyspark.sql.types.DataType.

    Args:
        arrow_type: Arrow data type to convert.
        cast_options: Optional casting options.

    Returns:
        Spark SQL data type.
    """
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
) -> T.StructField:
    """
    Convert a pyarrow.Field to a pyspark StructField.

    Args:
        field: Arrow field to convert.
        cast_options: Optional casting options.

    Returns:
        Spark StructField representation.
    """
    spark_type = arrow_type_to_spark_type(field.type, cast_options)

    return T.StructField(
        name=field.name,
        dataType=spark_type,
        nullable=field.nullable,
        metadata={},
    )


def spark_type_to_arrow_type(
    spark_type: T.DataType,
    cast_options: Any = None,
) -> pa.DataType:
    """
    Convert a pyspark.sql.types.DataType to a pyarrow.DataType.

    Args:
        spark_type: Spark SQL data type to convert.
        cast_options: Optional casting options.

    Returns:
        Arrow data type.
    """
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
    field: T.StructField,
    cast_options: Optional[CastOptions] = None,
) -> pa.Field:
    """
    Convert a pyspark StructField to a pyarrow.Field.

    Args:
        field: Spark StructField to convert.
        cast_options: Optional casting options.

    Returns:
        Arrow field.
    """
    arrow_type = spark_type_to_arrow_type(field.dataType, cast_options)

    return pa.field(
        name=field.name,
        type=arrow_type,
        nullable=field.nullable,
    )


@register_converter(SparkSQL.DataFrame, SparkSQL.DataFrame)
def cast_spark_dataframe(
    dataframe: SparkSQL.DataFrame,
    options: Optional[CastOptions] = None,
) -> SparkSQL.DataFrame:
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
    target_spark_schema = arrow_schema_to_spark_schema(sub_target_arrow_schema, None)

    source_name_to_index = {
        field.name: idx for idx, field in enumerate(source_arrow_fields)
    }

    if not options.strict_match_names:
        source_name_to_index.update({
            field.name.casefold(): idx for idx, field in enumerate(source_arrow_fields)
        })

    casted_columns: List[Tuple[T.StructField, SparkSQL.Column]] = []
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
            df_col: SparkSQL.Column = dataframe[source_index]

            casted_column = cast_spark_column(
                df_col,
                options.copy(
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


@register_converter(SparkSQL.Column, SparkSQL.Column)
def cast_spark_column(
    column: SparkSQL.Column,
    options: Optional[CastOptions] = None,
) -> SparkSQL.Column:
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
    target_spark_field = options.target_spark_field

    if target_spark_field is None:
        # Nothing to cast to, return the column as-is
        return column

    target_spark_type = target_spark_field.dataType
    source_spark_field = options.source_spark_field
    assert source_spark_field, "No source spark field found in cast options"

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
            source_field=source_spark_field,
            target_field=target_spark_field,
            mask=column.isNull()
        )
        .alias(target_spark_field.name)
    )


def check_column_nullability(
    column: SparkSQL.Column,
    source_field: T.StructField,
    target_field: T.StructField,
    mask: SparkSQL.Column
) -> SparkSQL.Column:
    """Fill nulls when the target field is non-nullable.

    Args:
        column: Spark column to adjust.
        source_field: Source Spark field.
        target_field: Target Spark field.
        mask: Null mask column.

    Returns:
        Updated Spark column.
    """
    source_nullable = True if source_field is None else source_field.nullable
    target_nullable = True if target_field is None else target_field.nullable

    if source_nullable and not target_nullable:
        dv = default_python_scalar(target_field)

        column = F.when(mask, F.lit(dv)).otherwise(column)

    return column


def cast_spark_column_to_list(
    column: SparkSQL.Column,
    options: Optional[CastOptions] = None,
) -> SparkSQL.Column:
    """
    Cast a Spark Column to an ArrayType using Arrow field type info.

    This is a pure Spark cast: no collection or Arrow arrays involved.
    """
    options = CastOptions.check_arg(options)

    target_arrow_field = options.target_arrow_field
    target_spark_field = options.target_spark_field

    if target_arrow_field is None:
        # No target type info, just pass through
        return column

    source_spark_field = options.source_spark_field
    source_spark_type = source_spark_field.dataType

    if not isinstance(source_spark_type, T.ArrayType):
        raise ValueError(f"Cannot cast {source_spark_field} to {target_spark_field}")

    # Options for casting individual elements
    element_cast_options = options.copy(
        source_field=options.source_child_arrow_field(index=0),
        target_field=options.target_child_arrow_field(index=0),
    )

    # Cast each element using the same Arrow-aware machinery
    casted = F.transform(
        column,
        lambda x: cast_spark_column(x, element_cast_options),
    )

    # Final cast to enforce the exact Spark ArrayType (element type + containsNull)
    casted = casted.cast(target_spark_field.dataType)

    return casted


def cast_spark_column_to_struct(
    column: SparkSQL.Column,
    options: Optional[CastOptions] = None,
) -> SparkSQL.Column:
    """
    Cast a Spark Column to a StructType using Arrow field type info.

    This is a pure Spark cast: no collection or Arrow arrays involved.
    """
    options = CastOptions.check_arg(options)

    target_arrow_field = options.target_field
    target_spark_field = options.target_spark_field

    if target_arrow_field is None:
        return column

    target_spark_type: T.StructType = target_spark_field.dataType

    source_spark_field = options.source_spark_field
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
                options.copy(
                    source_arrow_field=child_source_arrow_field,
                    target_arrow_field=child_target_arrow_field
                )
            )

        children.append(casted_column.alias(child_target_spark_field.name))

    return F.struct(*children)


def cast_spark_column_to_map(
    column: SparkSQL.Column,
    options: Optional[CastOptions] = None,
) -> SparkSQL.Column:
    """
    Cast a Spark Column to a MapType using Arrow field type info.

    This is a pure Spark cast: no collection or Arrow arrays involved.
    """
    options = CastOptions.check_arg(options)

    target_arrow_field = options.target_field
    target_spark_field = options.target_spark_field

    if target_arrow_field is None:
        return column

    target_spark_type: T.MapType = target_spark_field.dataType

    source_spark_field = options.source_spark_field
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
            cast_spark_column(entry["key"], key_cast_options).alias("key"),
            cast_spark_column(entry["value"], value_cast_options).alias("value"),
        ),
    )

    casted_map = F.map_from_entries(casted_entries)

    # Enforce exact target MapType (keyType, valueType, valueContainsNull)
    casted_map = casted_map.cast(target_spark_type)

    return casted_map


# ---------------------------------------------------------------------------
# Spark DataFrame <-> Arrow Table / RecordBatchReader
# ---------------------------------------------------------------------------

@register_converter(SparkSQL.DataFrame, pa.Table)
def spark_dataframe_to_arrow_table(
    dataframe: SparkSQL.DataFrame,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """Convert a Spark DataFrame to a pyarrow.Table.

    If ``options.target_schema`` is provided, the DataFrame is first cast
    using :func:`cast_spark_dataframe` before conversion. The resulting Arrow
    schema is derived from the cast target schema when available; otherwise it
    is inferred from the Spark schema via :func:`spark_field_to_arrow_field`.
    """
    opts = CastOptions.check_arg(options)

    if opts.target_arrow_schema is not None:
        dataframe = cast_spark_dataframe(dataframe, opts)
        arrow_schema = opts.target_arrow_schema
    else:
        arrow_schema = pa.schema([
            spark_field_to_arrow_field(f, options)
            for f in dataframe.schema
        ])

    return cast_arrow_tabular(dataframe.toArrow(), CastOptions.check_arg(arrow_schema))


@register_converter(pa.Table, SparkSQL.DataFrame)
def arrow_table_to_spark_dataframe(
    table: pa.Table,
    options: Optional[CastOptions] = None,
) -> SparkSQL.DataFrame:
    """Convert a pyarrow.Table to a Spark DataFrame.

    If a target schema is supplied, :func:`cast_arrow_table` is applied before
    creating the Spark DataFrame. Column types are derived from the Arrow
    schema using :func:`arrow_field_to_spark_field` to preserve nullability and
    metadata-driven mappings.
    """
    opts = CastOptions.check_arg(options)

    if opts.target_arrow_schema is not None:
        table = cast_arrow_tabular(table, opts)

    spark = pyspark.sql.SparkSession.getActiveSession()  # type: ignore[union-attr]
    if spark is None:
        raise RuntimeError(
            "An active SparkSession is required to convert Arrow data to Spark"
        )

    spark_schema = arrow_schema_to_spark_schema(table.schema, None)

    return spark.createDataFrame(table, schema=spark_schema)


@register_converter(Any, SparkSQL.DataFrame)
def any_to_spark_dataframe(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> SparkSQL.DataFrame:
    """Convert a pyarrow.Table to a Spark DataFrame.

    If a target schema is supplied, :func:`cast_arrow_table` is applied before
    creating the Spark DataFrame. Column types are derived from the Arrow
    schema using :func:`arrow_field_to_spark_field` to preserve nullability and
    metadata-driven mappings.
    """
    spark = pyspark.sql.SparkSession.getActiveSession()  # type: ignore[union-attr]
    if spark is None:
        raise RuntimeError(
            "An active SparkSession is required to convert Arrow data to Spark"
        )

    if not isinstance(obj, SparkSQL.DataFrame):
        options = CastOptions.check_arg(options)

        if obj is None:
            return spark.createDataFrame([], schema=options.target_spark_schema)

        from ..polars.cast import any_to_polars_dataframe, polars_dataframe_to_arrow_table

        arrow_table = polars_dataframe_to_arrow_table(any_to_polars_dataframe(obj, options), options)
        spark_schema = arrow_schema_to_spark_schema(arrow_table.schema, None)
        obj = spark.createDataFrame(arrow_table, schema=spark_schema)

    return cast_spark_dataframe(obj, options)


# ---------------------------------------------------------------------------
# Arrow <-> Spark type / field / schema converters (hooked into registry)
# ---------------------------------------------------------------------------
@register_converter(SparkSQL.DataFrame, T.DataType)
def spark_dataframe_to_spark_type(
    df: SparkSQL.DataFrame,
    options: Optional[CastOptions] = None,
) -> pa.DataType:
    """Return the Spark DataFrame schema as a Spark data type.

    Args:
        df: Spark DataFrame.
        options: Optional cast options.

    Returns:
        Spark DataType.
    """
    return df.schema


@register_converter(SparkSQL.DataFrame, T.StructField)
def spark_dataframe_to_spark_field(
    df: SparkSQL.DataFrame,
    options: Optional[CastOptions] = None,
) -> pa.DataType:
    """Return a Spark StructField for the DataFrame schema.

    Args:
        df: Spark DataFrame.
        options: Optional cast options.

    Returns:
        Spark StructField.
    """
    return T.StructField(
        df.getAlias() or "root",
        df.schema,
        nullable=False,
    )


@register_converter(SparkSQL.DataFrame, pa.Field)
def spark_dataframe_to_arrow_field(
    df: SparkSQL.DataFrame,
    options: Optional[CastOptions] = None,
) -> pa.DataType:
    """Return an Arrow field representation of the DataFrame schema.

    Args:
        df: Spark DataFrame.
        options: Optional cast options.

    Returns:
        Arrow field.
    """
    return spark_field_to_arrow_field(
        spark_dataframe_to_spark_field(df, options),
        options
    )


@register_converter(SparkSQL.DataFrame, pa.Schema)
def spark_dataframe_to_arrow_schema(
    df: SparkSQL.DataFrame,
    options: Optional[CastOptions] = None,
) -> pa.DataType:
    """Return an Arrow schema representation of the DataFrame.

    Args:
        df: Spark DataFrame.
        options: Optional cast options.

    Returns:
        Arrow schema.
    """
    return arrow_field_to_schema(
        spark_field_to_arrow_field(
            spark_dataframe_to_spark_field(df, options),
            options
        ),
        options
    )


@register_converter(pa.DataType, T.DataType)
def _arrow_type_to_spark_type_for_registry(
    dtype: pa.DataType,
    options: Optional[CastOptions] = None,
) -> T.DataType:  # type: ignore[name-defined]
    """
    Registry wrapper: pyarrow.DataType -> pyspark.sql.types.DataType
    """
    return arrow_type_to_spark_type(dtype, options)


@register_converter(pa.Field, T.StructField)
def _arrow_field_to_spark_field_for_registry(
    field: pa.Field,
    options: Optional[CastOptions] = None,
) -> T.StructField:  # type: ignore[name-defined]
    """
    Registry wrapper: pyarrow.Field -> pyspark.sql.types.StructField
    """
    return arrow_field_to_spark_field(field, options)


@register_converter(T.DataType, pa.DataType)
def _spark_type_to_arrow_type_for_registry(
    dtype: T.DataType,  # type: ignore[name-defined]
    options: Optional[CastOptions] = None,
) -> pa.DataType:
    """
    Registry wrapper: pyspark.sql.types.DataType -> pyarrow.DataType
    """
    return spark_type_to_arrow_type(dtype, options)


@register_converter(T.StructField, pa.Field)
def _spark_field_to_arrow_field_for_registry(
    field: T.StructField,  # type: ignore[name-defined]
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """
    Registry wrapper: pyspark.sql.types.StructField -> pyarrow.Field
    """
    return spark_field_to_arrow_field(field, options)


@register_converter(T.StructField, pa.Schema)
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


@register_converter(pa.Schema, T.StructField)
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
