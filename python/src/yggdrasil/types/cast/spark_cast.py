from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Optional, Tuple, List

import pyarrow as pa

from .arrow_cast import (
    ArrowCastOptions,
    cast_arrow_table,
    record_batch_reader_to_table,
    record_batch_to_table, arrow_field_to_schema,
    to_spark_arrow_type,
)
from .registry import register_converter
from ..python_defaults import default_from_arrow_hint
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
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pyspark.sql.DataFrame":
    """
    Cast a Spark DataFrame using Arrow *types* but without collecting to Arrow.

    - `cast_options` is normalized via ArrowCastOptions.check_arg.
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
    if pyspark is None:
        raise RuntimeError("pyspark is required to cast a Spark DataFrame")

    opts = ArrowCastOptions.check_arg(cast_options)
    target_schema = opts.target_schema

    # No target -> nothing to do
    if target_schema is None:
        return dataframe

    def _spark_compatible_schema(schema: pa.Schema) -> pa.Schema:
        return pa.schema(
            [
                pa.field(
                    f.name,
                    to_spark_arrow_type(f.type),
                    nullable=f.nullable,
                    metadata=f.metadata,
                )
                for f in schema
            ],
            metadata=schema.metadata,
        )

    def _cast_via_map_in_arrow(schema: pa.Schema) -> "pyspark.sql.DataFrame":
        compatible_schema = _spark_compatible_schema(schema)

        def _cast_batches(batches):
            compat_options = dc_replace(opts)
            compat_options.target_schema = compatible_schema

            for batch in batches:
                table = record_batch_to_table(batch, compat_options)
                casted = cast_arrow_table(table, compat_options)
                yield from casted.to_batches()

        spark_schema = arrow_schema_to_spark_schema(compatible_schema, opts)

        spark = pyspark.sql.SparkSession.getActiveSession()  # type: ignore[union-attr]
        if spark is None:
            raise RuntimeError("An active SparkSession is required to cast with mapInArrow")

        source_root = Path(__file__).resolve().parents[3]
        spark.sparkContext.addPyFile(str(source_root))

        return dataframe.mapInArrow(_cast_batches, spark_schema)

    cast_schema = target_schema
    try:
        spark_schema = arrow_schema_to_spark_schema(target_schema, opts)
    except TypeError:
        cast_schema = _spark_compatible_schema(target_schema)
        try:
            spark_schema = arrow_schema_to_spark_schema(cast_schema, opts)
        except TypeError:
            return _cast_via_map_in_arrow(target_schema)

    src_schema = dataframe.schema
    src_names = [f.name for f in src_schema.fields]

    exact_name_to_index = {name: idx for idx, name in enumerate(src_names)}
    folded_name_to_index = {name.casefold(): idx for idx, name in enumerate(src_names)}

    new_cols = []
    f: List[Tuple[int, pa.Field]] = list(enumerate(cast_schema))

    for i, field in f:
        # ----- resolve source column name -----
        source_name: Optional[str] = None
        source_nullable: Optional[bool] = None

        if field.name in exact_name_to_index:
            idx = exact_name_to_index[field.name]
            source_name = src_names[idx]
            source_nullable = src_schema[idx].nullable
        elif not opts.strict_match_names and field.name.casefold() in folded_name_to_index:
            idx = folded_name_to_index[field.name.casefold()]
            source_name = src_names[idx]
            source_nullable = src_schema[idx].nullable
        elif not opts.strict_match_names and len(src_names) > len(new_cols):
            idx = len(new_cols)
            source_name = src_names[idx]
            source_nullable = src_schema[idx].nullable
        elif opts.add_missing_columns:
            # No source column: we’ll synthesize one with defaults
            source_name = None
            source_nullable = None
        else:
            raise TypeError(f"Missing column {field.name} while casting Spark DataFrame")

        # ----- get target Spark type from Arrow field -----
        spark_struct_field = arrow_field_to_spark_field(field, cast_options)
        spark_type = spark_struct_field.dataType

        # ----- build column expression -----
        if source_name is None:
            # construct default column
            dv = default_from_arrow_hint(field).as_py()
            col = F.lit(dv).cast(spark_type)
        else:
            col = F.col(source_name).cast(spark_type)

            # If target field is non-nullable but Spark side might have nulls,
            # fill them with defaults
            if not field.nullable and (source_nullable is None or source_nullable):
                dv = default_from_arrow_hint(field).as_py()
                col = F.when(col.isNull(), F.lit(dv)).otherwise(col)

        # Final alias to target name
        new_cols.append(col.alias(field.name))

    # Drop extra columns unless allowed
    result = dataframe.select(*new_cols)

    spark = pyspark.sql.SparkSession.getActiveSession()  # type: ignore[union-attr]
    if spark is None:
        raise RuntimeError("An active SparkSession is required to cast a Spark DataFrame")

    if not opts.allow_add_columns:
        return spark.createDataFrame(result.rdd, schema=spark_schema)

    # Keep original columns + casted ones (casted override same names)
    # To keep order as target_schema, select casted first, then any extras.
    extra_cols = [c for c in dataframe.columns if c not in [f.name for f in target_schema]]
    full_schema = SparkSchema(list(spark_schema) + [src_schema[exact_name_to_index[c]] for c in extra_cols])
    result_with_extras = dataframe.select(*new_cols, *[F.col(c) for c in extra_cols])
    return spark.createDataFrame(result_with_extras.rdd, schema=full_schema)


@spark_converter(SparkColumn, SparkColumn)
def cast_spark_column(
    column: "pyspark.sql.Column",
    cast_options: Any = None,
) -> "pyspark.sql.Column":
    """
    Cast a single Spark Column using an Arrow target *type*.

    `cast_options` is interpreted via ArrowCastOptions.check_arg, and only the
    target_field is used. Supports:
      - pa.DataType
      - pa.Field
      - pa.Schema (we use the first field)
      - ArrowCastOptions (with target_field set)

    This is a pure Spark cast: no collection or Arrow arrays involved.
    """
    if pyspark is None:
        raise RuntimeError("pyspark is required to cast a Spark column")

    opts = ArrowCastOptions.check_arg(cast_options)
    target_field = opts.target_field

    if target_field is None:
        # Nothing to cast to, return the column as-is
        return column

    # If a schema is passed, normalize to a single field (first)
    if isinstance(target_field, pa.Schema):
        if len(target_field) != 1:
            raise ValueError("cast_spark_column: Schema target must have exactly one field")
        target_field = target_field[0]

    # ArrowCastOptions built from a Schema may wrap it in a root struct field; unwrap it
    if (
        isinstance(target_field, pa.Field)
        and pa.types.is_struct(target_field.type)
        and len(target_field.type) == 1
        and target_field.name == "root"
    ):
        target_field = target_field.type[0]

    if not isinstance(target_field, pa.Field):
        # treat DataType as a single anonymous field
        target_field = pa.field("value", target_field, nullable=True)

    compatible_field = pa.field(
        target_field.name,
        to_spark_arrow_type(target_field.type),
        nullable=target_field.nullable,
        metadata=target_field.metadata,
    )

    def _fill_default(field: pa.Field, col: "pyspark.sql.Column") -> "pyspark.sql.Column":
        if not field.nullable:
            dv = default_from_arrow_hint(field)
            if hasattr(dv, "as_py"):
                dv = dv.as_py()
            if isinstance(dv, dict) and pa.types.is_struct(field.type):
                default_fields = [
                    F.lit(
                        dv_value.as_py() if hasattr(dv_value := dv.get(child.name), "as_py") else dv_value
                    ).alias(child.name)
                    for child in field.type
                ]
                default_col = F.struct(*default_fields)
                return F.when(col.isNull(), default_col).otherwise(col)

            return F.when(col.isNull(), F.lit(dv)).otherwise(col)

        return col

    def _cast_to_field(field: pa.Field, col: "pyspark.sql.Column") -> "pyspark.sql.Column":
        spark_struct_field = arrow_field_to_spark_field(field, cast_options)
        spark_type = spark_struct_field.dataType

        if isinstance(spark_type, T.StructType) and pa.types.is_struct(field.type):
            child_cols = []
            for child_field in field.type:
                child_col = _cast_to_field(child_field, col.getField(child_field.name))
                child_cols.append(child_col.alias(child_field.name))

            struct_col = F.struct(*child_cols)
            return _fill_default(field, struct_col)

        base_col = col.cast(spark_type)
        return _fill_default(field, base_col)

    try:
        return _cast_to_field(target_field, column)
    except TypeError:
        # Normalize unsupported Arrow types (e.g., dictionary, large_*, extension)
        # to Spark-compatible equivalents and retry.
        return _cast_to_field(compatible_field, column)


# ---------------------------------------------------------------------------
# Spark DataFrame <-> Arrow Table / RecordBatchReader
# ---------------------------------------------------------------------------


@spark_converter(SparkDataFrame, pa.Table)
def spark_dataframe_to_arrow_table(
    dataframe: "pyspark.sql.DataFrame",
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.Table:
    """Convert a Spark DataFrame to a pyarrow.Table.

    If ``cast_options.target_schema`` is provided, the DataFrame is first cast
    using :func:`cast_spark_dataframe` before conversion. The resulting Arrow
    schema is derived from the cast target schema when available; otherwise it
    is inferred from the Spark schema via :func:`spark_field_to_arrow_field`.
    """
    if pyspark is None:
        raise RuntimeError("pyspark is required to convert Spark to Arrow")

    opts = ArrowCastOptions.check_arg(cast_options)

    if opts.target_schema is not None:
        dataframe = cast_spark_dataframe(dataframe, opts)
        arrow_schema = opts.target_schema
    else:
        arrow_schema = pa.schema(
            [spark_field_to_arrow_field(f, cast_options) for f in dataframe.schema]
        )

    return cast_arrow_table(dataframe.toArrow(), ArrowCastOptions.check_arg(arrow_schema))


@spark_converter(SparkDataFrame, pa.RecordBatchReader)
def spark_dataframe_to_record_batch_reader(
    dataframe: "pyspark.sql.DataFrame",
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatchReader:
    """Convert a Spark DataFrame to a pyarrow.RecordBatchReader."""
    table = spark_dataframe_to_arrow_table(dataframe, cast_options)
    batches = table.to_batches()
    return pa.RecordBatchReader.from_batches(table.schema, batches)


@spark_converter(pa.Table, SparkDataFrame)
def arrow_table_to_spark_dataframe(
    table: pa.Table,
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pyspark.sql.DataFrame":
    """Convert a pyarrow.Table to a Spark DataFrame.

    If a target schema is supplied, :func:`cast_arrow_table` is applied before
    creating the Spark DataFrame. Column types are derived from the Arrow
    schema using :func:`arrow_field_to_spark_field` to preserve nullability and
    metadata-driven mappings.
    """
    if pyspark is None:
        raise RuntimeError("pyspark is required to convert Arrow to Spark")

    opts = ArrowCastOptions.check_arg(cast_options)

    if opts.target_schema is not None:
        table = cast_arrow_table(table, opts)

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
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pyspark.sql.DataFrame":
    """Convert a pyarrow.RecordBatch to a Spark DataFrame."""
    table = record_batch_to_table(batch, cast_options)
    return arrow_table_to_spark_dataframe(table, cast_options)


@spark_converter(pa.RecordBatchReader, SparkDataFrame)
def arrow_record_batch_reader_to_spark_dataframe(
    reader: pa.RecordBatchReader,
    cast_options: Optional[ArrowCastOptions] = None,
) -> "pyspark.sql.DataFrame":
    """Convert a pyarrow.RecordBatchReader to a Spark DataFrame."""
    table = record_batch_reader_to_table(reader, cast_options)
    return arrow_table_to_spark_dataframe(table, cast_options)


# ---------------------------------------------------------------------------
# Arrow <-> Spark type / field / schema converters (hooked into registry)
# ---------------------------------------------------------------------------
@spark_converter(SparkDataFrame, SparkDataType)
def spark_dataframe_to_spark_type(
    df: SparkDataFrame,
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.DataType:
    return df.schema


@spark_converter(SparkDataFrame, SparkStructField)
def spark_dataframe_to_spark_field(
    df: SparkDataFrame,
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.DataType:
    return SparkStructField(
        df.getAlias() or "root",
        df.schema,
        nullable=False,
    )


@spark_converter(SparkDataFrame, pa.Field)
def spark_dataframe_to_arrow_field(
    df: SparkDataFrame,
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.DataType:
    return spark_field_to_arrow_field(
        spark_dataframe_to_spark_field(df, cast_options),
        cast_options
    )


@spark_converter(SparkDataFrame, pa.Schema)
def spark_dataframe_to_arrow_schema(
    df: SparkDataFrame,
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.DataType:
    return arrow_field_to_schema(
        spark_field_to_arrow_field(
            spark_dataframe_to_spark_field(df, cast_options),
            cast_options
        ),
        cast_options
    )


@spark_converter(pa.DataType, SparkDataType)
def _arrow_type_to_spark_type_for_registry(
    dtype: pa.DataType,
    cast_options: Optional[ArrowCastOptions] = None,
) -> "T.DataType":  # type: ignore[name-defined]
    """
    Registry wrapper: pyarrow.DataType -> pyspark.sql.types.DataType
    """
    return arrow_type_to_spark_type(dtype, cast_options)


@spark_converter(pa.Field, SparkStructField)
def _arrow_field_to_spark_field_for_registry(
    field: pa.Field,
    cast_options: Optional[ArrowCastOptions] = None,
) -> "T.StructField":  # type: ignore[name-defined]
    """
    Registry wrapper: pyarrow.Field -> pyspark.sql.types.StructField
    """
    return arrow_field_to_spark_field(field, cast_options)


@spark_converter(SparkDataType, pa.DataType)
def _spark_type_to_arrow_type_for_registry(
    dtype: "T.DataType",  # type: ignore[name-defined]
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.DataType:
    """
    Registry wrapper: pyspark.sql.types.DataType -> pyarrow.DataType
    """
    return spark_type_to_arrow_type(dtype, cast_options)


@spark_converter(SparkStructField, pa.Field)
def _spark_field_to_arrow_field_for_registry(
    field: "T.StructField",  # type: ignore[name-defined]
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.Field:
    """
    Registry wrapper: pyspark.sql.types.StructField -> pyarrow.Field
    """
    return spark_field_to_arrow_field(field, cast_options)


@spark_converter(SparkSchema, pa.Schema)
def spark_schema_to_arrow_schema(
    schema: "T.StructType",  # type: ignore[name-defined]
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.Schema:
    """
    Convert a pyspark StructType to a pyarrow.Schema.
    """
    opts = ArrowCastOptions.check_arg(cast_options)
    arrow_fields = [
        spark_field_to_arrow_field(field, opts)
        for field in schema.fields
    ]
    return pa.schema(arrow_fields)


@spark_converter(pa.Schema, SparkSchema)
def arrow_schema_to_spark_schema(
    schema: pa.Schema,
    cast_options: Optional[ArrowCastOptions] = None,
) -> "T.StructType":  # type: ignore[name-defined]
    """
    Convert a pyarrow.Schema to a pyspark StructType.
    """
    opts = ArrowCastOptions.check_arg(cast_options)
    spark_fields = [
        arrow_field_to_spark_field(field, opts)
        for field in schema
    ]
    return T.StructType(spark_fields)
