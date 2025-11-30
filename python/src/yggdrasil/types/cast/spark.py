from typing import Any, Optional

import pyarrow as pa

from .arrow import ArrowCastOptions, default_arrow_python_value
from ..cast.registry import register_converter
from ...libs.sparklib import (
    pyspark,
    require_pyspark,
    arrow_field_to_spark_field,
)

__all__ = [
    "cast_spark_dataframe",
    "cast_spark_column",
]


@require_pyspark(active_session=True)
def cast_spark_dataframe(
    dataframe: "pyspark.sql.DataFrame",
    cast_options: Optional[ArrowCastOptions] = None,
    default_value: Any = None,
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
        * otherwise `default_arrow_python_value(field.type)`
    """
    from pyspark.sql import functions as F

    opts = ArrowCastOptions.check_arg(cast_options)
    target_schema = opts.target_schema

    # No target -> nothing to do
    if target_schema is None:
        return dataframe

    src_schema = dataframe.schema
    src_names = [f.name for f in src_schema]

    exact_name_to_index = {name: idx for idx, name in enumerate(src_names)}
    folded_name_to_index = {name.casefold(): idx for idx, name in enumerate(src_names)}

    new_cols = []

    for i, field in enumerate(target_schema):
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
        spark_struct_field = arrow_field_to_spark_field(field, cast_options, default_value)
        spark_type = spark_struct_field.dataType

        # ----- build column expression -----
        if source_name is None:
            # construct default column
            dv = default_value
            if dv is None:
                dv = default_arrow_python_value(field.type)
            col = F.lit(dv).cast(spark_type)
        else:
            col = F.col(source_name).cast(spark_type)

            # If target field is non-nullable but Spark side might have nulls,
            # fill them with defaults
            if not field.nullable and (source_nullable is None or source_nullable):
                dv = default_value
                if dv is None:
                    dv = default_arrow_python_value(field.type)
                col = F.when(col.isNull(), F.lit(dv)).otherwise(col)

        # Final alias to target name
        new_cols.append(col.alias(field.name))

    # Drop extra columns unless allowed
    if not opts.allow_add_columns:
        return dataframe.select(*new_cols)

    # Keep original columns + casted ones (casted override same names)
    # To keep order as target_schema, select casted first, then any extras.
    extra_cols = [c for c in dataframe.columns if c not in [f.name for f in target_schema]]
    return dataframe.select(*new_cols, *extra_cols)


@require_pyspark(active_session=True)
def cast_spark_column(
    column: "pyspark.sql.Column",
    cast_options: Any = None,
    default_value: Any = None,
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
    from pyspark.sql import functions as F

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

    if not isinstance(target_field, pa.Field):
        # treat DataType as a single anonymous field
        target_field = pa.field("value", target_field, nullable=True)

    spark_struct_field = arrow_field_to_spark_field(target_field, cast_options, default_value)
    spark_type = spark_struct_field.dataType

    col = column.cast(spark_type)

    if not target_field.nullable:
        dv = default_value
        if dv is None:
            dv = default_arrow_python_value(target_field.type)
        col = F.when(col.isNull(), F.lit(dv)).otherwise(col)

    return col


# ---------------------------------------------------------------------------
# Converter registrations
# ---------------------------------------------------------------------------

if pyspark is not None:
    # Spark DF -> Spark DF (Arrow-type-driven cast, pure Spark)
    register_converter(pyspark.sql.DataFrame, pyspark.sql.DataFrame)(cast_spark_dataframe)

    # Spark Column -> Spark Column (Arrow-type-driven cast, pure Spark)
    register_converter(pyspark.sql.Column, pyspark.sql.Column)(cast_spark_column)
