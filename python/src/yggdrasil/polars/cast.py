"""Polars <-> Arrow casting helpers and converters."""
from typing import Optional, Tuple, Union, Dict, Any

import polars
import pyarrow as pa
import pyarrow.types as pat

from ..io.path import LocalDataPath, SystemPath
from ..pyutils.serde import ObjectSerde
from ..types.cast.arrow_cast import (
    cast_arrow_tabular,
    is_arrow_type_binary_like,
    is_arrow_type_string_like,
    is_arrow_type_list_like,
)
from ..types.cast.cast_options import CastOptions
from ..types.cast.registry import register_converter
from ..types.python_defaults import default_arrow_scalar

__all__ = [
    "register_converter",
    "cast_polars_array",
    "cast_polars_dataframe",
    "arrow_type_to_polars_type",
    "polars_type_to_arrow_type",
    "arrow_field_to_polars_field",
    "polars_field_to_arrow_field",
    "polars_strptime",
    "polars_dataframe_to_arrow_table",
    "arrow_table_to_polars_dataframe",
    "any_to_polars_dataframe",
    "any_polars_to_arrow_field"
]

# ---------------------------------------------------------------------------
# Polars type aliases + decorator wrapper (safe when Polars is missing)
# ---------------------------------------------------------------------------


# Primitive Arrow -> Polars dtype mapping (base, non-nested types).
# These are Polars *dtype classes* (not instances), so they can be used
# directly in schemas (e.g. pl.Struct({"a": pl.Int64})).
ARROW_TO_POLARS: Dict[pa.DataType, polars.DataType] = {
    pa.null(): polars.Null(),
    pa.bool_(): polars.Boolean(),

    pa.int8(): polars.Int8(),
    pa.int16(): polars.Int16(),
    pa.int32(): polars.Int32(),
    pa.int64(): polars.Int64(),

    pa.uint8(): polars.UInt8(),
    pa.uint16(): polars.UInt16(),
    pa.uint32(): polars.UInt32(),
    pa.uint64(): polars.UInt64(),

    pa.float16(): polars.Float32(),  # best-effort
    pa.float32(): polars.Float32(),
    pa.float64(): polars.Float64(),

    pa.string(): polars.Utf8(),
    pa.string_view(): polars.Utf8(),
    pa.large_string(): polars.Utf8(),

    pa.binary(): polars.Binary(),
    pa.binary_view(): polars.Binary(),
    pa.large_binary(): polars.Binary(),

    pa.date32(): polars.Date(),
}


POLARS_BASE_TO_ARROW = {
    v: k for k, v in ARROW_TO_POLARS.items()
}


def polars_type_classes():
    return tuple(v.__class__ for v in ARROW_TO_POLARS.values())


# ---------------------------------------------------------------------------
# Core casting: Polars <-> Arrow types
# ---------------------------------------------------------------------------
@register_converter(polars.Series, polars.Series)
@register_converter(polars.Expr, polars.Expr)
def cast_polars_array(
    array: Union[polars.Series, polars.Expr],
    options: Optional[CastOptions] = None,
) -> Union[polars.Series, polars.Expr]:
    """
    Cast a Polars Series to a target Arrow type using Polars casting rules.

    `options` is normalized via ArrowCastOptions.check_arg and its
    `target_field` is used as the Arrow target.

    - target_field can be a pa.DataType, pa.Field, or pa.Schema (schema → first field).
    - If a Field is provided, we also respect its nullability by filling nulls
      when nullable=False (using default_from_arrow_hint).
    """
    options = CastOptions.check_arg(options)

    if not options.need_polars_type_cast(source_obj=array):
        if options.need_nullability_check(source_obj=array):
            dv = default_arrow_scalar(
                options.target_field.type,
                nullable=options.target_field.nullable
            ).as_py()

            array = array.fill_null(dv)
        return array.alias(options.target_field_name)

    source_polars_field = options.source_polars_field
    target_polars_field = options.target_polars_field

    if source_polars_field.dtype == polars.Null():
        return array.cast(target_polars_field.dtype, strict=False)

    # strict=True => fail on lossy casts
    if isinstance(target_polars_field.dtype, polars.datatypes.TemporalType):
        if isinstance(source_polars_field.dtype, (polars.Utf8, polars.Binary)):
            casted = polars_strptime(array, options=options)
        else:
            casted = array.cast(
                target_polars_field.dtype,
                strict=options.safe
            )
    elif isinstance(target_polars_field.dtype, polars.List):
        # For Structs, we need to cast each subfield individually to respect nullability
        casted = cast_to_list_array(array, options=options)
    elif isinstance(target_polars_field.dtype, polars.Struct):
        # For Structs, we need to cast each subfield individually to respect nullability
        casted = cast_to_struct_array(array, options=options)
    else:
        casted = array.cast(target_polars_field.dtype, strict=options.safe)

    if options.need_nullability_check(source_obj=array):
        dv = default_arrow_scalar(
            options.target_field.type, nullable=options.target_field.nullable
        ).as_py()

        casted = casted.fill_null(dv)
    return casted.alias(options.target_field_name)


def cast_to_list_array(
    array: polars.Series,
    options: Optional[CastOptions] = None,
) -> polars.Series:
    """Cast a Polars list series to a target list Arrow type.

    Args:
        array: Polars Series with list dtype.
        options: Optional cast options.

    Returns:
        Cast Polars Series.
    """
    options = CastOptions.check_arg(options)

    if not options.need_polars_type_cast(source_obj=array):
        if options.need_nullability_check(source_obj=array):
            dv = default_arrow_scalar(
                options.target_field.type,
                nullable=options.target_field.nullable
            ).as_py()

            return array.fill_null(dv)
        return array

    if not isinstance(options.source_polars_field.dtype, polars.List):
        raise TypeError(f"expected List series, got {array.dtype}")

    element_options = options.copy(
        source_arrow_field=options.source_child_arrow_field(index=0),
        target_arrow_field=options.target_child_arrow_field(index=0)
    )

    # ✅ Fast path: cast elements in-place (no explode)
    df = array.to_frame("__x")
    out = df.select(
        polars.col("__x")
        .list.eval(
            cast_polars_array(polars.element(), element_options),
            parallel=True,
        )
        .alias("__x")
    )["__x"]

    # Enforce list-level nullability if required
    if options.need_nullability_check(source_obj=array):
        dv = default_arrow_scalar(
            options.target_field.type,
            nullable=options.target_field.nullable,
        ).as_py()
        out = out.fill_null(dv)

    return out


def cast_to_struct_array(
    array: polars.Series,
    options: Optional[CastOptions] = None,
) -> polars.Series:
    """
    Cast a Polars Struct Series to a target Arrow Struct type using Polars casting rules.

    Each subfield is cast individually.
    """
    options = CastOptions.check_arg(options)

    if not options.need_polars_type_cast(source_obj=array):
        if options.need_nullability_check(source_obj=array):
            dv = default_arrow_scalar(
                options.target_field.type,
                nullable=options.target_field.nullable
            ).as_py()

            return array.fill_null(dv)
        return array

    if not isinstance(options.source_polars_field.dtype, polars.Struct):
        raise ValueError(f"Cannot make struct polars series from {options.source_polars_field}")

    source_polars_fields = options.source_polars_field.dtype.fields
    source_arrow_fields = [polars_field_to_arrow_field(f, None) for f in source_polars_fields]

    target_arrow_fields: list[pa.Field] = list(options.target_arrow_field.type)
    target_polars_fields = [arrow_field_to_polars_field(f, None) for f in target_arrow_fields]

    name_to_index = {f.name: idx for idx, f in enumerate(source_polars_fields)}
    if not options.strict_match_names:
        name_to_index.update(
            {
                f.name.casefold(): idx
                for idx, f in enumerate(source_polars_fields)
            }
        )

    children = []

    for target_index, child_target_polars_field in enumerate(target_polars_fields):
        child_target_arrow_field: pa.Field = target_arrow_fields[target_index]

        find_name = (
            child_target_polars_field.name
            if options.strict_match_names
            else child_target_polars_field.name.casefold()
        )
        source_index = name_to_index.get(find_name)

        if source_index is None:
            if not options.add_missing_columns:
                raise ValueError(
                    f"Missing column {child_target_arrow_field!r} from {target_arrow_fields}"
                )

            dv = default_arrow_scalar(
                dtype=child_target_arrow_field.type,
                nullable=child_target_arrow_field.nullable,
            ).as_py()
            casted_child = polars.lit(
                value=dv,
                dtype=child_target_polars_field.dtype,
            )
        else:
            child_source_arrow_field: pa.Field = source_arrow_fields[source_index]
            child_source_polars_field: polars.Field = source_polars_fields[source_index]

            casted_child = cast_polars_array(
                array.struct.field(child_source_polars_field.name),
                options.copy(
                    source_arrow_field=child_source_arrow_field,
                    target_arrow_field=child_target_arrow_field,
                ),
            )

        children.append(casted_child.alias(child_target_polars_field.name))

    # Build the struct from the cast children
    result_struct = polars.struct(
        *children,
    )

    if options.source_field.nullable and options.target_field.nullable:
        return (
            polars.when(array.is_null())
            .then(None)
            .otherwise(result_struct)
            .alias(options.target_field.name)
        )
    return result_struct.alias(options.target_field.name)


def polars_strptime(
    series: polars.Series,
    options: Optional[CastOptions] = None,
) -> polars.Series:
    """
    Helper to parse strings to datetime in Polars using optional patterns.
    """
    options = CastOptions.check_arg(options)
    polars_field = options.target_polars_field

    if polars_field is None:
        polars_field = polars.Field(series.name, polars.Datetime("us", "UTC"))

    patterns = options.datetime_patterns or []

    if not patterns:
        # No patterns provided; use default parsing
        return (
            series.str
            .strptime(polars_field.dtype, strict=options.safe)
        )

    # Try each pattern in sequence until one works
    last_error = None
    for pattern in patterns:
        try:
            return (
                series
                .str.strptime(
                    polars_field.dtype,
                    format=pattern,
                    strict=True,
                    ambiguous="earliest"
                )
            )
        except Exception as e:
            last_error = e

    # If none worked, raise the last error
    raise last_error


@register_converter(polars.DataFrame, polars.DataFrame)
def cast_polars_dataframe(
    dataframe: polars.DataFrame,
    options: Optional[CastOptions] = None,
) -> polars.DataFrame:
    """
    Cast a Polars DataFrame to a target Arrow schema using Arrow casting rules.

    Uses:
    - name / case-insensitive / positional matching for columns
    - add_missing_columns to synthesize columns with defaults
    - allow_add_columns to keep or drop extra source columns
    """
    options = CastOptions.check_arg(options)
    target_arrow_schema = options.target_arrow_schema

    if target_arrow_schema is None:
        return dataframe

    sub_source_polars_fields = [
        polars.Field(name, d)
        for name, d in dataframe.schema.items()
    ]
    sub_source_arrow_fields = [
        polars_field_to_arrow_field(f, None)
        for f in sub_source_polars_fields
    ]
    sub_target_polars_fields = [
        arrow_field_to_polars_field(f, None)
        for f in target_arrow_schema
    ]

    source_name_to_index = {
        field.name: idx for idx, field in enumerate(sub_source_polars_fields)
    }

    if not options.strict_match_names:
        source_name_to_index.update({
            field.name.casefold(): idx for idx, field in enumerate(sub_source_polars_fields)
        })

    columns: list[Tuple[polars.Field, polars.Series]] = []
    found_column_names = set()

    for sub_target_index, sub_target_field in enumerate(sub_target_polars_fields):
        sub_target_field: polars.Field = sub_target_field
        target_arrow_field = target_arrow_schema.field(sub_target_index)
        source_index = source_name_to_index.get(sub_target_field.name)

        if source_index is None:
            if not options.add_missing_columns:
                raise pa.ArrowInvalid(f"Missing column '{sub_target_field.name}' in source polars dataframe {sub_source_polars_fields}")

            dv = default_arrow_scalar(
                target_arrow_field.type,
                nullable=target_arrow_field.nullable
            )
            series = polars.repeat(
                value=dv.as_py(),
                n=dataframe.shape[0],
                dtype=sub_target_field.dtype
            )
        else:
            source_field = sub_source_arrow_fields[source_index]
            found_column_names.add(source_field.name)
            source_series = dataframe[:, source_index]
            series = cast_polars_array(
                source_series,
                options.copy(
                    source_arrow_field=source_field,
                    target_arrow_field=target_arrow_field
                )
            )

        columns.append((sub_target_field, series.alias(sub_target_field.name)))

    # Start with only the casted schema columns
    result = dataframe.select(c for _, c in columns)

    # If we allow extra columns, horizontally concat them back
    if options.allow_add_columns:
        extra_cols = [
            name for name in dataframe.columns if name not in found_column_names
        ]

        if extra_cols:
            extra_df = dataframe.select(extra_cols)
            result = polars.concat([result, extra_df], how="horizontal")

    return result


@register_converter(polars.DataFrame, pa.Table)
def polars_dataframe_to_arrow_table(
    data: polars.DataFrame,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """
    Convert a Polars DataFrame to a pyarrow.Table.

    - If cast_options.target_schema is set, we apply `cast_polars_dataframe`
      first, then call `.to_arrow()`.
    - Otherwise, we directly call `data.to_arrow()`.
    """
    options = CastOptions.check_arg(options)
    target_field = options.target_field

    if target_field is not None:
        data = cast_polars_dataframe(data, options)

    compat_level = polars.CompatLevel.newest()

    try:
        table = data.to_arrow(compat_level=compat_level)
    except pa.ArrowInvalid:
        table = data.rechunk().to_arrow(compat_level=compat_level)

    # If you want Arrow-side casting too, keep this; otherwise it’s redundant.
    if target_field is not None:
        table = cast_arrow_tabular(table, options)

    return table


@register_converter(pa.Table, polars.DataFrame)
def arrow_table_to_polars_dataframe(
    table: pa.Table,
    options: Optional[CastOptions] = None,
) -> polars.DataFrame:
    """
    Convert a pyarrow.Table to a Polars DataFrame.

    - If cast_options.target_schema is set, we first apply `cast_arrow_table`
      then call `polars.from_arrow`.
    """
    options = CastOptions.check_arg(options)

    if options.target_arrow_schema is not None:
        table = cast_arrow_tabular(table, options)

    return polars.from_arrow(table)


@register_converter(Any, polars.DataFrame)
def any_to_polars_dataframe(
    obj: Any,
    options: Optional[CastOptions] = None
) -> polars.DataFrame:
    if not isinstance(obj, polars.DataFrame):
        options = CastOptions.check_arg(options)

        if obj is None:
            return polars.DataFrame(
                [],
                schema=options.target_polars_schema
            )
        elif isinstance(obj, (str, SystemPath)):
            path = LocalDataPath(obj)
            
            return path.read_polars(cast_options=options)

        namespace = ObjectSerde.full_namespace(obj)

        if namespace.startswith("pandas."):
            obj = polars.from_pandas(
                obj,
                nan_to_null=True,
                include_index=bool(obj.index.name)
            )
        elif namespace.startswith("pyarrow."):
            obj = polars.from_arrow(obj)
        elif namespace.startswith("pyspark."):
            from ..spark.lib import pyspark_sql

            obj: pyspark_sql.DataFrame = obj
            obj = polars.from_arrow(obj.toArrow())
        else:
            obj = polars.DataFrame(obj)

    return cast_polars_dataframe(obj, options)


# ---------------------------------------------------------------------------
# PyArrow Types <-> Polars Types
# ---------------------------------------------------------------------------
@register_converter(pa.DataType, polars.DataType)
def arrow_type_to_polars_type(
    arrow_type: Union[pa.DataType, pa.TimestampType, pa.ListType, pa.MapType],
    options: Optional[dict] = None,
) -> polars.DataType:
    """
    Convert a pyarrow.DataType to a Polars dtype.

    Returns a Polars dtype object (e.g. pl.Int64, pl.List(pl.Utf8), ...).
    Raises TypeError for unsupported types.
    """
    # Fast path: exact primitive mapping
    dtype = ARROW_TO_POLARS.get(arrow_type)

    if dtype is not None:
        return dtype

    # Timestamp
    if pat.is_timestamp(arrow_type):
        unit = arrow_type.unit  # "s", "ms", "us", "ns"
        tz = arrow_type.tz
        # Polars supports "ns", "us", "ms". Upcast seconds.
        if unit == "s":
            unit = "ms"
        return polars.Datetime(time_unit=unit, time_zone=tz)

    if pat.is_time(arrow_type):
        return polars.Time()

    # Duration
    if pat.is_duration(arrow_type):
        unit = arrow_type.unit  # "s", "ms", "us", "ns"
        if unit == "s":
            unit = "ms"
        return polars.Duration(time_unit=unit)

    if is_arrow_type_binary_like(arrow_type):
        return polars.Binary()

    if is_arrow_type_string_like(arrow_type):
        return polars.Utf8()

    # Dictionary -> Categorical (no categories info at dtype level)
    if pat.is_dictionary(arrow_type):
        return polars.Categorical()

    # Map -> represented as List(Struct({"key": ..., "value": ...}))
    if pat.is_map(arrow_type):
        key_type = arrow_type.key_type
        item_type = arrow_type.item_type
        pl_key = arrow_type_to_polars_type(key_type, None)
        pl_val = arrow_type_to_polars_type(item_type, None)
        # Struct fields: we prefer real pl.Field if available
        field_cls = getattr(polars, "Field", None)
        if callable(field_cls):
            struct_dtype = polars.Struct(
                [
                    field_cls("key", pl_key),
                    field_cls("value", pl_val),
                ]
            )
        else:
            struct_dtype = polars.Struct({"key": pl_key, "value": pl_val})
        return polars.List(struct_dtype)

    # List / LargeList
    if is_arrow_type_list_like(arrow_type):
        inner = arrow_type.value_type
        inner_pl = arrow_type_to_polars_type(inner, None)
        return polars.List(inner_pl)

    # Struct
    if pat.is_struct(arrow_type):
        field_cls = getattr(polars, "Field", None)
        if callable(field_cls):
            fields = [
                field_cls(f.name, arrow_type_to_polars_type(f.type, None))
                for f in arrow_type
            ]
            return polars.Struct(fields)
        else:
            return polars.Struct(
                {f.name: arrow_type_to_polars_type(f.type, None) for f in arrow_type}
            )

    raise TypeError(f"Unsupported or unknown Arrow type for Polars conversion: {arrow_type!r}")


@register_converter(pa.Field, polars.Field)
def arrow_field_to_polars_field(
    field: pa.Field,
    options: Optional[dict] = None,
) -> polars.Field:
    """
    Convert a pyarrow.Field to a Polars field representation.

    If polars.Field exists, returns a polars.Field(name, dtype).
    """
    built = polars.Field(field.name, arrow_type_to_polars_type(field.type, options))

    try:
        setattr(built, "nullable", field.nullable)
    except Exception:
        pass

    return built


@register_converter(polars.DataType, pa.DataType)
def polars_type_to_arrow_type(
    pl_type: polars.DataType | type,
    options: Optional[dict] = None,
) -> pa.DataType:
    """
    Convert a Polars dtype (class or instance) to a pyarrow.DataType.

    Handles primitives via POLARS_BASE_TO_ARROW and common nested/temporal types.
    """
    try:
        base = pl_type()
    except:
        base = pl_type

    # Primitive base mapping
    existing = POLARS_BASE_TO_ARROW.get(base) or POLARS_BASE_TO_ARROW.get(type(pl_type))

    if existing is not None:
        return existing

    if isinstance(pl_type, polars.Datetime):
        unit = pl_type.time_unit
        tz = pl_type.time_zone
        return pa.timestamp(unit, tz=tz)

    elif isinstance(pl_type, polars.Duration):
        unit = pl_type.time_unit
        return pa.duration(unit)

    elif isinstance(pl_type, polars.Decimal):
        precision = pl_type.precision
        scale = pl_type.scale
        return pa.decimal128(precision, scale) if precision <= 38 else pa.decimal256(precision, scale)

    elif isinstance(pl_type, polars.List):
        inner = pl_type.inner
        arrow_inner = polars_type_to_arrow_type(inner, None)

        return pa.list_(arrow_inner)

    elif isinstance(pl_type, polars.Struct):
        fields = [
            polars_field_to_arrow_field(_, None)
            for _ in pl_type.fields
        ]

        return pa.struct(fields)

    # Categorical / Enum -> Arrow dictionary<string>
    if isinstance(pl_type, polars.Categorical) or isinstance(pl_type, polars.Enum):
        # We don't have direct info on categories at the dtype level,
        # so choose a reasonable default: int32 index over string values.
        return pa.dictionary(index_type=pa.int32(), value_type=pa.string())

    raise TypeError(f"Unsupported or unknown Polars dtype for Arrow conversion: {pl_type!r}")


@register_converter(polars.Field, pa.Field)
def polars_field_to_arrow_field(
    field: polars.Field,
    options: Optional["CastOptions"] = None,
) -> pa.Field:
    """
    Convert a Polars field to a pyarrow.Field.

    Accepts:
      - polars.datatypes.Field instances
      - (name, dtype) tuples
    """
    arrow_type = polars_type_to_arrow_type(field.dtype, options)

    return pa.field(field.name, arrow_type, nullable=getattr(field, "nullable", True))


@register_converter(polars.Series, pa.Field)
@register_converter(polars.Expr, pa.Field)
def polars_array_to_arrow_field(
    array: Union[polars.Series, polars.Expr],
    options: Optional[CastOptions] = None
) -> pa.Field:
    """Infer an Arrow field from a Polars Series or Expr.

    Args:
        array: Polars Series or Expr.
        options: Optional cast options.

    Returns:
        Arrow field.
    """
    options = CastOptions.check_arg(options)

    if options.source_arrow_field:
        return options.source_arrow_field

    name = array.name
    atype = polars_type_to_arrow_type(array.dtype, None)
    nullable = array.null_count() > 0

    return pa.field(
        name,
        atype,
        nullable=nullable,
        metadata=None
    )

def any_polars_to_arrow_field(obj: Any, options: Optional[CastOptions]) -> pa.Field:
    options = CastOptions.check_arg(options)

    if isinstance(obj, polars.DataFrame):
        obj = pa.field(
            name="root",
            type=pa.struct([
                pa.field(name=name, type=polars_type_to_arrow_type(dtype, None))
                for name, dtype in obj.schema.items()
            ])
        )

        return obj

    elif isinstance(obj, polars.Series):
        obj = pa.field(
            name=obj.name,
            type=polars_type_to_arrow_type(obj.dtype, None),
        )

        return obj

    elif isinstance(obj, polars.Expr):
        obj = pa.field(
            name=obj.name,
            type=pa.null(),
        )

        return obj

    elif isinstance(obj, polars.DataType):
        return pa.field(
            name="root" if options.target_field is None else options.target_field.name,
            type=polars_type_to_arrow_type(obj, options),
            nullable=True if options.target_field is None else options.target_field.nullable,
            metadata=None if options.target_field is None else options.target_field.metadata,
        )

    elif isinstance(obj, polars.Field):
        return pa.field(
            name=obj.name,
            type=polars_type_to_arrow_type(obj.dtype, options),
            nullable=True if options.target_field is None else options.target_field.nullable,
            metadata=None if options.target_field is None else options.target_field.metadata,
        )

    elif obj in polars_type_classes():
        obj = obj()

        return pa.field(
            name="root" if options.target_field is None else options.target_field.name,
            type=polars_type_to_arrow_type(obj, options),
            nullable=True if options.target_field is None else options.target_field.nullable,
            metadata=None if options.target_field is None else options.target_field.metadata,
        )

    else:
        raise ValueError(f"Cannot convert polars object {type(obj)} to pyarrow.Field")