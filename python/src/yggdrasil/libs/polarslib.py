from typing import Any, Dict, Optional

import pyarrow as pa

from ..types.cast.registry import register_converter

try:
    import polars  # type: ignore

    polars = polars
    # Primitive Arrow -> Polars dtype mapping (base, non-nested types).
    # These are Polars *dtype classes* (not instances), so they can be used
    # directly in schemas (e.g. pl.Struct({"a": pl.Int64})).
    ARROW_TO_POLARS: Dict[pa.DataType, Any] = {
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
except ImportError:
    polars = None
    ARROW_TO_POLARS = {}


POLARS_BASE_TO_ARROW = {v: k for k, v in ARROW_TO_POLARS.items()}


__all__ = [
    "polars",
    "require_polars",
    "ARROW_TO_POLARS",
    "POLARS_BASE_TO_ARROW",
    "arrow_type_to_polars_type",
    "arrow_field_to_polars_field",
    "polars_type_to_arrow_type",
    "polars_field_to_arrow_field",
]


def require_polars():
    if polars is None:
        raise ImportError(
            "polars is required to use this function. "
            "Install it with `pip install polars`."
        )


def arrow_type_to_polars_type(
    arrow_type: pa.DataType,
    options: Optional[dict] = None,
) -> "polars.DataType":
    """
    Convert a pyarrow.DataType to a Polars dtype.

    Returns a Polars dtype object (e.g. pl.Int64, pl.List(pl.Utf8), ...).
    Raises TypeError for unsupported types.
    """
    import pyarrow.types as pat

    # Fast path: exact primitive mapping
    dtype = ARROW_TO_POLARS.get(arrow_type)
    if dtype is not None:
        return dtype

    # Dictionary -> Categorical (no categories info at dtype level)
    if pat.is_dictionary(arrow_type):
        return polars.Categorical()

    # Map -> represented as List(Struct({"key": ..., "value": ...}))
    if pat.is_map(arrow_type):
        key_type = arrow_type.key_type
        item_type = arrow_type.item_type
        pl_key = arrow_type_to_polars_type(key_type)
        pl_val = arrow_type_to_polars_type(item_type)
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
    if pat.is_list(arrow_type) or pat.is_large_list(arrow_type):
        inner = arrow_type.value_type
        inner_pl = arrow_type_to_polars_type(inner)
        return polars.List(inner_pl)

    # Fixed-size list -> Polars doesn't have a dedicated fixed-list dtype,
    # so we drop the fixed size and use List(inner).
    if pat.is_fixed_size_list(arrow_type):
        inner = arrow_type.value_type
        inner_pl = arrow_type_to_polars_type(inner)
        return polars.List(inner_pl)

    # Struct
    if pat.is_struct(arrow_type):
        field_cls = getattr(polars, "Field", None)
        if callable(field_cls):
            fields = [
                field_cls(f.name, arrow_type_to_polars_type(f.type))
                for f in arrow_type
            ]
            return polars.Struct(fields)
        else:
            return polars.Struct(
                {f.name: arrow_type_to_polars_type(f.type) for f in arrow_type}
            )

    # Timestamp
    if pat.is_timestamp(arrow_type):
        unit = arrow_type.unit  # "s", "ms", "us", "ns"
        tz = arrow_type.tz
        # Polars supports "ns", "us", "ms". Upcast seconds.
        if unit == "s":
            unit = "ms"
        return polars.Datetime(time_unit=unit, time_zone=tz)

    # Duration
    if pat.is_duration(arrow_type):
        unit = arrow_type.unit  # "s", "ms", "us", "ns"
        if unit == "s":
            unit = "ms"
        return polars.Duration(time_unit=unit)

    if pat.is_binary(arrow_type) or pat.is_large_binary(arrow_type) or pat.is_binary_view():
        return polars.Binary()

    if pat.is_string(arrow_type) or pat.is_large_string(arrow_type) or pat.is_string_view():
        return polars.Utf8()

    raise TypeError(f"Unsupported or unknown Arrow type for Polars conversion: {arrow_type!r}")


def arrow_field_to_polars_field(
    field: pa.Field,
    options: Optional[dict] = None,
) -> "polars.Field":
    """
    Convert a pyarrow.Field to a Polars field representation.

    If polars.Field exists, returns a polars.Field(name, dtype).
    """
    built = polars.Field(field.name, arrow_type_to_polars_type(field.type))

    try:
        setattr(built, "nullable", field.nullable)
    except Exception:
        pass

    return built


def _polars_base_type(pl_dtype: Any) -> Any:
    """
    Normalize a Polars dtype or dtype class to its base_type class,
    so we can key into POLARS_BASE_TO_ARROW.
    """
    # dtype is an instance
    base_method = getattr(pl_dtype, "base_type", None)
    if callable(base_method):
        return base_method()
    # dtype is a class (e.g. pl.Int64)
    try:
        instance = pl_dtype()
    except Exception:
        return pl_dtype
    base_method = getattr(instance, "base_type", None)
    if callable(base_method):
        return base_method()
    return pl_dtype


def polars_type_to_arrow_type(
    pl_type: "polars.DataType",
    options: Optional[dict] = None,
) -> pa.DataType:
    """
    Convert a Polars dtype (class or instance) to a pyarrow.DataType.

    Handles primitives via POLARS_BASE_TO_ARROW and common nested/temporal types.
    """
    base = _polars_base_type(pl_type)

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
        arrow_inner = polars_type_to_arrow_type(inner)

        return pa.list_(arrow_inner)

    elif isinstance(pl_type, polars.Struct):
        fields = [
            polars_field_to_arrow_field(_)
            for _ in pl_type.fields
        ]

        return pa.struct(fields)

    # Categorical / Enum -> Arrow dictionary<string>
    if isinstance(pl_type, polars.Categorical) or isinstance(pl_type, polars.Enum):
        # We don't have direct info on categories at the dtype level,
        # so choose a reasonable default: int32 index over string values.
        return pa.dictionary(index_type=pa.int32(), value_type=pa.string())

    raise TypeError(f"Unsupported or unknown Polars dtype for Arrow conversion: {pl_type!r}")


def polars_field_to_arrow_field(
    field: "polars.Field",
    options: Optional["CastOptions"] = None,
) -> pa.Field:
    """
    Convert a Polars field to a pyarrow.Field.

    Accepts:
      - polars.datatypes.Field instances
      - (name, dtype) tuples
    """
    arrow_type = polars_type_to_arrow_type(field.dtype)

    return pa.field(field.name, arrow_type, nullable=getattr(field, "nullable", True))


if polars is not None:
    register_converter(pa.DataType, polars.DataType)(arrow_type_to_polars_type)
    register_converter(pa.Field, polars.Field)(arrow_field_to_polars_field)
    register_converter(polars.DataType, pa.DataType)(polars_type_to_arrow_type)
    register_converter(polars.Field, pa.Field)(polars_field_to_arrow_field)
