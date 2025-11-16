import polars as pl
import pyarrow as pa

__all__ = [
    "POLARS_TO_ARROW_TYPE_MAP",
    "polars_to_arrow_type"
]

POLARS_TO_ARROW_TYPE_MAP = {
    # Boolean
    pl.Boolean(): pa.bool_(),

    # Integers
    pl.Int8(): pa.int8(),
    pl.Int16(): pa.int16(),
    pl.Int32(): pa.int32(),
    pl.Int64(): pa.int64(),
    pl.UInt8(): pa.uint8(),
    pl.UInt16(): pa.uint16(),
    pl.UInt32(): pa.uint32(),
    pl.UInt64(): pa.uint64(),

    # Floats
    pl.Float32(): pa.float32(),
    pl.Float64(): pa.float64(),

    # Strings / binary
    pl.Utf8(): pa.string(),
    pl.Binary(): pa.binary(),

    # Date / Time
    pl.Date(): pa.date32(),

    # Optional / Null
    pl.Null(): pa.null()
}


def polars_to_arrow_type(polars_type: pl.DataType) -> pa.DataType:
    """
    Convert a Polars dtype to PyArrow type.
    Uses dict lookup for primitives, falls back to handle nested/complex types.
    """
    # direct dict lookup
    result = POLARS_TO_ARROW_TYPE_MAP.get(polars_type)
    if result is not None:
        return result

    if isinstance(polars_type, pl.Datetime):
        return pa.timestamp(unit=polars_type.time_unit, tz=polars_type.time_zone)

    if isinstance(polars_type, pl.Time):
        return pa.time64("ns")



    if isinstance(polars_type, pl.Decimal):
        if polars_type.precision >= 38:
            return pa.decimal256(polars_type.precision, polars_type.scale)
        return pa.decimal128(polars_type.precision, polars_type.scale)

    # handle nested / complex types
    if isinstance(polars_type, pl.List):
        # recursively map inner type
        inner = polars_to_arrow_type(polars_type.inner)
        return pa.list_(inner)
    if isinstance(polars_type, pl.Categorical):
        # Arrow dictionary type: keys=int32, values=str
        return pa.dictionary(index_type=pa.int32(), value_type=pa.string())
    if isinstance(polars_type, pl.Struct):
        fields = [
            (pl_field.name, polars_to_arrow_type(pl_field.dtype))
            for pl_field in polars_type.fields
        ]
        return pa.struct(fields)

    raise TypeError(f"Unsupported or unknown Polars type: {polars_type}")
