import dataclasses
import datetime
import decimal
import types
import uuid
from collections.abc import Mapping, MutableMapping, MutableSequence, MutableSet
from typing import Annotated, Any, Tuple, Union, get_args, get_origin, Optional, Dict

import pyarrow as pa

__all__ = [
    "arrow_field_from_hint",
    "is_arrow_type_binary_like",
    "is_arrow_type_string_like",
    "is_arrow_type_list_like",
]

for key, func in [
    ("string_view", lambda: pa.string()),
    ("binary_view", lambda: pa.binary()),
    ("uuid", lambda: pa.binary(16))
]:
    if not hasattr(pa, key):
        setattr(pa, key, func)


for key, func in [
    ("is_string_view", lambda x: x == pa.string_view()),
    ("is_binary_view", lambda x: x == pa.binary_view()),
    ("is_uuid", lambda x: x == pa.uuid()),
]:
    if not hasattr(pa.types, key):
        setattr(pa.types, key, func)


_NONE_TYPE = type(None)

_PRIMITIVE_ARROW_TYPES = {
    _NONE_TYPE: pa.null(),
    str: pa.string(),
    int: pa.int64(),
    float: pa.float64(),
    bool: pa.bool_(),
    bytes: pa.binary(),
}

_SPECIAL_ARROW_TYPES = {
    datetime.datetime: pa.timestamp("us", tz="UTC"),
    datetime.date: pa.date32(),
    datetime.time: pa.time64("us"),
    datetime.timedelta: pa.duration("us"),
    uuid.UUID: pa.uuid(),
    decimal.Decimal: pa.decimal128(38, 18),
}

_INT_UNITS_ORDER = {"s": 0, "ms": 1, "us": 2, "ns": 3}


def _is_optional(hint) -> bool:
    origin = get_origin(hint)

    if origin is Annotated:
        return _is_optional(get_args(hint)[0])

    if origin in (Union, types.UnionType):
        return _NONE_TYPE in get_args(hint)

    return False


def _strip_optional(hint):
    origin = get_origin(hint)

    if origin is Annotated:
        base_hint, *metadata = get_args(hint)

        if _is_optional(base_hint):
            stripped_base = _strip_optional(base_hint)
            # Using __class_getitem__ to rebuild the Annotated type avoids the
            # unpacking syntax that is unsupported in older Python versions.
            return Annotated.__class_getitem__((stripped_base, *metadata))

        return hint

    if not _is_optional(hint):
        return hint

    return next(arg for arg in get_args(hint) if arg is not _NONE_TYPE)


def _field_name(hint, index: int | None) -> str:
    name = getattr(hint, "__name__", None)

    if name:
        return name

    if index is not None:
        return f"_{index}"

    return str(hint)


def _struct_from_dataclass(hint) -> pa.StructType:
    fields = []

    for field in dataclasses.fields(hint):
        if not field.init or field.name.startswith("_"):
            continue

        fields.append(arrow_field_from_hint(field.type, name=field.name))

    return pa.struct(fields)


def _struct_from_tuple(args, names: list[str] | None = None) -> pa.StructType:
    if names is not None and len(names) != len(args):
        raise TypeError("Tuple metadata names length must match tuple elements")

    return pa.struct(
        [
            arrow_field_from_hint(arg, name=names[idx] if names else f"_{idx + 1}")
            for idx, arg in enumerate(args)
        ]
    )


def _arrow_type_from_metadata(base_hint, metadata):
    merged_metadata: dict[str, Any] = {}

    for item in metadata:
        if isinstance(item, pa.DataType):
            return item

        if isinstance(item, Mapping):
            merged_metadata.update(item)
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
        ):
            merged_metadata[item[0]] = item[1]

    if merged_metadata:
        explicit_type = merged_metadata.get("arrow_type")

        if isinstance(explicit_type, pa.DataType):
            return explicit_type

        if get_origin(base_hint) in (tuple, Tuple):
            names = merged_metadata.get("names")

            if names is not None:
                return _struct_from_tuple(get_args(base_hint), list(names))

        if base_hint is decimal.Decimal:
            precision = merged_metadata.get("precision")

            if precision is not None:
                scale = merged_metadata.get("scale", 0)
                bit_width = merged_metadata.get("bit_width", 128)

                if bit_width > 128:
                    return pa.decimal256(precision, scale)

                return pa.decimal128(precision, scale)

        if base_hint is datetime.datetime:
            unit = merged_metadata.get("unit", "us")
            tz = merged_metadata.get("tz", "UTC")

            return pa.timestamp(unit, tz=tz)

        if base_hint is datetime.time:
            unit = merged_metadata.get("unit", "us")

            return pa.time64(unit)

        if base_hint is datetime.timedelta:
            unit = merged_metadata.get("unit", "us")

            return pa.duration(unit)

        if base_hint is bytes and "length" in merged_metadata:
            return pa.binary(merged_metadata["length"])

    return None


def _arrow_type_from_hint(hint):
    if get_origin(hint) is Annotated:
        base_hint, *metadata = get_args(hint)
        metadata_type = _arrow_type_from_metadata(base_hint, metadata)

        if metadata_type:
            return metadata_type

        return _arrow_type_from_hint(base_hint)

    if hint in _PRIMITIVE_ARROW_TYPES:
        return _PRIMITIVE_ARROW_TYPES[hint]

    if hint in _SPECIAL_ARROW_TYPES:
        return _SPECIAL_ARROW_TYPES[hint]

    origin = get_origin(hint)

    if hint in (list, set, dict, tuple):
        origin = hint

    if dataclasses.is_dataclass(hint):
        return _struct_from_dataclass(hint)

    if origin in (list, MutableSequence, set, MutableSet):
        item_hint = get_args(hint)[0] if get_args(hint) else Any
        return pa.list_(_arrow_type_from_hint(item_hint))

    if origin in (dict, MutableMapping, Mapping):
        key_hint, value_hint = get_args(hint) if get_args(hint) else (str, Any)
        return pa.map_(_arrow_type_from_hint(key_hint), _arrow_type_from_hint(value_hint))

    if origin in (tuple, Tuple):
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return pa.list_(_arrow_type_from_hint(args[0]))

        return _struct_from_tuple(args)

    raise TypeError(f"Cannot determine Arrow type for {hint!r}")


def arrow_field_from_hint(hint, name: str | None = None, index: int | None = None) -> pa.Field:
    nullable = _is_optional(hint)
    base_hint = _strip_optional(hint) if nullable else hint

    field_name = name or _field_name(base_hint, index)
    arrow_type = _arrow_type_from_hint(base_hint)

    return pa.field(field_name, arrow_type, nullable=nullable)


def is_arrow_type_list_like(arrow_type: pa.DataType) -> bool:
    """Check if an Arrow type is list-like."""
    return (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
        or pa.types.is_list_view(arrow_type)
    )


def is_arrow_type_string_like(arrow_type: pa.DataType) -> bool:
    """Check if an Arrow type is string-like."""
    return (
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_string_view(arrow_type)
    )


def is_arrow_type_binary_like(arrow_type: pa.DataType) -> bool:
    """Check if an Arrow type is string-like."""
    return (
        pa.types.is_binary(arrow_type)
        or pa.types.is_large_binary(arrow_type)
        or pa.types.is_fixed_size_binary(arrow_type)
        or pa.types.is_binary_view(arrow_type)
    )



def _merge_metadata(left: Optional[Dict[bytes, bytes]], right: Optional[Dict[bytes, bytes]]) -> Optional[Dict[bytes, bytes]]:
    if not left and not right:
        return None
    out: Dict[bytes, bytes] = {}
    if left:
        out.update(left)
    if right:
        # right wins on conflicts
        out.update(right)
    return out


def _is_null(dt: pa.DataType) -> bool:
    return pa.types.is_null(dt)


def _is_integer(dt: pa.DataType) -> bool:
    return pa.types.is_integer(dt)


def _is_signed_integer(dt: pa.DataType) -> bool:
    return pa.types.is_signed_integer(dt)


def _is_unsigned_integer(dt: pa.DataType) -> bool:
    return pa.types.is_unsigned_integer(dt)


def _is_floating(dt: pa.DataType) -> bool:
    return pa.types.is_floating(dt)


def _int_bit_width(dt: pa.DataType) -> int:
    # int8/int16/int32/int64/uint8/...
    return dt.bit_width


def _digits_for_uint_bits(bits: int) -> int:
    # max uint bits -> decimal digits upper bound:
    # uint64 max = 18446744073709551615 => 20 digits
    # 2**bits - 1 has ceil(bits*log10(2)) digits, use safe upper bound
    import math
    return int(math.ceil(bits * 0.3010299956639812))  # log10(2)


def _promote_int_types(left: pa.DataType, right: pa.DataType) -> pa.DataType:
    """
    Return an integer/decimal type that can represent values from both integer types.
    Strategy:
      - If both signed -> max bit width signed
      - If both unsigned -> max bit width unsigned
      - If mixed signed/unsigned -> try signed with enough bits (unsigned bits + 1)
        - If that exceeds 64, fall back to decimal128(precision>=digits, scale=0)
    """
    l_bits = _int_bit_width(left)
    r_bits = _int_bit_width(right)

    if _is_signed_integer(left) and _is_signed_integer(right):
        bits = max(l_bits, r_bits)
        return {8: pa.int8(), 16: pa.int16(), 32: pa.int32(), 64: pa.int64()}[bits]

    if _is_unsigned_integer(left) and _is_unsigned_integer(right):
        bits = max(l_bits, r_bits)
        return {8: pa.uint8(), 16: pa.uint16(), 32: pa.uint32(), 64: pa.uint64()}[bits]

    # mixed signed/unsigned
    u_bits = max(l_bits if _is_unsigned_integer(left) else 0,
                 r_bits if _is_unsigned_integer(right) else 0)
    s_bits = max(l_bits if _is_signed_integer(left) else 0,
                 r_bits if _is_signed_integer(right) else 0)

    # to hold uintN in signed, need N+1 bits
    needed_signed_bits = max(s_bits, u_bits + 1)

    if needed_signed_bits <= 64:
        # choose smallest signed type that can hold needed bits
        if needed_signed_bits <= 8:
            return pa.int8()
        if needed_signed_bits <= 16:
            return pa.int16()
        if needed_signed_bits <= 32:
            return pa.int32()
        return pa.int64()

    # uint64 + int64 is the classic overflow problem. Use decimal.
    # precision: digits to represent max uint bits + possible sign.
    digits = _digits_for_uint_bits(u_bits)
    # signed needs one extra digit sometimes; be conservative
    digits = max(digits + 1, 20)
    if digits <= 38:
        return pa.decimal128(digits, 0)
    return pa.decimal256(digits, 0)


def _promote_decimal_types(left: pa.Decimal128Type | pa.Decimal256Type,
                           right: pa.Decimal128Type | pa.Decimal256Type) -> pa.DataType:
    # Match scale, then set precision to fit both after scale alignment.
    scale = max(left.scale, right.scale)

    def adj_precision(d: pa.DataType) -> int:
        # Increasing scale can require increasing precision to keep same integer digits.
        # integer_digits = precision - scale
        integer_digits = d.precision - d.scale
        return integer_digits + scale

    precision = max(adj_precision(left), adj_precision(right))

    # Keep it safe: if precision > 38 use decimal256
    if pa.types.is_decimal256(left) or pa.types.is_decimal256(right) or precision > 38:
        return pa.decimal256(precision, scale)
    return pa.decimal128(precision, scale)


def _promote_numeric(left: pa.DataType, right: pa.DataType) -> pa.DataType:
    # decimal dominates ints/floats if present? Depends on your semantics.
    # Here: decimals keep exactness when mixing with ints; floats win when mixing float+anything non-decimal.
    if pa.types.is_decimal(left) and pa.types.is_decimal(right):
        return _promote_decimal_types(left, right)

    if pa.types.is_decimal(left) and _is_integer(right):
        # decimal vs int: keep decimal, ensure enough precision
        dec: pa.DataType = left
        # add digits for int range (safe upper bound)
        needed = _digits_for_uint_bits(_int_bit_width(right)) + (1 if _is_signed_integer(right) else 0)
        precision = max(dec.precision, max(needed, 1) + dec.scale)
        if pa.types.is_decimal256(dec) or precision > 38:
            return pa.decimal256(precision, dec.scale)
        return pa.decimal128(precision, dec.scale)

    if pa.types.is_decimal(right) and _is_integer(left):
        return _promote_numeric(right, left)

    if _is_floating(left) or _is_floating(right):
        # Be boring and safe: float64 if any float involved (avoids int overflow on cast paths).
        return pa.float64()

    # integer/integer
    return _promote_int_types(left, right)


def _merge_time_units(left_unit: str, right_unit: str) -> str:
    # choose finer resolution (higher order index)
    return left_unit if _INT_UNITS_ORDER[left_unit] >= _INT_UNITS_ORDER[right_unit] else right_unit


def merge_arrow_types(
    left: Union[pa.DataType, pa.TimestampType, pa.ListType, pa.MapType, pa.StructType],
    right: Union[pa.DataType, pa.TimestampType, pa.ListType, pa.MapType, pa.StructType],
    add_missing_columns: bool = True
) -> pa.DataType:
    # null is identity
    if _is_null(left):
        return right
    if _is_null(right):
        return left

    # identical => done
    if left.equals(right):
        return left

    # Extension types: keep only if truly same extension, else fall back to storage merge.
    if pa.types.is_extension(left) or pa.types.is_extension(right):
        if pa.types.is_extension(left) and pa.types.is_extension(right):
            if type(left) is type(right) and left.extension_name == right.extension_name:
                # assume compatible metadata; keep left
                return left
        # otherwise merge storage and drop extension wrapper
        l_storage = left.storage_type if pa.types.is_extension(left) else left
        r_storage = right.storage_type if pa.types.is_extension(right) else right
        return merge_arrow_types(l_storage, r_storage, add_missing_columns=add_missing_columns)

    # numeric promotion (int/float/decimal)
    if pa.types.is_decimal(left) or pa.types.is_decimal(right) or _is_integer(left) or _is_integer(right) or _is_floating(left) or _is_floating(right):
        # only if both are numeric-ish; otherwise fall through
        if (pa.types.is_decimal(left) or _is_integer(left) or _is_floating(left)) and (pa.types.is_decimal(right) or _is_integer(right) or _is_floating(right)):
            return _promote_numeric(left, right)

    # booleans
    if pa.types.is_boolean(left) and pa.types.is_boolean(right):
        return pa.bool_()

    # strings / binaries
    if pa.types.is_string(left) and pa.types.is_string(right):
        return pa.large_string() if (pa.types.is_large_string(left) or pa.types.is_large_string(right)) else pa.string()
    if pa.types.is_binary(left) and pa.types.is_binary(right):
        return pa.large_binary() if (pa.types.is_large_binary(left) or pa.types.is_large_binary(right)) else pa.binary()

    # timestamps
    if pa.types.is_timestamp(left) and pa.types.is_timestamp(right):
        if left.tz != right.tz:
            raise TypeError(f"Cannot merge timestamp timezones: {left.tz!r} vs {right.tz!r}")
        unit = _merge_time_units(left.unit, right.unit)
        return pa.timestamp(unit, tz=left.tz)

    # dates
    if pa.types.is_date32(left) and pa.types.is_date32(right):
        return pa.date32()
    if (pa.types.is_date32(left) and pa.types.is_date64(right)) or (pa.types.is_date64(left) and pa.types.is_date32(right)) or (pa.types.is_date64(left) and pa.types.is_date64(right)):
        return pa.date64()

    # times
    if pa.types.is_time32(left) and pa.types.is_time32(right):
        unit = _merge_time_units(left.unit, right.unit)  # s/ms
        # time32 supports s/ms only; if unit becomes us/ns, upgrade to time64
        if unit in ("s", "ms"):
            return pa.time32(unit)
        return pa.time64("us")
    if pa.types.is_time64(left) and pa.types.is_time64(right):
        unit = _merge_time_units(left.unit, right.unit)  # us/ns
        return pa.time64(unit)
    if (pa.types.is_time32(left) and pa.types.is_time64(right)) or (pa.types.is_time64(left) and pa.types.is_time32(right)):
        # upgrade to time64, choose finer among both
        l_unit = left.unit
        r_unit = right.unit
        unit = _merge_time_units(l_unit, r_unit)
        return pa.time64(unit if unit in ("us", "ns") else "us")

    # duration
    if pa.types.is_duration(left) and pa.types.is_duration(right):
        unit = _merge_time_units(left.unit, right.unit)
        return pa.duration(unit)

    # lists
    if pa.types.is_list(left) and pa.types.is_list(right):
        item = merge_arrow_fields(left.value_field, right.value_field, add_missing_columns=add_missing_columns)
        if pa.types.is_large_list(left) or pa.types.is_large_list(right):
            return pa.large_list(item)
        return pa.list_(item)

    # large list + list mix
    if (pa.types.is_list(left) and pa.types.is_large_list(right)) or (pa.types.is_large_list(left) and pa.types.is_list(right)):
        l = left if pa.types.is_large_list(left) or pa.types.is_list(left) else None
        r = right if pa.types.is_large_list(right) or pa.types.is_list(right) else None
        item = merge_arrow_fields(l.value_field, r.value_field, add_missing_columns=add_missing_columns)
        return pa.large_list(item)

    # fixed_size_list
    if pa.types.is_fixed_size_list(left) and pa.types.is_fixed_size_list(right):
        if left.list_size == right.list_size:
            item = merge_arrow_fields(left.value_field, right.value_field, add_missing_columns=add_missing_columns)
            return pa.fixed_size_list(item, left.list_size)
        # incompatible sizes -> degrade to normal list
        item = merge_arrow_fields(left.value_field, right.value_field, add_missing_columns=add_missing_columns)
        return pa.list_(item)

    # structs (diagonal schema merge)
    if pa.types.is_struct(left) and pa.types.is_struct(right):
        l_fields = {f.name: f for f in left}
        r_fields = {f.name: f for f in right}
        if add_missing_columns:
            names = sorted(set(l_fields) | set(r_fields))
        else:
            names = sorted(set(l_fields) & set(r_fields))

        out_fields = []
        for name in names:
            lf = l_fields.get(name)
            rf = r_fields.get(name)
            if lf is None:
                out_fields.append(rf)
            elif rf is None:
                out_fields.append(lf)
            else:
                out_fields.append(merge_arrow_fields(lf, rf, add_missing_columns=add_missing_columns))
        return pa.struct(out_fields)

    # maps
    if pa.types.is_map(left) and pa.types.is_map(right):
        # Arrow map is list<struct<key,value>> under the hood, but pyarrow exposes key/value types.
        # Keys should be non-nullable; we enforce that.
        key_t = merge_arrow_types(left.key_type, right.key_type, add_missing_columns=add_missing_columns)
        item_t = merge_arrow_types(left.item_type, right.item_type, add_missing_columns=add_missing_columns)
        return pa.map_(pa.field("key", key_t, nullable=False),
                       pa.field("value", item_t, nullable=True),
                       keys_sorted=bool(left.keys_sorted and right.keys_sorted))

    # dictionary
    if pa.types.is_dictionary(left) and pa.types.is_dictionary(right):
        # merge index type + value type
        idx = merge_arrow_types(left.index_type, right.index_type, add_missing_columns=add_missing_columns)
        if not _is_integer(idx):
            raise TypeError(f"Dictionary index types must be integer; got {idx}")
        val = merge_arrow_types(left.value_type, right.value_type, add_missing_columns=add_missing_columns)
        ordered = bool(left.ordered and right.ordered)
        return pa.dictionary(idx, val, ordered=ordered)

    # If we reach here, it's genuinely incompatible (or unsupported).
    raise TypeError(f"Cannot merge Arrow types: {left} vs {right}")


def merge_arrow_fields(
    left: pa.Field,
    right: pa.Field,
    add_missing_columns: bool = True
) -> pa.Field:
    if left.name != right.name:
        raise TypeError(f"Cannot merge fields with different names: {left.name!r} vs {right.name!r}")

    merged_type = merge_arrow_types(left.type, right.type, add_missing_columns=add_missing_columns)
    nullable = bool(left.nullable or right.nullable)
    metadata = _merge_metadata(left.metadata, right.metadata)

    # preserve dictionary-encoding? handled in merge_arrow_types
    return pa.field(left.name, merged_type, nullable=nullable, metadata=metadata)
