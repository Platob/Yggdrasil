import dataclasses
import datetime
import decimal
import types
import uuid
from collections.abc import Mapping, MutableMapping, MutableSequence, MutableSet
from typing import Annotated, Any, Tuple, Union, get_args, get_origin

import pyarrow as pa

__all__ = ["arrow_field_from_hint"]


_NONE_TYPE = type(None)

_PRIMITIVE_ARROW_TYPES = {
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
    decimal.Decimal: pa.decimal128(38),
}


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
