import dataclasses
import datetime
import decimal
import types
import uuid
from collections.abc import Mapping, MutableMapping, MutableSequence, MutableSet
from typing import Any, Tuple, Union, get_args, get_origin

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

    if origin in (Union, types.UnionType):
        return _NONE_TYPE in get_args(hint)

    return False


def _strip_optional(hint):
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
        if not field.init:
            continue

        fields.append(arrow_field_from_hint(field.type, name=field.name))

    return pa.struct(fields)


def _struct_from_tuple(args) -> pa.StructType:
    return pa.struct(
        [arrow_field_from_hint(arg, name=f"_{idx}") for idx, arg in enumerate(args, start=1)]
    )


def _arrow_type_from_hint(hint):
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
