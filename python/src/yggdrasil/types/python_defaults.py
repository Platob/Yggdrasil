import dataclasses
import datetime
import decimal
import types
import uuid
from collections.abc import Collection, Mapping, MutableMapping, MutableSequence, MutableSet
from typing import Any, Tuple, Union, get_args, get_origin

import pyarrow as pa

__all__ = [
    "default_from_hint",
    "default_from_arrow_hint"
]


_NONE_TYPE = type(None)
_PRIMITIVE_DEFAULTS = {
    str: "",
    int: 0,
    float: 0.0,
    bool: False,
    bytes: b"",
}

_SPECIAL_DEFAULTS = {
    datetime.datetime: lambda: datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc),
    datetime.date: lambda: datetime.date(1970, 1, 1),
    datetime.time: lambda: datetime.time(0, 0, 0, tzinfo=datetime.timezone.utc),
    datetime.timedelta: lambda: datetime.timedelta(0),
    uuid.UUID: lambda: uuid.UUID(int=0),
    decimal.Decimal: lambda: decimal.Decimal(0),
}


def _is_optional(hint) -> bool:
    origin = get_origin(hint)

    if origin in (Union, types.UnionType):
        return _NONE_TYPE in get_args(hint)

    return False


def _default_for_collection(origin):
    if origin in (list, MutableSequence):
        return []

    if origin in (set, MutableSet):
        return set()

    if origin in (dict, MutableMapping, Mapping):
        return {}

    if origin in (tuple, Tuple):
        return tuple()

    if origin and issubclass(origin, Collection):
        return origin()

    return None


def _default_for_tuple_args(args):
    if not args:
        return tuple()

    if len(args) == 2 and args[1] is Ellipsis:
        return tuple()

    return tuple(default_from_hint(arg) for arg in args)


def default_from_arrow_hint(hint):
    def _arrow_default_value(dtype: "pa.DataType"):
        if pa.types.is_struct(dtype):
            return {
                field.name: (
                    _arrow_default_value(field.type) if not field.nullable else None
                )
                for field in dtype
            }

        if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
            return []

        if pa.types.is_map(dtype):
            return {}

        if pa.types.is_integer(dtype) or pa.types.is_unsigned_integer(dtype):
            return 0

        if pa.types.is_floating(dtype) or pa.types.is_decimal(dtype):
            return decimal.Decimal(0) if pa.types.is_decimal(dtype) else 0.0

        if pa.types.is_boolean(dtype):
            return False

        if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) or pa.types.is_string_view(dtype):
            return ""

        if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype) or pa.types.is_binary_view(dtype):
            return b""

        if pa.types.is_fixed_size_binary(dtype):
            return b"\x00" * dtype.byte_width

        if (
            pa.types.is_timestamp(dtype)
            or pa.types.is_time(dtype)
            or pa.types.is_duration(dtype)
            or pa.types.is_interval(dtype)
        ):
            return 0

        return None

    def _arrow_default_scalar(dtype: "pa.DataType"):
        value = _arrow_default_value(dtype)
        return pa.scalar(value, type=dtype)

    if isinstance(hint, pa.Field):
        return pa.scalar(None, type=hint.type) if hint.nullable else _arrow_default_scalar(hint.type)

    if isinstance(hint, pa.DataType):
        return _arrow_default_scalar(hint)

    return dataclasses.MISSING


def _default_for_dataclass(hint):
    kwargs = {}

    for field in dataclasses.fields(hint):
        if not field.init or field.name.startswith("_"):
            continue

        if field.default is not dataclasses.MISSING:
            value = field.default
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            value = field.default_factory()  # type: ignore[misc]
        else:
            value = default_from_hint(field.type)

        kwargs[field.name] = value

    return hint(**kwargs)


def default_from_hint(hint: Any):
    if _is_optional(hint):
        return None

    if hint in _PRIMITIVE_DEFAULTS:
        return _PRIMITIVE_DEFAULTS[hint]

    if hint in _SPECIAL_DEFAULTS:
        return _SPECIAL_DEFAULTS[hint]()

    arrow_default = default_from_arrow_hint(hint)
    if arrow_default is not dataclasses.MISSING:
        return arrow_default

    origin = get_origin(hint)

    if hint in (list, set, dict, tuple):
        origin = hint

    if origin:
        if origin in (tuple, Tuple):
            return _default_for_tuple_args(get_args(hint))

        collection_default = _default_for_collection(origin)
        if collection_default is not None:
            return collection_default

    if dataclasses.is_dataclass(hint):
        return _default_for_dataclass(hint)

    try:
        return hint()
    except Exception as exc:
        raise TypeError(f"Cannot determine default value for {hint!r}") from exc
