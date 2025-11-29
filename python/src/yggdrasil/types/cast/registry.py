from __future__ import annotations

import datetime as _datetime
import re
import types
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Dict, Tuple, Union, get_args, get_origin

__all__ = ["register", "convert"]


Converter = Callable[[Any, Any, Any], Any]


_registry: Dict[Tuple[Any, Any], Converter] = {}


def register(from_hint: Any, to_hint: Any) -> Callable[[Converter], Converter]:
    """Register a converter from ``from_hint`` to ``to_hint``.

    The decorated callable receives ``(value, cast_options, default_value)`` and
    should return the converted value.
    """

    def decorator(func: Converter) -> Converter:
        _registry[(from_hint, to_hint)] = func
        return func

    return decorator


def _unwrap_optional(hint: Any) -> Tuple[bool, Any]:
    origin = get_origin(hint)
    if origin in {Union, types.UnionType}:
        args = get_args(hint)
        non_none = [arg for arg in args if arg is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return True, non_none[0]

    return False, hint


def _find_converter(from_value: Any, to_hint: Any) -> Converter | None:
    from_type = type(from_value)

    if (from_type, to_hint) in _registry:
        return _registry[(from_type, to_hint)]

    for (registered_from, registered_to), converter in _registry.items():
        try:
            if isinstance(from_value, registered_from) and to_hint == registered_to:
                return converter
        except TypeError:
            # ``registered_from`` might not be usable with ``isinstance`` (e.g. typing hints)
            continue

    return None


def _normalize_fractional_seconds(value: str) -> str:
    match = re.search(r"(\.)(\d+)(?=(?:[+-]\d{2}:?\d{2})?$)", value)
    if not match:
        return value

    start, end = match.span(2)
    fraction = match.group(2)
    normalized_fraction = fraction[:6].ljust(6, "0")
    return value[:start] + normalized_fraction + value[end:]


def convert(value: Any, target_hint: Any, *, cast_options: Any = None, default_value: Any = None) -> Any:
    """Convert ``value`` to ``target_hint`` using the registered converters."""

    is_optional, inner_hint = _unwrap_optional(target_hint)
    if is_optional and (value is None or value == ""):
        return None

    target = inner_hint if is_optional else target_hint

    origin = get_origin(target) or target
    args = get_args(target)

    if origin in {list, set}:
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise TypeError(f"Cannot convert {type(value)} to {origin.__name__}")

        element_hint = args[0] if args else Any
        converted = [
            convert(item, element_hint, cast_options=cast_options, default_value=default_value)
            for item in value
        ]
        return origin(converted)

    if origin is tuple:
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise TypeError("Cannot convert non-iterable to tuple")

        values = tuple(value)
        if len(args) == 2 and args[1] is Ellipsis:
            element_hint = args[0]
            return tuple(
                convert(item, element_hint, cast_options=cast_options, default_value=default_value)
                for item in values
            )

        if args and len(args) != len(values):
            raise TypeError("Tuple length does not match target annotation")

        return tuple(
            convert(item, args[idx] if args else Any, cast_options=cast_options, default_value=default_value)
            for idx, item in enumerate(values)
        )

    if origin in {dict, Mapping}:
        if not isinstance(value, Mapping):
            raise TypeError("Cannot convert non-mapping to dict")

        key_hint, value_hint = (args + (Any, Any))[:2]
        mapping_ctor = dict if origin is Mapping else origin
        return mapping_ctor(
            (
                convert(key, key_hint, cast_options=cast_options, default_value=default_value),
                convert(val, value_hint, cast_options=cast_options, default_value=default_value),
            )
            for key, val in value.items()
        )

    if target is Any or isinstance(value, target):
        return value

    converter = _find_converter(value, target)
    if converter is None:
        raise TypeError(f"No converter registered for {type(value)} -> {target}")

    return converter(value, cast_options, default_value)


@register(str, int)
def _str_to_int(value: str, cast_options: Any, default_value: Any) -> int:
    if value == "" and default_value is not None:
        return default_value
    return int(value)


@register(str, float)
def _str_to_float(value: str, cast_options: Any, default_value: Any) -> float:
    if value == "" and default_value is not None:
        return default_value
    return float(value)


@register(str, bool)
def _str_to_bool(value: str, cast_options: Any, default_value: Any) -> bool:
    if value == "" and default_value is not None:
        return default_value

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "t"}:
        return True
    if normalized in {"false", "0", "no", "n", "f"}:
        return False

    raise ValueError(f"Cannot parse boolean from {value!r}")


@register(str, _datetime.date)
def _str_to_date(value: str, cast_options: Any, default_value: Any) -> _datetime.date:
    if value == "" and default_value is not None:
        return default_value
    return _datetime.date.fromisoformat(value)


@register(str, _datetime.datetime)
def _str_to_datetime(value: str, cast_options: Any, default_value: Any) -> _datetime.datetime:
    if value == "" and default_value is not None:
        return default_value

    normalized = value.strip()
    if normalized == "now":
        return _datetime.datetime.now(tz=_datetime.timezone.utc)

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    normalized = _normalize_fractional_seconds(normalized)

    try:
        parsed = _datetime.datetime.fromisoformat(normalized)
    except ValueError:
        formats = [
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
        ]
        last_error: ValueError | None = None
        for fmt in formats:
            try:
                parsed = _datetime.datetime.strptime(normalized, fmt)
                break
            except ValueError as exc:  # pragma: no cover - inspected via test fallback
                last_error = exc
        else:
            raise last_error if last_error is not None else ValueError(
                f"Cannot parse datetime from {value!r}"
            )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_datetime.timezone.utc)

    return parsed


@register(str, _datetime.time)
def _str_to_time(value: str, cast_options: Any, default_value: Any) -> _datetime.time:
    if value == "" and default_value is not None:
        return default_value
    return _datetime.time.fromisoformat(value)


@register(_datetime.datetime, _datetime.date)
def _datetime_to_date(value: _datetime.datetime, cast_options: Any, default_value: Any) -> _datetime.date:
    return value.date()


@register(int, str)
def _int_to_str(value: int, cast_options: Any, default_value: Any) -> str:
    return str(value)
