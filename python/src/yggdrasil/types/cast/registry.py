from __future__ import annotations

import builtins
import dataclasses as _dataclasses
import datetime as _datetime
import enum
import inspect
import re
import types
from collections.abc import Iterable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints, Optional, List,
)

import pyarrow as pa

if TYPE_CHECKING:
    from .arrow import ArrowCastOptions

__all__ = ["register_converter", "convert"]


Converter = Callable[[Any, "ArrowCastOptions | dict | None"], Any]


_registry: Dict[Tuple[Any, Any], Converter] = {}


def register_converter(
    from_hint: Union[Any, List[Any]],
    to_hint: Any
) -> Callable[[Callable[..., Any]], Converter]:
    """Register a converter from ``from_hint`` to ``to_hint``.

    The decorated callable receives ``(value, options)`` and should return the
    converted value.
    """

    def decorator(func: Callable[..., Any]) -> Converter:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if any(
            param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            for param in params
        ):
            raise TypeError(
                "Converters must accept exactly two positional arguments: (value, options)"
            )

        if len(params) != 2:
            raise TypeError(
                "Converters must accept exactly two positional arguments: (value, options)"
            )

        if isinstance(from_hint, list):
            for h in from_hint:
                _registry[(h, to_hint)] = func
        else:
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


def _iter_mro(tp: Any) -> Iterable[Any]:
    """Return (tp, …) including its MRO if it's a class-like object."""
    try:
        mro = getattr(tp, "__mro__", None)
    except TypeError:
        mro = None

    if mro is None:
        # Not a normal class, just treat as single value
        return (tp,)
    return mro  # includes tp itself as mro[0]


def _find_converter(from_type: Any, to_hint: Any) -> "Converter | None":
    # Fast path: exact match
    conv = _registry.get((from_type, to_hint))
    if conv is not None:
        return conv

    # Build from_mro and to_mro
    from_mro = _iter_mro(from_type)
    to_mro = _iter_mro(to_hint)

    # 1) Try direct dict lookup on all (from_mro × to_mro) combinations.
    #    This is O(len(MRO)^2) but much cheaper than scanning the whole registry.
    for f in from_mro:
        for t in to_mro:
            conv = _registry.get((f, t))
            if conv is not None:
                return conv

    # 2) Fallback: scan registry with issubclass for more exotic keys
    #    (e.g. using typing hints, Protocols, etc.)
    for (registered_from, registered_to), conv in _registry.items():
        try:
            # registered_from may be a base of from_type
            from_ok = (
                from_type is registered_from
                or (
                    isinstance(registered_from, type)
                    and isinstance(from_type, type)
                    and issubclass(from_type, registered_from)
                )
            )

            # registered_to may be a base of to_hint
            to_ok = (
                to_hint is registered_to
                or (
                    isinstance(registered_to, type)
                    and isinstance(to_hint, type)
                    and issubclass(to_hint, registered_to)
                )
            )

            if from_ok and to_ok:
                return conv
        except TypeError:
            # Some registered types (e.g. typing constructs) may explode in issubclass
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


def convert(
    value: Any,
    target_hint: Union[
        type,
        pa.Field, pa.DataType, pa.Schema,
    ],
    options: Optional[ArrowCastOptions] = None,
    **kwargs,
) -> Any:
    """Convert ``value`` to ``target_hint`` using the registered converters."""
    from yggdrasil.types import default_from_hint
    from yggdrasil.types.cast.arrow import ArrowCastOptions

    is_optional, target_hint = _unwrap_optional(target_hint)

    if options is None and not kwargs:
        try:
            if isinstance(value, target_hint):
                return value
        except:
            pass

        if value is None:
            return None if is_optional else default_from_hint(target_hint)

    options = ArrowCastOptions.check_arg(arg=options, kwargs=kwargs)
    origin = get_origin(target_hint) or target_hint
    args = get_args(target_hint)
    source_hint = type(value)

    if isinstance(target_hint, (pa.Field, pa.DataType, pa.Schema)):
        options.target_field = convert(target_hint, pa.Field)

        converter = _find_converter(source_hint, source_hint)
    else:
        converter = _find_converter(source_hint, target_hint)

    if converter is not None:
        return converter(value, options)

    if isinstance(target_hint, type) and issubclass(target_hint, enum.Enum):
        if isinstance(value, target_hint):
            return value

        if isinstance(value, str):
            for member in target_hint:
                if member.name.lower() == value.lower():
                    return member

        try:
            first_member = next(iter(target_hint))
        except StopIteration:
            raise TypeError(f"Cannot convert to empty Enum {target_hint.__name__}")

        try:
            converted_value = convert(
                value,
                type(first_member.value),
                options=options,
            )
        except Exception:
            converted_value = value

        for member in target_hint:
            if member.value == converted_value:
                return member

        raise TypeError(f"No matching Enum member for {value!r} in {target_hint.__name__}")

    if isinstance(target_hint, type) and _dataclasses.is_dataclass(target_hint):
        if isinstance(value, target_hint):
            return value

        if not isinstance(value, Mapping):
            raise TypeError(f"Cannot convert {type(value)} to dataclass {target_hint.__name__}")

        hints = get_type_hints(target_hint)
        kwargs = {}
        for field in _dataclasses.fields(target_hint):
            if not field.init or field.name.startswith("_"):
                continue

            if field.name in value:
                field_value = convert(
                    value[field.name],
                    hints.get(field.name, Any),
                    options=options,
                )
            elif field.default is not _dataclasses.MISSING:
                field_value = field.default
            elif field.default_factory is not _dataclasses.MISSING:  # type: ignore[attr-defined]
                field_value = field.default_factory()  # type: ignore[misc]
            else:
                field_value = default_from_hint(field.type)

            kwargs[field.name] = field_value

        return target_hint(**kwargs)

    if origin in {list, set}:
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise TypeError(f"Cannot convert {type(value)} to {origin.__name__}")

        element_hint = args[0] if args else Any
        converted = [
            convert(item, element_hint, options=options)
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
                convert(item, element_hint, options=options)
                for item in values
            )

        if args and len(args) != len(values):
            raise TypeError("Tuple length does not match target annotation")

        return tuple(
            convert(
                item,
                target_hint=args[idx] if args else Any,
                options=options,
            )
            for idx, item in enumerate(values)
        )

    if origin in {dict, Mapping}:
        if not isinstance(value, Mapping):
            raise TypeError("Cannot convert non-mapping to dict")

        key_hint, value_hint = (args + (Any, Any))[:2]
        mapping_ctor = dict if origin is Mapping else origin
        return mapping_ctor(
            (
                convert(key, key_hint, options=options),
                convert(val, value_hint, options=options),
            )
            for key, val in value.items()
        )

    if target_hint is Any or isinstance(value, target_hint):
        return value

    raise TypeError(f"No converter registered for {type(value)} -> {target_hint}")


@register_converter(str, int)
def _str_to_int(value: str, cast_options: Any) -> int:
    if value == "":
        return 0
    return int(value)


@register_converter(str, float)
def _str_to_float(value: str, cast_options: Any) -> float:
    default_value = getattr(cast_options, "default_value", None)
    if value == "" and default_value is not None:
        return default_value
    return float(value)


@register_converter(str, bool)
def _str_to_bool(value: str, cast_options: Any) -> bool:
    default_value = getattr(cast_options, "default_value", None)
    if value == "" and default_value is not None:
        return default_value

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "t"}:
        return True
    if normalized in {"false", "0", "no", "n", "f"}:
        return False

    raise ValueError(f"Cannot parse boolean from {value!r}")


@register_converter(str, _datetime.date)
def _str_to_date(value: str, cast_options: Any) -> _datetime.date:
    default_value = getattr(cast_options, "default_value", None)
    if value == "" and default_value is not None:
        return default_value
    return _datetime.date.fromisoformat(value)


@register_converter(str, _datetime.datetime)
def _str_to_datetime(value: str, cast_options: Any) -> _datetime.datetime:
    default_value = getattr(cast_options, "default_value", None)
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


@register_converter(str, _datetime.time)
def _str_to_time(value: str, cast_options: Any) -> _datetime.time:
    default_value = getattr(cast_options, "default_value", None)
    if value == "" and default_value is not None:
        return default_value
    return _datetime.time.fromisoformat(value)


@register_converter(_datetime.datetime, _datetime.date)
def _datetime_to_date(value: _datetime.datetime, cast_options: Any) -> _datetime.date:
    return value.date()


@register_converter(int, str)
def _int_to_str(value: int, cast_options: Any) -> str:
    return str(value)


if not getattr(builtins, "register_converter", None):
    setattr(builtins, "register_converter", register_converter)


if not getattr(builtins, "convert", None):
    setattr(builtins, "convert", convert)
