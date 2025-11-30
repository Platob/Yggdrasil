from __future__ import annotations

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
    get_type_hints, Optional,
)

import pyarrow as pa


if TYPE_CHECKING:
    from .arrow import ArrowCastOptions

__all__ = ["register_converter", "convert"]


Converter = Callable[[Any, "ArrowCastOptions | dict | None"], Any]


_registry: Dict[Tuple[Any, Any], Converter] = {}


def register_converter(from_hint: Any, to_hint: Any) -> Callable[[Callable[..., Any]], Converter]:
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

        def wrapped(value: Any, options: "ArrowCastOptions | dict | None") -> Any:
            return func(value, options)

        _registry[(from_hint, to_hint)] = wrapped
        return wrapped

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


def convert(
    value: Any,
    target_hint: Any,
    *,
    options: Optional[Union[ArrowCastOptions, dict]] = None,
    default_value: Any = None,
) -> Any:
    """Convert ``value`` to ``target_hint`` using the registered converters."""
    from yggdrasil.types import default_from_hint

    arrow_hint_types = (pa.DataType, pa.Field, pa.Schema)

    try:
        from yggdrasil.types.cast.arrow import ArrowCastOptions
    except Exception:
        ArrowCastOptions = None  # type: ignore[assignment]

    is_optional, inner_hint = _unwrap_optional(target_hint)
    if is_optional and (value is None or value == ""):
        return None

    target_hint_value = inner_hint if is_optional else target_hint
    target_arrow_hint = (
        target_hint_value if isinstance(target_hint_value, arrow_hint_types) else None
    )
    source_hint: Any | None = None

    options_arg = options
    if ArrowCastOptions is not None:
        options = ArrowCastOptions.check_arg(options_arg)

        if options.target_hint is not None and not isinstance(options_arg, ArrowCastOptions):
            target_hint_value = options.target_hint
            target_arrow_hint = (
                target_hint_value
                if isinstance(target_hint_value, arrow_hint_types)
                else None
            )

        source_hint = options.source_hint

        replacements: dict[str, object] = {}

        if target_arrow_hint is not None and options.target_field is None:
            replacements["target_field"] = target_arrow_hint

        if (
            source_hint is not None
            and options.source_field is None
            and isinstance(source_hint, arrow_hint_types)
        ):
            replacements["source_field"] = source_hint

        if options.target_hint is None or options.target_hint != target_hint_value:
            replacements["target_hint"] = target_hint_value

        if options.source_hint is None and source_hint is not None:
            replacements["source_hint"] = source_hint

        if default_value is not None and options.default_value is None:
            replacements["default_value"] = default_value

        if replacements:
            options = _dataclasses.replace(options, **replacements)

    target = target_hint_value
    if target_arrow_hint is not None:
        if isinstance(value, pa.ChunkedArray):
            target = pa.ChunkedArray
        elif isinstance(value, pa.Table):
            target = pa.Table
        elif isinstance(value, pa.RecordBatch):
            target = pa.RecordBatch
        elif isinstance(value, pa.RecordBatchReader):
            target = pa.RecordBatchReader
        elif isinstance(value, pa.Array):
            target = pa.Array
        else:
            target = pa.Array

    origin = get_origin(target) or target
    args = get_args(target)

    if isinstance(target, type) and issubclass(target, enum.Enum):
        if isinstance(value, target):
            return value

        if isinstance(value, str):
            for member in target:
                if member.name.lower() == value.lower():
                    return member

        try:
            first_member = next(iter(target))
        except StopIteration:
            raise TypeError(f"Cannot convert to empty Enum {target.__name__}")

        try:
            converted_value = convert(
                value,
                type(first_member.value),
                options=options,
                default_value=default_value,
            )
        except Exception:
            converted_value = value

        for member in target:
            if member.value == converted_value:
                return member

        raise TypeError(f"No matching Enum member for {value!r} in {target.__name__}")

    if isinstance(target, type) and _dataclasses.is_dataclass(target):
        if isinstance(value, target):
            return value

        if not isinstance(value, Mapping):
            raise TypeError(f"Cannot convert {type(value)} to dataclass {target.__name__}")

        hints = get_type_hints(target)
        kwargs = {}
        for field in _dataclasses.fields(target):
            if not field.init or field.name.startswith("_"):
                continue

            if field.name in value:
                field_value = convert(
                    value[field.name],
                    hints.get(field.name, Any),
                    options=options,
                    default_value=default_value,
                )
            elif field.default is not _dataclasses.MISSING:
                field_value = field.default
            elif field.default_factory is not _dataclasses.MISSING:  # type: ignore[attr-defined]
                field_value = field.default_factory()  # type: ignore[misc]
            else:
                field_value = default_from_hint(field.type)

            kwargs[field.name] = field_value

        return target(**kwargs)

    if origin in {list, set}:
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise TypeError(f"Cannot convert {type(value)} to {origin.__name__}")

        element_hint = args[0] if args else Any
        converted = [
            convert(item, element_hint, options=options, default_value=default_value)
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
                convert(item, element_hint, options=options, default_value=default_value)
                for item in values
            )

        if args and len(args) != len(values):
            raise TypeError("Tuple length does not match target annotation")

        return tuple(
            convert(item, args[idx] if args else Any, options=options, default_value=default_value)
            for idx, item in enumerate(values)
        )

    if origin in {dict, Mapping}:
        if not isinstance(value, Mapping):
            raise TypeError("Cannot convert non-mapping to dict")

        key_hint, value_hint = (args + (Any, Any))[:2]
        mapping_ctor = dict if origin is Mapping else origin
        return mapping_ctor(
            (
                convert(key, key_hint, options=options, default_value=default_value),
                convert(val, value_hint, options=options, default_value=default_value),
            )
            for key, val in value.items()
        )

    if target is Any or (isinstance(value, target) and target_arrow_hint is None):
        return value

    converter = _find_converter(value, target)
    if converter is None:
        raise TypeError(f"No converter registered for {type(value)} -> {target}")

    return converter(value, options)


@register_converter(str, int)
def _str_to_int(value: str, cast_options: Any) -> int:
    default_value = getattr(cast_options, "default_value", None)
    if value == "" and default_value is not None:
        return default_value
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

