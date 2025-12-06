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
    get_type_hints, Optional, List,
)

import pyarrow as pa

if TYPE_CHECKING:
    from .arrow_cast import CastOptions

__all__ = ["register_converter", "convert"]


Converter = Callable[[Any, "ArrowCastOptions | dict | None"], Any]


_registry: Dict[Tuple[Any, Any], Converter] = {}


def identity(x, opt):
    return x

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


def _type_matches(actual: Any, registered: Any) -> bool:
    """Return True if `actual` can use a converter registered for `registered`."""
    if actual is registered:
        return True
    if isinstance(registered, type) and isinstance(actual, type):
        try:
            return issubclass(actual, registered)
        except TypeError:
            return False
    return False


def _find_converter(from_type: Any, to_hint: Any) -> Optional[Converter]:
    # 0) Fast path: exact key
    conv = _registry.get((from_type, to_hint))
    if conv is not None:
        return conv

    if from_type == to_hint:
        return identity

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
            from_ok = _type_matches(from_type, registered_from)
            to_ok = _type_matches(to_hint, registered_to)
            if from_ok and to_ok:
                return conv
        except TypeError:
            # Some registered types (e.g. typing constructs) may explode in issubclass
            continue

    # 3) One-level composition: from_type -> mid_type -> to_hint
    #
    # If we can find converters:
    #   conv1: from_type -> mid
    #   conv2: mid      -> to_hint
    # we build a composite converter that calls conv2(conv1(x)).
    #
    # To keep things predictable, we:
    #   - Only use registered keys (no recursive _find_converter calls here).
    #   - Respect the same _type_matches logic for from/mid/to.
    for (from1, mid_type), conv1 in _registry.items():
        try:
            if not _type_matches(from_type, from1):
                continue
        except TypeError:
            continue

        for (from2, to2), conv2 in _registry.items():
            try:
                # mid_type must be compatible with from2
                if not _type_matches(mid_type, from2):
                    continue
                # requested to_hint must be compatible with to2
                if not _type_matches(to_hint, to2):
                    continue
            except TypeError:
                continue

            # Build composite converter once we find the first viable chain.
            def composed(value, options=None, _c1=conv1, _c2=conv2):
                intermediate = _c1(value, options)
                return _c2(intermediate, options)

            return composed

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
    options: Optional[Union[CastOptions, pa.Field, pa.DataType, pa.Schema]] = None,
    **kwargs,
) -> Any:
    """Convert ``value`` to ``target_hint`` using the registered converters."""
    from yggdrasil.types.python_defaults import default_scalar
    from yggdrasil.types.cast.cast_options import CastOptions

    is_optional, target_hint = _unwrap_optional(target_hint)

    if options is None and not kwargs:
        try:
            if isinstance(value, target_hint):
                return value
        except:
            pass

        if value is None:
            return None if is_optional else default_scalar(target_hint)

    options = CastOptions.check_arg(target_field=options, kwargs=kwargs)
    origin = get_origin(target_hint) or target_hint
    args = get_args(target_hint)
    source_hint = type(value)

    if isinstance(target_hint, (pa.Field, pa.DataType, pa.Schema)):
        options.set_target_arrow_field(target_hint, cast=True)

        if isinstance(value, pa.Array):
            target_hint = pa.Array
        else:
            target_hint = value.__class__
        converter = _find_converter(source_hint, target_hint)
    else:
        converter = _find_converter(source_hint, target_hint)

    if converter is not None:
        return converter(value, options)

    if isinstance(target_hint, type) and issubclass(target_hint, enum.Enum):
        if isinstance(value, target_hint):
            return value

        if isinstance(value, str):
            vcsfld = value.casefold()

            for member in target_hint:
                if member.name.casefold() == vcsfld:
                    return member
                if str(member.value).casefold() == vcsfld:
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
                field_value = default_scalar(field.type)

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
    return _str_to_datetime(value, cast_options).date()


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


@register_converter(str, _datetime.timedelta)
def _str_to_timedelta(value: str, cast_options: Any) -> _datetime.timedelta:
    default_value = getattr(cast_options, "default_value", None)
    stripped = value.strip()

    if stripped == "" and default_value is not None:
        return default_value

    # 1) Match "Dd HH:MM:SS[.ffffff]" or "HH:MM:SS[.ffffff]"
    m = re.fullmatch(
        r"(?:(\d+)d\s+)?"
        r"(\d{1,2}):(\d{1,2})"
        r"(?::(\d{1,2})(?:\.(\d{1,6}))?)?",
        stripped,
    )
    if m:
        days = int(m.group(1)) if m.group(1) else 0
        hours = int(m.group(2))
        minutes = int(m.group(3))
        seconds = int(m.group(4)) if m.group(4) else 0
        frac = m.group(5) or "0"
        microseconds = int(frac.ljust(6, "0"))
        return _datetime.timedelta(
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
        )

    # 2) Match "<number><unit>" where unit in {s, m, h, d}
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)([smhd])", stripped)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit == "s":
            seconds = val
        elif unit == "m":
            seconds = val * 60
        elif unit == "h":
            seconds = val * 3600
        elif unit == "d":
            seconds = val * 86400
        else:  # shouldn't happen
            raise ValueError(f"Unknown timedelta unit {unit!r}")
        return _datetime.timedelta(seconds=seconds)

    # 3) Fallback: bare number = seconds
    try:
        seconds = float(stripped)
        return _datetime.timedelta(seconds=seconds)
    except ValueError:
        pass

    raise ValueError(f"Cannot parse timedelta from {value!r}")


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
