"""Type conversion registry and default converters.

Small, predictable conversion engine:
- Fast lookup for exact (from, to)
- MRO-aware fallback (subclasses)
- Optional[Any] unwrapping + sensible defaults
- Container support (list/set/tuple/dict/Mapping)
- Enum + dataclass helpers
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import inspect
import re
import types
from collections.abc import Iterable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import pyarrow as pa

if TYPE_CHECKING:
    from .cast_options import CastOptions

__all__ = ["register_converter", "convert"]

T = TypeVar("T")
R = TypeVar("R")

Converter = Callable[[Any, "CastOptions"], Any]
RegistryKey = tuple[Any, Any]


def _identity(x: Any, _: "CastOptions") -> Any:
    return x


_registry: dict[RegistryKey, Converter] = {}
_any_registry: dict[Any, Converter] = {}


def register_converter(from_hint: Any, to_hint: Any) -> Callable[[Converter], Converter]:
    """Decorator: register a converter from `from_hint` -> `to_hint`.

    Converter signature: (value, options) -> converted_value
    """

    def decorator(func: Converter) -> Converter:
        if from_hint in (Any, object):
            _any_registry[to_hint] = func
        else:
            _registry[(from_hint, to_hint)] = func
        return func

    return decorator


# ----------------------------
# Hint / type utilities
# ----------------------------


def _unwrap_optional(hint: Any) -> tuple[bool, Any]:
    """Return (is_optional, base_hint) for Optional[T] / T | None."""
    origin = get_origin(hint)
    if origin in {Union, types.UnionType}:
        args = get_args(hint)
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return True, non_none[0]
    return False, hint


def _iter_mro(tp: Any) -> Iterable[Any]:
    """(tp, ...) including MRO if tp is class-like; else (tp,)."""
    try:
        mro = getattr(tp, "__mro__", None)
    except TypeError:
        mro = None
    return (tp,) if mro is None else mro


def _type_matches(actual: Any, registered: Any) -> bool:
    """True if `actual` can use converter registered for `registered`."""
    if actual is registered:
        return True
    if isinstance(registered, type) and isinstance(actual, type):
        try:
            return issubclass(actual, registered)
        except TypeError:
            return False
    return False


def find_converter(from_type: Any, to_hint: Any) -> Optional[Converter]:
    """Find best converter for (from_type -> to_hint)."""

    # exact
    conv = _registry.get((from_type, to_hint))
    if conv is not None:
        return conv

    # cheap identities
    if from_type == to_hint or to_hint in (object, Any):
        return _identity

    # mro cross-product lookup (fast and deterministic)
    for f in _iter_mro(from_type):
        for t in _iter_mro(to_hint):
            conv = _registry.get((f, t))
            if conv is not None:
                return conv

    # scan with issubclass for odd registered keys
    for (rf, rt), conv in _registry.items():
        try:
            if _type_matches(from_type, rf) and _type_matches(to_hint, rt):
                return conv
        except TypeError:
            continue

    # one-level composition: from -> mid -> to
    for (rf, mid), c1 in _registry.items():
        try:
            if not _type_matches(from_type, rf):
                continue
        except TypeError:
            continue

        for (rmid, rt), c2 in _registry.items():
            try:
                if not _type_matches(mid, rmid):
                    continue
                if not _type_matches(to_hint, rt):
                    continue
            except TypeError:
                continue

            def composed(v: Any, o: "CastOptions", _c1=c1, _c2=c2) -> Any:
                return _c2(_c1(v, o), o)

            return composed

    return _any_registry.get(to_hint)


def _normalize_fractional_seconds(value: str) -> str:
    """Normalize fractional seconds to microsecond precision for fromisoformat()."""
    match = re.search(r"(\.)(\d+)(?=(?:[+-]\d{2}:?\d{2})?$)", value)
    if not match:
        return value
    start, end = match.span(2)
    frac = match.group(2)
    frac = frac[:6].ljust(6, "0")
    return value[:start] + frac + value[end:]


def is_runtime_value(x: Any) -> bool:
    """True for runtime values (42, [], MyClass()), False for type hints."""
    if inspect.isclass(x):
        return False
    return get_origin(x) is None


# ----------------------------
# Public API
# ----------------------------


def convert(
    value: Any,
    target_hint: type[T],
    options: Optional[Union["CastOptions", pa.Field, pa.DataType, pa.Schema]] = None,
    **kwargs: Any,
) -> T:
    """Convert `value` to `target_hint` using registered converters + built-ins."""
    from yggdrasil.types.python_defaults import default_scalar
    from yggdrasil.types.cast.cast_options import CastOptions

    is_optional, target_hint = _unwrap_optional(target_hint)

    # ultra-fast path: no options, no kwargs
    if options is None and not kwargs:
        try:
            if isinstance(value, target_hint):
                return value
        except Exception:
            pass

        if value is None:
            return None if is_optional else default_scalar(target_hint)  # type: ignore[return-value]

    options = CastOptions.check_arg(options=options, **kwargs)

    if value is None:
        return None if is_optional else default_scalar(target_hint)  # type: ignore[return-value]

    source_type = type(value)
    conv = find_converter(source_type, target_hint)
    if conv is not None:
        return conv(value, options)  # type: ignore[return-value]

    origin = get_origin(target_hint) or target_hint
    args = get_args(target_hint)

    if isinstance(target_hint, type) and issubclass(target_hint, enum.Enum):
        return convert_to_python_enum(value, target_hint, options=options)  # type: ignore[return-value]

    if isinstance(target_hint, type) and dataclasses.is_dataclass(target_hint):
        return convert_to_python_dataclass(value, target_hint, options=options)  # type: ignore[return-value]

    if origin in {list, set}:
        return convert_to_python_iterable(value, origin, args, options=options)  # type: ignore[return-value]

    if origin is tuple:
        return _convert_tuple(value, args, options)  # type: ignore[return-value]

    if origin in {dict, Mapping}:
        return _convert_mapping(value, origin, args, options)  # type: ignore[return-value]

    # last-resort identity-ish
    try:
        if target_hint is Any or isinstance(value, target_hint):
            return value
    except Exception:
        pass

    raise TypeError(f"No converter registered for {type(value)} -> {target_hint}")


def _convert_tuple(value: Any, args: tuple[Any, ...], options: "CastOptions") -> tuple[Any, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise TypeError("Cannot convert non-iterable to tuple")

    values = tuple(value)

    # tuple[T, ...]
    if len(args) == 2 and args[1] is Ellipsis:
        elem_hint = args[0]
        return tuple(convert(v, elem_hint, options=options) for v in values)

    # tuple[T1, T2, ...]
    if args and len(args) != len(values):
        raise TypeError("Tuple length does not match target annotation")

    return tuple(
        convert(v, target_hint=args[i] if args else Any, options=options)
        for i, v in enumerate(values)
    )


def _convert_mapping(
    value: Any,
    origin: Any,
    args: tuple[Any, ...],
    options: "CastOptions",
) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Cannot convert non-mapping to dict")

    key_hint, val_hint = (args + (Any, Any))[:2]
    ctor = dict if origin is Mapping else origin
    return ctor(
        (convert(k, key_hint, options=options), convert(v, val_hint, options=options))
        for k, v in value.items()
    )


# ----------------------------
# Built-in converters
# ----------------------------


def convert_to_python_enum(value: Any, target: type[enum.Enum], options: Optional["CastOptions"] = None) -> enum.Enum:
    if isinstance(value, target):
        return value

    if isinstance(value, str):
        s = value.casefold()
        for m in target:
            if m.name.casefold() == s or str(m.value).casefold() == s:
                return m

    try:
        first = next(iter(target))
    except StopIteration as e:
        raise TypeError(f"Cannot convert to empty Enum {target.__name__}") from e

    # try to coerce into underlying value type
    try:
        coerced = convert(value, type(first.value), options=options)
    except Exception:
        coerced = value

    for m in target:
        if m.value == coerced:
            return m

    raise TypeError(f"No matching Enum member for {value!r} in {target.__name__}")


def convert_to_python_dataclass(value: Any, target: type[T], options: Optional["CastOptions"] = None) -> T:
    from yggdrasil.types.python_defaults import default_scalar

    if isinstance(value, target):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"Cannot convert {type(value)} to dataclass {target.__name__}")

    hints = get_type_hints(target)
    out: dict[str, Any] = {}

    for f in dataclasses.fields(target):
        if not f.init or f.name.startswith("_"):
            continue

        if f.name in value:
            out[f.name] = convert(value[f.name], hints.get(f.name, Any), options=options)
            continue

        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            out[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            out[f.name] = default_scalar(f.type)

    return target(**out)


def convert_to_python_iterable(
    value: Any,
    origin: type,
    args: tuple[Any, ...],
    options: Optional["CastOptions"] = None,
) -> Any:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"No converter registered for {type(value)} -> {origin}")

    elem_hint = args[0] if args else Any

    # Arrow -> pylist (optionally cast through Arrow type hint)
    if isinstance(value, (pa.Array, pa.ChunkedArray, pa.Table, pa.RecordBatch)):
        from .. import arrow_field_from_hint

        try:
            value = convert(value, arrow_field_from_hint(elem_hint), options=options)
        except TypeError:
            pass
        value = value.to_pylist()

    return origin(convert(v, elem_hint, options=options) for v in value)


# ----------------------------
# Default registrations
# ----------------------------


@register_converter(str, int)
def _str_to_int(value: str, opts: Any) -> int:
    return 0 if value == "" else int(value)


@register_converter(str, float)
def _str_to_float(value: str, opts: Any) -> float:
    default_value = getattr(opts, "default_value", None)
    if value == "" and default_value is not None:
        return default_value
    return float(value)


@register_converter(str, bool)
def _str_to_bool(value: str, opts: Any) -> bool:
    default_value = getattr(opts, "default_value", None)
    if value == "" and default_value is not None:
        return default_value

    s = value.strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise ValueError(f"Cannot parse boolean from {value!r}")


@register_converter(str, dt.date)
def _str_to_date(value: str, opts: Any) -> dt.date:
    return _str_to_datetime(value, opts).date()


@register_converter(str, dt.datetime)
def _str_to_datetime(value: str, opts: Any) -> dt.datetime:
    default_value = getattr(opts, "default_value", None)
    if value == "" and default_value is not None:
        return default_value

    s = value.strip()
    if s == "now":
        return dt.datetime.now(tz=dt.timezone.utc)

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    s = _normalize_fractional_seconds(s)

    try:
        parsed = dt.datetime.fromisoformat(s)
    except ValueError:
        formats = (
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
        )
        last: Optional[ValueError] = None
        for fmt in formats:
            try:
                parsed = dt.datetime.strptime(s, fmt)
                break
            except ValueError as e:  # pragma: no cover
                last = e
        else:
            raise last or ValueError(f"Cannot parse datetime from {value!r}")

    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


@register_converter(str, dt.timedelta)
def _str_to_timedelta(value: str, opts: Any) -> dt.timedelta:
    default_value = getattr(opts, "default_value", None)
    s = value.strip()
    if s == "" and default_value is not None:
        return default_value

    # "Dd HH:MM:SS[.ffffff]" or "HH:MM:SS[.ffffff]" or "HH:MM"
    m = re.fullmatch(
        r"(?:(\d+)d\s+)?"
        r"(\d{1,2}):(\d{1,2})"
        r"(?::(\d{1,2})(?:\.(\d{1,6}))?)?",
        s,
    )
    if m:
        days = int(m.group(1)) if m.group(1) else 0
        hours = int(m.group(2))
        minutes = int(m.group(3))
        seconds = int(m.group(4)) if m.group(4) else 0
        micro = int((m.group(5) or "0").ljust(6, "0"))
        return dt.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=micro)

    # "<number><unit>" unit in {s,m,h,d}
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)([smhd])", s)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return dt.timedelta(seconds=val * scale)

    # bare number = seconds
    try:
        return dt.timedelta(seconds=float(s))
    except ValueError as e:
        raise ValueError(f"Cannot parse timedelta from {value!r}") from e


@register_converter(str, dt.time)
def _str_to_time(value: str, opts: Any) -> dt.time:
    default_value = getattr(opts, "default_value", None)
    if value == "" and default_value is not None:
        return default_value
    return dt.time.fromisoformat(value)


@register_converter(dt.datetime, dt.date)
def _datetime_to_date(value: dt.datetime, _: Any) -> dt.date:
    return value.date()


@register_converter(int, str)
def _int_to_str(value: int, _: Any) -> str:
    return str(value)
