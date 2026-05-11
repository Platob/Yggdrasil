# yggdrasil/data/cast/registry.py
"""
Type conversion registry and default converters.

This module is the *engine room* for yggdrasil casting.

Design goals
-----------
Small, predictable conversion engine:

- Fast lookup for exact (from_hint, to_hint) registrations.
- MRO-aware fallback (so subclasses work without extra registrations).
- Optional[T] / T | None handling (None passes through when optional, otherwise defaults).
- Container support (list / set / tuple / dict / Mapping) via recursive element casting.
- Enum + dataclass helpers (ergonomic for schemas / config objects).
- Namespace-triggered lazy imports to register ecosystem-specific converters
  (polars / pandas / pyspark) only when needed.

Key ideas
---------
- Converters are registered for *type hints*, not just raw Python types.
- We keep two registries:
  - `_registry[(from_hint, to_hint)]` for concrete registrations
  - `_any_registry[to_hint]` for wildcard "Any -> to_hint" handlers
- Dispatch prefers:
  1) exact match
  2) cheap identity
  3) Any-wildcard target handler
  4) MRO cross-product lookup
  5) scan-based fallback (issubclass checks for "odd" keys)
  6) one-hop composition: from -> mid -> to (single intermediate only)

The public API is `register_converter()` + `convert()`.

Notes on `options`
------------------
`options` is intentionally optional and can be:
- None
- CastOptions
- pyarrow Field / DataType / Schema

`CastOptions.check_arg(...)` normalizes it into a CastOptions instance, and kwargs
can override fields.

This module does *not* define CastOptions to avoid import cycles.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import types
from collections.abc import Iterable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import pyarrow as pa


if TYPE_CHECKING:
    from .options import CastOptions

__all__ = ["register_converter", "convert", "identity"]

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")

# Runtime converter shape used by the registry.
# Converter signature: func(value, options) -> converted_value
Converter = Callable[[Any, Optional["CastOptions"]], Any]
RegistryKey = tuple[Any, Any]


def identity(x: Any, *args, **kwargs) -> Any: # type: ignore
    """Return value as-is."""
    return x


# Concrete registrations: (from_hint, to_hint) -> converter
_registry: dict[RegistryKey, Converter] = {}

# Wildcard registrations: Any/object -> to_hint -> converter
_any_registry: dict[Any, Converter] = {}


def register_converter(from_hint: Any, to_hint: Any) -> Callable[[F], F]:
    """
    Decorator to register a converter from `from_hint` -> `to_hint`.

    This preserves the original function object (and its type signature),
    while registering it as a runtime `Converter`.

    Wildcard registrations
    ----------------------
    If `from_hint` is `typing.Any` or `object`, the converter is stored in
    `_any_registry[to_hint]` and is eligible for *any* source value type.

    Expected converter behavior:
      func(value, options) -> converted_value
    where `options` may be None.
    """

    def decorator(func: F) -> F:
        conv = func  # treated as Converter at runtime

        if from_hint in (Any, object):
            _any_registry[to_hint] = conv  # type: ignore[assignment]
        else:
            _registry[(from_hint, to_hint)] = conv  # type: ignore[assignment]

        return func

    return decorator


# ----------------------------
# Hint / type utilities
# ----------------------------


def unwrap_optional(hint: Any) -> tuple[bool, Any]:
    """
    Return (is_optional, base_hint) for Optional[T] / T | None.

    Examples:
      int | None -> (True, int)
      int | None -> (True, int)
      int -> (False, int)
    """
    origin = get_origin(hint)
    if origin in {Union, types.UnionType}:
        args = get_args(hint)
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return True, non_none[0]
    return False, hint


def iter_mro(tp: Any) -> Iterable[Any]:
    """
    Yield (tp, ...) including MRO if tp is class-like; else yield (tp,).

    This keeps lookups deterministic and cheap.
    """
    try:
        mro = getattr(tp, "__mro__", None)
    except TypeError:
        mro = None
    return (tp,) if mro is None else mro


def type_matches(actual: Any, registered: Any) -> bool:
    """
    True if `actual` can use converter registered for `registered`.

    This is slightly more permissive than plain `==` because it supports
    issubclass checks for class-like keys.
    """
    if actual is registered:
        return True
    if isinstance(registered, type) and isinstance(actual, type):
        try:
            return issubclass(actual, registered)
        except TypeError:
            return False
    return False


def find_converter(from_type: Any, to_hint: Any, check_namespace: bool = True) -> Optional[Converter]:
    """
    Find the best converter for (from_type -> to_hint).

    Dispatch order:
      1) exact (_registry[(from_type, to_hint)])
      2) identity-ish (same type, or target Any/object)
      3) wildcard Any->to_hint
      4) namespace-triggered late imports (polars/pandas/pyspark) once
      5) MRO cross-product lookup
      6) scan-based fallback with issubclass checks for odd keys
      7) one-hop composition: from -> mid -> to (single intermediate)
    """

    # 1) exact
    conv = _registry.get((from_type, to_hint))
    if conv is not None:
        return conv

    # 2) cheap identities
    if from_type == to_hint or to_hint in (object, Any):
        return identity

    # 3) wildcard Any -> to_hint
    any_converter = _any_registry.get(to_hint)
    if any_converter is not None:
        return any_converter

    # 4) late import side-effect: ensure namespace-specific converters are registered
    if check_namespace:
        from yggdrasil.pickle.serde import ObjectSerde

        from_namespace = ObjectSerde.full_namespace(from_type)
        to_namespace = ObjectSerde.full_namespace(to_hint)

        # IMPORTANT: these imports are *side effects* that register more converters.
        if from_namespace.startswith("polars") or to_namespace.startswith("polars"):
            from yggdrasil.polars import cast as _polars_cast  # noqa: F401
        elif from_namespace.startswith("pandas") or to_namespace.startswith("pandas"):
            from yggdrasil.pandas import cast as _pandas_cast  # noqa: F401
        elif from_namespace.startswith("pyspark") or to_namespace.startswith("pyspark"):
            from yggdrasil.spark import cast as _spark_cast  # noqa: F401
        elif from_namespace.startswith("pyarrow") or to_namespace.startswith("pyarrow"):
            from yggdrasil.arrow import cast as _arrow_cast  # noqa: F401

        return find_converter(from_type, to_hint, check_namespace=False)

    # 5) MRO cross-product lookup (fast and deterministic)
    for f in iter_mro(from_type):
        for t in iter_mro(to_hint):
            conv = _registry.get((f, t))
            if conv is not None:
                return conv

    # 6) scan with issubclass for odd registered keys
    for (rf, rt), conv in _registry.items():
        try:
            if type_matches(from_type, rf) and type_matches(to_hint, rt):
                return conv
        except TypeError:
            continue

    # 7) one-level composition: from -> mid -> to
    # This is intentionally limited: deterministic and avoids path-search explosions.
    for (rf, mid), c1 in _registry.items():
        try:
            if not type_matches(from_type, rf):
                continue
        except TypeError:
            continue

        for (rmid, rt), c2 in _registry.items():
            try:
                if not type_matches(mid, rmid):
                    continue
                if not type_matches(to_hint, rt):
                    continue
            except TypeError:
                continue

            def composed(v: Any, o: Optional["CastOptions"], _c1=c1, _c2=c2) -> Any:
                return _c2(_c1(v, o), o)

            return composed

    return None


def is_runtime_value(x: Any) -> bool:
    """
    True for runtime values (42, [], MyClass()), False for type hints.

    Used by some downstream logic that wants to distinguish "value" vs "hint".
    """
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
    """
    Convert `value` to `target_hint` using registered converters + built-ins.

    Dispatch order (cheapest first):
      1) ``Optional[T]`` unwrap.
      2) ``None`` → ``None`` if optional, else ``default_scalar(target)``.
      3) ``Any`` / ``object`` target → identity passthrough.
      4) ``isinstance(value, target_hint)`` → identity passthrough.
      5) Registry lookup (exact / wildcard / namespace / MRO / one-hop composition).
      6) ``Enum`` member resolution and ``dataclass`` from-mapping coercion.
      7) Container generics — ``list`` / ``set`` / ``tuple`` / ``dict`` / ``Mapping``.
      8) ``TypeError`` — no path found.

    Options are normalized through ``CastOptions.check`` only when the caller
    actually supplied one — the no-options call site (the common one) skips the
    allocation entirely.
    """
    from yggdrasil.arrow.python_defaults import default_scalar
    from yggdrasil.data.options import CastOptions

    is_optional, target_hint = unwrap_optional(target_hint)

    if value is None:
        return None if is_optional else default_scalar(target_hint)  # type: ignore[return-value]

    if target_hint is Any or target_hint is object:
        return value  # type: ignore[return-value]

    target_is_type = isinstance(target_hint, type)
    if target_is_type:
        try:
            if isinstance(value, target_hint):
                return value  # type: ignore[return-value]
        except TypeError:
            # Generic aliases / parameterized types raise here; ignore and dispatch.
            pass

    if options is not None or kwargs:
        options = CastOptions.check(options, **kwargs)

    conv = find_converter(type(value), target_hint)
    if conv is not None:
        return conv(value, options)  # type: ignore[return-value]

    if target_is_type:
        if issubclass(target_hint, enum.Enum):
            return convert_to_python_enum(value, target_hint, options=options)  # type: ignore[return-value]
        if dataclasses.is_dataclass(target_hint):
            return convert_to_python_dataclass(value, target_hint, options=options)  # type: ignore[return-value]

    origin = get_origin(target_hint) or target_hint
    args = get_args(target_hint)

    if origin is list or origin is set:
        return convert_to_python_iterable(value, origin, args, options=options)  # type: ignore[return-value]
    if origin is tuple:
        return convert_tuple(value, args, options)  # type: ignore[return-value]
    if origin is dict or origin is Mapping:
        return convert_mapping(value, origin, args, options)  # type: ignore[return-value]

    raise TypeError(f"No converter registered for {type(value)} -> {target_hint}")


def _is_noop_hint(hint: Any) -> bool:
    """True for hints where per-element conversion is a pure no-op."""
    return hint is Any or hint is object


def convert_tuple(value: Any, args: tuple[Any, ...], options: "CastOptions") -> tuple[Any, ...]:
    """
    Convert an iterable into a tuple type.

    Supports:
      - tuple[T, ...]
      - tuple[T1, T2, ...] (fixed-length)

    Fast path: a tuple with an ``Any`` / ``object`` element hint skips
    per-element recursion.
    """
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise TypeError("Cannot convert non-iterable to tuple")

    values = tuple(value)

    # tuple[T, ...]
    if len(args) == 2 and args[1] is Ellipsis:
        elem_hint = args[0]
        if _is_noop_hint(elem_hint):
            return values
        return tuple(convert(v, elem_hint, options=options) for v in values)

    # tuple[T1, T2, ...]
    if args and len(args) != len(values):
        raise TypeError("Tuple length does not match target annotation")

    if not args:
        return values

    if all(_is_noop_hint(a) for a in args):
        return values

    return tuple(
        convert(v, target_hint=args[i], options=options)
        for i, v in enumerate(values)
    )


def convert_mapping(
    value: Any,
    origin: Any,
    args: tuple[Any, ...],
    options: "CastOptions",
) -> Mapping[Any, Any]:
    """
    Convert a mapping into dict/Mapping[K, V] with recursive casting.

    Fast path: if both key and value hints are ``Any``/``object`` the
    mapping is constructed directly without per-element recursion.
    """
    if not isinstance(value, Mapping):
        raise TypeError("Cannot convert non-mapping to dict")

    key_hint, val_hint = (args + (Any, Any))[:2]
    ctor = dict if origin is Mapping else origin

    if _is_noop_hint(key_hint) and _is_noop_hint(val_hint):
        return ctor(value)

    if _is_noop_hint(key_hint):
        return ctor(
            (k, convert(v, val_hint, options=options))
            for k, v in value.items()
        )
    if _is_noop_hint(val_hint):
        return ctor(
            (convert(k, key_hint, options=options), v)
            for k, v in value.items()
        )
    return ctor(
        (convert(k, key_hint, options=options), convert(v, val_hint, options=options))
        for k, v in value.items()
    )


# ----------------------------
# Built-in converters
# ----------------------------

# Per-enum cache of (name_lookup, value_lookup, first_value_type).
# Built once per Enum class and reused on every subsequent conversion so
# `convert_to_python_enum` becomes two O(1) dict lookups instead of an
# O(n) linear scan over members.
_enum_lookup_cache: dict[type, tuple[dict[str, Any], dict[Any, Any], type]] = {}


def _enum_lookups(target: type[enum.Enum]) -> tuple[dict[str, Any], dict[Any, Any], type]:
    cached = _enum_lookup_cache.get(target)
    if cached is not None:
        return cached

    try:
        first = next(iter(target))
    except StopIteration as e:
        raise TypeError(f"Cannot convert to empty Enum {target.__name__}") from e

    name_lookup: dict[str, Any] = {}
    value_lookup: dict[Any, Any] = {}
    for m in target:
        name_lookup.setdefault(m.name.casefold(), m)
        name_lookup.setdefault(str(m.value).casefold(), m)
        try:
            value_lookup.setdefault(m.value, m)
        except TypeError:
            # Unhashable member values fall back to the name-only path.
            pass

    cached = (name_lookup, value_lookup, type(first.value))
    _enum_lookup_cache[target] = cached
    return cached


def convert_to_python_enum(value: Any, target: type[enum.Enum], options: Optional["CastOptions"] = None) -> enum.Enum:
    """
    Convert to an Enum member.

    Strategies (all O(1) lookups via a cached per-class map):
      1) Already an instance -> return as-is.
      2) ``str`` -> case-insensitive lookup against member names and stringified values.
      3) Otherwise -> coerce to the underlying value type of the first member, then equality-match.
    """
    if isinstance(value, target):
        return value

    name_lookup, value_lookup, first_value_type = _enum_lookups(target)

    if isinstance(value, str):
        hit = name_lookup.get(value.casefold())
        if hit is not None:
            return hit

    try:
        coerced = convert(value, first_value_type, options=options)
    except Exception:
        coerced = value

    try:
        hit = value_lookup.get(coerced)
    except TypeError:
        hit = None
    if hit is not None:
        return hit

    raise TypeError(f"No matching Enum member for {value!r} in {target.__name__}")


def convert_to_python_dataclass(value: Any, target: type[T], options: Optional["CastOptions"] = None) -> T:
    """
    Convert a mapping-like object into a dataclass instance.

    Rules:
    - If `value` is already an instance of `target`, return it.
    - Input must be a Mapping; keys are dataclass field names.
    - Fields with init=False or name starting with "_" are ignored.
    - For each init field:
        * if present in input -> cast using resolved type hints (get_type_hints)
        * else if dataclass default/default_factory exists -> use it
        * else -> default_scalar(resolved_field_type)

    Why resolved_field_type matters:
    - With `from __future__ import annotations`, `dataclasses.Field.type` can be a *string*
      (e.g. "int"), which would break default_scalar(...) because it expects real hints/types.
      So for defaults we must use `get_type_hints(target)` (already computed as `hints`).
    """
    from yggdrasil.arrow.python_defaults import default_scalar

    if isinstance(value, target):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"Cannot convert {type(value)} to dataclass {target.__name__}")

    # Resolved annotations (handles __future__.annotations and forward refs)
    hints = get_type_hints(target)
    out: dict[str, Any] = {}

    for f in dataclasses.fields(target):
        if not f.init or f.name.startswith("_"):
            continue

        field_hint = hints.get(f.name, f.type)

        if f.name in value:
            out[f.name] = convert(value[f.name], field_hint, options=options)
            continue

        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            out[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            out[f.name] = default_scalar(field_hint)

    return target(**out)


def convert_to_python_iterable(
    value: Any,
    origin: type,
    args: tuple[Any, ...],
    options: Optional["CastOptions"] = None,
) -> Any:
    """
    Convert `value` into list/set with recursive casting.

    Fast path: when the element hint is ``Any`` / ``object`` we skip the
    per-element ``convert()`` call and materialize the container directly
    from the iterator, which avoids a pure-Python loop with a function
    call per element.
    """
    if isinstance(value, (str, bytes)):
        raise TypeError(f"No converter registered for {type(value)} -> {origin}")

    elem_hint = args[0] if args else Any

    if _is_noop_hint(elem_hint):
        return origin(value)

    return origin(convert(v, elem_hint, options=options) for v in value)


# ----------------------------
# Default registrations — base Python types delegate to DataType._convert_pyobj
# ----------------------------
#
# Why delegate?
# -------------
# Each scalar Python type already has a canonical coercion path on its
# matching DataType subclass (IntegerType/FloatingPointType/BooleanType/
# StringType/...). Routing the registry through ``DataType._convert_pyobj``
# keeps a single source of truth for parse semantics — feature code that
# does ``convert("123", int)`` and engine code that hits ``IntegerType``
# during schema-driven ingest end up on the same code path.

# Cached primitive-DataType singletons keyed by Python type. ``DataType``
# instances are frozen, hashable, and idempotent — building one per
# convert() call would dwarf the actual coercion work.
_PRIMITIVE_DTYPE_CACHE: dict[type, Any] = {}


def _primitive_dtype(target: type) -> Any:
    """Return the cached :class:`DataType` matching *target* (e.g. ``int`` → ``IntegerType``)."""
    cached = _PRIMITIVE_DTYPE_CACHE.get(target)
    if cached is not None:
        return cached

    from yggdrasil.data.types.base import DataType

    dtype = DataType.from_pytype(target)
    _PRIMITIVE_DTYPE_CACHE[target] = dtype
    return dtype


@register_converter(str, int)
def str_to_int(value: str, opts: Any) -> int:
    """Parse int from string via ``IntegerType``. Empty string → 0."""
    if value == "":
        return 0
    return _primitive_dtype(int)._convert_pyobj(value, safe=True)


@register_converter(str, float)
def str_to_float(value: str, opts: Any) -> float:
    """Parse float from string via ``FloatingPointType``.

    Honors ``opts.default_value`` when input is the empty string — convenient
    for CSV/Excel ingest where missing cells round-trip as ``""``.
    """
    if value == "":
        default_value = getattr(opts, "default_value", None)
        if default_value is not None:
            return default_value
    return _primitive_dtype(float)._convert_pyobj(value, safe=True)


@register_converter(str, bool)
def str_to_bool(value: str, opts: Any) -> bool:
    """Parse bool from string via ``BooleanType``.

    Empty string is rejected (use ``opts.default_value`` to opt in to a
    fallback). The accepted truthy/falsy tokens are owned by ``BooleanType``.
    """
    if value == "":
        default_value = getattr(opts, "default_value", None)
        if default_value is not None:
            return default_value
        raise ValueError(f"Cannot parse boolean from {value!r}")
    return _primitive_dtype(bool)._convert_pyobj(value, safe=True)


@register_converter(int, str)
def int_to_str(value: int, _: Any) -> str:
    """Stringify int via ``StringType``."""
    return _primitive_dtype(str)._convert_pyobj(value, safe=True)