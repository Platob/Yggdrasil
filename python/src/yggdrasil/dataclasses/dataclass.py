"""Dataclass helpers that integrate with Arrow schemas and safe casting."""
from __future__ import annotations

from dataclasses import MISSING, Field, fields, is_dataclass
from inspect import isclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Optional, Callable, TypeVar, get_type_hints

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "DATACLASS_ARROW_FIELD_CACHE",
    "dataclass_to_arrow_field",
    "get_from_dict",
    "serialize_dataclass_state",
    "restore_dataclass_state",
    "default_value",
    "lazy_property"
]

DATACLASS_ARROW_FIELD_CACHE: dict[type, "pa.Field"] = {}
S = TypeVar("S")
T = TypeVar("T")


def dataclass_to_arrow_field(cls_or_instance: Any) -> "pa.Field":
    if not isinstance(cls_or_instance, type):
        cls = cls_or_instance.__class__
    else:
        cls = cls_or_instance

    existing = DATACLASS_ARROW_FIELD_CACHE.get(cls)
    if existing is not None:
        return existing

    from yggdrasil.data.data_field import Field
    built = Field.from_dataclass(cls).to_arrow_field()
    DATACLASS_ARROW_FIELD_CACHE[cls] = built
    return built


def get_from_dict(
    obj: Mapping[str, Any],
    keys: Sequence[str],
    prefix: Optional[str],
) -> Any:
    """Best-effort field lookup with optional prefix support.

    Lookup order for each key:
      1) obj[key]
      2) obj[prefix + key]

    Returns:
      - first non-MISSING value found
      - MISSING if nothing matched
    """
    for key in keys:
        found = obj.get(key, MISSING)
        if found is not MISSING:
            return found

        if prefix:
            found = obj.get(prefix + key, MISSING)
            if found is not MISSING:
                return found

    return MISSING


def default_value(f: Field[Any], with_factory: bool = True) -> Any:
    """Return the effective default value for a dataclass field.

    Returns:
        - f.default when present
        - f.default_factory() when present
        - MISSING otherwise
    """
    if f.default is not MISSING:
        return f.default

    if with_factory and f.default_factory is not MISSING:  # type: ignore[attr-defined]
        return f.default_factory()  # type: ignore[misc]

    return MISSING


def serialize_dataclass_state(obj: Any) -> dict[str, Any]:
    """Serialize constructor state for a dataclass instance.

    Rules:
      - only init=True fields are considered
      - private fields (name starts with "_") are skipped
      - None values are skipped
      - values equal to their effective default are skipped
      - output is a raw payload dict with no version envelope
    """
    payload: dict[str, Any] = {}

    for f in fields(obj):
        if not f.init or f.name.startswith("_"):
            continue

        value = getattr(obj, f.name)

        if value is None:
            continue

        default = default_value(f, with_factory=True)
        if default is not MISSING and value == default:
            continue

        payload[f.name] = value

    return payload


def restore_dataclass_state(obj: Any, state: Any) -> None:
    """Restore dataclass state from a raw payload dict.

    Rules:
      - None is treated as {}
      - unknown keys are ignored
      - missing init=True fields are filled from effective defaults
      - missing required init=True fields raise TypeError
      - non-init fields are reset to their effective defaults when available

    Raises:
        TypeError: If state is not a dict or a required field is missing.
    """
    if state is None:
        payload: dict[str, Any] = {}
    elif isinstance(state, dict):
        payload = state
    else:
        raise TypeError(f"Invalid pickle state for {type(obj).__name__}: {type(state)!r}")

    known_fields = {f.name: f for f in fields(obj)}

    for name, f in known_fields.items():
        if not f.init:
            continue

        if name in payload:
            value = payload[name]
        else:
            value = default_value(f)
            if value is MISSING:
                raise TypeError(
                    f"Cannot restore {type(obj).__name__}: missing required field {name!r}"
                )

        object.__setattr__(obj, name, value)

    for name, f in known_fields.items():
        if f.init:
            continue

        value = default_value(f)
        if value is not MISSING:
            object.__setattr__(obj, name, value)


def lazy_property(
    self: S,
    *,
    cache_attr: str,
    factory: Callable[[S], T],
    use_cache: bool,
) -> T:
    if use_cache:
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached

        created = factory(self)
        object.__setattr__(self, cache_attr, created)
        return created

    return factory(self)
