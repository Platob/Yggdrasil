"""Dataclass helpers that integrate with Arrow schemas and safe casting."""

import dataclasses
from inspect import isclass
from typing import Any, TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "dataclass_to_arrow_field",
    "get_from_dict"
]

DATACLASS_ARROW_FIELD_CACHE: dict[type, "pa.Field"] = {}


def dataclass_to_arrow_field(cls_or_instance: Any) -> "pa.Field":
    """Return a cached Arrow Field describing the dataclass type.

    Args:
        cls_or_instance: Dataclass class or instance.

    Returns:
        Arrow field describing the dataclass schema.
    """
    if dataclasses.is_dataclass(cls_or_instance):
        cls = cls_or_instance
        if not isclass(cls_or_instance):
            cls = cls_or_instance.__class__

        existing = DATACLASS_ARROW_FIELD_CACHE.get(cls, None)
        if existing is not None:
            return existing

        from yggdrasil.arrow.python_arrow import arrow_field_from_hint

        built = arrow_field_from_hint(cls)
        DATACLASS_ARROW_FIELD_CACHE[cls] = built
        return built

    raise ValueError(f"{cls_or_instance!r} is not a dataclass or yggdrasil dataclass")


def get_from_dict(
    obj: Mapping[str, Any],
    keys: Sequence[str],
    prefix: str,
) -> Any:
    """
    Best-effort field lookup with optional prefix support.

    Lookup order for each key:
      1) obj[key]
      2) obj[prefix + key]

    Returns:
      - first non-MISSING value found
      - MISSING if nothing matched
    """
    for key in keys:
        found = obj.get(key, dataclasses.MISSING)
        if found is not dataclasses.MISSING:
            return found

        found = obj.get(prefix + key, dataclasses.MISSING)
        if found is not dataclasses.MISSING:
            return found

    return dataclasses.MISSING