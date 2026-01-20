"""Dataclass helpers that integrate with Arrow schemas and safe casting."""

import dataclasses
from inspect import isclass
from typing import Any

import pyarrow as pa

__all__ = [
    "get_dataclass_arrow_field"
]

DATACLASS_ARROW_FIELD_CACHE: dict[type, pa.Field] = {}


def get_dataclass_arrow_field(cls_or_instance: Any) -> pa.Field:
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

        from yggdrasil.types.python_arrow import arrow_field_from_hint

        built = arrow_field_from_hint(cls)
        DATACLASS_ARROW_FIELD_CACHE[cls] = built
        return built

    raise ValueError(f"{cls_or_instance!r} is not a dataclass or yggdrasil dataclass")
