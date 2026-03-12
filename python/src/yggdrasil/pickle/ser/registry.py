from __future__ import annotations

from typing import Any

__all__ = ["SerdeRegistry", "REGISTRY"]


class SerdeRegistry:
    def __init__(self) -> None:
        self._tag_to_type: dict[int, type] = {}
        self._py_to_type: list[tuple[type, type]] = []

    def register_tag(self, tag: int, cls: type) -> None:
        prev = self._tag_to_type.get(tag)
        if prev is not None and prev is not cls:
            # dataclass(slots=True) may recreate class objects with the same identity
            # in terms of module/qualname; treat that as idempotent registration.
            same_decl = (
                getattr(prev, "__module__", None) == getattr(cls, "__module__", None)
                and getattr(prev, "__qualname__", None) == getattr(cls, "__qualname__", None)
            )
            if not same_decl:
                raise ValueError(f"Duplicate serialized tag registration: {tag} -> {cls!r}")
        self._tag_to_type[tag] = cls

    def register_python_type(self, py_type: type, cls: type) -> None:
        self._py_to_type.append((py_type, cls))

    def get_by_tag(self, tag: int) -> type:
        try:
            return self._tag_to_type[tag]
        except KeyError as e:
            raise ValueError(f"Unsupported tag for deserialization: {tag}") from e

    def get_by_python_value(self, obj: Any) -> type:
        if obj is None:
            for py_type, cls in self._py_to_type:
                if py_type is type(None):
                    return cls
            raise TypeError("No serializer registered for None")

        obj_type = type(obj)

        # Prefer exact type registrations first to avoid base-class shadowing
        # (e.g., OrderedDict should not resolve to dict serializer).
        for py_type, cls in reversed(self._py_to_type):
            if obj_type is py_type:
                return cls

        for py_type, cls in reversed(self._py_to_type):
            if isinstance(obj, py_type):
                return cls

        raise TypeError(f"Unsupported type for serialization: {type(obj)}")


REGISTRY = SerdeRegistry()