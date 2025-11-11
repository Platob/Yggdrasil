"""Utilities for casting Arrow arrays between compatible schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import types as pa_types


@dataclass(frozen=True)
class ArrowArrayCaster:
    """Cast :class:`pyarrow.Array` instances between two declared fields."""

    source_field: pa.Field
    target_field: pa.Field

    def cast(
        self,
        array: pa.Array,
        target_type: pa.DataType | None = None,
        safe: bool | None = None,
        options: pc.CastOptions | None = None,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.Array:
        """Cast *array* to the target field's type.

        Parameters
        ----------
        memory_pool
        options
        safe
        target_type
        array:
            The Arrow array whose type must match :pyattr:`source_field`.

        Returns
        -------
        :class:`pyarrow.Array`
            The cast array with the target type applied.
        """
        if not array.type.equals(self.source_field.type):
            raise ValueError(
                "Scalar type does not match caster source field: "
                f"{array.type} != {self.source_field.type}"
            )

        resolved_target = target_type or self.target_field.type

        kwargs: Dict[str, object] = {}

        if options is not None:
            kwargs["options"] = options
        else:
            kwargs["target_type"] = resolved_target
            if safe is not None:
                kwargs["safe"] = safe

        if memory_pool is not None:
            kwargs["memory_pool"] = memory_pool

        return array.cast(**kwargs)


class ArrowCastRegistry:
    """Maintain a registry of :class:`ArrowArrayCaster` objects.

    The registry can lazily derive casters for compatible nested schemas to
    minimise upfront configuration.
    """

    def __init__(self) -> None:
        self._casters: Dict[Tuple[str, str], ArrowArrayCaster] = {}

    def register(self, caster: ArrowArrayCaster) -> None:
        """Register an :class:`ArrowArrayCaster` for reuse."""

        self._casters[self._key(caster.source_field, caster.target_field)] = caster

    def get_or_build(
        self,
        source_field: pa.Field,
        target_field: pa.Field,
        safe: bool = True
    ) -> ArrowArrayCaster:
        """Return a caster, creating one for compatible nested schemas when needed."""

        key = self._key(source_field, target_field)
        cached = self._casters.get(key)
        if cached is not None:
            return cached

        if not self._compatible_nested(source_field, target_field, safe=safe):
            raise KeyError(
                "No caster registered and unable to derive one for the provided fields"
            )

        caster = ArrowArrayCaster(source_field=source_field, target_field=target_field)
        self.register(caster)
        return caster

    def registered_pairs(self) -> Iterable[Tuple[pa.Field, pa.Field]]:
        """Iterate over the currently registered field pairs."""

        for caster in self._casters.values():
            yield caster.source_field, caster.target_field

    def _safe_type(self, obj: pa.DataType | pa.Field):
        if isinstance(obj, pa.DataType):
            return obj
        return obj.type

    def _key(self, source_field: pa.Field, target_field: pa.Field) -> Tuple[str, str]:
        return (
            str(self._safe_type(source_field)),
            str(self._safe_type(target_field))
        )

    def _compatible_nested(
        self,
        source_field: pa.Field,
        target_field: pa.Field,
        safe: bool
    ) -> bool:
        """Return ``True`` when the provided fields can be safely cast."""
        source_type = self._safe_type(source_field)
        target_type = self._safe_type(target_field)

        if source_type.equals(target_type):
            return True

        return self._compatible_types(source_type, target_type, safe=safe)

    def _compatible_types(
        self,
        source_type: pa.DataType,
        target_type: pa.DataType,
        safe: bool
    ) -> bool:
        if source_type.equals(target_type):
            return True

        if pa.types.is_string(source_type) and not safe:
            return True

        if pa_types.is_integer(source_type) and pa_types.is_integer(target_type):
            return True
        if pa_types.is_floating(source_type) and pa_types.is_floating(target_type):
            return True
        if pa_types.is_list(source_type) and pa_types.is_list(target_type):
            return self._compatible_types(
                source_type.value_type, target_type.value_type,
                safe=safe
            )
        if pa_types.is_large_list(source_type) and pa_types.is_large_list(target_type):
            return self._compatible_types(
                source_type.value_type, target_type.value_type,
                safe=safe
            )
        if pa_types.is_fixed_size_list(source_type) and pa_types.is_fixed_size_list(target_type):
            return self._compatible_types(
                source_type.value_type, target_type.value_type,
                safe=safe
            )

        if pa_types.is_struct(source_type) and pa_types.is_struct(target_type):
            if source_type.num_fields != target_type.num_fields:
                return False
            source_children = {field.name: field for field in source_type}
            target_children = {field.name: field for field in target_type}

            if source_children.keys() != target_children.keys():
                return False

            return all(
                self._compatible_types(
                    source_children[name].type,
                    target_children[name].type,
                    safe=safe
                )
                for name in source_children
            )

        return False


__all__ = ["ArrowArrayCaster", "ArrowCastRegistry"]
