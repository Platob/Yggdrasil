"""Utilities for casting Arrow scalars between compatible schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pyarrow as pa
import pyarrow.compute as pc

from .array_cast import ArrowArrayCaster, ArrowCastRegistry


@dataclass(frozen=True)
class ArrowScalarCaster:
    """Cast :class:`pyarrow.Scalar` instances between two declared fields."""

    source_field: pa.Field
    target_field: pa.Field
    array_caster: ArrowArrayCaster | None = None

    def cast(
        self,
        scalar: pa.Scalar,
        target_type: pa.DataType | None = None,
        safe: bool | None = None,
        options: pc.CastOptions | None = None,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.Scalar:
        """Cast *scalar* to the target field's type."""

        if not scalar.type.equals(self.source_field.type):
            raise ValueError(
                "Scalar type does not match caster source field: "
                f"{scalar.type} != {self.source_field.type}"
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

        try:
            return scalar.cast(**kwargs)
        except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
            if self.array_caster is None:
                raise

        # Fallback to casting through an array when scalar casting is unsupported.
        array = pa.array([scalar.as_py()], type=self.source_field.type)
        cast_array = self.array_caster.cast(
            array,
            target_type=None if options is not None else resolved_target,
            safe=safe if options is None else None,
            options=options,
            memory_pool=memory_pool,
        )
        return cast_array[0]


class ArrowScalarCastRegistry:
    """Maintain :class:`ArrowScalarCaster` objects with array-based fallback."""

    def __init__(self, array_registry: ArrowCastRegistry | None = None) -> None:
        self._casters: Dict[Tuple[str, str], ArrowScalarCaster] = {}
        self._array_registry = array_registry or ArrowCastRegistry()

    @property
    def array_registry(self) -> ArrowCastRegistry:
        """Access the underlying array registry used for fallbacks."""

        return self._array_registry

    def register(self, caster: ArrowScalarCaster) -> None:
        """Register an :class:`ArrowScalarCaster` for reuse."""

        if caster.array_caster is None:
            array_caster = self._array_registry.get_or_build(
                caster.source_field, caster.target_field
            )
            caster = ArrowScalarCaster(
                source_field=caster.source_field,
                target_field=caster.target_field,
                array_caster=array_caster,
            )
        self._casters[self._key(caster.source_field, caster.target_field)] = caster

    def get_or_build(
        self, source_field: pa.Field, target_field: pa.Field
    ) -> ArrowScalarCaster:
        """Return a caster, deriving one from the array registry when necessary."""

        key = self._key(source_field, target_field)
        cached = self._casters.get(key)
        if cached is not None:
            return cached

        array_caster = self._array_registry.get_or_build(source_field, target_field)
        caster = ArrowScalarCaster(
            source_field=source_field,
            target_field=target_field,
            array_caster=array_caster,
        )
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


__all__ = ["ArrowScalarCaster", "ArrowScalarCastRegistry"]
