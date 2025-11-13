from __future__ import annotations

import dataclasses
from typing import ClassVar, TYPE_CHECKING, Any, Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types as pa_types

# Conditionally import polars to avoid hard dependency
if TYPE_CHECKING:
    try:
        import polars as pl
        HAS_POLARS = True
    except ImportError:
        HAS_POLARS = False
        pl = Any  # type: ignore
else:
    try:
        import polars as pl
        HAS_POLARS = True
    except ImportError:
        HAS_POLARS = False
        pl = None  # type: ignore

ArrowDataTypeOrField = pa.DataType | pa.Field
# Include Polars Series in ArrayLike
ArrayLike = Union[pa.Array, pa.ChunkedArray, "pl.Series"]


class ArrowUtility:
    @staticmethod
    def ensure_arrow_type(obj: ArrowDataTypeOrField):
        if isinstance(obj, pa.DataType):
            return obj
        return obj.type

    @staticmethod
    def ensure_arrow_field(obj: ArrowDataTypeOrField) -> pa.Field:
        if isinstance(obj, pa.Field):
            return obj
        return pa.field("value", obj, nullable=True)

    @staticmethod
    def get_nested_fields(dtype: ArrowDataTypeOrField) -> list[pa.Field]:
        dtype = ArrowUtility.ensure_arrow_type(dtype)

        if not pa_types.is_nested(dtype):
            return []

        if pa_types.is_list(dtype) or pa_types.is_large_list(dtype):
            return [dtype.value_field]
        if pa_types.is_map(dtype):
            return [dtype.key_field, dtype.value_field]
        if pa_types.is_struct(dtype):
            return dtype.fields

        raise pa.ArrowTypeError(f"Cannot get nested fields from {dtype}")

    @staticmethod
    def can_convert_arrow_fields(
        source_field: ArrowDataTypeOrField,
        target_field: ArrowDataTypeOrField,
        safe: bool | None = None,
        check_names: bool | None = None
    ) -> bool:
        source_field = ArrowUtility.ensure_arrow_field(source_field)
        target_field = ArrowUtility.ensure_arrow_field(target_field)

        if check_names and source_field.name != target_field.name:
            return False

        source_type = ArrowUtility.ensure_arrow_type(source_field)
        target_type = ArrowUtility.ensure_arrow_type(target_field)

        if source_type.equals(target_type):
            return True

        if safe:
            if pa.types.is_string(source_type) and pa.types.is_string(target_type):
                return True
            if pa_types.is_integer(source_type) and pa_types.is_integer(target_type):
                return True
            if pa_types.is_floating(source_type) and pa_types.is_floating(target_type):
                return True
            if pa_types.is_decimal(source_type) and pa_types.is_decimal(target_type):
                return True
            if pa_types.is_nested(source_type) and pa_types.is_nested(target_type):
                source_fields = ArrowUtility.get_nested_fields(source_type)
                target_fields = ArrowUtility.get_nested_fields(target_type)

                return all(
                    ArrowUtility.can_convert_arrow_fields(s, t, check_names=check_names)
                    for s, t in zip(source_fields, target_fields)
                )

            return False

        return True


@dataclasses.dataclass(frozen=True)
class ArrowCaster:
    """Cast :class:`pyarrow.Scalar` instances between two declared fields."""

    source_field: pa.Field
    target_field: pa.Field

    def cast_array(
        self,
        arr: ArrayLike,
        target_type: ArrowDataTypeOrField | None = None,
        safe: bool | None = None,
        memory_pool: pa.MemoryPool | None = None,
    ):
        checked_target_type = ArrowUtility.ensure_arrow_type(target_type or self.target_field)

        # Handle Polars Series - convert to Arrow array first
        if HAS_POLARS and isinstance(arr, pl.Series):
            arr = arr.to_arrow()

        return pc.cast(
            arr=arr,
            target_type=checked_target_type,
            safe=safe,
            memory_pool=memory_pool,
        )

    def cast_scalar(
        self,
        scalar: pa.Scalar,
        target_type: ArrowDataTypeOrField | None = None,
        safe: bool | None = None,
        memory_pool: pa.MemoryPool | None = None,
    ):
        checked_target_type = ArrowUtility.ensure_arrow_type(target_type or self.target_field)

        return scalar.cast(
            target_type=checked_target_type,
            safe=safe,
            memory_pool=memory_pool,
        )


@dataclasses.dataclass
class ArrowCastRegistry:
    cache: dict[tuple[str, str], ArrowCaster] = dataclasses.field(default_factory=dict)
    _instance: ClassVar[ArrowCastRegistry | None] = None

    @staticmethod
    def inner_key(source_field: ArrowDataTypeOrField, target_field: ArrowDataTypeOrField) -> tuple[str, str]:
        return (
            str(ArrowUtility.ensure_arrow_type(source_field)),
            str(ArrowUtility.ensure_arrow_type(target_field)),
        )

    def register(self, caster: ArrowCaster) -> ArrowCaster:
        key = self.inner_key(caster.source_field, caster.target_field)
        self.cache[key] = caster
        return caster

    def get(
        self,
        source_field: ArrowDataTypeOrField,
        target_field: ArrowDataTypeOrField,
    ) -> ArrowCaster | None:
        key = self.inner_key(source_field, target_field)
        return self.cache.get(key)

    def get_or_build(
        self,
        source_field: ArrowDataTypeOrField,
        target_field: ArrowDataTypeOrField,
        safe: bool | None = None,
    ) -> ArrowCaster:
        if not ArrowUtility.can_convert_arrow_fields(source_field, target_field, safe=safe):
            safe_str = " safely" if safe else ""
            raise pa.ArrowTypeError(
                f"Cannot convert {source_field} to {target_field}{safe_str}"
            )

        source_field = ArrowUtility.ensure_arrow_field(source_field)
        target_field = ArrowUtility.ensure_arrow_field(target_field)

        cached = self.get(source_field, target_field)
        if cached is not None:
            return cached

        caster = ArrowCaster(source_field=source_field, target_field=target_field)
        return self.register(caster)

    @classmethod
    def instance(cls) -> ArrowCastRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


__all__ = [
    "ArrowCaster",
    "ArrowCastRegistry",
    "ArrowUtility",
    "ARROW_CAST_REGISTRY",
    "ArrowDataTypeOrField",
    "ArrayLike",
    "HAS_POLARS",
]


ARROW_CAST_REGISTRY = ArrowCastRegistry.instance()
