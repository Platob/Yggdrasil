import dataclasses
from typing import Tuple, Dict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types as pa_types

ArrowDataTypeOrField = pa.DataType | pa.Field
ArrayLike = pa.Array | pa.ChunkedArray


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
    def can_convert_arrow_fields(
        source_field: ArrowDataTypeOrField,
        target_field: ArrowDataTypeOrField,
        safe: bool | None = None
    ) -> bool:
        source_field = ArrowUtility.ensure_arrow_field(source_field)
        target_field = ArrowUtility.ensure_arrow_field(target_field)

        source_type = ArrowUtility.ensure_arrow_type(source_field)
        target_type = ArrowUtility.ensure_arrow_type(target_field)

        if source_type.equals(target_type):
            return True

        if pa.types.is_string(source_type):
            if safe:
                return pa.types.is_string(target_type)
            return True

        if pa_types.is_integer(source_type) and pa_types.is_integer(target_type):
            return True

        if pa_types.is_floating(source_type) and pa_types.is_floating(target_type):
            return True

        if (
            (pa_types.is_list(source_type) or pa_types.is_large_list(source_type) or pa_types.is_fixed_size_list(source_type))
            and (pa_types.is_list(target_type) or pa_types.is_large_list(target_type) or pa_types.is_fixed_size_list(target_type))
        ):
            return ArrowUtility.can_convert_arrow_fields(
                source_type.value_field, target_type.value_field,
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
                ArrowUtility.can_convert_arrow_fields(
                    source_children[name],
                    target_children[name],
                    safe=safe
                )
                for name in source_children
            )

        return False


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
        memory_pool: pa.MemoryPool | None = None
    ):
        checked_target_type = ArrowUtility.ensure_arrow_type(target_type or self.target_field)

        return pc.cast(
            arr=arr,
            target_type=checked_target_type,
            safe=safe,
            memory_pool=memory_pool
        )

    def cast_scalar(
        self,
        scalar: pa.Scalar,
        target_type: ArrowDataTypeOrField | None = None,
        safe: bool | None = None,
        memory_pool: pa.MemoryPool | None = None
    ):
        checked_target_type = ArrowUtility.ensure_arrow_type(target_type or self.target_field)

        return scalar.cast(
            target_type=checked_target_type,
            safe=safe,
            memory_pool=memory_pool
        )


@dataclasses.dataclass(frozen=True)
class ArrowCastRegistry:
    cache: Dict[(str, str), ArrowCaster] = dataclasses.field(default_factory=dict)

    @staticmethod
    def inner_key(source_field: ArrowDataTypeOrField, target_field: ArrowDataTypeOrField) -> Tuple[str, str]:
        return (
            str(ArrowUtility.ensure_arrow_type(source_field)),
            str(ArrowUtility.ensure_arrow_type(target_field))
        )

    def get_caster(
        self,
        source_field: ArrowDataTypeOrField,
        target_field: ArrowDataTypeOrField,
        safe: bool | None = None
    ):
        if not ArrowUtility.can_convert_arrow_fields(source_field, target_field):
            safe_str = " safely" if safe else ""
            raise pa.ArrowTypeError(
                f"Cannot convert {source_field} to {source_field}{safe_str}"
            )

        key = self.inner_key(source_field, target_field)
        found = self.cache.get(key)
        if found:
            return found

        raise pa.ArrowTypeError(f"No caster found from {source_field} to {source_field}")


__all__ = [
    "ArrowCaster",
    "ArrowCastRegistry"
]