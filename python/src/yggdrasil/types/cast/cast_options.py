from dataclasses import replace as dc_replace
from typing import Optional, Union, List

import pyarrow as pa

from .registry import convert
from ...dataclasses import yggdataclass


__all__ = [
    "ArrowCastOptions",
    "DEFAULT_CAST_OPTIONS",
]


@yggdataclass
class ArrowCastOptions:
    """
    Options controlling Arrow casting behavior.

    Attributes
    ----------
    safe:
        If True, only allow "safe" casts (delegated to pyarrow.compute.cast).
    add_missing_columns:
        If True, create default-valued columns/fields when target schema has
        fields that are missing in the source.
    strict_match_names:
        If True, only match fields/columns by exact name (case-sensitive).
        If False, allows case-insensitive and positional matching.
    allow_add_columns:
        If True, allow additional columns beyond the target schema to remain.
        If False, extra columns are effectively ignored.
    rename:
        Reserved / placeholder for rename behavior (currently unused).
    memory_pool:
        Optional Arrow memory pool passed down to compute kernels.
    source_field:
        Description of the source field/schema. Used to infer nullability behavior.
        Can be a pa.Field, pa.Schema, or pa.DataType (normalized elsewhere).
    target_field:
        Description of the target field/schema. Can be pa.Field, pa.Schema,
        or pa.DataType (normalized elsewhere).
    """
    safe: bool = False
    add_missing_columns: bool = True
    strict_match_names: bool = False
    allow_add_columns: bool = False
    rename: bool = True
    memory_pool: Optional[pa.MemoryPool] = None
    source_field: Optional[pa.Field] = None
    target_field: Optional[pa.Field] = None
    datetime_formats: Optional[List[str]] = None

    @classmethod
    def default_instance(cls):
        return DEFAULT_CAST_OPTIONS

    def copy(
        self,
        safe: Optional[bool] = None,
        add_missing_columns: Optional[bool] = None,
        strict_match_names: Optional[bool] = None,
        allow_add_columns: Optional[bool] = None,
        rename: Optional[bool] = None,
        memory_pool: Optional[pa.MemoryPool] = None,
        source_field: Optional[pa.Field] = None,
        target_field: Optional[pa.Field] = None,
        **kwargs
    ):
        """
        Return a new ArrowCastOptions instance with updated fields.
        """
        return dc_replace(
            self,
            safe=self.safe if safe is None else safe,
            add_missing_columns=(
                self.add_missing_columns
                if add_missing_columns is None
                else add_missing_columns
            ),
            strict_match_names=(
                self.strict_match_names
                if strict_match_names is None
                else strict_match_names
            ),
            allow_add_columns=(
                self.allow_add_columns
                if allow_add_columns is None
                else allow_add_columns
            ),
            rename=self.rename if rename is None else rename,
            memory_pool=self.memory_pool if memory_pool is None else memory_pool,
            source_field=self.source_field if source_field is None else source_field,
            target_field=self.target_field if target_field is None else target_field,
        )

    @classmethod
    def check_arg(
        cls,
        target_field: Union[
            "ArrowCastOptions",
            dict,
            pa.DataType,
            pa.Field,
            pa.Schema,
            None,
        ] = None,
        kwargs: Optional[dict] = None,
        **options
    ) -> "ArrowCastOptions":
        """
        Normalize an argument into an ArrowCastOptions instance.

        - If `arg` is already ArrowCastOptions, return it.
        - Otherwise, treat `arg` as something convertible to pa.Field via
          the registry (`convert(arg, Optional[pa.Field])`) and apply it
          as `target_field` on top of DEFAULT_CAST_OPTIONS.
        - If arg is None, just use DEFAULT_CAST_OPTIONS.
        """
        if isinstance(target_field, ArrowCastOptions):
            result = target_field
        else:
            result = dc_replace(
                DEFAULT_CAST_OPTIONS,
                target_field=convert(target_field, Optional[pa.Field]),
            )

        if options:
            result = result.copy(**options)

        if kwargs:
            result = result.copy(**kwargs, **options)

        return result

    @property
    def target_schema(self) -> Optional[pa.Schema]:
        """
        Schema view of `target_field`.

        - If target_field is a struct, unwrap its children as schema fields.
        - Otherwise treat target_field as a single-field schema.
        """
        if self.target_field is not None:
            from .arrow_cast import arrow_field_to_schema

            return arrow_field_to_schema(self.target_field, None)
        return None

    @target_schema.setter
    def target_schema(self, value: pa.Schema) -> None:
        """
        Set `target_field` from a `pa.Schema`, wrapping it as a root struct field.
        """
        self.target_field = pa.field(
            "root",
            pa.struct(list(value)),
            nullable=False,
            metadata=value.metadata,
        )


DEFAULT_CAST_OPTIONS = ArrowCastOptions()
