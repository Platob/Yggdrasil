import dataclasses
from typing import Optional, Union, List, Any

import pyarrow as pa

from .registry import convert
from ..python_arrow import is_arrow_type_list_like
from ...dataclasses import yggdataclass

__all__ = [
    "CastOptions",
]

from ...libs import pyspark, polars


@yggdataclass
class CastOptions:
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
    source_arrow_field:
        Description of the source field/schema. Used to infer nullability behavior.
        Can be a pa.Field, pa.Schema, or pa.DataType (normalized elsewhere).
    target_arrow_field:
        Description of the target field/schema. Can be pa.Field, pa.Schema,
        or pa.DataType (normalized elsewhere).
    """
    safe: bool = False
    add_missing_columns: bool = True
    strict_match_names: bool = False
    allow_add_columns: bool = False
    eager: bool = False
    datetime_patterns: Optional[List[str]] = None

    source_arrow_field: Optional[pa.Field] = None
    _source_spark_field: Optional["pyspark.sql.types.StructField"] = dataclasses.field(default=None, init=False, repr=False)
    _source_polars_field: Optional["polars.Field"] = dataclasses.field(default=None, init=False, repr=False)

    target_arrow_field: Optional[pa.Field] = None
    _target_spark_field: Optional["pyspark.sql.types.StructField"] = dataclasses.field(default=None, init=False, repr=False)
    _target_polars_field: Optional["polars.Field"] = dataclasses.field(default=None, init=False, repr=False)

    arrow_memory_pool: Optional[pa.MemoryPool] = dataclasses.field(default=None, init=False, repr=False)

    @classmethod
    def safe_init(
        cls,
        safe: bool = False,
        add_missing_columns: bool = True,
        strict_match_names: bool = False,
        allow_add_columns: bool = False,
        eager: bool = False,
        datetime_patterns: Optional[List[str]] = None,
        source_field: pa.Field | pa.Schema | pa.DataType | None = None,
        target_field: pa.Field | pa.Schema | pa.DataType | None = None,
        **kwargs
    ):
        built = CastOptions(
            safe=safe,
            add_missing_columns=add_missing_columns,
            strict_match_names=strict_match_names,
            allow_add_columns=allow_add_columns,
            eager=eager,
            datetime_patterns=datetime_patterns,
            **kwargs
        )

        if source_field is not None:
            built.source_field = source_field

        if target_field is not None:
            built.target_field = target_field

        return built

    def copy(
        self,
        safe: bool = False,
        add_missing_columns: Optional[bool] = None,
        strict_match_names: bool = False,
        allow_add_columns: bool = False,
        eager: bool = False,
        datetime_patterns: Optional[List[str]] = None,
        source_field: pa.Field | pa.Schema | pa.DataType | None = None,
        source_arrow_field: pa.Field | None = None,
        target_field: pa.Field | pa.Schema | pa.DataType | None = None,
        target_arrow_field: pa.Field | None = None,
    ):
        """
        Return a new ArrowCastOptions instance with updated fields.
        """
        add_missing_columns = self.add_missing_columns if add_missing_columns is None else add_missing_columns

        built = self.safe_init(
            safe=self.safe or safe,
            add_missing_columns=add_missing_columns,
            strict_match_names=self.strict_match_names or strict_match_names,
            allow_add_columns=self.allow_add_columns or allow_add_columns,
            eager=self.eager or eager,
            datetime_patterns=self.datetime_patterns or datetime_patterns,
            source_arrow_field=self.source_arrow_field if source_arrow_field is None else source_arrow_field,
            target_arrow_field=self.target_arrow_field if target_arrow_field is None else target_arrow_field,
        )

        if source_field is not None:
            built.source_field = source_field

        if target_field is not None:
            built.target_field = target_field

        return built

    @classmethod
    def check_arg(
        cls,
        options: Union[
            "CastOptions",
            dict,
            pa.DataType,
            pa.Field,
            pa.Schema,
            None,
        ] = None,
        source_field: pa.Field | pa.Schema | pa.DataType | None = None,
        target_field: pa.Field | pa.Schema | pa.DataType | None = None,
        **kwargs
    ) -> "CastOptions":
        """
        Normalize an argument into an ArrowCastOptions instance.

        - If `arg` is already ArrowCastOptions, return it.
        - Otherwise, treat `arg` as something convertible to pa.Field via
          the registry (`convert(arg, Optional[pa.Field])`) and apply it
          as `target_field` on top of DEFAULT_CAST_OPTIONS.
        - If arg is None, just use DEFAULT_CAST_OPTIONS.
        """
        if isinstance(options, CastOptions):
            result = options
        else:
            t = target_field if target_field is not None else options

            return cls.safe_init(
                source_field=source_field,
                target_field=t,
                **kwargs
            )

        if kwargs or source_field is not None or target_field is not None:
            result = result.copy(
                target_field=target_field,
                source_field=source_field,
                **kwargs
            )

        return result

    def check_source(self, obj: Any):
        if self.source_field is not None or obj is None:
            return self

        self.source_field = obj

        return self

    def need_arrow_type_cast(self, source_obj: Any):
        if self.target_field is None:
            return False

        self.check_source(source_obj)

        return self.source_field.type != self.target_field.type

    def need_polars_type_cast(self, source_obj: Any):
        if self.target_polars_field is None:
            return False

        self.check_source(source_obj)

        return self.source_polars_field.dtype != self.target_polars_field.dtype

    def need_spark_type_cast(self, source_obj: Any):
        if self.target_spark_field is None:
            return False

        self.check_source(source_obj)

        return self.source_spark_field.dataType != self.target_spark_field.dataType

    def need_nullability_check(self, source_obj: Any):
        if self.target_field is None:
            return False

        self.check_source(source_obj)

        return self.source_field.nullable and not self.target_field.nullable

    @staticmethod
    def _child_arrow_field(
        arrow_field: pa.Field,
        index: int
    ):
        source_type: Union[
            pa.DataType, pa.ListType, pa.StructType, pa.MapType
        ] = arrow_field.type

        if pa.types.is_nested(source_type):
            if pa.types.is_struct(source_type):
                return source_type.field(index)
            elif is_arrow_type_list_like(source_type):
                return source_type.value_field
            elif pa.types.is_map(source_type):
                return pa.field(
                    "entries",
                    pa.struct([source_type.key_field, source_type.item_field]),
                    nullable=False
                )
            else:
                raise NotImplementedError()
        else:
            return arrow_field

    @property
    def source_field(self):
        return self.source_arrow_field

    @source_field.setter
    def source_field(self, value: Any):
        """
        Set the target_field used during casting operations.
        """
        if value is not None:
            value = value if isinstance(value, pa.Field) else convert(value, pa.Field)

        object.__setattr__(self, "source_arrow_field", value)

    def source_child_arrow_field(self, index: int):
        return self._child_arrow_field(self.source_arrow_field, index=index)

    @property
    def source_polars_field(self):
        if self.source_arrow_field is not None and self._source_polars_field is None:
            from ...types.cast.polars_cast import arrow_field_to_polars_field

            setattr(self, "_source_polars_field", arrow_field_to_polars_field(self.source_arrow_field))
        return self._source_polars_field

    @property
    def source_spark_field(self):
        if self.source_arrow_field is not None and self._source_spark_field is None:
            from ...types.cast.spark_cast import arrow_field_to_spark_field

            setattr(self, "_source_spark_field", arrow_field_to_spark_field(self.source_field))
        return self._source_spark_field

    @property
    def target_field(self) -> Optional[pa.Field]:
        """
        Set the target_field used during casting operations.
        """
        return self.target_arrow_field

    @property
    def target_field_name(self):
        if self.target_field is None:
            return ""

        if not self.target_field.name and self.source_field:
            return self.source_field.name
        return self.target_field.name

    @target_field.setter
    def target_field(self, value: Any) -> None:
        """
        Set the target_field used during casting operations.
        """
        if value is not None:
            value = value if isinstance(value, pa.Field) else convert(value, pa.Field)

        object.__setattr__(self, "target_arrow_field", value)

    def target_child_arrow_field(self, index: int):
        return self._child_arrow_field(self.target_arrow_field, index=index)

    @property
    def target_polars_field(self):
        if self.target_arrow_field is not None and self._target_polars_field is None:
            from ...types.cast.polars_cast import arrow_field_to_polars_field

            setattr(self, "_target_polars_field", arrow_field_to_polars_field(self.target_arrow_field))
        return self._target_polars_field

    @property
    def target_spark_field(self):
        if self.target_arrow_field is not None and self._target_spark_field is None:
            from ...types.cast.spark_cast import arrow_field_to_spark_field

            setattr(self, "_target_spark_field", arrow_field_to_spark_field(self.target_field))
        return self._target_spark_field

    @property
    def target_arrow_schema(self) -> Optional[pa.Schema]:
        """
        Schema view of `target_field`.

        - If target_field is a struct, unwrap its children as schema fields.
        - Otherwise treat target_field as a single-field schema.
        """
        if self.target_field is not None:
            from .arrow_cast import arrow_field_to_schema

            return arrow_field_to_schema(self.target_field, None)
        return None

    @property
    def target_spark_schema(self) -> Optional["pyspark.sql.types.StructType"]:
        arrow_schema = self.target_arrow_schema

        if arrow_schema is not None:
            from .spark_cast import arrow_schema_to_spark_schema

            return arrow_schema_to_spark_schema(arrow_schema)
        return arrow_schema


DEFAULT_INSTANCE = CastOptions()