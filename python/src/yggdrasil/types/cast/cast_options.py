import dataclasses
from typing import Optional, Union, List

import pyarrow as pa

from .registry import convert
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
    eager: bool = False
    datetime_patterns: Optional[List[str]] = None

    source_field: Optional[pa.Field] = None
    _spark_source_field: Optional["pyspark.sql.types.StructField"] = dataclasses.field(default=None, init=False, repr=False)
    _polars_source_field: Optional["polars.Field"] = dataclasses.field(default=None, init=False, repr=False)

    target_field: Optional[pa.Field] = None
    _spark_target_field: Optional["pyspark.sql.types.StructField"] = dataclasses.field(default=None, init=False, repr=False)
    _polars_target_field: Optional["polars.Field"] = dataclasses.field(default=None, init=False, repr=False)

    memory_pool: Optional[pa.MemoryPool] = dataclasses.field(default=None, init=False, repr=False)

    @classmethod
    def safe_init(cls, *args, **kwargs):
        return cls.__safe_init__(*args, **kwargs)

    def copy(
        self,
        **kwargs
    ):
        """
        Return a new ArrowCastOptions instance with updated fields.
        """
        if kwargs:
            return dataclasses.replace(self, **kwargs)
        return self

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
            result = CastOptions()

            result.set_target_arrow_field(value=options, cast=True)

        if kwargs:
            result = result.copy(**kwargs)

        return result

    def set_spark_session(self, value: "pyspark.sql.SparkSession") -> None:
        """
        Set the Spark session used during casting operations.
        """
        object.__setattr__(self, "_spark_session", value)

    def get_target_arrow_field(self) -> Optional[pa.Field]:
        """
        Set the target_field used during casting operations.
        """
        return self.target_field

    def set_target_arrow_field(self, value: pa.Field, cast: bool = False) -> None:
        """
        Set the target_field used during casting operations.
        """
        if value is not None and not isinstance(value, pa.Field) and cast:
            value = convert(value, Optional[pa.Field])

        object.__setattr__(self, "target_field", value)

    def get_target_polars_field(self):
        if self.target_field is not None and self._polars_target_field is None:
            from ...types.cast.polars_cast import arrow_field_to_polars_field

            setattr(self, "_polars_target_field", arrow_field_to_polars_field(self.target_field))
        return self._polars_target_field

    def set_target_polars_field(self, value: "polars.Field", cast: bool = False) -> None:
        object.__setattr__(self, "_polars_target_field", value)

    def get_source_spark_field(self):
        if self.source_field is not None and self._spark_source_field is None:
            from ...types.cast.spark_cast import arrow_field_to_spark_field

            setattr(self, "_spark_source_field", arrow_field_to_spark_field(self.source_field))
        return self._spark_source_field

    def set_source_spark_field(self, value: "pyspark.sql.types.StructField") -> None:
        object.__setattr__(self, "_spark_source_field", value)

    def get_target_spark_field(self):
        if self.target_field is not None and self._spark_target_field is None:
            from ...types.cast.spark_cast import arrow_field_to_spark_field

            setattr(self, "_spark_target_field", arrow_field_to_spark_field(self.target_field))
        return self._spark_target_field

    def set_target_spark_field(self, value: "pyspark.sql.types.StructField") -> None:
        object.__setattr__(self, "_spark_target_field", value)

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