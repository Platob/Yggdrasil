import dataclasses
from dataclasses import replace as dc_replace
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
    rename:
        Reserved / placeholder for rename behavior (currently unused).
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
    datetime_patterns: Optional[List[str]] = None

    source_field: Optional[pa.Field] = None
    _spark_source_field: Optional["pyspark.sql.types.StructField"] = dataclasses.field(default=None, init=False, repr=False)
    _polars_source_field: Optional["polars.Field"] = dataclasses.field(default=None, init=False, repr=False)

    target_field: Optional[pa.Field] = None
    _spark_target_field: Optional["pyspark.sql.types.StructField"] = dataclasses.field(default=None, init=False, repr=False)
    _polars_target_field: Optional["polars.Field"] = dataclasses.field(default=None, init=False, repr=False)

    _memory_pool: Optional[pa.MemoryPool] = dataclasses.field(default=None, init=False, repr=False)
    _spark_session: Optional["pyspark.sql.SparkSession"] = dataclasses.field(default=None, init=False, repr=False)

    @classmethod
    def safe_init(cls, *args, **kwargs):
        return cls.__safe_init__(*args, **kwargs)

    def copy(
        self,
        safe: Optional[bool] = None,
        add_missing_columns: Optional[bool] = None,
        strict_match_names: Optional[bool] = None,
        allow_add_columns: Optional[bool] = None,
        rename: Optional[bool] = None,
        datetime_patterns: Optional[List[str]] = None,
        source_field: Optional[pa.Field] = None,
        target_field: Optional[pa.Field] = None,
        memory_pool: Optional[pa.MemoryPool] = None,
        spark_session: Optional["pyspark.sql.SparkSession"] = None,
        **kwargs
    ):
        """
        Return a new ArrowCastOptions instance with updated fields.
        """
        instance = dc_replace(
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
            datetime_patterns=datetime_patterns,
            source_field=self.source_field if source_field is None else source_field,
            target_field=self.target_field if target_field is None else target_field,
        )

        memory_pool = memory_pool or self.get_memory_pool()
        if memory_pool is not None:
            instance.set_memory_pool(memory_pool)

        spark_session = spark_session or self.get_spark_session(raise_error=False)
        if spark_session is not None:
            instance.set_spark_session(spark_session)

        return instance

    @classmethod
    def check_arg(
        cls,
        target_field: Union[
            "CastOptions",
            dict,
            pa.DataType,
            pa.Field,
            pa.Schema,
            None,
        ] = None,
        kwargs: Optional[dict] = None,
        **options
    ) -> "CastOptions":
        """
        Normalize an argument into an ArrowCastOptions instance.

        - If `arg` is already ArrowCastOptions, return it.
        - Otherwise, treat `arg` as something convertible to pa.Field via
          the registry (`convert(arg, Optional[pa.Field])`) and apply it
          as `target_field` on top of DEFAULT_CAST_OPTIONS.
        - If arg is None, just use DEFAULT_CAST_OPTIONS.
        """
        if isinstance(target_field, CastOptions):
            result = target_field
        elif target_field is None:
            result = CastOptions()
        else:
            if target_field is None:
                result = CastOptions()
            else:
                target_field = target_field if isinstance(target_field, pa.Field) else convert(target_field, pa.Field)

                result = CastOptions(target_field=target_field)

        if options:
            result = result.copy(**options)

        if kwargs:
            result = result.copy(**kwargs, **options)

        return result

    def get_memory_pool(self) -> Optional[pa.MemoryPool]:
        """
        Arrow memory pool used during casting operations.
        """
        return self._memory_pool

    def set_memory_pool(self, value: pa.MemoryPool) -> None:
        """
        Set the Arrow memory pool used during casting operations.
        """
        object.__setattr__(self, "_memory_pool", value)

    def get_spark_session(self, raise_error: bool = True) -> Optional["pyspark.sql.SparkSession"]:
        """
        Spark session used during casting operations.
        """
        if self._spark_session is None and pyspark is not None:
            active = pyspark.sql.SparkSession.getActiveSession()

            if raise_error and active is None:
                raise ValueError("No active Spark session found. Please set the spark_session property explicitly.")

            object.__setattr__(self, "_spark_session", active)
        return self._spark_session

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
        if cast:
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