"""Casting options for Arrow- and engine-aware conversions."""

import dataclasses
from typing import Any, List, Optional, Union, TYPE_CHECKING

import pyarrow as pa

from ..python_arrow import is_arrow_type_list_like

if TYPE_CHECKING:
    import polars
    import pyspark

__all__ = [
    "CastOptions",
    "CastOptionsArg",
]

CastOptionsArg = Union[
    "CastOptions",
    dict,
    pa.DataType,
    pa.Field,
    pa.Schema,
    None,
]


@dataclasses.dataclass
class CastOptions:
    """Options controlling Arrow casting behavior.

    Attributes
    ----------
    safe:
        If True, only allow "safe" casts (delegated to pyarrow.compute.cast).
    add_missing_columns:
        If True, create default-valued columns/fields when the target schema
        has fields missing in the source.
    strict_match_names:
        If True, only match fields/columns by exact name (case-sensitive).
        If False, allows case-insensitive and positional matching.
    allow_add_columns:
        If True, allow additional columns beyond the target schema to remain.
        If False, extra columns are dropped.
    eager:
        If True, enable eager casting behavior.
    datetime_patterns:
        Optional list of datetime parsing patterns.
    merge:
        If True, enable merge casting.
    source_arrow_field:
        Describes the source field/schema. Used to infer nullability behavior.
    target_arrow_field:
        Describes the target field/schema.
    """

    safe: bool = False
    add_missing_columns: bool = True
    strict_match_names: bool = False
    allow_add_columns: bool = False
    eager: bool = False
    datetime_patterns: Optional[List[str]] = None
    merge: bool = False

    source_arrow_field: Optional[pa.Field] = None
    _source_spark_field: Optional["pyspark.sql.types.StructField"] = dataclasses.field(
        default=None, init=False, repr=False
    )
    _source_polars_field: Optional["polars.Field"] = dataclasses.field(
        default=None, init=False, repr=False
    )

    target_arrow_field: Optional[pa.Field] = None
    _target_spark_field: Optional["pyspark.sql.types.StructField"] = dataclasses.field(
        default=None, init=False, repr=False
    )
    _target_polars_field: Optional["polars.Field"] = dataclasses.field(
        default=None, init=False, repr=False
    )

    arrow_memory_pool: Optional[pa.MemoryPool] = dataclasses.field(
        default=None, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def safe_init(
        cls,
        safe: bool = False,
        add_missing_columns: bool = True,
        strict_match_names: bool = False,
        allow_add_columns: bool = False,
        eager: bool = False,
        datetime_patterns: Optional[List[str]] = None,
        merge: bool = False,
        source_field: Optional[Union[pa.Field, pa.Schema, pa.DataType]] = None,
        target_field: Optional[Union[pa.Field, pa.Schema, pa.DataType]] = None,
    ) -> "CastOptions":
        """Build a CastOptions instance with optional source/target fields.

        Parameters
        ----------
        safe:
            Enable safe casting.
        add_missing_columns:
            Add missing columns when True.
        strict_match_names:
            Require exact field-name matches when True.
        allow_add_columns:
            Preserve extra columns when True.
        eager:
            Enable eager casting when True.
        datetime_patterns:
            Optional datetime parsing patterns.
        merge:
            Enable merge casting when True.
        source_field:
            Optional source Arrow field / schema / type.
        target_field:
            Optional target Arrow field / schema / type.

        Returns
        -------
        CastOptions
        """
        instance = cls(
            safe=safe,
            add_missing_columns=add_missing_columns,
            strict_match_names=strict_match_names,
            allow_add_columns=allow_add_columns,
            eager=eager,
            datetime_patterns=datetime_patterns,
            merge=merge,
        )

        if source_field is not None:
            instance.source_field = source_field

        if target_field is not None:
            instance.target_field = target_field

        return instance

    def copy(
        self,
        safe: bool = False,
        add_missing_columns: Optional[bool] = None,
        strict_match_names: bool = False,
        allow_add_columns: bool = False,
        eager: bool = False,
        datetime_patterns: Optional[List[str]] = None,
        merge: bool = False,
        source_field: Optional[Union[pa.Field, pa.Schema, pa.DataType]] = None,
        target_field: Optional[Union[pa.Field, pa.Schema, pa.DataType]] = None,
    ) -> "CastOptions":
        """Return a new CastOptions with selected fields overridden.

        Boolean flags are OR-ed with the current instance's values so that
        enabling a flag is additive rather than destructive. ``add_missing_columns``
        falls back to ``self.add_missing_columns`` when *None* is passed.
        Fields (source / target) replace the current value only when explicitly
        provided.
        """
        return self.safe_init(
            safe=self.safe or safe,
            add_missing_columns=self.add_missing_columns if add_missing_columns is None else add_missing_columns,
            strict_match_names=self.strict_match_names or strict_match_names,
            allow_add_columns=self.allow_add_columns or allow_add_columns,
            eager=self.eager or eager,
            datetime_patterns=self.datetime_patterns or datetime_patterns,
            merge=self.merge or merge,
            source_field=source_field if source_field is not None else self.source_arrow_field,
            target_field=target_field if target_field is not None else self.target_arrow_field,
        )

    @classmethod
    def check_arg(
        cls,
        options: CastOptionsArg = None,
        source_field: Optional[Union[pa.Field, pa.Schema, pa.DataType]] = None,
        target_field: Optional[Union[pa.Field, pa.Schema, pa.DataType]] = None,
        **kwargs,
    ) -> "CastOptions":
        """Normalise an argument into a CastOptions instance.

        * If *options* is already a ``CastOptions``, return it (optionally
          updated via :meth:`copy` when extra arguments are supplied).
        * Otherwise treat *options* as a target-field hint (falls back to
          *target_field*) and build a fresh instance via :meth:`safe_init`.

        Parameters
        ----------
        options:
            Existing ``CastOptions``, a raw Arrow type / field / schema, or
            ``None``.
        source_field:
            Optional source field override.
        target_field:
            Optional target field override.
        **kwargs:
            Additional keyword arguments forwarded to :meth:`safe_init` /
            :meth:`copy`.
        """
        if isinstance(options, CastOptions):
            if kwargs or source_field is not None or target_field is not None:
                return options.copy(
                    source_field=source_field,
                    target_field=target_field,
                    **kwargs,
                )
            return options

        # Treat a bare type/field/schema as an implicit target_field.
        resolved_target = target_field if target_field is not None else options
        return cls.safe_init(
            source_field=source_field,
            target_field=resolved_target,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Source field
    # ------------------------------------------------------------------

    def check_source(self, obj: Any) -> "CastOptions":
        """Set the source field from *obj* if not already configured.

        Parameters
        ----------
        obj:
            Source object to infer the field from.

        Returns
        -------
        self
        """
        if self.source_arrow_field is None and obj is not None:
            self.source_field = obj
        return self

    @property
    def source_field(self) -> Optional[pa.Field]:
        """The configured source Arrow field."""
        return self.source_arrow_field

    @source_field.setter
    def source_field(self, value: Any) -> None:
        """Set the source Arrow field, converting from schema / type if needed.

        Resets cached Polars and Spark fields so they are recomputed from the
        new value on next access.
        """
        if value is not None and not isinstance(value, pa.Field):
            from .arrow_cast import any_to_arrow_field

            value = any_to_arrow_field(value, None)
        object.__setattr__(self, "source_arrow_field", value)
        object.__setattr__(self, "_source_polars_field", None)
        object.__setattr__(self, "_source_spark_field", None)

    def source_child_arrow_field(self, index: int) -> pa.Field:
        """Return a child source Arrow field by *index*."""
        return self._child_arrow_field(self.source_arrow_field, index=index)

    @property
    def source_polars_field(self) -> Optional["polars.Field"]:
        """Lazily computed Polars field for the source."""
        if self.source_arrow_field is not None and self._source_polars_field is None:
            from ...polars.cast import arrow_field_to_polars_field

            object.__setattr__(
                self,
                "_source_polars_field",
                arrow_field_to_polars_field(self.source_arrow_field, None),
            )
        return self._source_polars_field

    @property
    def source_spark_field(self) -> Optional["pyspark.sql.types.StructField"]:
        """Lazily computed Spark field for the source."""
        if self.source_arrow_field is not None and self._source_spark_field is None:
            from ...spark.cast import arrow_field_to_spark_field

            object.__setattr__(
                self,
                "_source_spark_field",
                arrow_field_to_spark_field(self.source_arrow_field, None),
            )
        return self._source_spark_field

    # ------------------------------------------------------------------
    # Target field
    # ------------------------------------------------------------------

    @property
    def target_field(self) -> Optional[pa.Field]:
        """The configured target Arrow field."""
        return self.target_arrow_field

    @target_field.setter
    def target_field(self, value: Any) -> None:
        """Set the target Arrow field, converting from schema / type if needed.

        Resets cached Polars and Spark fields so they are recomputed from the
        new value on next access.
        """
        if value is not None and not isinstance(value, pa.Field):
            from .arrow_cast import any_to_arrow_field

            value = any_to_arrow_field(value, None)
        object.__setattr__(self, "target_arrow_field", value)
        object.__setattr__(self, "_target_polars_field", None)
        object.__setattr__(self, "_target_spark_field", None)

    @property
    def target_field_name(self) -> Optional[str]:
        """Effective target field name, falling back to the source field name."""
        if self.target_field is None:
            return self.source_field.name if self.source_field is not None else None
        if not self.target_field.name and self.source_field is not None:
            return self.source_field.name
        return self.target_field.name

    def target_child_arrow_field(self, index: int) -> pa.Field:
        """Return a child target Arrow field by *index*."""
        return self._child_arrow_field(self.target_arrow_field, index=index)

    @property
    def target_polars_field(self) -> Optional["polars.Field"]:
        """Lazily computed Polars field for the target."""
        if self.target_arrow_field is not None and self._target_polars_field is None:
            from ...polars.cast import arrow_field_to_polars_field

            object.__setattr__(
                self,
                "_target_polars_field",
                arrow_field_to_polars_field(self.target_arrow_field, None),
            )
        return self._target_polars_field

    @property
    def target_polars_schema(self) -> Optional[dict[str, "polars.DataType"]]:
        """Polars schema dict derived from the target field (struct only)."""
        polars_field = self.target_polars_field
        if polars_field is None:
            return None

        from ...polars.lib import polars

        polars_type: polars.Struct = polars_field.dtype
        return {field.name: field.dtype for field in polars_type.fields}

    @property
    def target_spark_field(self) -> Optional["pyspark.sql.types.StructField"]:
        """Lazily computed Spark field for the target."""
        if self.target_arrow_field is not None and self._target_spark_field is None:
            from ...spark.cast import arrow_field_to_spark_field

            object.__setattr__(
                self,
                "_target_spark_field",
                arrow_field_to_spark_field(self.target_arrow_field, None),
            )
        return self._target_spark_field

    @property
    def target_arrow_schema(self) -> Optional[pa.Schema]:
        """Schema view of ``target_field``.

        For struct types the children are unwrapped into schema fields;
        otherwise the field is treated as a single-field schema.
        """
        if self.target_field is None:
            return None
        from .arrow_cast import arrow_field_to_schema

        return arrow_field_to_schema(self.target_field, None)

    @property
    def target_spark_schema(self) -> Optional["pyspark.sql.types.StructType"]:
        """Spark schema view of the target Arrow schema."""
        arrow_schema = self.target_arrow_schema
        if arrow_schema is None:
            return None
        from ...spark.cast import arrow_schema_to_spark_schema

        return arrow_schema_to_spark_schema(arrow_schema, None)

    # ------------------------------------------------------------------
    # Cast-need predicates
    # ------------------------------------------------------------------

    def need_arrow_type_cast(self, source_obj: Any) -> bool:
        """Return True when the source and target Arrow types differ."""
        if self.target_field is None:
            return False
        self.check_source(source_obj)
        return self.source_field.type != self.target_field.type

    def need_polars_type_cast(self, source_obj: Any) -> bool:
        """Return True when the source and target Polars dtypes differ."""
        if self.target_polars_field is None:
            return False
        self.check_source(source_obj)
        return self.source_polars_field.dtype != self.target_polars_field.dtype

    def need_spark_type_cast(self, source_obj: Any) -> bool:
        """Return True when the source and target Spark dataTypes differ."""
        if self.target_spark_field is None:
            return False
        self.check_source(source_obj)
        return self.source_spark_field.dataType != self.target_spark_field.dataType

    def need_nullability_fill(self, source_obj: Any) -> bool:
        """Return True when the source is nullable but the target is not."""
        if self.target_field is None:
            return False
        self.check_source(source_obj)
        return self.source_field.nullable and not self.target_field.nullable

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _child_arrow_field(arrow_field: pa.Field, index: int) -> pa.Field:
        """Return a child Arrow field by *index* for nested types.

        For structs, returns the field at *index*. For list-like types,
        returns the value field. For maps, synthesises an ``entries`` struct
        field containing the key and item fields. Non-nested types are
        returned unchanged.
        """
        source_type: Union[pa.DataType, pa.ListType, pa.StructType, pa.MapType] = (
            arrow_field.type
        )

        if not pa.types.is_nested(source_type):
            return arrow_field

        if pa.types.is_struct(source_type):
            return source_type.field(index)

        if is_arrow_type_list_like(source_type):
            return source_type.value_field

        if pa.types.is_map(source_type):
            return pa.field(
                "entries",
                pa.struct([source_type.key_field, source_type.item_field]),
                nullable=False,
            )

        raise NotImplementedError(f"Unsupported nested Arrow type: {source_type}")
