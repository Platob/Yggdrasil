"""Arrow DataClass implementation for Yggdrasil.

This module provides a decorator for creating dataclasses with PyArrow schema inference.
"""

from __future__ import annotations

import datetime as dt
import decimal as dec
from dataclasses import dataclass
from typing import TypeVar, Any, get_type_hints, get_origin, get_args, Union

import polars as pl
import pyarrow as pa

from ..utils.fake_module import make_fake_module

try:
    import pandas
except ImportError:
    pandas = make_fake_module(module_name="pandas")

from ..utils.py_utils import Annotated, safe_dict, safe_str, merge_dicts, safe_bool, safe_int
from ..utils.spark_utils import HAVE_SPARK, ARROW_TYPE_TO_SPARK_TYPE, cast_nested_spark_field, spark_to_arrow_type, spark_types, spark_sql, spark_functions

__all__ = [
    "DataField",
    "Annotated",
]

T = TypeVar("T")

# Type mapping from Python types to PyArrow types
_PYTHON_TO_ARROW_TYPE_MAP = {
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    str: pa.utf8(),
    bytes: pa.binary(),
    memoryview: pa.binary(),
    bytearray: pa.binary(),
    dec.Decimal: pa.decimal128(38,18),
    dt.datetime: pa.timestamp("us"),
    dt.date: pa.date32(),
}


@dataclass(frozen=True)
class DataField:
    name: str
    arrow_type: pa.DataType
    nullable: bool
    comment: str | None
    metadata: dict[str, str]
    children: list[DataField] | None

    def __post_init__(self):
        checked_metadata = safe_dict(
            self.metadata, default={},
            check_key=safe_str, check_value=safe_str
        )
        object.__setattr__(self, "metadata", checked_metadata)

        checked_comment = safe_str(self.comment, default=None)
        object.__setattr__(self, "comment", checked_comment)

        if pa.types.is_nested(self.arrow_type) and not self.children:
            if pa.types.is_struct(self.arrow_type):
                children = [
                    self.from_arrow_field(f)
                    for f in self.arrow_type
                ]
            elif (
                pa.types.is_list(self.arrow_type)
                or pa.types.is_large_list(self.arrow_type)
                or pa.types.is_fixed_size_list(self.arrow_type)
            ):
                list_type: pa.ListType = self.arrow_type
                children = [
                    self.from_arrow_field(field=list_type.value_field)
                ]
            elif pa.types.is_map(self.arrow_type):
                map_type: pa.MapType = self.arrow_type
                children = [
                    self.from_arrow_field(field=map_type.key_field),
                    self.from_arrow_field(field=map_type.item_field)
                ]
            else:
                raise ValueError(
                    f"Cannot initialize nested arrow type {self.arrow_type} without children set"
                )

            if children:
                children = [_.refine() for _ in children]

            object.__setattr__(self, "children", children)

    def refine(self):
        object.__setattr__(self, "comment", self.metadata.pop("comment", self.comment))

        if pa.types.is_timestamp(self.arrow_type):
            timeunit = self.metadata.pop("timeunit", self.arrow_type.unit)
            timezone = self.metadata.pop("timezone", self.arrow_type.tz)

            if self.arrow_type.unit != timeunit or self.arrow_type.tz != timezone:
                arrow_type = pa.timestamp(unit=timeunit, tz=timezone)
                object.__setattr__(self, "arrow_type", arrow_type)
        elif pa.types.is_time(self.arrow_type):
            timeunit = self.metadata.pop("timeunit", self.arrow_type.unit)

            if timeunit != self.arrow_type.unit:
                arrow_type = pa.time32(timeunit) if timeunit in ("s", "ms") else pa.time64(timeunit)
                object.__setattr__(self, "arrow_type", arrow_type)
        elif pa.types.is_decimal(self.arrow_type):
            precision = int(self.metadata.pop("precision", self.arrow_type.precision))
            scale = int(self.metadata.pop("scale", self.arrow_type.scale))

            if self.arrow_type.precision != precision or self.arrow_type.scale != scale:
                arrow_type = pa.decimal128(precision, scale) if precision <= 38 else pa.decimal256(precision, scale)
                object.__setattr__(self, "arrow_type", arrow_type)
        elif pa.types.is_map(self.arrow_type):
            keys_sorted = safe_bool(
                self.metadata.pop("keys_sorted", self.arrow_type.keys_sorted),
                default=False
            )

            if self.arrow_type.keys_sorted != keys_sorted:
                map_type: pa.MapType = self.arrow_type
                arrow_type = pa.map_(
                    map_type.key_field,
                    map_type.item_field,
                    keys_sorted=keys_sorted
                )
                object.__setattr__(self, "arrow_type", arrow_type)
        elif pa.types.is_list(self.arrow_type) or pa.types.is_large_list(self.arrow_type):
            fixed_size = safe_int(
                self.metadata.pop("fixed_size", -1),
                default=-1
            )

            if fixed_size > 0:
                list_type: pa.ListType = self.arrow_type
                arrow_type = pa.list_(list_type.value_field, fixed_size)
                object.__setattr__(self, "arrow_type", arrow_type)

        return self


    @classmethod
    def from_py_hint(
        cls,
        hint: type | Any,
        name: str | None = None,
        nullable: bool | None = None,
        comment: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> "DataField":
        """
        Create a DataField from a Python type hint.

        Args:
            name: Field name
            hint: Python type hint (e.g. int, str, list[int], Annotated[int, "metadata"])
            nullable: Whether the field can be null
            comment: Optional comment
            metadata: Optional metadata for the field

        Returns:
            A DataField instance with the corresponding PyArrow type
        """
        # Handle Annotated[T, ...] - extract the base type and metadata
        origin = get_origin(hint)
        args = get_args(hint)

        # Handle Annotated type
        if origin is Annotated:
            if args and len(args) >= 2:
                base_type = args[0]  # The original type
                field_metadata = safe_dict(args[1:], raise_error=False)
                return cls.from_py_hint(
                    name=name,
                    hint=base_type,
                    nullable=nullable,
                    comment=comment,
                    metadata=merge_dicts(
                        dicts=[field_metadata, metadata]
                    )
                )

            raise TypeError(f"Cannot create DataField from Python type hint {hint}")

        # Handle Optional[T] or Union[T, None]
        if origin is Union:
            if type(None) in args:
                # Extract the non-None type
                type_args = [arg for arg in args if arg is not type(None)]
                if len(type_args) == 0:
                    raise TypeError(f"Cannot create DataField from Python type hint {hint}")
                base_type = type_args[0]  # The original type
                nullable = True if nullable is None else nullable
                return cls.from_py_hint(
                    name=name,
                    hint=base_type,
                    nullable=nullable,
                    comment=comment,
                    metadata=metadata
                )

        arrow_type = None
        children_fields = None
        metadata = safe_dict(
            metadata, default={},
            check_key=safe_str, check_value=safe_str
        )
        # Check if it's a direct mapping from Python type to Arrow type
        if hint in _PYTHON_TO_ARROW_TYPE_MAP:
            arrow_type = _PYTHON_TO_ARROW_TYPE_MAP[hint]
        elif isinstance(hint, pa.DataType):
            # It's already a PyArrow type
            arrow_type = hint
        elif origin is list or hint is list:
            # Handle list[T]
            if args and len(args) > 0:
                item_hint = args[0]
                item_field = cls.from_py_hint(
                    name="item",
                    hint=item_hint,
                    nullable=None,
                    comment=None,
                    metadata=None
                )

            else:
                # Default to list of strings
                item_field = cls.from_arrow_type(name="item", dtype=pa.utf8(), nullable=True)

            arrow_type = pa.list_(item_field.to_arrow_field())
        elif origin is dict or hint is dict:
            # Handle dict[K, V]
            if args and len(args) == 2:
                key_type, value_type = args
                key_field = cls.from_py_hint(
                    name="key",
                    hint=key_type,
                    nullable=False,
                    comment=None,
                    metadata=None
                )
                value_field = cls.from_py_hint(
                    name="value",
                    hint=value_type,
                    nullable=None,
                    comment=None,
                    metadata=None
                )
            else:
                key_field = cls.from_arrow_type(name="key", dtype=pa.utf8(), nullable=False)
                value_field = cls.from_arrow_type(name="value", dtype=pa.utf8(), nullable=True)

            arrow_type = pa.map_(key_field.to_arrow_field(), value_field.to_arrow_field())
        else:
            type_hints = get_type_hints(hint, include_extras=True)
            children_fields = [
                cls.from_py_hint(name=child_name, hint=child_type)
                for child_name, child_type in type_hints.items()
                if not child_name.startswith("_")
            ]
            arrow_type = pa.struct([_.to_arrow_field() for _ in children_fields])

        if arrow_type is None:
            raise TypeError(f"Cannot create DataField from Python type hint {hint}")

        if nullable is None:
            nullable = False

        return cls(
            name=name or hint.__name__,
            arrow_type=arrow_type,
            nullable=nullable,
            comment=comment,
            children=children_fields or None,
            metadata=metadata
        ).refine()

    @classmethod
    def from_arrow_field(cls, field: pa.Field) -> "DataField":
        return cls(
            name=field.name,
            arrow_type=field.type,
            nullable=field.nullable,
            comment=None,
            children=None,
            metadata=field.metadata,
        )

    @classmethod
    def from_arrow_type(cls, name: str, dtype: pa.DataType, nullable: bool) -> "DataField":
        field = pa.field(name=name, type=dtype, nullable=nullable)

        return cls.from_arrow_field(field)

    @classmethod
    def from_spark_field(cls, spark_field: spark_types.StructField) -> "DataField":
        """Create a DataField from a Spark StructField.

        Args:
            spark_field: A PySpark StructField

        Returns:
            A DataField instance with the corresponding PyArrow type

        Raises:
            ImportError: If PySpark is not installed
            TypeError: If the Spark type cannot be converted to a PyArrow type
        """
        if not HAVE_SPARK:
            raise ImportError("PySpark is required for from_spark_field. Install it with 'pip install pyspark'.")

        name = spark_field.name
        nullable = spark_field.nullable
        metadata = getattr(spark_field, "metadata", {})
        arrow_type = spark_to_arrow_type(spark_field.dataType)

        return cls(
            name=name,
            arrow_type=arrow_type,
            nullable=nullable,
            comment=None,
            metadata=metadata,
            children=None
        ).refine()

    @classmethod
    def from_spark_type(cls, name: str, spark_type: spark_types.DataType, nullable: bool, metadata: dict[str, str] = None) -> "DataField":
        """Create a DataField from a Spark DataType.

        Args:
            name: The name of the DataField
            spark_type: A PySpark DataType
            nullable: Whether the DataField is nullable
            metadata: Optional metadata for the DataField

        Returns:
            A DataField instance with the corresponding PyArrow type

        Raises:
            ImportError: If PySpark is not installed
            TypeError: If the Spark type cannot be converted to a PyArrow type
        """
        if not HAVE_SPARK:
            raise ImportError("PySpark is required for from_spark_field. Install it with 'pip install pyspark'.")

        field = spark_types.StructField(name=name, dataType=spark_type, nullable=nullable, metadata=metadata)

        return cls.from_spark_field(field)

    # Properties
    def __eq__(self, other):
        return self.name == other.name and self.arrow_type == other.arrow_type

    def __hash__(self):
        return hash((self.name, self.arrow_type))

    def __repr__(self):
        return f"DataField(name={self.name}, arrow_type={self.arrow_type}, nullable={self.nullable}, metadata={self.metadata}, children={self.children})"

    def is_primitive(self):
        return not self.is_nested()

    def is_decimal(self):
        return pa.types.is_decimal(self.arrow_type)

    def is_timestamp(self):
        return pa.types.is_timestamp(self.arrow_type) or pa.types.is_date64(self.arrow_type)

    def is_date(self):
        return pa.types.is_date32(self.arrow_type)

    def is_time(self):
        return pa.types.is_time(self.arrow_type)

    def is_nested(self):
        return pa.types.is_nested(self.arrow_type)

    def is_list(self):
        return pa.types.is_list(self.arrow_type) or pa.types.is_large_list(self.arrow_type)

    def is_map(self):
        return pa.types.is_map(self.arrow_type)

    def is_struct(self):
        return pa.types.is_struct(self.arrow_type)

    # Transform to
    def to_arrow_field(self) -> pa.Field:
        return pa.field(
            name=self.name, type=self.arrow_type,
            metadata=self.metadata, nullable=self.nullable
        )

    def to_arrow_schema(self) -> pa.Schema:
        return pa.schema([
            field.to_arrow_field()
            for field in self.children or []
        ], metadata=self.metadata)

    def to_spark_field(self) -> spark_types.StructField:
        """Convert this DataField to a Spark StructField.

        Returns:
            A PySpark StructField object with the equivalent type.

        Raises:
            ImportError: If PySpark is not installed
            TypeError: If the arrow_type cannot be converted to a Spark type
        """
        if not HAVE_SPARK:
            raise ImportError("PySpark is required for to_spark_field. Install it with 'pip install pyspark'.")

        spark_type = ARROW_TYPE_TO_SPARK_TYPE.get(self.arrow_type)
        md = self.metadata.copy() or {}

        # Convert PyArrow type to Spark type
        if not spark_type:
            if self.is_primitive():
                if pa.types.is_decimal(self.arrow_type):
                    spark_type = spark_types.DecimalType(self.arrow_type.precision, self.arrow_type.scale)
                elif pa.types.is_timestamp(self.arrow_type):
                    md["timeunit"] = self.arrow_type.unit

                    if self.arrow_type.tz:
                        spark_type = spark_types.TimestampType()
                        md["timezone"] = self.arrow_type.unit
                    else:
                        spark_type = spark_types.TimestampNTZType()
                elif pa.types.is_time(self.arrow_type):
                    md["timeunit"] = self.arrow_type.unit
                    spark_type = spark_types.IntegerType() if pa.types.is_time32(
                        self.arrow_type) else spark_types.LongType()
            else:
                if self.is_struct():
                    spark_type = spark_types.StructType([
                        _.to_spark_field()
                        for _ in self.children
                    ])
                elif self.is_list():
                    item_field: DataField = self.children[0]
                    item_spark = item_field.to_spark_field()
                    spark_type = spark_types.ArrayType(item_spark.dataType, containsNull=item_field.nullable)
                elif self.is_map():
                    key_field: DataField = self.children[0]
                    key_spark = key_field.to_spark_field()

                    value_field: DataField = self.children[1]
                    value_spark = value_field.to_spark_field()

                    spark_type = spark_types.MapType(key_spark.dataType, value_spark.dataType, valueContainsNull=value_field.nullable)

        if spark_type is None:
            raise TypeError(f"Cannot convert {self.arrow_type} to Spark type")

        # Create and return a Spark StructField
        return spark_types.StructField(
            name=self.name,
            dataType=spark_type,
            nullable=self.nullable,
            metadata=md
        )

    def to_polars_field(self) -> pl.Field:
        primitives = {
            pa.utf8(): pl.Utf8,
            pa.binary(): pl.Binary(),
            pa.int8(): pl.Int8(),
            pa.int16(): pl.Int16(),
            pa.int32(): pl.Int32(),
            pa.int64(): pl.Int64(),
            pa.float32(): pl.Float32(),
            pa.float64(): pl.Float64(),
        }
        polars_type = primitives.get(self.arrow_type)

        if not polars_type:
            if self.is_timestamp():
                polars_type = pl.Datetime(
                    time_unit=self.arrow_type.unit, time_zone=self.arrow_type.tz
                )
            elif self.is_date():
                polars_type = pl.Date()
            elif self.is_time():
                polars_type = pl.Time()
            elif self.is_decimal():
                polars_type = pl.Decimal(self.arrow_type.precision, self.arrow_type.scale)
            elif self.is_struct():
                polars_type = pl.Struct(
                    fields=[
                        _.to_polars_field()
                        for _ in self.children
                    ]
                )
            elif self.is_list():
                polars_type = pl.List(inner=self.children[0].to_polars_field().dtype)

        if not polars_type:
            raise ValueError(f"Cannot convert {self} to polars field")

        return pl.Field(name=self.name, dtype=polars_type)

    def cast_spark_column(
        self,
        column_field: spark_types.StructField,
        column: spark_sql.Column,
    ) -> spark_sql.Column:
        """Cast a Spark Column from one DataField type to another."""
        if not HAVE_SPARK:
            raise ImportError(
                "PySpark is required for cast_spark_column. "
                "Install it with 'pip install pyspark'."
            )

        # Target Spark field (what this DataField wants)
        target_field: spark_types.StructField = self.to_spark_field()

        casted = cast_nested_spark_field(
            column,
            source_field=column_field,
            target_field=target_field,
        )

        # Keep the field name consistent with this DataField
        return casted.alias(self.name)

    def cast_spark_dataframe(self, df: spark_sql.DataFrame) -> spark_sql.DataFrame:
        """Cast a Spark DataFrame to match this DataField's schema.

        This method handles field name matching (case-insensitive), type casting,
        and handling missing or extra columns.

        Args:
            df: The source Spark DataFrame to cast

        Returns:
            A new Spark DataFrame with columns cast to the appropriate types

        Raises:
            ImportError: If PySpark is not installed
            ValueError: If required fields are missing from the DataFrame
        """
        if not HAVE_SPARK:
            raise ImportError("PySpark is required for cast_spark_dataframe. Install it with 'pip install pyspark'.")

        if not self.is_struct():
            raise TypeError(f"Cannot cast DataFrame to non-struct type {self}")

        if not self.children:
            return df

        # Get original column names
        original_columns = df.columns
        columns_lower = [col.lower() for col in original_columns]

        # Create mappings between field names in both directions
        # Map target field name -> source column name
        field_to_col_map = {}

        # Process each target field and find matching source column
        for field in self.children:
            field_name_lower = field.name.lower()

            # Look for case-insensitive match
            if field_name_lower in columns_lower:
                idx = columns_lower.index(field_name_lower)
                source_col_name = original_columns[idx]
                field_to_col_map[field.name] = source_col_name

        # Check if we have all required fields
        missing_fields = [field.name for field in self.children
                         if field.name not in field_to_col_map and not field.nullable]

        if missing_fields:
            raise ValueError(f"Required fields missing from DataFrame: {', '.join(missing_fields)}")

        # Build the select expressions for the output DataFrame
        select_exprs = []

        # Process each target field
        for field in self.children:
            target_spark_field = field.to_spark_field()

            if field.name in field_to_col_map:
                # Field exists in source, cast it
                source_col_name = field_to_col_map[field.name]
                source_col = df[source_col_name]
                source_field = df.schema[source_col_name]

                casted_col = cast_nested_spark_field(column=source_col, source_field=source_field, target_field=target_spark_field)
                select_exprs.append(casted_col)
            else:
                # Field is missing in source, add a null column
                null_lit = spark_functions.lit(None).cast(target_spark_field.dataType).alias(field.name)
                select_exprs.append(null_lit)

        # Create and return the new DataFrame with the casted columns
        return df.select(*select_exprs)

    # --------------------------- Polars support ---------------------------
    def cast_polars_column(self, series: "pl.Series") -> "pl.Series":
        """Cast a Polars Series to match this DataField's Arrow type.

        This is a best-effort cast and will raise if Polars is not available.
        For nested types (struct/list) the function will attempt to map to appropriate
        Polars dtypes where possible.
        """
        target_field = self.to_polars_field()
        casted = None

        if casted is None:
            casted = series.cast(target_field.dtype)

        return casted.alias(target_field.name)

    def cast_polars_dataframe(self, df: "pl.DataFrame") -> "pl.DataFrame":
        """Cast a Polars DataFrame to match this DataField's schema (when this DataField is a struct).

        Behavior mirrors cast_spark_dataframe: case-insensitive field matching, required fields check,
        and casting via cast_polars_column for each matched column. Missing nullable fields are filled with nulls.
        """
        if not self.is_struct():
            raise TypeError(f"Cannot cast DataFrame to non-struct type {self}")

        if not self.children:
            return df

        original_columns = df.columns
        columns_lower = [c.lower() for c in original_columns]

        # mapping target field name -> source column name
        field_to_col_map = {}
        for field in self.children:
            fname_lower = field.name.lower()
            if fname_lower in columns_lower:
                idx = columns_lower.index(fname_lower)
                field_to_col_map[field.name] = original_columns[idx]

        missing_fields = [field.name for field in self.children if
                          field.name not in field_to_col_map and not field.nullable]
        if missing_fields:
            raise ValueError(f"Required fields missing from DataFrame: {', '.join(missing_fields)}")

        exprs = []
        for field in self.children:
            if field.name in field_to_col_map:
                src = field_to_col_map[field.name]
                casted = field.cast_polars_column(df[src])
            else:
                target_field = self.to_polars_field()
                casted = pl.lit(None).cast(target_field.dtype).alias(field.name)

            exprs.append(casted)

        # build and return new dataframe; use select to preserve order
        return df.select(*exprs)

    def cast_arrow(self, df):
        pldf = pl.from_arrow(df)

        return self.cast_polars_dataframe(pldf).to_arrow()

    def cast_pandas(self, df):
        pldf = pl.from_pandas(df, include_index=bool(df.index.name))

        return self.cast_polars_dataframe(pldf).to_pandas()

    def cast_dataframe(self, df):
        if isinstance(df, (pa.RecordBatch, pa.Table)):
            return self.cast_arrow(df)
        elif isinstance(df, pandas.DataFrame):
            return self.cast_pandas(df)
        elif isinstance(df, spark_sql.DataFrame):
            return self.cast_spark_dataframe(df)

        raise ValueError(f"Cannot cast {df} with {self}")
