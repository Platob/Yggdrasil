"""Arrow DataClass implementation for Yggdrasil.

This module provides a decorator for creating dataclasses with PyArrow schema inference.
"""

from __future__ import annotations

import datetime as dt
import decimal as dec
import sys
from dataclasses import dataclass
from typing import TypeVar, Any, get_type_hints, get_origin, get_args, Union, Iterable

# Annotated is available in Python 3.9+
if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    try:
        from typing_extensions import Annotated
    except ImportError:
        # Define a dummy Annotated for backward compatibility
        class _AnnotatedAlias:
            def __class_getitem__(cls, params):
                if not isinstance(params, tuple):
                    params = (params,)
                return params[0]  # Return the original type

        Annotated = _AnnotatedAlias()

try:
    import pyarrow as pa
except ImportError:
    raise ImportError("PyArrow is required for arrow_dataclass. Install it with 'pip install pyarrow'.")

from .spark_utils import HAVE_SPARK, cast_nested_spark_field, spark_to_arrow_type, spark_types, spark_sql, spark_functions

__all__ = [
    "DataField",
    "Annotated",
    "merge_dicts",
    "safe_str",
    "annotation_args_to_metadata",
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


def annotation_args_to_metadata(args: Iterable) -> dict[str, str]:
    md = {}
    for arg in args:
        if isinstance(arg, tuple):
            if len(arg) == 2:
                md[safe_str(arg[0])] = safe_str(arg[1])
        elif isinstance(arg, dict):
            for k, v in arg.items():
                md[safe_str(k)] = safe_str(v)

    return md

def merge_dicts(*dicts: dict) -> dict:
    merged = {}

    for d in dicts:
        if d:
            merged.update(d)

    return merged


def safe_str(obj: Any) -> str:
    if isinstance(obj, str):
        return obj

    if isinstance(obj, bytes):
        return obj.decode("utf-8")

    return str(obj)


def safe_metadata_str(md: dict) -> dict[str, str]:
    if not md:
        return {}

    return {
        safe_str(k): safe_str(v)
        for k, v in md.items()
        if k and v
    }


@dataclass(frozen=True)
class DataField:
    name: str
    arrow_type: pa.DataType
    nullable: bool
    metadata: dict[str, str] | None
    children: list[DataField] | None

    @classmethod
    def from_py_hint(
        cls,
        name: str,
        hint: type | Any,
        nullable: bool | None = None,
        metadata: dict[str, str] | None = None
    ) -> "DataField":
        """
        Create a DataField from a Python type hint.

        Args:
            name: Field name
            hint: Python type hint (e.g. int, str, list[int], Annotated[int, "metadata"])
            nullable: Whether the field can be null
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
                field_metadata = annotation_args_to_metadata(args[1:])
                return cls.from_py_hint(name=name, hint=base_type, nullable=nullable, metadata=merge_dicts(field_metadata, metadata))

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
                return cls.from_py_hint(name=name, hint=base_type, nullable=nullable, metadata=metadata)

        arrow_type = None
        children_fields = None

        # Check if it's a direct mapping from Python type to Arrow type
        if hint in _PYTHON_TO_ARROW_TYPE_MAP:
            arrow_type = _PYTHON_TO_ARROW_TYPE_MAP[hint]
        elif isinstance(hint, pa.DataType):
            # It's already a PyArrow type
            arrow_type = hint
        elif origin is list:
            fixed_size = int(metadata.pop("fixed_size", 0))
            fixed_size = fixed_size if fixed_size > 0 else None

            # Handle list[T]
            if args and len(args) > 0:
                item_hint = args[0]
                item_field = cls.from_py_hint(name="item", hint=item_hint)

            else:
                # Default to list of strings
                item_field = cls.from_arrow_type(name="item", dtype=pa.utf8(), nullable=True)

            arrow_type = pa.list_(item_field.to_arrow_field(), fixed_size)
            children_fields = [item_field]
        elif origin is dict:
            keys_sorted = metadata.pop("keys_sorted", "false")
            keys_sorted = str(keys_sorted).lower().startswith("t") if keys_sorted else False

            # Handle dict[K, V]
            if args and len(args) == 2:
                key_type, value_type = args
                key_field = cls.from_py_hint(name="key", hint=key_type)
                value_field = cls.from_py_hint(name="value", hint=value_type)
            else:
                key_field = cls.from_arrow_type(name="key", dtype=pa.utf8(), nullable=True)
                value_field = cls.from_arrow_type(name="value", dtype=pa.utf8(), nullable=True)

            arrow_type = pa.map_(key_field.to_arrow_field(), value_field.to_arrow_field(), keys_sorted=keys_sorted)
        else:
            type_hints = get_type_hints(hint)
            children_fields = [
                cls.from_py_hint(name=child_name, hint=child_type)
                for child_name, child_type in type_hints.items()
                if not child_name.startswith("_")
            ]
            arrow_type = pa.struct([_.to_arrow_field() for _ in children_fields])

        if arrow_type is None:
            raise TypeError(f"Cannot create DataField from Python type hint {hint}")

        metadata = safe_metadata_str(metadata)
        time_unit = metadata.pop("unit", metadata.pop("timeunit", None))
        time_zone = metadata.pop("tz", metadata.pop("timezone", None))

        if time_unit:
            if pa.types.is_timestamp(arrow_type):
                if arrow_type.unit != time_unit:
                    arrow_type = pa.timestamp(time_unit)
            elif pa.types.is_timestamp(arrow_type):
                if arrow_type.unit != time_unit:
                    arrow_type = pa.time32(time_unit) if time_unit in {"s", "ms"} else pa.time64(time_unit)

        if time_zone:
            if pa.types.is_timestamp(arrow_type):
                if arrow_type.tz != time_zone:
                    arrow_type = pa.timestamp(arrow_type.unit, tz=time_zone)

        if pa.types.is_decimal(arrow_type):
            precision = int(metadata.pop("precision", arrow_type.precision))
            scale = int(metadata.pop("scale", arrow_type.scale))

            if arrow_type.precision != precision or arrow_type.scale != scale:
                arrow_type = pa.decimal128(precision, scale) if precision <= 38 else pa.decimal256(precision, scale)

        if nullable is None:
            nullable = False

        return cls(name=name, arrow_type=arrow_type, nullable=nullable, metadata=safe_metadata_str(metadata), children=children_fields or None)

    @classmethod
    def from_arrow_field(cls, field: pa.Field) -> "DataField":
        return cls(
            name=field.name,
            arrow_type=field.type,
            nullable=field.nullable,
        )

    @classmethod
    def from_arrow_type(cls, name: str, dtype: pa.DataType, nullable: bool) -> "DataField":
        field = pa.field(name=name, type=dtype, nullable=nullable)

        return cls.from_arrow_field(field)

    @classmethod
    def from_spark_field(cls, spark_field) -> "DataField":
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
        metadata = spark_field.metadata if hasattr(spark_field, "metadata") else None
        arrow_type = spark_to_arrow_type(spark_field.dataType)

        return cls(name=name, arrow_type=arrow_type, nullable=nullable,
                  metadata=safe_metadata_str(metadata) if metadata else None,
                  children=None)

    # Properties
    def __eq__(self, other):
        return self.name == other.name and self.arrow_type == other.arrow_type

    def __hash__(self):
        return hash((self.name, self.arrow_type))

    def __repr__(self):
        return f"DataField(name={self.name}, arrow_type={self.arrow_type}, nullable={self.nullable}, metadata={self.metadata}, children={self.children})"

    def is_primitive(self):
        return pa.types.is_primitive(self.arrow_type)

    def is_nested(self):
        return pa.types.is_nested(self.arrow_type)

    def is_list(self):
        return pa.types.is_list(self.children) or pa.types.is_large_list(self.children)

    def is_map(self):
        return pa.types.is_map(self)

    def is_struct(self):
        return pa.types.is_struct(self)

    # Transform to
    def to_arrow_field(self) -> pa.Field:
        return pa.field(name=self.name, type=self.arrow_type, metadata=self.metadata, nullable=self.nullable)

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

        spark_type = None
        primitives = {
            pa.utf8(): spark_types.StringType(),
            pa.binary(): spark_types.BinaryType(),
            pa.int8(): spark_types.IntegerType(),
            pa.int16(): spark_types.IntegerType(),
            pa.int32(): spark_types.IntegerType(),
            pa.int64(): spark_types.LongType(),
            pa.float32(): spark_types.FloatType(),
            pa.float64(): spark_types.DoubleType(),
        }
        md = self.metadata.copy() or {}

        # Convert PyArrow type to Spark type
        if self.is_primitive():
            spark_type = primitives[self.arrow_type]

            if not spark_type:
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
                    spark_type = spark_types.IntegerType() if pa.types.is_time32(self.arrow_type) else spark_types.LongType()
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