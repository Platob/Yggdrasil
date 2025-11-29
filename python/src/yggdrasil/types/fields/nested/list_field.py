from __future__ import annotations

from typing import Any, Dict, Optional

import pyarrow as pa

from ...libs import polars, pyspark
from .base import (
    ArrowNestedField,
    NestedField,
    PandasNestedField,
    PolarsNestedField,
    PythonNestedField,
    SparkNestedField,
    _pandas_dtype_fallback,
    _polars_field,
)

__all__ = [
    "ListField",
    "PythonListField",
    "PandasListField",
    "PolarsListField",
    "ArrowListField",
    "SparkListField",
]


class ListField(NestedField):
    def __init__(
        self,
        name: str,
        value_field: NestedField,
        *,
        nullable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.value_field = value_field
        arrow_type = pa.list_(self.value_field.to_arrow().inner.type)
        super().__init__(name, arrow_type, list, nullable=nullable, metadata=metadata)

    def to_python(self) -> "PythonListField":
        return PythonListField(self.name, list, self.nullable, self._metadata, self.value_field)

    def to_arrow(self) -> "ArrowListField":
        field = pa.field(self.name, self.type, nullable=self.nullable, metadata=self.metadata_bytes)
        return ArrowListField(field)

    def to_spark(self) -> "SparkListField":
        if pyspark is None:
            raise ImportError("pyspark is required to convert to Spark fields")

        value_spark = self.value_field.to_spark().type
        array_type = pyspark.sql.types.ArrayType(value_spark, containsNull=self.value_field.nullable)
        struct_field = pyspark.sql.types.StructField(
            self.name, array_type, self.nullable, metadata=self.metadata or {}
        )
        return SparkListField(struct_field)

    def to_polars(self) -> "PolarsListField":
        if polars is None:
            raise ImportError("polars is required to convert to Polars fields")

        value_polars = self.value_field.to_polars().type
        dtype = polars.List(value_polars)
        inner = _polars_field(self.name, dtype, self.nullable, self.metadata)
        return PolarsListField(inner, self.nullable, self.metadata)

    def to_pandas(self) -> "PandasListField":
        dtype = _pandas_dtype_fallback()
        return PandasListField(self.name, dtype, self.nullable, self.metadata, self.value_field)


class PythonListField(PythonNestedField):
    def __init__(
        self,
        name: str,
        hint: Any,
        nullable: bool,
        metadata: Optional[Dict[str, Any]],
        value_field: NestedField,
    ):
        super().__init__(name, hint, nullable, metadata)
        self.value_field = value_field


class PandasListField(PandasNestedField):
    def __init__(
        self,
        name: str,
        dtype: Any,
        nullable: bool,
        metadata: Optional[Dict[str, Any]],
        value_field: NestedField,
    ):
        super().__init__(name, dtype, nullable, metadata)
        self.value_field = value_field


class PolarsListField(PolarsNestedField):
    pass


class ArrowListField(ArrowNestedField):
    pass


class SparkListField(SparkNestedField):
    pass
