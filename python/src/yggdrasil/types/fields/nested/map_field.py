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
    "MapField",
    "PythonMapField",
    "PandasMapField",
    "PolarsMapField",
    "ArrowMapField",
    "SparkMapField",
]


class MapField(NestedField):
    def __init__(
        self,
        name: str,
        key_field: NestedField,
        value_field: NestedField,
        *,
        nullable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.key_field = key_field
        self.value_field = value_field
        key_arrow = self.key_field.to_arrow().inner.type
        value_arrow = self.value_field.to_arrow().inner.type
        arrow_type = pa.map_(key_arrow, value_arrow)
        super().__init__(name, arrow_type, dict, nullable=nullable, metadata=metadata)

    def to_python(self) -> "PythonMapField":
        return PythonMapField(
            self.name, dict, self.nullable, self._metadata, self.key_field, self.value_field
        )

    def to_arrow(self) -> "ArrowMapField":
        field = pa.field(self.name, self.type, nullable=self.nullable, metadata=self.metadata_bytes)
        return ArrowMapField(field)

    def to_spark(self) -> "SparkMapField":
        if pyspark is None:
            raise ImportError("pyspark is required to convert to Spark fields")

        key_spark = self.key_field.to_spark().type
        value_spark = self.value_field.to_spark().type
        map_type = pyspark.sql.types.MapType(key_spark, value_spark, valueContainsNull=self.value_field.nullable)
        struct_field = pyspark.sql.types.StructField(
            self.name, map_type, self.nullable, metadata=self.metadata or {}
        )
        return SparkMapField(struct_field)

    def to_polars(self) -> "PolarsMapField":
        if polars is None:
            raise ImportError("polars is required to convert to Polars fields")

        map_dtype = getattr(polars, "Map", None)
        if map_dtype is None:
            raise ValueError("Polars Map dtype is not available in this environment")

        dtype = map_dtype(self.key_field.to_polars().type, self.value_field.to_polars().type)
        inner = _polars_field(self.name, dtype, self.nullable, self.metadata)
        return PolarsMapField(inner, self.nullable, self.metadata)

    def to_pandas(self) -> "PandasMapField":
        dtype = _pandas_dtype_fallback()
        return PandasMapField(
            self.name, dtype, self.nullable, self.metadata, self.key_field, self.value_field
        )


class PythonMapField(PythonNestedField):
    def __init__(
        self,
        name: str,
        hint: Any,
        nullable: bool,
        metadata: Optional[Dict[str, Any]],
        key_field: NestedField,
        value_field: NestedField,
    ):
        super().__init__(name, hint, nullable, metadata)
        self.key_field = key_field
        self.value_field = value_field


class PandasMapField(PandasNestedField):
    def __init__(
        self,
        name: str,
        dtype: Any,
        nullable: bool,
        metadata: Optional[Dict[str, Any]],
        key_field: NestedField,
        value_field: NestedField,
    ):
        super().__init__(name, dtype, nullable, metadata)
        self.key_field = key_field
        self.value_field = value_field


class PolarsMapField(PolarsNestedField):
    pass


class ArrowMapField(ArrowNestedField):
    pass


class SparkMapField(SparkNestedField):
    pass
