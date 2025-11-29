from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

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
    "StructField",
    "PythonStructField",
    "PandasStructField",
    "PolarsStructField",
    "ArrowStructField",
    "SparkStructField",
]


class StructField(NestedField):
    def __init__(
        self,
        name: str,
        fields: Iterable["NestedField"],
        *,
        nullable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.fields: List[NestedField] = list(fields)
        arrow_type = pa.struct([child.to_arrow().inner for child in self.fields])
        super().__init__(name, arrow_type, dict, nullable=nullable, metadata=metadata)

    def to_python(self) -> "PythonStructField":
        return PythonStructField(self.name, dict, self.nullable, self._metadata, self.fields)

    def to_arrow(self) -> "ArrowStructField":
        field = pa.field(self.name, self.type, nullable=self.nullable, metadata=self.metadata_bytes)
        return ArrowStructField(field)

    def to_spark(self) -> "SparkStructField":
        if pyspark is None:
            raise ImportError("pyspark is required to convert to Spark fields")

        spark_children = [child.to_spark().inner for child in self.fields]
        spark_struct = pyspark.sql.types.StructType(spark_children)
        struct_field = pyspark.sql.types.StructField(
            self.name, spark_struct, self.nullable, metadata=self.metadata or {}
        )
        return SparkStructField(struct_field)

    def to_polars(self) -> "PolarsStructField":
        if polars is None:
            raise ImportError("polars is required to convert to Polars fields")

        polars_fields = [child.to_polars().inner for child in self.fields]
        dtype = polars.Struct(polars_fields) if hasattr(polars, "Struct") else polars.Struct
        inner = _polars_field(self.name, dtype, self.nullable, self.metadata)
        return PolarsStructField(inner, self.nullable, self.metadata)

    def to_pandas(self) -> "PandasStructField":
        dtype = _pandas_dtype_fallback()
        return PandasStructField(self.name, dtype, self.nullable, self.metadata, self.fields)


class PythonStructField(PythonNestedField):
    def __init__(
        self,
        name: str,
        hint: Any,
        nullable: bool,
        metadata: Optional[Dict[str, Any]],
        fields: List[NestedField],
    ):
        super().__init__(name, hint, nullable, metadata)
        self.fields = fields


class PandasStructField(PandasNestedField):
    def __init__(
        self,
        name: str,
        dtype: Any,
        nullable: bool,
        metadata: Optional[Dict[str, Any]],
        fields: List[NestedField],
    ):
        super().__init__(name, dtype, nullable, metadata)
        self.fields = fields


class PolarsStructField(PolarsNestedField):
    pass


class ArrowStructField(ArrowNestedField):
    pass


class SparkStructField(SparkNestedField):
    pass
