from __future__ import annotations

from typing import Any, Dict, Optional

import pyarrow as pa

from ...libs import pandas, polars, pyspark
from .base import (
    AbstractScalarField,
    ArrowScalarField,
    PandasScalarField,
    PolarsScalarField,
    PythonScalarField,
    SparkScalarField,
)

__all__ = [
    "ArrowFloatingField",
    "FloatingField",
    "PandasFloatingField",
    "PolarsFloatingField",
    "PythonFloatingField",
    "SparkFloatingField",
]


class FloatingField(AbstractScalarField):
    _ARROW_BY_SIZE = {
        2: pa.float16(),
        4: pa.float32(),
        8: pa.float64(),
    }

    _POLARS_BY_SIZE = {
        2: lambda: polars.Float16 if polars is not None and hasattr(polars, "Float16") else None,
        4: lambda: polars.Float32 if polars is not None else None,
        8: lambda: polars.Float64 if polars is not None else None,
    }

    _SPARK_BY_SIZE = {
        2: None,
        4: pyspark.sql.types.FloatType if pyspark is not None else None,
        8: pyspark.sql.types.DoubleType if pyspark is not None else None,
    }

    _PANDAS_BY_SIZE = {
        2: lambda: "float16",
        4: lambda: "float32",
        8: lambda: "float64",
    }

    def __init__(
        self,
        name: str,
        *,
        bytesize: int = 8,
        nullable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if bytesize not in self._ARROW_BY_SIZE:
            raise ValueError(f"Unsupported floating byte size: {bytesize}")

        self._bytesize = bytesize
        super().__init__(name, self._ARROW_BY_SIZE[bytesize], float, nullable=nullable, metadata=metadata)

    def to_python(self) -> "PythonFloatingField":
        return PythonFloatingField(self.name, float, self.nullable, self._metadata)

    def to_arrow(self) -> "ArrowFloatingField":
        field = pa.field(self.name, self.type, nullable=self.nullable, metadata=self.metadata_bytes)
        return ArrowFloatingField(field)

    def to_spark(self) -> "SparkFloatingField":
        if pyspark is None:
            raise ImportError("pyspark is required to convert to Spark fields")

        spark_dtype_factory = self._SPARK_BY_SIZE.get(self._bytesize)
        if spark_dtype_factory is None:
            raise ValueError(f"Unsupported Spark floating byte size: {self._bytesize}")

        struct_field = pyspark.sql.types.StructField(
            self.name,
            spark_dtype_factory(),
            self.nullable,
            metadata=self.metadata or {},
        )
        return SparkFloatingField(struct_field)

    def to_polars(self) -> "PolarsFloatingField":
        if polars is None:
            raise ImportError("polars is required to convert to Polars fields")

        dtype_factory = self._POLARS_BY_SIZE.get(self._bytesize)
        if dtype_factory is None:
            raise ValueError(f"Unsupported Polars floating byte size: {self._bytesize}")

        dtype = dtype_factory()
        if dtype is None:
            raise ValueError(f"Unsupported Polars floating byte size: {self._bytesize}")

        return PolarsFloatingField(self.name, dtype, self.nullable, self.metadata)

    def to_pandas(self) -> "PandasFloatingField":
        if pandas is None:
            raise ImportError("pandas is required to convert to Pandas fields")

        dtype_factory = self._PANDAS_BY_SIZE.get(self._bytesize)
        if dtype_factory is None:
            raise ValueError(f"Unsupported Pandas floating byte size: {self._bytesize}")

        dtype = dtype_factory()
        return PandasFloatingField(self.name, dtype, self.nullable, self.metadata)


class PythonFloatingField(PythonScalarField):
    def __init__(self, name: str, hint: Any, nullable: bool, metadata: Optional[Dict[str, Any]]):
        super().__init__(name, hint, nullable, metadata)


class ArrowFloatingField(ArrowScalarField):
    pass


class SparkFloatingField(SparkScalarField):
    pass


class PolarsFloatingField(PolarsScalarField):
    pass


class PandasFloatingField(PandasScalarField):
    pass
