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
    "ArrowIntegerField",
    "IntegerField",
    "PandasIntegerField",
    "PolarsIntegerField",
    "PythonIntegerField",
    "SparkIntegerField",
]


class IntegerField(AbstractScalarField):
    _ARROW_BY_SIZE = {
        1: pa.int8(),
        2: pa.int16(),
        4: pa.int32(),
        8: pa.int64(),
    }

    _POLARS_BY_SIZE = {
        1: lambda: polars.Int8 if polars is not None else None,
        2: lambda: polars.Int16 if polars is not None else None,
        4: lambda: polars.Int32 if polars is not None else None,
        8: lambda: polars.Int64 if polars is not None else None,
    }

    _SPARK_BY_SIZE = {
        1: pyspark.sql.types.ByteType if pyspark is not None else None,
        2: pyspark.sql.types.ShortType if pyspark is not None else None,
        4: pyspark.sql.types.IntegerType if pyspark is not None else None,
        8: pyspark.sql.types.LongType if pyspark is not None else None,
    }

    _PANDAS_BY_SIZE = {
        1: lambda: pandas.Int8Dtype() if pandas is not None else None,
        2: lambda: pandas.Int16Dtype() if pandas is not None else None,
        4: lambda: pandas.Int32Dtype() if pandas is not None else None,
        8: lambda: pandas.Int64Dtype() if pandas is not None else None,
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
            raise ValueError(f"Unsupported integer byte size: {bytesize}")

        self._bytesize = bytesize
        super().__init__(name, self._ARROW_BY_SIZE[bytesize], int, nullable=nullable, metadata=metadata)

    def to_python(self) -> "PythonIntegerField":
        return PythonIntegerField(self.name, int, self.nullable, self._metadata)

    def to_arrow(self) -> "ArrowIntegerField":
        field = pa.field(self.name, self.type, nullable=self.nullable, metadata=self.metadata_bytes)
        return ArrowIntegerField(field)

    def to_spark(self) -> "SparkIntegerField":
        if pyspark is None:
            raise ImportError("pyspark is required to convert to Spark fields")

        spark_dtype_factory = self._SPARK_BY_SIZE.get(self._bytesize)
        if spark_dtype_factory is None:
            raise ValueError(f"Unsupported Spark integer byte size: {self._bytesize}")

        struct_field = pyspark.sql.types.StructField(
            self.name,
            spark_dtype_factory(),
            self.nullable,
            metadata=self.metadata or {},
        )
        return SparkIntegerField(struct_field)

    def to_polars(self) -> "PolarsIntegerField":
        if polars is None:
            raise ImportError("polars is required to convert to Polars fields")

        dtype_factory = self._POLARS_BY_SIZE.get(self._bytesize)
        if dtype_factory is None:
            raise ValueError(f"Unsupported Polars integer byte size: {self._bytesize}")

        dtype = dtype_factory()
        if dtype is None:
            raise ValueError(f"Unsupported Polars integer byte size: {self._bytesize}")

        return PolarsIntegerField(self.name, dtype, self.nullable, self.metadata)

    def to_pandas(self) -> "PandasIntegerField":
        if pandas is None:
            raise ImportError("pandas is required to convert to Pandas fields")

        dtype_factory = self._PANDAS_BY_SIZE.get(self._bytesize)
        if dtype_factory is None:
            raise ValueError(f"Unsupported Pandas integer byte size: {self._bytesize}")

        dtype = dtype_factory()
        if dtype is None:
            raise ValueError(f"Unsupported Pandas integer byte size: {self._bytesize}")

        return PandasIntegerField(self.name, dtype, self.nullable, self.metadata)


class PythonIntegerField(PythonScalarField):
    def __init__(self, name: str, hint: Any, nullable: bool, metadata: Optional[Dict[str, Any]]):
        super().__init__(name, hint, nullable, metadata)


class ArrowIntegerField(ArrowScalarField):
    pass


class SparkIntegerField(SparkScalarField):
    pass


class PolarsIntegerField(PolarsScalarField):
    pass


class PandasIntegerField(PandasScalarField):
    pass
