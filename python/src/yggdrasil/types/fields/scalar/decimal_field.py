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
    "ArrowDecimalField",
    "DecimalField",
    "PandasDecimalField",
    "PolarsDecimalField",
    "PythonDecimalField",
    "SparkDecimalField",
]


class DecimalField(AbstractScalarField):
    def __init__(
        self,
        name: str,
        *,
        precision: int = 38,
        scale: int = 18,
        nullable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._precision = precision
        self._scale = scale
        super().__init__(name, pa.decimal128(precision, scale), float, nullable=nullable, metadata=metadata)

    def to_python(self) -> "PythonDecimalField":
        return PythonDecimalField(self.name, float, self.nullable, self.metadata)

    def to_arrow(self) -> "ArrowDecimalField":
        field = pa.field(self.name, self.type, nullable=self.nullable, metadata=self.metadata_bytes)
        return ArrowDecimalField(field)

    def to_spark(self) -> "SparkDecimalField":
        if pyspark is None:
            raise ImportError("pyspark is required to convert to Spark fields")

        struct_field = pyspark.sql.types.StructField(
            self.name,
            pyspark.sql.types.DecimalType(self._precision, self._scale),
            self.nullable,
            metadata=self.metadata or {},
        )
        return SparkDecimalField(struct_field)

    def to_polars(self) -> "PolarsDecimalField":
        if polars is None:
            raise ImportError("polars is required to convert to Polars fields")

        dtype = polars.Decimal(self.precision, self.scale)
        return PolarsDecimalField(self.name, dtype, self.nullable, self.metadata)

    def to_pandas(self) -> "PandasDecimalField":
        if pandas is None:
            raise ImportError("pandas is required to convert to Pandas fields")

        dtype = pandas.ArrowDtype(pa.decimal128(self.precision, self.scale))
        return PandasDecimalField(self.name, dtype, self.nullable, self.metadata)


class PythonDecimalField(PythonScalarField):
    def __init__(self, name: str, hint: Any, nullable: bool, metadata: Optional[Dict[str, Any]]):
        super().__init__(name, hint, nullable, metadata)


class ArrowDecimalField(ArrowScalarField):
    pass


class SparkDecimalField(SparkScalarField):
    pass


class PolarsDecimalField(PolarsScalarField):
    pass


class PandasDecimalField(PandasScalarField):
    pass
