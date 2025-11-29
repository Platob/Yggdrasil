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
    "ArrowDateField",
    "DateField",
    "PandasDateField",
    "PolarsDateField",
    "PythonDateField",
    "SparkDateField",
]


class DateField(AbstractScalarField):
    def __init__(self, name: str, *, nullable: bool = True, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(name, pa.date32(), "date", nullable=nullable, metadata=metadata)

    def to_python(self) -> "PythonDateField":
        return PythonDateField(self.name, "date", self.nullable, self._metadata)

    def to_arrow(self) -> "ArrowDateField":
        field = pa.field(self.name, self.type, nullable=self.nullable, metadata=self.metadata_bytes)
        return ArrowDateField(field)

    def to_spark(self) -> "SparkDateField":
        if pyspark is None:
            raise ImportError("pyspark is required to convert to Spark fields")

        struct_field = pyspark.sql.types.StructField(
            self.name,
            pyspark.sql.types.DateType(),
            self.nullable,
            metadata=self.metadata or {},
        )
        return SparkDateField(struct_field)

    def to_polars(self) -> "PolarsDateField":
        if polars is None:
            raise ImportError("polars is required to convert to Polars fields")

        dtype = polars.Date
        return PolarsDateField(self.name, dtype, self.nullable, self.metadata)

    def to_pandas(self) -> "PandasDateField":
        if pandas is None:
            raise ImportError("pandas is required to convert to Pandas fields")

        dtype = "datetime64[ns]"
        return PandasDateField(self.name, dtype, self.nullable, self.metadata)


class PythonDateField(PythonScalarField):
    def __init__(self, name: str, hint: Any, nullable: bool, metadata: Optional[Dict[str, Any]]):
        super().__init__(name, hint, nullable, metadata)


class ArrowDateField(ArrowScalarField):
    pass


class SparkDateField(SparkScalarField):
    pass


class PolarsDateField(PolarsScalarField):
    pass


class PandasDateField(PandasScalarField):
    pass
