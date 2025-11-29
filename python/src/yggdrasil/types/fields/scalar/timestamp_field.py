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
    "ArrowTimestampField",
    "PythonTimestampField",
    "PandasTimestampField",
    "PolarsTimestampField",
    "SparkTimestampField",
    "TimestampField",
]


class TimestampField(AbstractScalarField):
    def __init__(
        self,
        name: str,
        *,
        unit: str = "ns",
        tz: Optional[str] = None,
        nullable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._unit = unit
        self._tz = tz
        super().__init__(name, pa.timestamp(unit, tz=tz), "datetime", nullable=nullable, metadata=metadata)

    def to_python(self) -> "PythonTimestampField":
        return PythonTimestampField(self.name, "datetime", self.nullable, self.metadata)

    def to_arrow(self) -> "ArrowTimestampField":
        field = pa.field(self.name, self.type, nullable=self.nullable, metadata=self.metadata_bytes)
        return ArrowTimestampField(field)

    def to_spark(self) -> "SparkTimestampField":
        if pyspark is None:
            raise ImportError("pyspark is required to convert to Spark fields")

        struct_field = pyspark.sql.types.StructField(
            self.name,
            pyspark.sql.types.TimestampType(),
            self.nullable,
            metadata=self.metadata or {},
        )
        return SparkTimestampField(struct_field)

    def to_polars(self) -> "PolarsTimestampField":
        if polars is None:
            raise ImportError("polars is required to convert to Polars fields")

        dtype = polars.Datetime(self._unit, time_zone=self._tz)
        return PolarsTimestampField(self.name, dtype, self.nullable, self.metadata)

    def to_pandas(self) -> "PandasTimestampField":
        if pandas is None:
            raise ImportError("pandas is required to convert to Pandas fields")

        arrow_dtype = pa.timestamp(self._unit, tz=self._tz)
        dtype = pandas.ArrowDtype(arrow_dtype)
        return PandasTimestampField(self.name, dtype, self.nullable, self.metadata)


class PythonTimestampField(PythonScalarField):
    def __init__(self, name: str, hint: str, nullable: bool, metadata: Optional[Dict[str, Any]]):
        super().__init__(name, hint, nullable, metadata)


class ArrowTimestampField(ArrowScalarField):
    pass


class SparkTimestampField(SparkScalarField):
    pass


class PolarsTimestampField(PolarsScalarField):
    pass


class PandasTimestampField(PandasScalarField):
    pass
