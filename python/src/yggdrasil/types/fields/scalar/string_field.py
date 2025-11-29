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
    "ArrowStringField",
    "PandasStringField",
    "PolarsStringField",
    "PythonStringField",
    "SparkStringField",
    "StringField",
]


class StringField(AbstractScalarField):
    def __init__(
        self,
        name: str,
        *,
        large: bool = False,
        nullable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._large = large
        super().__init__(
            name,
            pa.large_string() if large else pa.string(),
            str,
            nullable=nullable,
            metadata=metadata,
        )

    def to_python(self) -> "PythonStringField":
        return PythonStringField(self.name, str, self.nullable, self._metadata)

    def to_arrow(self) -> "ArrowStringField":
        field = pa.field(self.name, self.type, nullable=self.nullable, metadata=self.metadata_bytes)
        return ArrowStringField(field)

    def to_spark(self) -> "SparkStringField":
        if pyspark is None:
            raise ImportError("pyspark is required to convert to Spark fields")

        struct_field = pyspark.sql.types.StructField(
            self.name,
            pyspark.sql.types.StringType(),
            self.nullable,
            metadata=self.metadata or {},
        )
        return SparkStringField(struct_field)

    def to_polars(self) -> "PolarsStringField":
        if polars is None:
            raise ImportError("polars is required to convert to Polars fields")

        dtype = polars.Utf8
        return PolarsStringField(self.name, dtype, self.nullable, self.metadata)

    def to_pandas(self) -> "PandasStringField":
        if pandas is None:
            raise ImportError("pandas is required to convert to Pandas fields")

        dtype = pandas.StringDtype()
        return PandasStringField(self.name, dtype, self.nullable, self.metadata)


class PythonStringField(PythonScalarField):
    def __init__(self, name: str, hint: Any, nullable: bool, metadata: Optional[Dict[str, Any]]):
        super().__init__(name, hint, nullable, metadata)


class ArrowStringField(ArrowScalarField):
    pass


class SparkStringField(SparkScalarField):
    pass


class PolarsStringField(PolarsScalarField):
    pass


class PandasStringField(PandasScalarField):
    pass
