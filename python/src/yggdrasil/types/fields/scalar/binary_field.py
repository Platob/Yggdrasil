from __future__ import annotations

from typing import Any, Dict, Optional

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
    "ArrowBinaryField",
    "BinaryField",
    "PandasBinaryField",
    "PolarsBinaryField",
    "PythonBinaryField",
    "SparkBinaryField",
]


class BinaryField(AbstractScalarField):
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
            pa.large_binary() if large else pa.binary(),
            bytes,
            nullable=nullable,
            metadata=metadata,
        )

    def to_python(self) -> "PythonBinaryField":
        return PythonBinaryField(self.name, bytes, self.nullable, self.metadata)

    def to_arrow(self) -> "ArrowBinaryField":
        field = pa.field(self.name, self.type, nullable=self.nullable, metadata=self.metadata_bytes)
        return ArrowBinaryField(field)

    def to_spark(self) -> "SparkBinaryField":
        if pyspark is None:
            raise ImportError("pyspark is required to convert to Spark fields")

        struct_field = pyspark.sql.types.StructField(
            self.name,
            pyspark.sql.types.BinaryType(),
            self.nullable,
            metadata=self.metadata or {},
        )
        return SparkBinaryField(struct_field)

    def to_polars(self) -> "PolarsBinaryField":
        if polars is None:
            raise ImportError("polars is required to convert to Polars fields")

        dtype = polars.Binary
        return PolarsBinaryField(self.name, dtype, self.nullable, self.metadata)

    def to_pandas(self) -> "PandasBinaryField":
        if pandas is None:
            raise ImportError("pandas is required to convert to Pandas fields")

        dtype = object
        return PandasBinaryField(self.name, dtype, self.nullable, self.metadata)


class PythonBinaryField(PythonScalarField):
    def __init__(self, name: str, hint: Any, nullable: bool, metadata: Optional[Dict[str, Any]]):
        super().__init__(name, hint, nullable, metadata)


class ArrowBinaryField(ArrowScalarField):
    pass


class SparkBinaryField(SparkScalarField):
    pass


class PolarsBinaryField(PolarsScalarField):
    pass


class PandasBinaryField(PandasScalarField):
    pass
