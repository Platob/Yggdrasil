from __future__ import annotations

from typing import Any, Dict, Optional

from typing import Any, Dict, Optional

import pyarrow as pa

from ...libs import pandas, polars
from .base import AbstractScalarField, ArrowScalarField, PandasScalarField, PolarsScalarField, PythonScalarField

__all__ = [
    "ArrowTimeField",
    "PandasTimeField",
    "PolarsTimeField",
    "PythonTimeField",
    "TimeField",
]


class TimeField(AbstractScalarField):
    def __init__(self, name: str, *, unit: str = "ns", nullable: bool = True, metadata: Optional[Dict[str, Any]] = None):
        self._unit = unit
        super().__init__(name, pa.time64(unit), "time", nullable=nullable, metadata=metadata)

    def to_python(self) -> "PythonTimeField":
        return PythonTimeField(self.name, "time", self.nullable, self.metadata)

    def to_arrow(self) -> "ArrowTimeField":
        field = pa.field(self.name, self.type, nullable=self.nullable, metadata=self.metadata_bytes)
        return ArrowTimeField(field)

    def to_spark(self):
        raise NotImplementedError("Spark conversion is not supported for time fields")

    def to_polars(self) -> "PolarsTimeField":
        if polars is None:
            raise ImportError("polars is required to convert to Polars fields")

        dtype = polars.Time
        return PolarsTimeField(self.name, dtype, self.nullable, self.metadata)

    def to_pandas(self) -> "PandasTimeField":
        if pandas is None:
            raise ImportError("pandas is required to convert to Pandas fields")

        dtype = "timedelta64[ns]" if self._unit != "s" else "timedelta64[s]"
        return PandasTimeField(self.name, dtype, self.nullable, self.metadata)


class PythonTimeField(PythonScalarField):
    def __init__(self, name: str, hint: str, nullable: bool, metadata: Optional[Dict[str, Any]]):
        super().__init__(name, hint, nullable, metadata)


class ArrowTimeField(ArrowScalarField):
    pass


class PolarsTimeField(PolarsScalarField):
    pass


class PandasTimeField(PandasScalarField):
    pass
