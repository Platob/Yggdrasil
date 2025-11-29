from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional

import pyarrow as pa

from ...libs import pandas, polars, pyspark
from ..abstract_field import (
    AbstractField,
    ArrowField,
    PandasField,
    PolarsField,
    PythonField,
    SparkField,
)
from ..scalar.base import metadata_bytes, metadata_str

__all__ = [
    "NestedField",
    "PythonNestedField",
    "PandasNestedField",
    "PolarsNestedField",
    "ArrowNestedField",
    "SparkNestedField",
]


class NestedField(AbstractField, ABC):
    def __init__(
        self,
        name: str,
        arrow_dtype: pa.DataType,
        python_hint: Any,
        *,
        nullable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._name = name
        self._arrow_dtype = arrow_dtype
        self._python_hint = python_hint
        self._nullable = nullable
        self._metadata = metadata

    @classmethod
    def _parse(cls, dtype: Any):
        raise NotImplementedError(f"Parsing is not supported for {cls.__name__}")

    @classmethod
    def validate_type(cls, dtype: Any):
        return isinstance(dtype, pa.DataType)

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> pa.DataType:
        return self._arrow_dtype

    @property
    def nullable(self) -> bool:
        return self._nullable

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return metadata_str(self._metadata)

    @property
    def metadata_bytes(self) -> Optional[Dict[bytes, bytes]]:
        return metadata_bytes(self._metadata)

    @property
    def metadata_str(self) -> Optional[Dict[str, str]]:
        return metadata_str(self._metadata)


class PythonNestedField(PythonField, ABC):
    def to_python(self) -> "PythonNestedField":
        return self


class PandasNestedField(PandasField, ABC):
    def to_pandas(self) -> "PandasNestedField":
        return self


class PolarsNestedField(PolarsField, ABC):
    def to_polars(self) -> "PolarsNestedField":
        return self


class ArrowNestedField(ArrowField, ABC):
    def to_arrow(self) -> "ArrowNestedField":
        return self


class SparkNestedField(SparkField, ABC):
    def to_spark(self) -> "SparkNestedField":
        return self


def _polars_field(name: str, dtype: Any, nullable: bool, metadata: Optional[Dict[str, Any]]):
    if polars is None:
        raise ImportError("polars is required to build Polars fields")

    if hasattr(polars, "Field"):
        return polars.Field(name, dtype, nullable=nullable, metadata=metadata)

    # Fallback for older polars versions that accept a tuple schema definition
    return (name, dtype)


def _pandas_dtype_fallback() -> Any:
    if pandas is None:
        raise ImportError("pandas is required to build Pandas fields")
    return object
