from __future__ import annotations

import decimal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from ._helpers import _BOOL_FALSE, _BOOL_TRUE, _coerce_str
from .base import PrimitiveType
from ..id import DataTypeId
from ..support import get_polars, get_spark_sql

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst


__all__ = ["BooleanType"]


@dataclass(frozen=True, repr=False)
class BooleanType(PrimitiveType):

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.BOOL

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        return f"{pad}bool"

    # ------------------------------------------------------------------
    # Engine probes / constructors
    # ------------------------------------------------------------------

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_boolean(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "BooleanType":
        if not pa.types.is_boolean(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")
        return cls.instance()

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return dtype == pl.Boolean

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "BooleanType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        return cls.instance()

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = get_spark_sql()
        return isinstance(dtype, spark.types.BooleanType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "BooleanType":
        if not cls.handles_spark_type(dtype):
            raise TypeError(f"Unsupported Spark data type: {dtype!r}")
        return cls.instance()

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.BOOL, "BOOLEAN")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BooleanType":
        return cls(byte_size=value.get("byte_size", 1))

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    def to_arrow(self) -> pa.DataType:
        return pa.bool_()

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.Boolean

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        return spark.types.BooleanType()

    def to_databricks_ddl(self) -> str:
        return "BOOLEAN"

    # ------------------------------------------------------------------
    # Defaults / conversion
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else False

    def _convert_pyobj(self, value: Any, safe: bool = False) -> bool | None:
        token = _coerce_str(value)
        if token is not None:
            normalized = token.strip().lower()
            if normalized in _BOOL_TRUE:
                return True
            if normalized in _BOOL_FALSE:
                return False
            if safe:
                raise ValueError(
                    f"Cannot parse bool from {value!r}. "
                    f"Expected one of true/false/yes/no/on/off/1/0."
                )
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, decimal.Decimal):
            return bool(value)
        if safe:
            raise ValueError(
                f"Cannot convert {type(value).__name__} to bool "
                f"for {type(self).__name__}: {value!r}."
            )
        return None
