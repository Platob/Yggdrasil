from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from ..id import DataTypeId
from ..support import get_polars, get_spark_sql
from .base import PrimitiveType

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst


__all__ = ["NullType"]


@dataclass(frozen=True, repr=False)
class NullType(PrimitiveType):
    """All-null column. Identity under merge; no-op on cast."""

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.NULL

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        return f"{pad}null"

    # ------------------------------------------------------------------
    # Engine probes / constructors
    # ------------------------------------------------------------------

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_null(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "NullType":
        if not pa.types.is_null(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")
        return cls()

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return dtype == pl.Null

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "NullType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        return cls()

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = get_spark_sql()
        return isinstance(dtype, spark.types.NullType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "NullType":
        if not cls.handles_spark_type(dtype):
            raise TypeError(f"Unsupported Spark data type: {dtype!r}")
        return cls()

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.NULL)

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "NullType":
        try:
            return cls.instance()
        except Exception as e:
            if default is ...:
                raise e
            return default

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    def to_arrow(self) -> pa.DataType:
        return pa.null()

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.Null

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        return spark.types.NullType()

    def to_databricks_ddl(self) -> str:
        return "VOID"

    # ------------------------------------------------------------------
    # Defaults / conversion
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        return None

    def _convert_pyobj(self, value: Any, safe: bool = False) -> None:
        return None
