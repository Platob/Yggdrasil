from __future__ import annotations

import datetime as dt
import decimal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from ._helpers import _bytes_to_str
from .base import PrimitiveType
from ..id import DataTypeId
from yggdrasil.lazy_imports import polars_module, spark_sql_module

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst

__all__ = ["StringType"]


@dataclass(frozen=True, repr=False)
class StringType(PrimitiveType):
    large: bool = False
    view: bool = False
    fixed_size: bool = False

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        s = "large_string" if self.large else "string"
        s = s + "_view" if self.view else s
        if self.fixed_size:
            s = s + "_fixed"
        return f"{pad}{s}"

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.STRING

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return (
            pa.types.is_string(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_string_view(dtype)
        )

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "StringType":
        if pa.types.is_string(dtype):
            return cls(large=False, view=False)
        if pa.types.is_large_string(dtype):
            return cls(large=True, view=False)
        if pa.types.is_string_view(dtype):
            return cls(large=False, view=True)
        raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = polars_module()
        return dtype == pl.String

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "StringType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        return cls()

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = spark_sql_module()
        return isinstance(dtype, spark.types.StringType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "StringType":
        if not cls.handles_spark_type(dtype):
            raise TypeError(f"Unsupported Spark data type: {dtype!r}")
        return cls()

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.STRING)

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "StringType":
        try:
            return cls(
                large=bool(value.get("large", False)),
                view=bool(value.get("view", False)),
                fixed_size=bool(value.get("fixed_size", False)),
                byte_size=value.get("byte_size"),
            )
        except Exception as e:
            if default is ...:
                raise e
            return default

    # ==================================================================
    # Exporters
    # ==================================================================

    def _default_pyhint(self) -> Any:
        # All string flavours (utf8 / large_utf8 / view) round-trip
        # as ``str`` in Python. Fixed-width / Unicode-narrower subtypes
        # rely on their ``_pyhint_cache`` stamp to preserve.
        return str

    def to_arrow(self) -> pa.DataType:
        if self.large:
            return pa.large_string()
        if self.view:
            return pa.string_view()
        return pa.string()

    def to_polars(self) -> "polars.DataType":
        pl = polars_module()
        return pl.String

    def to_spark(self) -> Any:
        spark = spark_sql_module()
        return spark.types.StringType()

    def to_spark_name(self) -> str:
        return "STRING"

    def as_spark(self) -> "StringType":
        # Spark Connect's Arrow gRPC transport rejects ``large_string``
        # and ``string_view`` with ``[UNSUPPORTED_ARROWTYPE]``; collapse
        # the storage flavor to plain ``pa.string()`` so the table sent
        # over the wire lands on the only variant Spark accepts.
        if not (self.large or self.view):
            return self
        return StringType(
            large=False,
            view=False,
            fixed_size=self.fixed_size,
            byte_size=self.byte_size,
        )

    # ==================================================================
    # Defaults / conversion
    # ==================================================================

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None

        if self.byte_size is not None and self.byte_size > 0:
            return "X" * self.byte_size
        return ""

    def _convert_pyobj(self, value: Any, safe: bool = False) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            decoded = _bytes_to_str(value)
            if decoded is None:
                if safe:
                    raise ValueError(
                        f"Cannot decode bytes as UTF-8 for {type(self).__name__}: {value!r}"
                    )
                return None
            return decoded
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float, decimal.Decimal)):
            return str(value)
        if isinstance(value, (dt.date, dt.time, dt.datetime, dt.timedelta)):
            return value.isoformat()
        try:
            return str(value)
        except Exception:
            if safe:
                raise ValueError(
                    f"Cannot convert value of type {type(value).__name__} to str "
                    f"for {type(self).__name__}: {value!r}"
                )
            return None
