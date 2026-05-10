from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from .base import PrimitiveType
from ..id import DataTypeId
from ..support import get_polars, get_spark_sql

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst


__all__ = ["BinaryType"]


@dataclass(frozen=True, repr=False)
class BinaryType(PrimitiveType):
    large: bool = False
    view: bool = False

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.BINARY

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        s = "large_binary" if self.large else "binary"
        s = s + "_view" if self.view else s
        return f"{pad}{s}"

    # ------------------------------------------------------------------
    # Engine probes / constructors
    # ------------------------------------------------------------------

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return (
            pa.types.is_binary(dtype)
            or pa.types.is_large_binary(dtype)
            or pa.types.is_binary_view(dtype)
            or pa.types.is_fixed_size_binary(dtype)
        )

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "BinaryType":
        if pa.types.is_binary(dtype):
            return cls(large=False, view=False, byte_size=None)
        if pa.types.is_large_binary(dtype):
            return cls(large=True, view=False, byte_size=None)
        if pa.types.is_binary_view(dtype):
            return cls(large=False, view=True, byte_size=None)
        if pa.types.is_fixed_size_binary(dtype):
            return cls(large=False, view=False, byte_size=dtype.byte_width)
        raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return dtype == pl.Binary

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "BinaryType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        return cls()

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = get_spark_sql()
        return isinstance(dtype, spark.types.BinaryType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "BinaryType":
        if not cls.handles_spark_type(dtype):
            raise TypeError(f"Unsupported Spark data type: {dtype!r}")
        return cls()

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.BINARY)

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "BinaryType":
        try:
            return cls(
                byte_size=value.get("byte_size"),
                large=bool(value.get("large", False)),
                view=bool(value.get("view", False)),
            )
        except Exception as e:
            if default is ...:
                raise e
            return default

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    def to_arrow(self) -> pa.DataType:
        if self.byte_size is not None:
            return pa.binary(self.byte_size)
        if self.large:
            return pa.large_binary()
        if self.view:
            return pa.binary_view()
        return pa.binary()

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.Binary

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        return spark.types.BinaryType()

    def to_spark_name(self) -> str:
        return "BINARY"

    # ------------------------------------------------------------------
    # Defaults / conversion
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None

        if self.byte_size is not None and self.byte_size > 0:
            return b"\x00" * self.byte_size
        return b""

    def _convert_pyobj(self, value: Any, safe: bool = False) -> bytes | None:
        if isinstance(value, bytes):
            out = value
        elif isinstance(value, (bytearray, memoryview)):
            out = bytes(value)
        elif isinstance(value, str):
            out = value.encode("utf-8")
        elif isinstance(value, bool):
            out = b"\x01" if value else b"\x00"
        elif isinstance(value, int):
            length = max(1, (value.bit_length() + 8) // 8)
            try:
                out = value.to_bytes(length, byteorder="big", signed=value < 0)
            except OverflowError:
                if safe:
                    raise ValueError(
                        f"Cannot encode {value!r} as bytes for {type(self).__name__}."
                    )
                return None
        else:
            try:
                out = bytes(value)
            except TypeError:
                if safe:
                    raise ValueError(
                        f"Cannot convert value of type {type(value).__name__} to bytes "
                        f"for {type(self).__name__}: {value!r}"
                    )
                return None

        if self.byte_size is not None:
            if len(out) < self.byte_size:
                out = out.ljust(self.byte_size, b"\x00")
            elif len(out) > self.byte_size:
                if safe:
                    raise ValueError(
                        f"Binary value of length {len(out)} exceeds fixed "
                        f"byte_size={self.byte_size}: {out!r}"
                    )
                out = out[: self.byte_size]
        return out
