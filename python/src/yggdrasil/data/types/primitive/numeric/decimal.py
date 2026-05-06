""":class:`DecimalType` — fixed-precision decimal numbers.

Decimals stay strict relative to integers / floats: their merge
behavior keeps ``DecimalType`` intact and never bridges across the
numeric kinds, so a ``DecimalType.merge_with(IntegerType(...))``
goes through the usual cross-id resolver rather than collapsing
the two sides into one. The byte_size auto-derives from
``precision`` (``≤38`` ⇒ 16-byte Decimal128, otherwise 32-byte
Decimal256) — the field is computed in ``__post_init__`` so
callers don't have to think about it.
"""
from __future__ import annotations

import decimal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from yggdrasil.data.enums import Mode

from .._helpers import _coerce_str
from ...id import DataTypeId
from ...support import get_polars, get_spark_sql
from .base import NumericType

if TYPE_CHECKING:
    import polars  # noqa: F401
    import pyspark.sql.types as pst  # noqa: F401
    from ...base import DataType  # noqa: F401


__all__ = ["DecimalType"]


@dataclass(frozen=True, repr=False)
class DecimalType(NumericType):
    precision: int = 38
    scale: int = 18

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * indent * level
        return f"{pad}decimal({self.precision}, {self.scale})"

    def __post_init__(self):
        if not self.byte_size:
            if self.precision < 38:
                object.__setattr__(self, "byte_size", 16)
            else:
                object.__setattr__(self, "byte_size", 32)

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.DECIMAL

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: Mode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "DecimalType":
        if not isinstance(other, DecimalType):
            raise TypeError(f"Cannot merge DecimalType with {other.__class__.__name__}")

        if downcast == upcast:
            return self

        if downcast:
            precision = min(self.precision, other.precision)
            scale = min(self.scale, other.scale)
        else:
            left_integer = self.precision - self.scale
            right_integer = other.precision - other.scale
            scale = max(self.scale, other.scale)
            precision = max(left_integer, right_integer) + scale

        byte_size = 16 if precision <= 38 else 32
        return self.__class__(
            byte_size=byte_size,
            precision=precision,
            scale=scale,
        )

    # ------------------------------------------------------------------
    # Engine probes / constructors
    # ------------------------------------------------------------------

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_decimal(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.Decimal128Type) -> "DecimalType":
        if not pa.types.is_decimal(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")
        bit_width = getattr(dtype, "bit_width", 128)
        return cls(
            byte_size=bit_width // 8,
            precision=dtype.precision,
            scale=dtype.scale,
        )

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return isinstance(dtype, pl.Decimal)

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "DecimalType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        precision = getattr(dtype, "precision", 38) or 38
        scale = getattr(dtype, "scale", 18) or 18
        return cls(precision=precision, scale=scale)

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = get_spark_sql()
        return isinstance(dtype, spark.types.DecimalType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "DecimalType":
        if not cls.handles_spark_type(dtype):
            raise TypeError(f"Unsupported Spark data type: {dtype!r}")
        return cls(precision=dtype.precision, scale=dtype.scale)

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.DECIMAL)

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "DecimalType":
        try:
            return cls(
                byte_size=value.get("byte_size"),
                precision=int(value.get("precision", 38)),
                scale=int(value.get("scale", 18)),
            )
        except Exception as e:
            if default is ...:
                raise ValueError(
                    f"Could not parse DecimalType from dict: {value!r}."
                ) from e
            return default

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    def to_arrow(self) -> pa.DataType:
        if self.precision <= 38:
            return pa.decimal128(self.precision, self.scale)
        return pa.decimal256(self.precision, self.scale)

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.Decimal(precision=self.precision, scale=self.scale)

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        return spark.types.DecimalType(self.precision, self.scale)

    def to_databricks_ddl(self) -> str:
        return f"DECIMAL({self.precision}, {self.scale})"

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["precision"] = self.precision
        base["scale"] = self.scale
        return base

    # ------------------------------------------------------------------
    # Defaults / conversion / autotag
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        return decimal.Decimal(0)

    def _convert_pyobj(self, value: Any, safe: bool = False) -> decimal.Decimal | None:
        token = _coerce_str(value)
        if token is not None:
            stripped = token.strip()
            if not stripped:
                if safe:
                    raise ValueError(
                        f"Cannot parse Decimal from empty string for {type(self).__name__}."
                    )
                return None
            try:
                return decimal.Decimal(stripped)
            except (decimal.InvalidOperation, ValueError):
                if safe:
                    raise ValueError(
                        f"Cannot parse Decimal from {value!r} for {type(self).__name__}."
                    )
                return None
        if isinstance(value, decimal.Decimal):
            return value
        if isinstance(value, bool):
            return decimal.Decimal(int(value))
        if isinstance(value, int):
            return decimal.Decimal(value)
        if isinstance(value, float):
            import math

            if math.isnan(value) or math.isinf(value):
                if safe:
                    raise ValueError(
                        f"Cannot convert non-finite float {value!r} to Decimal "
                        f"for {type(self).__name__}."
                    )
                return None
            return decimal.Decimal(str(value))
        if safe:
            raise ValueError(
                f"Cannot convert {type(value).__name__} to Decimal "
                f"for {type(self).__name__}: {value!r}."
            )
        return None

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        tags[b"precision"] = str(self.precision).encode("utf-8")
        tags[b"scale"] = str(self.scale).encode("utf-8")
        return tags

    def with_precision(self, precision: int, copy: bool = True):
        if precision is None or precision == self.precision:
            return self

        if copy:
            return DecimalType(precision=precision, scale=self.scale)

        object.__setattr__(self, "precision", precision)
        return self

    def with_scale(self, scale: int, copy: bool = True):
        if scale is None or scale == self.scale:
            return self

        if copy:
            return DecimalType(precision=self.precision, scale=scale)

        object.__setattr__(self, "scale", scale)
        return self
