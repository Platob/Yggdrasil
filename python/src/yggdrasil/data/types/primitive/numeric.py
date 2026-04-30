from __future__ import annotations

import datetime as dt
import decimal
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.io.enums import Mode
from ._helpers import (
    _coerce_str,
    _INT_ARROW_SIGNED,
    _INT_ARROW_UNSIGNED,
    _INT_DDL_SIGNED,
    _INT_DDL_UNSIGNED,
)
from .base import PrimitiveType
from ..base import DataType
from ..id import DataTypeId
from ..support import get_polars, get_spark_sql

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst
    from ...cast.options import CastOptions


__all__ = [
    "NumericType",
    "IntegerType",
    "FloatingPointType",
    "DecimalType",
]


_VIEW_TO_CONCRETE: dict[pa.DataType, pa.DataType] = {
    pa.string_view(): pa.large_string(),
    pa.binary_view(): pa.large_binary(),
}


# ======================================================================
# NumericType base — empty-string / empty-bytes → NULL before the cast
# ======================================================================
#
# Databricks / Spark ingests frequently land empty strings in string
# columns when the upstream CSV had no separator between two commas.
# Blindly casting ``""`` to a numeric target parses as NULL on Spark
# but can raise ArrowInvalid under pyarrow's ``safe=True`` kernel.
#
# We normalize the behaviour across engines: any empty-string / empty-
# bytes source value becomes NULL before the numeric cast runs.
#
# The old code called ``options.need_cast(<value>)`` positionally, which
# does not match the base-class signature (``need_cast(check_names,
# check_dtypes, check_metadata)``). All overrides now use the keyword
# form and guard ``source_field`` for None — a Field peeked from a
# naked Arrow Array / Polars Series may leave it unset.


@dataclass(frozen=True)
class NumericType(PrimitiveType, ABC):

    @staticmethod
    def _source_type_id(options: "CastOptions") -> DataTypeId | None:
        sf = options.source_field
        return sf.dtype.type_id if sf is not None else None

    # ------------------------------------------------------------------
    # Arrow
    # ------------------------------------------------------------------

    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.Array:
        if options.need_cast(array, self):
            src_id = self._source_type_id(options)
            if src_id in (DataTypeId.STRING, DataTypeId.BINARY):
                # Materialize view → concrete so the ``equal`` kernel resolves.
                # pyarrow's ``equal`` is not registered for the view variants.
                original_type = array.type
                concrete_type = _VIEW_TO_CONCRETE.get(original_type, original_type)

                if concrete_type is not original_type:
                    array = array.cast(concrete_type)

                empty = b"" if src_id == DataTypeId.BINARY else ""
                array = pc.if_else(
                    pc.equal(array, pa.scalar(empty, type=concrete_type)),
                    pa.scalar(None, type=concrete_type),
                    array,
                )

                # Round-trip back to the original view type if we materialized.
                if concrete_type is not original_type:
                    array = array.cast(original_type)

        return super()._cast_arrow_array(array, options)

    # ------------------------------------------------------------------
    # Polars
    # ------------------------------------------------------------------

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ) -> "polars.Series":
        if options.need_cast(series, self):
            src_id = self._source_type_id(options)
            if src_id == DataTypeId.STRING:
                # Replace empty strings with null before the numeric cast.
                series = series.set(series.str.len_bytes() == 0, None)
            elif src_id == DataTypeId.BINARY:
                # Polars Binary: length is bytes-length of the buffer.
                series = series.set(series.bin.size() == 0, None)

        return super()._cast_polars_series(series, options)

    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options: "CastOptions",
    ) -> "polars.Expr":
        if options.need_cast(expr, self):
            pl = get_polars()
            src_id = self._source_type_id(options)

            if src_id == DataTypeId.STRING:
                expr = (
                    pl.when(expr.str.len_bytes() == 0)
                    .then(None)
                    .otherwise(expr)
                )
            elif src_id == DataTypeId.BINARY:
                expr = (
                    pl.when(expr.bin.size() == 0)
                    .then(None)
                    .otherwise(expr)
                )

        return super()._cast_polars_expr(expr, options)

    # ------------------------------------------------------------------
    # Spark
    # ------------------------------------------------------------------

    def _cast_spark_column(
        self,
        column: Any,
        options: "CastOptions",
    ) -> Any:
        if options.need_cast(column, self):
            spark = get_spark_sql()
            F = spark.functions
            src_id = self._source_type_id(options)

            # Spark's ``length()`` returns character length on StringType and
            # byte length on BinaryType — exactly the semantic we want.
            if src_id in (DataTypeId.STRING, DataTypeId.BINARY):
                column = F.when(F.length(column) == 0, F.lit(None)).otherwise(column)

        return super()._cast_spark_column(column, options)


# ======================================================================
# Polars integer signedness helpers — kept at numeric-module scope
# because IntegerType needs them but they're not useful anywhere else.
# ======================================================================

def _polars_flip_int_signedness(dtype: Any) -> Any:
    pl = get_polars()
    flip = {
        pl.Int8: pl.UInt8,
        pl.Int16: pl.UInt16,
        pl.Int32: pl.UInt32,
        pl.Int64: pl.UInt64,
        pl.UInt8: pl.Int8,
        pl.UInt16: pl.Int16,
        pl.UInt32: pl.Int32,
        pl.UInt64: pl.Int64,
    }
    return flip[dtype]


def _polars_is_integer(dtype: Any) -> bool:
    pl = get_polars()
    return dtype in {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    }


def _polars_is_signed_int(dtype: Any) -> bool:
    pl = get_polars()
    return dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64}


# ======================================================================
# IntegerType
# ======================================================================

@dataclass(frozen=True, repr=False)
class IntegerType(NumericType):
    signed: bool = True

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        bits = (self.byte_size or 8) * 8
        prefix = "int" if self.signed else "uint"
        return f"{pad}{prefix}{bits}"

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.INTEGER

    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.Array:
        if options.need_cast(array, self):
            src_type = options.source_field.to_arrow_type()
            tgt_type = self.to_arrow()

            if (
                pa.types.is_integer(src_type)
                and pa.types.is_integer(tgt_type)
                and pa.types.is_signed_integer(src_type)
                != pa.types.is_signed_integer(tgt_type)
            ):
                casted = pc.cast(
                    array,
                    target_type=tgt_type,
                    safe=False,
                    memory_pool=options.arrow_memory_pool,
                )
                return self.fill_arrow_array_nulls(
                    casted, nullable=self._target_nullable(options)
                )

        return super()._cast_arrow_array(array, options)

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ):
        if options.need_cast(series, self):
            src_dtype = series.dtype
            tgt_dtype = self.to_polars()

            if (
                _polars_is_integer(src_dtype)
                and _polars_is_integer(tgt_dtype)
                and _polars_is_signed_int(src_dtype) != _polars_is_signed_int(tgt_dtype)
            ):
                flipped_src_dtype = _polars_flip_int_signedness(src_dtype)
                casted = series.reinterpret(
                    signed=_polars_is_signed_int(flipped_src_dtype)
                )
                if flipped_src_dtype != tgt_dtype:
                    # Width differs — cast between same-signedness widths where
                    # pyarrow/polars wraparound is well-defined.
                    casted = casted.cast(tgt_dtype, strict=False)
                return self.fill_polars_array_nulls(
                    casted, nullable=self._target_nullable(options)
                )

        return super()._cast_polars_series(series, options)

    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options: "CastOptions",
    ):
        if options.need_cast(expr, self):
            pl = get_polars()
            source_field = options.source_field
            src_dtype = (
                source_field.dtype.to_polars() if source_field is not None else None
            )
            if isinstance(src_dtype, type) and issubclass(src_dtype, pl.DataType):
                src_dtype = src_dtype()
            tgt_dtype = self.to_polars()

            if (
                src_dtype is not None
                and _polars_is_integer(src_dtype)
                and _polars_is_integer(tgt_dtype)
                and _polars_is_signed_int(src_dtype) != _polars_is_signed_int(tgt_dtype)
            ):
                flipped_src_dtype = _polars_flip_int_signedness(src_dtype)
                casted = expr.reinterpret(
                    signed=_polars_is_signed_int(flipped_src_dtype)
                )
                if flipped_src_dtype != tgt_dtype:
                    casted = casted.cast(tgt_dtype, strict=False)
                return self.fill_polars_array_nulls(
                    casted, nullable=self._target_nullable(options)
                )

        return super()._cast_polars_expr(expr, options)

    # ------------------------------------------------------------------
    # Engine probes / constructors
    # ------------------------------------------------------------------

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_integer(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "IntegerType":
        if pa.types.is_int8(dtype):
            return cls(byte_size=1, signed=True)
        if pa.types.is_int16(dtype):
            return cls(byte_size=2, signed=True)
        if pa.types.is_int32(dtype):
            return cls(byte_size=4, signed=True)
        if pa.types.is_int64(dtype):
            return cls(byte_size=8, signed=True)
        if pa.types.is_uint8(dtype):
            return cls(byte_size=1, signed=False)
        if pa.types.is_uint16(dtype):
            return cls(byte_size=2, signed=False)
        if pa.types.is_uint32(dtype):
            return cls(byte_size=4, signed=False)
        if pa.types.is_uint64(dtype):
            return cls(byte_size=8, signed=False)
        raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return dtype in {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        }

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "IntegerType":
        pl = get_polars()
        mapping = {
            pl.Int8: (1, True),
            pl.Int16: (2, True),
            pl.Int32: (4, True),
            pl.Int64: (8, True),
            pl.UInt8: (1, False),
            pl.UInt16: (2, False),
            pl.UInt32: (4, False),
            pl.UInt64: (8, False),
        }
        if dtype not in mapping:
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        byte_size, signed = mapping[dtype]
        return cls(byte_size=byte_size, signed=signed)

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = get_spark_sql()
        return isinstance(dtype, (
            spark.types.ByteType,
            spark.types.ShortType,
            spark.types.IntegerType,
            spark.types.LongType,
        ))

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "IntegerType":
        spark = get_spark_sql()
        if isinstance(dtype, spark.types.ByteType):
            return cls(byte_size=1, signed=True)
        if isinstance(dtype, spark.types.ShortType):
            return cls(byte_size=2, signed=True)
        if isinstance(dtype, spark.types.IntegerType):
            return cls(byte_size=4, signed=True)
        if isinstance(dtype, spark.types.LongType):
            return cls(byte_size=8, signed=True)
        raise TypeError(f"Unsupported Spark data type: {dtype!r}")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.INTEGER)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IntegerType":
        return cls(
            byte_size=value.get("byte_size", 8),
            signed=bool(value.get("signed", True)),
        )

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    @property
    def _size(self) -> int:
        return self.byte_size or 8

    def to_arrow(self) -> pa.DataType:
        mapping = _INT_ARROW_SIGNED if self.signed else _INT_ARROW_UNSIGNED
        return mapping[self._size]()

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        signed = {1: pl.Int8, 2: pl.Int16, 4: pl.Int32, 8: pl.Int64}
        unsigned = {1: pl.UInt8, 2: pl.UInt16, 4: pl.UInt32, 8: pl.UInt64}
        return (signed if self.signed else unsigned)[self._size]

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        t = spark.types
        if self.signed:
            mapping = {1: t.ByteType, 2: t.ShortType, 4: t.IntegerType, 8: t.LongType}
            return mapping[self._size]()
        # Spark doesn't really support uints natively — widen to preserve range.
        unsigned_factories = {
            1: t.ShortType,
            2: t.IntegerType,
            4: t.LongType,
            8: lambda: t.DecimalType(20, 0),
        }
        return unsigned_factories[self._size]()

    def to_databricks_ddl(self) -> str:
        mapping = _INT_DDL_SIGNED if self.signed else _INT_DDL_UNSIGNED
        return mapping[self._size]

    # ------------------------------------------------------------------
    # Defaults / conversion / autotag
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else 0

    def _convert_pyobj(self, value: Any, safe: bool = False) -> int | None:
        token = _coerce_str(value)
        if token is not None:
            stripped = token.strip()
            if not stripped:
                if safe:
                    raise ValueError(
                        f"Cannot parse int from empty string for {type(self).__name__}."
                    )
                return None
            try:
                # Go through int(str, 0) so "0x1a", "0b10", "0o7" are accepted;
                # fall back to float→int when the token carries a decimal point
                # or scientific notation.
                return int(stripped, 0)
            except ValueError:
                try:
                    return int(float(stripped))
                except (TypeError, ValueError):
                    if safe:
                        raise ValueError(
                            f"Cannot parse int from {value!r} for {type(self).__name__}."
                        )
                    return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            import math

            if math.isnan(value) or math.isinf(value):
                if safe:
                    raise ValueError(
                        f"Cannot convert non-finite float {value!r} to int "
                        f"for {type(self).__name__}."
                    )
                return None
            return int(value)
        if isinstance(value, decimal.Decimal):
            if value.is_nan() or value.is_infinite():
                if safe:
                    raise ValueError(
                        f"Cannot convert non-finite Decimal {value!r} to int "
                        f"for {type(self).__name__}."
                    )
                return None
            return int(value)
        if isinstance(value, dt.datetime):
            return int(value.timestamp())
        if isinstance(value, dt.timedelta):
            return int(value.total_seconds())
        if safe:
            raise ValueError(
                f"Cannot convert {type(value).__name__} to int "
                f"for {type(self).__name__}: {value!r}."
            )
        return None

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        tags[b"signed"] = b"true" if self.signed else b"false"
        return tags


# ======================================================================
# FloatingPointType
# ======================================================================

@dataclass(frozen=True, repr=False)
class FloatingPointType(NumericType):

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * indent * level
        bits = (self.byte_size or 8) * 8
        return f"{pad}float{bits}"

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.FLOAT

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: Mode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "FloatingPointType":
        if not isinstance(other, FloatingPointType):
            raise TypeError(
                f"Cannot merge FloatingPointType with {other.__class__.__name__}"
            )
        if downcast == upcast:
            return self

        left_size = self.byte_size or 8
        right_size = other.byte_size or 8
        byte_size = min(left_size, right_size) if downcast else max(left_size, right_size)
        return self.__class__(byte_size=byte_size)

    # ------------------------------------------------------------------
    # Engine probes / constructors
    # ------------------------------------------------------------------

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_floating(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "FloatingPointType":
        if pa.types.is_float32(dtype):
            return cls(byte_size=4)
        if pa.types.is_float64(dtype):
            return cls(byte_size=8)
        if pa.types.is_float16(dtype):
            return cls(byte_size=2)
        raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return dtype in {pl.Float32, pl.Float64}

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "FloatingPointType":
        pl = get_polars()
        if dtype == pl.Float32:
            return cls(byte_size=4)
        if dtype == pl.Float64:
            return cls(byte_size=8)
        raise TypeError(f"Unsupported Polars data type: {dtype!r}")

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = get_spark_sql()
        return isinstance(dtype, (spark.types.FloatType, spark.types.DoubleType))

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "FloatingPointType":
        spark = get_spark_sql()
        if isinstance(dtype, spark.types.FloatType):
            return cls(byte_size=4)
        if isinstance(dtype, spark.types.DoubleType):
            return cls(byte_size=8)
        raise TypeError(f"Unsupported Spark data type: {dtype!r}")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.FLOAT)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FloatingPointType":
        return cls(byte_size=value.get("byte_size", 8))

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    @property
    def _size(self) -> int:
        return self.byte_size or 8

    def to_arrow(self) -> pa.DataType:
        if self._size == 2:
            return pa.float16() if hasattr(pa, "float16") else pa.float32()
        return pa.float64() if self._size == 8 else pa.float32()

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.Float64 if self._size == 8 else pl.Float32

    def to_spark(self) -> Any:
        t = get_spark_sql().types
        return t.DoubleType() if self._size == 8 else t.FloatType()

    def to_databricks_ddl(self) -> str:
        return "DOUBLE" if self._size == 8 else "FLOAT"

    # ------------------------------------------------------------------
    # Defaults / conversion
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else 0.0

    def _convert_pyobj(self, value: Any, safe: bool = False) -> float | None:
        token = _coerce_str(value)
        if token is not None:
            stripped = token.strip()
            if not stripped:
                if safe:
                    raise ValueError(
                        f"Cannot parse float from empty string for {type(self).__name__}."
                    )
                return None
            try:
                return float(stripped)
            except (TypeError, ValueError):
                if safe:
                    raise ValueError(
                        f"Cannot parse float from {value!r} for {type(self).__name__}."
                    )
                return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, decimal.Decimal):
            return float(value)
        if safe:
            raise ValueError(
                f"Cannot convert {type(value).__name__} to float "
                f"for {type(self).__name__}: {value!r}."
            )
        return None


# ======================================================================
# DecimalType
# ======================================================================

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

    @property
    def type_id(self) -> DataTypeId:
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
    def from_dict(cls, value: dict[str, Any]) -> "DecimalType":
        return cls(
            byte_size=value.get("byte_size"),
            precision=value.get("precision", 38),
            scale=value.get("scale", 18),
        )

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
