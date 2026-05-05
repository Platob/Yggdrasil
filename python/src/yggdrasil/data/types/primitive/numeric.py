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
    "Int8Type",
    "Int16Type",
    "Int32Type",
    "Int64Type",
    "UInt8Type",
    "UInt16Type",
    "UInt32Type",
    "UInt64Type",
    "FloatingPointType",
    "Float16Type",
    "Float32Type",
    "Float64Type",
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

    def merge_with(
        self,
        other: "DataType",
        mode: Mode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ):
        # Specialized fixed-width subclasses (``Int32Type``, ``UInt8Type``,
        # ``Float64Type``, ...) carry distinct ``type_id`` values, but
        # they still merge as one kind: any two integers collapse via
        # ``_merge_with_same_id`` regardless of width / signedness, and
        # likewise for floats. Decimals stay strict — they have their
        # own ``DECIMAL`` id and don't bridge to ints / floats here.
        mode = Mode.from_(mode, Mode.UPSERT)
        if mode is Mode.IGNORE:
            return self

        same_kind = (
            self.type_id == other.type_id
            or (self.type_id.is_integer and other.type_id.is_integer)
            or (self.type_id.is_floating_point and other.type_id.is_floating_point)
        )
        if same_kind:
            return self._merge_with_same_id(
                other=other, mode=mode, downcast=downcast, upcast=upcast
            )
        return self._merge_with_different_id(
            other=other, mode=mode, downcast=downcast, upcast=upcast
        )

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

    def __new__(cls, byte_size: int | None = None, signed: bool = True, **kwargs):
        # When called on the abstract base, redirect to the registered
        # specialized subclass for ``(byte_size, signed)``. ``Int32Type`` /
        # ``UInt8Type`` / ... carry their own ``DataTypeId`` so they
        # round-trip through ``to_dict`` / ``from_dict`` without losing
        # signedness or width. Unusual sizes (16-byte hugeint, ``None``)
        # fall through to a plain ``IntegerType`` instance — the dynamic
        # fallback. Subclasses skip the redirect: ``Int32Type(...)`` is
        # always an ``Int32Type``.
        if cls is IntegerType:
            target = _SPECIALIZED_INTEGER_TYPES.get((byte_size, bool(signed)))
            if target is not None:
                return object.__new__(target)
        return object.__new__(cls)

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        bits = (self.byte_size or 8) * 8
        prefix = "int" if self.signed else "uint"
        return f"{pad}{prefix}{bits}"

    @classmethod
    def class_type_id(cls) -> DataTypeId:
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
        return cls._matches_dict(value, cls.class_type_id())

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "IntegerType":
        try:
            # Pull defaults from the class itself so specialized subclasses
            # (``UInt8Type``, ``Int64Type``, ...) recover the correct
            # signedness even when the serialized dict omits it.
            byte_size_default = cls.__dataclass_fields__["byte_size"].default
            if byte_size_default is None:
                byte_size_default = 8
            signed_default = cls.__dataclass_fields__["signed"].default

            return cls(
                byte_size=value.get("byte_size", byte_size_default),
                signed=bool(value.get("signed", signed_default)),
            )
        except Exception as e:
            if default is ...:
                raise ValueError(
                    f"Could not parse IntegerType from dict: {value!r}."
                ) from e
            return default

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["signed"] = self.signed
        return base

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

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: Mode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "IntegerType":
        # Accept any IntegerType subclass — the specialized fixed-width
        # subclasses (``Int8Type``, ``UInt32Type``, ...) are siblings,
        # not parent/child, so the default ``isinstance(other,
        # self.__class__)`` check from :class:`PrimitiveType` would
        # spuriously reject ``Int32Type.merge_with(Int64Type(...))``.
        if not isinstance(other, IntegerType):
            raise TypeError(
                f"Cannot merge {self.__class__.__name__} with {other.__class__.__name__}"
            )

        if mode is Mode.IGNORE:
            return self
        if mode is Mode.OVERWRITE:
            return other

        # Funnel construction back through the abstract base so the
        # ``__new__`` redirect picks the correct specialized class for
        # the resolved width / signedness.
        if mode is Mode.AUTO:
            byte_size = self.byte_size or other.byte_size
            signed = self.signed or other.signed
            if byte_size == self.byte_size and signed == self.signed:
                return self
            return IntegerType(byte_size=byte_size, signed=signed)

        if downcast == upcast:
            return self

        left = self.byte_size
        right = other.byte_size
        signed = self.signed or other.signed

        if left is None:
            return other if right is not None else self
        if right is None:
            return self
        if downcast:
            return IntegerType(byte_size=min(left, right), signed=signed)
        return IntegerType(byte_size=max(left, right), signed=signed)


# ======================================================================
# FloatingPointType
# ======================================================================

@dataclass(frozen=True, repr=False)
class FloatingPointType(NumericType):

    def __new__(cls, byte_size: int | None = None, **kwargs):
        # See :meth:`IntegerType.__new__` — same dispatch, keyed on
        # ``byte_size`` only. ``bfloat16`` rides the same Float16Type as
        # IEEE half-precision; we don't model BF separately here.
        if cls is FloatingPointType:
            target = _SPECIALIZED_FLOAT_TYPES.get(byte_size)
            if target is not None:
                return object.__new__(target)
        return object.__new__(cls)

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * indent * level
        bits = (self.byte_size or 8) * 8
        return f"{pad}float{bits}"

    @classmethod
    def class_type_id(cls) -> DataTypeId:
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

        if mode is Mode.IGNORE:
            return self
        if mode is Mode.OVERWRITE:
            return other

        if mode is Mode.AUTO:
            byte_size = self.byte_size or other.byte_size
            if byte_size == self.byte_size:
                return self
            return FloatingPointType(byte_size=byte_size)

        if downcast == upcast:
            return self

        left_size = self.byte_size or 8
        right_size = other.byte_size or 8
        byte_size = min(left_size, right_size) if downcast else max(left_size, right_size)
        # Build through the abstract base so the ``__new__`` redirect
        # selects the right ``FloatXX`` specialized class.
        return FloatingPointType(byte_size=byte_size)

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
        return cls._matches_dict(value, cls.class_type_id())

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "FloatingPointType":
        try:
            return cls(byte_size=value.get("byte_size", 8))
        except Exception as e:
            if default is ...:
                raise ValueError(
                    f"Could not parse FloatingPointType from dict: {value!r}."
                ) from e
            return default

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
# Specialized fixed-width integer / float subclasses
#
# Each one carries its own ``DataTypeId`` so ``to_dict`` / ``from_dict`` /
# ``autotag`` / ``pretty_format`` reflect the concrete width without
# having to read ``byte_size`` and ``signed`` out of metadata. The
# ``__new__`` redirect on the abstract bases (``IntegerType`` /
# ``FloatingPointType``) is what lifts a generic
# ``IntegerType(byte_size=4, signed=True)`` call into ``Int32Type`` —
# the registry below is the single source of truth for the mapping.
# ======================================================================


@dataclass(frozen=True, repr=False)
class Int8Type(IntegerType):
    byte_size: int | None = 1
    signed: bool = True

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.INT8


@dataclass(frozen=True, repr=False)
class Int16Type(IntegerType):
    byte_size: int | None = 2
    signed: bool = True

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.INT16


@dataclass(frozen=True, repr=False)
class Int32Type(IntegerType):
    byte_size: int | None = 4
    signed: bool = True

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.INT32


@dataclass(frozen=True, repr=False)
class Int64Type(IntegerType):
    byte_size: int | None = 8
    signed: bool = True

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.INT64


@dataclass(frozen=True, repr=False)
class UInt8Type(IntegerType):
    byte_size: int | None = 1
    signed: bool = False

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.UINT8


@dataclass(frozen=True, repr=False)
class UInt16Type(IntegerType):
    byte_size: int | None = 2
    signed: bool = False

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.UINT16


@dataclass(frozen=True, repr=False)
class UInt32Type(IntegerType):
    byte_size: int | None = 4
    signed: bool = False

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.UINT32


@dataclass(frozen=True, repr=False)
class UInt64Type(IntegerType):
    byte_size: int | None = 8
    signed: bool = False

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.UINT64


@dataclass(frozen=True, repr=False)
class Float16Type(FloatingPointType):
    byte_size: int | None = 2

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.FLOAT16


@dataclass(frozen=True, repr=False)
class Float32Type(FloatingPointType):
    byte_size: int | None = 4

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.FLOAT32


@dataclass(frozen=True, repr=False)
class Float64Type(FloatingPointType):
    byte_size: int | None = 8

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.FLOAT64


_SPECIALIZED_INTEGER_TYPES: dict[tuple[int | None, bool], type[IntegerType]] = {
    (1, True):  Int8Type,
    (2, True):  Int16Type,
    (4, True):  Int32Type,
    (8, True):  Int64Type,
    (1, False): UInt8Type,
    (2, False): UInt16Type,
    (4, False): UInt32Type,
    (8, False): UInt64Type,
}

_SPECIALIZED_FLOAT_TYPES: dict[int | None, type[FloatingPointType]] = {
    2: Float16Type,
    4: Float32Type,
    8: Float64Type,
}


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
