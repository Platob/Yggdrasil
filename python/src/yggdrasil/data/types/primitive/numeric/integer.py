"""Integer type family — :class:`IntegerType` + sized subclasses.

Layout:

* :class:`IntegerType` — abstract / unsized 2's-complement integer
  with a ``__new__`` redirect that promotes generic
  ``IntegerType(byte_size=N, signed=B)`` calls to the matching
  fixed-width subclass via :data:`_SPECIALIZED_INTEGER_TYPES`.
* :class:`Int8Type` … :class:`Int64Type` — signed widths.
* :class:`UInt8Type` … :class:`UInt64Type` — unsigned widths. Spark
  has no native unsigned integers; ``to_spark`` widens to the next
  signed type, ``as_spark`` does the same-width sign flip with
  two's-complement reinterpret semantics.

The dispatch table at the bottom is the single source of truth for
the redirect — ``__new__`` reads from it; ``IntegerType`` /
``UInt32Type`` / ... never see each other directly.
"""

from __future__ import annotations

import datetime as dt
import decimal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.enums import Mode

from .._helpers import (
    _coerce_str,
    _INT_ARROW_SIGNED,
    _INT_ARROW_UNSIGNED,
    _INT_DDL_SIGNED,
    _INT_DDL_UNSIGNED,
)
from ...base import _default_singleton
from ...id import DataTypeId
from yggdrasil.lazy_imports import polars_module, spark_sql_module
from .base import (
    NumericType,
    _polars_flip_int_signedness,
    _polars_is_integer,
    _polars_is_signed_int,
)

if TYPE_CHECKING:
    import polars  # noqa: F401
    import pyspark.sql.types as pst  # noqa: F401
    from ....cast.options import CastOptions  # noqa: F401
    from ...base import DataType  # noqa: F401


__all__ = [
    "IntegerType",
    "Int8Type",
    "Int16Type",
    "Int32Type",
    "Int64Type",
    "UInt8Type",
    "UInt16Type",
    "UInt32Type",
    "UInt64Type",
]


@dataclass(frozen=True, repr=False)
class IntegerType(NumericType):
    signed: bool = True

    def __new__(cls, byte_size: int | None = None, signed: bool = True, **kwargs):
        # Always redirect to the registered specialized subclass for
        # ``(byte_size, signed)`` — ``Int8Type`` / ``Int32Type`` /
        # ``UInt64Type`` / ... carry their own ``DataTypeId``. Two
        # branches collapse the cases:
        #
        # * ``IntegerType(byte_size=4, signed=True)`` redirects to
        #   ``Int32Type`` (the abstract → fixed promotion).
        # * ``Int8Type(byte_size=8, signed=True)`` *also* redirects, to
        #   ``Int64Type`` — a malformed specialized construction can't
        #   silently leave its declared width behind.
        #
        # Unusual sizes (16-byte hugeint, ``byte_size=None``) have no
        # registered specialized class, so the lookup misses and the
        # call lands on whichever ``cls`` the caller asked for —
        # typically the abstract :class:`IntegerType` for the dynamic
        # fallback, or a specialized class via ``Int64Type()`` (whose
        # dataclass default fills in ``byte_size=8`` during ``__init__``
        # after ``__new__`` returns).
        target = _SPECIALIZED_INTEGER_TYPES.get((byte_size, bool(signed)))
        resolved = target if (target is not None and target is not cls) else cls
        # Singleton fast path: leaf specialized subclasses
        # (``Int8Type`` … ``UInt64Type``) construct with the same field
        # values every time — share one instance so the lazy
        # ``to_arrow`` / ``to_polars`` / ``to_spark`` caches survive
        # across every caller. ``kwargs`` carries metadata-only knobs
        # in some subclasses, so it bypasses the singleton. The
        # abstract :class:`IntegerType` itself only singletons on the
        # default-arg call (``IntegerType()``) — non-default widths
        # need their own instance because the field values differ.
        if not kwargs and resolved in _SPECIALIZED_INTEGER_TYPES.values():
            return _default_singleton(resolved)
        if not kwargs and byte_size is None and signed is True and resolved is IntegerType:
            return _default_singleton(resolved)
        return object.__new__(resolved)

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
            src_type = options.source.to_arrow_type()
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
            pl = polars_module()
            source_field = options.source
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
        pl = polars_module()
        return dtype in {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        }

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "IntegerType":
        pl = polars_module()
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
        spark = spark_sql_module()
        return isinstance(
            dtype,
            (
                spark.types.ByteType,
                spark.types.ShortType,
                spark.types.IntegerType,
                spark.types.LongType,
            ),
        )

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "IntegerType":
        spark = spark_sql_module()
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

    def _default_pyhint(self) -> Any:
        # All integer widths / signedness collapse to ``int`` in Python.
        # Callers that need the original ``np.int64`` / ``np.uint8`` /
        # custom ``IntEnum`` alias rely on the ``_pyhint_cache`` stamp
        # ``from_pytype`` sets — this fallback is the safe round-trip
        # when no hint was preserved.
        return int

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
        pl = polars_module()
        signed = {1: pl.Int8, 2: pl.Int16, 4: pl.Int32, 8: pl.Int64}
        unsigned = {1: pl.UInt8, 2: pl.UInt16, 4: pl.UInt32, 8: pl.UInt64}
        return (signed if self.signed else unsigned)[self._size]

    def to_spark(self) -> Any:
        spark = spark_sql_module()
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

    def to_spark_name(self) -> str:
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
        # Specialized fixed-width subclasses (``Int8Type`` … ``Int64Type``,
        # ``UInt8Type`` … ``UInt64Type``) already encode signedness in
        # their ``type_id`` (and the ``type_name`` tag the base class
        # writes), so a ``signed`` tag would just be redundant noise.
        # Only the abstract ``IntegerType`` (``type_id=INTEGER``)
        # actually needs the flag, since its name alone doesn't pin
        # the signedness.
        if self.type_id is DataTypeId.INTEGER:
            tags[b"signed"] = b"true" if self.signed else b"false"
        return tags

    def as_spark(self) -> "IntegerType":
        # Spark has no native unsigned integers — flip ``signed`` while
        # keeping the same byte width. The cast goes through Arrow /
        # Polars / pyspark with two's-complement reinterpretation
        # (``safe=False``), so values that exceed the signed range wrap
        # negative: ``max(uint64) → -1`` as ``int64`` and vice versa.
        # Same width keeps storage cheap and the round-trip lossless
        # at the bit level.
        if self.signed:
            return self
        return IntegerType(byte_size=self._size, signed=True)

    def reinterpret_pyobj(self, value: int) -> int:
        """Reinterpret *value* as this type's two's-complement form.

        Mirrors what ``pyarrow.compute.cast(..., safe=False)`` does for
        signed↔unsigned casts at the Arrow level — values are masked
        to ``byte_size * 8`` bits and read back with this type's
        signedness. ``max(uint64)`` becomes ``-1`` when reinterpreted
        as ``int64``; ``-1`` becomes ``max(uint64)`` reinterpreted the
        other way.
        """
        bits = self._size * 8
        mask = (1 << bits) - 1
        raw = value & mask
        if self.signed and raw & (1 << (bits - 1)):
            return raw - (1 << bits)
        return raw

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


# ---------------------------------------------------------------------------
# Specialized fixed-width integer subclasses. Each one carries its own
# ``DataTypeId`` so ``to_dict`` / ``from_dict`` / ``autotag`` /
# ``pretty_format`` reflect the concrete width without having to read
# ``byte_size`` and ``signed`` out of metadata. The dispatch table
# below is the single source of truth for the ``__new__`` redirect.
# ---------------------------------------------------------------------------


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


_SPECIALIZED_INTEGER_TYPES: dict[tuple[int | None, bool], type[IntegerType]] = {
    (1, True): Int8Type,
    (2, True): Int16Type,
    (4, True): Int32Type,
    (8, True): Int64Type,
    (1, False): UInt8Type,
    (2, False): UInt16Type,
    (4, False): UInt32Type,
    (8, False): UInt64Type,
}
