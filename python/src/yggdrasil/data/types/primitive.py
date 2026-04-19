from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc
from yggdrasil.io import SaveMode

from .base import DataType
from .id import DataTypeId
from .support import get_polars, get_spark_sql

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst
    from ..data_field import Field
    from yggdrasil.data.cast.options import CastOptions


__all__ = [
    "PrimitiveType",
    "NullType",
    "BinaryType",
    "StringType",
    "BooleanType",
    "NumericType",
    "IntegerType",
    "XXHIntType",
    "FloatingPointType",
    "DecimalType",
    "TemporalType",
    "DateType",
    "TimeType",
    "TimestampType",
    "DurationType",
]


def _json_encode_helpers():
    from .nested._cast_json import (
        cast_arrow_json_encode_array,
        cast_polars_json_encode_series,
        cast_spark_json_encode_column,
        is_json_nested_source,
    )

    return (
        cast_arrow_json_encode_array,
        cast_polars_json_encode_series,
        cast_spark_json_encode_column,
        is_json_nested_source,
    )


class _JsonEncodeTargetMixin:
    """Mixin providing nested→string/binary cast via JSON serialisation.

    Used by ``StringType`` and ``BinaryType`` whose arrow cast path
    otherwise raises ``ArrowNotImplementedError`` when fed a
    list/struct/map source.
    """

    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.Array:
        options = options.check_source(array)

        (
            cast_arrow_json_encode_array,
            _,
            _,
            is_json_nested_source,
        ) = _json_encode_helpers()

        if is_json_nested_source(options.source_field.dtype.type_id):
            return cast_arrow_json_encode_array(array, options)

        return super()._cast_arrow_array(array, options)

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ):
        options.check_source(series)

        (
            _,
            cast_polars_json_encode_series,
            _,
            is_json_nested_source,
        ) = _json_encode_helpers()

        if is_json_nested_source(options.source_field.dtype.type_id):
            return cast_polars_json_encode_series(series, options)

        return super()._cast_polars_series(series, options)

    def _cast_polars_expr(
        self,
        expr: Any,
        options: "CastOptions",
    ):
        (
            _,
            _,
            _,
            is_json_nested_source,
        ) = _json_encode_helpers()

        if options.source_field is not None and is_json_nested_source(
            options.source_field.dtype.type_id
        ):
            raise TypeError(
                f"Cannot cast {options.source_field} to {options.target_field} "
                "as an expression; use cast_polars_series for a JSON-encoded "
                "nested source."
            )

        return super()._cast_polars_expr(expr, options)

    def _cast_spark_column(
        self,
        column: Any,
        options: "CastOptions",
    ):
        options.check_source(column)

        (
            _,
            _,
            cast_spark_json_encode_column,
            is_json_nested_source,
        ) = _json_encode_helpers()

        if is_json_nested_source(options.source_field.dtype.type_id):
            return cast_spark_json_encode_column(column, options)

        return super()._cast_spark_column(column, options)


@dataclass(frozen=True)
class PrimitiveType(DataType, ABC):
    byte_size: int | None = None

    @property
    def children_fields(self) -> list["Field"]:
        return []

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        if self.byte_size is not None:
            base["byte_size"] = self.byte_size
        return base

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        if self.byte_size is not None:
            tags[b"byte_size"] = str(self.byte_size).encode("utf-8")
        return tags

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "PrimitiveType":
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Cannot merge {self.__class__.__name__} with {other.__class__.__name__}"
            )
        if downcast == upcast:
            return self

        left = self.byte_size
        right = other.byte_size

        if left is None:
            return other if right is not None else self
        if right is None:
            return self

        if downcast:
            return self.__class__(byte_size=min(left, right))
        return self.__class__(byte_size=max(left, right))


@dataclass(frozen=True)
class NullType(PrimitiveType):

    def __str__(self):
        return "null"

    def __repr__(self):
        return self.__str__()

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.NULL

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
    def from_dict(cls, value: dict[str, Any]) -> "NullType":
        return cls()

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

    def default_pyobj(self, nullable: bool) -> Any:
        return None


@dataclass(frozen=True)
class BinaryType(_JsonEncodeTargetMixin, PrimitiveType):
    large: bool = False
    view: bool = False

    def __str__(self):
        s = "large_binary" if self.large else "binary"
        return s + "_view" if self.view else s

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.BINARY

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "BinaryType":
        if not isinstance(other, BinaryType):
            raise TypeError(f"Cannot merge BinaryType with {other.__class__.__name__}")
        if downcast == upcast:
            return self

        left_size = self.byte_size
        right_size = other.byte_size

        if left_size is None or right_size is None:
            byte_size = left_size if right_size is None else right_size if left_size is None else None
        else:
            byte_size = min(left_size, right_size) if downcast else max(left_size, right_size)

        if downcast:
            large = self.large and other.large
        else:
            large = self.large or other.large

        view = self.view and other.view

        if byte_size is not None:
            large = False
            view = False

        return self.__class__(
            byte_size=byte_size,
            large=large,
            view=view,
        )

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
    def from_dict(cls, value: dict[str, Any]) -> "BinaryType":
        return cls(
            byte_size=value.get("byte_size"),
            large=bool(value.get("large", False)),
            view=bool(value.get("view", False)),
        )

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

    def to_databricks_ddl(self) -> str:
        return "BINARY"

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else b""


_TEMPORAL_TYPE_IDS = frozenset(
    {
        DataTypeId.DATE,
        DataTypeId.TIME,
        DataTypeId.TIMESTAMP,
        DataTypeId.DURATION,
    }
)


@dataclass(frozen=True)
class StringType(_JsonEncodeTargetMixin, PrimitiveType):
    large: bool = False
    view: bool = False

    def __str__(self):
        s = "large_string" if self.large else "string"
        return s + "_view" if self.view else s


    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.STRING

    # ------------------------------------------------------------------
    # Temporal source support
    # ------------------------------------------------------------------

    @staticmethod
    def _source_is_temporal(options: "CastOptions") -> bool:
        sf = options.source_field
        return sf is not None and sf.dtype.type_id in _TEMPORAL_TYPE_IDS

    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.Array:
        options = options.check_source(array)

        if self._source_is_temporal(options):
            from .temporal import arrow_cast_to_string

            casted = arrow_cast_to_string(array)
            if self.to_arrow() != casted.type:
                casted = pa.compute.cast(casted, self.to_arrow())
            return options.fill_arrow_nulls(casted)

        return super()._cast_arrow_array(array, options)

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ):
        options.check_source(series)

        if self._source_is_temporal(options):
            pl = get_polars()
            from .temporal import arrow_cast_to_string

            arrow = arrow_cast_to_string(series.to_arrow())
            casted = pl.Series(name=series.name, values=arrow, dtype=pl.String)
            return self.fill_polars_array_nulls(
                casted, nullable=self._target_nullable(options)
            )

        return super()._cast_polars_series(series, options)

    def _cast_polars_expr(
        self,
        expr: Any,
        options: "CastOptions",
    ):
        if self._source_is_temporal(options):
            pl = get_polars()
            # Polars Expr path: format via the native dt accessor where possible.
            src_id = options.source_field.dtype.type_id
            if src_id in (DataTypeId.DATE, DataTypeId.TIMESTAMP):
                return expr.dt.to_string("iso").cast(pl.String)
            if src_id == DataTypeId.TIME:
                return expr.cast(pl.String, strict=options.safe)
            # Duration: render as integer count of source unit for round-trip.
            return expr.dt.total_microseconds().cast(pl.String)

        return super()._cast_polars_expr(expr, options)

    def _cast_spark_column(
        self,
        column: Any,
        options: "CastOptions",
    ):
        options = options.check_source(column)

        if self._source_is_temporal(options):
            from .temporal import spark_temporal_to_string

            casted = spark_temporal_to_string(column)
            return self.fill_spark_column_nulls(
                casted, nullable=self._target_nullable(options)
            )

        return super()._cast_spark_column(column, options)

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "StringType":
        if not isinstance(other, StringType):
            raise TypeError(f"Cannot merge StringType with {other.__class__.__name__}")
        if downcast == upcast:
            return self

        if self.byte_size is None or other.byte_size is None:
            byte_size = self.byte_size or other.byte_size
        else:
            byte_size = min(self.byte_size, other.byte_size) if downcast else max(
                self.byte_size,
                other.byte_size,
            )

        if downcast:
            large = self.large and other.large
        else:
            large = self.large or other.large

        view = self.view and other.view

        return self.__class__(
            byte_size=byte_size,
            large=large,
            view=view,
        )

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
        pl = get_polars()
        return dtype == pl.String

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "StringType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        return cls()

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = get_spark_sql()
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
    def from_dict(cls, value: dict[str, Any]) -> "StringType":
        return cls(
            large=bool(value.get("large", False)),
            view=bool(value.get("view", False)),
            byte_size=value.get("byte_size"),
        )

    def to_arrow(self) -> pa.DataType:
        if self.large:
            return pa.large_string()
        if self.view:
            return pa.string_view()
        return pa.string()

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.String

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        return spark.types.StringType()

    def to_databricks_ddl(self) -> str:
        return "STRING"

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else ""


@dataclass(frozen=True)
class BooleanType(PrimitiveType):

    def __str__(self):
        return "bool"

    def __repr__(self):
        return self.__str__()

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.BOOL

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "BooleanType":
        if not isinstance(other, BooleanType):
            raise TypeError(f"Cannot merge BooleanType with {other.__class__.__name__}")
        if downcast == upcast:
            return self
        return self.__class__(byte_size=self.byte_size or other.byte_size or 1)

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

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else False


@dataclass(frozen=True)
class NumericType(PrimitiveType, ABC):
    pass


_INT_ARROW_SIGNED = {1: pa.int8, 2: pa.int16, 4: pa.int32, 8: pa.int64}
_INT_ARROW_UNSIGNED = {1: pa.uint8, 2: pa.uint16, 4: pa.uint32, 8: pa.uint64}
_INT_DDL_SIGNED = {1: "BYTE", 2: "SHORT", 4: "INT", 8: "BIGINT"}
_INT_DDL_UNSIGNED = {1: "SHORT", 2: "INT", 4: "BIGINT", 8: "DECIMAL(20, 0)"}


def _flip_arrow_int_signedness(dtype: pa.DataType) -> pa.DataType:
    """Return the integer Arrow type with opposite signedness at the same bit width."""
    bytes_ = dtype.bit_width // 8
    mapping = _INT_ARROW_UNSIGNED if pa.types.is_signed_integer(dtype) else _INT_ARROW_SIGNED
    return mapping[bytes_]()


def _polars_flip_int_signedness(dtype: Any) -> Any:
    """Return the polars integer dtype with opposite signedness at the same bit width."""
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


@dataclass(frozen=True)
class IntegerType(NumericType):
    signed: bool = True

    def __str__(self):
        bits = (self.byte_size or 8) * 8
        prefix = "int" if self.signed else "uint"
        return f"{prefix}{bits}"

    def __repr__(self):
        return self.__str__()

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.INTEGER

    # ------------------------------------------------------------------
    # Round-trip safe signed <-> unsigned cast
    # ------------------------------------------------------------------
    # PyArrow's ``pc.cast`` with ``safe=True`` refuses values outside the
    # target's range; with ``safe=False`` it does C-style wraparound, which
    # is bit-preserving across the signed/unsigned flip. For integer-to-
    # integer casts that only differ in signedness (at any width), that
    # wraparound is the round-trip semantic we want: uint64 values above
    # INT64_MAX land as negative int64 via two's complement, and
    # round-tripping back gives the original bits. We force ``safe=False``
    # for this specific case so callers don't have to relax ``options.safe``
    # just to get the flip.
    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.Array:
        options = options.check_source(array)
        src_type = array.type
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
            return options.fill_arrow_nulls(casted)

        return super()._cast_arrow_array(array, options)

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ):
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

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "IntegerType":
        if not isinstance(other, IntegerType):
            raise TypeError(f"Cannot merge IntegerType with {other.__class__.__name__}")
        if downcast and upcast:
            raise pa.ArrowInvalid("Cannot set both downcast=True and upcast=True.")

        left_size = self.byte_size or 8
        right_size = other.byte_size or 8

        if downcast:
            byte_size = min(left_size, right_size)
            signed = self.signed and other.signed
        else:
            byte_size = max(left_size, right_size)
            signed = self.signed or other.signed

        return self.__class__(byte_size=byte_size, signed=signed)

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
        # Spark does not really support uints natively
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

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else 0

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        tags[b"signed"] = b"true" if self.signed else b"false"
        return tags


@dataclass(frozen=True)
class XXHIntType(IntegerType):
    """64-bit integer type carrying ``xxhash`` semantics.

    ``xxh3_64`` (and friends) produce unsigned 64-bit digests, but most
    downstream stores — Spark, Hive/Iceberg, Parquet in the analytics path,
    and every SQL engine that lacks a native ``UNSIGNED BIGINT`` — can only
    carry a signed ``int64``. ``XXHIntType`` captures that intent and makes
    the ``uint64 <-> int64`` flip a **bit-preserving round-trip**: a value
    above ``INT64_MAX`` lands as a negative ``int64`` via two's-complement
    reinterpretation, and casting back returns the original ``uint64`` bits.

    The default physical representation is signed ``int64`` so the value
    survives a trip through a signed-only store. Use
    ``XXHIntType(signed=False)`` when uint64 storage is available (Arrow,
    polars) and the unsigned domain should be exposed directly.
    """

    byte_size: int = 8
    signed: bool = True

    def __str__(self):
        return "xxhint64" if self.signed else "xxhuint64"

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        # XXHIntType is a caller-declared intent, not a type that can be
        # inferred from Arrow alone. Let IntegerType claim int64/uint64.
        return False

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "XXHIntType":
        if pa.types.is_int64(dtype):
            return cls(signed=True)
        if pa.types.is_uint64(dtype):
            return cls(signed=False)
        raise TypeError(
            f"XXHIntType is a 64-bit hash integer; got Arrow type {dtype!r}. "
            "Use XXHIntType(signed=True) for int64-backed hashes, "
            "XXHIntType(signed=False) for uint64-backed hashes, "
            "or IntegerType for generic integers."
        )

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        return False

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return False

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        if not cls._matches_dict(value, DataTypeId.INTEGER, "XXHINT", "XXHINT64"):
            return False
        name = str(value.get("name", "")).upper()
        return name in {"XXHINT", "XXHINT64", "XXHUINT64"}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "XXHIntType":
        name = str(value.get("name", "")).upper()
        signed = value.get("signed")
        if signed is None:
            signed = name != "XXHUINT64"
        return cls(byte_size=8, signed=bool(signed))

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["name"] = "XXHUINT64" if not self.signed else "XXHINT64"
        base["xxhint"] = True
        return base

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        tags[b"xxhint"] = b"true"
        return tags

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "IntegerType":
        # XXHIntType is a tagged int64; merging with another XXHIntType keeps
        # the tag. Merging with a plain IntegerType widens/narrows normally
        # and drops the xxhint intent so the wider schema is still correct.
        if isinstance(other, XXHIntType):
            if downcast == upcast:
                return self
            return XXHIntType(byte_size=8, signed=self.signed and other.signed)
        return super()._merge_with_same_id(
            other, mode=mode, downcast=downcast, upcast=upcast
        )


@dataclass(frozen=True)
class FloatingPointType(NumericType):

    def __str__(self):
        bits = (self.byte_size or 8) * 8
        return f"float{bits}"

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.FLOAT

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
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

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else 0.0


@dataclass(frozen=True)
class DecimalType(NumericType):
    precision: int = 38
    scale: int = 18

    def __str__(self):
        return f"decimal({self.precision}, {self.scale})"

    def __repr__(self):
        return self.__str__()

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
        mode: SaveMode | None = None,
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

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        from decimal import Decimal
        return Decimal(0)

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        tags[b"precision"] = str(self.precision).encode("utf-8")
        tags[b"scale"] = str(self.scale).encode("utf-8")
        return tags


_TEMPORAL_UNIT_ORDER = {
    "d": 0,
    "s": 1,
    "ms": 2,
    "us": 3,
    "ns": 4,
}


def _temporal_cast_module():
    """Lazy handle on the shared temporal cast helpers.

    Kept behind a callable so the import side-effects in ``data/types/`` stay
    linear and the helper module doesn't reach back into us while we're still
    being defined.
    """
    from . import temporal as tc

    return tc


@dataclass(frozen=True)
class TemporalType(PrimitiveType, ABC):
    unit: str = "us"
    tz: str | None = None

    # ------------------------------------------------------------------
    # Vectorized best-effort casting
    # ------------------------------------------------------------------

    def _arrow_target_kind(self) -> str:
        """One-letter tag used by the shared arrow dispatcher."""
        return {
            DataTypeId.DATE: "date",
            DataTypeId.TIME: "time",
            DataTypeId.TIMESTAMP: "timestamp",
            DataTypeId.DURATION: "duration",
        }[self.type_id]

    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.Array:
        options = options.check_source(array)
        tc = _temporal_cast_module()

        array = self._nullify_empty_arrow_strings(array)

        # options.safe=True means strict parsing. Best-effort ingestion
        # (the default) keeps fractional seconds and reinterprets wall-clock
        # when coercing naive timestamps to a target tz.
        best_effort = not options.safe

        kind = self._arrow_target_kind()
        if kind == "date":
            casted = tc.arrow_cast_to_date(array)
        elif kind == "time":
            casted = tc.arrow_cast_to_time(
                array, unit=self.unit, keep_fractional=best_effort
            )
        elif kind == "timestamp":
            casted = tc.arrow_cast_to_timestamp(
                array,
                unit=self.unit,
                tz=self.tz,
                keep_fractional=best_effort,
                unsafe_tz=best_effort,
            )
        else:  # duration
            casted = tc.arrow_cast_to_duration(array, unit=self.unit)

        return options.fill_arrow_nulls(casted)

    def _polars_dtype_instance(self):
        """Force ``to_polars`` into an instance.

        ``pl.Date`` / ``pl.Time`` are classes rather than instances, and Polars
        rejects second-precision on ``Datetime`` / ``Duration`` — normalize
        both quirks here so the cast helpers don't trip on them.
        """
        import dataclasses as _dc

        pl = get_polars()

        target = self
        if self.unit in {"s", "d"} and self.type_id in {
            DataTypeId.TIMESTAMP,
            DataTypeId.DURATION,
        }:
            target = _dc.replace(self, unit="ms")

        dtype = target.to_polars()
        if isinstance(dtype, type) and issubclass(dtype, pl.DataType):
            return dtype()
        return dtype

    def _needs_arrow_bridge(self) -> bool:
        """Polars cannot represent every Arrow temporal unit natively.

        ``Datetime`` / ``Duration`` both reject ``'s'``; ``DateType.unit == 'ms'``
        is also a Polars oddity. In those cases we roundtrip through Arrow so
        the caller still gets a correct value even if Polars can't store it in
        its highest-fidelity form.
        """
        if self.type_id in (DataTypeId.TIMESTAMP, DataTypeId.DURATION):
            return self.unit == "s"
        return False

    def _polars_from_arrow(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ):
        pl = get_polars()
        arrow = series.to_arrow()
        casted = self._cast_arrow_array(arrow, options)
        return pl.Series(name=series.name, values=casted)

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ):
        if self._needs_arrow_bridge():
            casted = self._polars_from_arrow(series, options)
            return self.fill_polars_array_nulls(
                casted, nullable=self._target_nullable(options)
            )

        from .temporal import cast_polars_array_to_temporal

        pl = get_polars()
        source_dtype = series.dtype
        target_dtype = self._polars_dtype_instance()

        casted = cast_polars_array_to_temporal(
            series,
            source=source_dtype,
            target=target_dtype,
            safe=options.safe,
            source_tz=(source_dtype.time_zone if isinstance(source_dtype, pl.Datetime) else None),
            target_tz=self.tz,
        )

        return self.fill_polars_array_nulls(
            casted, nullable=self._target_nullable(options)
        )

    def _cast_polars_expr(
        self,
        expr: Any,
        options: "CastOptions",
    ):
        from .temporal import cast_polars_array_to_temporal

        pl = get_polars()

        if self._needs_arrow_bridge():
            # Expressions can't roundtrip through Arrow cleanly. Fall back to the
            # stock polars cast and let the evaluator best-effort it.
            return expr.cast(self._polars_dtype_instance(), strict=options.safe)

        source_field = options.source_field
        source_dtype = source_field.dtype.to_polars() if source_field is not None else pl.String
        if isinstance(source_dtype, type) and issubclass(source_dtype, pl.DataType):
            source_dtype = source_dtype()
        target_dtype = self._polars_dtype_instance()

        casted = cast_polars_array_to_temporal(
            expr,
            source=source_dtype,
            target=target_dtype,
            safe=options.safe,
            source_tz=getattr(source_dtype, "time_zone", None)
            if isinstance(source_dtype, pl.Datetime)
            else None,
            target_tz=self.tz,
            to_expr=True,
        )

        return self.fill_polars_array_nulls(
            casted, nullable=self._target_nullable(options)
        )

    def _cast_spark_column(
        self,
        column: Any,
        options: "CastOptions",
    ):
        tc = _temporal_cast_module()
        options = options.check_source(column)

        column = self._nullify_empty_spark_strings(column, options)

        best_effort = not options.safe

        kind = self._arrow_target_kind()
        if kind == "date":
            casted = tc.spark_to_date(column)
        elif kind == "timestamp":
            casted = tc.spark_to_timestamp(
                column, unit=self.unit, tz=self.tz, unsafe_tz=best_effort
            )
        elif kind == "time":
            # Spark has no TIME; we land as string per to_spark().
            casted = tc.spark_to_time_string(column)
        else:  # duration — yggdrasil stores it as LongType count of unit
            casted = tc.spark_to_duration_seconds(column, unit=self.unit)

        return self.fill_spark_column_nulls(
            casted, nullable=self._target_nullable(options)
        )

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "TemporalType":
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Cannot merge {self.__class__.__name__} with {other.__class__.__name__}"
            )
        if downcast and upcast:
            raise pa.ArrowInvalid("Cannot set both downcast=True and upcast=True.")

        left_rank = _TEMPORAL_UNIT_ORDER.get(self.unit, 3)
        right_rank = _TEMPORAL_UNIT_ORDER.get(other.unit, 3)

        if downcast:
            unit = self.unit if left_rank <= right_rank else other.unit
            byte_size = min(self.byte_size or 8, other.byte_size or 8)
            tz = self.tz if self.tz == other.tz else None
        else:
            unit = self.unit if left_rank >= right_rank else other.unit
            byte_size = max(self.byte_size or 8, other.byte_size or 8)
            if self.tz == other.tz:
                tz = self.tz
            else:
                tz = self.tz or other.tz

        return self.__class__(
            byte_size=byte_size,
            unit=unit,
            tz=tz,
        )

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        return {
            **base,
            "unit": self.unit,
            "tz": self.tz,
        }

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        if self.unit:
            tags[b"unit"] = self.unit.encode("utf-8")
        if self.tz:
            tags[b"tz"] = self.tz.encode("utf-8")
        return tags


@dataclass(frozen=True)
class DateType(TemporalType):
    unit: str = "d"
    tz: str | None = None

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.DATE

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_date(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "DateType":
        if pa.types.is_date32(dtype):
            return cls(byte_size=4, unit="d")
        if pa.types.is_date64(dtype):
            return cls(byte_size=8, unit="ms")
        raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return dtype == pl.Date

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "DateType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        return cls(byte_size=4, unit="d")

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = get_spark_sql()
        return isinstance(dtype, spark.types.DateType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "DateType":
        if not cls.handles_spark_type(dtype):
            raise TypeError(f"Unsupported Spark data type: {dtype!r}")
        return cls(byte_size=4, unit="d")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.DATE)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DateType":
        return cls(
            byte_size=value.get("byte_size", 4),
            unit=value.get("unit", "d"),
            tz=value.get("tz"),
        )

    def to_arrow(self) -> pa.DataType:
        return pa.date64() if self.unit == "ms" else pa.date32()

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.Date

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        return spark.types.DateType()

    def to_databricks_ddl(self) -> str:
        return "DATE"

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        import datetime as dt
        return dt.date(1970, 1, 1)


@dataclass(frozen=True)
class TimeType(TemporalType):
    unit: str = "us"
    tz: str | None = None

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.TIME

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_time(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "TimeType":
        if not pa.types.is_time(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")
        return cls(byte_size=dtype.bit_width // 8, unit=dtype.unit)

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return dtype == pl.Time

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "TimeType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        return cls(byte_size=8, unit="ns")

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return False

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "TimeType":
        raise TypeError(f"Spark has no native time-only type: {dtype!r}")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.TIME)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TimeType":
        return cls(
            byte_size=value.get("byte_size", 8),
            unit=value.get("unit", "us"),
            tz=value.get("tz"),
        )

    def to_arrow(self) -> pa.DataType:
        if self.unit in {"s", "ms"}:
            return pa.time32(self.unit)
        return pa.time64(self.unit)

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.Time

    def to_spark(self) -> Any:
        # Spark doesn't support time-only
        spark = get_spark_sql()
        return spark.types.StringType()

    def to_databricks_ddl(self) -> str:
        # Databricks doesn't support TIME type natively in SQL DDL
        return "STRING"

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        import datetime as dt
        return dt.time(0, 0, 0)


@dataclass(frozen=True)
class TimestampType(TemporalType):
    unit: str = "us"
    tz: str | None = None

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.TIMESTAMP

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_timestamp(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.TimestampType) -> "TimestampType":
        if not pa.types.is_timestamp(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")
        return cls(byte_size=8, unit=dtype.unit, tz=dtype.tz)

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return isinstance(dtype, pl.Datetime)

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "TimestampType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        unit = getattr(dtype, "time_unit", "us") or "us"
        tz = getattr(dtype, "time_zone", None)
        return cls(byte_size=8, unit=unit, tz=tz)

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = get_spark_sql()
        return isinstance(dtype, (spark.types.TimestampType, spark.types.TimestampNTZType))

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "TimestampType":
        spark = get_spark_sql()
        if isinstance(dtype, spark.types.TimestampType):
            return cls(byte_size=8, unit="us", tz="UTC")
        if isinstance(dtype, spark.types.TimestampNTZType):
            return cls(byte_size=8, unit="us", tz=None)
        raise TypeError(f"Unsupported Spark data type: {dtype!r}")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.TIMESTAMP)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TimestampType":
        return cls(
            byte_size=value.get("byte_size", 8),
            unit=value.get("unit", "us"),
            tz=value.get("tz"),
        )

    def to_arrow(self) -> pa.DataType:
        return pa.timestamp(self.unit, self.tz)

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.Datetime(time_unit=self.unit, time_zone=self.tz)

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        if self.tz is None and hasattr(spark.types, "TimestampNTZType"):
            return spark.types.TimestampNTZType()
        return spark.types.TimestampType()

    def to_databricks_ddl(self) -> str:
        if self.tz is None:
            return "TIMESTAMP_NTZ"
        return "TIMESTAMP"

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        import datetime as dt
        if self.tz:
            return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        return dt.datetime(1970, 1, 1)


@dataclass(frozen=True)
class DurationType(TemporalType):
    unit: str = "us"
    tz: str | None = None

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.DURATION

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_duration(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DurationType) -> "DurationType":
        if not pa.types.is_duration(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")
        return cls(byte_size=8, unit=dtype.unit)

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return isinstance(dtype, pl.Duration)

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "DurationType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        unit = getattr(dtype, "time_unit", "us") or "us"
        return cls(byte_size=8, unit=unit)

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return False

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "DurationType":
        raise TypeError(f"Spark has no native duration type: {dtype!r}")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.DURATION)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DurationType":
        return cls(
            byte_size=value.get("byte_size", 8),
            unit=value.get("unit", "us"),
            tz=value.get("tz"),
        )

    def to_arrow(self) -> pa.DataType:
        return pa.duration(self.unit)

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.Duration(time_unit=self.unit)

    def to_spark(self) -> Any:
        # no native Spark duration
        spark = get_spark_sql()
        return spark.types.LongType()

    def to_databricks_ddl(self) -> str:
        # no native Databricks duration
        return "BIGINT"

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        import datetime as dt
        return dt.timedelta(0)

