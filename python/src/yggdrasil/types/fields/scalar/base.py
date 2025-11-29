from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional

import pyarrow as pa

from ...libs import pyspark
from ..abstract_field import (
    AbstractField,
    ArrowField,
    PandasField,
    PolarsField,
    PythonField,
    SparkField,
)

__all__ = [
    "AbstractScalarField",
    "PythonScalarField",
    "PandasScalarField",
    "PolarsScalarField",
    "ArrowScalarField",
    "SparkScalarField",
    "metadata_bytes",
    "metadata_str",
]


def _encode_metadata_value(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    return str(value).encode()


def metadata_bytes(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[bytes, bytes]]:
    if not metadata:
        return None
    return {k.encode() if isinstance(k, str) else bytes(k): _encode_metadata_value(v) for k, v in metadata.items()}


def metadata_str(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not metadata:
        return None
    def _to_str(value: Any) -> str:
        if isinstance(value, bytes):
            try:
                return value.decode()
            except Exception:
                return str(value)
        return str(value)

    return {_to_str(k): _to_str(v) for k, v in metadata.items()}


class AbstractScalarField(AbstractField, ABC):
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
        if isinstance(dtype, pa.Field):
            return cls._from_arrow_field(dtype)

        if pyspark is not None and isinstance(dtype, pyspark.sql.types.StructField):
            return cls._from_spark_field(dtype)

        if isinstance(dtype, tuple) and len(dtype) == 2 and isinstance(dtype[0], str):
            name, inner_dtype = dtype  # type: ignore[misc]

            if isinstance(inner_dtype, pa.DataType):
                return cls._from_arrow_components(name, inner_dtype, True, None)

            if pyspark is not None and isinstance(inner_dtype, pyspark.sql.types.DataType):
                spark_struct = pyspark.sql.types.StructField(name, inner_dtype, True)
                return cls._from_spark_field(spark_struct)

        raise TypeError(f"Cannot parse {dtype!r} into {cls.__name__}")

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

    def to_python(self) -> "PythonScalarField":
        raise NotImplementedError

    def to_arrow(self) -> "ArrowScalarField":
        raise NotImplementedError

    def to_spark(self) -> "SparkScalarField":
        raise NotImplementedError

    def to_polars(self) -> "PolarsScalarField":
        raise NotImplementedError

    def to_pandas(self) -> "PandasScalarField":
        raise NotImplementedError

    @staticmethod
    def _from_arrow_components(
        name: str, dtype: pa.DataType, nullable: bool, metadata: Optional[Dict[bytes, bytes]]
    ) -> "AbstractScalarField":
        from .binary_field import BinaryField
        from .date_field import DateField
        from .decimal_field import DecimalField
        from .floating_field import FloatingField
        from .integer_field import IntegerField
        from .string_field import StringField
        from .time_field import TimeField
        from .timestamp_field import TimestampField

        if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
            return StringField(
                name,
                large=pa.types.is_large_string(dtype),
                nullable=nullable,
                metadata=metadata_str(metadata),
            )
        if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
            return BinaryField(
                name,
                large=pa.types.is_large_binary(dtype),
                nullable=nullable,
                metadata=metadata_str(metadata),
            )
        if pa.types.is_integer(dtype):
            bit_width = getattr(dtype, "bit_width", 64)
            return IntegerField(
                name,
                bytesize=bit_width // 8,
                nullable=nullable,
                metadata=metadata_str(metadata),
            )
        if pa.types.is_floating(dtype):
            bit_width = getattr(dtype, "bit_width", 64)
            return FloatingField(
                name,
                bytesize=bit_width // 8,
                nullable=nullable,
                metadata=metadata_str(metadata),
            )
        if pa.types.is_decimal(dtype):
            return DecimalField(
                name,
                precision=dtype.precision,  # type: ignore[attr-defined]
                scale=dtype.scale,  # type: ignore[attr-defined]
                nullable=nullable,
                metadata=metadata_str(metadata),
            )
        if pa.types.is_date(dtype):
            return DateField(name, nullable=nullable, metadata=metadata_str(metadata))
        if pa.types.is_time(dtype):
            return TimeField(
                name,
                unit=getattr(dtype, "unit", "ns"),
                nullable=nullable,
                metadata=metadata_str(metadata),
            )
        if pa.types.is_timestamp(dtype):
            return TimestampField(
                name,
                unit=getattr(dtype, "unit", "ns"),
                tz=getattr(dtype, "tz", None),
                nullable=nullable,
                metadata=metadata_str(metadata),
            )
        raise TypeError(f"Unsupported Arrow scalar type: {dtype!r}")

    @classmethod
    def _from_arrow_field(cls, field: pa.Field) -> "AbstractScalarField":
        return cls._from_arrow_components(field.name, field.type, field.nullable, field.metadata)

    @classmethod
    def _from_spark_field(cls, field: "pyspark.sql.types.StructField") -> "AbstractScalarField":
        if pyspark is None:
            raise ImportError("pyspark is required to parse Spark fields")

        from .binary_field import BinaryField
        from .date_field import DateField
        from .decimal_field import DecimalField
        from .floating_field import FloatingField
        from .integer_field import IntegerField
        from .string_field import StringField
        from .timestamp_field import TimestampField

        dtype = field.dataType
        if isinstance(dtype, pyspark.sql.types.StringType):
            return StringField(field.name, nullable=field.nullable, metadata=field.metadata)
        if isinstance(dtype, pyspark.sql.types.BinaryType):
            return BinaryField(field.name, nullable=field.nullable, metadata=field.metadata)
        if isinstance(dtype, pyspark.sql.types.ByteType):
            return IntegerField(field.name, bytesize=1, nullable=field.nullable, metadata=field.metadata)
        if isinstance(dtype, pyspark.sql.types.ShortType):
            return IntegerField(field.name, bytesize=2, nullable=field.nullable, metadata=field.metadata)
        if isinstance(dtype, pyspark.sql.types.IntegerType):
            return IntegerField(field.name, bytesize=4, nullable=field.nullable, metadata=field.metadata)
        if isinstance(dtype, pyspark.sql.types.LongType):
            return IntegerField(field.name, bytesize=8, nullable=field.nullable, metadata=field.metadata)
        if isinstance(dtype, pyspark.sql.types.FloatType):
            return FloatingField(field.name, bytesize=4, nullable=field.nullable, metadata=field.metadata)
        if isinstance(dtype, pyspark.sql.types.DoubleType):
            return FloatingField(field.name, bytesize=8, nullable=field.nullable, metadata=field.metadata)
        if isinstance(dtype, pyspark.sql.types.DecimalType):
            return DecimalField(
                field.name,
                precision=dtype.precision,
                scale=dtype.scale,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        if isinstance(dtype, pyspark.sql.types.DateType):
            return DateField(field.name, nullable=field.nullable, metadata=field.metadata)
        if isinstance(dtype, pyspark.sql.types.TimestampType):
            return TimestampField(field.name, nullable=field.nullable, metadata=field.metadata)

        raise TypeError(f"Unsupported Spark scalar type: {dtype!r}")


class PythonScalarField(PythonField, ABC):
    def to_python(self) -> "PythonScalarField":
        return self


class PandasScalarField(PandasField, ABC):
    def to_pandas(self) -> "PandasScalarField":
        return self


class PolarsScalarField(PolarsField, ABC):
    def to_polars(self) -> "PolarsScalarField":
        return self


class ArrowScalarField(ArrowField, ABC):
    def to_arrow(self) -> "ArrowScalarField":
        return self


class SparkScalarField(SparkField, ABC):
    def to_spark(self) -> "SparkScalarField":
        return self
