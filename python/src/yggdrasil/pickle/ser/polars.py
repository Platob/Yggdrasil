from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO as StdBytesIO
from typing import ClassVar, Generic, TypeVar

from yggdrasil.polars.lib import polars as pl
import pyarrow as pa

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

from .pyarrow import (
    _merge_metadata,
    _table_from_ipc_file_buffer,
    _table_to_ipc_file_buffer,
)

__all__ = [
    "TPolars",
    "PolarsSerialized",
    "PolarsDataFrameSerialized",
    "PolarsSeriesSerialized",
    "PolarsLazyFrameSerialized",
    "PolarsExprSerialized",
    "PolarsSchemaSerialized",
    "PolarsDataTypeSerialized",
]

TPolars = TypeVar("TPolars", bound=object)

_SENTINEL_FIELD_NAME = "__ygg_value__"


def _dataframe_to_arrow_table(df: pl.DataFrame) -> pa.Table:
    return df.to_arrow()


def _series_to_arrow_table(series: pl.Series) -> pa.Table:
    name = series.name if series.name is not None else _SENTINEL_FIELD_NAME
    return series.to_frame(name=name).to_arrow()


def _arrow_table_to_dataframe(table: pa.Table) -> pl.DataFrame:
    return pl.from_arrow(table)


def _arrow_table_to_series(table: pa.Table) -> pl.Series:
    df = pl.from_arrow(table)
    if df.width != 1:
        raise ValueError(
            f"POLARS_SERIES payload must contain exactly 1 column, got {df.width}"
        )
    return df.to_series(0)


def _simple_dtype_name(dtype: object) -> str | None:
    simple_names = (
        "Null",
        "Boolean",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Int128",
        "Float32",
        "Float64",
        "String",
        "Binary",
        "Date",
        "Time",
        "Object",
        "Unknown",
        "Categorical",
    )

    for name in simple_names:
        candidate = getattr(pl, name, None)
        if candidate is not None and dtype == candidate:
            return name
    return None


def _simple_dtype_from_name(name: str) -> pl.DataType | None:
    candidate = getattr(pl, name, None)
    if candidate is None:
        return None
    return candidate


def _dtype_to_python_payload(dtype: pl.DataType | type[pl.DataType]) -> object:
    if isinstance(dtype, type):
        return {"kind": "class", "name": dtype.__name__}

    simple_name = _simple_dtype_name(dtype)
    if simple_name is not None:
        return {"kind": "class", "name": simple_name}

    if isinstance(dtype, pl.List):
        return {
            "kind": "list",
            "inner": _dtype_to_python_payload(dtype.inner),
        }

    if isinstance(dtype, pl.Array):
        return {
            "kind": "array",
            "inner": _dtype_to_python_payload(dtype.inner),
            "shape": list(dtype.shape),
        }

    if isinstance(dtype, pl.Struct):
        return {
            "kind": "struct",
            "fields": [
                {
                    "name": field.name,
                    "dtype": _dtype_to_python_payload(field.dtype),
                }
                for field in dtype.fields
            ],
        }

    if isinstance(dtype, pl.Datetime):
        return {
            "kind": "datetime",
            "time_unit": dtype.time_unit,
            "time_zone": dtype.time_zone,
        }

    if isinstance(dtype, pl.Duration):
        return {
            "kind": "duration",
            "time_unit": dtype.time_unit,
        }

    if isinstance(dtype, pl.Decimal):
        return {
            "kind": "decimal",
            "precision": dtype.precision,
            "scale": dtype.scale,
        }

    if isinstance(dtype, pl.Enum):
        categories = getattr(dtype, "categories", None)
        return {
            "kind": "enum",
            "categories": list(categories) if categories is not None else None,
        }

    return {
        "kind": "string_repr",
        "value": str(dtype),
    }


def _dtype_from_python_payload(payload: object) -> pl.DataType:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict fallback payload for DataType, got {type(payload)!r}")

    kind = payload.get("kind")
    if not isinstance(kind, str):
        raise TypeError("DataType payload missing string field 'kind'")

    if kind == "class":
        name = payload.get("name")
        if not isinstance(name, str):
            raise TypeError("DataType payload missing string field 'name'")
        dtype = _simple_dtype_from_name(name)
        if dtype is None:
            raise ValueError(f"Unsupported simple Polars dtype: {name!r}")
        return dtype

    if kind == "list":
        return pl.List(_dtype_from_python_payload(payload["inner"]))

    if kind == "array":
        inner = _dtype_from_python_payload(payload["inner"])
        shape = payload.get("shape")
        if isinstance(shape, list):
            shape_tuple = tuple(int(x) for x in shape)
            if len(shape_tuple) == 1:
                return pl.Array(inner, shape_tuple[0])
            return pl.Array(inner, shape_tuple)
        raise TypeError("Array dtype payload missing list field 'shape'")

    if kind == "struct":
        raw_fields = payload.get("fields")
        if not isinstance(raw_fields, list):
            raise TypeError("Struct dtype payload missing list field 'fields'")
        return pl.Struct(
            [
                pl.Field(
                    field["name"],
                    _dtype_from_python_payload(field["dtype"]),
                )
                for field in raw_fields
            ]
        )

    if kind == "datetime":
        return pl.Datetime(
            time_unit=payload.get("time_unit"),
            time_zone=payload.get("time_zone"),
        )

    if kind == "duration":
        return pl.Duration(
            time_unit=payload.get("time_unit"),
        )

    if kind == "decimal":
        return pl.Decimal(
            precision=payload.get("precision"),
            scale=payload.get("scale"),
        )

    if kind == "enum":
        categories = payload.get("categories")
        if categories is None:
            raise ValueError("Enum dtype payload missing categories")
        if not isinstance(categories, list):
            raise TypeError("Enum dtype payload field 'categories' must be a list")
        return pl.Enum(categories)

    if kind == "string_repr":
        value = payload.get("value")
        if not isinstance(value, str):
            raise TypeError("DataType payload missing string field 'value'")

        # minimal compatibility escape hatch
        simple = _simple_dtype_from_name(value)
        if simple is not None:
            return simple

        raise ValueError(f"Unsupported Polars dtype string representation: {value!r}")

    raise ValueError(f"Unsupported Polars dtype payload kind: {kind!r}")


def _coerce_dtype(obj: object) -> pl.DataType | type[pl.DataType] | None:
    if isinstance(obj, type):
        module = getattr(obj, "__module__", "")
        if module.startswith("polars"):
            return obj

    if isinstance(obj, str):
        return _simple_dtype_from_name(obj)

    try:
        payload = _dtype_to_python_payload(obj)  # type: ignore[arg-type]
        _ = _dtype_from_python_payload(payload)
        return obj  # type: ignore[return-value]
    except Exception:
        return None


def _dataframe_to_python_payload(df: pl.DataFrame) -> dict[str, object]:
    return {
        "columns": [
            {
                "name": name,
                "values": df.get_column(name).to_list(),
            }
            for name in df.columns
        ],
        "schema": [
            {
                "name": name,
                "dtype": _dtype_to_python_payload(dtype),
            }
            for name, dtype in df.schema.items()
        ],
    }


def _dataframe_from_python_payload(payload: object) -> pl.DataFrame:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict fallback payload for DataFrame, got {type(payload)!r}")

    column_entries = payload.get("columns")
    schema_entries = payload.get("schema")

    if not isinstance(column_entries, list):
        raise TypeError("DataFrame fallback payload missing list 'columns'")

    data: dict[str, list[object]] = {}
    column_order: list[str] = []

    for entry in column_entries:
        if not isinstance(entry, dict):
            raise TypeError("DataFrame fallback payload column entry must be a dict")
        if "name" not in entry or "values" not in entry:
            raise TypeError(
                "DataFrame fallback payload column entry must contain 'name' and 'values'"
            )

        name = entry["name"]
        values = entry["values"]

        if not isinstance(name, str):
            raise TypeError("DataFrame fallback payload column 'name' must be a str")
        if not isinstance(values, list):
            raise TypeError("DataFrame fallback payload column 'values' must be a list")

        data[name] = values
        column_order.append(name)

    schema: dict[str, pl.DataType] | None = None
    if isinstance(schema_entries, list):
        maybe_schema: dict[str, pl.DataType] = {}
        try:
            for entry in schema_entries:
                if not isinstance(entry, dict):
                    raise TypeError("DataFrame fallback schema entry must be a dict")

                name = entry.get("name")
                dtype_payload = entry.get("dtype")

                if not isinstance(name, str):
                    raise TypeError("DataFrame fallback schema field 'name' must be a str")

                maybe_schema[name] = _dtype_from_python_payload(dtype_payload)
            schema = maybe_schema
        except Exception:
            schema = None

    if schema:
        try:
            return pl.DataFrame(data, schema=schema).select(column_order)
        except Exception:
            pass

    return pl.DataFrame(data).select(column_order)


def _series_to_python_payload(series: pl.Series) -> dict[str, object]:
    return {
        "name": series.name,
        "values": series.to_list(),
        "dtype": _dtype_to_python_payload(series.dtype),
    }


def _series_from_python_payload(payload: object) -> pl.Series:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict fallback payload for Series, got {type(payload)!r}")

    name = payload.get("name")
    values = payload.get("values")
    dtype_payload = payload.get("dtype")

    if name is not None and not isinstance(name, str):
        raise TypeError("Series fallback payload field 'name' must be a str or None")
    if not isinstance(values, list):
        raise TypeError("Series fallback payload missing list 'values'")

    kwargs: dict[str, object] = {"name": name, "values": values}

    if dtype_payload is not None:
        try:
            kwargs["dtype"] = _dtype_from_python_payload(dtype_payload)
        except Exception:
            pass

    try:
        return pl.Series(**kwargs)
    except Exception:
        return pl.Series(name=name, values=values)


def _lazyframe_to_bytes(lf: pl.LazyFrame) -> bytes:
    data = lf.serialize(format="binary")
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


def _lazyframe_from_bytes(data: bytes) -> pl.LazyFrame:
    return pl.LazyFrame.deserialize(StdBytesIO(data), format="binary")


def _expr_to_bytes(expr: pl.Expr) -> bytes:
    data = expr.meta.serialize(format="binary")
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


def _expr_from_bytes(data: bytes) -> pl.Expr:
    return pl.Expr.deserialize(StdBytesIO(data), format="binary")


def _schema_items_from_object(obj: object) -> list[tuple[str, pl.DataType | type[pl.DataType]]] | None:
    if isinstance(obj, pl.Schema):
        return list(obj.items())

    if isinstance(obj, Mapping):
        items: list[tuple[str, pl.DataType | type[pl.DataType]]] = []
        for key, value in obj.items():
            if not isinstance(key, str):
                return None
            dtype = _coerce_dtype(value)
            if dtype is None:
                return None
            items.append((key, dtype))
        return items

    return None


def _schema_to_python_payload(schema: object) -> dict[str, object]:
    items = _schema_items_from_object(schema)
    if items is None:
        raise TypeError(f"Unsupported schema object: {type(schema)!r}")

    return {
        "items": [
            {
                "name": name,
                "dtype": _dtype_to_python_payload(dtype),
            }
            for name, dtype in items
        ]
    }


def _schema_from_python_payload(payload: object) -> pl.Schema:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict fallback payload for Schema, got {type(payload)!r}")

    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise TypeError("Schema fallback payload missing list 'items'")

    items: list[tuple[str, pl.DataType]] = []
    for entry in raw_items:
        if not isinstance(entry, dict):
            raise TypeError("Schema fallback payload entry must be a dict")

        name = entry.get("name")
        dtype_payload = entry.get("dtype")

        if not isinstance(name, str):
            raise TypeError("Schema fallback payload field 'name' must be a str")

        dtype = _dtype_from_python_payload(dtype_payload)
        items.append((name, dtype))

    return pl.Schema(items)


@dataclass(frozen=True, slots=True)
class PolarsSerialized(Serialized[TPolars], Generic[TPolars]):
    TAG: ClassVar[int]

    @property
    def value(self) -> TPolars:
        raise NotImplementedError

    def as_python(self) -> TPolars:
        return self.value

    def decode_arrow_buffer(self) -> pa.Buffer:
        return pa.py_buffer(self.decode())

    @staticmethod
    def _deserialize_nested_payload(data: bytes) -> object:
        nested = Serialized.read_from(BytesIO(data), pos=0)
        return nested.as_python()

    @staticmethod
    def _serialize_nested_payload(
        payload: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        return Serialized.from_python_object(
            payload,
            metadata=metadata,
            codec=codec,
        )

    @staticmethod
    def _nested_serialized_bytes(nested: Serialized[object]) -> bytes:
        buf = BytesIO()
        nested.write_to(buf)
        return buf.to_bytes()

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        if isinstance(obj, pl.DataFrame):
            return PolarsDataFrameSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pl.Series):
            return PolarsSeriesSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pl.LazyFrame):
            return PolarsLazyFrameSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pl.Expr):
            return PolarsExprSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pl.Schema) or isinstance(obj, Mapping):
            items = _schema_items_from_object(obj)
            if items is not None:
                return PolarsSchemaSerialized.from_value(
                    obj,
                    metadata=metadata,
                    codec=codec,
                )

        dtype = _coerce_dtype(obj)
        if dtype is not None:
            return PolarsDataTypeSerialized.from_value(
                dtype,
                metadata=metadata,
                codec=codec,
            )

        return None


@dataclass(frozen=True, slots=True)
class PolarsDataFrameSerialized(PolarsSerialized[pl.DataFrame]):
    TAG: ClassVar[int] = Tags.POLARS_DATAFRAME

    @property
    def value(self) -> pl.DataFrame:
        metadata = self.metadata or {}
        serialization_format = metadata.get(b"serialization_format", b"arrow")

        if serialization_format == b"python_serialized":
            payload = self._deserialize_nested_payload(self.decode())
            return _dataframe_from_python_payload(payload)

        table = _table_from_ipc_file_buffer(self.decode_arrow_buffer())
        return _arrow_table_to_dataframe(table)

    @classmethod
    def from_value(
        cls,
        df: pl.DataFrame,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        try:
            table = _dataframe_to_arrow_table(df)
            buf = _table_to_ipc_file_buffer(table, metadata=None)
            return cls.build(
                tag=cls.TAG,
                data=buf,
                metadata=_merge_metadata(metadata, {b"serialization_format": b"arrow"}),
                codec=codec,
            )
        except Exception:
            payload = _dataframe_to_python_payload(df)
            nested = cls._serialize_nested_payload(
                payload,
                metadata=None,
                codec=codec,
            )
            return cls.build(
                tag=cls.TAG,
                data=cls._nested_serialized_bytes(nested),
                metadata=_merge_metadata(
                    metadata,
                    {b"serialization_format": b"python_serialized"},
                ),
                codec=codec,
            )


@dataclass(frozen=True, slots=True)
class PolarsSeriesSerialized(PolarsSerialized[pl.Series]):
    TAG: ClassVar[int] = Tags.POLARS_SERIES

    @property
    def value(self) -> pl.Series:
        metadata = self.metadata or {}
        serialization_format = metadata.get(b"serialization_format", b"arrow")

        if serialization_format == b"python_serialized":
            payload = self._deserialize_nested_payload(self.decode())
            return _series_from_python_payload(payload)

        table = _table_from_ipc_file_buffer(self.decode_arrow_buffer())
        return _arrow_table_to_series(table)

    @classmethod
    def from_value(
        cls,
        series: pl.Series,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        try:
            table = _series_to_arrow_table(series)
            buf = _table_to_ipc_file_buffer(table, metadata=None)
            return cls.build(
                tag=cls.TAG,
                data=buf,
                metadata=_merge_metadata(metadata, {b"serialization_format": b"arrow"}),
                codec=codec,
            )
        except Exception:
            payload = _series_to_python_payload(series)
            nested = cls._serialize_nested_payload(
                payload,
                metadata=None,
                codec=codec,
            )
            return cls.build(
                tag=cls.TAG,
                data=cls._nested_serialized_bytes(nested),
                metadata=_merge_metadata(
                    metadata,
                    {b"serialization_format": b"python_serialized"},
                ),
                codec=codec,
            )


@dataclass(frozen=True, slots=True)
class PolarsLazyFrameSerialized(PolarsSerialized[pl.LazyFrame]):
    TAG: ClassVar[int] = Tags.POLARS_LAZYFRAME

    @property
    def value(self) -> pl.LazyFrame:
        return _lazyframe_from_bytes(self.decode())

    @classmethod
    def from_value(
        cls,
        lf: pl.LazyFrame,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        payload = _lazyframe_to_bytes(lf)
        return cls.build(
            tag=cls.TAG,
            data=payload,
            metadata=_merge_metadata(
                metadata,
                {b"serialization_format": b"polars_binary"},
            ),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class PolarsExprSerialized(PolarsSerialized[pl.Expr]):
    TAG: ClassVar[int] = Tags.POLARS_EXPR

    @property
    def value(self) -> pl.Expr:
        return _expr_from_bytes(self.decode())

    @classmethod
    def from_value(
        cls,
        expr: pl.Expr,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        payload = _expr_to_bytes(expr)
        return cls.build(
            tag=cls.TAG,
            data=payload,
            metadata=_merge_metadata(
                metadata,
                {b"serialization_format": b"polars_binary"},
            ),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class PolarsSchemaSerialized(PolarsSerialized[pl.Schema]):
    TAG: ClassVar[int] = Tags.POLARS_SCHEMA

    @property
    def value(self) -> pl.Schema:
        payload = self._deserialize_nested_payload(self.decode())
        return _schema_from_python_payload(payload)

    @classmethod
    def from_value(
        cls,
        schema: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        payload = _schema_to_python_payload(schema)
        nested = cls._serialize_nested_payload(
            payload,
            metadata=None,
            codec=codec,
        )
        return cls.build(
            tag=cls.TAG,
            data=cls._nested_serialized_bytes(nested),
            metadata=_merge_metadata(
                metadata,
                {b"serialization_format": b"python_serialized"},
            ),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class PolarsDataTypeSerialized(
    PolarsSerialized[pl.DataType | type[pl.DataType]]
):
    TAG: ClassVar[int] = Tags.POLARS_DATATYPE

    @property
    def value(self) -> pl.DataType:
        payload = self._deserialize_nested_payload(self.decode())
        return _dtype_from_python_payload(payload)

    @classmethod
    def from_value(
        cls,
        dtype: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        coerced = _coerce_dtype(dtype)
        if coerced is None:
            raise TypeError(f"Unsupported Polars dtype object: {type(dtype)!r}")

        nested = cls._serialize_nested_payload(
            _dtype_to_python_payload(coerced),
            metadata=None,
            codec=codec,
        )
        return cls.build(
            tag=cls.TAG,
            data=cls._nested_serialized_bytes(nested),
            metadata=_merge_metadata(
                metadata,
                {b"serialization_format": b"python_serialized"},
            ),
            codec=codec,
        )


for cls in PolarsSerialized.__subclasses__():
    Tags.register_class(cls, tag=cls.TAG)

PolarsDataFrameSerialized = Tags.get_class(Tags.POLARS_DATAFRAME) or PolarsDataFrameSerialized
PolarsSeriesSerialized = Tags.get_class(Tags.POLARS_SERIES) or PolarsSeriesSerialized
PolarsLazyFrameSerialized = Tags.get_class(Tags.POLARS_LAZYFRAME) or PolarsLazyFrameSerialized
PolarsExprSerialized = Tags.get_class(Tags.POLARS_EXPR) or PolarsExprSerialized
PolarsSchemaSerialized = Tags.get_class(Tags.POLARS_SCHEMA) or PolarsSchemaSerialized
PolarsDataTypeSerialized = Tags.get_class(Tags.POLARS_DATATYPE) or PolarsDataTypeSerialized

