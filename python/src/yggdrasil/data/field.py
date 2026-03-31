from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from yggdrasil.arrow.cast import any_to_arrow_field
from yggdrasil.data.cast import CastOptions, convert

if TYPE_CHECKING:
    import polars as pl
    import pyspark.sql.types as pst


__all__ = [
    "Field",
    "field",
    "_normalize_metadata",
    "_to_bytes",
    "_merge_metadata_and_tags",
    "_decode_metadata_dict",
]


def _to_bytes(value: object) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _normalize_metadata(
    metadata: dict[bytes | str, bytes | str | object] | None,
    tags: dict[bytes | str, bytes | str | object] | None,
) -> dict[bytes, bytes] | None:
    if not metadata and not tags:
        return None

    normalized = {
        _to_bytes(key): _to_bytes(value)
        for key, value in (metadata or {}).items()
        if key and value is not None
    }

    if tags:
        normalized.update(
            {
                b"t:" + _to_bytes(key): _to_bytes(value)
                for key, value in tags.items()
                if key and value is not None
            }
        )

    return normalized or None


def _merge_metadata_and_tags(
    metadata: dict[bytes, bytes] | None,
    tags: dict[bytes, bytes] | None,
) -> dict[bytes, bytes] | None:
    merged: dict[bytes, bytes] = dict(metadata or {})

    if tags:
        merged.update(
            {
                key if key.startswith(b"t:") else b"t:" + key: value
                for key, value in tags.items()
            }
        )

    return merged or None


def _decode_metadata_dict(
    metadata: dict[bytes, bytes] | None,
) -> dict[str, object]:
    if not metadata:
        return {}

    result: dict[str, object] = {}
    for key, value in metadata.items():
        decoded_key = key.decode("utf-8") if isinstance(key, bytes) else str(key)

        try:
            result[decoded_key] = json.loads(value.decode("utf-8"))
        except Exception:
            result[decoded_key] = value.decode("utf-8", errors="replace")

    return result


def field(
    name: str,
    arrow_type: pa.DataType,
    *,
    nullable: bool = True,
    metadata: dict[bytes | str, bytes | str | object] | None = None,
    tags: dict[bytes | str, bytes | str | object] | None = None,
) -> "Field":
    return Field(
        name=name,
        arrow_type=arrow_type,
        nullable=nullable,
        metadata=_normalize_metadata(metadata, tags=tags),
    )


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    arrow_type: pa.DataType
    nullable: bool = True
    metadata: dict[bytes, bytes] | None = None

    @property
    def partition_by(self) -> bool:
        if self.metadata:
            v = self.metadata.get(b"t:partition_by")
            if v:
                return v.startswith(b"t")
        return False

    @property
    def cluster_by(self) -> bool:
        if self.metadata:
            v = self.metadata.get(b"t:cluster_by")
            if v:
                return v.startswith(b"t")
        return False

    @property
    def tags(self) -> dict[bytes, bytes]:
        if not self.metadata:
            return {}

        return {k[2:]: v for k, v in self.metadata.items() if k.startswith(b"t:")}

    def copy(
        self,
        *,
        name: str | None = None,
        arrow_type: pa.DataType | None = None,
        nullable: bool | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Field":
        return Field(
            name=self.name if name is None else name,
            arrow_type=self.arrow_type if arrow_type is None else arrow_type,
            nullable=self.nullable if nullable is None else nullable,
            metadata=(
                dict(self.metadata)
                if metadata is None and tags is None
                else _normalize_metadata(metadata, tags=tags)
            ),
        )

    def autotag(self) -> "Field":
        inferred: dict[str, object] = {}

        name = self.name.lower()
        dtype = self.arrow_type

        # ---- dtype family tags
        if pa.types.is_boolean(dtype):
            inferred["kind"] = "boolean"
        elif pa.types.is_integer(dtype):
            inferred["kind"] = "integer"
            inferred["numeric"] = True
        elif pa.types.is_floating(dtype):
            inferred["kind"] = "float"
            inferred["numeric"] = True
        elif pa.types.is_decimal(dtype):
            inferred["kind"] = "decimal"
            inferred["numeric"] = True
        elif pa.types.is_timestamp(dtype):
            inferred["kind"] = "timestamp"
            inferred["temporal"] = True
        elif pa.types.is_date(dtype):
            inferred["kind"] = "date"
            inferred["temporal"] = True
        elif pa.types.is_time(dtype):
            inferred["kind"] = "time"
            inferred["temporal"] = True
        elif pa.types.is_duration(dtype):
            inferred["kind"] = "duration"
            inferred["temporal"] = True
        elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
            inferred["kind"] = "string"
        elif pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
            inferred["kind"] = "binary"
        elif (
            pa.types.is_list(dtype)
            or pa.types.is_large_list(dtype)
            or pa.types.is_fixed_size_list(dtype)
        ):
            inferred["kind"] = "list"
            inferred["nested"] = True
        elif pa.types.is_struct(dtype):
            inferred["kind"] = "struct"
            inferred["nested"] = True
        elif pa.types.is_map(dtype):
            inferred["kind"] = "map"
            inferred["nested"] = True
        else:
            inferred["kind"] = str(dtype)

        inferred["nullable"] = self.nullable

        # ---- timezone from Arrow timestamp type
        if pa.types.is_timestamp(dtype):
            unit = getattr(dtype, "unit", None)

            if unit:
                inferred["unit"] = str(unit)

            tz = getattr(dtype, "tz", None)
            if tz:
                inferred["tz"] = tz

        # ---- semantic name tags
        if name == "id" or name.endswith("_id") or name.endswith("id"):
            inferred.setdefault("role", "identifier")

        if (
            name in {"ts", "timestamp"}
            or name.endswith("_ts")
            or name.endswith("_timestamp")
        ):
            inferred["temporal"] = True
            inferred.setdefault("role", "event_time")

        if name == "date" or name.endswith("_date"):
            inferred["temporal"] = True
            inferred.setdefault("role", "date")

        if name == "dt" or name.endswith("_dt"):
            inferred["temporal"] = True

        if "created_at" in name:
            inferred["temporal"] = True
            inferred["role"] = "created_at"

        if "updated_at" in name:
            inferred["temporal"] = True
            inferred["role"] = "updated_at"

        if "deleted_at" in name:
            inferred["temporal"] = True
            inferred["role"] = "deleted_at"

        if name.startswith("is_") or name.startswith("has_") or name.endswith("_flag"):
            inferred.setdefault("role", "flag")

        if "price" in name:
            inferred.setdefault("role", "price")
            inferred["numeric"] = True

        if any(
            token in name
            for token in ("qty", "quantity", "volume", "count", "size", "amount")
        ):
            inferred.setdefault("role", "measure")
            inferred["numeric"] = True

        if any(
            token in name
            for token in ("name", "label", "desc", "description", "comment")
        ):
            inferred.setdefault("role", "attribute")

        if any(
            token in name
            for token in ("country", "region", "area", "zone", "market", "book", "desk")
        ):
            inferred.setdefault("role", "dimension")

        if self.partition_by:
            inferred["partition_by"] = True

        if self.cluster_by:
            inferred["cluster_by"] = True

        # explicit existing tags win
        merged_tags: dict[bytes, bytes] = {
            _to_bytes(key): _to_bytes(value)
            for key, value in inferred.items()
            if key and value is not None
        }
        merged_tags.update(self.tags)

        return Field(
            name=self.name,
            arrow_type=self.arrow_type,
            nullable=self.nullable,
            metadata=_merge_metadata_and_tags(self.metadata, merged_tags),
        )

    @classmethod
    def from_any(cls, obj: Any) -> "Field":
        if isinstance(obj, cls):
            return obj

        return cls.from_arrow(any_to_arrow_field(obj))

    @classmethod
    def from_arrow(cls, value: pa.Field) -> "Field":
        if not isinstance(value, pa.Field):
            value = any_to_arrow_field(value)

        return cls(
            name=value.name,
            arrow_type=value.type,
            nullable=value.nullable,
            metadata=value.metadata,
        )

    def to_arrow_field(self) -> pa.Field:
        return pa.field(
            name=self.name,
            type=self.arrow_type,
            nullable=self.nullable,
            metadata=self.metadata,
        )

    @classmethod
    def from_polars(
        cls,
        obj: Any = None,
        *,
        name: str | None = None,
        dtype: "pl.DataType | None" = None,
        nullable: bool = True,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Field":
        if obj is not None:
            return cls.from_arrow(any_to_arrow_field(obj))

        if name is None or dtype is None:
            raise ValueError("name and dtype are required when obj is not provided")

        from yggdrasil.polars.cast import polars_type_to_arrow_type

        return cls(
            name=name,
            arrow_type=polars_type_to_arrow_type(dtype),
            nullable=nullable,
            metadata=_normalize_metadata(metadata, tags=tags),
        )

    def to_polars_field(self) -> "pl.Field":
        from yggdrasil.polars.cast import arrow_field_to_polars_field

        return arrow_field_to_polars_field(self.to_arrow_field())

    @classmethod
    def from_pyspark(
        cls,
        obj: Any = None,
        *,
        name: str | None = None,
        dtype: "pst.DataType | None" = None,
        nullable: bool = True,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Field":
        if obj is not None:
            from yggdrasil.spark.cast import spark_field_to_arrow_field
            from yggdrasil.spark.lib import pyspark_sql

            if isinstance(obj, pyspark_sql.types.StructField):
                return cls.from_arrow(spark_field_to_arrow_field(obj))
            return cls.from_arrow(any_to_arrow_field(obj))

        if name is None or dtype is None:
            raise ValueError("name and dtype are required when obj is not provided")

        from yggdrasil.spark.cast import spark_type_to_arrow_type

        return cls(
            name=name,
            arrow_type=spark_type_to_arrow_type(dtype),
            nullable=nullable,
            metadata=_normalize_metadata(metadata, tags=tags),
        )

    def to_pyspark_field(self) -> "pst.StructField":
        from yggdrasil.spark.cast import arrow_field_to_spark_field

        return arrow_field_to_spark_field(self.to_arrow_field())

    def cast(self, value: Any, *, safe: bool = True) -> Any:
        options = CastOptions(target_field=self.to_arrow_field(), safe=safe)
        scalar: pa.Scalar = convert(value, target_hint=pa.Scalar, options=options)
        return scalar.as_py()
