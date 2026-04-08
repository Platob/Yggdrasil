from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

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


def _to_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bool):
        return b"true" if value else b"false"
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _normalize_metadata(
    metadata: dict[Any, Any] | None,
    tags: dict[Any, Any] | None,
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
                if key and value
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
    def primary_key(self) -> bool:
        if self.metadata:
            v = self.metadata.get(b"t:primary_key")
            if v:
                return v.startswith(b"t")
        return False

    @property
    def foreign_key(self) -> str | None:
        """Dotted FK reference ``catalog.schema.table.column`` stored in
        ``t:foreign_key`` field metadata, or ``None`` when not set."""
        if self.metadata:
            v = self.metadata.get(b"t:foreign_key")
            if v:
                return v.decode("utf-8") if isinstance(v, bytes) else str(v)
        return None

    @property
    def tags(self) -> dict[bytes, bytes]:
        if not self.metadata:
            return {}

        return {k[2:]: v for k, v in self.metadata.items() if k.startswith(b"t:")}

    def is_timestamp(self):
        return pa.types.is_timestamp(self.arrow_type) or pa.types.is_date(self.arrow_type)

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
                (dict(self.metadata) if self.metadata is not None else None)
                if metadata is None and tags is None
                else _normalize_metadata(metadata, tags=tags)
            ),
        )

    def autotag(self) -> "Field":
        inferred: dict[str, object] = {}

        name = self.name.lower()
        dtype = self.arrow_type

        # ---- semantic name tags
        if not self.primary_key:
            if name == "id" and pa.types.is_integer(dtype):
                inferred["primary_key"] = True

        if self.partition_by:
            inferred["partition_by"] = True

        if self.cluster_by:
            inferred["cluster_by"] = True

        if self.primary_key:
            inferred["primary_key"] = True

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
            metadata=_merge_metadata_and_tags(
                self.metadata,
                _normalize_metadata(None, tags=merged_tags)
            ),
        )

    @classmethod
    def from_any(cls, obj: Any) -> "Field":
        if isinstance(obj, cls):
            return obj

        from yggdrasil.arrow.cast import any_to_arrow_field

        return cls.from_arrow(any_to_arrow_field(obj))

    @classmethod
    def from_arrow(cls, value: pa.Field) -> "Field":
        if not isinstance(value, pa.Field):
            from yggdrasil.arrow.cast import any_to_arrow_field
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
            from yggdrasil.arrow.cast import any_to_arrow_field
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
            from pyspark.sql.types import StructField
            from yggdrasil.spark.cast import spark_field_to_arrow_field

            if isinstance(obj, StructField):
                return cls.from_arrow(spark_field_to_arrow_field(obj))

            from yggdrasil.arrow.cast import any_to_arrow_field
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
