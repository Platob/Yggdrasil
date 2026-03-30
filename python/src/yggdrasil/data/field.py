from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    import polars as pl


__all__ = [
    "Field",
    "field",
    "_normalize_metadata",
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
) -> dict[bytes, bytes] | None:
    if not metadata:
        return None

    normalized = {
        (key if isinstance(key, bytes) else str(key).encode("utf-8")): _to_bytes(value)
        for key, value in metadata.items()
        if value is not None
    }
    return normalized or None


def _split_metadata_and_tags(
    metadata: dict[bytes, bytes] | None,
) -> tuple[dict[bytes, bytes] | None, dict[bytes, bytes] | None]:
    if not metadata:
        return None, None

    base: dict[bytes, bytes] = {}
    tags: dict[bytes, bytes] = {}

    for key, value in metadata.items():
        if key.startswith(b"t:"):
            tags[key[2:]] = value
        else:
            base[key] = value

    return base or None, tags or None


def _merge_metadata_and_tags(
    metadata: dict[bytes, bytes] | None,
    tags: dict[bytes, bytes] | None,
) -> dict[bytes, bytes] | None:
    merged: dict[bytes, bytes] = dict(metadata or {})

    if tags:
        merged.update(
            {
                (key if key.startswith(b"t:") else b"t:" + key): value
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
        metadata=_normalize_metadata(metadata),
        tags=_normalize_metadata(tags),
    )


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    arrow_type: pa.DataType
    nullable: bool = True
    metadata: dict[bytes, bytes] | None = None
    tags: dict[bytes, bytes] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))
        object.__setattr__(self, "tags", _normalize_metadata(self.tags))

    @property
    def merged_metadata(self) -> dict[bytes, bytes] | None:
        return _merge_metadata_and_tags(self.metadata, self.tags)

    @property
    def decoded_metadata(self) -> dict[str, object]:
        return _decode_metadata_dict(self.metadata)

    @property
    def decoded_tags(self) -> dict[str, object]:
        return _decode_metadata_dict(self.tags)

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
            metadata=dict(self.metadata) if metadata is None else _normalize_metadata(metadata),
            tags=dict(self.tags) if tags is None else _normalize_metadata(tags),
        )

    @classmethod
    def from_any(cls, obj: Any) -> "Field":
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, pa.Field):
            return cls.from_arrow(obj)

        try:
            import polars as pl

            if isinstance(obj, pl.Field):
                return cls.from_polars(obj)
        except Exception:
            pass

        from yggdrasil.arrow.cast import any_to_arrow_field

        try:
            return cls.from_arrow(any_to_arrow_field(obj))
        except Exception as exc:
            raise TypeError(f"Cannot build Field from {type(obj)!r}") from exc

    @classmethod
    def from_arrow(cls, value: pa.Field) -> "Field":
        if not isinstance(value, pa.Field):
            from yggdrasil.arrow.cast import any_to_arrow_field

            value = any_to_arrow_field(value)

        metadata, tags = _split_metadata_and_tags(value.metadata or None)

        return cls(
            name=value.name,
            arrow_type=value.type,
            nullable=value.nullable,
            metadata=metadata,
            tags=tags,
        )

    def to_arrow_field(self) -> pa.Field:
        return pa.field(
            name=self.name,
            type=self.arrow_type,
            nullable=self.nullable,
            metadata=self.merged_metadata,
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
            metadata=_normalize_metadata(metadata),
            tags=_normalize_metadata(tags),
        )

    def to_polars_field(self) -> "pl.Field":
        from yggdrasil.polars.cast import arrow_field_to_polars_field

        return arrow_field_to_polars_field(self.to_arrow_field())