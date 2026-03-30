from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, MutableMapping
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING

import pyarrow as pa

from .field import Field, _normalize_metadata

if TYPE_CHECKING:
    import polars as pl


__all__ = [
    "Schema",
    "schema",
]


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


def schema(
    fields: Iterable[Field | pa.Field],
    *,
    metadata: dict[bytes | str, bytes | str | object] | None = None,
    tags: dict[bytes | str, bytes | str | object] | None = None,
) -> "Schema":
    return Schema.from_fields(
        fields,
        metadata=metadata,
        tags=tags,
    )


@dataclass
class Schema(MutableMapping[str, Field]):
    inner_fields: OrderedDict[str, Field] = dc_field(default_factory=OrderedDict)
    metadata: dict[bytes, bytes] | None = None
    tags: dict[bytes, bytes] | None = None

    def __post_init__(self) -> None:
        normalized_fields: OrderedDict[str, Field] = OrderedDict()

        for key, value in self.inner_fields.items():
            field = Field.from_any(value)
            if field.name != key:
                field = field.copy(name=key)
            normalized_fields[key] = field

        self.inner_fields = normalized_fields
        self.metadata = _normalize_metadata(self.metadata)
        self.tags = _normalize_metadata(self.tags)

    @property
    def merged_metadata(self) -> dict[bytes, bytes] | None:
        return _merge_metadata_and_tags(self.metadata, self.tags)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.inner_fields.keys())

    @property
    def fields(self) -> tuple[Field, ...]:
        return tuple(self.inner_fields.values())

    def copy(self) -> "Schema":
        return Schema(
            inner_fields=OrderedDict(
                (name, field.copy()) for name, field in self.inner_fields.items()
            ),
            metadata=dict(self.metadata) if self.metadata else None,
            tags=dict(self.tags) if self.tags else None,
        )

    def append(self, *more_fields: Field | pa.Field) -> "Schema":
        out = self.copy()
        for value in more_fields:
            field = Field.from_any(value)
            out[field.name] = field
        return out

    def extend(self, fields: Iterable[Field | pa.Field]) -> "Schema":
        out = self.copy()
        for value in fields:
            field = Field.from_any(value)
            out[field.name] = field
        return out

    def __getitem__(self, key: str) -> Field:
        return self.inner_fields[key]

    def __setitem__(self, key: str, value: Field | pa.Field) -> None:
        field = Field.from_any(value)
        if field.name != key:
            field = field.copy(name=key)
        self.inner_fields[key] = field

    def __delitem__(self, key: str) -> None:
        del self.inner_fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.inner_fields)

    def __len__(self) -> int:
        return len(self.inner_fields)

    def __contains__(self, key: object) -> bool:
        return key in self.inner_fields

    @classmethod
    def from_fields(
        cls,
        fields: Iterable[Field | pa.Field],
        *,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Schema":
        inner_fields: OrderedDict[str, Field] = OrderedDict()

        for value in fields:
            field = Field.from_any(value)
            inner_fields[field.name] = field

        return cls(
            inner_fields=inner_fields,
            metadata=_normalize_metadata(metadata),
            tags=_normalize_metadata(tags),
        )

    @classmethod
    def from_arrow_schema(cls, value: pa.Schema) -> "Schema":
        metadata, tags = _split_metadata_and_tags(dict(value.metadata or {}))

        inner_fields: OrderedDict[str, Field] = OrderedDict()
        for arrow_field in value:
            field = Field.from_arrow(arrow_field)
            inner_fields[field.name] = field

        return cls(
            inner_fields=inner_fields,
            metadata=metadata,
            tags=tags,
        )

    def to_arrow_schema(self) -> pa.Schema:
        return pa.schema(
            fields=[field.to_arrow_field() for field in self.inner_fields.values()],
            metadata=self.merged_metadata,
        )

    @classmethod
    def from_polars(
        cls,
        obj: "pl.Schema | pl.DataFrame | pl.LazyFrame",
        *,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Schema":
        try:
            import polars as pl
        except ImportError as exc:
            raise RuntimeError("polars is required for Schema.from_polars()") from exc

        if isinstance(obj, pl.Schema):
            fields = [
                Field.from_polars(name=name, dtype=dtype)
                for name, dtype in obj.items()
            ]
            return cls.from_fields(fields, metadata=metadata, tags=tags)

        if isinstance(obj, pl.DataFrame):
            fields = [
                Field.from_polars(name=name, dtype=dtype)
                for name, dtype in obj.schema.items()
            ]
            return cls.from_fields(fields, metadata=metadata, tags=tags)

        if isinstance(obj, pl.LazyFrame):
            fields = [
                Field.from_polars(name=name, dtype=dtype)
                for name, dtype in obj.schema.items()
            ]
            return cls.from_fields(fields, metadata=metadata, tags=tags)

        raise TypeError(
            "Expected a polars.Schema, polars.DataFrame, or polars.LazyFrame; "
            f"got {type(obj).__name__}"
        )

    def to_polars_schema(self) -> "pl.Schema":
        try:
            import polars as pl
        except ImportError as exc:
            raise RuntimeError("polars is required for Schema.to_polars_schema()") from exc

        return pl.Schema(
            {
                field.name: field.to_polars_field().dtype
                for field in self.inner_fields.values()
            }
        )