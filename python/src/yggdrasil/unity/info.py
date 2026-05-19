"""Info dataclasses — the metadata carried by every :mod:`yggdrasil.unity` resource.

Each :class:`UnityResource` exposes an ``info`` property returning one of
the frozen dataclasses below. ``from_dict`` / ``to_dict`` keep the on-disk
JSON shape canonical so any backend (filesystem, S3, a SQL catalog) can
round-trip the same payload without dragging in a backend-specific type.

The :class:`TableInfo` carries the table's :class:`Schema` (column
:class:`Field` list) and partitioning intent; :class:`ViewInfo` carries a
fully-qualified ``catalog.schema.name`` source reference plus an optional
SQL projection that the backend resolves on read.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Mapping

from yggdrasil.data.enums.media_type import MediaType, MediaTypes
from yggdrasil.data.schema import Schema


__all__ = [
    "CatalogInfo",
    "SchemaInfo",
    "TableInfo",
    "ViewInfo",
]


def _now() -> float:
    return time.time()


def _props(value: "Mapping[str, str] | None") -> dict[str, str]:
    return dict(value) if value else {}


@dataclasses.dataclass(frozen=True)
class CatalogInfo:
    name: str
    comment: str | None = None
    owner: str | None = None
    properties: Mapping[str, str] = dataclasses.field(default_factory=dict)
    created_at: float = dataclasses.field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "catalog",
            "name": self.name,
            "comment": self.comment,
            "owner": self.owner,
            "properties": dict(self.properties),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CatalogInfo":
        return cls(
            name=value["name"],
            comment=value.get("comment"),
            owner=value.get("owner"),
            properties=_props(value.get("properties")),
            created_at=float(value.get("created_at") or _now()),
        )


@dataclasses.dataclass(frozen=True)
class SchemaInfo:
    catalog_name: str
    name: str
    comment: str | None = None
    owner: str | None = None
    properties: Mapping[str, str] = dataclasses.field(default_factory=dict)
    created_at: float = dataclasses.field(default_factory=_now)

    @property
    def full_name(self) -> str:
        return f"{self.catalog_name}.{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "schema",
            "catalog_name": self.catalog_name,
            "name": self.name,
            "comment": self.comment,
            "owner": self.owner,
            "properties": dict(self.properties),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SchemaInfo":
        return cls(
            catalog_name=value["catalog_name"],
            name=value["name"],
            comment=value.get("comment"),
            owner=value.get("owner"),
            properties=_props(value.get("properties")),
            created_at=float(value.get("created_at") or _now()),
        )


@dataclasses.dataclass(frozen=True)
class TableInfo:
    """Stored metadata for a managed table.

    ``schema`` is the canonical :class:`Schema` (column :class:`Field`
    list). ``format`` is a :class:`MediaType` — the wire format the
    backend uses to materialize rows (Parquet / Arrow IPC / …).
    ``partition_by`` enumerates partition columns; ``None`` means
    unpartitioned.
    """

    catalog_name: str
    schema_name: str
    name: str
    schema: Schema
    format: MediaType = MediaTypes.PARQUET
    partition_by: tuple[str, ...] = ()
    comment: str | None = None
    owner: str | None = None
    properties: Mapping[str, str] = dataclasses.field(default_factory=dict)
    created_at: float = dataclasses.field(default_factory=_now)

    @property
    def full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}.{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "table",
            "catalog_name": self.catalog_name,
            "schema_name": self.schema_name,
            "name": self.name,
            "schema": self.schema.to_dict(),
            "format": self.format.full_extension,
            "partition_by": list(self.partition_by),
            "comment": self.comment,
            "owner": self.owner,
            "properties": dict(self.properties),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TableInfo":
        fmt_raw = value.get("format")
        fmt = (
            MediaType.from_(fmt_raw, default=MediaTypes.PARQUET)
            if fmt_raw is not None
            else MediaTypes.PARQUET
        )
        return cls(
            catalog_name=value["catalog_name"],
            schema_name=value["schema_name"],
            name=value["name"],
            schema=Schema.from_dict(value["schema"]),
            format=fmt,
            partition_by=tuple(value.get("partition_by") or ()),
            comment=value.get("comment"),
            owner=value.get("owner"),
            properties=_props(value.get("properties")),
            created_at=float(value.get("created_at") or _now()),
        )


@dataclasses.dataclass(frozen=True)
class ViewInfo:
    """Stored metadata for a view.

    A view is a read-only :class:`Tabular` that resolves to another
    registered table by dotted ``catalog.schema.name``. ``definition``
    optionally carries the SQL projection applied when the view is read;
    when ``None``, the view is a transparent passthrough of its source.
    """

    catalog_name: str
    schema_name: str
    name: str
    source_full_name: str
    definition: str | None = None
    comment: str | None = None
    owner: str | None = None
    properties: Mapping[str, str] = dataclasses.field(default_factory=dict)
    created_at: float = dataclasses.field(default_factory=_now)

    @property
    def full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}.{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "view",
            "catalog_name": self.catalog_name,
            "schema_name": self.schema_name,
            "name": self.name,
            "source_full_name": self.source_full_name,
            "definition": self.definition,
            "comment": self.comment,
            "owner": self.owner,
            "properties": dict(self.properties),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ViewInfo":
        return cls(
            catalog_name=value["catalog_name"],
            schema_name=value["schema_name"],
            name=value["name"],
            source_full_name=value["source_full_name"],
            definition=value.get("definition"),
            comment=value.get("comment"),
            owner=value.get("owner"),
            properties=_props(value.get("properties")),
            created_at=float(value.get("created_at") or _now()),
        )
