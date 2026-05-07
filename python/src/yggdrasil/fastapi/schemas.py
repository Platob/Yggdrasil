"""Pydantic request/response models for JSON-shaped endpoints.

The hot data path is Arrow IPC bytes — these models only describe
the catalog navigation and source-registration surface, where JSON
is the right wire format.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


__all__ = [
    "TableEntry",
    "SchemaListing",
    "CatalogListing",
    "EngineListing",
    "FieldInfo",
    "SchemaInfo",
    "RegisterPathRequest",
    "RegisterInlineRequest",
    "RegisterResult",
    "QueryRequest",
    "InsertRowsRequest",
    "WriteResult",
    "DeleteResult",
]


class TableEntry(BaseModel):
    """One ``catalog.schema.name`` row exposed to the client."""

    catalog: str
    schema_name: str = Field(alias="schema")
    name: str
    qualified_name: str
    tabular_class: str

    model_config = {"populate_by_name": True}


class SchemaListing(BaseModel):
    catalog: str
    schema_name: str = Field(alias="schema")
    tables: list[str]

    model_config = {"populate_by_name": True}


class CatalogListing(BaseModel):
    catalog: str
    schemas: list[str]


class EngineListing(BaseModel):
    catalogs: list[str]
    qualified_names: list[str]


class FieldInfo(BaseModel):
    name: str
    dtype: dict[str, Any] | str
    nullable: bool = True
    metadata: dict[str, str] | None = None


class SchemaInfo(BaseModel):
    catalog: str
    schema_name: str = Field(alias="schema")
    name: str
    fields: list[FieldInfo]

    model_config = {"populate_by_name": True}


class RegisterPathRequest(BaseModel):
    """Register a Tabular pointing at a path / URL.

    The path string is dispatched through :class:`yggdrasil.io.path.Path`,
    so anything :meth:`Path.from_` accepts (local filesystem, ``s3://``,
    ``dbfs://``, …) is fair game. The optional ``media_type`` overrides
    extension-based sniffing when the path doesn't carry one.
    """

    path: str
    media_type: str | None = None


class RegisterInlineRequest(BaseModel):
    """Register a Tabular from inline rows / columns.

    ``rows`` is a list of dicts (one per row); ``columns`` is a dict
    mapping column name → list of values. Pass exactly one — the
    request rejects both at once because there's no obvious merge.
    """

    rows: list[dict[str, Any]] | None = None
    columns: dict[str, list[Any]] | None = None


class RegisterResult(BaseModel):
    catalog: str
    schema_name: str = Field(alias="schema")
    name: str
    qualified_name: str
    rows: int | None = None
    field_count: int

    model_config = {"populate_by_name": True}


class QueryRequest(BaseModel):
    """Builder-style query: optional projection, predicate, paging.

    The ``where`` field accepts any SQL boolean expression
    :func:`yggdrasil.io.tabular.execution.expr.Expression.from_sql`
    parses (``"price > 100 AND region = 'EU'"`` etc.). ``select`` is
    a list of column names. ``limit`` / ``offset`` are applied after
    projection / filtering.
    """

    select: list[str] | None = None
    where: str | None = None
    limit: int | None = None
    offset: int | None = None


class InsertRowsRequest(BaseModel):
    """Inline row / column data for ``POST /data/.../rows``.

    Mirrors :class:`RegisterInlineRequest` — pass exactly one of
    ``rows`` (list of dicts) or ``columns`` (dict of name → list).
    Binary inserts go through the same endpoint with a non-JSON
    content type.
    """

    rows: list[dict[str, Any]] | None = None
    columns: dict[str, list[Any]] | None = None


class WriteResult(BaseModel):
    catalog: str
    schema_name: str = Field(alias="schema")
    name: str
    rows_written: int
    mode: str

    model_config = {"populate_by_name": True}


class DeleteResult(BaseModel):
    catalog: str
    schema_name: str = Field(alias="schema")
    name: str
    rows_deleted: int

    model_config = {"populate_by_name": True}
