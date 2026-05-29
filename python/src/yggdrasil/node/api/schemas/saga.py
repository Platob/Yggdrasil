"""Saga catalog schemas — the catalog/schema/table hierarchy + SQL editor wire
contracts. Unity-Catalog-shaped, but every table is just a reference to a file
on the network filesystem; the bytes never enter the catalog store.
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel

# Table kinds: EXTERNAL points at a pre-existing file; MANAGED is one Saga
# created (and may delete) under its catalog's storage location.
TableType = str  # "EXTERNAL" | "MANAGED"


# -- column + statistics metadata ------------------------------------------

class ColumnSpec(StrictModel):
    name: str
    dtype: str = "string"
    nullable: bool = True
    comment: str = ""


class ColumnStat(StrictModel):
    column: str
    null_count: int | None = None
    distinct_count: int | None = None
    min: Any | None = None
    max: Any | None = None


class TableStatistics(StrictModel):
    row_count: int | None = None
    size_bytes: int | None = None
    columns: list[ColumnStat] = Field(default_factory=list)
    computed_at: str | None = None


# -- catalog ----------------------------------------------------------------

class CatalogCreate(StrictModel):
    name: str
    comment: str = ""
    owner: str = ""
    # Default SQL dialect for the editor when a query doesn't pin one.
    dialect: str | None = None
    # Relative path (under the node files root) where MANAGED tables land.
    storage_location: str | None = None
    properties: dict[str, str] = Field(default_factory=dict)


class CatalogUpdate(StrictModel):
    comment: str | None = None
    owner: str | None = None
    dialect: str | None = None
    storage_location: str | None = None
    properties: dict[str, str] | None = None


class CatalogEntry(StrictModel):
    id: int
    name: str
    comment: str = ""
    owner: str = ""
    dialect: str = "postgres"
    storage_location: str = ""
    node_id: str = ""
    schema_count: int = 0
    properties: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class CatalogResponse(StrictModel):
    catalog: CatalogEntry


class CatalogListResponse(StrictModel):
    node_id: str
    catalogs: list[CatalogEntry]


# -- schema (database) ------------------------------------------------------

class SchemaCreate(StrictModel):
    name: str
    comment: str = ""
    properties: dict[str, str] = Field(default_factory=dict)


class SchemaUpdate(StrictModel):
    comment: str | None = None
    properties: dict[str, str] | None = None


class SchemaEntry(StrictModel):
    id: int
    catalog: str
    name: str
    full_name: str
    comment: str = ""
    table_count: int = 0
    properties: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class SchemaResponse(StrictModel):
    schema_: SchemaEntry = Field(alias="schema")

    model_config = {"populate_by_name": True}


class SchemaListResponse(StrictModel):
    node_id: str
    catalog: str
    schemas: list[SchemaEntry]


# -- table ------------------------------------------------------------------

class TableCreate(StrictModel):
    name: str
    # A filesystem path or npfs:// URL to the backing data file.
    source_url: str
    # Node id that physically holds the bytes (None = local / network fs).
    node: str | None = None
    table_type: TableType = "EXTERNAL"
    format: str | None = None  # inferred from the extension when omitted
    comment: str = ""
    # Skip the schema/statistics scan on register (faster, refresh later).
    infer: bool = True
    columns: list[ColumnSpec] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)


class TableUpdate(StrictModel):
    source_url: str | None = None
    node: str | None = None
    table_type: TableType | None = None
    format: str | None = None
    comment: str | None = None
    columns: list[ColumnSpec] | None = None
    properties: dict[str, str] | None = None


class TableEntry(StrictModel):
    id: int
    catalog: str
    schema_name: str = Field(alias="schema")
    name: str
    full_name: str
    table_type: TableType = "EXTERNAL"
    format: str = ""
    source_url: str = ""
    node: str | None = None
    comment: str = ""
    columns: list[ColumnSpec] = Field(default_factory=list)
    statistics: TableStatistics = Field(default_factory=TableStatistics)
    properties: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str

    model_config = {"populate_by_name": True}


class TableResponse(StrictModel):
    table: TableEntry


class TableListResponse(StrictModel):
    node_id: str
    catalog: str
    schema_: str = Field(alias="schema")
    tables: list[TableEntry]

    model_config = {"populate_by_name": True}


# -- discovery (auto-register a folder of files) ----------------------------

class DiscoverRequest(StrictModel):
    catalog: str
    schema_: str = Field(alias="schema")
    # Folder (under the node files root) to scan for tabular files.
    path: str = ""
    node: str | None = None
    recursive: bool = True
    infer: bool = True

    model_config = {"populate_by_name": True}


# -- SQL editor -------------------------------------------------------------

class SqlRequest(StrictModel):
    sql: str
    dialect: str | None = None
    # Default catalog/schema used to qualify unqualified table references.
    catalog: str | None = None
    schema_: str | None = Field(default=None, alias="schema")
    # Node to run the query on (None = this node). A peer id proxies the run.
    node: str | None = None
    # Row cap for the JSON grid response.
    limit: int | None = None

    model_config = {"populate_by_name": True}


class SqlColumn(StrictModel):
    name: str
    dtype: str


class SqlResult(StrictModel):
    node_id: str
    columns: list[SqlColumn]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    elapsed_ms: float
    # The plan re-emitted as SQL — the "execution plan" view for the editor.
    plan_sql: str = ""
    referenced_tables: list[str] = Field(default_factory=list)


class ExplainResult(StrictModel):
    node_id: str
    dialect: str
    plan: str
    plan_sql: str
    referenced_tables: list[str]
    statement: str
