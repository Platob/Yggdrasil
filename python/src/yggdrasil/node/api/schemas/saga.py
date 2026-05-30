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
    # Node ids that hold a local replica of this table's data.
    replicas: list[str] = Field(default_factory=list)
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


# -- one-shot register (ensure catalog + schema, infer name/format) ---------

class RegisterRequest(StrictModel):
    # A filesystem path or npfs:// URL to the backing data file.
    source_url: str
    catalog: str = "main"
    schema_: str = Field(default="default", alias="schema")
    # Table name; defaults to the source file's stem.
    table: str | None = None
    node: str | None = None
    table_type: TableType = "EXTERNAL"
    # Dialect for the catalog if it has to be created.
    dialect: str | None = None

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
    # When set (an npfs:// NodePath URL), the executing node writes the Arrow
    # result there instead of streaming it back — lets a remote node stage the
    # output near whoever asked for it.
    staging_path: str | None = None

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


class StagedResult(StrictModel):
    node_id: str
    staging_path: str
    columns: list[SqlColumn]
    row_count: int
    bytes: int
    elapsed_ms: float


# -- execution plan graph (DAG + analyze timings) ---------------------------

class PlanOp(StrictModel):
    id: str
    # scan | join | union | filter | aggregate | having | distinct | project | sort | limit
    op: str
    title: str
    detail: str = ""
    inputs: list[str] = Field(default_factory=list)
    # Filled by analyze: rows out of this op + its measured time.
    rows: int | None = None
    elapsed_ms: float | None = None


class PlanGraph(StrictModel):
    node_id: str
    dialect: str
    statement: str
    plan_sql: str
    ops: list[PlanOp]
    analyzed: bool = False
    total_ms: float | None = None
    sampled: bool = False


class PlanEdit(StrictModel):
    # set_limit | set_offset | drop_filter | drop_group | drop_order | drop_limit | drop_distinct
    op: str
    value: int | None = None


class PlanEditRequest(StrictModel):
    sql: str
    dialect: str | None = None
    catalog: str | None = None
    schema_: str | None = Field(default=None, alias="schema")
    edits: list[PlanEdit] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class PlanEditResult(StrictModel):
    node_id: str
    sql: str
    plan_sql: str


# -- replication ------------------------------------------------------------

class ReplicateRequest(StrictModel):
    catalog: str
    schema_: str = Field(alias="schema")
    table: str
    # Target node id to replicate to.
    target: str
    # "metadata": register the table on the target (shared fs / same source).
    # "data": copy the data file to the target's staging, then register it there.
    mode: str = "data"

    model_config = {"populate_by_name": True}


class TablePayload(StrictModel):
    """Self-contained table registration for cross-node import."""
    catalog: str
    schema_: str = Field(alias="schema")
    table: TableEntry
    catalog_dialect: str = "postgres"

    model_config = {"populate_by_name": True}


class ReplicateResult(StrictModel):
    source_node: str
    target_node: str
    full_name: str
    mode: str
    bytes_copied: int = 0
    target_source_url: str = ""


# -- operation log ----------------------------------------------------------

class OpLogEntry(StrictModel):
    ts: str
    op: str
    user: str = ""
    node: str = ""
    statement: str = ""
    rows: int | None = None
    detail: str = ""


class OpLogResponse(StrictModel):
    node_id: str
    asset: str
    entries: list[OpLogEntry]
