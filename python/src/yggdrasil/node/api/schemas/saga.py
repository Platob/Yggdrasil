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
# Asset object kinds — a schema's children are no longer just tables.
# FORECAST is a registered forecasting workflow: a JSON spec in `definition`
# that resolves (live, or from a materialised snapshot) to a history+forecast
# table the plan engine can query like a view.
ObjectType = str  # "TABLE" | "VIEW" | "FUNCTION" | "MODEL" | "FORECAST" | "OTHER"
OBJECT_TYPES = ("TABLE", "VIEW", "FUNCTION", "MODEL", "FORECAST", "OTHER")


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
    # A filesystem path or npfs:// URL to the backing data file (TABLE), or "".
    source_url: str = ""
    object_type: ObjectType = "TABLE"
    # VIEW: the SQL definition. FUNCTION/MODEL: the code/spec.
    definition: str = ""
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
    object_type: ObjectType | None = None
    definition: str | None = None
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
    object_type: ObjectType = "TABLE"
    definition: str = ""
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
    # A filesystem path or npfs:// URL to the backing data file (TABLE).
    source_url: str = ""
    catalog: str = "main"
    schema_: str = Field(default="default", alias="schema")
    # Table name; defaults to the source file's stem.
    table: str | None = None
    object_type: ObjectType = "TABLE"
    definition: str = ""
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


# -- path mounts ------------------------------------------------------------
# A Mount is a named alias for a base path or URL — the same idea as the
# @function feature, but for path objects. ``mount://<alias>/<sub>`` (or just
# ``<alias>/<sub>``) expands to ``<target>/<sub>`` everywhere a source is
# resolved: the SQL engine (``SELECT * FROM 'prod_vol/2024/jan.parquet'``), a
# registered table's ``source_url``, and the file browser. The target is any
# URL the yggdrasil ``Path``/``Tabular`` layer can open — a Databricks volume
# (``/Volumes/cat/sch/vol``), an S3 prefix (``s3://bucket/key``), a remote node
# (``npfs://node-2:8100/data``) or a node-home-relative folder.

class MountCreate(StrictModel):
    name: str                          # the alias (e.g. "prod_vol")
    target: str                        # base path/URL the alias expands to
    comment: str = ""
    read_only: bool = True             # navigation/SQL never mutates unless False
    properties: dict[str, str] = Field(default_factory=dict)


class MountUpdate(StrictModel):
    target: str | None = None
    comment: str | None = None
    read_only: bool | None = None
    properties: dict[str, str] | None = None


class MountEntry(StrictModel):
    id: int
    name: str
    target: str
    comment: str = ""
    read_only: bool = True
    # The path family of the target, sniffed from its scheme/prefix — purely
    # informational for the UI (databricks_volume | s3 | node | local | ...).
    kind: str = "local"
    node_id: str = ""
    properties: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class MountResponse(StrictModel):
    mount: MountEntry


class MountListResponse(StrictModel):
    node_id: str
    mounts: list[MountEntry]


# -- mount navigation (lazy browse through the Path layer) ------------------

class MountNode(StrictModel):
    name: str
    path: str                          # the mount-relative subpath of this child
    is_dir: bool = False
    size: int = 0
    is_tabular: bool = False           # a tabular file → queryable / previewable


class MountListing(StrictModel):
    mount: str
    subpath: str = ""
    target: str                        # the resolved absolute target it expanded to
    entries: list[MountNode]
    truncated: bool = False


# -- forecast workflow ------------------------------------------------------

class ForecastSpec(StrictModel):
    """The persisted definition of a FORECAST asset — everything needed to
    rebuild the history+forecast view anywhere. Serialised to JSON in the
    asset's ``definition`` field, so it rides the same store + replication."""
    source: str                       # registered table name/full_name or a file path
    column: str                       # value column to forecast
    x: str | None = None              # time / order column (else row index)
    keys: list[str] = Field(default_factory=list)  # aggregation keys (per-key series)
    horizon: int = 24                 # steps ahead
    model: str = "auto"               # auto | xgboost | gbr | ridge
    period: int | None = None         # seasonal period (Fourier features)
    agg: str = "mean"                 # collapse duplicate x: mean|sum|last|max|min
    materialized: bool = False        # snapshot to a managed parquet (vs live recompute)


class ForecastRegisterRequest(StrictModel):
    """Register (upsert) a forecast workflow as a catalog asset."""
    catalog: str = "main"
    schema_: str = Field(default="default", alias="schema")
    name: str
    spec: ForecastSpec
    node: str | None = None
    comment: str = ""
    # Materialise immediately (run once, write the snapshot) on register.
    materialize: bool = False

    model_config = {"populate_by_name": True}


class ForecastAssetResult(StrictModel):
    node_id: str
    table: "TableEntry"
    model_used: str
    rmse: float | None = None
    rows: int                         # rows in the materialised view (history + forecast)
    materialized_url: str | None = None
    sampled: bool = False


# -- SQL editor -------------------------------------------------------------

class SqlRequest(StrictModel):
    sql: str
    dialect: str | None = None
    # Default catalog/schema used to qualify unqualified table references.
    catalog: str | None = None
    schema_: str | None = Field(default=None, alias="schema")
    # Node to run the query on (None = this node). A peer id proxies the run.
    node: str | None = None
    # Row cap for the JSON grid response (maps to CastOptions.row_limit).
    limit: int | None = None
    # Arrow IPC chunk size in rows (CastOptions.row_size) for the stream path.
    batch_rows: int | None = None
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


class MaterializeResult(StrictModel):
    node_id: str
    # Node-home-relative path to the result parquet (drives /tabular + /analysis).
    path: str
    columns: list[SqlColumn]
    row_count: int
    elapsed_ms: float


# -- staged result session (lazy windowed display of a heavy result) ---------

class SessionResult(StrictModel):
    node_id: str
    # Node-home-relative Arrow IPC file the session windows read from.
    path: str
    columns: list[SqlColumn]
    row_count: int
    elapsed_ms: float


class SagaFilter(StrictModel):
    column: str
    op: str  # == | != | > | >= | < | <= | contains | in | is_null | not_null
    value: Any | None = None


class WindowTransform(StrictModel):
    op: str  # explode | unnest
    column: str


class WindowRequest(StrictModel):
    path: str
    node: str | None = None
    offset: int = 0
    limit: int = 200
    filters: list[SagaFilter] = Field(default_factory=list)
    sort: str | None = None
    descending: bool = False
    transforms: list[WindowTransform] = Field(default_factory=list)
    columns: list[str] | None = None


class SqlExportRequest(StrictModel):
    sql: str
    dialect: str | None = None
    catalog: str | None = None
    schema_: str | None = Field(default=None, alias="schema")
    node: str | None = None
    # csv | parquet | json | ndjson | arrow | xlsx | tsv | feather
    fmt: str = "csv"
    # Cap the exported rows (None = the whole result).
    max_rows: int | None = None

    model_config = {"populate_by_name": True}


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


# -- search -----------------------------------------------------------------

class SearchHit(StrictModel):
    kind: str  # "catalog" | "schema" | "table"
    name: str
    full_name: str
    object_type: ObjectType = "TABLE"
    catalog: str = ""
    schema_: str = Field(default="", alias="schema")
    comment: str = ""

    model_config = {"populate_by_name": True}


class SearchResponse(StrictModel):
    node_id: str
    query: str
    hits: list[SearchHit]
    total: int
    truncated: bool


# -- activity (asset monitoring dashboard) ----------------------------------

class ActivityResponse(StrictModel):
    node_id: str
    asset: str
    op_counts: dict[str, int]
    total_ops: int
    last_op_at: str | None = None
    # Per-day op counts (most recent last) for a sparkline.
    daily: list[int] = Field(default_factory=list)
    recent: list[OpLogEntry] = Field(default_factory=list)


# -- overview (the catalog-wide monitoring dashboard) -----------------------

class TopAsset(StrictModel):
    """A heaviest/most-active asset row for the overview leaderboards."""
    full_name: str
    object_type: ObjectType = "TABLE"
    catalog: str = ""
    schema_: str = Field(default="", alias="schema")
    rows: int | None = None
    size_bytes: int | None = None
    ops: int = 0
    last_op_at: str | None = None

    model_config = {"populate_by_name": True}


class SagaOverview(StrictModel):
    """One-call rollup of every Saga asset for the management dashboard:
    counts by kind, totals, per-day activity, recent ops across all assets, and
    leaderboards (largest tables, busiest assets)."""
    node_id: str
    catalog_count: int = 0
    schema_count: int = 0
    table_count: int = 0
    view_count: int = 0
    forecast_count: int = 0
    other_count: int = 0
    mount_count: int = 0
    mount_kinds: dict[str, int] = Field(default_factory=dict)
    total_rows: int = 0
    total_bytes: int = 0
    total_ops: int = 0
    op_counts: dict[str, int] = Field(default_factory=dict)
    daily: list[int] = Field(default_factory=list)        # last 14 days, ops/day
    recent: list[OpLogEntry] = Field(default_factory=list)  # newest ops, any asset
    largest: list[TopAsset] = Field(default_factory=list)   # by size_bytes
    busiest: list[TopAsset] = Field(default_factory=list)   # by op count
    mounts: list["MountEntry"] = Field(default_factory=list)
