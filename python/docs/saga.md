# Saga — distributed data catalog

Saga is Yggdrasil's equivalent of Unity Catalog: a network-wide registry of
data sources organised as a **catalog → schema → table** hierarchy, plus a SQL
editor that executes queries over registered tables and raw filesystem URLs.

It is a thin, metadata-only layer on top of machinery that already exists in
the repo:

| Saga needs | Reuses |
|------------|--------|
| Parse SQL (postgres + other dialects) | `yggdrasil.plan.sql_parser.parse_sql` |
| Execution plan over named tables / URLs | `yggdrasil.plan.execute.execute_plan` (already resolves `catalog.schema.table` keys **and** auto-resolves file URLs, with predicate/projection pushdown) |
| Lazy, pushdown, streaming scans | `LazyTabular` + polars lazy (`AnalysisService`) |
| Zero-copy result wire | `transport` Arrow IPC stream |
| Disk spill when heavy | `ArrowTabular` auto-spill + node `spill_root`/`tmp` |
| Cross-node data + compute | `NetworkService` peer mesh + `?node=` proxy |

## Concepts

- **Catalog** — top-level namespace. Owns a default storage location (a path
  under the node files root) for *managed* tables and a default SQL dialect.
- **Schema** — a database inside a catalog (`catalog.schema`).
- **Table** — registers one data source (`catalog.schema.table`):
  - `source_url` — a file path or `npfs://node/path` URL (parquet/csv/ndjson/
    json/arrow/xlsx). The bytes live on the filesystem (network fs by default),
    never copied into the catalog.
  - `table_type` — `EXTERNAL` (points at an existing file) or `MANAGED` (Saga
    created it under the catalog storage location).
  - `columns` — schema metadata (name, dtype, nullable, comment).
  - `statistics` — `row_count`, `size_bytes`, per-column null/min/max/distinct,
    refreshed on demand.
  - `node` — which node holds the bytes (`None` = local / network fs).

Identifiers are int64 via `make_id(full_name)` — deterministic per name, so the
same `catalog.schema.table` gets the same id on every node.

## Persistence

Metadata is a single JSON document at `~/.node/{id}/saga/store.json`
(`{catalogs, schemas, tables}`), loaded on startup and written atomically
(temp + rename) under a lock on every mutation. The *data* is never stored
here — only references to filesystem locations.

## SQL execution

```
POST /api/v2/saga/sql         → bounded JSON grid (editor)
POST /api/v2/saga/sql.arrow   → Arrow IPC stream (zero-copy; disk-spilled when heavy)
POST /api/v2/saga/explain     → parsed plan + emitted SQL + referenced tables (no run)
```

Pipeline:

1. `parse_sql(sql, dialect)` → immutable `PlanNode` tree (default dialect
   `postgres`, overridable per request / per catalog).
2. Walk the tree for `TableRef`s. Unqualified names are completed with the
   request's `catalog`/`schema` context. Each `catalog.schema.table` that maps
   to a registered table resolves to `Tabular.from_(source_url)` (lazy, pushdown
   capable). Unregistered path-shaped names fall through to the executor's
   built-in URL auto-resolution, so `SELECT * FROM 'data/x.parquet'` works too.
3. `execute_plan(node, tables)` runs it — predicates/projection/limit push into
   the scan; GROUP BY/JOIN/ORDER BY dispatch to Arrow kernels.
4. Result streams back as Arrow IPC. Bounded results stream straight from the
   Arrow buffer (one zero-copy frame). Heavy results spill to an Arrow IPC file
   under `spill_root` and stream from disk in 64 KiB chunks, so the node never
   holds a multi-GB result whole.

`compute_node`: when a request names a peer `node`, the whole call is proxied to
that node so compute runs where it's cheapest / where the data lives. Default is
the local node.

## tmp / spill cleanup daemon

A background loop (registered in the node lifespan) sweeps `tmp/` and
`spill_root` every hour, deleting entries older than `tmp_ttl` seconds
(default `86400` = 1 day, env `YGG_NODE_TMP_TTL`). This reclaims SQL spill files
and scratch downloads without operator intervention.

## Optimisations made to the shared layer

- `transport`: added genuine streaming helpers — `iter_arrow_ipc_stream`
  (incremental, framing-preserving IPC encode of a record-batch iterator) and
  `iter_file_chunks` (64 KiB disk streaming) — so query results are never fully
  buffered server-side.
- `LazyTabular._collect_schema`: fast-path that returns the source schema
  directly for schema-preserving plans (filter/limit/offset only), avoiding a
  full plan execution just to learn column types.
