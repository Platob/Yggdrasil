# Skill: data modeling — schema layout, `raw_` tables, PK / FK via `Field` metadata

## When to use

The user asks "where should this table live?", "how do I organise
this catalog?", "what schema should I put the raw data in?", "how do
I link orders to customers?", "add a primary key", "add a foreign
key", or pastes a flat table and asks "model this properly". Also
when chaining from
[`ygg-ingestion-pipeline`](ygg-ingestion-pipeline.md) — the recipe
calls into this skill at the *reconcile target* step.

## One schema per data source

A schema in Unity Catalog is the natural ownership boundary. Default
layout for any new data source:

```
<catalog>
└── <source>                       ← schema = source / project name
    ├── raw_<entity>                ← landed, immutable, source-shaped
    ├── raw_<entity>_changes        ← optional CDC / audit
    ├── <entity>                    ← curated view (see ygg-curated-views)
    └── _meta                       ← lookup / dim tables, source-of-truth refs
```

Real examples:

```
main.vendor_orders.raw_orders
main.vendor_orders.raw_customers
main.vendor_orders.orders                 (curated view over raw_orders)
main.vendor_orders.customers              (curated view over raw_customers)
main.iso.country                          (shared dim — ISO 3166-1)
main.iso.currency                         (shared dim — ISO 4217)
main.iso.timezone                         (shared dim — IANA)
```

`<catalog>` is shared across the org (`main` / `dev` / `prod`);
`<source>` is one source / project / vendor — never a mix. Schemas
are cheap; *combining unrelated sources into one schema is the
mistake*. Cross-source joins go through shared `main.iso.*`
dimensions and stable ISO codes — not vendor A's "country" column
joining vendor B's "country" column directly.

`dbc.catalog(...)` / `dbc.schema(...)` / `dbc.table(...)` are
singleton resources — call their `.ensure_created()` methods, not
`ws.catalogs.create(...)` / `ws.schemas.create(...)`. See
[`ygg-databricks-client`](ygg-databricks-client.md).

## The `raw_` prefix is a contract

Any table prefixed `raw_` carries these guarantees:

- **Source-shaped.** Column names and types come from the source,
  not a curated model. If the API ships `customer_id_str`, the raw
  table is `customer_id_str string`.
- **Immutable.** Once written, rows are not edited. CDC / corrections
  land in a sibling `raw_<entity>_changes` table or via a new
  load partition.
- **Carries ingestion metadata.** Every `raw_` table includes the
  ingestion provenance columns below — none of them come from the
  source payload.
- **Idempotent on rerun.** Use `MERGE` keyed on the source's natural
  id when present, `(natural_id, ingested_at)` otherwise. A re-run of
  the same window must not duplicate.

### Geographic shared dims

When the source has a place reference (country, region, exchange,
bidding zone, station), the **shared `main.iso.*` dim is the only
right join target**, and that dim ships `lat` + `lon` (+
`boundary_geojson` for polygons) so the curated row is *renderable*
without a second lookup. See
[`ygg-curated-views`](ygg-curated-views.md#3b-geographic-data--always-carry-latlon--optional-polygon)
for the full geo-display convention.

### Standard `raw_` provenance columns

Add these to every `raw_<entity>` schema, after the source-shaped
columns:

```python
from yggdrasil.data import Field, DataType, Schema

PROVENANCE = [
    Field("_ingested_at",
              DataType.timestamp("UTC"),  nullable=False,
              comment="When the row landed in this table (UTC)."),
    Field("_source",
              DataType.string(),          nullable=False,
              comment="Logical source name — matches the schema."),
    Field("_source_url",
              DataType.string(),          nullable=True,
              comment="Endpoint / file URL the row came from."),
    Field("_payload_hash",
              DataType.string(),          nullable=False,
              comment="Hash of the source row (xxhash64) — dedup key."),
    Field("_batch_id",
              DataType.string(),          nullable=False,
              comment="Run / job id that produced the row."),
]
```

`Response` (`yggdrasil.io.response`) already carries most of these as
`received_at`, `body_hash`, `partition_key`. When the source is an
HTTP `SchemaSession` cache, the `RESPONSE_SCHEMA` columns *are* the
provenance — don't re-derive them.

## Field metadata drives DDL

`yggdrasil.data.Field` carries metadata + tag flags that the table
layer reads when emitting `CREATE TABLE`. Set them once on the
`Schema`; the rest is automatic.

```python
from yggdrasil.data import Field, DataType, Schema

ORDERS_SCHEMA = Schema.from_fields([
    Field(
        "order_id", DataType.string(), nullable=False,
        comment="Vendor order id, ULID format.",
        tags={"primary_key": True},
    ),
    Field(
        "customer_id", DataType.string(), nullable=False,
        comment="FK → main.vendor_orders.raw_customers.customer_id",
        tags={"foreign_key": True},
        metadata={
            "references": "main.vendor_orders.raw_customers(customer_id)",
        },
    ),
    Field("amount",        DataType.decimal(18, 2),    nullable=False),
    Field(
        "paid_at", DataType.timestamp("UTC"), nullable=False,
        tags={"partition_by": True},      # daily partition by date(paid_at)
    ),
    Field(
        "currency_iso", DataType.string(), nullable=False,
        comment="ISO 4217 — see yggdrasil.data.enums.Currency.",
        tags={"cluster_by": True},
    ),
])
```

What each tag does on `Table.ensure_created(schema=...)`:

| Field tag | Effect on the CREATE TABLE |
| --- | --- |
| `primary_key=True` | Composite PK, RELY constraint inline DDL. Multiple fields → one composite key. |
| `foreign_key=True` + `metadata["references"]` | Post-create constraint via `TableConstraints.create_constraint`. |
| `partition_by=True` | Column lands in `PARTITIONED BY (...)`. |
| `cluster_by=True` | Column lands in `CLUSTER BY (...)` (Delta liquid clustering). |
| `comment="..."` | Column-level `COMMENT '...'`. |
| `metadata={...}` | Round-trips through Arrow field metadata + the SDK column properties. |

Helpers when you'd rather not use a literal `tags={}` dict:

```python
field = (
    Field("order_id", DataType.string(), nullable=False)
    .with_primary_key()
    .with_partition_by(False)     # explicit off, in case a parent set it
)
```

`Field.with_primary_key()` / `with_foreign_key()` /
`with_partition_by()` / `with_cluster_by()` all return a new
`Field` with the tag flag set (or replace in place when
`inplace=True`).

## Composite primary keys + auto-naming

For `raw_` tables, the natural PK is usually
`(source_id, _ingested_at)` — a re-fetch lands a new row, but the
pair stays unique. The table layer's `_build_inline_constraints`
emits a single `CONSTRAINT pk_<table>_<cols> PRIMARY KEY(col1, col2)
RELY` clause; no need to name it by hand.

```python
RAW_ORDERS_SCHEMA = Schema.from_fields([
    Field("order_id",     DataType.string(), nullable=False,
              tags={"primary_key": True}),
    Field("_ingested_at", DataType.timestamp("UTC"), nullable=False,
              tags={"primary_key": True, "partition_by": True}),
    Field("_payload_hash", DataType.string(), nullable=False),
    # ... source-shaped columns ...
])
```

## Foreign keys land via the constraint service

PK landings happen inline in `CREATE TABLE`; FK creation needs a
follow-up because the parent table must exist first. The Table layer
calls `TableConstraints.create_constraint` post-create when a Field
carries `foreign_key=True` + `metadata["references"]="cat.sch.tbl(col)"`.

Re-running `ensure_created` after the parent exists is the trigger:

```python
# 1. parent (raw_customers) created first
dbc.table("main.vendor_orders.raw_customers").ensure_created(
    schema=RAW_CUSTOMERS_SCHEMA,
)

# 2. child (raw_orders) — FK lands on this call
dbc.table("main.vendor_orders.raw_orders").ensure_created(
    schema=RAW_ORDERS_SCHEMA,    # contains foreign_key Fields w/ references
)
```

FK creation is best-effort: the table layer logs and continues if the
constraint already exists (`CONSTRAINT_ALREADY_EXISTS_IN_SCHEMA`) or
the references syntax doesn't parse. Don't pre-`exists()` — let the
typed exception flow.

## Cross-schema joins via shared ISO dimensions

Vendor A's `country_code` should not join vendor B's `country` —
they drift independently. Both join `main.iso.country(iso_alpha2)`
instead:

```sql
-- bad: vendor-specific shapes drift independently
JOIN main.vendor_a.raw_orders ord
  ON ord.country_code = main.vendor_b.raw_returns.country

-- good: both bridge through a shared ISO dim
JOIN main.iso.country c ON ord.country_iso = c.iso_alpha2
JOIN main.iso.country c ON ret.country_iso = c.iso_alpha2
```

Build the shared dim once from `yggdrasil.data.enums.geozone` /
`Currency` / `Timezone` (see
[`ygg-curated-views`](ygg-curated-views.md) for the standardisation
recipes that put the right ISO codes into the curated layer).

## `_meta` and reference data

Per-source lookups (vendor's own currency table, vendor's status
enum, vendor's country list) belong in `<source>._meta` — not at the
catalog root. That way nothing leaks across sources, and the `iso.*`
schema stays canonical.

```
main.vendor_orders._meta.status_codes        ← vendor-specific
main.iso.country                              ← shared
```

## Don'ts

- Don't rename source columns on the way into `raw_` — preserve the
  source's shape. Renaming + casting belongs in the curated view.
- Don't `DROP` and re-`CREATE` to "change the schema" — let
  `Table.ensure_created(schema=...)` reconcile via `ALTER`, or evolve
  through a new `raw_<entity>_v2` when the source shape genuinely
  broke.
- Don't store country / currency / timezone as free text. Use ISO
  codes and join through the shared dim.
- Don't combine two sources in one schema even if "they look similar".
  Schemas are free; cross-source joins go through the shared
  dimensions.
- Don't hand-write `CREATE TABLE` SQL when a `Schema` literal + tags
  produces the same DDL with PK/FK/partition/cluster baked in.
- Don't put PK / FK / partition flags as ad-hoc dict keys; use the
  `with_primary_key()` / `with_foreign_key()` / `with_partition_by()`
  / `with_cluster_by()` helpers so the metadata round-trips through
  Arrow and the SDK.
- Don't omit the provenance columns. `_ingested_at` /
  `_payload_hash` / `_source` are the difference between a
  reproducible pipeline and "why is this row here?".
