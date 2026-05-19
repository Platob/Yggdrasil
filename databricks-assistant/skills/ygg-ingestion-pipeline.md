# Skill: build an ingestion pipeline end-to-end (HTTP / API / S3 → Unity Catalog)

## When to use

The user pastes API documentation, an OpenAPI / Swagger URL, an S3
bucket, an FTP endpoint, a webhook payload, or a vendor data feed and
asks to "ingest into Databricks", "build an ETL job", "schedule a
pipeline", "load this API into a Delta table". Also when they hand
you a problem in prose — *"I need every order from <vendor> refreshed
hourly into `main.sales.orders`"* — and expect a working pipeline,
not just snippets.

This skill orchestrates the others — it does **not** re-explain HTTP,
schema, table, or job mechanics. Each section ends with a pointer to
the dedicated skill for the call-site details.

## The seven steps that always happen

1. **Read the source contract.** Documentation, OpenAPI spec, sample
   payload, or — when nothing is provided — call the endpoint and
   inspect the response.
2. **Discover the schema.** Sample 50–500 rows, infer field types via
   `Field.from_pandas` / `Field.from_arrow` / Polars inference, then
   *fix* the inferred schema (nullability, decimal precision, timezone
   intent). See [`ygg-schema-discovery`](ygg-schema-discovery.md).
3. **Pick the schema layout.** One Unity Catalog schema per data
   source, `raw_<entity>` for landings + provenance columns, curated
   tables / views downstream. See
   [`ygg-data-modeling`](ygg-data-modeling.md).
4. **Reconcile the target.** Catalog / schema / table singletons,
   `ensure_created(schema=...)`, PK / FK / partition / cluster all
   ride on `Field` metadata. See
   [`ygg-data-modeling`](ygg-data-modeling.md) +
   [`ygg-databricks-tables`](ygg-databricks-tables.md).
5. **Write the fetch-and-load callable.** Pull pages via
   `HTTPSession` (or `SchemaSession` when the responses themselves
   are the cache — see the decision tree below), cast through the
   schema, write via `Table.insert` / `async_insert` / `merge`. See
   [`ygg-http`](ygg-http.md) +
   [`ygg-statement-result`](ygg-statement-result.md).
6. **Stage the callable as a Databricks Job task** with a schedule.
   See [`ygg-databricks-job-workflows`](ygg-databricks-job-workflows.md).
7. **Build the curated layer.** Standardise UTC timestamps, decimal
   money, ISO codes, naming. See
   [`ygg-curated-views`](ygg-curated-views.md). Benchmark the hot
   transform before merging — see [`ygg-benchmarks`](ygg-benchmarks.md).

## HTTP source — should the responses *be* the raw table?

Two shapes for the raw landing when the source is HTTP:

| Shape | When | How |
| --- | --- | --- |
| **Response cache = raw table.** Persist every `Response` row as-is, schema = `RESPONSE_SCHEMA`. | API is per-id `GET`s, slow-changing, idempotent. Replay = re-render from cached rows. | `SchemaSession(schema=…, local_cache=…, mode=…)` — see [`ygg-http`](ygg-http.md#choose-local-vs-remote-cache-via-schemasession). The Delta table behind the session *is* `raw_<entity>_responses`. |
| **Parse-then-write.** Pull payload, normalise into Arrow, write `raw_<entity>` with provenance columns. | API returns lists / pages / aggregates. Curated layer needs columnar access to fields. | Plain `HTTPSession` + `Table.insert / merge`; provenance columns (`_ingested_at`, `_payload_hash`, `_source`) on the schema. See [`ygg-data-modeling`](ygg-data-modeling.md#standard-raw_-provenance-columns). |

Use the response-cache shape (`SchemaSession`) when the source spec
says *"every call is cheap, results are stable per id"* — replay
becomes free, and the local-vs-remote cache decision tree in
[`ygg-http`](ygg-http.md) picks the right tier for the workload
(single-process notebook → `local_cache=True`; Spark / `send_many`
job → `local_cache=False`, remote-only is the synchronization point).
Use the parse-then-write shape for paginated lists, streaming
deltas, or any source where the curated layer needs predicate-pushed
column access.

## Worked example — REST API → Delta table, hourly

```python
from datetime import datetime, timedelta, timezone

from yggdrasil.databricks import DatabricksClient
from yggdrasil.data import Field, DataType, Schema
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.io.http_ import HTTPSession, HTTPRequest

dbc = DatabricksClient()


# ---- 1. source contract: paginated GET /v1/orders?since=<iso> ----------
ORDERS_API = "https://api.vendor.example.com"
ORDERS_PATH = "/v1/orders"


# ---- 2. discovered schema (see ygg-schema-discovery for sampling) ------
# Two layers — raw (vendor-shaped, immutable) + curated (standardised).
# See ygg-data-modeling for the raw_ + provenance convention.

import datetime as dt

RAW_ORDERS_SCHEMA = Schema.from_fields([
    # Source-shaped columns — names + types match the vendor payload.
    Field("order_id",     DataType.string(),  nullable=False,
              tags={"primary_key": True}),
    Field("customer_id",  DataType.string(),  nullable=False,
              tags={"foreign_key": True},
              metadata={"references": "main.vendor_orders.raw_customers(customer_id)"}),
    Field("created",      DataType.string(),  nullable=False,
              comment="Vendor ISO-8601 string, +offset varies."),
    Field("amount",       DataType.float64(), nullable=False,
              comment="Source ships float — curated layer demotes to decimal(18, 2)."),
    Field("ccy",          DataType.string(),  nullable=False),
    Field("country",      DataType.string(),  nullable=True),
    Field("status",       DataType.string(),  nullable=False),
    # Provenance — never from the source. See ygg-data-modeling.
    Field("_ingested_at", DataType.timestamp("UTC"), nullable=False,
              tags={"primary_key": True, "partition_by": True}),
    Field("_source",      DataType.string(),  nullable=False,
              comment="Logical source — matches the schema name."),
    Field("_source_url",  DataType.string(),  nullable=True),
    Field("_payload_hash", DataType.string(), nullable=False,
              comment="xxhash64 of the source row — dedup key."),
    Field("_batch_id",    DataType.string(),  nullable=False),
])


# ---- 3. one schema per source -----------------------------------------
dbc.catalog("main").ensure_created()
dbc.schema("main.vendor_orders").ensure_created(
    comment="Vendor orders source — raw_ landings + curated views.",
)


# ---- 4. reconcile target (PK / FK / partition all from Field tags) ----
raw_tbl = dbc.table("main.vendor_orders.raw_orders")
raw_tbl.ensure_created(
    schema=RAW_ORDERS_SCHEMA,
    comment="Bronze landing for vendor orders. Immutable, source-shaped.",
)


# ---- 5. fetch-and-load callable ---------------------------------------
def ingest_orders_since(since_iso: str) -> int:
    """Pull every order updated since *since_iso*, write to raw_orders.

    Returns the row count written so the caller can log / alert on
    empty windows.
    """
    import uuid

    session = HTTPSession.from_url(
        ORDERS_API,
        headers={"Authorization": f"Bearer {dbc.secrets.get('vendor', 'api_token')}"},
        retry={"total": 5, "backoff_factor": 0.5,
               "status_forcelist": (429, 500, 502, 503, 504)},
    )

    rows: list[dict] = []
    next_page: str | None = ORDERS_PATH + f"?since={since_iso}"
    while next_page:
        resp = session.send(HTTPRequest(method="GET", url=next_page))
        payload = resp.json()
        rows.extend(payload["data"])
        next_page = payload.get("next")  # vendor cursor pagination

    if not rows:
        return 0

    # Stamp provenance columns on every row before the cast.
    from yggdrasil.xxhash import xxh64
    from yggdrasil.pickle import json as ygg_json
    now = dt.datetime.now(dt.timezone.utc)
    batch_id = str(uuid.uuid4())
    for r in rows:
        r["_ingested_at"] = now
        r["_source"] = "vendor_orders"
        r["_source_url"] = f"{ORDERS_API}{ORDERS_PATH}"
        r["_payload_hash"] = xxh64(ygg_json.dumps(r, to_bytes=True)).hexdigest()
        r["_batch_id"] = batch_id

    # Cast Python dicts → Arrow with the raw schema in one pass —
    # no row loops, the cast registry does the work.
    from pyarrow import Table as ArrowTable
    arrow = ArrowTable.from_pylist(rows, schema=RAW_ORDERS_SCHEMA.to_arrow())

    # MERGE on (order_id, _ingested_at) so a re-run with the same
    # window doesn't duplicate — but a fresh fetch lands a new row.
    raw_tbl.merge(
        arrow,
        keys=["order_id", "_ingested_at"],
        update_columns=["customer_id", "created", "amount", "ccy",
                        "country", "status", "_source_url",
                        "_payload_hash", "_batch_id"],
    )
    return arrow.num_rows


# ---- 6. stage + schedule (hourly) -------------------------------------
from databricks.sdk.service.jobs import CronSchedule

job = dbc.jobs.create_or_update(
    name="ingest_vendor_orders",
    tasks=[],
    schedule=CronSchedule(
        quartz_cron_expression="0 0 * * * ?",   # top of every hour
        timezone_id="UTC",
    ),
    **dbc.jobs.userinfo_defaults(),
)

job.pytask(
    ingest_orders_since,
    (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    task_key="ingest_orders",
).create()

# ---- 7. curated view on top (see ygg-curated-views) -------------------
# main.vendor_orders.orders — standardised UTC timestamps + decimal money
# + ISO currency / country codes — feeds BI / ML. Built once at curate
# time, or rebuilt on every ingestion run.
#
# Benchmark the hot transform path before merging — see ygg-benchmarks.
```

That whole file is the pipeline. Everything else — auth, retries,
schema cast, staging Parquet, idempotent merge, schedule, dependency
resolution — is delegated to the singletons. Don't re-implement any
of it at the call site.

## S3 / object-store source variant

```python
from yggdrasil.io.path import Path as YggPath

src = YggPath.from_("s3://vendor-feeds/orders/2026/05/")
for f in src.iterdir():
    if not f.name.endswith(".parquet"):
        continue
    import pyarrow.parquet as pq
    arrow = pq.read_table(f.open("rb"))
    tbl.insert(arrow)  # schema reconciliation in insert()
```

`yggdrasil.io.path` covers s3 / gs / az / local / dbfs / volume /
workspace through one `Path` abstraction. Same `read_bytes` /
`iterdir` / `open` surface.

## Choosing the write path

| Frame size | Re-run shape | Use |
| --- | --- | --- |
| < 100 K rows | append-only | `tbl.insert(arrow)` |
| < 100 K rows | upsert / re-run | `tbl.merge(arrow, keys=[...])` |
| > 1 M rows | append | `tbl.async_insert(arrow, staging_volume=...)` |
| > 1 M rows | upsert | `tbl.async_insert + tbl.merge_from_volume(...)` |

The async path stages Parquet on a Volume and runs `COPY INTO` /
`MERGE` server-side — keep the warehouse / cluster sized for the
SQL, not the local Python frame.

## When the schema is unknown

Spend the first iteration *discovering* it (see
[`ygg-schema-discovery`](ygg-schema-discovery.md)), then **commit the
final Schema to source** as a Python literal. Don't infer at runtime
in the job — schema drift becomes silent corruption when the cast
registry's "best effort" lands on a different decimal precision than
last week.

## Don'ts

- Don't infer the schema on every run; commit the validated
  `Schema(...)` literal and let `tbl.insert` reconcile against it.
- Don't call the API once per row to "enrich" — pull a page, cast
  once, write once.
- Don't write a parallel pagination loop with `requests` — use
  `HTTPSession.send_many` with `SendManyConfig(max_workers=...)`.
- Don't stage Parquet by hand on DBFS when `tbl.async_insert` already
  does it on a Volume.
- Don't `ws.jobs.create(...)` raw — go through
  `dbc.jobs.create / create_or_update / create_for_user`. They wrap
  retries, name caching, volume-creation on `InvalidParameterValue`,
  and `userinfo_defaults` (git source, notifications, tags).
- Don't write a custom retry loop around an HTTP call — configure
  `retry=` on the session once.
- Don't ship a Python `for row in payload:` cast — let the Arrow
  schema cast handle the whole batch.
