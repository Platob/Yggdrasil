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

## The six steps that always happen

1. **Read the source contract.** Documentation, OpenAPI spec, sample
   payload, or — when nothing is provided — call the endpoint and
   inspect the response.
2. **Discover the schema.** Sample 50–500 rows, infer field types via
   `Field.from_pandas` / `Field.from_arrow` / Polars inference, then
   *fix* the inferred schema (nullability, decimal precision, timezone
   intent). See [`ygg-schema-discovery`](ygg-schema-discovery.md).
3. **Reconcile the target.** Build a `Schema` from the discovered
   shape, ensure the catalog / schema / volume / table exist via the
   resource singletons. See
   [`ygg-databricks-tables`](ygg-databricks-tables.md) +
   [`ygg-schema-fields`](ygg-schema-fields.md).
4. **Write the fetch-and-load callable** — a single Python function
   that pulls a page / range from the source, casts to the target
   schema, and writes via `Table.insert` / `async_insert` / `merge`.
   See [`ygg-http`](ygg-http.md) +
   [`ygg-statement-result`](ygg-statement-result.md).
5. **Stage the callable as a Databricks Job task** with a schedule.
   See [`ygg-databricks-job-workflows`](ygg-databricks-job-workflows.md).
6. **Benchmark the hot path during development** so the pipeline
   doesn't ship a row-loop trap. See
   [`ygg-benchmarks`](ygg-benchmarks.md).

## Worked example — REST API → Delta table, hourly

```python
from datetime import datetime, timedelta, timezone

from yggdrasil.databricks import DatabricksClient
from yggdrasil.data import DataField, DataType, Schema
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.io.http_ import HTTPSession, HTTPRequest

dbc = DatabricksClient()


# ---- 1. source contract: paginated GET /v1/orders?since=<iso> ----------
ORDERS_API = "https://api.vendor.example.com"
ORDERS_PATH = "/v1/orders"


# ---- 2. discovered schema (see ygg-schema-discovery for sampling) ------
ORDERS_SCHEMA = Schema.from_fields([
    DataField("order_id",   DataType.string(),               nullable=False),
    DataField("customer_id", DataType.string(),              nullable=False),
    DataField("amount",     DataType.decimal(18, 2),          nullable=False),
    DataField("currency",   DataType.string(),                nullable=False),
    DataField("paid_at",    DataType.timestamp("UTC"),        nullable=False),
    DataField("status",     DataType.string(),                nullable=False),
])


# ---- 3. reconcile target ----------------------------------------------
tbl = dbc.table("main.sales.orders")
tbl.ensure_created(schema=ORDERS_SCHEMA, comment="vendor orders v1")


# ---- 4. fetch-and-load callable ---------------------------------------
def ingest_orders_since(since_iso: str) -> int:
    """Pull every order updated since *since_iso*, write to Delta.

    Returns the row count written so the caller can log / alert on
    empty windows.
    """
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

    # Cast Python dicts → Arrow with the target schema in one pass —
    # no row loops, the cast registry does the work.
    from pyarrow import Table as ArrowTable
    arrow = ArrowTable.from_pylist(rows, schema=ORDERS_SCHEMA.to_arrow())

    # MERGE on (order_id) so a re-run is idempotent.
    tbl.merge(
        arrow,
        keys=["order_id"],
        update_columns=["customer_id", "amount", "currency",
                        "paid_at", "status"],
    )
    return arrow.num_rows


# ---- 5. stage + schedule (hourly) -------------------------------------
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

# ---- 6. benchmark the hot path before merging (see ygg-benchmarks) ----
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
