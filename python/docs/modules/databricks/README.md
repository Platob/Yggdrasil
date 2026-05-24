# yggdrasil.databricks

Production-ready Databricks toolkit for Yggdrasil.

This package gives you a **single entrypoint** (`DatabricksClient`) and service helpers for:

- SQL execution and warehouse lifecycle
- Unity Catalog navigation (catalog/schema/table/column)
- Compute cluster management and remote execution contexts
- DBFS / Volume / Workspace file operations
- Secrets and IAM administration
- Jobs parameter parsing and typed notebook config
- Genie conversational analytics

---

## 1) Quick start (copy/paste)

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient(host="https://<workspace>", token="<token>")
print(client.sql.execute("SELECT current_user() AS me").to_pandas())
```

One-line style works across services:

- SQL: `DatabricksClient().sql.execute("SELECT 1")`
- Warehouses: `DatabricksClient().warehouses.find_default().start()`
- Catalogs: `DatabricksClient().catalogs["main"]["default"]["orders"]`
- Compute: `DatabricksClient().compute.clusters.all_purpose_cluster(name="etl")`
- FS: `DatabricksClient().dbfs_path("dbfs:/tmp/demo.txt").write_text("ok")`
- Secrets: `DatabricksClient().secrets.create_secret("scope/key", "value")`
- IAM: `next(DatabricksClient().iam.users.list(limit=1), None)`
- Genie: `DatabricksClient().genie.ask("<space-id>", "weekly revenue")`

---

## 2) Authentication patterns

```python
from yggdrasil.databricks import DatabricksClient
```

### PAT token

```python
client = DatabricksClient(host="https://<workspace>", token="<token>")
```

### OAuth client credentials

```python
client = DatabricksClient(
    host="https://<workspace>",
    client_id="<client-id>",
    client_secret="<client-secret>",
)
```

### Environment-driven (best for local + CI)

```python
client = DatabricksClient()  # reads DATABRICKS_* variables
```

Common env vars: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_ACCOUNT_ID`, `DATABRICKS_CLUSTER_ID`, `DATABRICKS_CONFIG_PROFILE`.

---

## 3) End-to-end workflows

### A. SQL + table lifecycle

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient(host="https://<workspace>", token="<token>")

c.sql.execute("CREATE TABLE IF NOT EXISTS main.default.demo (id BIGINT, name STRING) USING DELTA")
c.sql.insert_into("main.default.demo", [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}])

stmt = c.sql.execute("SELECT * FROM main.default.demo ORDER BY id")
print(stmt.to_arrow_table())
print(stmt.to_pandas())
print(stmt.to_polars())
```

### B. Files + secrets + SQL in one flow

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient(host="https://<workspace>", token="<token>")

c.secrets.create_secret("demo/api_key", "abc123")
path = c.tmp_path(extension="json")
path.write_text('{"event":"created"}')

c.sql.execute("SELECT current_timestamp() AS ts")
print(path.read_text())
```

### C. Compute execution context

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.compute import ExecutionContext

c = DatabricksClient(host="https://<workspace>", token="<token>")
cluster = c.compute.clusters.create_or_update(cluster_name="docs-cluster", num_workers=1)

with ExecutionContext(cluster=cluster) as ctx:
    print(ctx.execute("print('hello from databricks')"))
```

---

## 4) Service feature map

| Service | What it covers | Best first call |
|---|---|---|
| `client.sql` | Query execution, DDL/DML, result conversion | `client.sql.execute("SELECT 1")` |
| `client.warehouses` | Warehouse discovery/start/stop/update | `client.warehouses.find_default()` |
| `client.catalogs` / `client.tables` | Unity Catalog hierarchy + table resources | `client.catalogs["main"]["default"]["orders"]` |
| `client.compute` | Cluster lifecycle/version selection | `client.compute.clusters.all_purpose_cluster(name="etl")` |
| `client.dbfs_path(...)` | DBFS/Volumes path operations | `client.dbfs_path("dbfs:/tmp/a.txt")` |
| `client.secrets` | Scope/secret CRUD helpers | `client.secrets.create_secret("scope/key", "value")` |
| `client.iam` | Users/groups in workspace/account scope | `client.iam.users.current_user` |
| `client.genie` | Conversational BI workflows | `client.genie.ask("<space-id>", "top customers")` |
| `client.compute` | Cluster lifecycle / remote execution | `client.compute.clusters.all_purpose_cluster(name="etl")` |

---

## 5) Troubleshooting

- **Auth errors (`401/403`)**: verify host + token pair, and whether you need workspace scope vs account scope.
- **Warehouse query issues**: ensure a running warehouse exists (`client.warehouses.find_default().start()`).
- **Cluster code execution fails**: verify cluster policy, permissions, and runtime version compatibility.
- **Path not found**: ensure DBFS vs `/Volumes/...` prefixes are correct for the target path type.
- **Optional package missing**: install the right extra (`ygg[databricks]`, `ygg[data]`, `ygg[http]`, etc.).

---

## 6) Full module docs

- [sql](sql/README.md)
- [compute](compute/README.md)
- [compute.remote](compute/remote/README.md)
- [workspaces](workspaces/README.md)
- [fs](fs/README.md)
- [secrets](secrets/README.md)
- [iam](iam/README.md)
- [genie](genie/README.md)
- [jobs](jobs/README.md)
- [workflow](workflow/README.md) — Prefect-style `@flow` / `@task` API
- [account](account/README.md)

---

## 7) Schema-driven table creation with Field tags

`Field` metadata drives DDL: `primary_key`, `foreign_key`, `partition_by`, and `cluster_by` tags become SQL constraints and physical layout directives automatically.

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.data import field, Schema

c = DatabricksClient()

orders_schema = Schema([
    field("order_id",     "int64",       nullable=False, tags={"primary_key": True}),
    field("customer_id",  "int64",       nullable=False,
          tags={"foreign_key": True}, metadata={"references": "main.customers.customers(id)"}),
    field("amount",       "decimal(18,2)"),
    field("currency_iso", "string"),
    field("placed_at_utc","timestamp[us, UTC]"),
    field("region",       "string",      tags={"partition_by": True}),
], name="orders")

table = c.sql.table("main", "default", "orders")
table.ensure_created(schema=orders_schema, comment="Order fact table")
print(table.exists)   # True
```

---

## 8) Async / parallel insert into sharded tables

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.concurrent import Job, JobPoolExecutor
import pyarrow as pa

c = DatabricksClient()

# Sharded raw table pattern: one table per symbol, no MERGE contention
def ingest_symbol(symbol: str, rows: list[dict]) -> None:
    table_name = f"main.raw.raw_ohlcv_{symbol.lower()}"
    tbl = pa.table(rows)
    c.sql.insert_into(table_name, tbl, mode="append", create_if_missing=True)

payloads = [
    ("AAPL", [{"ts": "2026-05-01", "open": 175.0, "close": 178.5}]),
    ("GOOG", [{"ts": "2026-05-01", "open": 140.0, "close": 141.2}]),
    ("MSFT", [{"ts": "2026-05-01", "open": 420.0, "close": 425.0}]),
]

jobs = [Job.make(ingest_symbol, sym, rows) for sym, rows in payloads]
with JobPoolExecutor(max_workers=3) as pool:
    for res in pool.as_completed(jobs):
        res.raise_for_exception()
print("All shards written")
```

---

## 9) Full pipeline: ingest → curate → dash (workflow)

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.workflow import flow, task

c = DatabricksClient()

@task(name="ingest-orders")
def ingest(date: str) -> str:
    from yggdrasil.fxrate import FxRate
    df = FxRate().fetch([("EUR", "USD"), ("GBP", "USD")], start=date, end=date, sampling="1d")
    c.sql.insert_into("main.iso.raw_fxrate", df.to_arrow(), mode="append")
    return "main.iso.raw_fxrate"

@task(name="curate-fxrate", depends_on=["ingest-orders"])
def curate(source_table: str) -> str:
    c.sql.execute(f"""
        INSERT OVERWRITE main.iso.fxrate
        SELECT source, target, from_timestamp, to_timestamp, sampling,
               ROUND(value, 8) AS value
        FROM {source_table}
        WHERE value IS NOT NULL
    """)
    return "main.iso.fxrate"

@task(name="dash-fxrate", depends_on=["curate-fxrate"])
def dash(curated_table: str) -> None:
    c.sql.execute(f"""
        CREATE OR REPLACE TABLE main.iso.dash_fxrate AS
        SELECT source, target, DATE(from_timestamp) AS date,
               AVG(value) AS avg_rate, MAX(value) AS high, MIN(value) AS low
        FROM {curated_table}
        GROUP BY 1, 2, 3
    """)

@flow(name="daily-fxrate", schedule="0 6 * * *")
def daily_pipeline(date: str = "2026-05-01"):
    raw   = ingest(date)
    curated = curate(raw)
    dash(curated)

# Deploy to Databricks
job = daily_pipeline.deploy(client=c, catalog="main", schema="iso")
print(job.url)
```

---

## 10) DatabricksPath + Arrow filesystem

```python
from yggdrasil.databricks import DatabricksClient
import pyarrow.parquet as pq
import pyarrow as pa

c = DatabricksClient()
vol_path = c.dbfs_path("/Volumes/main/default/raw/orders/")
vol_path.mkdir(parents=True, exist_ok=True)

# Write via Arrow filesystem
tbl = pa.table({"id": [1, 2, 3], "amount": [10.5, 20.0, 5.75]})
fs = c.volumes["main"]["default"]["raw"].arrow_filesystem()
pq.write_table(tbl, "orders/batch_001.parquet", filesystem=fs)

# Read back
result = pq.read_table("orders/batch_001.parquet", filesystem=fs)
print(result)
```
