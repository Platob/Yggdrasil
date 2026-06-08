# Databricks

`yggdrasil.databricks` wraps the Databricks SDK with a single entrypoint, `DatabricksClient`, plus typed result wrappers and Arrow-first conversions.

```bash
pip install "ygg[databricks]"
```

## One client, many services

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient(host="https://<workspace>", token="<token>")
```

| Service | What it covers | First call |
|---|---|---|
| `c.sql` | Query execution, DDL/DML, result conversion | `c.sql.execute("SELECT 1")` |
| `c.warehouses` | Warehouse discovery / start / stop / update | `c.warehouses.find_default()` |
| `c.catalogs` / `c.tables` | Unity Catalog hierarchy + table resources | `c.catalogs["main"]["default"]["orders"]` |
| `c.compute` | Cluster lifecycle / runtime selection | `c.compute.clusters.all_purpose_cluster(name="etl")` |
| `c.dbfs_path(...)` | DBFS / Volumes / Workspace files | `c.dbfs_path("dbfs:/tmp/a.txt")` |
| `c.secrets` | Scope/secret CRUD | `c.secrets.create_secret("scope/key", "value")` |
| `c.iam` | Users/groups (workspace or account) | `c.iam.users.current_user` |

## Authentication

```python
# PAT
DatabricksClient(host="https://<workspace>", token="<token>")

# OAuth client credentials
DatabricksClient(host="https://<workspace>", client_id="...", client_secret="...")

# Environment-driven (best for local + CI)
DatabricksClient()   # reads DATABRICKS_HOST / TOKEN / CONFIG_PROFILE / ACCOUNT_ID / CLUSTER_ID
```

Reuse the global singleton: `DatabricksClient.current()`.

## SQL execution

```python
stmt = c.sql.execute("SELECT current_user() AS me")
stmt.to_arrow_table()
stmt.to_pandas()
stmt.to_polars()
stmt.to_spark()
stmt.to_pylist()
```

## End-to-end: write from many engines, read into any

```python
import pyarrow as pa
from yggdrasil.databricks import DatabricksClient
from yggdrasil.lazy_imports import pandas as pd
from yggdrasil.lazy_imports import polars as pl

c = DatabricksClient(host="https://<workspace>", token="<token>")
sql = c.sql

table = "main.default.demo_ingest"

sql.execute(f"""
CREATE TABLE IF NOT EXISTS {table} (
  id BIGINT, source STRING, payload STRING
) USING DELTA
""")
sql.execute(f"DELETE FROM {table}")

# Write from pyarrow
sql.arrow_insert_into(table, pa.table({
    "id": [1], "source": ["pyarrow"], "payload": ['{"k":"arrow"}'],
}))

# Write from pandas
sql.insert_into(table, pd.DataFrame([{"id": 2, "source": "pandas", "payload": '{"k":"pandas"}'}]))

# Write from polars
sql.insert_into(table, pl.DataFrame({
    "id": [3], "source": ["polars"], "payload": ['{"k":"polars"}'],
}))

# Write from a Spark DataFrame
sdf = spark.createDataFrame([{"id": 4, "source": "pyspark", "payload": '{"k":"spark"}'}])
sql.spark_insert_into(table, sdf)

# Write unstructured rows
sql.insert_into(table, [
    {"id": 5, "source": "raw", "payload": "free-form note"},
    {"id": 6, "source": "raw", "payload": '{"freeform": [1,2,3]}'},
])

# Read once, project anywhere
stmt = sql.execute(f"SELECT * FROM {table} ORDER BY id")
arrow_table = stmt.to_arrow_table()
pandas_df   = stmt.to_pandas()
polars_df   = stmt.to_polars()
spark_df    = stmt.to_spark(spark=spark)
pylist      = stmt.to_pylist()
```

## Files: DBFS, Volumes, Workspace

```python
p = c.dbfs_path("/Volumes/main/default/raw/data.parquet")

p.exists(), p.is_file(), p.stat()
p.parent.mkdir(parents=True, exist_ok=True)
p.write_bytes(b"...")
p.read_text()
list(p.parent.ls())
p.rename("/Volumes/main/default/raw/data.archive.parquet")
p.remove()
```

Temp file workflow:

```python
tmp = c.tmp_path(extension="json", max_lifetime=1800)
tmp.write_text('{"step": "created"}')
print(tmp.exists(), tmp.read_text())
c.clean_tmp_folder()
```

## Secrets

```python
c.secrets.create_scope("demo")
c.secrets.create_secret("api-key", "<value>", scope="demo")

# dict-style shortcuts
c.secrets["demo/api-key"] = "rotated"
del c.secrets["demo/api-key"]
```

## Compute

```python
clusters = c.compute.clusters

cluster = clusters.create_or_update(cluster_name="etl", num_workers=1)
cluster = clusters.all_purpose_cluster(name="shared-etl")
cluster = clusters.find_cluster("shared-etl")
clusters.latest_spark_version(photon=True, python_version="3.12")
```

Run code on a cluster:

```python
from yggdrasil.databricks.compute import ExecutionContext

with ExecutionContext(cluster=cluster) as ctx:
    print(ctx.execute("print('hello from databricks')"))
```

Function-level remote dispatch:

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(cluster_name="shared-etl")
def add(x: int, y: int) -> int:
    return x + y
```

## IAM

```python
iam = c.iam
iam.users.current_user
iam.users.create("analyst@company.com")
list(iam.users.list(limit=20))

grp = iam.groups.create("data-engineering")
list(iam.groups.list(name="data-engineering", limit=5))
iam.groups.delete(grp)
```

## Genie (AI/BI)

`c.genie` is code-oriented access to Databricks AI/BI **Genie** spaces:
list / find / create spaces, ask a space a question, and read the rows it
computed — all through the same `StatementResult` surface as `c.sql`.

```python
space = c.genie.find("Sales analytics")     # or c.genie["01ef…space_id"]
answer = space.ask("top 10 customers by revenue last quarter")

answer.text            # Genie's narrative reply
answer.query           # the SQL it generated
answer.to_polars()     # the rows, re-read from the executed statement (no re-run)

follow = space.follow_up(answer.conversation_id, "now split by product")
```

Genie runs its SQL on the space's own warehouse; `answer.result()`
re-attaches to that finished statement (falling back to the engine's
default warehouse when the space has none), so reading the rows never
re-runs the query.

## Troubleshooting

- **401 / 403** — verify host + token, and whether you need workspace vs account scope.
- **Warehouse query issues** — make sure a warehouse is running: `c.warehouses.find_default().start()`.
- **Cluster code execution fails** — check cluster policy, permissions, runtime version compatibility.
- **Path not found** — pick the right prefix (`dbfs:/...` for DBFS, `/Volumes/...` for Volumes).
- **Optional package missing** — install the right extra (`ygg[databricks]`, `ygg[data]`, `ygg[bigdata]`, `ygg[http]`).
- **Local integration tests skipped** — they require `DATABRICKS_HOST` (gated by the `integration` marker).

## See also

- [databricks/sql](../modules/databricks/sql/README.md)
- [databricks/compute](../modules/databricks/compute/README.md), [remote](../modules/databricks/compute/remote/README.md)
- [databricks/workspaces](../modules/databricks/workspaces/README.md), [fs](../modules/databricks/fs/README.md)
- [databricks/secrets](../modules/databricks/secrets/README.md), [iam](../modules/databricks/iam/README.md)
- [databricks/account](../modules/databricks/account/README.md)
