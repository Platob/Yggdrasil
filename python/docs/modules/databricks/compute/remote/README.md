# yggdrasil.databricks.compute.remote

`databricks_remote_compute` is a function decorator that transparently redirects a local Python function to run on a Databricks cluster. When called from inside a Databricks environment (notebook, job) it is a no-op — the function runs locally as if the decorator wasn't there.

## One-liner

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(cluster_name="shared-etl")
def transform(x: int, y: int) -> int:
    return x + y

result = transform(3, 4)   # runs on the cluster, returns 7
```

---

## How it works

1. **Inside Databricks (notebook / job):** the decorator is a transparent no-op. `transform(3, 4)` runs in-process.
2. **Outside Databricks with `DATABRICKS_HOST` set:** the decorator serializes the call, ships it to the resolved cluster via an `ExecutionContext`, and returns the result.
3. **Outside Databricks without `DATABRICKS_HOST`:** the decorator is a no-op (no workspace to connect to).

The cluster is resolved once at decoration time using `DatabricksClient`, cached, and reused across calls.

---

## Cluster selection

### By name

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(cluster_name="shared-etl")
def heavy_compute(data: list) -> list:
    return [x ** 2 for x in data]
```

### By cluster ID

```python
@databricks_remote_compute(cluster_id="0423-142519-abc12345")
def summarize(rows: list) -> dict:
    return {"count": len(rows), "sum": sum(r["amount"] for r in rows)}
```

### Explicit cluster object

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.compute.remote import databricks_remote_compute

client  = DatabricksClient()
cluster = client.compute.clusters.all_purpose_cluster(name="etl-cluster")

@databricks_remote_compute(cluster=cluster)
def run_etl(date: str) -> str:
    from yggdrasil.databricks import DatabricksClient
    dbc = DatabricksClient()
    dbc.sql.execute(f"OPTIMIZE main.sales.orders WHERE dt = '{date}'")
    return f"done:{date}"
```

### Explicit workspace

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(
    workspace="https://my-workspace.azuredatabricks.net",
    cluster_name="shared-etl",
)
def process():
    ...
```

---

## Forward environment variables

Pass a list of local environment variable names to forward to the cluster. Useful for secrets that you don't want baked into the function body:

```python
import os
from yggdrasil.databricks.compute.remote import databricks_remote_compute

os.environ["VENDOR_API_KEY"] = "secret-key-from-vault"

@databricks_remote_compute(cluster_name="shared-etl", env_keys=["VENDOR_API_KEY"])
def fetch_vendor_data(date: str) -> list:
    import os
    key = os.environ["VENDOR_API_KEY"]   # forwarded from local env
    return call_vendor_api(key, date)
```

---

## Force local execution

Set `force_local=True` to always run locally regardless of environment. Useful for tests:

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(cluster_name="etl", force_local=True)
def my_fn(x: int) -> int:
    return x * 2

assert my_fn(5) == 10   # runs locally in tests
```

---

## Use as a factory (no-arg call)

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

# Build the decorator first, then apply it to multiple functions
remote = databricks_remote_compute(cluster_name="etl")

@remote
def step_one(data: list) -> list:
    return [x + 1 for x in data]

@remote
def step_two(data: list) -> int:
    return sum(data)

result = step_two(step_one([1, 2, 3]))   # 9 — both run on the same cluster
```

---

## End-to-end: local ETL → cluster execution

```python
import os
from yggdrasil.databricks.compute.remote import databricks_remote_compute

os.environ.setdefault("DATABRICKS_HOST", "https://my-workspace.azuredatabricks.net")

@databricks_remote_compute(cluster_name="etl-cluster", env_keys=["DATABRICKS_TOKEN"])
def rebuild_curated_table(catalog: str, schema: str, table: str) -> dict:
    from yggdrasil.databricks import DatabricksClient

    dbc = DatabricksClient()
    stmt = dbc.sql.execute(
        f"SELECT COUNT(*) AS n FROM {catalog}.{schema}.{table}"
    )
    return stmt.to_pylist()[0]

# Runs on the cluster; local caller gets the dict back
stats = rebuild_curated_table("main", "sales", "orders")
print(stats)   # {"n": 123456}
```

---

## Signature reference

```python
def databricks_remote_compute(
    _func=None,                          # function to decorate (None → use as factory)
    cluster_id: str | None = None,       # target cluster by ID
    cluster_name: str | None = None,     # target cluster by name
    workspace=None,                      # DatabricksClient, host string, or None (env-driven)
    cluster=None,                        # pre-resolved Cluster object
    env_keys: list[str] | None = None,   # env vars to forward to the cluster
    force_local: bool = False,           # always run locally (useful in tests)
)
```
