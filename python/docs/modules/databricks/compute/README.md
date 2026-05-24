# yggdrasil.databricks.compute

Cluster lifecycle, execution contexts, and remote Python execution on Databricks.

---

## Surface map

| Symbol | Use for |
|---|---|
| `DatabricksClient().compute.clusters` | Cluster service — list, find, create/update, start/stop |
| `Cluster` | Individual cluster resource |
| `ExecutionContext` | Interactive command execution on a running cluster |
| `@databricks_remote_compute` | Decorator to run a Python function on a Databricks cluster |

---

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

cluster = DatabricksClient().compute.clusters.all_purpose_cluster(name="shared-etl")
```

---

## 1) Cluster service

```python
from yggdrasil.databricks import DatabricksClient

clusters = DatabricksClient(
    host="https://<workspace>",
    token="<token>",
).compute.clusters
```

---

## 2) Find or create a cluster

```python
# Reuse an existing cluster by name, or create it if absent
cluster = clusters.all_purpose_cluster(name="shared-etl")
print(cluster.cluster_id, cluster.state)

# Find only (raises if not found)
cluster = clusters.find_cluster("shared-etl")

# Create or update from config
cluster = clusters.create_or_update(
    cluster_name="etl-v2",
    num_workers=4,
    spark_version=clusters.latest_spark_version(photon=True),
    node_type_id="Standard_DS3_v2",
)
```

---

## 3) List clusters

```python
for c in clusters.list(limit=50):
    print(c.cluster_name, c.state, c.cluster_id)

# Filter by state
running = [c for c in clusters.list() if c.state == "RUNNING"]
```

---

## 4) Runtime version selection

```python
# Latest Photon + Python 3.12 runtime
version = clusters.latest_spark_version(photon=True, python_version="3.12")
print(version)   # e.g. "14.3.x-photon-scala2.12"

# All available runtimes
for v in clusters.spark_versions(photon=False, allow_ml=True):
    print(v.key, v.name)
```

---

## 5) Cluster lifecycle

```python
cluster = clusters.find_cluster("etl-v2")

# Start (wait for RUNNING)
cluster.start(wait=True)

# Restart
cluster.restart(wait=True)

# Stop / terminate
cluster.terminate(wait=True)

# Delete
cluster.delete(wait=True)

# Check current state
print(cluster.state)          # 'RUNNING', 'TERMINATED', ...
print(cluster.is_running())   # True / False
```

---

## 6) ExecutionContext — run code on a cluster

```python
from yggdrasil.databricks.compute import ExecutionContext
from yggdrasil.databricks import DatabricksClient

cluster = DatabricksClient().compute.clusters.find_cluster("shared-etl")

with ExecutionContext(cluster=cluster) as ctx:
    # Run arbitrary Python
    result = ctx.execute("print('hello from Databricks')")
    print(result)

    # Multi-line code block
    result = ctx.execute("""
import sys
import platform
print(sys.version)
print(platform.node())
""")
    print(result)

    # Run dbutils
    result = ctx.execute("dbutils.fs.ls('/tmp')[:3]")
    print(result)
```

---

## 7) One-shot remote execution

```python
from yggdrasil.databricks.compute import ExecutionContext
from yggdrasil.databricks import DatabricksClient

cluster = DatabricksClient().compute.clusters.find_cluster("shared-etl")
print(ExecutionContext(cluster=cluster).execute("import sys; print(sys.version)"))
```

---

## 8) `@databricks_remote_compute` decorator

Run a Python function as a remote computation on a Databricks cluster with zero boilerplate:

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.compute.remote import databricks_remote_compute

c = DatabricksClient()

@databricks_remote_compute(client=c, cluster_name="shared-etl")
def count_events(catalog: str, schema: str, table: str) -> int:
    from yggdrasil.databricks import DatabricksClient
    dbc = DatabricksClient()
    result = dbc.sql.execute(f"SELECT COUNT(*) AS n FROM {catalog}.{schema}.{table}")
    return result.to_pylist()[0]["n"]

n = count_events("main", "curated", "events")
print(f"Events: {n:,}")
```

---

## 9) End-to-end: ephemeral job cluster

For one-shot heavy computations, create a cluster, run the job, then terminate:

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.compute import ExecutionContext

c = DatabricksClient()
clusters = c.compute.clusters

# Create ephemeral cluster
cluster = clusters.create_or_update(
    cluster_name="ephemeral-ml-job",
    num_workers=8,
    spark_version=clusters.latest_spark_version(photon=False, allow_ml=True),
    node_type_id="Standard_DS4_v2",
)
cluster.start(wait=True)

try:
    with ExecutionContext(cluster=cluster) as ctx:
        ctx.execute("""
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
# ... training code
mlflow.log_metric("accuracy", 0.94)
""")
finally:
    cluster.terminate(wait=True)
    cluster.delete(wait=True)
```

---

## 10) Multi-cluster fan-out

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.compute import ExecutionContext
from yggdrasil.concurrent import Job, JobPoolExecutor

c = DatabricksClient()

cluster_names = ["etl-us-east", "etl-eu-west", "etl-ap-south"]
script = "from yggdrasil.databricks import DatabricksClient; print(DatabricksClient().sql.execute('SELECT current_user()').to_pylist())"

def run_on(name: str) -> str:
    cluster = c.compute.clusters.find_cluster(name)
    with ExecutionContext(cluster=cluster) as ctx:
        return ctx.execute(script)

jobs = [Job.make(run_on, name) for name in cluster_names]
with JobPoolExecutor(max_workers=3) as pool:
    for result in pool.as_completed(jobs):
        result.raise_for_exception()
        print(result.value)
```
