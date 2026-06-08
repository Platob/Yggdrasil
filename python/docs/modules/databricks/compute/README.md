# yggdrasil.databricks.compute

Cluster lifecycle and remote command execution helpers.

## Recommended one-liner

```python
from yggdrasil.databricks import DatabricksClient

cluster = DatabricksClient().compute.clusters.all_purpose_cluster(name="shared-etl")
```

## Cluster service features

```python
from yggdrasil.databricks import DatabricksClient

clusters = DatabricksClient(host="https://<workspace>", token="<token>").compute.clusters
```

- Reuse-or-create all-purpose cluster: `cluster = clusters.all_purpose_cluster(name="shared-etl")`
- List clusters: `for c in clusters.list(limit=20): print(c.cluster_name, c.state)`
- Find by name quickly: `cluster = clusters.find_cluster("shared-etl")`
- Create/update from one call: `cluster = clusters.create_or_update(cluster_name="shared-etl", num_workers=2)`
- Pick runtime versions: `clusters.latest_spark_version(photon=True, python_version="3.12")`
- Enumerate compatible runtimes: `clusters.spark_versions(photon=False, allow_ml=True)`

## Execution context examples

```python
from yggdrasil.databricks.compute import ExecutionContext

with ExecutionContext(cluster=cluster) as ctx:
    print(ctx.execute("print('hello from Databricks')"))
```

One-liners:

- `ExecutionContext(cluster=cluster).execute("dbutils.fs.ls('/')")`
- `ExecutionContext(cluster=cluster).execute("import sys; print(sys.version)")`

## Remote decorator

See [compute.remote](remote/README.md) for function-level remote execution with `@databricks_remote_compute`.

## Extended example: create cluster, run code, and terminate

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.compute import ExecutionContext

client = DatabricksClient(host="https://<workspace>", token="<token>")
clusters = client.compute.clusters

cluster = clusters.create_or_update(
    cluster_name="demo-docs-cluster",
    num_workers=1,
)

with ExecutionContext(cluster=cluster) as ctx:
    result = ctx.execute("print('docs smoke test')")
    print(result)

# Optional cleanup when this is an ephemeral cluster
# cluster.delete(wait=True)
```
