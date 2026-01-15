# yggdrasil.databricks.compute

Helpers for managing Databricks clusters and running remote compute workloads.

## When to use
- You need to create/update clusters with shared configuration defaults.
- You want to execute code in remote Databricks execution contexts.

## Core APIs
- `Cluster` wraps cluster lifecycle operations (ensure running, install libraries, submit commands).
- `ExecutionContext` models a remote execution context for running commands on a cluster.
- `databricks_remote_compute` builds a Spark-aware remote execution helper.

## Submodules
- [remote](remote/README.md) for the decorator that runs local functions on a Databricks cluster.

```python
from yggdrasil.databricks.compute import Cluster

cluster = Cluster(cluster_name="demo")
cluster.ensure_running()
```

## Use cases
### Run a Python snippet on a cluster
```python
from yggdrasil.databricks.compute import Cluster

cluster = Cluster(cluster_name="demo")
cluster.ensure_running()
cluster.execute("print('hello from cluster')")
```

### Execute with a shared execution context
```python
from yggdrasil.databricks.compute import Cluster, ExecutionContext

cluster = Cluster(cluster_name="demo")
with ExecutionContext(cluster=cluster) as context:
    context.execute("spark.range(5).count()")
```

## Related modules
- [yggdrasil.databricks.workspaces](../workspaces/README.md) for workspace configuration helpers.
- [yggdrasil.databricks.sql](../sql/README.md) for SQL execution against Databricks warehouses.
