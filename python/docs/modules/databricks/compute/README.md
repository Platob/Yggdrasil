# yggdrasil.databricks.compute

Cluster and remote execution helpers.

## Exports

```python
from yggdrasil.databricks.compute import Cluster, ExecutionContext, databricks_remote_compute
```

## Cluster + execution context

```python
from yggdrasil.databricks.compute import Cluster, ExecutionContext

cluster = Cluster(cluster_name="shared-etl")

with ExecutionContext(cluster=cluster) as ctx:
    out = ctx.execute("print('hello from Databricks')")
    print(out)
```
