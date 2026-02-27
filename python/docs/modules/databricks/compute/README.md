# yggdrasil.databricks.compute

Cluster lifecycle management and remote command execution for Databricks.

Use from CI/CD, local scripts, or orchestration services that need to target a running cluster.

## Key exports

```python
from yggdrasil.databricks.compute import Cluster, ExecutionContext
```

---

## Bootstrap: start a cluster and run a command

```python
from yggdrasil.databricks.compute import Cluster

cluster = Cluster(cluster_name="shared-etl")
cluster.ensure_running()   # starts cluster if stopped, waits until ready

output = cluster.execute("print('hello from cluster')")
print(output)
```

---

## Bootstrap: context-managed multi-step execution

```python
from yggdrasil.databricks.compute import Cluster, ExecutionContext

cluster = Cluster(cluster_name="analytics-jobs")
cluster.ensure_running()

with ExecutionContext(cluster=cluster) as ctx:
    ctx.execute("import pandas as pd; print(pd.__version__)")
    count = ctx.execute("spark.range(10).count()")
    print(count)
# context is closed automatically
```

---

## Bootstrap: Spark SQL via remote cluster

```python
from yggdrasil.databricks.compute import Cluster

cluster = Cluster(cluster_name="prod-etl")
cluster.ensure_running()

cluster.execute("""
spark = SparkSession.getActiveSession()
result = spark.sql("SELECT count(*) FROM main.analytics.events").collect()
print(result)
""")
```

---

## Bootstrap: automation / nightly script

```python
from yggdrasil.databricks.workspaces import Workspace
from yggdrasil.databricks.compute import Cluster

ws = Workspace().connect()
cluster = Cluster(cluster_name="nightly-maintenance", workspace=ws)
cluster.ensure_running()

cluster.execute("spark.sql('OPTIMIZE main.analytics.events ZORDER BY (user_id)')")
```

---

## `Cluster` API

```python
Cluster(
    cluster_name=None,    # resolve by name
    cluster_id=None,      # resolve by ID
    workspace=None,       # Workspace instance
)

cluster.ensure_running()          # start + wait if not running
cluster.execute(command: str)     # run Python/Spark code string on cluster
```

## `ExecutionContext` API

```python
with ExecutionContext(cluster=cluster) as ctx:
    ctx.execute(command: str)   # run command in the context
# context destroyed on exit
```

---

## Related

- [compute.remote](remote/README.md) — decorator-based remote dispatch
