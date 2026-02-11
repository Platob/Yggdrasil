# yggdrasil.databricks.compute

This module focuses on remote compute operations for Databricks clusters.

Use it when your code runs outside notebooks (CI/CD, services, local scripts) and still needs to:
- Resolve or manage target clusters
- Execute commands in cluster contexts
- Reuse cluster connection patterns across pipelines

---

## Core APIs

- `Cluster`: cluster selection, lifecycle control, command execution.
- `ExecutionContext`: scoped remote execution context.
- `databricks_remote_compute`: decorator for function-level remote execution.

---

## Bootstrap: basic cluster command execution

```python
from yggdrasil.databricks.compute import Cluster

cluster = Cluster(cluster_name="shared-etl-cluster")
cluster.ensure_running()

result = cluster.execute("print('hello from databricks cluster')")
print(result)
```

---

## Bootstrap: context-managed remote session

```python
from yggdrasil.databricks.compute import Cluster, ExecutionContext

cluster = Cluster(cluster_name="analytics-jobs")
cluster.ensure_running()

with ExecutionContext(cluster=cluster) as context:
    output = context.execute("spark.range(10).count()")
    print(output)
```

---

## Bootstrap: cluster bootstrap in automation script

```python
from yggdrasil.databricks.compute import Cluster

cluster = Cluster(cluster_name="nightly-maintenance")

# Ensure availability before running batch logic
cluster.ensure_running()

# Run remote administrative script
cluster.execute(
    """
from pyspark.sql import SparkSession
spark = SparkSession.getActiveSession()
print(spark.sql('SELECT current_date()').collect())
"""
)
```

---

## Related submodule

- [compute.remote](remote/README.md): function decorators for remote dispatch.

## Recommendations

- Keep cluster naming conventions deterministic (`env-purpose-size`).
- Prefer context managers for multi-step command execution workflows.
- Add retry and timeout controls in surrounding orchestration logic.
