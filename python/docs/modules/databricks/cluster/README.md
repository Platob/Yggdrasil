# yggdrasil.databricks.cluster

All-purpose cluster lifecycle — find, start, stop, poll state, install libraries, and execute commands remotely.

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

cluster = DatabricksClient().compute.clusters.all_purpose_cluster(name="etl")
print(cluster.state)
```

## Find a cluster

```python
from yggdrasil.databricks import DatabricksClient

client   = DatabricksClient()
clusters = client.compute.clusters

# By name
cluster = clusters.all_purpose_cluster(name="etl-cluster")

# By cluster ID
cluster = clusters.cluster(cluster_id="0601-123456-abc12345")

# List all clusters
for c in clusters.list():
    print(c.name, c.state, c.id)
```

## Inspect state

```python
cluster = DatabricksClient().compute.clusters.all_purpose_cluster(name="etl")

print(cluster.name)
print(cluster.id)
print(cluster.state)           # State.RUNNING / State.TERMINATED / ...
print(cluster.is_running)
print(cluster.spark_version)   # "14.3.x-scala2.12"
print(cluster.runtime_version) # e.g. "14.3"
print(cluster.explore_url)     # Databricks UI link
```

## Start / wait / stop

```python
cluster = DatabricksClient().compute.clusters.all_purpose_cluster(name="etl")

# Start and wait until RUNNING
cluster.start().wait_for_status()

# Refresh state from API
cluster.refresh()
print(cluster.is_running)

# Raise immediately if the cluster is in an error state
cluster.raise_for_status()

# Stop
cluster.stop()
```

## Install libraries

```python
cluster = DatabricksClient().compute.clusters.all_purpose_cluster(name="etl")

cluster.install_libraries([
    {"pypi": {"package": "requests>=2.31"}},
    {"pypi": {"package": "polars==1.0.0"}},
])
```

## Execute a Python command remotely

```python
from yggdrasil.databricks import DatabricksClient

client  = DatabricksClient()
cluster = client.compute.clusters.all_purpose_cluster(name="etl")

result = cluster.execute_python("print('hello from the cluster')")
print(result.stdout)
```

## Remote execution via `@databricks_remote_compute`

The preferred high-level entry point for running a Python callable on a Databricks cluster is the `@databricks_remote_compute` decorator — it serializes the function and its arguments, ships them to the cluster, and deserializes the return value:

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(cluster_name="etl")
def heavy_computation(n: int) -> int:
    return sum(range(n))

result = heavy_computation(1_000_000)
print(result)   # 499999500000
```

See [compute.remote](../compute/remote/README.md) for the full remote execution guide.
