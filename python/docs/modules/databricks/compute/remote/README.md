# yggdrasil.databricks.compute.remote

Decorator for running local functions on Databricks compute.

## Export

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute
```

## Example

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(cluster_name="shared-etl")
def add(x: int, y: int) -> int:
    return x + y
```
