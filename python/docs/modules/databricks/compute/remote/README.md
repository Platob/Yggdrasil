# yggdrasil.databricks.compute.remote

Decorator utilities for running Python callables on a Databricks cluster.

## When to use
- You want to run a local function on a remote Databricks cluster without rewriting it.
- You need a simple guard that executes locally when no Databricks host is configured.

## Core API
- `databricks_remote_compute` returns a decorator that dispatches a function to a cluster via the Databricks command execution API.

## Use cases
### Decorate a function for remote execution
```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(cluster_name="demo")
def remote_sum(x, y):
    return x + y

result = remote_sum(2, 3)
```

### Provide an explicit Workspace host
```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(workspace="https://my-workspace.cloud.databricks.com")
def remote_upper(value: str) -> str:
    return value.upper()
```

## Notes
- If `DATABRICKS_HOST` is unset (and no workspace is provided), the decorator becomes a no-op and executes locally.
- Use `force_local=True` to always run locally (useful for tests).
