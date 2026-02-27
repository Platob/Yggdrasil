# yggdrasil.databricks.compute.remote

`@databricks_remote_compute` — run a local Python function on a Databricks cluster with no business-logic changes.

## Key export

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute
```

---

## Bootstrap: decorate a function for remote execution

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(cluster_name="shared-etl")
def add(x: int, y: int) -> int:
    return x + y

result = add(2, 3)   # executes on cluster, returns 5
```

---

## Bootstrap: explicit workspace host

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(
    workspace="https://<workspace>.azuredatabricks.net",
    cluster_name="analytics-jobs",
)
def normalize(name: str) -> str:
    return name.strip().lower()
```

---

## Bootstrap: forward environment variables

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(
    cluster_name="shared-etl",
    env_keys=["ENV", "LOG_LEVEL", "FEATURE_FLAG_X"],
)
def pipeline_step() -> dict:
    import os
    return {"env": os.getenv("ENV"), "log": os.getenv("LOG_LEVEL")}
```

---

## Bootstrap: local-safe fallback (tests / CI without cluster)

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(force_local=True)
def transform(value: str) -> str:
    return value.strip().upper()

assert transform("  hello  ") == "HELLO"
```

---

## Decorator options

```python
@databricks_remote_compute(
    cluster_name=None,      # cluster resolved by name
    cluster_id=None,        # cluster resolved by ID
    workspace=None,         # host string or Workspace instance
    env_keys=None,          # list[str] — env vars to forward
    force_local=False,      # True: always execute locally (for tests)
)
```

**Execution rules:**
- If `force_local=True` → always local.
- If running inside a Databricks notebook → local (cluster already available).
- Otherwise → serializes and executes on the specified cluster.
