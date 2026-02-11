# yggdrasil.databricks.compute.remote

`databricks_remote_compute` enables decorator-based remote execution, so you can keep local Python functions and run them on Databricks when configured.

This is especially useful for teams that want:
- local development ergonomics
- remote execution in CI/prod
- minimal branching in business logic

---

## Core API

- `databricks_remote_compute(...)`: decorator factory with cluster/workspace controls.

Key options include:
- `cluster_id` / `cluster_name`
- `workspace` (host string or workspace object)
- `env_keys` (forward selected environment variables)
- `force_local` (always execute locally)

---

## Bootstrap: basic remote decoration

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(cluster_name="shared-etl-cluster")
def add(x: int, y: int) -> int:
    return x + y

print(add(2, 3))
```

---

## Bootstrap: explicit workspace host

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(
    workspace="https://<workspace-host>",
    cluster_name="analytics-jobs",
)
def normalize(name: str) -> str:
    return name.strip().lower()
```

---

## Bootstrap: local-safe fallback for tests

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(force_local=True)
def deterministic_logic(value: str) -> str:
    return value.upper()

assert deterministic_logic("ok") == "OK"
```

---

## Bootstrap: environment key forwarding

```python
from yggdrasil.databricks.compute.remote import databricks_remote_compute

@databricks_remote_compute(
    cluster_name="shared-etl-cluster",
    env_keys=["ENV", "LOG_LEVEL", "FEATURE_FLAG_X"],
)
def run_with_runtime_flags() -> str:
    return "done"
```

---

## Behavior notes

- If Databricks host information is unavailable, decorator behavior may remain local.
- In Databricks runtime environments, local execution can be preferred automatically.
- For deterministic CI tests, use `force_local=True`.
