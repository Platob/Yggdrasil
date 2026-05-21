# Skill: install Yggdrasil in a Databricks notebook or job

## When to use

The user asks to install / set up / pin `ygg`, `yggdrasil`,
`ygg[data]`, or `ygg[databricks]` in a notebook, on a cluster, or in a
job task; or asks "how do I get Yggdrasil on this cluster?"; or hits
`ModuleNotFoundError: yggdrasil` / `databricks-sdk` not found.

## What to do

### In a notebook (cell-scoped)

```python
%pip install "ygg[data,databricks]"
dbutils.library.restartPython()
```

- `[data]` adds pandas + numpy + sqlglot.
- `[databricks]` adds `databricks-sdk>=0.107`.
- **Do not** add `[bigdata]` on a Databricks runtime — Spark is
  already on the cluster.

### Pinning a version

```python
%pip install "ygg[data,databricks]==0.7.93"
```

Check the latest at https://pypi.org/project/ygg/.

### On a job cluster (libraries config)

In the cluster's **Libraries** tab, add a PyPI library with the
coordinate `ygg[data,databricks]`. For job tasks, set the same on the
task's environment / dependencies block.

### Auth (env vars)

`DatabricksClient` picks these up automatically:

| Var | Purpose |
| --- | --- |
| `DATABRICKS_HOST` | workspace URL |
| `DATABRICKS_TOKEN` | PAT (notebook-side; jobs get one auto-injected) |
| `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` | service principal |
| `DATABRICKS_CLUSTER_ID` / `DATABRICKS_SERVERLESS_COMPUTE_ID` | default compute |

Inside a notebook on Databricks the host + token are usually injected
already; `DatabricksClient()` with no args picks them up.

## Quick check

```python
from yggdrasil.databricks import DatabricksClient
dbc = DatabricksClient()
print(dbc)  # __repr__ confirms host + auth_type
```

## Don'ts

- Don't `pip install yggdrasil` — the PyPI name is `ygg`.
- Don't skip `dbutils.library.restartPython()` after `%pip install`;
  the runtime keeps the old `yggdrasil` (often partial) loaded
  otherwise.
- Don't install `databricks-sdk` separately — `[databricks]` pins it
  at a version Yggdrasil tests against.
