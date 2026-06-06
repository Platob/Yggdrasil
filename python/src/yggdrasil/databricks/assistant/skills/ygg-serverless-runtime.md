# Skill: ygg on Databricks serverless (no CLI)

## When to use

Always, as the baseline for any Databricks work in this workspace — and
specifically whenever you're tempted to open a terminal, run `%sh`, `!cmd`,
or the `ygg` / `databricks` CLI. On serverless you can't, and you don't
need to: every CLI verb has a Python equivalent.

## The two rules

1. **No terminal.** Serverless notebooks and jobs have no shell. `%sh`,
   `!command`, `subprocess`, and the `ygg` / `databricks` CLIs are
   unavailable. Do the work in Python through `DatabricksClient`.
2. **Use the default environment.** This workspace seeds a **pre-built ygg
   image** into the default serverless environment, so `import yggdrasil`
   just works. Don't create a custom environment; don't reinstall unless
   the import fails.

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()          # auth from the serverless runtime
```

If — and only if — `import yggdrasil` raises `ModuleNotFoundError`:

```python
%pip install "ygg[databricks]"
dbutils.library.restartPython()
```

## CLI → Python cheatsheet

Reach for the right side, never the left.

| Instead of the CLI… | Call |
| --- | --- |
| `ygg databricks sql query "…"` | `dbc.sql.execute("…").to_polars()` |
| `ygg databricks fs ls /Volumes/…` | `dbc.path("/Volumes/…").iterdir()` |
| `ygg databricks fs put a b` | `dbc.path(b).write_bytes(local_bytes)` |
| `ygg databricks fs cat p` | `dbc.path(p).read_bytes()` |
| `ygg databricks job run name` | `dbc.jobs.get(name="name").run()` |
| `ygg databricks job runs name` | `dbc.jobs.get(name="name").list_runs()` |
| `ygg databricks warehouses list` | `list(dbc.warehouses.list_warehouses())` |
| `ygg databricks clusters list` | `list(dbc.clusters.list_clusters())` |
| `databricks secrets put` | `dbc.secrets.create_secret(key, value, scope=…)` |
| `pip install ygg` (a shell) | `%pip install "ygg[databricks]"` (only if missing) |

## Don'ts

- Don't run `%sh` / `!cmd` / `subprocess` to call a CLI — it will fail on
  serverless. Use the Python API.
- Don't build a bespoke cluster or environment to "get ygg" — it's already
  on the default serverless image.
- Don't `%pip install` on every run — import first; install only on
  `ModuleNotFoundError`.
- Don't `import databricks.sdk` directly — go through `dbc.<service>`.
