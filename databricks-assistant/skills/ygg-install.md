# Skill: install ygg on Databricks

## When to use

The user asks to install or set up `ygg` / `yggdrasil`, or hits
`ModuleNotFoundError: yggdrasil`.

## Install

```python
%pip install "ygg[databricks]"
dbutils.library.restartPython()
```

- `[databricks]` adds `databricks-sdk`.
- Do **not** add `[bigdata]` — Spark is already on the cluster.
- Add `[data]` for local pandas + numpy + sqlglot.
- Add `[http]` for the HTTP-session helpers (API ingestion).

Combine extras when needed: `"ygg[databricks,http]"`.

### Pin a version

```python
%pip install "ygg[databricks]==0.8.45"
```

Pin in production jobs so a notebook and its scheduled run install the
same build.

### Quick check

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
print(dbc)              # repr shows the resolved host
dbc.sql.execute("SELECT current_user() AS me").to_pylist()
```

## Don'ts

- Don't `pip install yggdrasil` — the PyPI distribution name is `ygg`
  (the import name is `yggdrasil`).
- Don't skip `dbutils.library.restartPython()` after `%pip install`.
- Don't install `databricks-sdk` separately — the `[databricks]` extra
  pulls a compatible version.
