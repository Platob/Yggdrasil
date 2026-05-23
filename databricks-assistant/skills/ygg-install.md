# Skill: install ygg on Databricks

## When to use

The user asks to install or set up `ygg` / `yggdrasil`, or hits
`ModuleNotFoundError`.

## Install

```python
%pip install "ygg[databricks]"
dbutils.library.restartPython()
```

- `[databricks]` adds `databricks-sdk`.
- Do **not** add `[bigdata]` — Spark is already on the cluster.
- Add `[data]` when you need pandas + numpy + sqlglot.
- Add `[http]` when you need HTTP session features for API ingestion.

### Pin a version

```python
%pip install "ygg[databricks]==0.7.94"
```

### Quick check

```python
from yggdrasil.databricks import DatabricksClient
dbc = DatabricksClient()
print(dbc)
```

## Don'ts

- Don't `pip install yggdrasil` — the PyPI name is `ygg`.
- Don't skip `dbutils.library.restartPython()` after `%pip install`.
- Don't install `databricks-sdk` separately.
