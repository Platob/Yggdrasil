# Skill: install / import ygg on Databricks

## When to use

The user asks to install or set up `ygg` / `yggdrasil`, or hits
`ModuleNotFoundError: yggdrasil`.

## First, just import it

This workspace ships a **pre-built ygg image** on the default serverless
environment (seeded by `ygg databricks seed`). So the normal path is *no
install at all*:

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
dbc.sql.execute("SELECT current_user() AS me").to_pylist()
```

## Only if the import fails

Install into the **default environment** — never a custom one, never a
shell `pip`:

```python
%pip install "ygg[databricks]"
dbutils.library.restartPython()
```

- `[databricks]` adds `databricks-sdk`.
- Do **not** add `[bigdata]` — Spark is already on the cluster.
- Add `[data]` for local pandas + numpy + sqlglot, `[http]` for the
  HTTP-session helpers. Combine extras: `"ygg[databricks,http]"`.

### Pin a version (production jobs)

```python
%pip install "ygg[databricks]==0.8.45"
```

Pin so a notebook and its scheduled run install the same build.

## Don'ts

- Don't `%pip install` reflexively — `import yggdrasil` works on the seeded
  image; install only on `ModuleNotFoundError`.
- Don't run `!pip` / `%sh pip` / the `pip` CLI — serverless can't shell
  out; use the `%pip` magic into the default environment.
- Don't `pip install yggdrasil` — the PyPI distribution name is `ygg`
  (the import name is `yggdrasil`).
- Don't skip `dbutils.library.restartPython()` after `%pip install`.
- Don't install `databricks-sdk` separately — the `[databricks]` extra
  pulls a compatible version.
