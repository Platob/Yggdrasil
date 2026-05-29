# Skill: connect and use DatabricksClient

## When to use

The user asks to connect to Databricks, authenticate, use a service
(SQL, tables, volumes, jobs, secrets, compute, genie), or needs the
client in a notebook or job.

## Connect

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()  # picks up DATABRICKS_HOST/TOKEN from env
```

Singleton by config — same `(host, auth)` → same instance. Picklable
across Spark workers and job tasks.

### Auth patterns

| Auth | How |
| --- | --- |
| PAT (env) | set `DATABRICKS_HOST` + `DATABRICKS_TOKEN` |
| PAT (explicit) | `DatabricksClient(host="https://…", token="dapi…")` |
| Service principal | set `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` |
| Account-level | `DatabricksClient(account_id="…")` |

Inside a Databricks notebook, host + token are auto-injected.

## Services

```python
dbc.sql           # SQL execution → StatementResult
dbc.tables        # Unity Catalog tables
dbc.schemas       # schemas
dbc.catalogs      # catalogs
dbc.columns       # column metadata
dbc.volumes       # Volume resources
dbc.warehouses    # SQL warehouses
dbc.compute       # clusters + instance pools
dbc.jobs          # Jobs (create, run, wait)
dbc.secrets       # secret scopes
dbc.iam           # users, groups
dbc.genie         # Genie spaces + Q&A
dbc.ai            # vector search
```

## Resource singletons

`Table`, `Volume`, `Schema`, `Catalog`, `Job`, `Cluster`,
`Warehouse`, `Secret` are singleton resources with lifecycle methods:

```python
tbl = dbc.tables["main.default.orders"]
tbl.exists()
tbl.ensure_created(schema=schema)
tbl.read_info()
tbl.delete()

vol = dbc.volumes["main.default.staging"]
vol.ensure_created()
vol.path   # → VolumePath
```

Route through these methods — don't call `ws.tables.create()`
directly.

## SparkTabular / Dataset

```python
ds = dbc.dataset("SELECT * FROM main.sales.orders")
ds = dbc.dataset("main.sales.orders")
ds = dbc.parallelize(fetch_data, urls, schema=output_schema)
```

See the `ygg-spark-tabular` skill for the full Dataset API.

## Don'ts

- Don't `import databricks.sdk` directly — go through `dbc.<service>`.
- Don't construct multiple clients for the same workspace — the
  singleton cache handles it.
- Don't pickle `WorkspaceClient`; pickle `DatabricksClient`.
