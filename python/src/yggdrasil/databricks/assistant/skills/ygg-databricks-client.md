# Skill: connect and use DatabricksClient

## When to use

The user asks to connect to Databricks, authenticate, reach a service
(SQL, tables, volumes, jobs, secrets, compute, IAM, vector search), or
needs the client inside a notebook or job.

## Connect

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()   # inside a serverless notebook: auth from the runtime
```

- **Singleton by config** — same constructor args → the same instance
  (process-wide). `DatabricksClient.current()` returns the global one.
- **Picklable** — capture it in Spark workers / job tasks; it
  re-authenticates from the runtime on the far side.

> On serverless you don't install anything first — ygg is on the pre-built
> image (see `ygg-serverless-runtime`). And you never reach for the `ygg` /
> `databricks` CLI: every service below is a plain Python call.

### Auth patterns

| Auth | How |
| --- | --- |
| Runtime (notebook/job) | nothing — host + token are injected |
| PAT (env) | set `DATABRICKS_HOST` + `DATABRICKS_TOKEN` |
| PAT (explicit) | `DatabricksClient(host="https://…", token="dapi…")` |
| OAuth M2M (env) | set `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` |
| OAuth M2M (explicit) | `DatabricksClient(client_id="…", client_secret="…")` |
| Config profile | `DatabricksClient(profile="prod")` (`~/.databrickscfg`) |
| Account-level | `DatabricksClient(account_id="…")` |

You can also default the working catalog/schema:
`DatabricksClient(catalog_name="main", schema_name="default")`.

## Services

Reach everything through `dbc.<service>` — never `import databricks.sdk`,
never the CLI.

```python
dbc.sql            # SQL execution → StatementResult
dbc.tables         # Unity Catalog tables (dbc.views is an alias)
dbc.schemas        # schemas
dbc.catalogs       # catalogs
dbc.columns        # column metadata
dbc.volumes        # Volume resources
dbc.warehouses     # SQL warehouses
dbc.compute        # clusters + instance pools  (dbc.clusters == dbc.compute.clusters)
dbc.jobs           # Jobs (create_or_update, get, run)
dbc.job_runs       # Job runs
dbc.secrets        # secret scopes + secrets
dbc.iam            # users, groups
dbc.ai             # dbc.ai.vector_search
dbc.genie          # AI/BI Genie spaces (ask questions, manage spaces)
dbc.external       # external locations + storage credentials
dbc.entity_tags    # UC entity tags
```

Plus helpers: `dbc.dataset(...)`, `dbc.parallelize(...)`, `dbc.spark()`,
`dbc.path(...)`, `dbc.tmp_path(...)`, `dbc.open(...)`.

## Resource objects

`Table`, `Volume`, `Schema`, `Catalog`, `Job`, `JobRun`, `Cluster`,
`Warehouse` are resource objects with lifecycle methods. Get them through
the service, then call methods on them:

```python
tbl = dbc.tables["main.default.orders"]
tbl.exists()
tbl.ensure_created(schema)          # schema is positional
tbl.read_infos()
tbl.delete()

vol = dbc.volumes["main.default.staging"]
vol.create()                        # idempotent (or vol.get_or_create())
vol.path("sub/file.parquet")        # → VolumePath
```

## Secrets

```python
dbc.secrets.create_scope("vendor")
dbc.secrets.create_secret("api-key", "<value>", scope="vendor")

dbc.secrets["vendor/api-key"]       # dict-style read → Secret
dbc.secrets.secret("api-key", scope="vendor")
list(dbc.secrets.list_scopes())
```

## Distributed work

```python
ds = dbc.dataset("SELECT * FROM main.sales.orders")   # → SparkDataset
results = dbc.parallelize(urls, fetch, schema=out)    # inputs first, fn second
```

See the `ygg-spark-tabular` skill for the full Dataset API.

## Don'ts

- Don't `import databricks.sdk` directly — go through `dbc.<service>`.
  (The raw SDK clients are reachable via `dbc.workspace_client()` if you
  truly need them, but prefer the service layer.)
- Don't shell out to the `ygg` / `databricks` CLI — serverless can't, and
  every CLI verb is a `dbc.<service>` method.
- Don't build multiple clients for one workspace — the singleton handles it.
- Don't use `dbc.dbfs_path(...)` — it's deprecated; use `dbc.path(...)`.
