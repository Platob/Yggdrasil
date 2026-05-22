# Skill: connect to Databricks with `DatabricksClient`

## When to use

The user asks to "connect to Databricks", "set up a workspace
client", "authenticate against Databricks", "switch workspace /
account / service principal", "use a serverless warehouse / cluster
by default", or "pickle a Databricks client across Spark workers /
job tasks".

## Primary surface

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()  # picks up DATABRICKS_HOST/TOKEN from env
```

`DatabricksClient` is a **singleton-by-config**: same
`(host, auth_type, account_id, workspace_id, …)` → same instance. It
is picklable across Spark workers, multiprocessing pools, FastAPI
forks, and Power Query bridges. Hash + equality follow the config,
not `id(self)`, so it is safe as a dict key.

## Common auth patterns

| Auth | How |
| --- | --- |
| PAT (env) | set `DATABRICKS_HOST` + `DATABRICKS_TOKEN`, then `DatabricksClient()` |
| PAT (explicit) | `DatabricksClient(host="https://…", token="dapi…")` |
| Service principal (OAuth M2M) | set `DATABRICKS_HOST` + `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` |
| Azure SP | set `ARM_RESOURCE_ID` + `ARM_CLIENT_ID` + `ARM_CLIENT_SECRET` |
| Account-level | `DatabricksClient(account_id="…")` |
| Default compute | set `DATABRICKS_CLUSTER_ID` or `DATABRICKS_SERVERLESS_COMPUTE_ID` |

Inside a Databricks notebook the host/token are usually injected, so
`DatabricksClient()` with no args works.

## Services hung off the client

Don't reach for `databricks.sdk.WorkspaceClient` directly — go
through the client's service properties:

```python
dbc.sql                # SQLEngine — execute statements, return Tabular
dbc.catalogs           # Unity Catalog catalogs
dbc.schemas            # schemas
dbc.tables             # tables (read_info, describe, ensure_created)
dbc.columns            # column-level metadata
dbc.volumes            # Volume resources
dbc.warehouses         # SQL warehouses
dbc.compute            # all-purpose / job clusters
dbc.clusters           # alias for compute clusters
dbc.jobs               # Jobs (run_now, wait, etc.)
dbc.secrets            # secret scopes
dbc.iam                # users, groups, service principals
dbc.workspaces         # Workspaces (account-scoped)
dbc.genie              # Genie spaces
```

Each service uses `ExpiringDict` for its internal cache (TTL,
thread-safe, picklable). Don't add a parallel dict + lock — extend
the existing cache.

## Resource singletons

`Volume` / `Schema` / `Catalog` / `Table` / `Warehouse` / `Cluster` /
`Job` / `Secret` / `WorkspaceFile` are singleton resources. Use
**their own** `.create(...)`, `.delete(...)`, `.read_info(...)`,
`.ensure_created(...)`, `.exists` methods — they wrap the underlying
SDK with project defaults (managed-volume-type, owner / comment
normalization), the `_store_infos` cache warm-up, and `missing_ok`
/ `missing_ok` ergonomics.

```python
vol = dbc.volume("main.default.staging")  # builds the singleton
vol.ensure_created()                       # create if missing, idempotent
vol.exists                                  # bool, uses stat cache
```

**Don't** call `ws.volumes.create(...)` / `ws.schemas.create(...)`
directly from feature code — even from inside the resource class's
own `_ensure_X` paths. Route through `self.create(...)`.

## Don'ts

- Don't construct one `DatabricksClient` per call site — let the
  singleton cache hand you the same instance.
- Don't subclass `databricks.sdk.WorkspaceClient` to add helpers; add
  them to the appropriate `DatabricksService` subclass in
  `yggdrasil.databricks`.
- Don't pickle a `WorkspaceClient`; pickle `DatabricksClient` (the
  workspace handle rebuilds from config on the worker).
