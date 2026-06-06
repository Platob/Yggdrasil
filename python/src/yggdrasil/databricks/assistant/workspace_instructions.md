# Workspace instructions — `ygg[databricks]`

This workspace uses [Yggdrasil](https://github.com/Platob/Yggdrasil)
(PyPI name: `ygg`, import: `yggdrasil`). Write notebook and job code
against `yggdrasil.databricks`, not the raw `databricks-sdk`.

## Runtime — you are running on serverless

You (the Assistant) execute Python in a **serverless notebook on the
pre-built ygg image**. Two hard rules follow from that:

1. **Never use a terminal.** Serverless cannot shell out — no `%sh`, no
   `!command`, and **no `ygg` / `databricks` CLI**. Everything the CLI does
   is also a Python call on `DatabricksClient`; use that. (When you catch
   yourself reaching for `ygg databricks …`, stop and call the matching
   `dbc.<service>` method instead — see the table below.)
2. **ygg is already installed** — the workspace seeds a pre-built ygg image
   into the default serverless environment (`ygg databricks seed`). Just
   `import yggdrasil`. Only if the import fails, install into the **default
   environment** once:

   ```python
   %pip install "ygg[databricks]"
   dbutils.library.restartPython()
   ```

   Don't add `[bigdata]` on a cluster (Spark is already there); don't pin a
   custom environment — use the default one.

## Connect

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()   # host/token come from the serverless runtime
```

- **Singleton by config** — same constructor args → the same instance.
- **Picklable** — safe to capture in Spark workers and job tasks (it
  re-authenticates from the runtime on the other side).
- Reuse the process-global instance with `DatabricksClient.current()`.

Outside a notebook, set `DATABRICKS_HOST` + `DATABRICKS_TOKEN` (PAT) or
`DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` (OAuth M2M), or pass
`host=` / `token=` / `profile=` / `client_id=` explicitly.

## Services — the CLI replacement

Every terminal task maps to a Python call. Reach everything through
`dbc.<service>` — never `import databricks.sdk`, never the CLI.

| You want… | Use (Python — not the CLI) |
| --- | --- |
| Run SQL | `dbc.sql.execute(q)` → `StatementResult` |
| Read/write a table | `dbc.tables["cat.sch.tbl"]` → `Table` |
| Catalogs / schemas / columns | `dbc.catalogs`, `dbc.schemas`, `dbc.columns` |
| Volumes | `dbc.volumes["cat.sch.vol"]` → `Volume` |
| Files (Volumes/DBFS/Workspace) | `dbc.path(uri)` / `DatabricksPath.from_(uri)` |
| Warehouses / compute | `dbc.warehouses`, `dbc.compute` (`dbc.clusters`) |
| Jobs & runs | `dbc.jobs`, `dbc.job_runs` |
| Secrets | `dbc.secrets` |
| Users / groups | `dbc.iam` |
| Vector Search | `dbc.ai.vector_search` |
| Distributed Spark transforms | `dbc.dataset(...)`, `dbc.parallelize(...)` |

## SQL

```python
result = dbc.sql.execute(
    "SELECT * FROM main.default.orders WHERE id = :id",
    parameters={"id": 42},          # bind params — never f-string SQL
)

result.to_arrow_table()   # pyarrow.Table
result.to_polars()        # polars.DataFrame
result.to_pandas()        # pandas.DataFrame
result.to_spark()         # pyspark.sql.DataFrame
result.to_pylist()        # list[dict] — only for genuinely small results

dbc.sql.execute_many([
    "CREATE TABLE IF NOT EXISTS main.default.t (id BIGINT) USING DELTA",
    "INSERT INTO main.default.t SELECT 1",
])
```

## Tables

```python
from yggdrasil.data import Schema, Field, DataType

tbl = dbc.tables["main.default.orders"]

# Create (idempotent). Schema is the first positional arg — a yggdrasil
# Schema or a pyarrow.Schema.
tbl.ensure_created(Schema([
    Field("id",     DataType.int64(), nullable=False),
    Field("amount", DataType.decimal(18, 2)),
    Field("ts",     DataType.timestamp("UTC")),
]))

# Insert — accepts Arrow / pandas / Polars / Spark frames, dicts, or lists.
tbl.insert(arrow_table)
tbl.insert([{"id": 1, "amount": 9.99}, {"id": 2, "amount": 12.0}])

# Upsert / merge — match on key columns.
tbl.insert(updates, match_by=["id"])

# Full overwrite.
tbl.insert(arrow_table, mode="overwrite")
```

`Table` also exposes `exists()`, `columns`, `arrow_schema`,
`read_infos()`, `rename(...)`, `clone(...)`, `delete(...)`. For large
warehouse loads it stages Parquet to a Volume automatically
(`arrow_insert` / `stage_insert`).

## Files

```python
from yggdrasil.databricks import DatabricksPath, VolumePath, DBFSPath, WorkspacePath

# Auto-dispatch by shape (or use dbc.path(uri)):
p = DatabricksPath.from_("/Volumes/main/default/staging/data.parquet")  # VolumePath
p = DatabricksPath.from_("dbfs:/tmp/data.parquet")                      # DBFSPath
p = DatabricksPath.from_("/Workspace/Users/me/script.py")               # WorkspacePath

p.read_bytes(); p.write_bytes(b"...")     # write replaces by default
p.read_text();  p.write_text("...")
p.exists(); p.is_dir(); p.is_file(); p.size
p.iterdir(); p.ls(recursive=True)
p.mkdir(parents=True, exist_ok=True)
p.unlink(missing_ok=True)                 # file
p.remove(recursive=True)                  # tree
child = p.parent / "other.parquet"        # join with /
```

Paths carry their bound client and are picklable. Stat is cached ~60s.
There is **no** `glob()` and **no** `rename()` on paths — use `ls`/`iterdir`
and `remove`/`unlink`.

### Volumes

```python
vol = dbc.volumes["main.default.staging"]
vol.create()                              # idempotent (or vol.get_or_create())
dst = vol.path("orders/2026-05-19.parquet")   # .path(sub) is a METHOD → VolumePath
dst.write_bytes(parquet_bytes)
```

## Distributed transforms (`Dataset` / SparkTabular)

```python
ds = dbc.dataset("SELECT * FROM main.sales.orders")   # query or table name
ds = dbc.parallelize(urls, fetch_fn, schema=out_schema)  # inputs FIRST, then fn

(dbc.dataset("main.raw.events")
   .map(clean)                                   # 1:1 row transform
   .filter(lambda row: row["amount"] > 0)
   .to_table("main.curated.events", mode="overwrite"))

ds.toArrow(); ds.toPolars(); ds.toPandas()        # collect to driver
for row in ds.to_local_iterator(): ...            # stream, bounded memory
```

Methods: `map`, `apply`, `filter`, `explode`, `cast`, `to_table`,
`toArrow`/`toPolars`/`toPandas`, `collect`, `count`, `to_local_iterator`,
`infer_schema`, `persist`/`unpersist`. Dataset auto-ships `yggdrasil` +
your function's imports to executors.

## Jobs

```python
# Run an existing job
job = dbc.jobs.get(name="my-pipeline")
run = job.run()                # → JobRun (awaitable); or job.run_and_wait()
run.wait()
run.result_state               # SUCCESS / FAILED

# Define & deploy with @task / @flow (import from databricks.job)
from yggdrasil.databricks.job import task, flow

@task
def ingest(date: str) -> str:
    from yggdrasil.databricks import DatabricksClient
    dbc = DatabricksClient()
    dbc.dataset(f"SELECT * FROM vendor.raw WHERE date = '{date}'") \
       .map(clean).to_table("main.curated.events")
    return "main.curated.events"

@flow(name="daily-ingest")
def daily(date: str = "2025-01-01"):
    ingest(date)

daily.deploy(dbc)              # upsert the Databricks Job (serverless, ygg image)
daily(date="2026-05-23")       # run it (in-process inside Databricks, else remote)
```

`@task` / `@flow` default to **serverless** and ship your live code as a
wheel against the seeded ygg image — no cluster wiring, no CLI. Schedules
use the SDK's `CronSchedule` (quartz), passed to `create_or_update` or as
`@flow(trigger=...)`.

## Secrets

```python
dbc.secrets.create_scope("vendor")
dbc.secrets.create_secret("api-key", "<value>", scope="vendor")
dbc.secrets["vendor/api-key"]            # dict-style access → Secret
```

## Rules

- **No terminal / CLI.** Serverless can't run `%sh`, `!cmd`, or the
  `ygg` / `databricks` CLI — every such task is a `dbc.<service>` call.
- **Use the default environment.** ygg is pre-installed on the serverless
  image; `import yggdrasil` and go. Don't build a custom environment.
- **No row-by-row Python loops over data.** Vectorise with
  `pyarrow.compute`, Polars expressions, or `Dataset` transforms.
- **Don't pre-check then act.** `ensure_created` / `vol.create` are
  idempotent; do the op and handle errors, don't `exists()` first.
- **Route through resource methods** — `tbl.insert()`, `vol.create()`,
  `tbl.ensure_created()` — not raw `dbc.workspace_client().tables...`.
- **Upsert with `match_by=[keys]`**, not a hand-written MERGE. There is no
  `tbl.merge()` / `tbl.async_insert()` / `tbl.delete_where()`.
- **Use `yggdrasil.pickle.json`** instead of stdlib `json` — it wraps
  `orjson` with datetime / UUID / Path / Enum coverage.
