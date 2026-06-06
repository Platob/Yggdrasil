# Skill: run SQL and manage tables

## When to use

The user asks to run SQL, query Databricks, get a DataFrame from a query,
or create / insert / upsert a Unity Catalog table.

## Run SQL

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
result = dbc.sql.execute("SELECT * FROM main.default.orders LIMIT 100")
```

`execute(...)` returns a `StatementResult` — materialise to any engine:

```python
result.to_arrow_table()   # pyarrow.Table
result.to_polars()        # polars.DataFrame
result.to_pandas()        # pandas.DataFrame
result.to_spark()         # pyspark.sql.DataFrame
result.to_pylist()        # list[dict] — only for genuinely small results
```

(These `to_*` names are the public aliases; the underlying `read_*`
methods work too.)

### Parameters

```python
dbc.sql.execute(
    "SELECT * FROM main.default.orders WHERE id = :id",
    parameters={"id": 42},
)
```

Bind parameters — don't f-string user values into SQL. Useful kwargs:
`warehouse_name=` / `warehouse_id=` to pick compute, `engine="spark"` to
run over Spark instead of a SQL warehouse, `row_limit=`.

### Run many

```python
dbc.sql.execute_many([
    "CREATE TABLE IF NOT EXISTS main.default.t (id BIGINT) USING DELTA",
    "INSERT INTO main.default.t SELECT 1",
    "OPTIMIZE main.default.t",
])
```

## Tables

```python
tbl = dbc.tables["main.default.orders"]   # "cat.sch.tbl", or "sch.tbl"/"tbl" with client defaults
```

### Create

```python
from yggdrasil.data import Schema, Field, DataType

schema = Schema([
    Field("id",     DataType.int64(), nullable=False),
    Field("amount", DataType.decimal(18, 2)),
    Field("ts",     DataType.timestamp("UTC")),
])

tbl.ensure_created(schema)        # idempotent; schema is the FIRST positional arg
```

A `pyarrow.Schema` is accepted too: `tbl.ensure_created(pa.schema([...]))`.

### Insert

`insert` accepts Arrow tables/batches, pandas / Polars / Spark frames,
dicts, or lists of dicts — type is auto-detected and cast to the table
schema:

```python
tbl.insert(arrow_table)
tbl.insert(polars_df)
tbl.insert([{"id": 1, "amount": 9.99}, {"id": 2, "amount": 12.0}])
tbl.insert(arrow_table, mode="overwrite")     # replace all rows
```

### Upsert / merge

Match on key columns — yggdrasil builds the MERGE:

```python
tbl.insert(updates, match_by=["id"])                 # update matched, insert new
```

### Large / staged loads

`insert` auto-routes big warehouse writes through staged Parquet on a
Volume. To stage explicitly without running the insert:

```python
staged_path = tbl.stage_insert(arrow_table)          # → VolumePath
```

You can also drive the warehouse path directly: `tbl.arrow_insert(data)`,
or the Spark path: `tbl.spark_insert(data)`.

### Inspect

```python
tbl.exists()
tbl.columns          # list[Column]
tbl.arrow_schema     # pyarrow.Schema
tbl.read_infos()     # fresh TableInfo from the metastore
```

### Rename / clone / delete

```python
tbl.rename("orders_v2")
tbl.clone("main.default.orders_backup")
tbl.delete()         # drop
```

## Engine-level inserts (no Table object)

When you only have a name and data:

```python
dbc.sql.insert_into(arrow_table, location="main.default.orders", match_by=["id"])
dbc.sql.arrow_insert_into(arrow_table, location="main.default.orders")
dbc.sql.spark_insert_into(spark_df,   location="main.default.orders")
```

The data is the first positional arg; the target is the `location=` kwarg.

## Dataset shortcut

For Spark-native reads/writes, use a `Dataset` (see `ygg-spark-tabular`):

```python
dbc.dataset("SELECT * FROM main.raw.events").map(transform).to_table("main.curated.events")
```

## Don'ts

- **No `tbl.merge()`, `tbl.async_insert()`, or `tbl.delete_where()`** —
  they don't exist. Upsert via `insert(data, match_by=[...])`; delete rows
  with a SQL `DELETE` (`dbc.sql.execute("DELETE FROM … WHERE …")`).
- Don't pass `schema=` as a kwarg to `ensure_created` — the schema is the
  first positional argument.
- Don't pre-check `exists()` before inserting — `ensure_created` is
  idempotent.
- Don't loop rows to insert — pass a whole frame to `insert`.
- Don't f-string SQL values — use `parameters={...}`.
