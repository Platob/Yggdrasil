# Skill: run SQL and manage tables

## When to use

The user asks to run SQL, query Databricks, get a DataFrame from a
query, create/insert/merge a table, or work with `StatementResult`.

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
result.to_pylist()        # list[dict] — only for genuine row endpoints
```

### Parameters

```python
dbc.sql.execute(
    "SELECT * FROM orders WHERE id = :id",
    parameters={"id": 42},
)
```

Don't f-string SQL — use parameter binding.

### Run many

```python
batch = dbc.sql.execute_many([
    "CREATE TABLE IF NOT EXISTS ...",
    "INSERT INTO ... SELECT ...",
    "OPTIMIZE ...",
])
```

## Tables

```python
tbl = dbc.tables["main.default.orders"]
```

### Create

```python
from yggdrasil.data import Schema, Field, DataType

schema = Schema.from_fields([
    Field("id", DataType.int64(), nullable=False),
    Field("amount", DataType.decimal(18, 2)),
    Field("ts", DataType.timestamp("UTC")),
])

tbl.ensure_created(schema=schema)   # idempotent
```

### Insert

Accepts Arrow, pandas, Polars, Spark frames, or dicts:

```python
tbl.insert(arrow_table)
tbl.insert(polars_df)
tbl.insert({"id": [1, 2], "amount": [9.99, 12.00]})
```

### Merge / upsert

```python
tbl.merge(data, keys=["id"], update_columns=["amount", "ts"])
```

### Async insert (large data via Volume staging)

```python
job = tbl.async_insert(data, staging_volume="main.default.staging")
job.wait()
```

### Delete

```python
tbl.delete_where("ts < '2024-01-01'")
tbl.delete()   # drop table
```

### Inspect

```python
tbl.exists
tbl.schema         # yggdrasil Schema
tbl.arrow_schema   # pyarrow.Schema
tbl.columns        # list[Field]
tbl.read_info()    # refresh metadata
```

## Dataset shortcut

For Spark-native reads and writes, use Dataset:

```python
ds = dbc.dataset("SELECT * FROM main.raw.events")
ds.map(transform).to_table("main.curated.events")
```

See the `ygg-spark-tabular` skill.

## Don'ts

- Don't pre-check `exists()` before inserting — `ensure_created` is
  idempotent.
- Don't loop rows to insert — pass a frame to `insert` / `merge`.
- Don't call `ws.tables.create(...)` directly — use
  `tbl.ensure_created()`.
- Don't `to_pylist()` for type coercion — pass a target schema via
  `CastOptions(target_field=...)`.
