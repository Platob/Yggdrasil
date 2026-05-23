# Skill: distributed transforms with Dataset (SparkTabular)

## When to use

The user asks to run a function across Spark executors, transform a
Spark DataFrame with yggdrasil, map/filter/apply over distributed
data, collect results as Arrow/Polars/pandas, or write a transform
pipeline to a Delta table.

## Create a Dataset

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()

# From SQL or table name (auto-detected)
ds = dbc.dataset("SELECT * FROM main.sales.orders")
ds = dbc.dataset("main.sales.orders")

# Distribute a function over inputs via Spark executors
ds = dbc.parallelize(fetch_data, urls, schema=output_schema)
```

Or directly without a client:

```python
from yggdrasil.spark.tabular import Dataset

ds = Dataset.from_sql("SELECT 1", spark_session=spark)
ds = Dataset.from_table("main.sales.orders", spark_session=spark)
ds = Dataset.from_iterable([{"a": 1}, {"a": 2}], schema=my_schema, spark_session=spark)
ds = Dataset.parallelize(fn, inputs, schema=output_schema, spark_session=spark)
```

## Transforms

```python
# 1:1 row transform
ds = ds.map(transform_fn, schema=output_schema)

# Batch-level vectorised transform
ds = ds.apply(transform_fn, schema=output_schema)

# Filter rows
ds = ds.filter(lambda row: row["status"] == "active")

# Flatten iterables into rows
ds = ds.explode(schema=output_schema)

# Re-cast to a new schema
ds = ds.cast(target_schema)
```

Chain them:

```python
(dbc.dataset("main.raw.events")
 .map(clean)
 .filter(lambda row: row["amount"] > 0)
 .cast(curated_schema)
 .to_table("main.curated.events"))
```

## Collect to driver

```python
arrow_table = ds.toArrow()
polars_df   = ds.toPolars()
pandas_df   = ds.toPandas()
rows        = ds.collect()              # list of Python objects
count       = ds.count()
```

For large results, stream instead of collecting:

```python
for row in ds.to_local_iterator():
    process(row)
```

## Write to Delta

```python
ds.to_table("main.curated.events", mode="overwrite")
ds.to_table("main.curated.events", mode="append")
```

## Dynamic vs typed mode

- **Typed** (`schema` set) — rows are dicts matching the schema.
  Transforms go through `mapInArrow` for vectorised batch processing.
  Prefer this when the schema is known.
- **Dynamic** (`schema=None`) — rows are arbitrary pickled Python
  objects. Good for heterogeneous data. Use `infer_schema()` to
  discover the shape, then `cast(schema)` to switch to typed mode.

```python
ds = dbc.parallelize(fetch, urls)       # dynamic
schema = ds.infer_schema()              # discover
ds = ds.cast(schema)                    # → typed
ds.to_table("main.raw.fetched")
```

## Executor dependency shipping

Dataset auto-ships `yggdrasil` and user-function dependencies to
Spark executors. No manual `addArtifacts` needed — the first
transform that uses a function resolves imports via AST walking.

## Caching

```python
ds = ds.persist()     # cache on executors (idempotent)
# ... reuse ds multiple times ...
ds.unpersist()        # release executor cache
```

## Don'ts

- Don't loop over rows in Python — use `map`, `apply`, or `filter`.
- Don't `collect()` large frames to the driver — use `to_table()` or
  `to_local_iterator()`.
- Don't call `SparkSession.builder` yourself — use `dbc.spark()` or
  let Dataset resolve the session.
