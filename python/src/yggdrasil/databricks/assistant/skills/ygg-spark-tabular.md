# Skill: distributed transforms with Dataset (SparkTabular)

## When to use

The user asks to run a function across Spark executors, transform a Spark
DataFrame with yggdrasil, map/filter/apply over distributed data, collect
results as Arrow/Polars/pandas, or write a transform pipeline to Delta.

## Create a Dataset

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()

# From SQL or a table name (auto-detected)
ds = dbc.dataset("SELECT * FROM main.sales.orders")
ds = dbc.dataset("main.sales.orders")

# Distribute a function over inputs — inputs FIRST, function SECOND
ds = dbc.parallelize(urls, fetch_data, schema=output_schema)

# Wrap inputs as a Dataset without a function
ds = dbc.parallelize(rows, schema=output_schema)
```

Or build directly without a client (pass the Spark session):

```python
from yggdrasil.spark.tabular import SparkDataset

ds = SparkDataset.from_sql("SELECT 1", spark_session=spark)
ds = SparkDataset.from_table("main.sales.orders", spark_session=spark)
ds = SparkDataset.from_iterable([{"a": 1}, {"a": 2}], schema=my_schema, spark_session=spark)
ds = SparkDataset.parallelize(inputs, fetch, schema=output_schema, spark_session=spark)
```

## Transforms

```python
ds = ds.map(transform_fn, schema=output_schema)      # 1:1 row transform
ds = ds.apply(transform_fn, schema=output_schema)    # rich/vectorised batch transform
ds = ds.filter(lambda row: row["status"] == "active")
ds = ds.explode(schema=output_schema)                # flatten iterable rows (dynamic mode)
ds = ds.cast(target_schema)                          # recast to a new schema
```

Chain them:

```python
(dbc.dataset("main.raw.events")
   .map(clean)
   .filter(lambda row: row["amount"] > 0)
   .cast(curated_schema)
   .to_table("main.curated.events", mode="overwrite"))
```

## Collect to the driver

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

## Write to a table

```python
ds.to_table("main.curated.events", mode="overwrite")           # or mode="append"
ds.to_table("main.curated.events", format="delta", partition_by=["dt"])
```

## Typed vs dynamic mode

- **Typed** (`schema` set) — rows are dicts matching the schema; transforms
  go through `mapInArrow` for vectorised batches. Prefer when the schema is
  known.
- **Dynamic** (`schema=None`) — rows are arbitrary pickled Python objects.
  Good for heterogeneous data. Discover the shape with `infer_schema()`,
  then `cast(schema)` to switch to typed.

```python
ds = dbc.parallelize(urls, fetch)       # dynamic
schema = ds.infer_schema()              # discover the shape
ds = ds.cast(schema)                    # → typed
ds.to_table("main.raw.fetched")
```

## Executor dependency shipping

Dataset auto-ships `yggdrasil` plus your function's imports to Spark
executors — no manual `addArtifacts`. Imports are resolved by AST walking
the function the first time a transform uses it.

## Caching

```python
ds = ds.persist()      # cache on executors (skip-if-already-cached)
# ... reuse ds ...
ds.unpersist()         # release
```

## Don'ts

- Don't loop over rows in Python — use `map`, `apply`, or `filter`.
- Don't `collect()` a large frame to the driver — use `to_table()` or
  `to_local_iterator()`.
- Don't build `SparkSession.builder` yourself — use `dbc.spark()` or let
  the Dataset resolve the session.
- Remember `parallelize` is `(inputs, function)` — inputs come first.
