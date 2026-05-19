# Skill: run SQL on Databricks and get an Arrow / Polars / pandas / Spark frame

## When to use

The user asks to "run / execute a query", "query Databricks SQL", "get
a DataFrame from a warehouse", "fetch query results as Arrow / pandas
/ Polars / Spark / a list of dicts", or to bind parameters / stream
results / read into a target schema.

## Primary surface

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
stmt = dbc.sql.execute("SELECT * FROM main.default.orders LIMIT 100")
```

`dbc.sql` is a `SQLEngine`. `execute(...)` returns a
`StatementResult` (subclass of `yggdrasil.data.DataTable`'s `Tabular`)
— a typed handle to the result, not yet materialised.

## Materialise to whichever engine you want

```python
stmt.to_arrow_table()   # pyarrow.Table
stmt.to_pandas()        # pandas.DataFrame
stmt.to_polars()        # polars.DataFrame
stmt.to_spark()         # pyspark.sql.DataFrame
stmt.to_pylist()        # list[dict] — only when rows ARE the endpoint
```

Each materialiser preserves schema intent (names, order, nullability,
nested structure, timezone). Don't post-process the result through
another `.cast()` chain — pass a target schema instead (below).

## Pin a target schema

```python
from yggdrasil.data.cast.options import CastOptions
import yggdrasil.arrow as pa

target = pa.schema([
    pa.field("id",    pa.int64(),   nullable=False),
    pa.field("amount", pa.decimal128(18, 2)),
])

stmt = dbc.sql.execute(
    "SELECT id, amount FROM main.default.orders",
    options=CastOptions(target_field=target),
)
out = stmt.to_arrow_table()  # already conforming to `target`
```

## Parameters / bindings

```python
dbc.sql.execute(
    "SELECT * FROM main.default.orders WHERE id = :id",
    parameters={"id": 42},
)
```

Don't string-format SQL with f-strings; the engine binds parameters
safely.

## Warehouse vs. compute

`SQLEngine` routes to a Databricks SQL warehouse by default. Choose
explicitly:

```python
dbc.sql.execute(q, warehouse_id="…")        # specific warehouse
dbc.sql.execute(q, cluster_id="…")          # all-purpose cluster (Spark Connect)
```

Defaults come from `DATABRICKS_WAREHOUSE_ID`,
`DATABRICKS_CLUSTER_ID`, `DATABRICKS_SERVERLESS_COMPUTE_ID`.

## DML / merges / inserts

For inserts on a target table, prefer the `Table` API (see the
`ygg-databricks-table` skill) — it handles staging, async inserts,
Delta MERGE, prune-by predicates, and schema reconciliation. For
one-off DML, `dbc.sql.execute("DELETE FROM …")` is fine.

## Don'ts

- Don't `to_pylist()` then comprehension over rows for type coercion
  — pass a `target_field` schema and let the cast registry do it.
- Don't open a fresh warehouse / cluster connection per query;
  `SQLEngine` already pools and retries (`retry_sdk_call`).
- Don't pre-check `exists` on a table before querying it — let the
  query fail and handle `NotFound` if needed.
