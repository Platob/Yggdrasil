# Skill: consume query results with `StatementResult` / `Tabular` / `DataTable`

## When to use

The user has a query result, an external SDK response, or any
"rows + schema" object and asks to materialise it, iterate it,
stream it, persist it, project a subset of columns, or hand it off
to another sink. Triggers include "the result of `dbc.sql.execute`",
"convert this `StatementResult`", "stream a large query", "iterate
batches", "save query results to Parquet / Delta / a Volume", or
"why is `.to_pandas()` slow on this result?".

## Primary surface

```python
from yggdrasil.data import DataTable
from yggdrasil.data.statement import StatementResult, PreparedStatement
```

- `DataTable` is the read-side abstraction every engine result
  implements: `schema`, `arrow_schema`, `to_arrow_table()`,
  `to_polars()`, `to_pandas()`, `to_spark()`, `to_pylist()`,
  `iter_record_batches()`.
- `StatementResult` (subclass of `Tabular`) is what
  `dbc.sql.execute(...)` returns — same surface plus the executed
  statement, parameters, and any `CastOptions` that were applied.
- `PreparedStatement` is the explicit "build a statement, execute it
  later" handle when you need to re-run the same query with
  different bindings.

## Materialise once, at the right granularity

```python
stmt = dbc.sql.execute("SELECT id, amount FROM main.default.orders")

stmt.to_arrow_table()        # pyarrow.Table — preferred when staying in Arrow
stmt.to_polars()             # polars.DataFrame — zero-copy from Arrow
stmt.to_pandas()             # pandas.DataFrame — Arrow → pandas C bridge
stmt.to_spark()              # pyspark.sql.DataFrame
```

Don't chain `stmt.to_pandas().to_dict("records")` to get a list of
dicts — call `stmt.to_pylist()` directly, and only when rows ARE
the endpoint (NDJSON / Kafka / Mongo writer, HTTP JSON response).

## Stream large results in batches

```python
for batch in stmt.iter_record_batches(batch_size=50_000):
    sink.write(batch)        # pyarrow.RecordBatch — bounded memory
```

Use `iter_record_batches` (Arrow) or `iter_polars(batch_size=…)`
instead of materialising a 10 GB result through `.to_pandas()`.

## Pin the target schema at execution time

Instead of `stmt.to_arrow_table()` followed by a `cast`:

```python
from yggdrasil.data.cast.options import CastOptions

target = schema.to_arrow()
stmt = dbc.sql.execute(q, options=CastOptions(target_field=target))
out = stmt.to_arrow_table()        # already conforming
```

The cast happens during materialisation, with one pass through the
Arrow C++ runtime — not after, with a Python-level rebuild.

## Re-use a `PreparedStatement` for re-runs

```python
ps = dbc.sql.prepare("SELECT * FROM main.default.orders WHERE id = :id")
ps.execute(parameters={"id": 42}).to_arrow_table()
ps.execute(parameters={"id": 43}).to_arrow_table()
```

`PreparedStatement` caches the compiled query handle and parameter
binding shape — cheaper than calling `dbc.sql.execute(...)` twice.

## Implement `DataTable` for new sources

A new connector (REST API, message queue, custom file format)
should implement `DataTable` (or subclass `StatementResult`) so it
plugs into the same `to_arrow_table()` / `to_polars()` /
`iter_record_batches()` surface every other Yggdrasil consumer
already speaks. Don't invent a parallel `MyResult` class.

## Don'ts

- Don't call `.to_pandas()` then `.to_dict("records")` — use
  `.to_pylist()`, and only at a true row endpoint.
- Don't iterate `stmt.to_arrow_table().to_pylist()` for type
  coercion — pass `CastOptions(target_field=...)` to `execute`.
- Don't materialise a multi-GB result eagerly; stream via
  `iter_record_batches`.
- Don't keep a `StatementResult` alive longer than you need — its
  underlying cursor / staged files release on `__exit__` / disposal.
