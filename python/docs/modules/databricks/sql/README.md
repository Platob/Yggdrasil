# yggdrasil.databricks.sql

`yggdrasil.databricks.sql` offers a unified SQL execution surface for Databricks automation.

It can execute through:
- **Spark SQL** (inside Databricks runtime with active Spark session)
- **Databricks SQL statement API** (external or warehouse-oriented execution)

---

## Core APIs

- `SQLEngine`: query execution, table targeting, and insert helpers.
- `StatementResult`: structured results + conversion helpers.
- `SqlStatementError`: typed error for statement failures.

---

## Bootstrap: initialize engine

```python
from yggdrasil.databricks.sql import SQLEngine

engine = SQLEngine(
    catalog_name="main",
    schema_name="analytics",
)
```

---

## Bootstrap: execute SQL and inspect results

```python
from yggdrasil.databricks.sql import SQLEngine

engine = SQLEngine(catalog_name="main", schema_name="analytics")
result = engine.execute("SELECT 1 AS healthcheck")

# Useful access patterns
print(result.status)
print(result.statement_id)
print(result.rows)
```

---

## Bootstrap: convert to Arrow for downstream processing

```python
from yggdrasil.databricks.sql import SQLEngine

engine = SQLEngine(catalog_name="main", schema_name="analytics")
result = engine.execute("SELECT id, amount FROM main.analytics.transactions LIMIT 100")
arrow_table = result.to_arrow_table()
```

---

## Bootstrap: load Arrow table to Delta

```python
import pyarrow as pa
from yggdrasil.databricks.sql import SQLEngine

engine = SQLEngine(catalog_name="main", schema_name="analytics")

data = pa.table({
    "id": [1, 2, 3],
    "country": ["US", "FR", "IN"],
})

engine.insert_into(data, table_name="country_dim", mode="append")
```

---

## Bootstrap: external orchestration mode (API-first)

```python
from yggdrasil.databricks.sql import SQLEngine

engine = SQLEngine(catalog_name="main", schema_name="analytics")

result = engine.execute(
    statement="SELECT current_timestamp() AS ts",
    engine="api",  # force SQL statement API
    warehouse_name="analytics_wh",
    wait=True,
)
```

---

## Best practices

- Keep `catalog_name` and `schema_name` explicit in pipelines.
- Use `engine="api"` when running from external orchestrators without Spark.
- Use `row_limit` for lightweight checks and validation probes.
