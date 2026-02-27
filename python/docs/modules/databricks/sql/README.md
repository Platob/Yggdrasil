# yggdrasil.databricks.sql

Unified SQL execution for Databricks — auto-selects Spark SQL (inside runtime) or warehouse Statement API (external).

## Key exports

```python
from yggdrasil.databricks.sql import SQLEngine, StatementResult, SqlStatementError
```

---

## Bootstrap: initialize an engine

```python
from yggdrasil.databricks.workspaces import Workspace
from yggdrasil.databricks.sql import SQLEngine

ws = Workspace().connect()
engine = SQLEngine(catalog_name="main", schema_name="analytics", workspace=ws)
```

---

## Bootstrap: execute and convert results

```python
result = engine.execute("SELECT id, amount FROM transactions LIMIT 100")

arrow_table = result.to_arrow_table()   # pyarrow.Table
pandas_df   = result.to_pandas()        # pandas.DataFrame
```

---

## Bootstrap: DML and DDL

```python
engine.execute("OPTIMIZE main.analytics.events ZORDER BY (user_id)")

engine.execute("""
    CREATE TABLE IF NOT EXISTS main.analytics.dim_market (
        id     BIGINT NOT NULL,
        name   STRING NOT NULL
    )
    USING DELTA
""")
```

---

## Bootstrap: insert Arrow data into Delta

```python
import pyarrow as pa

data = pa.table({
    "id":      [1, 2, 3],
    "country": ["US", "FR", "DE"],
})

engine.insert_into(data, table_name="country_dim", mode="append")
# mode: "append" | "overwrite"
```

---

## Bootstrap: force warehouse API (external orchestrator)

```python
result = engine.execute(
    statement="SELECT current_timestamp() AS ts",
    engine="api",                    # "spark" or "api"
    warehouse_name="analytics_wh",
    wait=True,                       # block until done
)
arrow_table = result.to_arrow_table()
```

---

## Bootstrap: async execution (fire and poll)

```python
result = engine.execute(
    "SELECT * FROM large_table",
    wait=False,     # returns immediately with a handle
)

# ... do other work ...

result.wait()       # block until terminal state
table = result.to_arrow_table()
```

---

## Bootstrap: health check with row limit

```python
result = engine.execute("SELECT 1 AS ok", row_limit=1)
assert result.to_arrow_table().num_rows == 1
```

---

## `SQLEngine.execute` signature

```python
engine.execute(
    statement: str,
    *,
    engine=None,             # "spark" | "api" | None (auto)
    warehouse_name=None,     # warehouse name (API engine)
    warehouse_id=None,       # warehouse ID (API engine)
    wait=True,               # block for completion
    row_limit=None,          # cap returned rows
    catalog_name=None,       # override engine default
    schema_name=None,        # override engine default
) -> StatementResult
```

## `StatementResult` methods

```python
result.to_arrow_table()           # → pa.Table
result.to_pandas()                # → pandas.DataFrame
result.to_arrow_batches()         # → Iterator[pa.RecordBatch]
result.to_arrow_dataset()         # → pyarrow.dataset.Dataset
result.wait()                     # block until done (if wait=False was used)
result.raise_for_status()         # raise SqlStatementError on failure
result.statement_id               # str
result.done                       # bool
result.failed                     # bool
```
