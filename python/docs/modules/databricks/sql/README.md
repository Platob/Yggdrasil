# yggdrasil.databricks.sql

Databricks SQL execution helpers for statement execution and Spark-backed workflows.

## When to use
- You need to execute SQL against Databricks SQL warehouses or clusters.
- You want structured results that preserve schema metadata.

## Core APIs
- `SQLEngine` wraps statement execution and provides convenience helpers for fully qualified table names.
- `StatementResult` captures rows plus metadata for Arrow conversion.

```python
from yggdrasil.databricks.sql import SQLEngine

engine = SQLEngine(catalog_name="main", schema_name="demo")
engine.execute("SELECT 1 AS value")
```

## Use cases
### Execute a statement and fetch Arrow data
```python
from yggdrasil.databricks.sql import SQLEngine

engine = SQLEngine(catalog_name="main", schema_name="demo")
result = engine.execute("SELECT 1 AS value")
table = result.to_arrow_table()
```

### Insert an Arrow table into Delta
```python
import pyarrow as pa
from yggdrasil.databricks.sql import SQLEngine

engine = SQLEngine(catalog_name="main", schema_name="demo")
table = pa.table({"id": [1, 2], "value": ["a", "b"]})
engine.insert_into(table, table_name="demo_table", mode="append")
```

## Related modules
- [yggdrasil.databricks.workspaces](../workspaces/README.md) for workspace configuration.
