# yggdrasil.databricks.sql

Unified SQL engine and typed result wrappers.

## Key exports

```python
from yggdrasil.databricks.sql import SQLEngine, StatementResult, SqlStatementError
```

## Execute SQL and consume Arrow/pandas/Polars

```python
from yggdrasil.databricks.workspaces import Workspace
from yggdrasil.databricks.sql import SQLEngine

client = Workspace(host="https://<workspace>", token="<token>")
engine = SQLEngine(client=client, catalog_name="main", schema_name="default")

stmt = engine.execute("SELECT 1 AS value")
print(stmt.to_arrow_table())
print(stmt.to_pandas())
print(stmt.to_polars())
```

## Wait and status handling

```python
stmt.wait()
stmt.raise_for_status()
```
