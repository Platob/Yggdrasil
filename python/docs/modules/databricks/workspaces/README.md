# yggdrasil.databricks.workspaces

Workspace client + path and filesystem abstractions (with path exports now sourced from `yggdrasil.databricks.fs`).

## Common exports

```python
from yggdrasil.databricks.workspaces import Workspace, DatabricksPath, DBFSPath, VolumePath
```

## Parse and work with paths

```python
from yggdrasil.databricks.workspaces import DatabricksPath

p = DatabricksPath.parse("dbfs:/tmp/example.parquet")
print(type(p).__name__, p)
```

## Build client from environment or explicit credentials

```python
from yggdrasil.databricks.workspaces import Workspace

client = Workspace()  # uses DATABRICKS_* env vars when present
# or
client = Workspace(host="https://<workspace>", token="<token>")
```
