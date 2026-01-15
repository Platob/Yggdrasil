# yggdrasil.databricks.workspaces

Workspace, filesystem, and path abstractions for Databricks assets (DBFS, workspace files, Unity Catalog volumes).

## When to use
- You want a unified path abstraction that works across DBFS, workspace, and volumes.
- You need file-like IO for Databricks paths with Arrow/Pandas/Polars helpers.

## Core APIs
- `Workspace` wraps `databricks.sdk.WorkspaceClient` setup and caching.
- `DatabricksPath` is a unified path abstraction with filesystem-style helpers.
- `DatabricksIO` provides file-like read/write utilities for Databricks paths.

```python
from yggdrasil.databricks.workspaces import DatabricksPath, Workspace

workspace = Workspace()
path = DatabricksPath.parse("dbfs:/tmp/demo.txt", workspace=workspace)

with path.open("w") as handle:
    handle.write("hello")
```

## Use cases
### Copy data between DBFS and workspace files
```python
from yggdrasil.databricks.workspaces import DatabricksPath

src = DatabricksPath.parse("dbfs:/tmp/input.csv")
dest = DatabricksPath.parse("/Workspace/Users/me/output.csv")
src.copy_to(dest)
```

### Read a DBFS file as a pandas dataframe
```python
from yggdrasil.databricks.workspaces import DatabricksPath

path = DatabricksPath.parse("dbfs:/tmp/data.parquet")
df = path.read_pandas()
```

## Related modules
- [yggdrasil.databricks.sql](../sql/README.md) for SQL execution helpers.
- [yggdrasil.databricks.compute](../compute/README.md) for cluster execution helpers.
