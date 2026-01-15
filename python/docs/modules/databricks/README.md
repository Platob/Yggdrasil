# yggdrasil.databricks

Databricks integrations built on the Databricks SDK plus Spark-aware helpers.

## When to use
- You need workspace-aware file/path utilities across DBFS, workspace files, and Unity Catalog volumes.
- You want a thin SQL execution layer that can use the Databricks SQL API or Spark when available.
- You want helper utilities for cluster management or notebook widget configuration.

## Common use cases
- **Path-aware file access**: read/write data across DBFS, workspace, or Unity Catalog volumes with a single API.
- **SQL automation**: execute SQL statements with structured results and Arrow-friendly metadata.
- **Cluster orchestration**: create clusters and run commands from CI or automation scripts.
- **Notebook configuration**: parse widgets/job parameters into typed dataclasses.

## Submodules
- [workspaces](workspaces/README.md) for workspace, path, and IO abstractions.
- [sql](sql/README.md) for statement execution helpers.
- [compute](compute/README.md) for cluster lifecycle and remote execution utilities.
- [jobs](jobs/README.md) for notebook configuration helpers.

## Workspace and paths
Modules under `yggdrasil.databricks.workspaces` provide:
- `Workspace` – configuration wrapper around `databricks.sdk.WorkspaceClient` with caching and context-manager support.
- `DatabricksPath` / `DatabricksPathKind` – unified path abstraction for DBFS, workspace, and volumes.
- `DatabricksIO` – file-like read/write interfaces that operate on Databricks paths.

```python
from yggdrasil.databricks.workspaces import Workspace, DatabricksPath

workspace = Workspace(host="https://...")
path = DatabricksPath.parse("dbfs:/tmp/demo.txt", workspace=workspace)
```

Use case: copy files between DBFS and workspace paths.

```python
from yggdrasil.databricks.workspaces import DatabricksPath

src = DatabricksPath.parse("dbfs:/tmp/input.csv")
dest = DatabricksPath.parse("/Workspace/Users/me/output.csv")
src.copy_to(dest)
```

## SQL execution
`yggdrasil.databricks.sql` includes `SQLEngine` and `StatementResult` to execute SQL via the SQL Statement Execution API or Spark.

Key capabilities:
- Build fully qualified table names and issue SQL statements.
- Insert Arrow tables or Spark dataframes into Delta tables.
- Convert SQL metadata into Arrow fields.

```python
from yggdrasil.databricks.sql import SQLEngine

engine = SQLEngine(catalog_name="main", schema_name="demo")
engine.execute("SELECT 1 AS value")
```

Use case: insert a small Arrow table into a Delta table.

```python
import pyarrow as pa
from yggdrasil.databricks.sql import SQLEngine

engine = SQLEngine(catalog_name="main", schema_name="demo")
table = pa.table({"id": [1, 2], "value": ["a", "b"]})
engine.insert_into(table, table_name="demo_table", mode="append")
```

## Compute helpers
`yggdrasil.databricks.compute` provides:
- `Cluster` – create/update clusters, install libraries, and execute commands remotely.
- `ExecutionContext` – helper for remote execution contexts.

```python
from yggdrasil.databricks.compute import Cluster

cluster = Cluster()
cluster.ensure_running()
```

Use case: run a small Python snippet on a cluster.

```python
from yggdrasil.databricks.compute import Cluster

cluster = Cluster(cluster_name="demo")
cluster.ensure_running()
cluster.execute("print('hello from cluster')")
```

## Jobs and widgets
`yggdrasil.databricks.jobs` focuses on notebook configuration helpers:
- `NotebookConfig` – dataclass base that reads Databricks widgets or job parameters.
- `WidgetType` – enum of supported widget types.

Use case: parse widget inputs into a typed config.

```python
from yggdrasil.databricks.jobs import NotebookConfig

class IngestConfig(NotebookConfig):
    source: str
    target_table: str

config = IngestConfig.from_environment()
```

## Notes
- These helpers rely on the Databricks SDK and PySpark when running inside a Spark-enabled environment.
- Arrow-based insert and cast helpers use `yggdrasil.types.cast` for schema enforcement.
