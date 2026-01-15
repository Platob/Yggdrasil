# yggdrasil.databricks

Databricks integrations built on the Databricks SDK plus Spark-aware helpers.

## When to use
- You need workspace-aware file/path utilities across DBFS, workspace files, and Unity Catalog volumes.
- You want a thin SQL execution layer that can use the Databricks SQL API or Spark when available.
- You want helper utilities for cluster management or notebook widget configuration.

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

## Compute helpers
`yggdrasil.databricks.compute` provides:
- `Cluster` – create/update clusters, install libraries, and execute commands remotely.
- `ExecutionContext` – helper for remote execution contexts.

```python
from yggdrasil.databricks.compute import Cluster

cluster = Cluster()
cluster.ensure_running()
```

## Jobs and widgets
`yggdrasil.databricks.jobs` focuses on notebook configuration helpers:
- `NotebookConfig` – dataclass base that reads Databricks widgets or job parameters.
- `WidgetType` – enum of supported widget types.

## Notes
- These helpers rely on the Databricks SDK and PySpark when running inside a Spark-enabled environment.
- Arrow-based insert and cast helpers use `yggdrasil.types.cast` for schema enforcement.
