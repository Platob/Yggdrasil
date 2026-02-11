# yggdrasil.databricks

`yggdrasil.databricks` is the integration surface for Databricks-first workflows in Yggdrasil.

It is designed for teams who want one Python toolkit that works for:
- Notebook development
- Databricks Jobs orchestration
- CI/CD automation scripts
- Service-style data platform tooling

---

## What this module family solves

### 1) Workspace-aware file operations
Read/write/copy files across DBFS, Workspace files, and Unity Catalog Volumes with one path model.

### 2) SQL execution with environment-aware behavior
Run SQL through Spark when inside a Databricks runtime, or through statement APIs for external orchestration.

### 3) Cluster orchestration and remote execution
Create/resolve clusters, run commands, and execute local functions remotely without rewriting business logic.

### 4) Typed job configuration
Parse widgets/job parameters into strongly typed dataclasses for reproducible notebook contracts.

---

## Submodules

- [workspaces](workspaces/README.md): workspace client, paths, and Databricks IO.
- [sql](sql/README.md): SQL engine + statement results.
- [compute](compute/README.md): cluster lifecycle + execution context.
- [compute.remote](compute/remote/README.md): decorator-based remote function execution.
- [jobs](jobs/README.md): `NotebookConfig` and widget typing.

---

## Bootstrap: minimal setup for Databricks integrations

```python
import os
from yggdrasil.databricks.workspaces import Workspace

# Recommended: set credentials from secrets manager or CI env
os.environ["DATABRICKS_HOST"] = "https://<workspace-host>"
os.environ["DATABRICKS_TOKEN"] = "<token>"

workspace = Workspace().connect()
print(workspace.safe_host)
```

---

## Bootstrap: end-to-end ingestion pattern

```python
from yggdrasil.databricks.workspaces import DatabricksPath
from yggdrasil.databricks.sql import SQLEngine

# 1) Read raw source from DBFS
source = DatabricksPath.parse("dbfs:/pipelines/raw/orders.parquet")
orders = source.read_arrow()

# 2) Apply your transformation layer (pseudo-step)
orders = orders.append_column("source", orders.column("order_id"))

# 3) Write using SQL engine into Unity Catalog table
engine = SQLEngine(catalog_name="main", schema_name="analytics")
engine.insert_into(orders, table_name="orders_curated", mode="append")
```

---

## Bootstrap: production notebook pattern

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig
from yggdrasil.databricks.sql import SQLEngine

@dataclass
class JobArgs(NotebookConfig):
    catalog: str
    schema: str
    table: str
    days_back: int = 1

cfg = JobArgs.from_environment()

engine = SQLEngine(catalog_name=cfg.catalog, schema_name=cfg.schema)
engine.execute(f"""
DELETE FROM {cfg.catalog}.{cfg.schema}.{cfg.table}
WHERE event_date >= date_sub(current_date(), {cfg.days_back})
""")
```

---

## Design guidance

- Use `workspaces` for all path manipulation and file handling.
- Use `sql` for DDL/DML and controlled result retrieval.
- Use `jobs` for parameter contracts and type casting.
- Use `compute` / `compute.remote` only when external execution contexts are required.
