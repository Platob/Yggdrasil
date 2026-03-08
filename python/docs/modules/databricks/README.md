# yggdrasil.databricks

Databricks integration: workspace auth, SQL, compute, and typed job configuration.

## Submodules

| Module | Key exports |
|---|---|
| [workspaces](workspaces/README.md) | `Workspace`, `DatabricksPath` |
| [sql](sql/README.md) | `SQLEngine`, `StatementResult` |
| [compute](compute/README.md) | `Cluster`, `ExecutionContext` |
| [compute.remote](compute/remote/README.md) | `databricks_remote_compute` |
| [jobs](jobs/README.md) | `NotebookConfig` |

---

## Bootstrap: connect to a workspace

```python
from yggdrasil.databricks.workspaces import Workspace

# Explicit credentials
ws = Workspace(host="https://<workspace>.azuredatabricks.net", token="<pat>").connect()

# Or from environment: DATABRICKS_HOST + DATABRICKS_TOKEN
ws = Workspace().connect()
```

---

## Bootstrap: full ingestion pipeline

```python
from yggdrasil.databricks.workspaces import Workspace, DatabricksPath
from yggdrasil.databricks.sql import SQLEngine

ws = Workspace().connect()

# 1. Read raw source
source = DatabricksPath.parse("dbfs:/pipelines/raw/orders.parquet", client=ws)
orders = source.read_arrow()

# 2. Cast to target schema (optional)
# orders = cast_arrow_tabular(orders, CastOptions(target_field=target_schema))

# 3. Write to Unity Catalog
engine = SQLEngine(catalog_name="main", schema_name="analytics", workspace=ws)
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
    days_back: int = 7

cfg = JobArgs.init_job()   # init widgets + load from environment

engine = SQLEngine(catalog_name=cfg.catalog, schema_name=cfg.schema)
engine.execute(f"""
    DELETE FROM {cfg.catalog}.{cfg.schema}.{cfg.table}
    WHERE event_date >= date_sub(current_date(), {cfg.days_back})
""")
```

---

## Auth patterns

| Scenario | How to configure |
|---|---|
| Personal Access Token | `Workspace(host=..., token=...)` |
| Environment variables | `Workspace()` — reads `DATABRICKS_HOST` / `DATABRICKS_TOKEN` |
| Azure MSI / Service Principal | Set `DATABRICKS_AZURE_*` env vars, use `Workspace()` |
| Inside Databricks runtime | `Workspace()` — auto-detects context |
