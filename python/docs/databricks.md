# yggdrasil.databricks

Helpers built on the Databricks SDK for workspaces, SQL execution, jobs, and compute.

## Workspaces (`yggdrasil.databricks.workspaces`)
- `Workspace` dataclass wraps `databricks.sdk.WorkspaceClient` configuration (host/token, Azure/GCP settings) and provides helpers to upload/download, list, and delete files across DBFS, Unity Catalog Volumes, and Workspace paths.
- `WorkspaceObject` tracks remote file metadata and provides convenience methods for reading/writing.
- `AuthType` enumerates supported auth flows (e.g., external-browser).

```python
from yggdrasil.databricks.workspaces import Workspace

ws = Workspace(host="https://...", token="...")
with ws.client() as sdk:
    print(sdk.current_user.me())
```

## SQL (`yggdrasil.databricks.sql`)
- `SQLEngine(workspace, **kwargs)` issues SQL statements via the Databricks SQL API.
- `StatementResult` encapsulates statement status, waiting/polling, and fetching results as Arrow tables/record batches with casting helpers (`cast_spark_dataframe`, `convert`).
- `column_info_to_arrow_field` maps SQL metadata to Arrow fields.

```python
from yggdrasil.databricks.sql import SQLEngine
from yggdrasil.databricks.workspaces import Workspace

engine = SQLEngine(Workspace(host="https://...", token="..."))
result = engine.execute("SELECT 1 AS v").wait(engine)
print(result.arrow_table().to_pandas())
```

### Inserting data
- Use `SQLEngine.insert_into(data, ...)` to load Arrow tables/record batches or Spark DataFrames into Delta tables.
- Arrow inputs land in a temporary Unity Catalog volume as Parquet before `INSERT INTO`/`MERGE` statements run; Spark inputs call DataFrame writes directly.
- Set `match_by` to perform an upsert/merge on key columns; omit it for plain appends or specify `mode="overwrite"` for replace semantics.
- Optional optimization hooks: `zorder_by` (Z-ORDER), `optimize_after_merge` (OPTIMIZE), and `vacuum_hours` (VACUUM) are issued after inserts.

```python
import pyarrow as pa
from yggdrasil.databricks.sql import SQLEngine
from yggdrasil.databricks.workspaces import Workspace

engine = SQLEngine(Workspace(host="https://...", token="..."))

# Append with automatic table creation if it does not exist
data = pa.table({"id": [1, 2], "name": ["alice", "bob"]})
engine.insert_into(
    data,
    catalog_name="main",
    schema_name="demo",
    table_name="users",
)

# Upsert on matching keys and run post-merge optimization
engine.insert_into(
    data,
    catalog_name="main",
    schema_name="demo",
    table_name="users",
    match_by=["id"],  # MERGE ... WHEN MATCHED/NOT MATCHED
    optimize_after_merge=True,
    zorder_by=["id"],
    vacuum_hours=168,  # keep 7 days of history
)

# Insert from Spark DataFrame
from pyspark.sql import SparkSession

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
df = spark.createDataFrame([(3, "carol")], ["id", "name"])

engine.insert_into(
    df,
    catalog_name="main",
    schema_name="demo",
    table_name="users",
    match_by=["id"],
)
```

## Jobs (`yggdrasil.databricks.jobs`)
- `DBXJobSettings` and `TaskConfig` dataclasses (via `config.py`) to compose job/task payloads.
- Utilities for building `Task` structures with defaults and type-safe casting.

## Compute (`yggdrasil.databricks.compute`)
- `DBXCluster` dataclass and `DBXCompute` helpers to create/get/delete clusters and manage remote file uploads for Spark jobs.

### Notes
- Functions guard imports with `require_databricks_sdk()` and `require_pyspark()` to give clear errors when dependencies are missing.
- Many methods accept `CastOptions` to control Arrow/Pandas/Spark casting of SQL results.
