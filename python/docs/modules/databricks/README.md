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
Use `Cluster` to provision and manage clusters, track metadata via custom tags, install libraries, and execute commands directly on the driver node.

```python
from yggdrasil.databricks.compute import Cluster
from yggdrasil.databricks.workspaces import Workspace

cluster = Cluster(
    workspace=Workspace(host="https://...", token="..."),
    metadata={"owner": "platform-team", "env": "dev"},
)

# Create a small, on-demand cluster and persist metadata tags (prefixed with "yggdrasil:")
cluster_id = cluster.create(
    cluster_name="demo-cluster",
    spark_version="14.3.x-scala2.12",
    num_workers=1,
    node_type_id="i3.xlarge",
)

# Discover an existing cluster by name and inspect cached info
demo = cluster.find_cluster(name="demo-cluster")
print(demo.info.metadata)

# Update or prune metadata stored in custom tags
cluster.update_metadata({"purpose": "ad-hoc"}, cluster_id=cluster_id)
cluster.remove_metadata_keys(["env"], cluster_id=cluster_id)

# Manage installed Python packages
cluster.install_python_libraries([
    "polars==1.0.0",
    "pandas>=2.2",
], cluster_id=cluster_id)

# Bump to a runtime advertising Python 3.10
cluster.update_runtime_by_python_version("3.10", cluster_id=cluster_id)

# Ensure the cluster is running and execute a command on the driver
cluster.check_started(cluster_id=cluster_id)
print(
    cluster.execute_command(
        "import platform; print(platform.python_version())",
        cluster_id=cluster_id,
    )
)
```

### Notes
- Functions guard imports with `require_databricks_sdk()` and `require_pyspark()` to give clear errors when dependencies are missing.
- Many methods accept `CastOptions` to control Arrow/Pandas/Spark casting of SQL results.

## Navigation
- [Module overview](../../modules.md)
- [Dataclasses](../dataclasses/README.md)
- [Libs](../libs/README.md)
- [Requests](../requests/README.md)
- [Types](../types/README.md)
- [Databricks](./README.md)
