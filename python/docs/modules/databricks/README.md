# yggdrasil.databricks

Helpers built on the Databricks SDK for workspaces, SQL execution, jobs, and compute.

## When to use
- You need a lightweight wrapper around Databricks SDK clients with sane defaults.
- You want typed SQL execution that returns Arrow tables or Spark DataFrames.
- You manage clusters, jobs, or workspace files programmatically.

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

### Tips
- Requires the `databricks-sdk` extra. For Azure-hosted workspaces, ensure the `host` points to the correct region-specific URL.
- File helpers support DBFS and Unity Catalog volume prefixes; choose the storage type that matches your workspace permissions.

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

data = pa.table({"id": [1, 2], "name": ["alice", "bob"]})
engine.insert_into(
    data,
    catalog_name="main",
    schema_name="demo",
    table_name="users",
)
```

### Common pitfalls
- Ensure the workspace has permission to create the temporary volume used for Arrow ingestion.
- When using `match_by`, the key columns must exist in the incoming data and target table.
- Casting options control how timestamps/timezones are handled—pass `CastOptions` if you need strict enforcement.

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

cluster_id = cluster.create(
    cluster_name="demo-cluster",
    spark_version="14.3.x-scala2.12",
    num_workers=1,
    node_type_id="i3.xlarge",
)

cluster.check_started(cluster_id=cluster_id)
print(
    cluster.execute_command(
        "import platform; print(platform.python_version())",
        cluster_id=cluster_id,
    )
)
```

### Common operations
- **Create or update a cluster with sensible defaults**

```python
from yggdrasil.databricks.compute import Cluster
from yggdrasil.databricks.workspaces import Workspace
import datetime as dt

cluster = Cluster(Workspace(host="https://...", token="..."))

cluster.create_or_update(
    cluster_name="ci-dev",
    spark_version="17.3.x-scala2.13",
    num_workers=1,
    node_type_id="i3.xlarge",
    autotermination_minutes=30,
)

cluster.ensure_running()  # starts the cluster if needed
```

- **Install local or PyPI libraries onto the cluster**

```python
# Install a local package from the current environment
cluster.install_libraries(libraries=["my_package"], upload_local_lib=True)

# Install directly from PyPI without building a wheel locally
cluster.install_libraries(libraries=["pydantic==2.8.0"])
```

- **Execute code remotely on the driver node**

```python
# Run a one-off Python statement
output = cluster.execute_command(
    "import platform; print(platform.python_version())",
    cluster_id=cluster.cluster_id,
)
print(output)

# Send a function to run remotely with automatic serialization
@cluster.remote_execute(timeout=dt.timedelta(seconds=10))
def remote_sum(x, y):
    return x + y

print(remote_sum(2, 3))
```

- **Restart or delete clusters when finished**

```python
cluster.restart()
cluster.delete()
```

### Notes
- Functions guard imports with `require_databricks_sdk()` and `require_pyspark()` to give clear errors when dependencies are missing.
- Many methods accept `CastOptions` to control Arrow/Pandas/Spark casting of SQL results.

## Related modules
- [yggdrasil.libs](../libs/README.md) for Spark dependency guards.
- [yggdrasil.types](../types/README.md) for casting Arrow/Spark data.
