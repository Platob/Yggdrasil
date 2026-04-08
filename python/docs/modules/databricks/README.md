# yggdrasil.databricks

Production-ready Databricks toolkit for Yggdrasil.

This package gives you a **single entrypoint** (`DatabricksClient`) and service helpers for:

- SQL execution and warehouse lifecycle
- Unity Catalog navigation (catalog/schema/table/column)
- Compute cluster management and remote execution contexts
- DBFS / Volume / Workspace file operations
- Secrets and IAM administration
- Jobs parameter parsing and typed notebook config
- Genie conversational analytics

---

## 1) Quick start (copy/paste)

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient(host="https://<workspace>", token="<token>")
print(client.sql.execute("SELECT current_user() AS me").to_pandas())
```

One-line style works across services:

- SQL: `DatabricksClient().sql.execute("SELECT 1")`
- Warehouses: `DatabricksClient().warehouses.find_default().start()`
- Catalogs: `DatabricksClient().catalogs["main"]["default"]["orders"]`
- Compute: `DatabricksClient().compute.clusters.all_purpose_cluster(name="etl")`
- FS: `DatabricksClient().dbfs_path("dbfs:/tmp/demo.txt").write_text("ok")`
- Secrets: `DatabricksClient().secrets.create_secret("scope/key", "value")`
- IAM: `next(DatabricksClient().iam.users.list(limit=1), None)`
- Genie: `DatabricksClient().genie.ask("<space-id>", "weekly revenue")`

---

## 2) Authentication patterns

```python
from yggdrasil.databricks import DatabricksClient
```

### PAT token

```python
client = DatabricksClient(host="https://<workspace>", token="<token>")
```

### OAuth client credentials

```python
client = DatabricksClient(
    host="https://<workspace>",
    client_id="<client-id>",
    client_secret="<client-secret>",
)
```

### Environment-driven (best for local + CI)

```python
client = DatabricksClient()  # reads DATABRICKS_* variables
```

Common env vars: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_ACCOUNT_ID`, `DATABRICKS_CLUSTER_ID`, `DATABRICKS_CONFIG_PROFILE`.

---

## 3) End-to-end workflows

### A. SQL + table lifecycle

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient(host="https://<workspace>", token="<token>")

c.sql.execute("CREATE TABLE IF NOT EXISTS main.default.demo (id BIGINT, name STRING) USING DELTA")
c.sql.insert_into("main.default.demo", [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}])

stmt = c.sql.execute("SELECT * FROM main.default.demo ORDER BY id")
print(stmt.to_arrow_table())
print(stmt.to_pandas())
print(stmt.to_polars(stream=False))
```

### B. Files + secrets + SQL in one flow

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient(host="https://<workspace>", token="<token>")

c.secrets.create_secret("demo/api_key", "abc123")
path = c.tmp_path(extension="json")
path.write_text('{"event":"created"}')

c.sql.execute("SELECT current_timestamp() AS ts")
print(path.read_text())
```

### C. Compute execution context

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.compute import ExecutionContext

c = DatabricksClient(host="https://<workspace>", token="<token>")
cluster = c.compute.clusters.create_or_update(cluster_name="docs-cluster", num_workers=1)

with ExecutionContext(cluster=cluster) as ctx:
    print(ctx.execute("print('hello from databricks')"))
```

---

## 4) Service feature map

| Service | What it covers | Best first call |
|---|---|---|
| `client.sql` | Query execution, DDL/DML, result conversion | `client.sql.execute("SELECT 1")` |
| `client.warehouses` | Warehouse discovery/start/stop/update | `client.warehouses.find_default()` |
| `client.catalogs` / `client.tables` | Unity Catalog hierarchy + table resources | `client.catalogs["main"]["default"]["orders"]` |
| `client.compute` | Cluster lifecycle/version selection | `client.compute.clusters.all_purpose_cluster(name="etl")` |
| `client.dbfs_path(...)` | DBFS/Volumes path operations | `client.dbfs_path("dbfs:/tmp/a.txt")` |
| `client.secrets` | Scope/secret CRUD helpers | `client.secrets.create_secret("scope/key", "value")` |
| `client.iam` | Users/groups in workspace/account scope | `client.iam.users.current_user` |
| `client.genie` | Conversational BI workflows | `client.genie.ask("<space-id>", "top customers")` |
| `client.spark_connect()` | Spark Connect session bootstrapping | `spark = client.spark_connect()` |

---

## 5) Troubleshooting

- **Auth errors (`401/403`)**: verify host + token pair, and whether you need workspace scope vs account scope.
- **Warehouse query issues**: ensure a running warehouse exists (`client.warehouses.find_default().start()`).
- **Cluster code execution fails**: verify cluster policy, permissions, and runtime version compatibility.
- **Path not found**: ensure DBFS vs `/Volumes/...` prefixes are correct for the target path type.
- **Optional package missing**: install the right extra (`ygg[databricks]`, `ygg[data]`, `ygg[http]`, etc.).

---

## 6) Full module docs

- [sql](sql/README.md)
- [compute](compute/README.md)
- [compute.remote](compute/remote/README.md)
- [workspaces](workspaces/README.md)
- [fs](fs/README.md)
- [secrets](secrets/README.md)
- [iam](iam/README.md)
- [ai.genie](ai/genie/README.md)
- [jobs](jobs/README.md)
- [account](account/README.md)
