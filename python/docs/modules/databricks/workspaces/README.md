# yggdrasil.databricks.workspaces

Workspace-oriented client setup, path helpers, and the `Workspace` resource for multi-workspace management.

---

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient()   # reads DATABRICKS_HOST + DATABRICKS_TOKEN from env
print(c.workspace)
```

---

## 1) Authentication patterns

```python
from yggdrasil.databricks import DatabricksClient

# Environment variables (DATABRICKS_HOST, DATABRICKS_TOKEN)
c = DatabricksClient()

# PAT token
c = DatabricksClient(host="https://<workspace>", token="<token>")

# OAuth client credentials (M2M)
c = DatabricksClient(
    host="https://<workspace>",
    client_id="<client-id>",
    client_secret="<client-secret>",
)

# Named profile from ~/.databrickscfg
c = DatabricksClient(profile="staging")

# URL form — useful when client config is serialized
c = DatabricksClient.from_("dbks://:<token>@<workspace-host>/")

# Singleton — always returns the same instance for the same config
c1 = DatabricksClient(host="https://<workspace>", token="<token>")
c2 = DatabricksClient(host="https://<workspace>", token="<token>")
assert c1 is c2   # True — singleton per config
```

---

## 2) Workspace resource

```python
from yggdrasil.databricks import DatabricksClient, Workspace

c = DatabricksClient()
ws = c.workspace

print(ws.workspace_url)
print(ws.workspace_id)
print(ws.deployment_name)
```

---

## 3) Build and navigate file paths

```python
c = DatabricksClient()

# DBFS path
dbfs = c.dbfs_path("dbfs:/tmp/data.parquet")

# Volume path (auto-detected from /Volumes/ prefix)
vol = c.dbfs_path("/Volumes/main/default/landing/events.json")

# Workspace path
ws = c.dbfs_path("/Workspace/Shared/notebooks/etl.py")

# Path navigation
print(dbfs.parent)         # dbfs:/tmp/
print(vol / "subdir")      # /Volumes/main/default/landing/subdir/
print(dbfs.with_suffix(".csv"))
```

---

## 4) Temp managed paths

```python
c = DatabricksClient()

# Auto-managed temp path — cleaned up by clean_tmp_folder
tmp_json = c.tmp_path(extension="json", max_lifetime=1800)   # 30 min TTL
tmp_parquet = c.tmp_path(extension="parquet")

tmp_json.write_text('{"event": "test"}')
print(tmp_json.exists(), tmp_json.read_text())

# Cleanup stale temp files (older than their max_lifetime)
c.clean_tmp_folder()
```

---

## 5) Connect / close lifecycle

```python
c = DatabricksClient()

c.connect()   # warm up connection pool
# ... work ...
c.close()     # release pool

# Or use as context manager
with DatabricksClient() as c:
    print(c.sql.execute("SELECT 1").to_pylist())
```

---

## 6) Multi-workspace client routing

```python
from yggdrasil.databricks import DatabricksClient

WORKSPACES = {
    "prod":    "https://prod.azuredatabricks.net",
    "staging": "https://staging.azuredatabricks.net",
    "dev":     "https://dev.azuredatabricks.net",
}
TOKEN = "<shared-service-principal-token>"

clients = {
    env: DatabricksClient(host=url, token=TOKEN)
    for env, url in WORKSPACES.items()
}

for env, c in clients.items():
    result = c.sql.execute("SELECT current_user()").to_pylist()
    print(f"{env}: {result[0]}")
```

---

## 7) Workspace-level file operations

```python
c = DatabricksClient()

# List Workspace folder
ws_path = c.dbfs_path("/Workspace/Shared/")
for entry in ws_path.ls():
    print(entry.name, "dir" if entry.is_dir() else "file")

# Read a notebook export
nb = c.dbfs_path("/Workspace/Shared/notebooks/etl.py")
source = nb.read_text()
print(source[:200])
```

---

## 8) Listing all workspaces (account-level)

```python
from yggdrasil.databricks import DatabricksClient

# Account-level client lists all workspaces
account = DatabricksClient(
    host="https://accounts.cloud.databricks.com",
    account_id="<account-id>",
    token="<account-token>",
)

for ws in account.workspaces.list():
    print(ws.workspace_name, ws.workspace_id, ws.workspace_status)
```

---

## 9) Pickle round-trip (Spark worker safe)

`DatabricksClient` is singleton-cached and picklable — the same instance is reused when workers unpickle it inside a Spark executor:

```python
import pickle
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient(host="https://<workspace>", token="<token>")
blob = pickle.dumps(c)

c2 = pickle.loads(blob)
assert c2 is c   # same singleton — no new auth round-trip
```
