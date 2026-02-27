# yggdrasil.databricks.workspaces

Unified path abstraction for DBFS, Workspace files, and Unity Catalog Volumes.

## Key exports

```python
from yggdrasil.databricks.workspaces import Workspace, DatabricksPath
```

## Supported path schemes

| Scheme | Example |
|---|---|
| DBFS | `dbfs:/tmp/data.parquet` |
| Workspace | `/Workspace/Users/me/file.txt` |
| Unity Catalog Volume | `/Volumes/main/analytics/bronze/data.parquet` |

---

## Bootstrap: connect and parse a path

```python
from yggdrasil.databricks.workspaces import Workspace, DatabricksPath

ws = Workspace(host="https://<workspace>.azuredatabricks.net", token="<pat>").connect()
# or: ws = Workspace().connect()  # uses DATABRICKS_HOST + DATABRICKS_TOKEN

path = DatabricksPath.parse("dbfs:/tmp/orders.parquet", workspace=ws)
```

---

## Bootstrap: read into dataframes

```python
from yggdrasil.databricks.workspaces import DatabricksPath

path = DatabricksPath.parse("dbfs:/curated/users.parquet")

arrow_tbl = path.read_arrow()    # pyarrow.Table
pandas_df = path.read_pandas()   # pandas.DataFrame
polars_df = path.read_polars()   # polars.DataFrame
```

---

## Bootstrap: write and read text

```python
from yggdrasil.databricks.workspaces import DatabricksPath

path = DatabricksPath.parse("dbfs:/tmp/hello.txt")

with path.open("w") as f:
    f.write("hello from yggdrasil")

with path.open("r") as f:
    print(f.read())
```

---

## Bootstrap: copy between storage domains

```python
from yggdrasil.databricks.workspaces import DatabricksPath

src  = DatabricksPath.parse("dbfs:/tmp/raw/events.parquet")
dest = DatabricksPath.parse("/Volumes/main/analytics/bronze/events.parquet")

src.copy_to(dest)    # copy, source remains
# src.move_to(dest)  # move, source is deleted
```

---

## Bootstrap: Unity Catalog Volume read

```python
from yggdrasil.databricks.workspaces import DatabricksPath

path = DatabricksPath.parse("/Volumes/main/analytics/bronze/trades.parquet")
table = path.read_arrow()
```

---

## Bootstrap: Workspace file write

```python
from yggdrasil.databricks.workspaces import DatabricksPath

path = DatabricksPath.parse("/Workspace/Users/me@corp.com/outputs/report.json")

with path.open("w") as f:
    import json
    json.dump({"status": "ok", "rows": 1000}, f)
```

---

## `Workspace` API

```python
ws = Workspace(host=None, token=None)
ws.connect()          # authenticate, returns self
ws.connected          # bool
ws.safe_host          # sanitized host URL
ws.sdk()              # databricks.sdk.WorkspaceClient
```

## `DatabricksPath` API

```python
path = DatabricksPath.parse(uri, workspace=None)

path.open(mode)        # context manager ("r", "w", "rb", "wb")
path.read_arrow()      # → pa.Table
path.read_pandas()     # → pandas.DataFrame
path.read_polars()     # → polars.DataFrame
path.copy_to(dest)     # copy to DatabricksPath
path.move_to(dest)     # move to DatabricksPath
```
