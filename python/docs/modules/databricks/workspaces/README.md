# yggdrasil.databricks.workspaces

This module standardizes Databricks filesystem interactions across different storage domains:
- DBFS (`dbfs:/...`)
- Workspace files (`/Workspace/...`)
- Unity Catalog volumes (`/Volumes/...`)

It provides a single API surface for parsing, opening, and transferring files across those domains.

---

## Core components

- `Workspace`: wraps workspace client configuration and access.
- `DatabricksPath`: path abstraction with parse + operation helpers.
- `DatabricksIO`: file-like operations on Databricks paths.

---

## Bootstrap: connect once, reuse everywhere

```python
from yggdrasil.databricks.workspaces import Workspace, DatabricksPath

workspace = Workspace(host="https://<workspace-host>").connect()

path = DatabricksPath.parse("dbfs:/tmp/demo.csv", workspace=workspace)
print(path)
```

---

## Bootstrap: write and read text

```python
from yggdrasil.databricks.workspaces import DatabricksPath

path = DatabricksPath.parse("dbfs:/tmp/bootstrap/hello.txt")

with path.open("w") as handle:
    handle.write("hello from yggdrasil")

with path.open("r") as handle:
    text = handle.read()

print(text)
```

---

## Bootstrap: transfer data between domains

```python
from yggdrasil.databricks.workspaces import DatabricksPath

src = DatabricksPath.parse("dbfs:/tmp/raw/events.parquet")
dest = DatabricksPath.parse("/Volumes/main/analytics/bronze/events.parquet")

src.copy_to(dest)
```

---

## Bootstrap: DataFrame-friendly usage

```python
from yggdrasil.databricks.workspaces import DatabricksPath

# Read parquet into pandas
pdf = DatabricksPath.parse("dbfs:/tmp/curated/users.parquet").read_pandas()

# Read parquet into polars
pl_df = DatabricksPath.parse("dbfs:/tmp/curated/users.parquet").read_polars()

# Read parquet into pyarrow
arrow_tbl = DatabricksPath.parse("dbfs:/tmp/curated/users.parquet").read_arrow()
```

---

## Operational recommendations

- Prefer explicit `DatabricksPath.parse(...)` over string concatenation.
- Centralize workspace initialization and inject it where possible.
- Keep file movement operations (`copy_to`, `move_to`) idempotent in jobs.
