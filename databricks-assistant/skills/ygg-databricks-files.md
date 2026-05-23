# Skill: read/write files on Databricks and other filesystems

## When to use

The user asks to read/write/list/delete files on DBFS, Unity Catalog
Volumes, the Workspace tree, or to stage data for table inserts.

## DatabricksPath — universal entry point

```python
from yggdrasil.databricks import DatabricksPath, VolumePath, DBFSPath, WorkspacePath

# Auto-dispatch by path shape
p = DatabricksPath.from_("/Volumes/main/default/staging/data.parquet")  # → VolumePath
p = DatabricksPath.from_("dbfs:/tmp/data.parquet")                      # → DBFSPath
p = DatabricksPath.from_("/Workspace/Users/me/script.py")               # → WorkspacePath
```

Or via the client:

```python
from yggdrasil.databricks import DatabricksClient
dbc = DatabricksClient()

p = dbc.dbfs_path("dbfs:/tmp/data.parquet")
```

## Read / write

```python
# Bytes
data = p.read_bytes()
p.write_bytes(b"content")

# Text
text = p.read_text()
p.write_text("content")

# File-like
with p.open("rb") as f:
    content = f.read()
```

## Directory operations

```python
p.exists                           # bool (stat-cached)
p.is_dir()
p.is_file()
p.size                             # file size in bytes

p.iterdir()                        # iterate children
p.glob("*.parquet")                # pattern matching

p.mkdir(parents=True, exist_ok=True)
p.unlink(missing_ok=True)          # delete file
```

## Volume paths

Volumes are the preferred filesystem for Unity Catalog workloads:

```python
vol = dbc.volumes["main.default.staging"]
vol.ensure_created()

# Navigate
dst = vol.path / "orders" / "2026-05-19.parquet"
dst.write_bytes(parquet_bytes)

# List contents
for child in vol.path.iterdir():
    print(child)
```

## Stage Parquet for table inserts

```python
vol = dbc.volumes["main.default.staging"]
vol.ensure_created()

dst = vol.path / "batch.parquet"
dst.write_bytes(arrow_buf.getvalue())

# Then insert via Table.async_insert or COPY INTO
tbl = dbc.tables["main.default.orders"]
tbl.async_insert(arrow_table, staging_volume="main.default.staging")
```

## Path properties

- Paths carry their bound `DatabricksClient` automatically.
- Singleton-cached: same `(client, url)` → same instance.
- Picklable and hashable — safe across Spark workers and job tasks.
- Stat results are cached with a 5-minute TTL.

## Don'ts

- Don't pass raw strings around when a `DatabricksPath` carries the
  client + cache.
- Don't `exists()` before `read_bytes()` — do the read, catch
  `NotFoundError` if needed. The retry policy handles transient errors.
- Don't call `ws.files.upload(...)` directly — use
  `path.write_bytes(...)` for retry + cache invalidation.
