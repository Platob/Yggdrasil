# Skill: read/write files on Databricks filesystems

## When to use

The user asks to read / write / list / delete files on Unity Catalog
Volumes, DBFS, or the Workspace tree, or to stage data for a table load.

## DatabricksPath — one entry point

```python
from yggdrasil.databricks import DatabricksPath, VolumePath, DBFSPath, WorkspacePath

# Auto-dispatch by path shape:
p = DatabricksPath.from_("/Volumes/main/default/staging/data.parquet")  # → VolumePath
p = DatabricksPath.from_("dbfs:/tmp/data.parquet")                      # → DBFSPath
p = DatabricksPath.from_("/Workspace/Users/me/script.py")               # → WorkspacePath
```

Or through the client (carries auth for you):

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
p = dbc.path("/Volumes/main/default/staging/data.parquet")
```

## Read / write

```python
data = p.read_bytes()
p.write_bytes(b"content")            # whole-content write replaces by default

text = p.read_text()
p.write_text("content")

with p.open("rb") as f:
    chunk = f.read()
```

## Directory operations

```python
p.exists()                           # bool (stat cached ~60s)
p.is_dir(); p.is_file()
p.size                               # file size in bytes
p.full_path(); p.name; p.parent

p.iterdir()                          # immediate children
p.ls(recursive=True)                 # recursive walk

p.mkdir(parents=True, exist_ok=True)
p.unlink(missing_ok=True)            # remove a file
p.remove(recursive=True)             # remove a tree

child = p.parent / "other.parquet"   # join with /
```

There is **no** `glob()` and **no** `rename()` on paths. Filter the
results of `iterdir()` / `ls()` yourself; to "move", read+write to the new
path then `unlink` the old.

## Volumes — preferred for Unity Catalog

```python
vol = dbc.volumes["main.default.staging"]
vol.create()                         # idempotent (or vol.get_or_create())

dst = vol.path("orders/2026-05-19.parquet")   # .path(sub) is a METHOD → VolumePath
dst.write_bytes(parquet_bytes)

for child in vol.path().iterdir():   # vol.path() with no arg = volume root
    print(child.full_path())
```

## Stage Parquet for a table load

```python
vol = dbc.volumes["main.default.staging"]
vol.create()
(vol.path("batch.parquet")).write_bytes(arrow_buf.getvalue())

# Or let the Table stage + load for you:
tbl = dbc.tables["main.default.orders"]
tbl.arrow_insert(arrow_table)        # stages Parquet to a Volume, then loads
```

## Path properties

- Paths carry their bound `DatabricksClient`.
- Picklable and hashable — safe across Spark workers and job tasks.
- Stat is cached with a ~60s TTL (and excluded from the pickle).
- Paths are **not** singleton-cached (the UC *resources* like `Volume`
  and `Table` are).

## Don'ts

- Don't pass raw path strings around when a `DatabricksPath` carries the
  client + cache.
- Don't `exists()` before `read_bytes()` — do the read and handle the
  not-found error if it matters.
- Don't reach for `glob()` / `rename()` — they aren't implemented; use
  `iterdir`/`ls` and read+write+`unlink`.
- Don't call `dbc.dbfs_path(...)` — deprecated; use `dbc.path(...)`.
