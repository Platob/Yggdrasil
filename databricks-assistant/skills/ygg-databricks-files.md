# Skill: read / write DBFS, Unity Catalog Volumes, and Workspace files

## When to use

The user asks to read / write / list / delete a file on DBFS, on a
Unity Catalog Volume, in the Workspace tree, or to stage Parquet /
JSON / bytes for a later table insert. Triggers include
"`dbfs:/...`", "`/Volumes/...`", "Volume path", "workspace file",
"upload a file", "download a notebook", "iterate a directory".

## Primary surface

```python
from yggdrasil.databricks import (
    DatabricksClient, DatabricksPath, DBFSPath, VolumePath, WorkspacePath,
)

dbc = DatabricksClient()
```

`DatabricksPath.from_(...)` is the universal entry point — give it a
URL or string, get back the right subclass:

```python
DatabricksPath.from_("dbfs:/tmp/raw.parquet")
# -> DBFSPath

DatabricksPath.from_("/Volumes/main/default/staging/raw.parquet")
# -> VolumePath

DatabricksPath.from_("/Users/me/notebooks/setup.py")
# -> WorkspacePath
```

Or build directly via the client:

```python
dbc.dbfs_path("dbfs:/tmp/raw.parquet")
dbc.volume_path("/Volumes/main/default/staging/raw.parquet")
dbc.workspace_path("/Users/me/notebooks/setup.py")
```

## Common operations

```python
p = dbc.volume_path("/Volumes/main/default/staging/raw.parquet")

p.exists                  # cached stat
p.read_bytes()            # bytes
p.read_text()             # str
p.write_bytes(b"…")       # bytes
p.write_text("…")         # str
p.open("rb")              # file-like, integrates with yggdrasil.io.BytesIO
p.iterdir()               # iterable of DatabricksPath children
p.unlink(missing_ok=True) # delete
p.mkdir(parents=True, exist_ok=True)
```

Paths carry their bound `DatabricksClient` — `__repr__` shows the
full identity. They are picklable + hashable; same `(client, url)` →
same instance.

## Staging Parquet for table inserts

```python
vol = dbc.volume("main.default.staging")
vol.ensure_created()

dst = vol.path / "orders/2026-05-19.parquet"
dst.write_bytes(arrow_buf.getvalue())

# then hand `dst` to Table.async_insert / Table.copy_into / SQL COPY INTO.
```

See the `ygg-databricks-table` skill for the insert side.

## Fail fast, don't pre-check

Don't gate a `download` / `upload` / `delete` / `read_bytes` /
`iterdir` on a preceding `exists()` / `stat()` / HEAD probe — the
probe doubles the round trip, races concurrent writers, and lies
under eventual consistency. Catch `NotFound` / `FileNotFoundError`
from the real call instead. The library's retry policy
(`retry_sdk_call`) absorbs transient 5xx / throttling / connect
timeouts automatically; deterministic errors propagate immediately.

After a mutation, the path's stat cache auto-invalidates — don't
manually clear it.

## Don'ts

- Don't pass raw `dbfs:/...` / `/Volumes/...` strings around if a
  `DatabricksPath` would carry the bound client + cache.
- Don't loop `read_bytes()` over many small files — batch via
  `iterdir()` + parallel `Job` / `concurrent.futures` (see the
  `yggdrasil.concurrent` module).
- Don't call `ws.files.upload(...)` / `ws.dbfs.put(...)` directly
  when a `path.write_bytes(...)` does the same thing with retry +
  cache invalidation.
