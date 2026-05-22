# yggdrasil.io.path

Backend-agnostic filesystem abstractions — a `Path` that unifies local
disk, DBFS, S3, Volumes, and WorkspaceFiles behind a single
`pathlib`-like interface.

## Surface map

| Class | Backed by |
|---|---|
| `Path` | Abstract URL-addressed byte holder |
| `LocalPath` | `os` syscalls — local disk |
| `RemotePath` | Abstract base for all network-backed backends |
| (registered) `DBFSPath` / `VolumePath` / `WorkspacePath` | Databricks backends |
| (registered) `S3Path` | AWS S3 |

Backends register themselves via `Path._register(scheme, cls)` on import.
`Path.from_(url)` dispatches to the right subclass by URL scheme; defaults
to `LocalPath` for bare paths with no scheme.

---

## 1) One-liners

```python
from yggdrasil.io.path import Path, LocalPath

# Local file
p = LocalPath.from_path("/tmp/data.parquet")

# Auto-dispatch by URL scheme (dbfs:// → DBFSPath, s3:// → S3Path, etc.)
p = Path.from_("dbfs:/tmp/events.parquet")

# Read / write text
p = LocalPath.from_path("/tmp/demo.txt")
p.write_text("hello, world")
print(p.read_text())
```

---

## 2) `LocalPath` — local filesystem

```python
from yggdrasil.io.path import LocalPath

# Construction
p = LocalPath.from_path("/data/events")

# pathlib-like navigation
child = p / "2026-05" / "events.parquet"
print(child.name)       # "events.parquet"
print(child.suffix)     # ".parquet"
print(child.parent)     # /data/events/2026-05

# Stat
print(child.exists)
print(child.is_file)
print(child.is_dir)
stat = child.stat()     # IOStats(kind, size, mtime)
print(stat.size)        # bytes
print(stat.mtime)       # datetime

# Directory ops
p.mkdir(parents=True, exist_ok=True)
for item in p.iterdir():
    print(item)

# Read / write bytes
child.write_bytes(b"raw payload")
data = child.read_bytes()

# Read / write text
child.write_text("csv header\na,b\n1,2")
text = child.read_text()

# Read / write Arrow (Parquet round-trip)
import pyarrow as pa
table = pa.table({"id": [1, 2, 3], "val": [0.1, 0.2, 0.3]})
child.write_arrow_table(table)
back = child.read_arrow_table()

# Delete
child.unlink()
p.rmtree()
```

---

## 3) `Path.from_()` — scheme-dispatched construction

```python
from yggdrasil.io.path import Path

# Local — no scheme or file://
local = Path.from_("/tmp/data.parquet")

# Databricks — requires yggdrasil.databricks.fs to be imported
import yggdrasil.databricks.fs   # registers dbfs:// VolumePath WorkspacePath
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
dbfs   = Path.from_("dbfs:/tmp/demo.txt",            client=client)
vol    = Path.from_("/Volumes/main/raw/landing/a.pq", client=client)

# AWS S3 — requires yggdrasil.aws.fs to be imported
import yggdrasil.aws.fs          # registers s3://
from yggdrasil.aws import AWSClient
s3 = Path.from_("s3://my-bucket/prefix/data.parquet", client=AWSClient())
```

---

## 4) Common operations on any `Path`

All of these work identically on `LocalPath`, `VolumePath`, `S3Path`, etc.

```python
from yggdrasil.io.path import Path

p: Path = ...   # any concrete backend

# Pure-path API (delegates to URL — no I/O)
print(p.name)
print(p.stem)
print(p.suffix)
print(p.parent)
child = p / "subdir" / "file.txt"
renamed = child.with_suffix(".parquet")

# Predicates (one I/O round-trip each)
print(p.exists)
print(p.is_file)
print(p.is_dir)
print(p.stat().size)

# Byte I/O
p.write_bytes(b"payload")
raw = p.read_bytes()

# Text I/O
p.write_text("hello")
txt = p.read_text(encoding="utf-8")

# Arrow I/O (Parquet by default for .parquet suffix, Arrow IPC for .arrow)
import pyarrow as pa
p.write_arrow_table(pa.table({"x": [1, 2, 3]}))
tbl = p.read_arrow_table()

# Directory ops
p.mkdir(parents=True, exist_ok=True)
for child in p.iterdir():
    print(child)
p.unlink(missing_ok=True)    # delete file
p.rmtree()                   # delete directory tree
```

---

## 5) `RemotePath` — shared behaviour for network backends

Remote backends inherit `RemotePath` and get for free:

- **Stat cache** — 5-minute TTL; invalidated on write/delete.
- **Singleton cache** — same URL + client → same instance, so stat is never
  re-fetched twice in the same loop.

```python
from yggdrasil.databricks.fs import VolumePath
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
path   = VolumePath.from_("/Volumes/main/raw/landing/events.parquet", client=client)

# First access — fetches stat from Databricks Files API
print(path.exists)   # True / False
print(path.stat().size)

# Second access — served from the 5-minute in-process stat cache
print(path.exists)   # no extra API call

# Invalidate after a write
path.write_bytes(b"...")
# stat cache auto-invalidated; next .exists re-fetches
```

---

## 6) Copy between backends

`Path.copy_to(dst)` streams bytes from source to destination —
works across any pair of backends.

```python
from yggdrasil.io.path import LocalPath
from yggdrasil.databricks.fs import VolumePath
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
src    = LocalPath.from_path("/tmp/export.parquet")
dst    = VolumePath.from_("/Volumes/main/raw/landing/export.parquet", client=client)

src.copy_to(dst)
print(dst.stat().size)
```

---

## 7) Pattern: write-then-read with the Tabular layer

`Path` implements the full `Tabular` contract, so it plugs directly into
any method that accepts a `Tabular`.

```python
from yggdrasil.io.path import LocalPath
import pyarrow as pa

path = LocalPath.from_path("/tmp/orders.parquet")

# Write Arrow → Parquet
path.write_arrow_table(pa.table({"order_id": [1, 2], "amount": [99.0, 149.5]}))

# Read back as Polars
import polars as pl
df = path.to_polars()
print(df)

# Stream in Arrow batches (memory-efficient for large files)
for batch in path.iter_arrow_batches(batch_size=10_000):
    print(batch.num_rows)
```
