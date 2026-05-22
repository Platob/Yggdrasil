# yggdrasil.io.nested

Nested-Tabular containers — directories of files, Zip archives, and Delta
Lake tables — all implementing the same Arrow record-batch `Tabular`
read/write contract.

## Surface map

| Class | What it wraps |
|---|---|
| `FolderPath` | A directory tree of Parquet/CSV/Arrow/NDJSON files |
| `ZipFile` / `ZipEntryFile` | A `.zip` archive (as a whole or one named entry) |
| `DeltaFolder` | A Delta Lake table — log replay, snapshots, time-travel |

> **Delta convenience import:** `yggdrasil.delta` is a thin shim that
> re-exports everything from `yggdrasil.io.nested.delta`. New code should
> import from the canonical location (`yggdrasil.io.nested.delta`).

---

## 1) One-liners

```python
from yggdrasil.io.nested import FolderPath, DeltaFolder, ZipFile

# Read all Parquet files under a directory as a single table
tbl = FolderPath("/data/events/").read_arrow_table()

# Latest Delta snapshot
snap = DeltaFolder("/data/orders_delta").snapshot()
print(snap.version, len(snap.add_files))

# Zip round-trip
import pyarrow as pa
zf = ZipFile("/tmp/export.zip")
zf.write_arrow_table(pa.table({"id": [1, 2]}))
```

---

## 2) `FolderPath` — directory of tabular files

A `FolderPath` treats a directory (local or remote) as a single virtual
table. Files inside it can be Parquet, CSV, NDJSON, Arrow IPC, or XLSX —
mixed formats are supported.

```python
from yggdrasil.io.nested import FolderPath

# Local directory — all .parquet files
folder = FolderPath("/data/events/2026-05/")
tbl    = folder.read_arrow_table()
print(tbl.num_rows)

# Stream in batches (memory-efficient for large directories)
for batch in folder.iter_arrow_batches(batch_size=50_000):
    process(batch)

# Filter to a specific format
folder = FolderPath("/data/mixed/", media_type="application/vnd.apache.parquet")

# Write a table into the folder (creates date-partitioned files)
import pyarrow as pa
folder.write_arrow_table(
    pa.table({"id": [1, 2, 3], "ts": [...]}),
    partition_cols=["date"],
)
```

### Remote folder (Databricks Volume)

```python
from yggdrasil.io.nested import FolderPath
from yggdrasil.databricks.fs import VolumePath
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
folder = FolderPath(
    VolumePath.from_("/Volumes/main/raw/events/", client=client)
)
tbl = folder.read_arrow_table()
```

### Options

```python
from yggdrasil.io.nested import FolderPath, FolderOptions

opts   = FolderOptions(recursive=True, pattern="*.parquet")
folder = FolderPath("/data/", options=opts)
tbl    = folder.read_arrow_table()
```

---

## 3) `ZipFile` — zip archive

```python
from yggdrasil.io.nested import ZipFile, ZipEntryFile, ZipOptions
import pyarrow as pa

# Write a table into a named entry inside a zip
zf = ZipFile("/tmp/export.zip")
zf.write_arrow_table(
    pa.table({"product": ["A", "B"], "qty": [100, 50]}),
    entry_name="products.parquet",
)

# Read back all entries as one table
tbl = zf.read_arrow_table()
print(tbl)

# Read one named entry
entry = ZipEntryFile(zf, "products.parquet")
back  = entry.read_arrow_table()
```

### Stream entries one by one

```python
from yggdrasil.io.nested import ZipFile

zf = ZipFile("/tmp/archive.zip")
for entry in zf.iter_entries():
    print(entry.name)
    tbl = entry.read_arrow_table()
```

---

## 4) `DeltaFolder` — Delta Lake

`DeltaFolder` is a pure-Python Delta log reader. No Spark required — it
replays the transaction log using pyarrow JSON, resolves snapshots, and
reports file manifests. Writes are **not** supported (use `DatabricksClient`
for Delta DML).

### Read the latest snapshot

```python
from yggdrasil.io.nested import DeltaFolder

folder = DeltaFolder("/mnt/data/orders_delta")
snap   = folder.snapshot()

print("Version:", snap.version)
print("Schema: ", snap.schema)          # yggdrasil.data.Schema
print("Files:  ", len(snap.add_files))  # number of active Parquet files
```

### Time travel

```python
from yggdrasil.io.nested import DeltaFolder

folder  = DeltaFolder("/mnt/data/orders_delta")
snap_v3 = folder.snapshot(version=3)
print("Version 3 files:", len(snap_v3.add_files))
```

### Inspect `AddFile` entries

```python
from yggdrasil.io.nested import DeltaFolder

snap = DeltaFolder("/mnt/data/orders_delta").snapshot()

for f in snap.add_files:
    print(f.path)
    print(f.size)
    print(f.modification_time)
    print(f.partition_values)   # {"year": "2026", "month": "05"}
    print(f.stats)              # min/max/null JSON string
```

### Walk the Delta log

```python
from yggdrasil.io.nested import DeltaFolder
from yggdrasil.io.nested.delta import DeltaLog

folder = DeltaFolder("/mnt/data/orders_delta")
log    = folder.log()   # DeltaLog

for entry in log.entries():
    print(entry.version)
    print(entry.commit_info)
    print("adds:", len(entry.add_files))
    print("removes:", len(entry.remove_files))
```

### Parse commit actions directly

```python
from yggdrasil.io.nested.delta import parse_action, AddFile, RemoveFile, Metadata, Protocol

line = '{"add": {"path": "part-0.parquet", "size": 1234, ...}}'
action = parse_action(line)
if isinstance(action, AddFile):
    print("Added:", action.path)
```

### `DeltaOptions` — configure the log reader

```python
from yggdrasil.io.nested import DeltaFolder, DeltaOptions

opts   = DeltaOptions(checkpoint_interval=10)
folder = DeltaFolder("/mnt/data/orders_delta", options=opts)
snap   = folder.snapshot()
```

---

## 5) End-to-end: Delta → transform → Parquet export

```python
from yggdrasil.io.nested import DeltaFolder
from yggdrasil.io.path import LocalPath
import pyarrow as pa
import pyarrow.compute as pc

# Read the Delta snapshot
folder = DeltaFolder("/mnt/data/orders_delta")
snap   = folder.snapshot()

# Collect from active Parquet files (no Spark)
batches = []
for add_file in snap.add_files:
    file_path = LocalPath.from_path(f"/mnt/data/orders_delta/{add_file.path}")
    batches.extend(file_path.read_arrow_table().to_batches())

orders = pa.Table.from_batches(batches)

# Filter to completed orders (using pyarrow.compute — no Python loop)
mask   = pc.equal(orders.column("status"), "completed")
done   = orders.filter(mask)

# Export to Parquet
out = LocalPath.from_path("/tmp/completed_orders.parquet")
out.write_arrow_table(done)
print(f"Exported {done.num_rows} rows → {out}")
```
