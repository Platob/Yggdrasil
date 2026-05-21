# yggdrasil.delta / yggdrasil.io.nested.delta

Delta Lake log reader — parse Delta transaction logs, snapshots, and file manifests without Spark.

> **Import path:** `yggdrasil.delta` is a back-compat shim; new code should import from `yggdrasil.io.nested.delta`.

## One-liner

```python
from yggdrasil.io.nested.delta import DeltaFolder

folder = DeltaFolder("/path/to/delta/table")
snap   = folder.snapshot()
print(snap.version, len(snap.add_files))
```

## Read a Delta snapshot

```python
from yggdrasil.io.nested.delta import DeltaFolder, Snapshot

# Local filesystem
folder = DeltaFolder("/mnt/data/events_delta")

# Latest snapshot
snap: Snapshot = folder.snapshot()
print("Version:", snap.version)
print("Schema:", snap.schema)
print("Files:", len(snap.add_files))

# Specific version
snap_v3 = folder.snapshot(version=3)
```

## Inspect log entries

```python
from yggdrasil.io.nested.delta import DeltaFolder, DeltaLog

folder = DeltaFolder("/mnt/data/events_delta")
log    = folder.log()   # DeltaLog

for entry in log.entries():
    print(entry.version, entry.commit_info, len(entry.add_files))
```

## File manifest (AddFile)

```python
from yggdrasil.io.nested.delta import DeltaFolder

snap = DeltaFolder("/mnt/data/events_delta").snapshot()

for add_file in snap.add_files:
    print(add_file.path)
    print(add_file.size)
    print(add_file.modification_time)
    print(add_file.partition_values)   # {"year": "2026", "month": "05"}
    print(add_file.stats)              # row-count / min / max JSON string
```

## Metadata and protocol

```python
snap = DeltaFolder("/mnt/data/events_delta").snapshot()

meta: Metadata = snap.metadata
print(meta.schema_string)    # Delta JSON schema string
print(meta.format)
print(meta.partition_columns)
print(meta.configuration)    # table-level Delta properties

proto: Protocol = snap.protocol
print(proto.min_reader_version, proto.min_writer_version)
```

## DeltaOptions

```python
from yggdrasil.io.nested.delta import DeltaFolder, DeltaOptions

opts = DeltaOptions(
    version=5,            # pin to a specific version
    timestamp=None,       # or time-travel by timestamp
)
snap = DeltaFolder("/mnt/data/events_delta").snapshot(options=opts)
```

## Deletion vectors

```python
snap = DeltaFolder("/mnt/data/events_delta").snapshot()

for add_file in snap.add_files:
    dv = add_file.deletion_vector
    if dv is not None:
        print(dv.storage_type, dv.offset, dv.size_in_bytes)
```

## Back-compat import path

Old import paths still work:

```python
# Legacy (still works via back-compat shim)
from yggdrasil.delta import DeltaFolder, DeltaLog, Snapshot, AddFile

# Preferred — import from the canonical location
from yggdrasil.io.nested.delta import DeltaFolder, DeltaLog, Snapshot, AddFile
```

## Read Delta files into Arrow (without Spark)

```python
from yggdrasil.io.nested.delta import DeltaFolder
import pyarrow.parquet as pq
import pyarrow as pa

folder = DeltaFolder("/mnt/data/events_delta")
snap   = folder.snapshot()

# Build Arrow table from the live Parquet files
parts = []
for add_file in snap.add_files:
    tbl = pq.read_table(folder.path / add_file.path)
    parts.append(tbl)

result = pa.concat_tables(parts)
print(result.num_rows)
```
