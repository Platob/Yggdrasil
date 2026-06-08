# yggdrasil.delta

Pure-Python Delta Lake read/write — no Spark, no JVM. Works on local, S3, DBFS, or any yggdrasil `Path`.

> **Import:** `from yggdrasil.delta import DeltaFolder` or `from yggdrasil.io.delta import DeltaFolder`

---

## Quick start

```python
from yggdrasil.delta import DeltaFolder
import pyarrow as pa

# Write
folder = DeltaFolder(path="/tmp/my_table")
folder.write_arrow_table(pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]}))

# Read
table = folder.read_arrow_table()
print(table.to_pandas())
```

## Read

### Arrow

```python
folder = DeltaFolder(path="/data/events")
table = folder.read_arrow_table()
```

### Polars

```python
df = folder.read_polars_frame()
```

### Pandas

```python
df = folder.read_pandas_frame()
```

### Spark (distributed — no collect)

```python
# Scatters parquet reads to Spark executors via mapInArrow.
# Data never touches the driver — each executor reads directly from storage.
spark_df = folder.read_spark_frame()
```

### Streaming batches

```python
for batch in folder.read_arrow_batches():
    process(batch)
```

## Write

### Arrow

```python
from yggdrasil.delta import DeltaFolder, DeltaOptions
from yggdrasil.enums import Mode

folder = DeltaFolder(path="/data/events")

# First write (creates table)
folder.write_arrow_table(table)

# Append
folder.write_arrow_table(new_data, options=DeltaOptions(mode=Mode.APPEND))

# Overwrite (replaces all data + updates schema)
folder.write_arrow_table(new_data, options=DeltaOptions(mode=Mode.OVERWRITE))
```

### Spark

```python
folder.write_spark_frame(spark_df)
```

### Polars / Pandas

```python
folder.write_polars_frame(polars_df)
folder.write_pandas_frame(pandas_df)
```

### Write modes

| Mode | Behavior |
|------|----------|
| `AUTO` / `APPEND` | Add new files, keep existing |
| `OVERWRITE` / `TRUNCATE` | Replace all files + update schema |
| `UPSERT` / `MERGE` | Key-aware merge (requires `match_by`) |
| `IGNORE` | Skip if table non-empty |
| `ERROR_IF_EXISTS` | Raise if table non-empty |

### Upsert (merge by key)

```python
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.primitive import Int64Type, StringType

folder.write_arrow_table(
    pa.table({"id": [2, 5, 7], "val": ["B", "E", "g"]}),
    options=DeltaOptions(
        mode=Mode.UPSERT,
        match_by=[Field(name="id", dtype=Int64Type())],
    ),
)
```

## Time travel

```python
# Read at a specific version
v0 = folder.read_arrow_table(options=DeltaOptions(version=0))

# Inspect snapshot at version 3
snap = folder.snapshot(version=3)
print(snap.version, snap.num_active_files())
```

## Partitioned tables

```python
from yggdrasil.data.data_field import Field
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import Int64Type, StringType

schema = Schema()
schema.with_field(Field(name="id", dtype=Int64Type()))
schema.with_field(Field(name="region", dtype=StringType()).with_partition_by(True))
schema.with_field(Field(name="val", dtype=StringType()))

folder = DeltaFolder(path="/data/events")
folder.write_arrow_table(table, options=DeltaOptions(target=schema))

# Reads auto-prune partitions from predicates
from yggdrasil.saga.expr import col
filtered = folder.read_arrow_table(
    options=DeltaOptions(predicate=col("region") == "us"),
)
```

## Deletion vectors

```python
# Enable DV-based deletes (keeps original parquet, marks rows as deleted)
folder = DeltaFolder(path="/data/events")
folder.write_arrow_table(data, options=DeltaOptions(delete_via_dv=True))

# The read path automatically masks deleted rows via DVs
table = folder.read_arrow_table()
```

## Checkpoints

```python
# V1 (single parquet) — default, compatible with all engines
folder.write_arrow_table(data, options=DeltaOptions(
    checkpoint_interval=10,  # checkpoint every 10 commits
    checkpoint_kind="v1",
))

# V2 (manifest + sidecars) — modern format, multi-sidecar support
folder.write_arrow_table(data, options=DeltaOptions(
    checkpoint_interval=10,
    checkpoint_kind="v2",
))
```

## Snapshot introspection

```python
snap = folder.snapshot()

snap.version                # int — current version
snap.num_active_files()     # int — number of live parquet files
snap.partition_columns      # list[str] — partition column names
snap.schema_string          # str — Spark JSON schema
snap.configuration          # dict — table configuration
snap.has_deletion_vectors   # bool — any file has a DV
snap.num_rows_approx        # int — approximate row count from stats

# Iterate active files
for add in snap.active_files.values():
    print(add.path, add.size, add.partition_values, add.stats)

# Resolve file path
file_path = snap.resolve(add)  # Path object
```

## Schema introspection

```python
# Yggdrasil Schema (native type system)
schema = folder.collect_schema()
for field in schema.fields:
    print(field.name, field.dtype)

# Arrow schema
arrow_schema = schema.to_arrow_schema()

# Spark schema
spark_schema = schema.to_spark_schema()
```

## Remote storage (S3, DBFS)

```python
from yggdrasil.aws.fs.path import S3Path
from yggdrasil.delta import DeltaFolder

# S3 — same API, reads go through S3Path
folder = DeltaFolder(path=S3Path("s3://my-bucket/delta/events"))
table = folder.read_arrow_table()

# DBFS
from yggdrasil.databricks.fs.volume_path import VolumePath
folder = DeltaFolder(path=VolumePath("catalog.schema.volume/delta/events"))
```

## Concurrent writes

```python
# DeltaFolder uses O_EXCL (local) or check-then-write (remote)
# for atomic commits. Version races are retried automatically.
folder.write_arrow_table(data, options=DeltaOptions(
    commit_max_retries=8,        # retry budget
    commit_retry_backoff=0.05,   # exponential backoff base
    commit_retry_jitter=0.05,    # random jitter
    commit_retry_max_delay=1.0,  # max per-attempt delay
))
```

## Per-file statistics

```python
# Stats (numRecords, minValues, maxValues, nullCount) are collected
# by default and written into the AddFile.stats JSON.
folder.write_arrow_table(data, options=DeltaOptions(collect_stats=True))

# Disable for faster writes when stats aren't needed
folder.write_arrow_table(data, options=DeltaOptions(collect_stats=False))
```

## Idempotent writes (txn)

```python
# Application-level idempotency: if a txn with this app_id + version
# already committed, the write is a no-op.
folder.write_arrow_table(data, options=DeltaOptions(
    txn_app_id="my-pipeline",
    txn_version=42,
))
```

## Protocol features

```python
# Fresh tables get protocol versions based on enabled features:
# - Base: reader=1, writer=2
# - delete_via_dv=True: reader=3, writer=7, deletionVectors feature
# - checkpoint_kind="v2": reader=3, writer=7, v2Checkpoint feature

snap = folder.snapshot()
print(snap.protocol.min_reader_version)   # 1
print(snap.protocol.min_writer_version)   # 2
print(snap.protocol.reader_features)      # []
print(snap.protocol.writer_features)      # []
```

## Cache control

```python
# DeltaFolder caches the log listing + snapshot per instance.
# After external writes, call refresh() to pick up new commits.
folder.refresh()
table = folder.read_arrow_table()

# Commit JSON content is cached in a module-level ExpiringDict
# (60s TTL, 1024 max entries, skip > 1 MiB) to reduce remote
# round trips on repeated reads.
```

## Interop with deltalake package

```python
import deltalake

# Write with yggdrasil, read with deltalake
folder = DeltaFolder(path="/tmp/interop")
folder.write_arrow_table(pa.table({"id": [1, 2, 3]}))

dt = deltalake.DeltaTable("/tmp/interop")
print(dt.to_pyarrow_table())

# Write with deltalake, read with yggdrasil
deltalake.write_deltalake("/tmp/interop2", pa.table({"id": [4, 5]}))

folder2 = DeltaFolder(path="/tmp/interop2")
print(folder2.read_arrow_table())
```

## DeltaOptions reference

| Option | Default | Description |
|--------|---------|-------------|
| `version` | `None` | Pin read to specific version (None = HEAD) |
| `checkpoint_interval` | `10` | Commits between automatic checkpoints (0 = disable) |
| `checkpoint_kind` | `"v1"` | `"v1"` (single parquet) or `"v2"` (manifest + sidecars) |
| `operation` | `"WRITE"` | Operation name in commitInfo |
| `engine_info` | `"yggdrasil"` | Engine name in commitInfo |
| `txn_app_id` | `None` | Application ID for idempotent writes |
| `txn_version` | `None` | Application version for idempotent writes |
| `min_reader_version` | `1` | Min reader protocol version for new tables |
| `min_writer_version` | `2` | Min writer protocol version for new tables |
| `delete_via_dv` | `False` | Use deletion vectors instead of file rewrite |
| `commit_max_retries` | `8` | Max retries on version race |
| `commit_retry_backoff` | `0.05` | Exponential backoff base (seconds) |
| `commit_retry_jitter` | `0.05` | Random jitter cap (seconds) |
| `commit_retry_max_delay` | `1.0` | Max per-attempt delay (seconds) |
| `collect_stats` | `True` | Collect min/max/null stats per file |
| `target_file_size` | `128 MiB` | Target parquet file size |
| `mode` | `AUTO` | Write disposition (APPEND, OVERWRITE, UPSERT, etc.) |
| `predicate` | `None` | Row/partition filter for reads |
| `match_by` | `None` | Key columns for UPSERT mode |

## Architecture

```
DeltaFolder(Folder)
  ├── DeltaLog          # parses _delta_log directory
  │   ├── segment()     # resolves checkpoint + commits → LogSegment
  │   ├── replay()      # yields typed DeltaAction stream
  │   └── _content_cache  # ExpiringDict for commit JSON (60s, 1024 max)
  ├── Snapshot          # collapsed table state at a version
  │   ├── active_files  # Dict[path, AddFile]
  │   ├── protocol      # Protocol (reader/writer versions)
  │   └── metadata      # Metadata (schema, partitions, config)
  ├── schema_codec      # Spark JSON ↔ Schema/Field/DataType (native)
  │                     # Arrow/Spark/Polars are peer projections
  ├── deletion_vector   # Roaring bitmap encode/decode + batch masking
  └── checkpoint        # V1/V2 checkpoint writers
```
