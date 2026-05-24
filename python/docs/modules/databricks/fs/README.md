# yggdrasil.databricks.fs

DBFS, Unity Catalog Volume, and Workspace file-path abstractions — all with a uniform `pathlib`-style API for read, write, list, stat, move, copy, and delete.

---

## Path types

| Class | Scheme | Example |
|---|---|---|
| `DBFSPath` | `dbfs:/` | `dbfs:/tmp/scratch/data.parquet` |
| `VolumePath` | `/Volumes/` | `/Volumes/main/default/raw/events.json` |
| `WorkspacePath` | `/Workspace/` | `/Workspace/Shared/notebooks/etl.py` |
| `DatabricksPath` | (auto) | dispatch based on path prefix |

---

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

DatabricksClient().dbfs_path("dbfs:/tmp/hello.txt").write_text("hello world")
```

---

## 1) Resolve a path

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient()

# DBFS
dbfs = c.dbfs_path("dbfs:/tmp/demo.parquet")

# Unity Catalog Volume (auto-detected from /Volumes/ prefix)
vol = c.dbfs_path("/Volumes/main/default/raw/events.parquet")

# Workspace
ws = c.dbfs_path("/Workspace/Shared/.ygg/jobs/config.json")
```

---

## 2) Read and write

```python
from yggdrasil.databricks import DatabricksClient

p = DatabricksClient().dbfs_path("dbfs:/tmp/ygg_demo.txt")

# Text
p.write_text("id,name\n1,alice\n2,bob")
print(p.read_text())

# Bytes
p.with_suffix(".bin").write_bytes(b"\x00\x01\x02")
raw = p.with_suffix(".bin").read_bytes()
```

---

## 3) Directory operations

```python
from yggdrasil.databricks import DatabricksClient

base = DatabricksClient().dbfs_path("dbfs:/tmp/ygg_demo_dir/")

# Create
base.mkdir(parents=True, exist_ok=True)

# List
for entry in base.ls():
    print(entry.name, entry.is_file(), entry.size)

# Recursive
for entry in base.ls_recursive():
    print(entry.path)

# Remove directory
base.rmdir(recursive=True)
```

---

## 4) Stat and existence checks

```python
p = DatabricksClient().dbfs_path("dbfs:/tmp/demo.txt")

print(p.exists())        # True / False
print(p.is_file())
print(p.is_dir())

info = p.stat()
print(info.file_size, info.modification_time)
```

---

## 5) Path transforms

```python
from yggdrasil.databricks import DatabricksClient

p = DatabricksClient().dbfs_path("dbfs:/tmp/data/report.csv.gz")

print(p.name)           # report.csv.gz
print(p.stem)           # report.csv
print(p.suffix)         # .gz
print(p.parent)         # dbfs:/tmp/data/
print(p.with_suffix(".parquet"))  # dbfs:/tmp/data/report.csv.parquet
print(p.relative_to("dbfs:/tmp/"))  # data/report.csv.gz
```

---

## 6) Move, copy, delete

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient()
src = c.dbfs_path("dbfs:/tmp/source.txt")
dst = c.dbfs_path("dbfs:/tmp/dest.txt")

src.write_text("payload")
src.copy_to(dst)          # copy without removing source
print(dst.read_text())    # payload

src.rename(c.dbfs_path("dbfs:/tmp/source_moved.txt"))  # move
c.dbfs_path("dbfs:/tmp/source_moved.txt").remove()
dst.remove()
```

---

## 7) Volume path SQL integration

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient()
p = c.dbfs_path("/Volumes/main/default/raw/orders.parquet")

# Resolve the SQL triple (catalog, schema, volume_name)
catalog, schema, volume = p.sql_volume_or_table_parts()
print(catalog, schema, volume)   # main default raw

# Write Arrow table directly to the volume path
import pyarrow as pa
tbl = pa.table({"id": [1, 2, 3], "amount": [10.5, 20.0, 5.75]})
p.write_arrow(tbl)
print(c.sql.execute(f"SELECT COUNT(*) FROM parquet.`{p}`").to_pylist())
```

---

## 8) Temp managed paths

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient()

# Temp path with a 1-hour lifetime (auto-cleaned by clean_tmp_folder)
tmp = c.tmp_path(extension="json", max_lifetime=3600)
tmp.write_text('{"status": "ok"}')
print(tmp.exists(), tmp.read_text())
tmp.remove()

# Bulk cleanup of stale temp files
c.clean_tmp_folder()
```

---

## 9) Arrow filesystem integration

Volume paths expose the Arrow `pyarrow.fs.FileSystem` interface so Arrow's Parquet/IPC readers can target them directly:

```python
from yggdrasil.databricks import DatabricksClient
import pyarrow.parquet as pq

c = DatabricksClient()
vol = c.volumes["main"]["default"]["raw"]

# Write and read via Arrow FS
fs = vol.arrow_filesystem()
pq.write_table(pa.table({"x": [1, 2]}), "data.parquet", filesystem=fs)
print(pq.read_table("data.parquet", filesystem=fs))
```

---

## 10) End-to-end: DBFS → Volume migration

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient()

# Source DBFS files
src_dir = c.dbfs_path("dbfs:/mnt/legacy/orders/")

# Target Volume
dst_dir = c.dbfs_path("/Volumes/main/default/landing/orders/")
dst_dir.mkdir(parents=True, exist_ok=True)

for src_file in src_dir.ls():
    if src_file.is_file():
        dst = dst_dir / src_file.name
        src_file.copy_to(dst)
        print(f"Copied {src_file.name} → {dst.path}")
```
