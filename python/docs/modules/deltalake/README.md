# yggdrasil.deltalake

Read and write Delta Lake transaction logs without a Spark runtime. Works with any PyArrow-supported filesystem (S3, GCS, Azure Blob, local).

## Key exports

```python
from yggdrasil.deltalake import DeltaTable, DeltaLog  # DeltaLog = alias
```

---

## Bootstrap: open an existing table (S3)

```python
from pyarrow.fs import S3FileSystem
from yggdrasil.deltalake import DeltaTable

fs = S3FileSystem(region="us-east-1")
tbl = DeltaTable(fs=fs, storage_location="s3://my-bucket/data/orders/")

# list files in the latest snapshot
files = tbl.select()
print(files)
```

---

## Bootstrap: open a table with partition filter

```python
from pyarrow.fs import S3FileSystem
from yggdrasil.deltalake import DeltaTable

fs = S3FileSystem(region="us-east-1")
tbl = DeltaTable(fs=fs, storage_location="s3://my-bucket/data/trades/")

files = tbl.select(partition_filter={"market": "equities", "date": "2024-01-15"})
```

---

## Bootstrap: read into Arrow

```python
import pyarrow as pa
from pyarrow.fs import S3FileSystem
from yggdrasil.deltalake import DeltaTable

fs = S3FileSystem(region="us-east-1")
tbl = DeltaTable(fs=fs, storage_location="s3://my-bucket/data/trades/")

schema = pa.schema([
    pa.field("trade_id", pa.int64()),
    pa.field("price", pa.float64()),
    pa.field("ts", pa.timestamp("us", tz="UTC")),
])

files = tbl.select()
dataset = tbl.to_arrow_dataset(files, schema=schema)
table = dataset.to_table()
```

---

## Bootstrap: initialize a new table

```python
import pyarrow as pa
from pyarrow.fs import LocalFileSystem
from yggdrasil.deltalake import DeltaTable

schema = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("symbol", pa.string(), nullable=False),
    pa.field("price", pa.float64()),
])

fs = LocalFileSystem()
tbl = DeltaTable.init(
    fs=fs,
    storage_location="/tmp/my_delta_table/",
    schema=schema,
)
```

---

## Bootstrap: write a commit (append)

```python
import pyarrow as pa
from pyarrow.fs import LocalFileSystem
import pyarrow.parquet as pq
from yggdrasil.deltalake import DeltaTable

fs = LocalFileSystem()
tbl = DeltaTable(fs=fs, storage_location="/tmp/my_delta_table/")

data = pa.table({"id": [1, 2], "symbol": ["AAPL", "MSFT"], "price": [190.5, 420.0]})
pq.write_table(data, "/tmp/my_delta_table/part-0.parquet", filesystem=fs)

tbl.commit(added_files=["part-0.parquet"])
```

---

## Value objects

```python
from yggdrasil.deltalake import (
    DeltaProtocol,
    DeltaMetadata,
    DeltaStats,
    DeltaFile,
    DeletionVector,
)
```

- `DeltaProtocol` — min reader/writer version
- `DeltaMetadata` — table name, description, schema string, partition columns
- `DeltaStats` — per-file row counts and column statistics
- `DeltaFile` — represents a file in the Delta log (path, size, stats)
- `DeletionVector` — deletion vector metadata for row-level deletes
