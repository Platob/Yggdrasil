# yggdrasil.io.primitive

Single-buffer tabular format handlers — Parquet, CSV, JSON, NDJSON, Arrow
IPC, and XLSX. Each leaf is a `BytesIO` subclass registered under its
MIME type; every format implements the same `Tabular` read/write contract.

## Surface map

| Class | MIME type | Extension | Notes |
|---|---|---|---|
| `ParquetFile` | `application/vnd.apache.parquet` | `.parquet` | Columnar, footer-indexed, pushdown-capable |
| `CSVFile` | `text/csv` | `.csv` | Delimiter-separated text |
| `JSONFile` | `application/json` | `.json` | JSON array or newline-delimited |
| `NDJSONFile` | `application/x-ndjson` | `.ndjson` / `.jsonl` | Newline-delimited JSON — preferred for streaming |
| `ArrowIPCFile` | `application/vnd.apache.arrow.file` | `.arrow` | Zero-copy IPC file format |
| `XLSXFile` | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` | `.xlsx` | Excel workbook |

---

## 1) One-liners

```python
from yggdrasil.io.primitive import ParquetFile
import pyarrow as pa

buf = ParquetFile()
buf.write_arrow_table(pa.table({"id": [1, 2, 3]}))
buf.seek(0)
tbl = buf.read_arrow_table()
```

---

## 2) Parquet

```python
from yggdrasil.io.primitive import ParquetFile
import pyarrow as pa

table = pa.table({
    "id":    pa.array([1, 2, 3], type=pa.int64()),
    "score": pa.array([9.1, 8.7, 7.4], type=pa.float64()),
    "label": pa.array(["a", "b", "c"], type=pa.string()),
})

# In-memory round-trip
buf = ParquetFile()
buf.write_arrow_table(table)
buf.seek(0)
back = buf.read_arrow_table()
print(back.equals(table))   # True

# Stream in row-group-sized batches
buf.seek(0)
for batch in buf.iter_arrow_batches(batch_size=1_000):
    print(batch.num_rows)

# To pandas / Polars
buf.seek(0)
df  = buf.to_pandas()
buf.seek(0)
lf  = buf.to_polars()

# File path round-trip
from yggdrasil.io.path import LocalPath
path = LocalPath.from_path("/tmp/orders.parquet")
path.write_arrow_table(table)
back = path.read_arrow_table()
```

### Parquet with column projection and target schema

```python
from yggdrasil.io.primitive import ParquetFile
from yggdrasil.data.cast.options import CastOptions
import pyarrow as pa

buf = ParquetFile()
buf.write_arrow_table(pa.table({"id": ["1", "2"], "score": ["9.1", "8.7"], "extra": [1, 2]}))
buf.seek(0)

target = pa.schema([
    pa.field("id",    pa.int64(),   nullable=False),
    pa.field("score", pa.float64()),
])
tbl = buf.read_arrow_table(options=CastOptions(target=target))
print(tbl.schema)   # id: int64, score: float64 — extra dropped, types cast
```

---

## 3) CSV

```python
from yggdrasil.io.primitive import CSVFile
import pyarrow as pa

table = pa.table({"name": ["alice", "bob"], "score": [9.1, 8.7]})

buf = CSVFile()
buf.write_arrow_table(table)
buf.seek(0)

# Read back with type inference
back = buf.read_arrow_table()

# Read with explicit schema
from yggdrasil.data.cast.options import CastOptions
buf.seek(0)
tbl  = buf.read_arrow_table(options=CastOptions(
    target=pa.schema([
        pa.field("name",  pa.string()),
        pa.field("score", pa.float64()),
    ])
))

# Local file
from yggdrasil.io.path import LocalPath
path = LocalPath.from_path("/tmp/report.csv")
path.write_arrow_table(table)
df = path.to_pandas()
```

---

## 4) NDJSON (newline-delimited JSON)

NDJSON is the preferred format for streaming JSON — one record per line,
no wrapping array.

```python
from yggdrasil.io.primitive import NDJSONFile
import pyarrow as pa

table = pa.table({
    "event": ["click", "view", "purchase"],
    "user":  [1, 2, 1],
})

buf = NDJSONFile()
buf.write_arrow_table(table)
buf.seek(0)

# Read with vectorised NDJSON decoder (pyarrow.json)
back = buf.read_arrow_table()

# Stream large NDJSON files in batches
buf.seek(0)
for batch in buf.iter_arrow_batches(batch_size=500):
    print(batch.num_rows)
```

---

## 5) JSON

```python
from yggdrasil.io.primitive import JSONFile
import pyarrow as pa

table = pa.table({"k": [1, 2], "v": ["a", "b"]})

buf = JSONFile()
buf.write_arrow_table(table)
buf.seek(0)
back = buf.read_arrow_table()
```

---

## 6) Arrow IPC (`.arrow`)

Arrow IPC is the fastest format for in-process or same-host transfers —
zero-copy reads when mapped directly.

```python
from yggdrasil.io.primitive import ArrowIPCFile
import pyarrow as pa

table = pa.table({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})

buf = ArrowIPCFile()
buf.write_arrow_table(table)
buf.seek(0)

back = buf.read_arrow_table()

# Batch iteration
buf.seek(0)
for batch in buf.iter_arrow_batches():
    print(batch)
```

---

## 7) XLSX (Excel)

```python
from yggdrasil.io.primitive import XLSXFile
import pyarrow as pa

table = pa.table({
    "product":  ["Widget A", "Widget B", "Widget C"],
    "price":    [9.99, 14.99, 4.99],
    "quantity": [100, 50, 200],
})

buf = XLSXFile()
buf.write_arrow_table(table)
buf.seek(0)

back = buf.read_arrow_table()
print(back.to_pydict())

# Save to disk
from yggdrasil.io.path import LocalPath
path = LocalPath.from_path("/tmp/report.xlsx")
path.write_arrow_table(table)
```

---

## 8) Auto-dispatch from MIME type or file extension

The format registry lets you pick a class by MIME type or path:

```python
from yggdrasil.io.tabular import Tabular

# Auto-detects Parquet from extension
t = Tabular.from_("data.parquet")

# Explicit MIME type
from yggdrasil.io.holder import Holder
cls = Holder.class_for_media_type("text/csv")   # → CSVFile

# In conversion: let the path extension drive format selection
from yggdrasil.io.path import LocalPath
p = LocalPath.from_path("/tmp/export.csv")
p.write_arrow_table(pa.table({"a": [1]}))   # writes as CSV (extension → CSVFile)
```

---

## 9) HTTP response → typed format

When `HTTPSession` downloads a response body as tabular data, it uses the
`Content-Type` header to pick the right format class automatically:

```python
from yggdrasil.io.http_ import HTTPSession

http = HTTPSession()
resp = http.get("https://api.example.com/export?format=parquet")

# Server returns Content-Type: application/vnd.apache.parquet
tbl = resp.to_arrow_table()   # dispatches to ParquetFile.read_arrow_table()
```
