# yggdrasil.io

Backend-agnostic IO layer: URL parsing and composition, filesystem paths, byte buffers, tabular format handlers, and HTTP transport primitives.

---

## Surface map

| Class / module | What it does | Import from |
|---|---|---|
| `URL` | Immutable URL value — parse, compose, sign, match patterns | `yggdrasil.io` |
| `BytesIO` | Spill-to-disk byte buffer with media/codec detection | `yggdrasil.io` |
| `Path` | Backend-agnostic `pathlib.Path`-like filesystem object | `yggdrasil.io.path` |
| Tabular formats | Parquet / Arrow IPC / CSV / NDJSON / XLSX handlers | `yggdrasil.io.primitive` |
| `HTTPSession` | Preferred HTTP client | `yggdrasil.http_` |

---

## 1) URL — immutable URL value

```python
from yggdrasil.io import URL

u = URL.from_str("https://api.example.com/v1/data?page=1&lang=en")

print(u.scheme)           # "https"
print(u.host)             # "api.example.com"
print(u.path)             # "/v1/data"
print(u.query_dict)       # {"page": "1", "lang": "en"}
```

### Build and mutate

```python
from yggdrasil.io import URL

base = URL.from_str("https://api.example.com")

# Path joining (immutable — returns a new URL each time)
endpoint = base / "v1" / "orders"
print(endpoint.to_string())    # "https://api.example.com/v1/orders"

# Query params
with_params = endpoint.with_query_items({"page": 2, "limit": 100})
print(with_params.to_string())

# Add a single param (replace=True removes existing)
url2 = endpoint.add_param("format", "arrow", replace=True)

# Sort query keys for cache-stable URLs
url3 = endpoint.with_query_items({"b": 2, "a": 1}, sort_keys=True)
# → ?a=1&b=2
```

### Scrub credentials before logging

```python
u = URL.from_str("https://user:secret@api.example.com/data?token=abc")
print(u.anonymize(mode="redact"))
# "https://***:***@api.example.com/data?token=***"
print(u.anonymize(mode="remove"))
# "https://api.example.com/data"
```

### Pattern matching

```python
from yggdrasil.io import URL

u = URL.from_str("https://api.example.com/v1/orders/42")

print(u.match_pattern("https://api.example.com/v1/*/"))  # True
print(u.matches_patterns(["https://api.*", "https://cdn.*"]))  # True
```

### S3 / DBFS / Volumes paths

```python
from yggdrasil.io import URL

s3  = URL.from_str("s3://my-bucket/data/events.parquet")
vol = URL.from_str("dbfs+volume://main.sales.uploads/raw/batch.parquet")

print(s3.scheme)   # "s3"
print(s3.host)     # "my-bucket"
print(s3.path)     # "/data/events.parquet"
```

---

## 2) BytesIO — in-memory byte buffer

```python
from yggdrasil.io import BytesIO

buf = BytesIO()
buf.write(b"hello world")
buf.seek(0)
print(buf.read())         # b"hello world"
print(buf.size)           # 11
```

Media-type and codec detection:

```python
with BytesIO() as buf:
    import pyarrow.parquet as pq
    import pyarrow as pa
    pq.write_table(pa.table({"id": [1, 2]}), buf)
    buf.seek(0)
    print(buf.media_type)   # MediaType.PARQUET (detected from magic bytes)
    print(buf.compression)  # Codec.NONE (or the active codec)
```

---

## 3) Path — backend-agnostic filesystem

`yggdrasil.io.path.Path` is a `pathlib.Path`-like interface that dispatches to the right backend (local, S3, DBFS, Volumes, Workspace) based on the URL scheme.

```python
from yggdrasil.io.path import Path

# Local file
p = Path.from_("file:///tmp/data.parquet")
p.write_bytes(b"content")
print(p.exists())           # True
print(p.read_bytes()[:4])   # b"PAR1"  (Parquet magic)
print(p.name)               # "data.parquet"
print(p.stem)               # "data"
print(p.suffix)             # ".parquet"

# Directory listing
for child in Path.from_("file:///tmp/").iterdir():
    print(child.name)
```

```python
# S3 (via yggdrasil.aws.fs.S3Path)
p = Path.from_("s3://my-bucket/prefix/file.csv")
content = p.read_text()

# Databricks DBFS (via yggdrasil.databricks.path)
p = Path.from_("dbfs:/tmp/output.json")
p.write_text('{"key": "value"}')
```

Path arithmetic:

```python
from yggdrasil.io.path import Path

base = Path.from_("s3://my-bucket/data/")
child = base / "2026" / "05" / "events.parquet"
print(child.url.to_string())   # "s3://my-bucket/data/2026/05/events.parquet"
```

---

## 4) Tabular format handlers (`yggdrasil.io.primitive`)

Single-buffer format handlers for Parquet, Arrow IPC, CSV, NDJSON, and XLSX. Each implements the same `read_arrow_table()` / `write_arrow_table()` surface.

### Parquet

```python
import io, pyarrow as pa
from yggdrasil.io.primitive import ParquetFile

table = pa.table({"id": [1, 2, 3], "score": [9.1, 8.7, 7.4]})

# Write to buffer
buf = io.BytesIO()
pf = ParquetFile(buf)
pf.write_arrow_table(table)

# Read back
buf.seek(0)
out = pf.read_arrow_table()
print(out.schema)
```

### Arrow IPC

```python
import io, pyarrow as pa
from yggdrasil.io.primitive import ArrowIPCFile

table = pa.table({"x": [1, 2], "y": [3.0, 4.0]})

buf = io.BytesIO()
ArrowIPCFile(buf).write_arrow_table(table)
buf.seek(0)
out = ArrowIPCFile(buf).read_arrow_table()
```

### CSV

```python
import io, pyarrow as pa
from yggdrasil.io.primitive import CSVFile
from yggdrasil.data.cast.options import CastOptions

csv_bytes = b"id,name,score\n1,alice,9.5\n2,bob,8.1\n"
buf = io.BytesIO(csv_bytes)

target = pa.schema([
    pa.field("id",    pa.int64()),
    pa.field("name",  pa.string()),
    pa.field("score", pa.float64()),
])
out = CSVFile(buf).read_arrow_table(CastOptions(target_field=target))
print(out)
```

### NDJSON

```python
import io
from yggdrasil.io.primitive import NDJSONFile

ndjson = b'{"id":1,"v":2.5}\n{"id":2,"v":3.0}\n'
buf = io.BytesIO(ndjson)
out = NDJSONFile(buf).read_arrow_table()
print(out.schema)
```

### XLSX (Excel)

```python
import io, pyarrow as pa
from yggdrasil.io.primitive import XLSXFile

table = pa.table({"product": ["A", "B"], "revenue": [1000.0, 2500.0]})

buf = io.BytesIO()
XLSXFile(buf).write_arrow_table(table)
buf.seek(0)
out = XLSXFile(buf).read_arrow_table()
```

---

## 5) IO curation (`yggdrasil.io.curation`)

Auto-typing rules that promote raw string columns into their correct Arrow types based on heuristics (ISO 8601 dates, numeric-looking strings, boolean literals, etc.).

```python
import pyarrow as pa
from yggdrasil.io.curation import curate_arrow_table

raw = pa.table({
    "order_id":   ["101", "102", "103"],
    "placed_at":  ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
    "amount":     ["99.50", "149.00", "79.99"],
    "is_paid":    ["true", "false", "true"],
})

curated = curate_arrow_table(raw)
print(curated.schema)
# order_id:  int64          (numeric-looking string → int)
# placed_at: timestamp[us, UTC]  (ISO 8601 → UTC timestamp)
# amount:    double         (decimal string → float)
# is_paid:   bool           (boolean literal)
```

---

## 6) HTTP client (`yggdrasil.http_`)

For HTTP requests, reach for `HTTPSession` from `yggdrasil.http_` — see the full [http_ module docs](../http_/README.md).

```python
from yggdrasil.http_ import HTTPSession

http = HTTPSession()
resp = http.get("https://httpbin.org/json")
print(resp.status, resp.json())
```

---

## 7) End-to-end: fetch JSON → Parquet → S3

```python
import io, pyarrow as pa
from yggdrasil.http_ import HTTPSession
from yggdrasil.io.primitive import ParquetFile, NDJSONFile
from yggdrasil.aws.fs import S3Path

# Fetch NDJSON from API
http = HTTPSession()
resp = http.get("https://api.example.com/events", params={"format": "ndjson"})

# Parse into Arrow table
table = NDJSONFile(io.BytesIO(resp.content)).read_arrow_table()

# Write as Parquet to S3
buf = io.BytesIO()
ParquetFile(buf).write_arrow_table(table)
buf.seek(0)
S3Path("s3://my-bucket/raw/events.parquet").write_bytes(buf.read())
```
