# yggdrasil.io

IO and transport layer — byte buffers, URL composition, filesystem paths,
tabular format handlers, and the `HTTPSession` HTTP client.

## Module map

| Sub-module | What it covers |
|---|---|
| `yggdrasil.io` | `URL`, `BytesIO`, `Memory`, `IOStats` — root primitives |
| `yggdrasil.io.path` | `Path`, `LocalPath`, `RemotePath` — filesystem abstraction |
| `yggdrasil.io.http_` | `HTTPSession`, `PreparedRequest`, `Response` — HTTP client |
| `yggdrasil.io.tabular` | `ArrowTabular`, `LazyTabular`, `UnionTabular` — in-memory Tabular |
| `yggdrasil.io.primitive` | `ParquetFile`, `CSVFile`, `NDJSONFile`, `ArrowIPCFile`, `XLSXFile` |
| `yggdrasil.io.nested` | `FolderPath`, `ZipFile`, `DeltaFolder` — multi-file containers |

---

## 1) URL — immutable URL parsing and composition

```python
from yggdrasil.io import URL

# Parse
u = URL.from_str("https://api.example.com/v2/events?since=2026-01-01&limit=100")
print(u.scheme)    # "https"
print(u.host)      # "api.example.com"
print(u.path)      # "/v2/events"
print(u.query)     # "since=2026-01-01&limit=100"
print(u.params)    # {"since": "2026-01-01", "limit": "100"}

# Compose — immutable: every method returns a new URL
base    = URL.from_str("https://api.example.com")
events  = base.with_path("/v2/events")
paged   = events.with_query_items({"page": 2, "limit": 50})
print(paged.to_string())
# "https://api.example.com/v2/events?limit=50&page=2"

# With sorted params (deterministic cache key)
canonical = events.with_query_items(
    {"since": "2026-01-01", "format": "parquet"},
    sort_keys=True,
)

# Path joining
detail = base / "v2" / "events" / "12345"
print(detail.path)   # "/v2/events/12345"

# Mutate individual components
secure = URL.from_str("http://example.com/api").with_scheme("https")
authed = URL.from_str("https://api.example.com").with_userinfo("user", "secret")

# Query-param helpers
u2 = u.with_query_param("limit", 200)
u3 = u.without_query_param("since")
print(u3.to_string())
```

---

## 2) BytesIO — spill-to-disk byte buffer

`BytesIO` is a `Tabular`-capable, compression-aware byte buffer with
MIME-type detection and optional disk spill.

```python
from yggdrasil.io import BytesIO

# In-memory buffer with context manager
with BytesIO() as buf:
    buf.write(b"hello, world")
    buf.seek(0)
    print(buf.read())           # b"hello, world"
    print(buf.media_type)       # auto-detected MIME type
    print(buf.compression)      # None / "gzip" / "zstd" / ...

# Write then read Arrow (Parquet round-trip via the buffer)
import pyarrow as pa
from yggdrasil.io.primitive import ParquetFile

pq = ParquetFile()
pq.write_arrow_table(pa.table({"id": [1, 2, 3]}))
pq.seek(0)
tbl = pq.read_arrow_table()

# With compression
from yggdrasil.io import BytesIO
import gzip

buf = BytesIO()
buf.write(gzip.compress(b"compressed payload"))
buf.seek(0)
print(buf.compression)   # "gzip"
print(buf.decompress().read())   # b"compressed payload"
```

---

## 3) Memory — named in-memory store

```python
from yggdrasil.io import Memory
import pyarrow as pa

# Write a table to a named in-memory slot
mem = Memory("my_table")
mem.write_arrow_table(pa.table({"k": [1, 2], "v": ["a", "b"]}))

# Read it back — zero bytes over the wire
mem2 = Memory("my_table")
tbl  = mem2.read_arrow_table()
print(tbl)
```

---

## 4) IOStats — file metadata

```python
from yggdrasil.io import IOStats
from yggdrasil.io.path import LocalPath

p    = LocalPath.from_path("/tmp/data.parquet")
stat = p.stat()

print(stat.size)    # bytes (int)
print(stat.mtime)   # datetime
print(stat.kind)    # IOKind.FILE or IOKind.DIRECTORY
print(stat.is_file)
print(stat.is_dir)
```

---

## 5) Quick reference to sub-module docs

| What you need | See |
|---|---|
| File read/write (local or remote path) | [io.path](path/README.md) |
| HTTP GET/POST/batch requests | [io.http_](http_/README.md) |
| Parquet / CSV / JSON / XLSX in memory | [io.primitive](primitive/README.md) |
| Arrow in-memory tabular container | [io.tabular](tabular/README.md) |
| Directory / Zip / Delta Lake | [io.nested](nested/README.md) |

---

## 6) End-to-end: HTTP → Parquet → local file

```python
from yggdrasil.io import URL
from yggdrasil.io.http_ import HTTPSession
from yggdrasil.io.primitive import CSVFile
from yggdrasil.io.path import LocalPath
import pyarrow as pa

# 1. Fetch CSV from an API
http = HTTPSession()
resp = http.get("https://data.example.com/orders.csv")
resp.raise_for_status()

# 2. Parse into an Arrow table via CSVFile
buf = CSVFile()
buf.write(resp.content)
buf.seek(0)
orders = buf.read_arrow_table()

# 3. Write as Parquet to local disk
out = LocalPath.from_path("/tmp/orders.parquet")
out.write_arrow_table(orders)
print(f"Saved {orders.num_rows} rows → {out}")
```
