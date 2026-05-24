# yggdrasil.io

Yggdrasil's I/O layer: immutable URLs, spill-to-disk buffers, request/response models, media-type and codec detection, and the full primitive/nested file-format hierarchy.

---

## Surface map

| Symbol | Module | Use for |
|---|---|---|
| `URL` | `yggdrasil.io.url` | Parse, compose, and manipulate any URL |
| `BytesIO` | `yggdrasil.io.bytes_io` | Spill-to-disk byte buffer with media/codec awareness |
| `SendConfig` | `yggdrasil.io.send_config` | Single-request send options |
| `SendManyConfig` | `yggdrasil.io.send_config` | Batch-send options (max_workers, ordered, backpressure) |
| `CacheConfig` | `yggdrasil.io.send_config` | Local + remote cache routing for `HTTPSession` |
| `ParquetFile` | `yggdrasil.io.primitive` | Read/write `.parquet` files |
| `ArrowIPCFile` | `yggdrasil.io.primitive` | Read/write Arrow IPC stream/file |
| `CSVFile` | `yggdrasil.io.primitive` | Read/write `.csv` files |
| `JSONFile` | `yggdrasil.io.primitive` | Read/write `.json` files |
| `NDJSONFile` | `yggdrasil.io.primitive` | Read/write newline-delimited JSON |
| `ZipFile` | `yggdrasil.io.nested` | Multi-entry zip archives |
| `DeltaFolder` | `yggdrasil.io.nested.delta` | Delta Lake log reader (no Spark) |
| `Tabular` | `yggdrasil.io.tabular` | Abstract frame I/O surface |

---

## 1) URL — one-liner

```python
from yggdrasil.io import URL

u = URL.from_str("https://api.example.com/v2/items?page=1&limit=20")
print(u.host)           # api.example.com
print(u.path)           # /v2/items
print(u.query_dict)     # {'page': ('1',), 'limit': ('20',)}
```

---

## 2) URL — construction patterns

```python
from yggdrasil.io import URL

# From string
u1 = URL.from_str("https://example.com/a/b")

# From dict
u2 = URL.from_dict({"scheme": "https", "host": "example.com", "path": "/data"})

# From pathlib.Path (becomes file://)
from pathlib import Path
u3 = URL.from_pathlib(Path("/home/user/data.parquet"))

# Division operator — pathlib-style join
base = URL.from_str("https://api.example.com/v1")
endpoint = base / "orders" / "123"
print(endpoint.to_string())   # https://api.example.com/v1/orders/123

# Relative join (RFC 3986)
print(base.join("../v2/orders"))   # https://api.example.com/v2/orders
```

---

## 3) URL — query manipulation

```python
from yggdrasil.io import URL

u = URL.from_str("https://api.example.com/search")

# Replace query entirely (keys sorted for cache-stable hash)
u2 = u.with_query_items({"q": "ygg", "page": "1", "limit": "20"})
print(u2.to_string())
# https://api.example.com/search?limit=20&page=1&q=ygg

# Add or replace a single param
u3 = u2.add_param("page", "2", replace=True)

# Add multiple params (merging, not replacing)
u4 = u2.add_params({"source": "docs", "format": "json"})

# Round-trip query as items
for key, val in u2.query_items():
    print(key, val)
```

---

## 4) URL — path operations

```python
from yggdrasil.io import URL

u = URL.from_str("https://example.com/data/2026/05/report.csv.gz")

print(u.name)          # report.csv.gz
print(u.stem)          # report.csv
print(u.extensions)    # ['csv', 'gz']   (lowercased, no leading dot)
print(u.parent)        # https://example.com/data/2026/05/
print(u.parts)         # ['data', '2026', '05', 'report.csv.gz']

# Navigate up
print(u.parent.parent.parent)   # https://example.com/data/

# Test ancestry
print(u.is_relative_to("https://example.com/data/"))   # True

# Relative path
rel = u.relative_to("https://example.com/data/")
print(rel.path)   # 2026/05/report.csv.gz

# With-suffix replacement
print(u.with_path(u.path.replace(".csv.gz", ".parquet")))
```

---

## 5) URL — media type and codec detection

```python
from yggdrasil.io import URL

# Media type inferred from extensions
u = URL.from_str("s3://bucket/prefix/data.parquet.zst")
print(u.extensions)    # ['parquet', 'zst']
print(u.mime_type)     # MimeType.PARQUET
print(u.codec)         # Codec.ZSTD
print(u.media_type)    # MediaType(mime=PARQUET, codec=ZSTD)

# Hive-partition segments
from yggdrasil.io.url import hive_split, hive_encode, hive_decode

seg = "date=2026-05-01"
print(hive_split(seg))    # ('date', '2026-05-01')

encoded = hive_encode(3.14)
print(hive_decode(encoded))   # 3.14
```

---

## 6) URL — anonymise for logs

```python
from yggdrasil.io import URL

# Credential-bearing URL (OAuth token in query, password in userinfo)
u = URL.from_str("https://admin:secret@api.example.com/data?token=abc123&source=prod")

# Remove sensitive fields entirely
print(u.anonymize("remove").to_string())
# https://api.example.com/data?source=prod

# Or redact to fixed placeholder
print(u.anonymize("redact").to_string())
# https://***:***@api.example.com/data?token=***&source=prod

# __repr__ is always safe for logs (uses redact by default)
print(repr(u))
```

---

## 7) URL — S3 / DBFS / Volume paths

```python
from yggdrasil.io import URL

# S3
s3 = URL.from_str("s3://my-bucket/prefix/2026/data.parquet")
print(s3.scheme, s3.host, s3.path)

# Databricks DBFS
dbfs = URL.from_str("dbfs:/tmp/scratch/run-001.csv")
print(dbfs.scheme, dbfs.path)

# Unity Catalog Volume
vol = URL.from_str("dbfs+volume://main/default/raw/events.parquet")
print(vol.scheme, vol.host, vol.path)

# Match by glob pattern
print(s3.match_pattern("*.parquet"))   # True
print(s3.matches_patterns(["*.csv", "*.parquet"]))   # True
```

---

## 8) BytesIO — buffer with codec and media-type awareness

```python
from yggdrasil.io import BytesIO

# Write and inspect metadata
with BytesIO() as buf:
    buf.write(b"id,name\n1,alice\n2,bob\n")
    buf.seek(0)
    print(buf.compression)    # None
    print(buf.media_type)     # None (raw bytes)
    print(buf.size)           # byte count

# Attach a URL so media type is inferred from extension
from yggdrasil.io import URL

buf = BytesIO(url=URL.from_str("file:///tmp/out.parquet"))
print(buf.mime_type)    # MimeType.PARQUET
```

---

## 9) SendConfig — request behavior knobs

```python
from yggdrasil.io.send_config import SendConfig

# Default: raise on non-2xx, no streaming, no cache
cfg = SendConfig()

# Inspect response instead of raising
cfg_soft = SendConfig(raise_error=False)

# Streaming large response
cfg_stream = SendConfig(stream=True)

# Collect result as Tabular (Arrow-backed) when endpoint returns tabular JSON
cfg_tabular = SendConfig(as_tabular=True)
```

---

## 10) SendManyConfig — batch dispatch

```python
from yggdrasil.io.send_config import SendManyConfig

# 5 concurrent workers, preserve submission order, up to 32 requests per batch
cfg = SendManyConfig(
    max_workers=5,
    ordered=True,
    batch_size=32,
    raise_error=False,          # collect errors rather than short-circuit
    max_in_flight=4,            # backpressure: at most 4 in-flight futures
)
```

---

## 11) CacheConfig — response caching

```python
from yggdrasil.io.send_config import CacheConfig, SendConfig
from yggdrasil.data.enums import Mode
import datetime

# Cache by URL + headers, store for 10 minutes
cache_cfg = CacheConfig(
    mode=Mode.APPEND,
    received_from=datetime.datetime.now(tz=datetime.timezone.utc),
    received_to=datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(minutes=10),
    anonymize="remove",           # strip credentials before computing cache key
)

# Wire into a send
send_cfg = SendConfig(local_cache=cache_cfg)
```

---

## 12) Primitive file formats

```python
from yggdrasil.io.primitive.parquet_file import ParquetFile
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile
from yggdrasil.io.primitive.csv_file import CSVFile
import pyarrow as pa

tbl = pa.table({"id": [1, 2, 3], "val": [1.1, 2.2, 3.3]})

# Parquet
pq = ParquetFile("/tmp/demo.parquet")
pq.write(tbl)
print(pq.read().num_rows)   # 3

# Arrow IPC
ipc = ArrowIPCFile("/tmp/demo.arrow")
ipc.write(tbl)
print(ipc.read().num_rows)

# CSV (detect types on read)
csv = CSVFile("/tmp/demo.csv")
csv.write(tbl)
print(csv.read())
```

---

## 13) Zip archives

```python
from yggdrasil.io.nested.zip_file import ZipFile
import pyarrow as pa

# Read entry from zip
z = ZipFile("/tmp/archive.zip")
for entry in z.iter_entries():
    print(entry.name, entry.size)

# Write multiple Arrow tables into one zip
tbl = pa.table({"x": [1, 2]})
with ZipFile("/tmp/out.zip") as z:
    z.write_arrow(tbl, name="data.parquet")
    z.write_arrow(tbl, name="copy.parquet")
```

---

## 14) Delta Lake log reader (no Spark)

```python
from yggdrasil.io.nested.delta import DeltaFolder

folder = DeltaFolder("/mnt/delta/events")

# Latest snapshot metadata
snap = folder.snapshot()
print("Version:", snap.version)
print("Schema:", snap.schema)
print("Files:", len(snap.add_files))

# Time-travel to version 5
snap_v5 = folder.snapshot(version=5)

# Walk add files — no Spark required
import pyarrow.parquet as pq, pyarrow as pa

parts = [pq.read_table(folder.path / f.path) for f in snap.add_files]
result = pa.concat_tables(parts)
print(result.num_rows)
```

---

## 15) IOStats — observability

```python
from yggdrasil.io import IOStats, URL

# Accumulate read/write stats across a session
stats = IOStats()
stats.record_read(url=URL.from_str("s3://bucket/a.parquet"), bytes_read=1_048_576)
stats.record_read(url=URL.from_str("s3://bucket/b.parquet"), bytes_read=524_288)
print(stats.total_bytes_read)   # 1 572 864
print(stats.operations)
```

---

## 16) Complex pattern: paginated API → Parquet archive

```python
import pyarrow as pa
from yggdrasil.io import URL
from yggdrasil.io.primitive.parquet_file import ParquetFile
from yggdrasil.http_ import HTTPSession
from yggdrasil.io.send_config import SendManyConfig

http = HTTPSession()
base = URL.from_str("https://api.example.com/events")

# Build all page requests
requests = [
    http.prepare_request("GET", base.add_param("page", str(p)).to_string())
    for p in range(1, 11)
]

# Fan-out concurrently
responses = list(http.send_many(requests, send_config=SendManyConfig(max_workers=5, ordered=True)))

# Concatenate and write
batches = [pa.table(r.json()["rows"]) for r in responses if r.ok]
result = pa.concat_tables(batches)

ParquetFile("/tmp/events.parquet").write(result)
print(f"Wrote {result.num_rows} rows")
```
