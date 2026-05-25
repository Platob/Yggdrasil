# yggdrasil.io

Transport layer for buffers, URLs, request/response models, media detection, and HTTP execution. `yggdrasil.io` is the byte substrate that every other layer (HTTP, Databricks, Mongo, Kafka, Parquet) reads and writes through.

## Surface map

| Symbol | Purpose |
| --- | --- |
| `URL` | Immutable URL parsing, composition, and path algebra |
| `BytesIO` | Spill-to-disk byte buffer with media/compression helpers |
| `Memory` | In-process byte buffer with Arrow/Parquet/JSON/IPC codec awareness |
| `MemoryStream` | Streaming read wrapper around `Memory` |
| `IOStats` | Timing, byte counts, and status for a single IO operation |
| `SendConfig` | Per-request send configuration (raise-on-error, cache config) |
| `SendManyConfig` | Batch dispatch config (`max_workers`, concurrency) |

HTTP execution lives in `yggdrasil.http_` — see [http_](../http_/README.md).

---

## URL

Immutable, hashable, and picklable. Parsed once, composed incrementally. Never mutates — all `with_*` and `join` methods return a new instance.

### Parse a URL

```python
from yggdrasil.io import URL

u = URL.from_str("https://api.example.com/v2/orders?page=1&limit=50")

print(u.scheme)   # "https"
print(u.host)     # "api.example.com"
print(u.port)     # 0 (not specified)
print(u.path)     # "/v2/orders"
print(u.query)    # "page=1&limit=50"
print(u.stem)     # "orders"
print(u.parts)    # ["v2", "orders"]
```

### Build from parts

```python
from yggdrasil.io import URL

u = URL(scheme="https", host="api.example.com", path="/v2/orders", query="page=1")
```

### From a pathlib.Path (local file URL)

```python
from pathlib import Path
from yggdrasil.io import URL

u = URL.from_pathlib(Path("/tmp/data/events.parquet"))
print(u.scheme)   # "file"
print(u.path)     # "/tmp/data/events.parquet"
```

### From a dict

```python
from yggdrasil.io import URL

u = URL.from_dict({"scheme": "https", "host": "example.com", "path": "/api"})
```

### Compose query parameters

```python
from yggdrasil.io import URL

base = URL.from_str("https://api.example.com/data")

# Add or replace individual params
u1 = base.with_query_items({"page": 2, "format": "arrow"})
print(u1)   # https://api.example.com/data?page=2&format=arrow

# Stable sort for cache-key reproducibility
u2 = base.with_query_items({"z": 1, "a": 2}, sort_keys=True)
```

### Path algebra

```python
from yggdrasil.io import URL

base = URL.from_str("https://api.example.com/v1")

# Append a path segment
child = base / "orders" / "123"
print(child)   # https://api.example.com/v1/orders/123

# Navigate up
print(child.parent)   # https://api.example.com/v1/orders

# Relative-to
print(child.relative_to(base))   # orders/123

# is_child_of
print(child.is_child_of(base))   # True
```

### Change scheme, host, or path

```python
from yggdrasil.io import URL

u = URL.from_str("https://api.example.com/v1/items?q=foo")

# Change just the host
u2 = URL(scheme=u.scheme, host="api-v2.example.com", path=u.path, query=u.query)
```

### Normalize for cache keys

```python
from yggdrasil.io import URL

# Always use sort_keys=True so the same logical query has a stable hash
canonical = URL.from_str("https://api.example.com/data").with_query_items(
    {"end": "2026-05-01T00:00:00Z", "start": "2026-04-01T00:00:00Z"},
    sort_keys=True,
)
```

---

## BytesIO

A spill-to-disk byte buffer. Behaves like `io.BytesIO` from stdlib but:

- Detects media type and compression from magic bytes.
- Preserves Arrow/Parquet/IPC framing.
- Spills to a temp file when a memory limit is exceeded (when configured).

### Basic read/write

```python
from yggdrasil.io import BytesIO

with BytesIO() as buf:
    buf.write(b"hello, world")
    buf.seek(0)
    print(buf.read())   # b"hello, world"
```

### Detect media type / compression

```python
from yggdrasil.io import BytesIO

with BytesIO() as buf:
    buf.write(b"\x1f\x8b" + b"\x00" * 20)  # gzip magic bytes
    buf.seek(0)
    print(buf.compression)   # Codec.GZIP (or similar)
    print(buf.media_type)    # MediaType based on content sniff
```

### Wrap existing bytes

```python
from yggdrasil.io import BytesIO

data = b"{'key': 'value'}"
buf = BytesIO.from_bytes(data)
buf.seek(0)
print(buf.read())
```

### Use as an Arrow/Parquet sink

```python
import pyarrow as pa
import pyarrow.parquet as pq
from yggdrasil.io import BytesIO

table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})

with BytesIO() as buf:
    pq.write_table(table, buf)
    buf.seek(0)
    read_back = pq.read_table(buf)

print(read_back.schema)
```

---

## Memory

An in-process byte buffer with codec and media-type awareness. Unlike `BytesIO`, `Memory` carries metadata (codec, media type, encoding) alongside the bytes and exposes Arrow/JSON/IPC read helpers directly.

### Create from bytes

```python
from yggdrasil.io import Memory

m = Memory.from_bytes(b'{"key": "value"}')
print(m.size)   # 16
```

### Create from a JSON-serializable object

```python
from yggdrasil.io import Memory

m = Memory.from_json({"event": "login", "user_id": 42})
print(m.to_json())   # {"event": "login", "user_id": 42}
```

### Create from an Arrow table

```python
import pyarrow as pa
from yggdrasil.io import Memory

table = pa.table({"id": [1, 2], "v": [3.0, 4.0]})
m = Memory.from_arrow_table(table)
tbl = m.to_arrow_table()
print(tbl.schema)
```

### Codec and media type inspection

```python
from yggdrasil.io import Memory
from yggdrasil.data.enums import Codec, MediaType

m = Memory.from_bytes(b"...", codec=Codec.ZSTD, media_type=MediaType.PARQUET)
print(m.codec)
print(m.media_type)
```

---

## MemoryStream

A streaming read wrapper around a `Memory` object. Use when you want to read a `Memory` object incrementally (e.g., pass it to a library that expects a file-like).

```python
from yggdrasil.io import Memory, MemoryStream

m = Memory.from_bytes(b"hello, stream")
stream = MemoryStream(m)

print(stream.read(5))   # b"hello"
print(stream.read())    # b", stream"
```

---

## SendConfig

Per-request behavior config for a single HTTP send. The main knob callers use is `raise_error`.

```python
from yggdrasil.io import SendConfig   # or from yggdrasil.io.send_config

# Default: raise on non-2xx status
cfg = SendConfig()

# Don't raise — inspect the response manually
cfg_no_raise = SendConfig(raise_error=False)
```

Usage with `HTTPSession`:

```python
from yggdrasil.http_ import HTTPSession
from yggdrasil.io.send_config import SendConfig

http = HTTPSession()
resp = http.send(
    http.prepare_request("GET", "https://httpbin.org/status/404"),
    send_config=SendConfig(raise_error=False),
)
print(resp.status, resp.ok)   # 404 False — no exception
```

---

## SendManyConfig

Batch dispatch config for `HTTPSession.send_many`. Controls concurrency and per-request behavior.

```python
from yggdrasil.io.send_config import SendManyConfig

# Basic: 5 concurrent workers
cfg = SendManyConfig(max_workers=5)

# Don't raise on per-response errors — collect all, filter after
cfg_collect = SendManyConfig(max_workers=8, raise_error=False)
```

### Fan-out example

```python
from yggdrasil.http_ import HTTPSession
from yggdrasil.io.send_config import SendManyConfig

http = HTTPSession()
requests = [
    http.prepare_request("GET", "https://httpbin.org/get", params={"i": i})
    for i in range(20)
]

cfg = SendManyConfig(max_workers=5)
responses = list(http.send_many(requests, send_config=cfg))
print([r.status for r in responses])   # [200, 200, ...]
```

### Collect all, skip failures

```python
from yggdrasil.http_ import HTTPSession
from yggdrasil.io.send_config import SendManyConfig

http = HTTPSession()
urls = ["https://httpbin.org/get", "https://httpbin.org/status/500", "https://httpbin.org/get"]
reqs = [http.prepare_request("GET", url) for url in urls]

responses = list(http.send_many(reqs, send_config=SendManyConfig(max_workers=3, raise_error=False)))
ok = [r for r in responses if r.ok]
print(f"{len(ok)}/{len(responses)} successful")
```

---

## IOStats

Carries timing, byte counts, and status for a completed IO operation. Surfaced by `HTTPSession.send` / `send_many` on the response object.

```python
from yggdrasil.http_ import HTTPSession

resp = HTTPSession().get("https://httpbin.org/get")
stats = resp.stats   # IOStats

print(stats.elapsed_ms)   # round-trip latency in milliseconds
print(stats.bytes_sent)
print(stats.bytes_received)
print(stats.status)
```

---

## Full pipeline: paginated API → Arrow table

```python
import pyarrow as pa
from yggdrasil.http_ import HTTPSession
from yggdrasil.io import URL
from yggdrasil.io.send_config import SendManyConfig

http = HTTPSession()
base = URL.from_str("https://httpbin.org/get")

# Build per-page requests with a canonical sort for cache key stability
reqs = [
    http.prepare_request(
        "GET",
        str(base.with_query_items({"page": p, "size": 50}, sort_keys=True)),
    )
    for p in range(1, 6)
]

# Fetch concurrently
responses = list(http.send_many(reqs, send_config=SendManyConfig(max_workers=3)))

# Collect into Arrow
rows = []
for r in responses:
    payload = r.json()
    rows.append({
        "page": int(payload.get("args", {}).get("page", 0)),
        "url":  payload.get("url", ""),
    })

table = pa.table({
    "page": pa.array([r["page"] for r in rows], type=pa.int64()),
    "url":  pa.array([r["url"]  for r in rows]),
})
print(table)
```
