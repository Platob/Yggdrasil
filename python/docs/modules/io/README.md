# yggdrasil.io

In-memory buffer with spill-to-disk, compression detection, and MIME-type inference.

## Key exports

```python
from yggdrasil.io import BytesIO, BufferConfig, Codec, MediaType
```

---

## `BytesIO` — smart byte buffer

Wraps `io.BytesIO` with automatic spill-to-disk when data exceeds a threshold, plus compression and media-type detection by magic bytes.

```python
BytesIO(config: BufferConfig = DEFAULT_CONFIG)
```

Properties (available after writing + seeking to 0):
- `buf.compression` → `Codec | None`
- `buf.media_type` → `MediaType | None`

---

## Bootstrap: write and inspect a buffer

```python
from yggdrasil.io import BytesIO

buf = BytesIO()
buf.write(b"some bytes")
buf.seek(0)

print(buf.compression)   # e.g. Codec.ZSTD or None
print(buf.media_type)    # e.g. MediaType('application/vnd.apache.parquet')

buf.close()
```

---

## Bootstrap: context manager

```python
from yggdrasil.io import BytesIO
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})

with BytesIO() as buf:
    pq.write_table(table, buf)
    buf.seek(0)
    print(buf.media_type)    # application/vnd.apache.parquet
    payload = buf.read()
```

---

## Bootstrap: spill-to-disk threshold

```python
from yggdrasil.io import BytesIO, BufferConfig

# spill to disk when buffer exceeds 64 MB
config = BufferConfig(max_memory_bytes=64 * 1024 * 1024)
buf = BytesIO(config)
# write large data ...
```

---

## `Codec` — compression detection

Detects compression format from magic bytes.

```python
from yggdrasil.io import Codec

Codec.GZIP
Codec.ZSTD
Codec.SNAPPY
Codec.LZ4
Codec.BROTLI
```

```python
from yggdrasil.io import Codec, BytesIO

buf = BytesIO()
buf.write(compressed_data)
buf.seek(0)

codec = buf.compression    # Codec enum value or None
if codec == Codec.ZSTD:
    print("zstd compressed")
```

---

## `MediaType` — MIME type inference

Infers MIME type from magic bytes or file extension.

```python
from yggdrasil.io import MediaType

# Common values
MediaType("application/vnd.apache.parquet")
MediaType("application/json")
MediaType("text/csv")
MediaType("application/octet-stream")
```

```python
from yggdrasil.io import BytesIO

with BytesIO() as buf:
    buf.write(parquet_bytes)
    buf.seek(0)
    print(buf.media_type.value)   # "application/vnd.apache.parquet"
```

---

## Import alias

The same module is also available at:

```python
from yggdrasil.pyutils.dynamic_buffer import BytesIO, BufferConfig, Codec, MediaType
```
