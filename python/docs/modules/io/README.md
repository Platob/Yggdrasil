# yggdrasil.io

Arrow/data pipeline IO subsystem with URL utilities, request/response models, buffer codecs, and HTTP sessions.

## Key exports from `yggdrasil.io`

```python
from yggdrasil.io import BytesIO, BufferConfig, Codec, URL, SendConfig, SendManyConfig
```

## Smart byte buffer

```python
from yggdrasil.io import BytesIO

with BytesIO() as buf:
    buf.write(b"hello")
    buf.seek(0)
    print(buf.compression)
    print(buf.media_type)
```

## URL parsing

```python
from yggdrasil.io import URL

u = URL.parse_str("https://example.com/a/b?q=1")
print(u.host, u.path)
```

## Preferred HTTP client

```python
from yggdrasil.io.http_ import HTTPSession

session = HTTPSession()
resp = session.get("https://example.com")
print(resp.status)
```
