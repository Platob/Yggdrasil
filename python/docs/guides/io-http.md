# IO and HTTP Guide

## Preferred HTTP client

Use `HTTPSession` for new network-facing functionality.

```python
from yggdrasil.io.http_ import HTTPSession

session = HTTPSession()
response = session.get("https://example.com")
print(response.status)
```

## BytesIO buffer

```python
from yggdrasil.io import BytesIO

with BytesIO() as buf:
    buf.write(b"hello")
    buf.seek(0)
    print(buf.media_type)
```

## URL resources

`URL` and `URLResource` support immutable parsing and resource registration for URL-addressable handlers.
