# yggdrasil.io

Yggdrasil IO and transport layer for buffers, URLs, request/response models, media detection, and HTTP execution.

## What this module gives you

- `BytesIO` for spill-to-disk byte buffers with media/compression helpers
- `URL` for immutable URL parsing and composition
- `SendConfig` / `SendManyConfig` for request behavior and batching
- `HTTPSession` (`yggdrasil.io.http_`) as the preferred HTTP client

---

## 1) BytesIO quick example

```python
from yggdrasil.io import BytesIO

with BytesIO() as buf:
    buf.write(b"hello")
    buf.seek(0)
    print(buf.compression)
    print(buf.media_type)
```

---

## 2) URL parsing/composition

```python
from yggdrasil.io import URL

u = URL.parse_str("https://example.com/a/b?q=1")
print(u.host, u.path)
print(u.with_query_items({"q": 2, "lang": "en"}).to_string())
```

---

## 3) Preferred HTTP client

```python
from yggdrasil.io.http_ import HTTPSession

http = HTTPSession()
resp = http.get("https://example.com")
print(resp.status)
```

For advanced patterns (prepared requests, `send_many`, cache config, response conversions), see:

- [http_](http_/README.md)
