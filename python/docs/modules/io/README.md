# yggdrasil.io

Yggdrasil IO and transport layer for URLs, request/response models, media detection, and HTTP execution.

## What this module gives you

- `URL` for immutable URL parsing and composition
- `SendConfig` (in `yggdrasil.http_`) for request behavior; batch via `HTTPSession.send_many`
- `HTTPSession` (`yggdrasil.http_`) as the preferred HTTP client

---

## 1) URL parsing/composition

```python
from yggdrasil.io import URL

u = URL.from_str("https://example.com/a/b?q=1")
print(u.host, u.path)
print(u.with_query_items({"q": 2, "lang": "en"}).to_string())
```

---

## 2) Preferred HTTP client

```python
from yggdrasil.http_ import HTTPSession

http = HTTPSession()
resp = http.get("https://example.com")
print(resp.status)
```

For advanced patterns (prepared requests, `send_many`, cache config, response conversions), see:

- [http_](../http_/README.md)
