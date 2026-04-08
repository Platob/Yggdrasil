# yggdrasil.io.http_

`HTTPSession` is the preferred HTTP client in Yggdrasil for:

- typed request/response objects,
- retry + request preparation,
- batch dispatch (`send_many`),
- cache-aware transport config,
- easy conversion of tabular responses to Arrow/pandas/Polars/Spark.

---

## 1) Fast start

```python
from yggdrasil.io.http_ import HTTPSession

http = HTTPSession()
resp = http.get("https://httpbin.org/get", params={"source": "docs"})
print(resp.status, resp.ok)
print(resp.json())
```

---

## 2) Everyday request patterns

### GET / POST / PUT / PATCH / DELETE

```python
from yggdrasil.io.http_ import HTTPSession

http = HTTPSession()
print(http.get("https://httpbin.org/get").status)
print(http.post("https://httpbin.org/post", json={"name": "alice"}).status)
print(http.put("https://httpbin.org/put", json={"enabled": True}).status)
print(http.patch("https://httpbin.org/patch", json={"op": "replace"}).status)
print(http.delete("https://httpbin.org/delete").status)
```

### Headers and auth

```python
from yggdrasil.io.http_ import HTTPSession

http = HTTPSession(x_api_key="my-api-key")
resp = http.get(
    "https://httpbin.org/headers",
    headers={"x-trace-id": "run-001", "accept": "application/json"},
)
print(resp.json())
```

### Strict status handling

```python
from yggdrasil.io.http_ import HTTPSession

http = HTTPSession()
resp = http.get("https://httpbin.org/status/404")
resp.raise_for_status()  # raises on non-2xx
```

---

## 3) Prepared request workflow (`prepare_request` + `send`)

Use this when you need explicit control over a request before transport.

```python
from yggdrasil.io.http_ import HTTPSession

http = HTTPSession()
prepared = http.prepare_request(
    method="POST",
    url="https://httpbin.org/post",
    json={"event": "order_created", "id": 123},
    headers={"x-source": "ygg-docs"},
)

resp = http.send(prepared)
print(resp.status)
print(resp.json().get("json"))
```

---

## 4) Parallel/batch dispatch (`send_many`)

```python
from yggdrasil.io.http_ import HTTPSession
from yggdrasil.io import SendManyConfig

http = HTTPSession()
requests = [
    http.prepare_request("GET", "https://httpbin.org/get", params={"idx": i})
    for i in range(10)
]

cfg = SendManyConfig(max_workers=5)
responses = list(http.send_many(requests, send_config=cfg))
print([r.status for r in responses])
```

---

## 5) Cache-aware requests (`SendConfig`)

```python
from yggdrasil.io.http_ import HTTPSession
from yggdrasil.io import SendConfig

http = HTTPSession()
cache_cfg = SendConfig(cache_for="15m")

r1 = http.get("https://httpbin.org/get", params={"q": "cache"}, send_config=cache_cfg)
r2 = http.get("https://httpbin.org/get", params={"q": "cache"}, send_config=cache_cfg)

print(r1.status, r2.status)
```

---

## 6) Response handling and conversions

```python
from yggdrasil.io.http_ import HTTPSession

resp = HTTPSession().get("https://httpbin.org/json")

print(resp.text[:80])
print(resp.json())
print(resp.ok)
```

If your endpoint returns tabular JSON/Arrow-compatible payloads, you can project to analytics formats:

```python
# table = resp.to_arrow_table()
# pdf = resp.to_pandas()
# plf = resp.to_polars()
# sdf = resp.to_spark()
```

---

## 7) Practical recipe: resilient paged pull + normalization

```python
from yggdrasil.io.http_ import HTTPSession
from yggdrasil.io import SendManyConfig

http = HTTPSession()

# stage 1: fetch multiple pages concurrently
reqs = [
    http.prepare_request("GET", "https://httpbin.org/get", params={"page": p})
    for p in range(1, 6)
]
responses = list(http.send_many(reqs, send_config=SendManyConfig(max_workers=3)))

# stage 2: normalize to python rows (replace with your own extraction logic)
rows = []
for r in responses:
    payload = r.json()
    rows.append({"page": payload.get("args", {}).get("page"), "url": payload.get("url")})

print(rows)
```

---

## 8) When to use `yggdrasil.requests.YGGSession` instead

Use `yggdrasil.io.http_.HTTPSession` for modern features (prepared requests, cache options, typed response conversions, batch dispatch).

Use `yggdrasil.requests.YGGSession` only when you want a minimal retry-only wrapper and do not need the richer IO/session features.
