# yggdrasil.http_

`HTTPSession` is the preferred HTTP client in Yggdrasil for:

- typed request/response objects,
- retry + request preparation,
- batch dispatch (`send_many`),
- cache-aware transport config,
- easy conversion of tabular responses to Arrow/pandas/Polars/Spark.

---

## 1) Fast start

```python
from yggdrasil.http_ import HTTPSession

http = HTTPSession()
resp = http.get("https://httpbin.org/get", params={"source": "docs"})
print(resp.status, resp.ok)
print(resp.json())
```

---

## 2) Everyday request patterns

### GET / POST / PUT / PATCH / DELETE

```python
from yggdrasil.http_ import HTTPSession

http = HTTPSession()
print(http.get("https://httpbin.org/get").status)
print(http.post("https://httpbin.org/post", json={"name": "alice"}).status)
print(http.put("https://httpbin.org/put", json={"enabled": True}).status)
print(http.patch("https://httpbin.org/patch", json={"op": "replace"}).status)
print(http.delete("https://httpbin.org/delete").status)
```

### Headers and auth

```python
from yggdrasil.http_ import HTTPSession

http = HTTPSession(x_api_key="my-api-key")
resp = http.get(
    "https://httpbin.org/headers",
    headers={"x-trace-id": "run-001", "accept": "application/json"},
)
print(resp.json())
```

### Strict status handling

```python
from yggdrasil.http_ import HTTPSession

http = HTTPSession()
resp = http.get("https://httpbin.org/status/404")
resp.raise_for_status()  # raises on non-2xx
```

---

## 3) Prepared request workflow (`prepare_request` + `send`)

Use this when you need explicit control over a request before transport.

```python
from yggdrasil.http_ import HTTPSession

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
from yggdrasil.http_ import HTTPSession

http = HTTPSession()
requests = [
    http.prepare_request("GET", "https://httpbin.org/get", params={"idx": i})
    for i in range(10)
]

responses = list(http.send_many(requests, max_in_flight=5))
print([r.status for r in responses])
```

---

## 6) Proxy support

```python
from yggdrasil.http_ import HTTPSession

# Explicit proxy
http = HTTPSession(proxy="http://proxy.corp:8080")

# Proxy with authentication
http = HTTPSession(proxy="http://user:pass@proxy.corp:8080")

# Bypass proxy for specific hosts
http = HTTPSession(
    proxy="http://proxy.corp:8080",
    no_proxy="localhost,127.0.0.1,.internal.corp",
)
```

When no explicit `proxy` is passed, the session reads the standard environment
variables: `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` / `NO_PROXY`.

HTTPS targets use an HTTP CONNECT tunnel — the proxy sees only opaque bytes.
Plain HTTP targets send the absolute URL as the request path.

## 7) SSL verification

```python
from yggdrasil.http_ import HTTPSession

# Default: full certificate verification
http = HTTPSession(verify=True)

# Disable verification (emits InsecureRequestWarning)
http = HTTPSession(verify=False)

# Custom CA bundle
http = HTTPSession(verify="/path/to/ca-bundle.crt")

# Quick toggle from an existing session
insecure = http.insecure()
```

Suppress the warning when verification is intentionally disabled:

```python
from yggdrasil.http_.exceptions import disable_warnings, InsecureRequestWarning
disable_warnings(InsecureRequestWarning)
```

## 8) Response handling and conversions

```python
from yggdrasil.http_ import HTTPSession

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

## 9) Practical recipe: resilient paged pull + normalization

```python
from yggdrasil.http_ import HTTPSession

http = HTTPSession()

# stage 1: fetch multiple pages concurrently
reqs = [
    http.prepare_request("GET", "https://httpbin.org/get", params={"page": p})
    for p in range(1, 6)
]
responses = list(http.send_many(reqs, max_in_flight=3))

# stage 2: normalize to python rows (replace with your own extraction logic)
rows = []
for r in responses:
    payload = r.json()
    rows.append({"page": payload.get("args", {}).get("page"), "url": payload.get("url")})

print(rows)
```

