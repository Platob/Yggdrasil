# IO and HTTP

`yggdrasil.io` is the transport surface — buffers, URLs, requests/responses, codecs, media types, sessions, pagination, caching. Use it instead of reaching directly for `urllib3`/`requests`/`io`.

## Preferred HTTP client: `HTTPSession`

```python
from yggdrasil.http_ import HTTPSession

http = HTTPSession()
print(http.get("https://httpbin.org/get").json())
```

### Verbs

```python
http.get("https://httpbin.org/get")
http.post("https://httpbin.org/post", json={"name": "alice"})
http.put("https://httpbin.org/put", json={"enabled": True})
http.patch("https://httpbin.org/patch", json={"op": "replace"})
http.delete("https://httpbin.org/delete")
```

### Headers and auth

```python
http = HTTPSession(x_api_key="my-api-key")
http.get("https://httpbin.org/headers", headers={"x-trace-id": "run-001"})
```

### Strict status

```python
resp = http.get("https://httpbin.org/status/404")
resp.raise_for_status()        # raises on non-2xx
```

## Prepared request workflow

Useful when you need to inspect, mutate, or sign a request before it goes out:

```python
prepared = http.prepare_request(
    method="POST",
    url="https://httpbin.org/post",
    json={"event": "order_created", "id": 123},
    headers={"x-source": "ygg-docs"},
)
resp = http.send(prepared)
print(resp.status, resp.json()["json"])
```

## Batch / parallel dispatch

```python
from yggdrasil.io import SendManyConfig

reqs = [
    http.prepare_request("GET", "https://httpbin.org/get", params={"idx": i})
    for i in range(10)
]
cfg = SendManyConfig(max_workers=5)
responses = list(http.send_many(reqs, send_config=cfg))
print([r.status for r in responses])
```

## Response → analytics formats

If the server returns tabular JSON / Arrow, project straight into your engine:

```python
resp = http.get("https://api.example.com/v1/orders?format=arrow")
resp.to_arrow_table()
resp.to_pandas()
resp.to_polars()
resp.to_spark()
```

For free-form JSON:

```python
resp = http.get("https://httpbin.org/json")
resp.text[:80]
resp.json()
resp.ok
```

## Resilient paged pull (recipe)

```python
from yggdrasil.http_ import HTTPSession
from yggdrasil.io import SendManyConfig

http = HTTPSession()

# Stage 1: fetch pages concurrently.
reqs = [http.prepare_request("GET", "https://httpbin.org/get", params={"page": p})
        for p in range(1, 6)]
responses = list(http.send_many(reqs, send_config=SendManyConfig(max_workers=3)))

# Stage 2: normalize.
rows = []
for r in responses:
    payload = r.json()
    rows.append({"page": payload.get("args", {}).get("page"),
                 "url":  payload.get("url")})
print(rows)
```

## URL parsing and composition

```python
from yggdrasil.io import URL

u = URL.from_str("https://example.com/a/b?q=1")
print(u.host, u.path)
print(u.with_query_items({"q": 2, "lang": "en"}).to_string())
```

`URL` is immutable. Mutate via `with_*` methods that return a new instance.

## Observability fields

Tooling downstream relies on the request/response models preserving:

- normalized URL parts,
- promoted/remaining headers,
- body bytes,
- payload hashes,
- timestamps,
- status / timing.

If you write a wrapper, keep these fields populated.
