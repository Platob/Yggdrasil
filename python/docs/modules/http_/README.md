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

## 6) Response handling and conversions

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

## 7) Practical recipe: resilient paged pull + normalization

```python
from yggdrasil.http_ import HTTPSession
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

## 8) Subclass HTTPSession for custom auth

The recommended pattern for vendor integrations: one small subclass per source, not a mega-class with flags.

```python
from yggdrasil.http_ import HTTPSession
from yggdrasil.io import URL

class MyVendorSession(HTTPSession):
    """HTTPSession that injects Bearer auth and snaps time params to hourly buckets."""

    def __init__(self, api_key: str, *, base_url: str = "https://api.vendor.example.com") -> None:
        super().__init__(
            base_url=URL.from_str(base_url),
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._api_key = api_key

    def prepare_request_before_send(self, request):
        # Normalise all requests to the same base; add version header
        req = super().prepare_request_before_send(request)
        return req._replace(headers={**req.headers, "x-api-version": "2026-05"})

    def get_timeseries(self, symbol: str, start: str, end: str) -> list:
        # Snap timestamps to hourly grid for cache stability
        import datetime as dt
        snap = lambda s: dt.datetime.fromisoformat(s).replace(minute=0, second=0).isoformat()
        return self.get(
            "/timeseries",
            params={"symbol": symbol, "start": snap(start), "end": snap(end)},
        ).json()["data"]


vendor = MyVendorSession(api_key="<your-key>")
data = vendor.get_timeseries("AAPL", "2026-05-01T10:23:00", "2026-05-01T14:37:00")
print(len(data), "rows")
```

---

## 9) Streaming large response body

```python
from yggdrasil.http_ import HTTPSession
from yggdrasil.io.send_config import SendConfig

http = HTTPSession()
cfg  = SendConfig(stream=True, raise_error=True)

resp = http.get("https://example.com/large-export.ndjson", send_config=cfg)

import pyarrow.json as paj
import pyarrow as pa

batches = []
for chunk in resp.iter_lines():
    if chunk:
        batches.append(paj.read_json(pa.py_buffer(chunk)))

result = pa.concat_tables(batches)
print(result.num_rows)
```

---

## 10) Retry policy — rate-limit aware

`HTTPSession` retries 429 / 5xx automatically and respects `Retry-After`. To customise the policy, subclass and override `_build_retry`:

```python
from yggdrasil.http_ import HTTPSession
from urllib3.util.retry import Retry

class ResilientSession(HTTPSession):
    def _build_retry(self) -> Retry:
        return Retry(
            total=10,
            backoff_factor=1.5,
            status_forcelist={429, 500, 502, 503, 504},
            allowed_methods={"GET", "POST", "PUT"},
            respect_retry_after_header=True,
            raise_on_status=False,
        )

http = ResilientSession()
resp = http.get("https://api.example.com/data")
```

---

## 11) Collect Arrow table from tabular JSON endpoint

```python
from yggdrasil.http_ import HTTPSession
from yggdrasil.io.send_config import SendConfig
import pyarrow as pa
import pyarrow.json as paj

http = HTTPSession()

# Endpoint returns newline-delimited JSON rows
resp = http.get("https://api.example.com/export.ndjson")
tbl  = paj.read_json(pa.py_buffer(resp.content))
print(tbl.schema)
print(tbl.num_rows)
```

