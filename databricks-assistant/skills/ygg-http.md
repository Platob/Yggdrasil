# Skill: call HTTP APIs with `HTTPSession` / `PreparedRequest` / `Response`

## When to use

The user asks to "call a REST API", "fetch JSON from a URL",
"download a file", "POST a payload with retries", "use a session
for keep-alive", "rate-limit my calls", or to wire an external
service into a Databricks notebook (Slack, OpenAI, custom internal
API, vendor data feed). Also when the user pastes raw `requests` /
`httpx` code that would benefit from caching, retries, and a
typed `Response` envelope.

## Primary surface

```python
from yggdrasil.io.http_ import HTTPSession, HTTPRequest, HTTPResponse
from yggdrasil.io import URL, SendConfig, SendManyConfig
```

`HTTPSession` is a singleton-by-config session: same
`(base_url, headers, auth, retry, cache, ...)` → same instance.
It is picklable across Spark workers / job tasks and safe as a dict
key. Don't construct a fresh `requests.Session` per call site.

## A single call

```python
session = HTTPSession.from_url("https://api.example.com")
req = session.prepare_request_before_send(HTTPRequest(method="GET", url="/v1/items"))
resp = session.send(req)

resp.status_code               # int
resp.headers                   # normalized + promoted
resp.content                   # bytes
resp.json()                    # dict / list via yggdrasil.pickle.json
resp.elapsed                   # timing
```

`HTTPResponse` preserves observability fields the rest of the
codebase relies on: normalized URL parts, promoted / remaining
headers, body bytes, payload hashes, timestamps, status / timing.
Don't strip them by re-wrapping into a plain dict.

## Many calls, batched + parallel

```python
reqs = [HTTPRequest(method="GET", url=f"/v1/items/{i}") for i in ids]
responses = session.send_many(reqs, config=SendManyConfig(
    max_workers=16,
    fail_fast=False,
))
```

`send_many` runs the requests through the session's bounded job
pool with the configured retry policy, surfaces per-request
exceptions on the matching `Response`, and respects the cache
config.

## Retries, caching, rate limits

Configure once on the session (or via `SendConfig` per call):

```python
session = HTTPSession.from_url(
    "https://api.example.com",
    headers={"Authorization": f"Bearer {token}"},
    retry={"total": 5, "backoff_factor": 0.5, "status_forcelist": (429, 500, 502, 503, 504)},
)
```

`HTTPSession` ships a `_TieredRetry` (subclass of `urllib3.Retry`)
that backs off differently per status class — don't hand-roll a
retry loop around `session.send(...)`. For local response caching
pass a `CacheConfig` (see `yggdrasil/io/session.py`); for remote /
shared caching, the same `Session` supports a Mongo-backed cache.

## URLs are values, not strings

```python
from yggdrasil.io import URL

u = URL.from_("https://api.example.com/v1/items?page=2")
u.scheme         # Scheme.HTTPS
u.host           # "api.example.com"
u.path           # "/v1/items"
u.query          # MultiDict-like
u / "extra"      # path join, returns a new URL
```

Pass `URL` objects around (picklable + hashable + singleton-keyed)
rather than re-parsing the same string at every layer.

## Don'ts

- Don't `import requests` and call `requests.get(...)` at the call
  site — `HTTPSession` gives you retries, caching, observability,
  and reuse across workers for free.
- Don't write a parallel `MyAPIClient` that wraps `requests`
  internally; subclass `HTTPSession` or compose it.
- Don't string-concat URL fragments (`base + "/" + path + "?" +
  ...`) — use `URL.from_(...)` + `url / "segment"` + query updates.
- Don't strip / rewrap an `HTTPResponse` into a plain dict — the
  promoted headers and timing fields are what makes downstream
  diagnostics tractable.
- Don't sleep-and-retry on a 429 in your own loop; configure
  `retry={"backoff_factor": ...}` on the session.
