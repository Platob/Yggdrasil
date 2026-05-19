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

For failures that **outlast** the retry budget (vendor down for an
hour, auth token revoked, persistent 5xx on one record), wrap the
session in `ErrorNotifyingHTTPSession` — it fires a notifier
callback on persistent failure and returns a synthetic
`status_code=0` response instead of raising, so a flaky upstream
doesn't take the ingestion pipeline down. See
[`ygg-resilient-ingestion`](ygg-resilient-ingestion.md) for the
full pattern (notifier shapes, quarantine tables, dead-letter,
idempotent re-runs).

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

## Persist responses with `Response` / `RESPONSE_SCHEMA`

`HTTPResponse` (`yggdrasil.io.response.Response`) is itself a
`Tabular[ResponseOptions]` — every reply is a single Arrow row
shaped by `RESPONSE_SCHEMA`. The schema is the contract every
ingestion path can land into Delta without re-modeling:

| Column | Meaning |
| --- | --- |
| `hash` + `body_size` | Composite primary key (RELY). |
| `partition_key` | Derived from URL / route — co-locates request + response. |
| `received_at` | UTC timestamp (the `time_column`). |
| `status_code`, `headers`, `body`, `body_hash` | Response payload + envelope. |
| `request_*` | Flattened, prefixed copy of the matching request fields (no nested struct lookups). |

Schema tags: `domain="http"`, `entity="response"`, `layer="bronze"`,
`namespace="yggdrasil.io.response"`. Use these as the basis for the
`raw_<entity>` table when the source is an HTTP API — the raw layer
*is* the response cache (see [`ygg-data-modeling`](ygg-data-modeling.md)).

```python
from yggdrasil.io.response import RESPONSE_SCHEMA

dbc.table("main.vendor_orders.raw_orders_responses").ensure_created(
    schema=RESPONSE_SCHEMA,            # PK + partition_by tags ride along
    comment="Bronze cache of /v1/orders responses (yggdrasil RESPONSE_SCHEMA).",
)
```

## Choose local vs remote cache via `SchemaSession`

[`yggdrasil.databricks.schema.session.SchemaSession`](https://github.com/Platob/Yggdrasil/blob/main/python/src/yggdrasil/databricks/schema/session.py)
is an `HTTPSession` subclass that maps every outbound request to a
per-path Delta table inside a bound `Schema`. The parent's `_send`
pipeline already owns the local → remote → network → writeback
flow; the subclass stamps each request with the right
`CacheConfig`.

```python
from yggdrasil.data.enums import Mode
from yggdrasil.databricks.schema.session import SchemaSession

schema = dbc.schema("main.vendor_orders")
schema.ensure_created()

session = SchemaSession(
    schema=schema,
    base_url="https://api.vendor.example.com",
    mode=Mode.APPEND,         # read-through cache, write-once per URL
    local_cache=True,          # ~/.yggdrasil/cache/response/<host>/<path>/<hash>.arrow
    table_cache_ttl=3600,
)

resp = session.send(HTTPRequest(method="GET", url="/v1/orders/12345"))
# 1st call: network → write to per-path Delta table → local Arrow file
# 2nd call: local Arrow file (no network, no Delta read)
```

### Decision tree — which cache tier to enable

| Source spec | `local_cache` | `mode` | Notes |
| --- | --- | --- | --- |
| Static reference data (countries, currencies, exchange holidays) | `True` (default) | `APPEND` | Read once, never refetch. |
| Slow-changing per-id resource (`GET /customers/<id>`) | `True` | `APPEND` | Refresh by trashing the row. |
| Vendor with strict rate limit (e.g. 100 req/day) | `True` | `APPEND` | Local fast-path is decisive. |
| Multi-worker job (Spark / `send_many`) | `False` | `APPEND` | Remote cache is the only shared tier; local files don't help workers see each other's hits. |
| Time-series / streaming data | `False` | `UPSERT` | Always refetch; remote tier replays the last response only. |
| One-shot batch (no rerun needed) | `False` | `APPEND` *or* don't use `SchemaSession` | Plain `HTTPSession` + direct `Table.insert` is cheaper. |
| Dev / ad-hoc notebook | `True` | `APPEND` | Local Arrow cache survives kernel restarts. |

Rules of thumb:

- `local_cache=True` wins **single-process** repeat reads. Picks up
  the same response in ~ms vs the ~100 ms Delta read.
- `local_cache=False` is correct **whenever multiple workers share
  the same job** — each worker's local dir is invisible to the others,
  and the remote Delta tier is the synchronization point.
- `Mode.APPEND` is read-through (cache wins, network on miss).
  `Mode.UPSERT` is bypass-the-read (network wins, cache stores the
  latest). Use `UPSERT` when you'd otherwise be tempted to manually
  invalidate.

### Per-call override

Stamp a different `CacheConfig` per request when one URL on the same
session needs a different policy:

```python
from yggdrasil.io.send_config import SendConfig, CacheConfig

resp = session.send(
    HTTPRequest(method="GET", url="/v1/health"),
    config=SendConfig(cache=CacheConfig(enabled=False)),   # never cache /v1/health
)
```

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
