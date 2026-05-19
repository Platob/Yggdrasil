# Skill: resilient ingestion — keep pipelines green when external data fails

## When to use

The user asks for "robust ingestion", "don't break when the vendor
API is down", "minimum strict failing", "graceful degradation",
"alert me but keep running", "what if a few rows are malformed",
"how do I make this retry-safe", or pastes a brittle `requests.get()
→ json() → insert` loop and asks to harden it. Also when chaining
from [`ygg-ingestion-pipeline`](ygg-ingestion-pipeline.md) — this
skill is the **failure-handling layer** that wraps step 5 (fetch +
load) so a 429 storm, a 502 from a flaky upstream, or a single
malformed row doesn't abort the whole job.

Pair with [`ygg-http`](ygg-http.md) (transport surface + retry
config), [`ygg-databricks-tables`](ygg-databricks-tables.md)
(`merge` / `async_insert` for idempotent writes), and
[`ygg-data-modeling`](ygg-data-modeling.md) (provenance columns
that make re-runs safe).

## The three failure shapes

Every ingestion call has three categorically different failure modes
— don't conflate them.

| Shape | Example | Right tool |
| --- | --- | --- |
| **Transient** | 429 Too Many Requests, 503 Service Unavailable, connect timeout, TLS reset | `HTTPSession` `retry=` config (`_TieredRetry` already backs off per status class + honours `Retry-After`) |
| **Persistent** | 429 / 5xx outlasts the retry budget, vendor is down for an hour, auth token revoked | `ErrorNotifyingHTTPSession` — notify ops, return synthetic 0-status response, **keep running** |
| **Poison data** | One row in the page has a bad decimal / missing PK / unparseable timestamp | Per-row cast fallback + quarantine table — drop the offender, **keep the batch** |

The "minimum strict failing" stance means: the pipeline stops only
when the *workspace* is broken (UC unreachable, schema reconciliation
fails, secret missing). Every other class of failure routes to a
notifier + a quarantine row + a metric, and the schedule keeps
running.

## `ErrorNotifyingHTTPSession` — notify + continue

The wrap-and-continue session lives in
`yggdrasil.io.http_.notifying_session`. It subclasses `HTTPSession`,
intercepts `_send` so persistent failures fire a notifier callback
**and** return a response to the caller (synthetic `status_code=0`
when the wire-level send raised). Set `raise_on_failure=True` only
when you want the strict shape (notify *and* re-raise — pipeline
stops).

```python
from yggdrasil.io.http_ import (
    ErrorNotifyingHTTPSession, HTTPRequest, smtp_email_notifier,
)

session = ErrorNotifyingHTTPSession.from_url(
    "https://api.vendor.example.com",
    headers={"Authorization": f"Bearer {dbc.secrets.get('vendor', 'api_token')}"},
    retry={"total": 5, "backoff_factor": 0.5,
           "status_forcelist": (429, 500, 502, 503, 504)},
    notifier=smtp_email_notifier(
        host="smtp.example.com",
        from_addr="data-platform@example.com",
        to_addrs=["data-oncall@example.com"],
        subject_prefix="[ingest-orders]",
    ),
    raise_on_failure=False,   # default — keep pipeline running on persistent failures
)

resp = session.send(HTTPRequest(method="GET", url="/v1/orders?page=1"))
if resp.status_code == 0:
    # Wire-level failure — notifier already fired. Skip this page,
    # next scheduled run picks the window up again.
    return 0
if not resp.ok:
    # Persistent HTTP-level failure — notifier already fired.
    # Decide per-route: skip (idempotent re-run), or route to
    # raw_<entity>_failures for forensics. Don't raise.
    return 0
```

`notifier=` is **transient state** — not part of the singleton key,
not pickled across Spark workers. Re-attach on the receiving side
if a worker needs the alert channel (most jobs only need it on the
driver).

## Notifier shapes

Notifier signature: `(response, exc, session) -> None`. Exactly one of
`response` / `exc` is set per invocation. Notifier exceptions are
caught + logged — never load-bearing.

### SMTP (stdlib, no extra deps)

```python
from yggdrasil.io.http_ import smtp_email_notifier

notifier = smtp_email_notifier(
    host="smtp.example.com",
    port=587,
    from_addr="data-platform@example.com",
    to_addrs=("data-oncall@example.com", "data-lead@example.com"),
    use_tls=True,
    username=dbc.secrets.get("smtp", "user"),
    password=dbc.secrets.get("smtp", "password"),
    subject_prefix="[ygg-ingest]",
)
```

Body carries method / URL / status / exception type + 2 KB body
excerpt — enough to triage without spamming the whole payload.

### Slack webhook

```python
def slack_notifier(response, exc, session):
    from yggdrasil.io.http_ import HTTPSession, HTTPRequest
    from yggdrasil.pickle import json as ygg_json

    webhook = HTTPSession.from_url(dbc.secrets.get("slack", "ingest_webhook"))
    method = getattr(getattr(response, "request", None), "method", "?")
    url = str(getattr(getattr(response, "request", None), "url", "?"))
    status = getattr(response, "status_code", 0)
    text = (
        f":rotating_light: {session!r}\n"
        f"`{method} {url}` → `{status}`"
        + (f"\nerror: `{type(exc).__name__}: {exc}`" if exc else "")
    )
    webhook.send(HTTPRequest(
        method="POST",
        url="/",
        body=ygg_json.dumps({"text": text}, to_bytes=True),
        headers={"Content-Type": "application/json"},
    ))
```

### Log-only (dev / staging)

```python
import logging
LOGGER = logging.getLogger("ingest.notifier")

def log_notifier(response, exc, session):
    LOGGER.warning(
        "Ingestion failure on %r (status=%s, exc=%r)",
        session, getattr(response, "status_code", None), exc,
    )
```

Use this in dev — same code path as prod, no email noise during
schema exploration.

### Compose notifiers

```python
def composite(*notifiers):
    def fan_out(response, exc, session):
        for n in notifiers:
            try:
                n(response, exc, session)
            except Exception:
                LOGGER.exception("Notifier %r raised; suppressing", n)
    return fan_out

session.notifier = composite(slack_notifier, log_notifier)
```

The session's own `_fire_notifier` already swallows + logs notifier
exceptions, but compose-and-catch makes a per-target failure visible
without taking down the rest of the alert fan-out.

## Synthetic `status_code=0` — the "I never got a reply" marker

When `_local_send` raises (DNS, TLS, connect timeout, socket reset
post-retry), the notifier fires and the session returns an
`HTTPResponse` with `status_code=0`, the request preserved, an
`x-ygg-error` header carrying `"<ExceptionType>: <message>"`, and an
empty body. **`status_code=0` is the only HTTP-illegal status the
codebase uses — it's safe to branch on.**

```python
def fetch_or_skip(session, request):
    resp = session.send(request)
    if resp.status_code == 0:
        # Wire-level failure already alerted. Next scheduled run
        # picks the window up because raw_<entity> MERGE is idempotent.
        LOGGER.warning("Skipping %r — wire-level failure (%s)",
                       request, resp.headers.get("x-ygg-error"))
        return None
    if not resp.ok:
        LOGGER.warning("Skipping %r — persistent %d", request, resp.status_code)
        return None
    return resp.json()
```

Don't write a parallel `try / except RequestException` around the
session — `ErrorNotifyingHTTPSession` already absorbs the exception
into the synthetic response.

## Partial-batch tolerance — quarantine, don't abort

One bad row should not kill a 50 K-row page. The pattern:

1. **Cast the whole batch in lenient mode** (silently demote, capture
   per-row errors).
2. **Split valid rows → target, invalid rows → `raw_<entity>_quarantine`**
   with the cast error attached.
3. **Alert when quarantine ratio exceeds a threshold**, not on every
   row — quarantine is the *expected* dump for a flaky vendor.

```python
from yggdrasil.data import Schema, Field, DataType
from yggdrasil.data.cast import convert
from yggdrasil.data.cast.options import CastOptions

QUARANTINE_SCHEMA = Schema.from_fields([
    Field("_payload_json", DataType.string(), nullable=False,
          comment="Raw source row JSON — preserved verbatim for forensics."),
    Field("_cast_error",   DataType.string(), nullable=False,
          comment="Exception type + message from the failed cast."),
    Field("_ingested_at",  DataType.timestamp("UTC"), nullable=False,
          tags={"primary_key": True, "partition_by": True}),
    Field("_source",       DataType.string(), nullable=False),
    Field("_source_url",   DataType.string(), nullable=True),
    Field("_batch_id",     DataType.string(), nullable=False,
          tags={"primary_key": True}),
])

quarantine = dbc.table("main.vendor_orders.raw_orders_quarantine")
quarantine.ensure_created(
    schema=QUARANTINE_SCHEMA,
    comment="Rows that failed cast into raw_orders. Forensic-only.",
)


def cast_and_split(rows, target_schema, *, source, source_url, batch_id):
    """Cast a batch; return (good_arrow, bad_arrow). Never raises on row errors."""
    from yggdrasil.pickle import json as ygg_json
    import datetime as dt
    import pyarrow as pa

    good, bad = [], []
    now = dt.datetime.now(dt.timezone.utc)
    for r in rows:
        try:
            # Per-row cast probe — only the rare path; the bulk cast
            # below is the vectorised one.
            convert(r, target_schema,
                    options=CastOptions(target_field=target_schema, strict=True))
            good.append(r)
        except Exception as exc:
            bad.append({
                "_payload_json": ygg_json.dumps(r),
                "_cast_error":   f"{type(exc).__name__}: {exc}",
                "_ingested_at":  now,
                "_source":       source,
                "_source_url":   source_url,
                "_batch_id":     batch_id,
            })

    good_arrow = pa.Table.from_pylist(good, schema=target_schema.to_arrow()) if good else None
    bad_arrow  = pa.Table.from_pylist(bad,  schema=QUARANTINE_SCHEMA.to_arrow()) if bad  else None
    return good_arrow, bad_arrow
```

The per-row probe is the **documented fallback** from CLAUDE.md's
"No Python for-loops over data" rule — exempt because the vectorised
bulk cast can't isolate which row caused a failure. Run the bulk
cast first on the optimistic path; only iterate when the bulk cast
raises.

```python
def cast_lenient(rows, target_schema, **provenance):
    import pyarrow as pa
    try:
        # Vectorised happy path — one C++ pass, no Python loop.
        return pa.Table.from_pylist(rows, schema=target_schema.to_arrow()), None
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
        # One bad row poisoned the bulk cast — fall back to per-row probe.
        return cast_and_split(rows, target_schema, **provenance)
```

Then write both sides in the same transaction window:

```python
good, bad = cast_lenient(rows, RAW_ORDERS_SCHEMA,
                         source="vendor_orders",
                         source_url=f"{ORDERS_API}{ORDERS_PATH}",
                         batch_id=batch_id)
if good is not None:
    raw_tbl.merge(good, keys=["order_id", "_ingested_at"])
if bad is not None:
    quarantine.insert(bad)
    if bad.num_rows / max(len(rows), 1) > 0.05:    # > 5 % poisoned
        session.notifier(None, RuntimeError(
            f"Quarantine ratio {bad.num_rows}/{len(rows)} on {source_url}"
        ), session)
```

## Idempotent re-run — the safety net for everything above

Notify-and-continue only works because the **next scheduled run can
safely re-fetch the same window**. The two mechanics that make that
true:

1. **MERGE on `(natural_id, _ingested_at)`**, not `INSERT`. Re-fetching
   the same row in a later run lands a new `_ingested_at` row instead
   of duplicating; the curated layer reads `MAX(_ingested_at)` per
   `natural_id` (see [`ygg-curated-views`](ygg-curated-views.md)).
2. **`_payload_hash` (xxhash64) dedup column**, so the curated layer
   can also collapse identical re-fetches to one row when needed.

```python
raw_tbl.merge(
    arrow,
    keys=["order_id", "_ingested_at"],
    update_columns=["customer_id", "amount", "ccy", "country", "status",
                    "_source_url", "_payload_hash", "_batch_id"],
)
```

A 429 storm that skips three hourly runs has no permanent effect:
when the vendor recovers, the next run's `since=<last_success - 1h>`
window picks the gap up, the MERGE lands the missed rows with the
real `_ingested_at`, and the curated layer's `MAX(_ingested_at)`
serves the right value.

## Strict mode — when to flip `raise_on_failure=True`

Default is **lenient** (notify + continue). Flip to strict when:

- The operation is **non-idempotent on partial failure** — a payment
  webhook ack, a one-shot reconciliation that must be all-or-nothing.
- The downstream consumer **already has its own resilience** — e.g.
  a Job task that the orchestrator (Airflow / Databricks Workflows)
  is meant to retry by re-running the task.
- The failure **is the signal** — a "vendor is down" alert that
  should page someone, not log silently into the next run.

```python
session = ErrorNotifyingHTTPSession.from_url(
    "https://api.vendor.example.com",
    notifier=pagerduty_notifier,
    raise_on_failure=True,    # pipeline stops; PagerDuty handles human alert
)
```

Even in strict mode, the notifier still fires *first* — alerts beat
re-raises, never the other way around.

## End-to-end resilient pipeline (paginated, rate-limited, lossy)

```python
import datetime as dt
import logging
import uuid

from yggdrasil.databricks import DatabricksClient
from yggdrasil.data import Schema, Field, DataType
from yggdrasil.io.http_ import (
    ErrorNotifyingHTTPSession, HTTPRequest, smtp_email_notifier,
)
from yggdrasil.pickle import json as ygg_json
from yggdrasil.xxhash import xxh64

LOGGER = logging.getLogger(__name__)

dbc = DatabricksClient()


def ingest_orders_since(since_iso: str) -> dict:
    """Resilient ingestion: vendor outage → skipped page; bad row → quarantine.

    Returns counts for the scheduler to log / alert on:
        {"good": int, "bad": int, "skipped_pages": int}
    """
    session = ErrorNotifyingHTTPSession.from_url(
        "https://api.vendor.example.com",
        headers={"Authorization": f"Bearer {dbc.secrets.get('vendor', 'api_token')}"},
        retry={"total": 5, "backoff_factor": 0.5,
               "status_forcelist": (429, 500, 502, 503, 504)},
        notifier=smtp_email_notifier(
            host="smtp.example.com", port=587, use_tls=True,
            from_addr="data-platform@example.com",
            to_addrs=("data-oncall@example.com",),
            username=dbc.secrets.get("smtp", "user"),
            password=dbc.secrets.get("smtp", "password"),
            subject_prefix="[ingest-orders]",
        ),
        raise_on_failure=False,
    )

    raw_tbl    = dbc.table("main.vendor_orders.raw_orders")
    quarantine = dbc.table("main.vendor_orders.raw_orders_quarantine")
    batch_id   = str(uuid.uuid4())
    now        = dt.datetime.now(dt.timezone.utc)

    good_total = bad_total = skipped = 0
    next_page = f"/v1/orders?since={since_iso}"

    while next_page:
        resp = session.send(HTTPRequest(method="GET", url=next_page))

        if resp.status_code == 0 or not resp.ok:
            # Persistent failure — notifier already fired. Skip page,
            # next scheduled run picks the window up.
            LOGGER.warning("Skipping page %r — status=%s, x-ygg-error=%s",
                           next_page, resp.status_code,
                           resp.headers.get("x-ygg-error", ""))
            skipped += 1
            break

        payload = resp.json()
        rows = payload.get("data", [])
        for r in rows:
            r["_ingested_at"]  = now
            r["_source"]       = "vendor_orders"
            r["_source_url"]   = str(resp.request.url)
            r["_payload_hash"] = xxh64(ygg_json.dumps(r, to_bytes=True)).hexdigest()
            r["_batch_id"]     = batch_id

        good, bad = cast_lenient(
            rows, RAW_ORDERS_SCHEMA,
            source="vendor_orders",
            source_url=str(resp.request.url),
            batch_id=batch_id,
        )
        if good is not None:
            raw_tbl.merge(good, keys=["order_id", "_ingested_at"])
            good_total += good.num_rows
        if bad is not None:
            quarantine.insert(bad)
            bad_total += bad.num_rows

        next_page = payload.get("next")

    if bad_total and bad_total / max(good_total + bad_total, 1) > 0.05:
        # Above 5 % poison — escalate.
        session.notifier(None, RuntimeError(
            f"Quarantine ratio {bad_total}/{good_total + bad_total} for batch {batch_id}"
        ), session)

    LOGGER.info(
        "Ingested vendor_orders (good=%d, bad=%d, skipped_pages=%d, batch_id=%s)",
        good_total, bad_total, skipped, batch_id,
    )
    return {"good": good_total, "bad": bad_total, "skipped_pages": skipped}
```

What you get:

- **Transient failures** absorbed by `_TieredRetry` — invisible to the
  caller, no alert noise.
- **Persistent failures** notify ops + skip the page; the next scheduled
  run's MERGE re-lands the missed window.
- **Bad rows** drop into `raw_orders_quarantine` with the cast error
  attached; the good rows land normally.
- **Quarantine ratio > 5 %** fires a second alert (data-quality, not
  vendor-availability).
- **Counts come back to the scheduler** so the Job's tags / metrics
  surface the run health.

## When persistence ≠ vendor availability — the dead-letter pattern

Sometimes the vendor *is* up but a specific URL keeps 5xx-ing (e.g.
one customer's record is corrupted server-side). For that case,
land the failing request itself in a dead-letter table so future
runs skip it explicitly:

```python
DEAD_LETTER_SCHEMA = Schema.from_fields([
    Field("request_url",   DataType.string(), nullable=False,
          tags={"primary_key": True}),
    Field("last_status",   DataType.int32(),  nullable=False),
    Field("first_seen_at", DataType.timestamp("UTC"), nullable=False),
    Field("last_seen_at",  DataType.timestamp("UTC"), nullable=False,
          tags={"partition_by": True}),
    Field("attempts",      DataType.int32(),  nullable=False),
    Field("last_error",    DataType.string(), nullable=True),
])

dead_letter = dbc.table("main.vendor_orders.raw_orders_dead_letter")
dead_letter.ensure_created(schema=DEAD_LETTER_SCHEMA)


def with_dead_letter(session, request, dead_letter_tbl):
    """Skip URLs already in the dead-letter table; record new persistent failures."""
    resp = session.send(request)
    if resp.status_code == 0 or not resp.ok:
        dead_letter_tbl.merge(
            pa.table({
                "request_url":   [str(request.url)],
                "last_status":   [resp.status_code],
                "first_seen_at": [dt.datetime.now(dt.timezone.utc)],
                "last_seen_at":  [dt.datetime.now(dt.timezone.utc)],
                "attempts":      [1],
                "last_error":    [resp.headers.get("x-ygg-error", "")],
            }),
            keys=["request_url"],
            update_columns=["last_status", "last_seen_at", "last_error"],
            # Note: increment `attempts` via a `WHEN MATCHED THEN UPDATE SET attempts = attempts + 1`
            # SQL — the merge helper supports custom update expressions.
        )
        return None
    return resp
```

The dead-letter table is the **opt-out registry** — an operator
inspects it, decides whether the URL is genuinely permanently
broken, and either re-queues it (delete the row) or accepts the
loss (leave it). The schedule keeps running either way.

## Don'ts

- Don't wrap `session.send(...)` in `try / except RequestException` —
  `ErrorNotifyingHTTPSession` already converts the exception into a
  synthetic `status_code=0` response.
- Don't `raise_on_failure=True` by default; the whole point of the
  notifying session is to keep the pipeline running on persistent
  failures. Flip to strict only when you genuinely want the schedule
  to stop.
- Don't abort the batch on one bad row. Cast lenient → split good /
  bad → quarantine the bad rows. Aborting hides the fact that the
  vendor's schema changed for one optional field.
- Don't pickle the `notifier` across Spark workers — it's marked
  transient on the session. Build the alert channel on the driver,
  or re-attach on the worker side.
- Don't write a parallel retry loop around the session — configure
  `retry=` once at session construction (see
  [`ygg-http`](ygg-http.md#retries-caching-rate-limits)).
- Don't email on every transient retry; the notifier only fires on
  **persistent** failures (post-retry-exhaustion). The
  `_TieredRetry` schedule is the noise filter — let it do its job.
- Don't put quarantine / dead-letter tables in a *different* schema
  from the source's `raw_<entity>`. Same schema → same ownership →
  one place for ops to look.
- Don't suppress notifier exceptions silently outside the session's
  own catch — if you compose notifiers, catch + log inside the
  composite so a flaky Slack webhook doesn't kill the email channel.
- Don't treat `status_code=0` as "success because no exception".
  Branch on it explicitly; it's the wire-level-failure marker.
