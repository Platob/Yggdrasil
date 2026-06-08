"""Benchmark the HTTP layer: :class:`PreparedRequest`, :class:`Response`,
and :class:`HTTPSession`.

What this covers
----------------

The in-process parts of an HTTP send — everything that runs before the
socket and after the body lands. We skip the wire entirely (no fake
server, no urllib3 round-trip) so the numbers measure yggdrasil's own
overhead rather than the OS network stack.

* :class:`PreparedRequest` — ``prepare`` (the canonical build path),
  ``copy`` (pagination), the cached identity surface (``hash``,
  ``public_hash``, ``private_url_hash``, ``public_url_hash``,
  ``partition_key``, ``body_hash``), and ``arrow_values`` /
  ``to_arrow_batch`` (logging / cache row build).
* :class:`Response` — construct from raw inputs, the same cached
  identity surface, ``arrow_values`` / ``to_arrow_batch`` (metadata
  projection), ``media_type``, and ``content`` / ``text``.
* :class:`HTTPSession` — singleton lookup (``HTTPSession(base_url=…)``),
  ``prepare_request_before_send`` (per-send hot path), and
  ``SendConfig.from_`` (every ``session.send`` builds one).

Usage::

    PYTHONPATH=src python tests/test_yggdrasil/test_http_/benchmarks/bench_http.py
    PYTHONPATH=src python tests/test_yggdrasil/test_http_/benchmarks/bench_http.py --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable

from yggdrasil.url import URL
from yggdrasil.http_.session import HTTPSession
from yggdrasil.path.memory import Memory
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response
from yggdrasil.http_.send_config import SendConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


HTTPS_STR = "https://api.example.com:8443/v1/accounts/12345/transactions?from=2024-01-01&to=2024-12-31&page=3"
URL_HTTPS = URL.from_str(HTTPS_STR)

HEADERS_SMALL = {
    "Content-Type": "application/json",
    "User-Agent": "yggdrasil/0.x",
    "Accept-Encoding": "gzip,deflate",
}
HEADERS_LARGE = {f"X-Header-{i:02d}": f"value-{i:02d}" for i in range(20)}

JSON_BODY = {"id": 42, "name": "alice", "filters": [1, 2, 3, 4, 5]}
RAW_BODY = b'{"id":42,"name":"alice"}'
MEDIUM_BODY = b"x" * 64_000

# Pre-built request fixtures — most identity / projection benchmarks
# hit these so we measure the cached fast path. A fresh-per-call
# variant lives next to each "cold" scenario.
REQ_NO_BODY = PreparedRequest.prepare("GET", HTTPS_STR, headers=dict(HEADERS_SMALL))
REQ_WITH_BODY = PreparedRequest.prepare(
    "POST", HTTPS_STR, headers=dict(HEADERS_SMALL), body=RAW_BODY,
)
# Warm the identity / projection cache so the "warm" runs measure the
# memoized fast path. Cold runs build a fresh request per iteration.
_ = REQ_NO_BODY.hash, REQ_NO_BODY.public_hash, REQ_NO_BODY.arrow_values
_ = REQ_WITH_BODY.hash, REQ_WITH_BODY.public_hash, REQ_WITH_BODY.arrow_values

RESP_NO_BODY = Response(
    request=REQ_NO_BODY,
    status_code=200,
    headers={"Content-Type": "application/json", "Content-Length": "0"},
    tags={},
    buffer=Memory(),
    received_at=dt.datetime.now(dt.timezone.utc),
)
RESP_WITH_BODY = Response(
    request=REQ_WITH_BODY,
    status_code=200,
    headers={"Content-Type": "application/json"},
    tags={},
    buffer=Memory(binary=RAW_BODY),
    received_at=dt.datetime.now(dt.timezone.utc),
)
_ = RESP_NO_BODY.hash, RESP_NO_BODY.public_hash, RESP_NO_BODY.arrow_values
_ = RESP_WITH_BODY.hash, RESP_WITH_BODY.public_hash, RESP_WITH_BODY.arrow_values

SESSION = HTTPSession(base_url="https://api.example.com")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 1000)):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
    }


def _fmt(r: dict) -> str:
    scale = 1e6
    unit = "us"
    if r["best"] < 1e-6:
        scale = 1e9
        unit = "ns"
    return (
        f"{r['label']:<60s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# PreparedRequest scenarios
# ---------------------------------------------------------------------------


def _request_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # Build paths — what every outbound request pays once.
    out.append(_time_one(
        "PreparedRequest.prepare GET (small headers)",
        lambda: PreparedRequest.prepare(
            "GET", HTTPS_STR, headers=dict(HEADERS_SMALL),
        ),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "PreparedRequest.prepare POST (json body)",
        lambda: PreparedRequest.prepare(
            "POST", HTTPS_STR, headers=dict(HEADERS_SMALL), json=JSON_BODY,
        ),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "PreparedRequest.prepare POST (raw bytes body)",
        lambda: PreparedRequest.prepare(
            "POST", HTTPS_STR, headers=dict(HEADERS_SMALL), body=RAW_BODY,
        ),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "PreparedRequest.prepare GET (large headers)",
        lambda: PreparedRequest.prepare(
            "GET", HTTPS_STR, headers=dict(HEADERS_LARGE),
        ),
        repeat=repeat, inner=2_000,
    ))

    # ``copy`` — fires per page on paginated fetches.
    out.append(_time_one(
        "request.copy(url=…) (pagination clone)",
        lambda: REQ_WITH_BODY.copy(url=URL_HTTPS),
        repeat=repeat, inner=10_000,
    ))

    # Cached identity surface — warm. ``arrow_values`` walk reaches every
    # column; downstream cache code does this on every send.
    out.append(_time_one(
        "request.hash (warm)",
        lambda: REQ_NO_BODY.hash,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "request.public_hash (warm)",
        lambda: REQ_NO_BODY.public_hash,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "request.private_url_hash (warm)",
        lambda: REQ_NO_BODY.private_url_hash,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "request.public_url_hash (warm)",
        lambda: REQ_NO_BODY.public_url_hash,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "request.partition_key (warm)",
        lambda: REQ_NO_BODY.partition_key,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "request.body_hash (warm, has body)",
        lambda: REQ_WITH_BODY.body_hash,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "request.arrow_values (warm)",
        lambda: REQ_NO_BODY.arrow_values,
        repeat=repeat, inner=500_000,
    ))

    # Cold identity — every property recomputed because the cache token
    # shifted. Approximates the first-touch cost for a fresh request.
    def _cold_hash():
        r = PreparedRequest.prepare("GET", HTTPS_STR, headers=dict(HEADERS_SMALL))
        return r.hash, r.public_hash
    out.append(_time_one(
        "request: prepare + hash + public_hash (cold)",
        _cold_hash,
        repeat=repeat, inner=2_000,
    ))

    def _cold_arrow_values():
        r = PreparedRequest.prepare("GET", HTTPS_STR, headers=dict(HEADERS_SMALL))
        return r.arrow_values
    out.append(_time_one(
        "request: prepare + arrow_values (cold)",
        _cold_arrow_values,
        repeat=repeat, inner=1_000,
    ))

    out.append(_time_one(
        "request.to_arrow_batch(parse=False)",
        lambda: REQ_NO_BODY.to_arrow_batch(parse=False),
        repeat=repeat, inner=5_000,
    ))

    # match_value / match_tuple — used by the cache lookup layer.
    keys = ("method", "public_hash", "public_url_hash", "partition_key")
    out.append(_time_one(
        "request.match_tuple(public keys)",
        lambda: REQ_NO_BODY.match_tuple(keys),
        repeat=repeat, inner=100_000,
    ))

    out.append(_time_one(
        "request.anonymize('remove')",
        lambda: REQ_NO_BODY.anonymize(mode="remove"),
        repeat=repeat, inner=5_000,
    ))

    return out


# ---------------------------------------------------------------------------
# Response scenarios
# ---------------------------------------------------------------------------


def _response_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    now = dt.datetime.now(dt.timezone.utc)
    out.append(_time_one(
        "Response(...) construct (no body)",
        lambda: Response(
            request=REQ_NO_BODY,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=Memory(),
            received_at=now,
        ),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "Response(...) construct (with body bytes)",
        lambda: Response(
            request=REQ_WITH_BODY,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=Memory(binary=RAW_BODY),
            received_at=now,
        ),
        repeat=repeat, inner=2_000,
    ))

    out.append(_time_one(
        "response.hash (warm)",
        lambda: RESP_NO_BODY.hash,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "response.public_hash (warm)",
        lambda: RESP_NO_BODY.public_hash,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "response.partition_key (warm)",
        lambda: RESP_NO_BODY.partition_key,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "response.arrow_values (warm)",
        lambda: RESP_NO_BODY.arrow_values,
        repeat=repeat, inner=500_000,
    ))

    def _cold_resp_hash():
        r = Response(
            request=REQ_NO_BODY,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=Memory(binary=RAW_BODY),
            received_at=now,
        )
        return r.hash, r.public_hash
    out.append(_time_one(
        "response: construct + hash + public_hash (cold)",
        _cold_resp_hash,
        repeat=repeat, inner=1_000,
    ))

    out.append(_time_one(
        "response.to_arrow_batch(parse=False)",
        lambda: RESP_NO_BODY.to_arrow_batch(parse=False),
        repeat=repeat, inner=2_000,
    ))

    out.append(_time_one(
        "response.media_type (warm)",
        lambda: RESP_WITH_BODY.media_type,
        repeat=repeat, inner=100_000,
    ))

    out.append(_time_one(
        "response.content (uncompressed)",
        lambda: RESP_WITH_BODY.content,
        repeat=repeat, inner=20_000,
    ))

    out.append(_time_one(
        "response.text (uncompressed)",
        lambda: RESP_WITH_BODY.text,
        repeat=repeat, inner=20_000,
    ))

    return out


# ---------------------------------------------------------------------------
# HTTPSession scenarios
# ---------------------------------------------------------------------------


def _session_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # --- Singleton lookup ----------------------------------------------
    # Most code calls ``HTTPSession(base_url=…)`` as if it were free.
    # The lookup is keyed on the config so re-entering the same kwargs
    # collapses to the cached instance; this measures that fast path.
    out.append(_time_one(
        "HTTPSession(base_url) (singleton hit)",
        lambda: HTTPSession(base_url="https://api.example.com"),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "HTTPSession(base_url, headers) (singleton hit)",
        lambda: HTTPSession(
            base_url="https://api.example.com",
            headers={"X-Tenant": "acme"},
        ),
        repeat=repeat, inner=20_000,
    ))

    # --- Send/SendMany config validation -------------------------------
    out.append(_time_one(
        "SendConfig.from_(None) (no-cache default)",
        lambda: SendConfig.from_(None, wait=None, raise_error=True, stream=True),
        repeat=repeat, inner=20_000,
    ))
    cached_cfg = SendConfig.from_(None)
    out.append(_time_one(
        "SendConfig.from_(SendConfig) (pass-through)",
        lambda: SendConfig.from_(cached_cfg),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "SendConfig.from_(None)",
        lambda: SendConfig.from_(None),
        repeat=repeat, inner=20_000,
    ))

    # --- prepare_request_before_send fan-out ---------------------------
    # Three shapes a session pays per send:
    #   * default no-header session — cheapest path, only sets sent_at;
    #   * session-level headers merge — every request inherits tenant /
    #     trace headers from the session;
    #   * session-level auth handler — every request re-resolves the
    #     handler's ``authorization`` and stamps the header.
    session_plain = HTTPSession(base_url="https://api.example.com/bench-plain")
    session_with_headers = HTTPSession(
        base_url="https://api.example.com",
        headers={"X-Tenant": "acme", "X-Trace-Id": "abc123"},
    )

    # Stand-in auth handler that returns a static token; the real MSAL /
    # OAuth handlers go through the same property surface, so this
    # measures the per-send overhead, not the token refresh.
    from yggdrasil.http_.authorization.base import Authorization

    class _StaticAuth(Authorization):
        scheme = "Bearer"
        token = "bench-token"

        @property
        def authorization(self) -> str:
            return f"Bearer {self.token}"

        def refresh(self, *, force: bool = False) -> bool:
            return False

    session_with_auth = HTTPSession(
        base_url="https://api.example.com/bench-auth",
        headers={"X-Tenant": "acme"},
        auth=_StaticAuth(),
    )

    def _prepare_plain():
        r = PreparedRequest.prepare("GET", HTTPS_STR, headers=dict(HEADERS_SMALL))
        return session_plain.prepare_request_before_send(r)
    out.append(_time_one(
        "session.prepare_request_before_send (no extras)",
        _prepare_plain,
        repeat=repeat, inner=2_000,
    ))

    def _prepare_headers():
        r = PreparedRequest.prepare("GET", HTTPS_STR, headers=dict(HEADERS_SMALL))
        return session_with_headers.prepare_request_before_send(r)
    out.append(_time_one(
        "session.prepare_request_before_send (session headers)",
        _prepare_headers,
        repeat=repeat, inner=2_000,
    ))

    def _prepare_auth():
        r = PreparedRequest.prepare("GET", HTTPS_STR, headers=dict(HEADERS_SMALL))
        return session_with_auth.prepare_request_before_send(r)
    out.append(_time_one(
        "session.prepare_request_before_send (auth handler)",
        _prepare_auth,
        repeat=repeat, inner=2_000,
    ))

    # --- Session.prepare_request (base_url + params merge) -------------
    out.append(_time_one(
        "session.prepare_request(relative)",
        lambda: session_plain.prepare_request("GET", "/v1/accounts/12345"),
        repeat=repeat, inner=2_000,
    ))
    params = {"from": "2024-01-01", "to": "2024-12-31", "page": "3"}
    out.append(_time_one(
        "session.prepare_request(relative, params=…)",
        lambda: session_plain.prepare_request(
            "GET", "/v1/accounts/12345", params=params,
        ),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "URL.join (base_url + relative)",
        lambda: URL.from_str("https://api.example.com").join("/v1/accounts/12345"),
        repeat=repeat, inner=10_000,
    ))

    # --- Full no-cache send dispatch through a fake transport ----------
    # ``_local_send`` is the only async-coupled step in ``_send``; stub
    # it so the bench measures the rest of the pipeline (config merge,
    # ``prepare_request_before_send``, cache short-circuits, post hooks)
    # without involving a socket.
    class _FakeHTTPSession(HTTPSession):
        def _local_send(self, request):  # type: ignore[override]
            resp = Response(
                request=request,
                status_code=200,
                headers={"Content-Type": "application/json", "Content-Length": "0"},
                tags={},
                buffer=Memory(),
                received_at=dt.datetime.now(dt.timezone.utc),
            )
            return resp

    fake_session = _FakeHTTPSession(
        base_url="https://api.example.com/bench-fake",
        headers={"X-Tenant": "acme", "X-Trace-Id": "abc123"},
    )
    # ``_send`` short-circuits on cache hits, but here both caches
    # default to disabled — every iteration walks the no-cache branch:
    # cache lookup → ``prepare_request_before_send`` → ``_local_send``
    # → ``prepare_response_after_received``.
    def _send_via_fake():
        r = PreparedRequest.prepare("GET", HTTPS_STR, headers=dict(HEADERS_SMALL))
        return fake_session.send(r, raise_error=False)
    out.append(_time_one(
        "session.send (fake transport, no cache)",
        _send_via_fake,
        repeat=repeat, inner=1_000,
    ))

    # Combined "build + check_arg" — closer to the steady-state cost
    # the caller sees per ``session.send(prepare(...))`` line.
    def _build_and_check():
        r = PreparedRequest.prepare("GET", HTTPS_STR, headers=dict(HEADERS_SMALL))
        cfg = SendConfig.from_(None, wait=None, raise_error=True, stream=True)
        SESSION.prepare_request_before_send(r)
        return r, cfg
    out.append(_time_one(
        "build + SendConfig.from_ + prepare_request",
        _build_and_check,
        repeat=repeat, inner=2_000,
    ))

    return out


# ---------------------------------------------------------------------------
# Arrow conversion scenarios (request / response, both directions)
# ---------------------------------------------------------------------------


def _arrow_scenarios(repeat: int) -> list[dict]:
    """Bench every Arrow-side conversion the cache pipeline depends on.

    Covers both directions for both shapes:

    * ``PreparedRequest.to_arrow_batch`` / ``.to_arrow_table`` — fires on
      every cache *write* (local fast-path and remote insert) and again
      on every ``_split_remote_cache`` lookup that has to project a
      request row for ``request_tuple`` matching.
    * ``Response.to_arrow_batch`` / ``.to_arrow_table`` — fires once per
      successful network fetch on the writeback path (``_persist_remote``,
      ``_store_cached_response``).
    * Bulk ``pa.Table.from_batches([r.to_arrow_batch(...) for r in N])``
      — the shape ``_persist_remote`` builds before handing off to
      ``Tabular.write_arrow_batches``. Measures the per-row build
      amortised over a 64-row batch, which is the relevant cost for
      "batched insert" vs. "row-at-a-time insert".
    * ``PreparedRequest.from_arrow`` / ``Response.from_arrow_tabular``
      / ``Response.from_records`` — fires on every cache *read* (remote
      lookup → :class:`Response` reconstruction, local fast-path read).
      Both single-row and 64-row batches.
    """
    import pyarrow as pa
    out: list[dict] = []

    # --- to_arrow: single row ---
    out.append(_time_one(
        "PreparedRequest.to_arrow_batch(parse=False)",
        lambda: REQ_NO_BODY.to_arrow_batch(parse=False),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "PreparedRequest.to_arrow_batch(parse=False) (with body)",
        lambda: REQ_WITH_BODY.to_arrow_batch(parse=False),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "PreparedRequest.to_arrow_table(parse=False)",
        lambda: REQ_NO_BODY.to_arrow_table(parse=False),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "Response.to_arrow_batch(parse=False) (no body)",
        lambda: RESP_NO_BODY.to_arrow_batch(parse=False),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "Response.to_arrow_batch(parse=False) (with body)",
        lambda: RESP_WITH_BODY.to_arrow_batch(parse=False),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "Response.to_arrow_table(parse=False)",
        lambda: RESP_NO_BODY.to_arrow_table(parse=False),
        repeat=repeat, inner=2_000,
    ))

    # --- to_arrow: bulk-batch shape (the _persist_remote build) ---
    BULK = 64
    bulk_reqs = [
        PreparedRequest.prepare(
            "GET",
            f"https://api.example.com/v1/r{i:05d}?page={i % 7}",
            headers=dict(HEADERS_SMALL),
        )
        for i in range(BULK)
    ]
    bulk_resps = [
        Response(
            request=r,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=Memory(binary=RAW_BODY),
            received_at=dt.datetime.now(dt.timezone.utc),
        )
        for r in bulk_reqs
    ]
    # Warm the identity / projection caches — the cold cost is
    # already covered above; bulk shape measures the conversion
    # itself, not the first-touch hash work.
    for r in bulk_reqs:
        _ = r.public_hash, r.arrow_values
    for r in bulk_resps:
        _ = r.public_hash, r.arrow_values

    out.append(_time_one(
        f"pa.Table.from_batches([req.to_arrow_batch] x {BULK}) (per-row build)",
        lambda: pa.Table.from_batches(
            [r.to_arrow_batch(parse=False) for r in bulk_reqs]
        ),
        repeat=repeat, inner=50,
    ))
    out.append(_time_one(
        f"PreparedRequest.values_to_arrow_batch ({BULK} rows, bulk)",
        lambda: PreparedRequest.values_to_arrow_batch(bulk_reqs),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"pa.Table.from_batches([resp.to_arrow_batch] x {BULK}) (per-row build)",
        lambda: pa.Table.from_batches(
            [r.to_arrow_batch(parse=False) for r in bulk_resps]
        ),
        repeat=repeat, inner=50,
    ))
    out.append(_time_one(
        f"Response.values_to_arrow_batch ({BULK} rows, bulk)",
        lambda: Response.values_to_arrow_batch(bulk_resps),
        repeat=repeat, inner=500,
    ))

    # --- from_arrow: single row + bulk-batch ---
    req_batch_1 = REQ_NO_BODY.to_arrow_batch(parse=False)
    resp_batch_1 = RESP_NO_BODY.to_arrow_batch(parse=False)
    req_table_n = pa.Table.from_batches(
        [r.to_arrow_batch(parse=False) for r in bulk_reqs]
    ).combine_chunks()
    resp_table_n = pa.Table.from_batches(
        [r.to_arrow_batch(parse=False) for r in bulk_resps]
    ).combine_chunks()

    out.append(_time_one(
        "PreparedRequest.from_arrow (1 row, batch)",
        lambda: list(PreparedRequest.from_arrow(req_batch_1, normalize=False)),
        repeat=repeat, inner=1_000,
    ))
    out.append(_time_one(
        f"PreparedRequest.from_arrow ({BULK} rows, table)",
        lambda: list(PreparedRequest.from_arrow(req_table_n, normalize=False)),
        repeat=repeat, inner=50,
    ))
    out.append(_time_one(
        "Response.from_arrow_tabular (1 row, batch)",
        lambda: list(Response.from_arrow_tabular(resp_batch_1, normalize=False)),
        repeat=repeat, inner=1_000,
    ))
    out.append(_time_one(
        f"Response.from_arrow_tabular ({BULK} rows, table)",
        lambda: list(Response.from_arrow_tabular(resp_table_n, normalize=False)),
        repeat=repeat, inner=20,
    ))

    # Records path used by `_send_many` to flatten HTTPResponseBatch holders.
    resp_records = list(
        Response.from_arrow_tabular(resp_table_n, normalize=False)
    )
    record_dicts = [r.arrow_values for r in resp_records]
    out.append(_time_one(
        f"Response.from_records ({BULK} mappings)",
        lambda: list(Response.from_records(record_dicts, normalize=False)),
        repeat=repeat, inner=50,
    ))

    # arrow_values walk — the underlying column source for to_arrow_batch.
    # Cold (no cache) so this measures the projection cost, not the
    # cached attribute read benched elsewhere.
    def _cold_req_arrow_values():
        r = PreparedRequest.prepare("GET", HTTPS_STR, headers=dict(HEADERS_SMALL))
        return r.arrow_values
    out.append(_time_one(
        "PreparedRequest.arrow_values (cold)",
        _cold_req_arrow_values,
        repeat=repeat, inner=500,
    ))
    def _cold_resp_arrow_values():
        r = Response(
            request=REQ_NO_BODY,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=Memory(binary=RAW_BODY),
            received_at=dt.datetime.now(dt.timezone.utc),
        )
        return r.arrow_values
    out.append(_time_one(
        "Response.arrow_values (cold)",
        _cold_resp_arrow_values,
        repeat=repeat, inner=500,
    ))

    return out


# ---------------------------------------------------------------------------
# Aggregator + CLI
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    return [
        *_request_scenarios(repeat),
        *_response_scenarios(repeat),
        *_session_scenarios(repeat),
        *_arrow_scenarios(repeat),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<60s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
