"""Benchmark the HTTP caching strategies in :mod:`yggdrasil.io`.

What this covers
----------------

The two caching layers that wrap :class:`yggdrasil.io.Session.send`
plus their config plumbing:

* :class:`CacheConfig` — ``from_`` (the polymorphic entry point
  every ``send`` call funnels through), the predicate properties used
  to gate the cache pipeline (``cache_enabled``, ``local_cache_enabled``,
  ``remote_cache_enabled``, ``request_by_is_public``, ``match_by``,
  ``match_by_columns``), and the :class:`Predicate` builders that
  drive *both* backends through :meth:`Tabular.read_arrow_batches`
  (``make_lookup_predicate``, ``make_batch_lookup_predicate``).
* :class:`FolderPath` — partition-aware write
  (``write_arrow_batches`` with a batch whose schema carries
  ``partition_by`` tags), partition-aware read (predicate-pushed,
  sidecar-driven), the URL-driven ``static_values`` parse, and the
  ``.ygg/schema.arrow`` collect / persist round trip.
* :class:`Session` integration — :meth:`Session._load_cached_response`
  (``source="local"``) on a miss (the common case for cold cache)
  and on a hit (warm re-fetch), plus
  :meth:`Session._store_cached_response` (``source="local"``,
  fire-and-forget queueing of the actual write).

Skips out-of-process work (real Databricks SQL, real disk on hot
read benches where we use a pre-stored fixture, no network) so the
numbers measure yggdrasil's own caching overhead, not the FS or
remote service.

Usage::

    PYTHONPATH=src python tests/test_yggdrasil/test_http_/benchmarks/bench_http_cache.py
    PYTHONPATH=src python tests/test_yggdrasil/test_http_/benchmarks/bench_http_cache.py --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable

from yggdrasil.io import URL
from yggdrasil.http_.session import HTTPSession
from yggdrasil.io.memory import Memory
from yggdrasil.io.nested.folder_path import FolderPath, FolderOptions
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response
from yggdrasil.io.send_config import CacheConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


HTTPS_STR = "https://api.example.com:8443/v1/accounts/12345/transactions?from=2024-01-01&to=2024-12-31&page=3"
URL_HTTPS = URL.from_str(HTTPS_STR)

REQ = PreparedRequest.prepare(
    "GET", HTTPS_STR, headers={"Content-Type": "application/json"},
)
# Warm caches so identity is paid once outside the bench.
_ = REQ.public_hash, REQ.public_url_hash, REQ.partition_key

# Build a batch of requests with distinct URLs — make_batch_lookup_predicate
# folds them into one SQL clause; per-request work scales linearly.
BATCH_SIZE = 64
REQ_BATCH = [
    PreparedRequest.prepare(
        "GET",
        f"https://api.example.com/v1/accounts/{i:05d}/transactions?page={i % 7}",
        headers={"Content-Type": "application/json"},
    )
    for i in range(BATCH_SIZE)
]
for r in REQ_BATCH:
    _ = r.public_hash, r.partition_key

RESP = Response(
    request=REQ,
    status_code=200,
    headers={"Content-Type": "application/json"},
    tags={},
    buffer=Memory(binary=b'{"ok":true,"rows":[1,2,3]}'),
    received_at=dt.datetime.now(dt.timezone.utc),
)
_ = RESP.hash, RESP.public_hash, RESP.partition_key

# Three CacheConfig shapes worth measuring separately:
# - default (no caching) — the predicate gates short-circuit;
# - local-only — path is set, drives the disk fast-path branch;
# - remote-only with request_by — drives the SQL builders and
#   request_by_is_public branch.
CFG_DEFAULT = CacheConfig()
CFG_LOCAL = CacheConfig(tabular=tempfile.mkdtemp(prefix="ygg-bench-cache-"))
CFG_REMOTE_BY_PUBLIC = CacheConfig(
    request_by=["public_hash", "public_url_hash"],
    received_ttl=dt.timedelta(days=1),
)
CFG_REMOTE_PRIVATE = CacheConfig(
    request_by=["method", "private_url_hash"],
)


# A session is the bridge between CacheConfig and the fast-path
# helpers — singleton-cached so repeated construction is free.
SESSION = HTTPSession(base_url="https://api.example.com/bench-cache")


# Pre-store one response into the local cache so the
# ``_load_cached_response`` (local) hit path has something to read.
_LOCAL_TABULAR = CFG_LOCAL.cache_tabular()
_LOCAL_TABULAR.write_arrow_batches(
    (RESP.to_arrow_batch(parse=False),),
    options=FolderOptions(mode=CFG_LOCAL.mode),
)


def _cleanup_tmp() -> None:
    if CFG_LOCAL.is_local:
        shutil.rmtree(str(CFG_LOCAL.tabular.path), ignore_errors=True)


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
# CacheConfig.from_ dispatch scenarios
# ---------------------------------------------------------------------------


def _check_arg_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "CacheConfig.from_(None)",
        lambda: CacheConfig.from_(None),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "CacheConfig.from_(existing CacheConfig)",
        lambda: CacheConfig.from_(CFG_LOCAL),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "CacheConfig.from_(path str)",
        lambda: CacheConfig.from_("/tmp/ygg-bench"),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "CacheConfig.from_(Path)",
        lambda: CacheConfig.from_(Path("/tmp/ygg-bench")),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "CacheConfig.from_(timedelta=1d)",
        lambda: CacheConfig.from_(dt.timedelta(days=1)),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "CacheConfig.from_(mapping)",
        lambda: CacheConfig.from_({
            "path": "/tmp/ygg-bench",
            "mode": "APPEND",
            "request_by": ["public_hash"],
        }),
        repeat=repeat, inner=10_000,
    ))

    return out


# ---------------------------------------------------------------------------
# CacheConfig predicate / property scenarios
# ---------------------------------------------------------------------------


def _predicate_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "CacheConfig.cache_enabled (default)",
        lambda: CFG_DEFAULT.cache_enabled,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "CacheConfig.local_cache_enabled (local)",
        lambda: CFG_LOCAL.local_cache_enabled,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "CacheConfig.remote_cache_enabled (no tabular)",
        lambda: CFG_REMOTE_BY_PUBLIC.remote_cache_enabled,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "CacheConfig.match_by (request+response)",
        lambda: CFG_REMOTE_BY_PUBLIC.match_by,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "CacheConfig.match_by_columns",
        lambda: CFG_REMOTE_BY_PUBLIC.match_by_columns,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "CacheConfig.request_by_is_public (true)",
        lambda: CFG_REMOTE_BY_PUBLIC.request_by_is_public,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "CacheConfig.request_by_is_public (false)",
        lambda: CFG_REMOTE_PRIVATE.request_by_is_public,
        repeat=repeat, inner=200_000,
    ))

    return out


# ---------------------------------------------------------------------------
# CacheConfig request/response matching scenarios
# ---------------------------------------------------------------------------


def _matching_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "CacheConfig.request_values(request)",
        lambda: CFG_REMOTE_BY_PUBLIC.request_values(REQ),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "CacheConfig.request_tuple(request)",
        lambda: CFG_REMOTE_BY_PUBLIC.request_tuple(REQ),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "CacheConfig.filter_response(response, request)",
        lambda: CFG_REMOTE_BY_PUBLIC.filter_response(RESP, request=REQ),
        repeat=repeat, inner=50_000,
    ))

    return out


# ---------------------------------------------------------------------------
# Predicate builders — the single lookup surface for both backends
# ---------------------------------------------------------------------------


def _predicate_builder_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "CacheConfig.make_lookup_predicate(single request)",
        lambda: CFG_REMOTE_BY_PUBLIC.make_lookup_predicate(request=REQ),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        f"CacheConfig.make_batch_lookup_predicate({BATCH_SIZE} requests)",
        lambda: CFG_REMOTE_BY_PUBLIC.make_batch_lookup_predicate(requests=REQ_BATCH),
        repeat=repeat, inner=1_000,
    ))

    return out


# ---------------------------------------------------------------------------
# Local partitioned-folder cache primitives
# ---------------------------------------------------------------------------


def _local_folder_cache_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # FolderPath surface: URL-driven static_values, schema-driven
    # partition_columns, sidecar collect/persist round trip.
    partition_folder = FolderPath(
        path=Path(str(CFG_LOCAL.tabular.path)) / f"partition_key={REQ.partition_key}",
    )
    out.append(_time_one(
        "FolderPath.static_values (Hive partition leaf)",
        lambda: dict(partition_folder.static_values),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "FolderPath.partition_columns (root, sidecar hit)",
        lambda: _LOCAL_TABULAR.partition_columns(),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "FolderPath.collect_schema (sidecar hit, cached)",
        lambda: _LOCAL_TABULAR.collect_schema(),
        repeat=repeat, inner=200_000,
    ))

    # On-disk round trip — partition-aware write + predicate read.
    # Separate temp dirs per scenario so the write bench's accumulated
    # part files (one ``part-*.<ext>`` per iteration under
    # ``partition_key=<v>/``) don't poison the read bench's iter_children
    # walk, which would otherwise scan every prior-iteration leaf.
    batch = RESP.to_arrow_batch(parse=False)
    write_opts = FolderOptions(mode=CFG_LOCAL.mode)
    with tempfile.TemporaryDirectory(prefix="ygg-bench-folder-write-") as wtmp:
        write_tab = FolderPath(path=wtmp)
        # Prime the in-memory schema cache so the bench measures the
        # steady-state hot path (sidecar already persisted, in-memory
        # cache short-circuits the sidecar rewrite via the
        # ``prior == schema`` check in ``_persist_schema``).
        write_tab.write_arrow_batches((batch,), options=write_opts)
        out.append(_time_one(
            "FolderPath.write_arrow_batches (partitioned, schema unchanged)",
            lambda: write_tab.write_arrow_batches((batch,), options=write_opts),
            repeat=repeat, inner=200,
        ))

    with tempfile.TemporaryDirectory(prefix="ygg-bench-folder-read-") as rtmp:
        read_tab = FolderPath(path=rtmp)
        # Seed exactly one part file under one partition so the read
        # measures the partition-prune + predicate filter cost, not a
        # 500-leaf iterdir scan.
        read_tab.write_arrow_batches((batch,), options=write_opts)
        predicate = CacheConfig().make_lookup_predicate(request=REQ)
        read_opts = FolderOptions(predicate=predicate)
        out.append(_time_one(
            "FolderPath.read_arrow_batches (predicate hit)",
            lambda: list(read_tab.read_arrow_batches(options=read_opts)),
            repeat=repeat, inner=500,
        ))

    return out


# ---------------------------------------------------------------------------
# Session-level integration
# ---------------------------------------------------------------------------


def _session_cache_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "Session._load_cached_response (local hit)",
        lambda: SESSION._load_cached_response(REQ, CFG_LOCAL, source="local"),
        repeat=repeat, inner=500,
    ))

    miss_req = PreparedRequest.prepare(
        "GET",
        "https://api.example.com/v1/missing?cold=1",
        headers={"Content-Type": "application/json"},
    )
    _ = miss_req.public_hash
    out.append(_time_one(
        "Session._load_cached_response (local miss)",
        lambda: SESSION._load_cached_response(miss_req, CFG_LOCAL, source="local"),
        repeat=repeat, inner=2_000,
    ))

    # Use a dedicated folder so the fire-and-forget writeback queue
    # doesn't accumulate part files in the shared CFG_LOCAL — every
    # subsequent inner iteration would otherwise pay an iterdir over
    # the running total of written parts.
    store_cfg = CacheConfig(tabular=tempfile.mkdtemp(prefix="ygg-bench-store-"))
    store_tabular = store_cfg.cache_tabular()
    out.append(_time_one(
        "Session._store_cached_response (local, fire-and-forget)",
        lambda: SESSION._store_cached_response(
            RESP, store_cfg, source="local", tabular=store_tabular,
        ),
        repeat=repeat, inner=500,
    ))

    return out


# ---------------------------------------------------------------------------
# send_many_batches: per-chunk cache machinery
# ---------------------------------------------------------------------------


def _send_many_cache_scenarios(repeat: int) -> list[dict]:
    """Hot loops inside ``_send_many_batches`` that scale with the chunk size.

    Targets the per-chunk machinery that wraps the actual local/remote
    cache calls:

    * the ``key_to_remote_cfg`` / ``key_to_local_cfg`` dict-builds that
      key per-request effective configs (one entry per request);
    * the per-response key lookup pattern used by ``_persist_remote``,
      ``_mirror_local_hits_to_remote`` and ``_backfill_local_cache`` to
      resolve a response back to its originating per-request config;
    * ``Session._split_local_cache`` — the stage-1 scan that decides
      which requests can short-circuit to the on-disk fast-path.

    Numbers reported are *per chunk* (not per request), so a 1024-batch
    figure of 30 ms means the dict-build is paying ~30 us / request.
    """
    out: list[dict] = []

    # Pre-warm the public_url_hash / public_hash caches so the
    # optimized key path measures the warm fast-path. The status-quo
    # ``anonymize().url`` path doesn't have a warm cache — every call
    # rebuilds.
    for r in REQ_BATCH:
        _ = r.public_url_hash, r.public_hash, r.partition_key

    # The current ``_send_many_batches`` shape: one full anonymize per
    # request, twice over (once for remote, once for local). This is
    # the dict-build cost stage 4 / mirror / backfill all queue behind.
    def _build_url_keyed_maps():
        url_to_remote = {
            str(r.anonymize(mode="remove").url): CFG_REMOTE_BY_PUBLIC
            for r in REQ_BATCH
        }
        url_to_local = {
            str(r.anonymize(mode="remove").url): CFG_LOCAL
            for r in REQ_BATCH
        }
        return url_to_remote, url_to_local
    out.append(_time_one(
        f"_send_many_batches: anonymize-url dict-build ({BATCH_SIZE} req)",
        _build_url_keyed_maps,
        repeat=repeat, inner=20,
    ))

    # The optimized shape: a single pass over the chunk, keyed by
    # ``request.public_url_hash`` (already cached on the request) so
    # the per-request anonymize cost collapses to one cached attribute
    # read per request.
    def _build_hash_keyed_maps():
        url_to_remote: dict[int, CacheConfig] = {}
        url_to_local: dict[int, CacheConfig] = {}
        for r in REQ_BATCH:
            k = r.public_url_hash
            url_to_remote[k] = CFG_REMOTE_BY_PUBLIC
            url_to_local[k] = CFG_LOCAL
        return url_to_remote, url_to_local
    out.append(_time_one(
        f"_send_many_batches: public_url_hash dict-build ({BATCH_SIZE} req)",
        _build_hash_keyed_maps,
        repeat=repeat, inner=200,
    ))

    # Per-response lookup pattern used by ``_persist_remote`` /
    # ``_mirror_local_hits_to_remote``. Today this re-anonymizes the
    # response's request once per row — the optimized path just reads
    # ``response.request.public_url_hash``.
    resp_batch = [
        Response(
            request=r,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=Memory(binary=b'{"ok":true}'),
            received_at=dt.datetime.now(dt.timezone.utc),
        )
        for r in REQ_BATCH
    ]

    url_keyed = {
        str(r.anonymize(mode="remove").url): CFG_REMOTE_BY_PUBLIC
        for r in REQ_BATCH
    }
    hash_keyed = {r.public_url_hash: CFG_REMOTE_BY_PUBLIC for r in REQ_BATCH}

    def _lookup_via_anonymize_url():
        for resp in resp_batch:
            url_key = str(resp.request.anonymize(mode="remove").url)
            _ = url_keyed.get(url_key)
    out.append(_time_one(
        f"_persist_remote: anonymize-url per-response lookup ({BATCH_SIZE} resp)",
        _lookup_via_anonymize_url,
        repeat=repeat, inner=20,
    ))

    def _lookup_via_public_url_hash():
        for resp in resp_batch:
            _ = hash_keyed.get(resp.request.public_url_hash)
    out.append(_time_one(
        f"_persist_remote: public_url_hash per-response lookup ({BATCH_SIZE} resp)",
        _lookup_via_public_url_hash,
        repeat=repeat, inner=2_000,
    ))

    # Stage-1 cache scan over a misses-only chunk so the on-disk read
    # path doesn't dominate — the bench targets the per-request
    # dispatch, not the IPC decode (covered by
    # ``_read_fast_path_arrow_batch`` above).
    cold_batch = [
        PreparedRequest.prepare(
            "GET",
            f"https://api.example.com/v1/cold/{i:05d}?p={i % 9}",
            headers={"Content-Type": "application/json"},
        )
        for i in range(BATCH_SIZE)
    ]
    for r in cold_batch:
        _ = r.public_hash, r.public_url_hash
    out.append(_time_one(
        f"Session._split_local_cache (all miss, {BATCH_SIZE} req)",
        lambda: SESSION._split_local_cache(cold_batch, CFG_LOCAL),
        repeat=repeat, inner=200,
    ))

    # Same scan when no local cache is active anywhere in the batch —
    # the early-exit predicate matters because hot Spark workers /
    # cache-disabled callers walk this path on every chunk.
    out.append(_time_one(
        f"Session._split_local_cache (cache off, {BATCH_SIZE} req)",
        lambda: SESSION._split_local_cache(cold_batch, CFG_DEFAULT),
        repeat=repeat, inner=2_000,
    ))

    return out


# ---------------------------------------------------------------------------
# End-to-end Session.send / Session.send_many with local cache
# ---------------------------------------------------------------------------


class _StubBenchSession(HTTPSession):
    """``HTTPSession`` double for end-to-end bench timing.

    Returns a canned :class:`Response` from :meth:`_local_send` so the
    wire transport drops out of the measurement and the cache pipeline
    is what's being timed. Mirrors ``tests._helpers.StubSession`` but
    skips the per-call recording (counter cost would noise the bench).
    """

    def __init__(self, *args, response: Response | None = None, **kwargs) -> None:
        if getattr(self, "_initialized", False):
            if response is not None:
                self._response = response
            return
        super().__init__(*args, **kwargs)
        self._response = response

    def _local_send(self, request: PreparedRequest) -> Response:
        # Hand back a fresh shallow copy so the caller can stamp
        # ``response.request`` without mutating the bench template
        # across iterations.
        return Response(
            request=request,
            status_code=self._response.status_code,
            headers=dict(self._response.headers),
            tags={},
            buffer=Memory(binary=b'{"ok":true}'),
            received_at=dt.datetime.now(dt.timezone.utc),
        )


def _e2e_send_scenarios(repeat: int) -> list[dict]:
    """End-to-end ``Session.send`` timing for the cache hot paths.

    Measures the **full** ``send`` pipeline (effective-cfg resolve →
    local lookup → optional remote lookup → optional ``_local_send``
    → writeback dispatch), not the isolated ``_load_cached_response``
    call covered above. Numbers here include every Python-side cost a
    real caller pays per request — the only thing skipped is the
    actual wire transport.
    """
    out: list[dict] = []

    bench_url = "https://api.example.com/bench-send/x"
    bench_req = PreparedRequest.prepare(
        "GET", bench_url, headers={"Content-Type": "application/json"},
    )
    _ = bench_req.public_url_hash, bench_req.public_hash, bench_req.partition_key

    bench_resp = Response(
        request=bench_req, status_code=200,
        headers={"Content-Type": "application/json"}, tags={},
        buffer=Memory(binary=b'{"ok":true}'),
        received_at=dt.datetime.now(dt.timezone.utc),
    )

    # Baseline: no cache configured. Pure ``send`` dispatch cost.
    cold_root = tempfile.mkdtemp(prefix="ygg-bench-e2e-cold-")
    cold_cache = CacheConfig(tabular=cold_root)
    stub = _StubBenchSession(
        base_url="https://api.example.com/bench-send",
        response=bench_resp,
    )
    out.append(_time_one(
        "Session.send (no cache, stub wire)",
        lambda: stub.send(bench_req, raise_error=False),
        repeat=repeat, inner=500,
    ))

    # Pre-seed one row matching ``bench_req`` so every ``send`` call
    # in the inner loop short-circuits on the local cache.
    hit_root = tempfile.mkdtemp(prefix="ygg-bench-e2e-hit-")
    hit_cache = CacheConfig(tabular=hit_root)
    hit_cache.cache_tabular().write_arrow_batches(
        (bench_resp.to_arrow_batch(parse=False),),
        options=FolderOptions(mode=hit_cache.mode),
    )
    out.append(_time_one(
        "Session.send (local cache HIT, stub wire)",
        lambda: stub.send(bench_req, local_cache=hit_cache, raise_error=False),
        repeat=repeat, inner=200,
    ))

    # All-miss: distinct URL per call so neither the local cache nor
    # the in-memory dedup catches it. Includes the fire-and-forget
    # writeback to a temp folder; the writeback queue is drained by
    # the job pool, so we're measuring the dispatch cost not the
    # actual disk write.
    miss_root = tempfile.mkdtemp(prefix="ygg-bench-e2e-miss-")
    miss_cache = CacheConfig(tabular=miss_root)
    miss_counter = [0]

    def _miss_send() -> None:
        i = miss_counter[0]
        miss_counter[0] = i + 1
        r = PreparedRequest.prepare(
            "GET",
            f"https://api.example.com/bench-miss/{i:09d}",
            headers={"Content-Type": "application/json"},
        )
        stub.send(r, local_cache=miss_cache, raise_error=False)

    out.append(_time_one(
        "Session.send (local cache MISS + writeback, stub wire)",
        _miss_send,
        repeat=repeat, inner=200,
    ))

    return out


def _e2e_send_many_scenarios(repeat: int) -> list[dict]:
    """End-to-end ``Session.send_many`` timing for batch cache paths.

    Three shapes, all with ``BATCH_SIZE`` requests per call:

    * **all-hit** — every row pre-seeded; pipeline short-circuits on
      stage 1 and never enters stage 3. The tight loop in the
      ``not after_local`` branch is what's timed.
    * **all-miss** — none seeded; stage 1 finds nothing, every row
      drops to the stub wire + fire-and-forget writeback.
    * **mixed (50/50)** — half seeded, half fresh; exercises both
      branches plus the merge of hits + new responses in
      :class:`HTTPResponseBatch`.
    """
    out: list[dict] = []

    bench_resp_template = Response(
        request=REQ, status_code=200,
        headers={"Content-Type": "application/json"}, tags={},
        buffer=Memory(binary=b'{"ok":true}'),
        received_at=dt.datetime.now(dt.timezone.utc),
    )
    stub = _StubBenchSession(
        base_url="https://api.example.com/bench-send-many",
        response=bench_resp_template,
    )

    def _make_reqs(prefix: str, n: int = BATCH_SIZE) -> list[PreparedRequest]:
        reqs = [
            PreparedRequest.prepare(
                "GET",
                f"https://api.example.com/{prefix}/{i:06d}?p={i % 7}",
                headers={"Content-Type": "application/json"},
            )
            for i in range(n)
        ]
        for r in reqs:
            _ = r.public_url_hash, r.public_hash, r.partition_key
        return reqs

    def _seed(cache: CacheConfig, reqs: list[PreparedRequest]) -> None:
        tab = cache.cache_tabular()
        batches = []
        for i, r in enumerate(reqs):
            resp = Response(
                request=r, status_code=200,
                headers={"Content-Type": "application/json"}, tags={},
                buffer=Memory(binary=f'{{"i":{i}}}'.encode()),
                received_at=dt.datetime.now(dt.timezone.utc),
            )
            batches.append(resp.to_arrow_batch(parse=False))
        tab.write_arrow_batches(batches, options=FolderOptions(mode=cache.mode))

    # all-hit
    hit_root = tempfile.mkdtemp(prefix="ygg-bench-many-hit-")
    hit_cache = CacheConfig(tabular=hit_root)
    hit_reqs = _make_reqs("many-hit")
    _seed(hit_cache, hit_reqs)
    out.append(_time_one(
        f"Session.send_many (local cache ALL HIT, {BATCH_SIZE} req)",
        lambda: list(stub.send_many(iter(hit_reqs), local_cache=hit_cache)),
        repeat=repeat, inner=50,
    ))

    # all-miss — every call uses fresh URLs so the cache never catches.
    miss_root = tempfile.mkdtemp(prefix="ygg-bench-many-miss-")
    miss_cache = CacheConfig(tabular=miss_root)
    miss_counter = [0]

    def _miss_batch():
        base = miss_counter[0]
        miss_counter[0] = base + BATCH_SIZE
        reqs = [
            PreparedRequest.prepare(
                "GET",
                f"https://api.example.com/many-miss/{base + i:09d}",
                headers={"Content-Type": "application/json"},
            )
            for i in range(BATCH_SIZE)
        ]
        list(stub.send_many(iter(reqs), local_cache=miss_cache))

    out.append(_time_one(
        f"Session.send_many (local cache ALL MISS + writeback, {BATCH_SIZE} req)",
        _miss_batch,
        repeat=repeat, inner=20,
    ))

    # mixed: half seeded, half fresh. Reuse seeded reqs across iterations
    # but supply fresh misses each call so the misses don't write into
    # the seeded folder and degrade the "half hit" balance.
    mixed_root = tempfile.mkdtemp(prefix="ygg-bench-many-mixed-")
    mixed_cache = CacheConfig(tabular=mixed_root)
    half = BATCH_SIZE // 2
    seeded_reqs = _make_reqs("many-mixed-hit", n=half)
    _seed(mixed_cache, seeded_reqs)
    mixed_counter = [0]

    def _mixed_batch():
        base = mixed_counter[0]
        mixed_counter[0] = base + half
        fresh = [
            PreparedRequest.prepare(
                "GET",
                f"https://api.example.com/many-mixed-fresh/{base + i:09d}",
                headers={"Content-Type": "application/json"},
            )
            for i in range(half)
        ]
        # Interleave so the implementation can't stripe the work by
        # input position.
        chunk: list[PreparedRequest] = []
        for i in range(half):
            chunk.append(seeded_reqs[i])
            chunk.append(fresh[i])
        list(stub.send_many(iter(chunk), local_cache=mixed_cache))

    out.append(_time_one(
        f"Session.send_many (local cache 50% HIT 50% MISS, {BATCH_SIZE} req)",
        _mixed_batch,
        repeat=repeat, inner=20,
    ))

    return out


# ---------------------------------------------------------------------------
# Aggregator + CLI
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    return [
        *_check_arg_scenarios(repeat),
        *_predicate_scenarios(repeat),
        *_matching_scenarios(repeat),
        *_predicate_builder_scenarios(repeat),
        *_local_folder_cache_scenarios(repeat),
        *_session_cache_scenarios(repeat),
        *_send_many_cache_scenarios(repeat),
        *_e2e_send_scenarios(repeat),
        *_e2e_send_many_scenarios(repeat),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<60s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    try:
        for row in scenarios(args.repeat):
            print(_fmt(row))
    finally:
        _cleanup_tmp()


if __name__ == "__main__":
    main()
