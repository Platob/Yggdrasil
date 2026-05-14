"""Benchmark :class:`yggdrasil.databricks.schema.SchemaSession`.

What this covers
----------------

The in-process hot paths SchemaSession adds on top of
:class:`HTTPSession` — none of these scenarios talk to Databricks.
They measure the cost of the per-request remote-cache injection that
sits in front of :meth:`Session._send`:

* :meth:`SchemaSession.path_to_table_name` — the URL path → safe
  identifier conversion every request runs through. Hot loop on the
  send pipeline.
* :meth:`SchemaSession.table_for` — the per-path :class:`Table` cache
  miss (first call for a new path) vs. hit (the :class:`ExpiringDict`
  fast path).
* :meth:`SchemaSession.cache_config_for` — building a fresh
  :class:`CacheConfig` from the resolved table.
* :meth:`SchemaSession._attach_cache` — the end-to-end "stamp the
  per-path config onto a :class:`PreparedRequest`" cost, which is the
  one piece of work every outbound request pays for the cache layer
  even when the caller doesn't otherwise touch the Databricks stack.

Stub :class:`Tabular` and :class:`Schema` are used so the bench
measures SchemaSession's own overhead, not the SDK. CacheConfig's
``_is_tabular_io`` duck-test accepts anything with
``read_arrow_batches`` + ``write_arrow_batches``, so the stub flows
through the real validation path.

Usage::

    PYTHONPATH=src python benchmarks/databricks/bench_schema_session.py
    PYTHONPATH=src python benchmarks/databricks/bench_schema_session.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

from yggdrasil.data.enums import Mode
from yggdrasil.databricks.schema.session import SchemaSession
from yggdrasil.io.http_.session import HTTPSession
from yggdrasil.io.request import PreparedRequest


# ---------------------------------------------------------------------------
# Stubs — stand in for Schema / Table so we measure SchemaSession overhead
# only, not the Databricks SDK round-trip.
# ---------------------------------------------------------------------------


class _StubTabular:
    """Tabular-shaped sentinel; matches ``_is_tabular_io`` duck-test."""

    def __init__(self, name: str) -> None:
        self.name = name

    def read_arrow_batches(self, *a, **kw):  # pragma: no cover - bench stub
        raise NotImplementedError

    def write_arrow_batches(self, *a, **kw):  # pragma: no cover - bench stub
        raise NotImplementedError

    def full_name(self, safe=None) -> str:  # pragma: no cover - bench stub
        return f"main.bench.{self.name}"


class _StubSchema:
    def __init__(self) -> None:
        self.calls = 0
        self.full_name_str = "main.bench"

    def full_name(self, safe=None) -> str:
        return self.full_name_str

    def table(self, name: str) -> _StubTabular:
        # Count to confirm the per-path cache is actually short-circuiting
        # in the hot scenarios — if the bench accidentally rebuilt a
        # fresh SchemaSession every iteration the counter would explode.
        self.calls += 1
        return _StubTabular(name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SCHEMA = _StubSchema()

SESSION_APPEND = SchemaSession(
    SCHEMA,
    base_url="https://api.example.com",
    key="bench-append",
    mode=Mode.APPEND,
)
SESSION_UPSERT = SchemaSession(
    SCHEMA,
    base_url="https://api.example.com",
    key="bench-upsert",
    mode=Mode.UPSERT,
)
# Remote-only variant: local cache disabled so ``_attach_cache``
# pays only the remote-side cost. Used to isolate the local-cache
# attachment overhead in the comparison below.
SESSION_REMOTE_ONLY = SchemaSession(
    SCHEMA,
    base_url="https://api.example.com",
    key="bench-remote-only",
    mode=Mode.APPEND,
    local_cache=False,
)
# Plain HTTPSession baseline — same base_url with a different ``key``
# so the singleton cache keeps it distinct from the SchemaSession
# instances sharing the same URL.
BASE_SESSION = HTTPSession(base_url="https://api.example.com", key="bench-base")


PATHS = [
    "/v1/accounts/12345/transactions",
    "/v1/accounts/12345",
    "/v1/users/42/orders",
    "/v2/products/super-long-name-here/details?page=3",
    "/",
    "/api/v1.2/data.json",
    "/" + "x/" * 200,  # exercises safe_table_name truncate+hash
]

REQS = [
    PreparedRequest.prepare("GET", f"https://api.example.com{p}",
                            headers={"Content-Type": "application/json"})
    for p in PATHS
]
for _r in REQS:
    _ = _r.public_hash, _r.public_url_hash  # warm caches once

# Single request reused across hot scenarios where one path is enough.
REQ_ONE = REQS[0]


# Warm the per-path table cache on the APPEND session so the "hit"
# scenarios actually exercise the fast path. The UPSERT session is
# left cold for the "cold-build" scenarios.
for _r in REQS:
    _ = SESSION_APPEND.table_for(_r)


# ---------------------------------------------------------------------------
# Timing helpers (mirrors bench_http_cache.py)
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], object], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 1_000)):
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
    scale, unit = (1e9, "ns") if r["best"] < 1e-6 else (1e6, "us")
    return (
        f"{r['label']:<60s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _name_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    sess = SESSION_APPEND

    # Sanity sweep: the derived name must always fit Unity Catalog's
    # identifier limit, regardless of input shape. Asserting at bench
    # entry catches a future regression in ``safe_table_name``'s
    # split-and-hash logic before the bench numbers can mask it.
    from yggdrasil.databricks.sql.sql_utils import MAX_TABLE_NAME_LEN
    _shapes = [
        "/",
        "/v1/accounts/12345/transactions",
        "/" + "x/" * 200,                                   # 400 chars, 200 tokens
        "/" + "/".join(f"part-{i:03d}" for i in range(40)),  # 40 named tokens
        "/" + "a" * 500,                                    # one giant token
    ]
    for _shape in _shapes:
        _name = sess.path_to_table_name(_shape)
        assert _name and len(_name) <= MAX_TABLE_NAME_LEN, (
            f"length-cap regression: path_to_table_name({_shape!r}) "
            f"→ {_name!r} (len={len(_name) if _name else 0})"
        )

    out.append(_time_one(
        "path_to_table_name (plain)",
        lambda: sess.path_to_table_name("/v1/accounts/12345/transactions"),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "path_to_table_name (empty)",
        lambda: sess.path_to_table_name("/"),
        repeat=repeat, inner=200_000,
    ))
    long_path = "/" + "x/" * 200
    out.append(_time_one(
        "path_to_table_name (overflow → truncate+hash)",
        lambda: sess.path_to_table_name(long_path),
        repeat=repeat, inner=50_000,
    ))
    one_giant_token = "/" + "a" * 500
    out.append(_time_one(
        "path_to_table_name (single overlong token → digest fallback)",
        lambda: sess.path_to_table_name(one_giant_token),
        repeat=repeat, inner=50_000,
    ))
    return out


def _table_for_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    sess_hot = SESSION_APPEND  # cache warmed at module load
    out.append(_time_one(
        "table_for (cache hit)",
        lambda: sess_hot.table_for(REQ_ONE),
        repeat=repeat, inner=200_000,
    ))

    # Cold path: bypass the cache so we measure the schema lookup +
    # ExpiringDict insert per call. The stub schema's ``table`` is
    # essentially a constructor, so this isolates SchemaSession's
    # per-miss overhead (re.sub + safe_table_name + get_or_set miss
    # branch).
    def _cold():
        sess_hot._table_cache.clear()
        sess_hot.table_for(REQ_ONE)
    out.append(_time_one(
        "table_for (cache miss, then clear)",
        _cold,
        repeat=repeat, inner=20_000,
    ))
    return out


def _cache_config_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    sess = SESSION_APPEND
    out.append(_time_one(
        "cache_config_for (remote, warm)",
        lambda: sess.cache_config_for(REQ_ONE),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "local_cache_config_for (mode match → template return)",
        lambda: sess.local_cache_config_for(REQ_ONE),
        repeat=repeat, inner=500_000,
    ))

    # When request.mode disagrees with the template's, the local config
    # is rebuilt via ``merge`` — measure that branch too because it's
    # the steady-state cost when callers flip individual requests off
    # the session default.
    req_override = REQ_ONE.copy()
    req_override.mode = Mode.UPSERT
    out.append(_time_one(
        "local_cache_config_for (mode override → merge)",
        lambda: sess.local_cache_config_for(req_override),
        repeat=repeat, inner=20_000,
    ))
    return out


def _attach_cache_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    sess = SESSION_APPEND
    sess_remote_only = SESSION_REMOTE_ONLY

    # ``_attach_cache`` mutates the request in place when no config is
    # set; copy each iteration so we measure the cold (uncached on the
    # request) branch on every loop. Local + remote both fire.
    def _attach_cold():
        req = REQ_ONE.copy()
        sess._attach_cache(req)
    out.append(_time_one(
        "_attach_cache (local+remote, request had no config)",
        _attach_cold,
        repeat=repeat, inner=50_000,
    ))

    # Remote-only variant — local cache disabled so we see what the
    # local-cache attachment alone adds in the comparison above.
    def _attach_cold_remote_only():
        req = REQ_ONE.copy()
        sess_remote_only._attach_cache(req)
    out.append(_time_one(
        "_attach_cache (remote-only, request had no config)",
        _attach_cold_remote_only,
        repeat=repeat, inner=50_000,
    ))

    # Already-stamped request — the short-circuit path. This is what
    # repeated sends of a reused PreparedRequest pay.
    pre_stamped = REQ_ONE.copy()
    pre_stamped.remote_cache_config = sess.cache_config_for(pre_stamped)
    pre_stamped.local_cache_config = sess.local_cache_config_for(pre_stamped)
    out.append(_time_one(
        "_attach_cache (request already has both configs)",
        lambda: sess._attach_cache(pre_stamped),
        repeat=repeat, inner=500_000,
    ))
    return out


def _mode_override_scenarios(repeat: int) -> list[dict]:
    """Measure the per-request :attr:`PreparedRequest.mode` setter.

    The setter is the natural knob for a caller that wants to flip a
    single request from the session-default APPEND read-through to an
    OVERWRITE / UPSERT round-trip — used to repair a stale cache row
    without rewiring the session. Two cases: (a) the request has no
    cache config yet (just stashes ``_mode``); (b) a remote cache
    config is already attached and the setter rebuilds it via
    :meth:`CacheConfig.merge`. (b) is the steady-state cost on a
    request that already passed through ``_attach_cache``.
    """
    out: list[dict] = []

    # Fresh request per iteration — setter walks the no-config branch.
    def _set_cold():
        req = REQ_ONE.copy()
        req.mode = Mode.UPSERT
    out.append(_time_one(
        "PreparedRequest.mode setter (no cache config)",
        _set_cold,
        repeat=repeat, inner=100_000,
    ))

    # Request already has a CacheConfig — setter rebuilds it via merge.
    def _set_with_cfg():
        req = REQ_ONE.copy()
        req.remote_cache_config = SESSION_APPEND.cache_config_for(req)
        req.mode = Mode.UPSERT
    out.append(_time_one(
        "PreparedRequest.mode setter (rebuilds remote_cache_config)",
        _set_with_cfg,
        repeat=repeat, inner=20_000,
    ))

    # End-to-end: the SchemaSession honors ``req.mode`` even when the
    # session-level mode disagrees. Build a fresh request per loop,
    # assign UPSERT, then let _attach_cache pick that up.
    def _attach_with_override():
        req = REQ_ONE.copy()
        req.mode = Mode.UPSERT
        SESSION_APPEND._attach_cache(req)
    out.append(_time_one(
        "_attach_cache (request.mode override honored)",
        _attach_with_override,
        repeat=repeat, inner=20_000,
    ))
    return out


def _from_scenarios(repeat: int) -> list[dict]:
    """:meth:`SchemaSession.from_` polymorphic-resolve fast path.

    Same-instance dispatch is the steady-state cost when a helper
    keeps re-wrapping a session it already has (idempotent), and the
    Schema-pass-through path is the most common shape when the caller
    has already resolved a schema once. The string / Schemas-service
    paths talk to the Databricks SDK and aren't measurable in this
    bench — they go through ``Schemas.schema`` which is benched
    separately under ``benchmarks/databricks``.
    """
    out: list[dict] = []
    sess = SESSION_APPEND
    schema = sess.schema  # the bench stub schema

    out.append(_time_one(
        "SchemaSession.from_(SchemaSession) — idempotent",
        lambda: SchemaSession.from_(sess),
        repeat=repeat, inner=500_000,
    ))

    # Schema pass-through: real Schema (or duck-typed) → SchemaSession.
    # The stub schema doesn't subclass the real Schema, so this path
    # falls through to the ``isinstance(_Schemas)`` / DatabricksClient
    # / str branches and raises. Skip the cold scenario — covered by
    # the from_ unit test instead.
    _ = schema  # keep ruff happy
    return out


def _send_pipeline_overhead_scenarios(repeat: int) -> list[dict]:
    """Measure the overhead SchemaSession adds vs. HTTPSession on a request
    that *doesn't* actually go to the wire.

    The work compared here is the part of the send pipeline that
    happens before the network — request normalization, config
    resolution, and (for SchemaSession) the per-path cache attachment.
    We invoke the helpers directly instead of ``send`` because ``send``
    would hit the network; the predicates and config builders are the
    paths that run on every send regardless of cache state.
    """
    out: list[dict] = []
    sess_schema = SESSION_APPEND
    sess_base = BASE_SESSION

    # Walk the path-distinct request set so the cache hit case is
    # representative of a real burst over many endpoints (one entry per
    # path warmed during module load).
    def _attach_burst():
        for r in REQS:
            req = r.copy()
            sess_schema._attach_cache(req)

    def _baseline_burst():
        # HTTPSession does no per-request injection — copy is the same
        # both sides so the delta isolates SchemaSession's added work.
        for r in REQS:
            req = r.copy()
            _ = sess_base  # touch to keep the lookup honest
            _ = req

    out.append(_time_one(
        f"SchemaSession._attach_cache burst (n={len(REQS)})",
        _attach_burst,
        repeat=repeat, inner=5_000,
    ))
    out.append(_time_one(
        f"HTTPSession baseline burst (n={len(REQS)}, no cache work)",
        _baseline_burst,
        repeat=repeat, inner=5_000,
    ))
    return out


# ---------------------------------------------------------------------------
# Aggregator + CLI
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    return [
        *_name_scenarios(repeat),
        *_table_for_scenarios(repeat),
        *_cache_config_scenarios(repeat),
        *_attach_cache_scenarios(repeat),
        *_mode_override_scenarios(repeat),
        *_from_scenarios(repeat),
        *_send_pipeline_overhead_scenarios(repeat),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repeat", type=int, default=5,
        help="Outer repeat count per scenario (median across).",
    )
    args = ap.parse_args()
    print(f"# repeat={args.repeat}")
    print(f"# {'label':<60s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
