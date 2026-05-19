"""Benchmark :class:`yggdrasil.fxrate.FxRate` orchestration overhead.

What this measures (and what it does NOT)
-----------------------------------------
This benchmark stubs the wire — no real HTTP. It measures everything
:class:`FxRate` does *around* the network call: input coercion, pair
grouping, backend fan-out + fallback, the long-frame assembly into
polars, geography enrichment via :class:`GeoZoneCatalog`. Those are
the costs you pay per call regardless of which upstream you hit, and
the things worth optimising.

It does NOT measure HTTP round trips, Frankfurter / Fawaz / ER-API
parsing of large payloads, or polars' internal aggregation cost on
massive frames. Different bench, different concerns.

Usage::

    PYTHONPATH=src python benchmarks/fxrate/bench_fxrate.py
    PYTHONPATH=src python benchmarks/fxrate/bench_fxrate.py --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import statistics
import time
from typing import Callable, Sequence

from yggdrasil.data.enums.currency import Currency
from yggdrasil.fxrate import (
    BackendError,
    FxQuote,
    FxRate,
)
from yggdrasil.fxrate.backends import Backend

# Silence the orchestration logger — the fallback-walk scenarios fire
# the "backend X failed, falling back" warning a few thousand times per
# bench, and we'd rather not drown the timing summary in log spam.
logging.getLogger("yggdrasil.fxrate").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Stub backend — deterministic, in-memory, zero overhead per call
# ---------------------------------------------------------------------------


def _make_quotes(
    source: str,
    targets: Sequence[str],
    *,
    start_date: dt.date,
    days: int,
) -> list[FxQuote]:
    """Pre-build ``days * len(targets)`` quotes for the *(source, targets)* group.

    Done once at fixture time so the benchmarked path doesn't pay the
    quote construction cost — we only want to measure FxRate itself.
    """
    out: list[FxQuote] = []
    for d in range(days):
        from_ts = dt.datetime.combine(
            start_date + dt.timedelta(days=d), dt.time.min, tzinfo=dt.timezone.utc,
        )
        to_ts = from_ts + dt.timedelta(days=1)
        for i, tgt in enumerate(targets):
            out.append(FxQuote(
                source=source,
                target=tgt,
                from_timestamp=from_ts,
                to_timestamp=to_ts,
                sampling="1d",
                value=1.0 + 0.001 * (d + i),
            ))
    return out


class _StubBackend(Backend):
    """In-memory backend that hands out a pre-built quote list per call.

    ``raise_with`` flips the backend into "always fail" mode so we can
    bench the fallback walk.
    """

    name: str = "stub"
    base_url: str = "stub://"
    default_sampling: str = "1d"

    def __init__(self, quotes: list[FxQuote] | None = None, raise_with: Exception | None = None) -> None:
        self._quotes = quotes or []
        self._raise = raise_with

    def fetch_timeseries(self, session, *, source, targets, start, end, sampling):
        if self._raise is not None:
            raise self._raise
        return self._quotes

    def fetch_latest(self, session, *, source, targets, at):
        if self._raise is not None:
            raise self._raise
        return self._quotes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_START = dt.date(2024, 1, 1)
_QUOTES_30D_3TGT = _make_quotes(
    "EUR", ("USD", "GBP", "JPY"), start_date=_START, days=30,
)
_QUOTES_30D_1TGT = _make_quotes(
    "EUR", ("USD",), start_date=_START, days=30,
)
_QUOTES_1D_1TGT = _make_quotes(
    "EUR", ("USD",), start_date=_START, days=1,
)


# Sessions — singletons by base_url, so we instantiate once and reuse.
_FAST_BACKEND_3TGT = _StubBackend(_QUOTES_30D_3TGT)
_FAST_BACKEND_1TGT = _StubBackend(_QUOTES_30D_1TGT)
_FAST_BACKEND_SHORT = _StubBackend(_QUOTES_1D_1TGT)


def _make_session(*backends: Backend) -> FxRate:
    """Each call returns a fresh-configured singleton FxRate."""
    return FxRate(backends=backends)


# ---------------------------------------------------------------------------
# Timing harness — mirrors benchmarks/io/bench_http.py
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 50)):
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
    scale, unit = 1e6, "us"
    if r["best"] < 1e-6:
        scale, unit = 1e9, "ns"
    return (
        f"{r['label']:<60s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _coercion_scenarios(repeat: int) -> list[dict]:
    """Per-call input shaping — pair / date parsing."""
    from yggdrasil.fxrate.session import (
        _coerce_currency, _coerce_datetime, _coerce_pair,
        _group_pairs_by_source,
    )
    out: list[dict] = []
    out.append(_time_one(
        "coerce_currency('EUR')",
        lambda: _coerce_currency("EUR"),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "coerce_currency('$') [alias]",
        lambda: _coerce_currency("$"),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "coerce_currency(Currency.EUR) [identity]",
        lambda: _coerce_currency(Currency.EUR),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "coerce_pair(('EUR','USD'))",
        lambda: _coerce_pair(("EUR", "USD")),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "coerce_datetime('2024-01-01')",
        lambda: _coerce_datetime("2024-01-01"),
        repeat=repeat, inner=10_000,
    ))
    iso = "2024-01-01T10:00:00+00:00"
    out.append(_time_one(
        "coerce_datetime(ISO w/ tz)",
        lambda: _coerce_datetime(iso),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "coerce_datetime(epoch int)",
        lambda: _coerce_datetime(1704067200),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "coerce_datetime(datetime instance)",
        lambda: _coerce_datetime(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)),
        repeat=repeat, inner=10_000,
    ))
    pairs10 = [
        (Currency.EUR, Currency.USD),
        (Currency.EUR, Currency.GBP),
        (Currency.EUR, Currency.JPY),
        (Currency.USD, Currency.JPY),
        (Currency.USD, Currency.EUR),
        (Currency.GBP, Currency.USD),
        (Currency.GBP, Currency.EUR),
        (Currency.CHF, Currency.USD),
        (Currency.CHF, Currency.EUR),
        (Currency.JPY, Currency.USD),
    ]
    out.append(_time_one(
        "group_pairs_by_source (10 pairs, 5 sources)",
        lambda: _group_pairs_by_source(pairs10),
        repeat=repeat, inner=5_000,
    ))
    return out


def _fetch_scenarios(repeat: int) -> list[dict]:
    """End-to-end ``fetch`` through the stub — measures orchestration only."""
    out: list[dict] = []

    fx_fast = _make_session(_FAST_BACKEND_SHORT)
    out.append(_time_one(
        "fetch 1 pair, 1 day (eager frame)",
        lambda: fx_fast.fetch(
            pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01",
        ),
        repeat=repeat, inner=500,
    ))

    fx30 = _make_session(_FAST_BACKEND_1TGT)
    out.append(_time_one(
        "fetch 1 pair, 30 days (eager frame)",
        lambda: fx30.fetch(
            pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-30",
        ),
        repeat=repeat, inner=500,
    ))

    fx30_3 = _make_session(_FAST_BACKEND_3TGT)
    out.append(_time_one(
        "fetch 3 pairs, 30 days (90 rows)",
        lambda: fx30_3.fetch(
            pairs=[("EUR", "USD"), ("EUR", "GBP"), ("EUR", "JPY")],
            start="2024-01-01", end="2024-01-30",
        ),
        repeat=repeat, inner=500,
    ))

    out.append(_time_one(
        "fetch 3 pairs, 30 days (lazy)",
        lambda: fx30_3.fetch(
            pairs=[("EUR", "USD"), ("EUR", "GBP"), ("EUR", "JPY")],
            start="2024-01-01", end="2024-01-30",
            lazy=True,
        ),
        repeat=repeat, inner=500,
    ))

    # Latest path — same orchestration, smaller payload
    fx_short = _make_session(_FAST_BACKEND_SHORT)
    out.append(_time_one(
        "latest 1 pair",
        lambda: fx_short.latest(pairs=[("EUR", "USD")]),
        repeat=repeat, inner=1_000,
    ))

    return out


def _fallback_scenarios(repeat: int) -> list[dict]:
    """Backend fallback walk — measures the cost of a one-fail roll-over."""
    out: list[dict] = []
    primary = _StubBackend(raise_with=BackendError("simulated"))
    secondary = _StubBackend(_QUOTES_1D_1TGT)
    fx = _make_session(primary, secondary)
    out.append(_time_one(
        "fetch with 1 fail+1 success backend",
        lambda: fx.fetch(
            pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01",
        ),
        repeat=repeat, inner=500,
    ))
    # Two failing backends + one success — the realistic worst case
    # for the default chain (primary down, secondary down, tertiary
    # answers).
    p1 = _StubBackend(raise_with=BackendError("down 1"))
    p2 = _StubBackend(raise_with=BackendError("down 2"))
    p3 = _StubBackend(_QUOTES_1D_1TGT)
    fx2 = _make_session(p1, p2, p3)
    out.append(_time_one(
        "fetch with 2 fail+1 success backends",
        lambda: fx2.fetch(
            pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01",
        ),
        repeat=repeat, inner=500,
    ))
    return out


def _geo_scenarios(repeat: int) -> list[dict]:
    """Geography enrichment — joins per-currency catalog lookups."""
    out: list[dict] = []
    fx = _make_session(_FAST_BACKEND_3TGT)
    out.append(_time_one(
        "fetch 3 pairs, 30 days (no geo)",
        lambda: fx.fetch(
            pairs=[("EUR", "USD"), ("EUR", "GBP"), ("EUR", "JPY")],
            start="2024-01-01", end="2024-01-30",
        ),
        repeat=repeat, inner=500,
    ))
    # Warm the catalog so geo=True measures the steady-state cost
    # (the first call pays a one-time HTTP fetch + index build).
    try:
        fx.fetch(
            pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01",
            geo=True,
        )
    except Exception:
        pass
    out.append(_time_one(
        "fetch 3 pairs, 30 days (geo=True)",
        lambda: fx.fetch(
            pairs=[("EUR", "USD"), ("EUR", "GBP"), ("EUR", "JPY")],
            start="2024-01-01", end="2024-01-30",
            geo=True,
        ),
        repeat=repeat, inner=500,
    ))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    sections: list[tuple[str, list[dict]]] = [
        ("Input coercion", _coercion_scenarios(args.repeat)),
        ("End-to-end fetch (stubbed transport)", _fetch_scenarios(args.repeat)),
        ("Fallback walk", _fallback_scenarios(args.repeat)),
        ("Geography enrichment", _geo_scenarios(args.repeat)),
    ]
    for title, results in sections:
        print()
        print(f"--- {title} ---")
        for r in results:
            print(_fmt(r))


if __name__ == "__main__":
    main()
