"""Benchmark the market data path and signal computation (no HTTP).

Measures:
 1. TTL cache lookup vs cold fetch cost
 2. Signal computation on varying price windows
 3. ENTSOE XML parse → polars frame (with stubbed XML)

Usage::

    PYTHONPATH=src python benchmarks/bot_v2/bench_market_signal.py
"""
from __future__ import annotations

import datetime as dt
import statistics
import time
from pathlib import Path


INNER = 200

_HDR = f"{'scenario':<48}  {'best µs':>10}  {'median µs':>10}"
_SEP = "-" * len(_HDR)


def _time(fn, *, repeat: int = 5, inner: int = INNER) -> list[float]:
    for _ in range(min(inner, 20)):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return samples


def _fmt(label: str, samples: list[float]) -> str:
    best = min(samples) * 1e6
    med = statistics.median(samples) * 1e6
    return f"{label:<48}  {best:>10.1f}  {med:>10.1f}"


# ---------------------------------------------------------------------------
# 1. TTL cache
# ---------------------------------------------------------------------------

def bench_cache() -> None:
    print("\n--- TTL cache (hit vs miss) ---")
    print(_HDR)
    print(_SEP)

    from yggdrasil.bot.market import _TTLCache, _cache

    cache = _TTLCache()
    val = list(range(168))           # week of hourly values

    # cold miss
    samples_miss = _time(lambda: cache.get("key:miss"))
    print(_fmt("cache.get() — miss (absent)", samples_miss))

    # warm hit
    cache.set("key:hit", val, ttl=300)
    samples_hit = _time(lambda: cache.get("key:hit"))
    print(_fmt("cache.get() — hit (present)", samples_hit))

    # set cost
    samples_set = _time(lambda: cache.set("key:w", val, ttl=300))
    print(_fmt("cache.set() (list[168])", samples_set))


# ---------------------------------------------------------------------------
# 2. Signal computation
# ---------------------------------------------------------------------------

def bench_signals() -> None:
    print("\n--- signal computation (zscore → BUY/SELL/HOLD) ---")
    print(_HDR)
    print(_SEP)

    from yggdrasil.bot.signals import compute_signals

    base_ts = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)

    def _prices(n: int, spike: bool = False) -> list[dict]:
        vals = [50.0 + (i % 10) * 0.5 for i in range(n)]
        if spike:
            vals[-1] = 200.0   # force SELL
        return [{"timestamp": base_ts + dt.timedelta(hours=i), "value": v,
                 "unit": "MWh", "currency": "EUR"} for i, v in enumerate(vals)]

    for label, prices in [
        ("24 pts (1 day)", _prices(24)),
        ("168 pts (1 week)", _prices(168)),
        ("720 pts (30 days hourly)", _prices(720)),
        ("720 pts — spike (SELL signal)", _prices(720, spike=True)),
    ]:
        s = _time(lambda p=prices: compute_signals(p, "DE_LU", "day_ahead_prices"))
        print(_fmt(label, s))


# ---------------------------------------------------------------------------
# 3. ENTSOE XML → frame
# ---------------------------------------------------------------------------

def bench_entsoe_parse() -> None:
    from yggdrasil.loki.entsoe import parse_timeseries_xml, to_frame

    print("\n--- ENTSOE XML parse → polars frame ---")
    print(_HDR)
    print(_SEP)

    def _xml(days: int, step_min: int = 60) -> str:
        base = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
        pts = (days * 24 * 60) // step_min
        resolution = f"PT{step_min}M" if step_min < 60 else "PT60M"
        rows_xml = "\n".join(
            f"    <Point><position>{i+1}</position><price.amount>{50.0+i*0.01:.2f}</price.amount></Point>"
            for i in range(pts)
        )
        return f"""<?xml version="1.0"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>{base.strftime('%Y-%m-%dT%H:%MZ')}</start>
        <end>{(base + dt.timedelta(days=days)).strftime('%Y-%m-%dT%H:%MZ')}</end>
      </timeInterval>
      <resolution>{resolution}</resolution>
{rows_xml}
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""

    for label, days, step_min in [
        ("1d @ 60m (24 pts)", 1, 60),
        ("7d @ 60m (168 pts)", 7, 60),
        ("30d @ 60m (720 pts)", 30, 60),
        ("7d @ 15m (672 pts)", 7, 15),
    ]:
        xml = _xml(days, step_min)
        s_parse = _time(lambda x=xml: parse_timeseries_xml(x), inner=50)
        print(_fmt(f"parse_timeseries_xml — {label}", s_parse))

        s_frame = _time(lambda x=xml: to_frame(x, zone="DE_LU", series="day_ahead_prices"), inner=50)
        print(_fmt(f"to_frame — {label}", s_frame))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run() -> None:
    print()
    print("=" * 75)
    print("  ygg-bot market+signal benchmark")
    print("=" * 75)
    bench_cache()
    bench_signals()
    bench_entsoe_parse()
    print()
    print("=" * 75)
    print()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    run()
