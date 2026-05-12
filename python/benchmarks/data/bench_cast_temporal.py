"""Benchmark the temporal converters in :mod:`yggdrasil.data.cast.datetime`
*and* the engine-side temporal type kernels (DateType / TimeType /
TimestampType / DurationType) on Arrow / Polars / pandas.

Why this exists
---------------

Temporal coercions sit on every CSV/JSON ingest hot path: Excel/Power
Query timestamps land as strings, REST APIs hand back ISO 8601, log
shippers emit epoch seconds/millis/micros, and the FastAPI routers
re-coerce them back into ``datetime`` before they cross another wire.
The converters in :mod:`yggdrasil.data.cast.datetime` are the canonical
scalar funnel; the per-engine kernels in
:mod:`yggdrasil.data.types.primitive.temporal` fan that out to
vectorised paths across Arrow / Polars / pandas (Spark intentionally
skipped — its cast path is opaque-side and out of scope here).

This bench measures both the registry-side dispatch (``convert(value,
dt.datetime)``) and the direct converter call so we can tell registry
overhead from kernel cost. Coverage:

* ISO-8601 with / without ``Z`` / fractional seconds / explicit offset.
* Compact ``YYYYMMDD`` / ``YYYYMMDDTHHMMSS`` form.
* Slash-form ``YYYY/MM/DD``.
* Epoch numerics (seconds / millis / micros) routed through the
  auto-scale heuristic in ``_numeric_timestamp_to_seconds``.
* ``str_to_tzinfo`` for fixed offsets and IANA names.
* ``any_to_*`` polymorphic dispatch with each accepted source type.
* Identity / no-op datetime->datetime path (target tz=None).
* Array-shaped: cast 10k strings/ints in a Python list to exercise the
  registry's per-element overhead end to end.
* Engine kernels: DateType / TimeType / TimestampType (with / without
  tz) / DurationType cast on Arrow Array, Polars Series, pandas Series
  for the four shapes that matter on the hot path:

  - **MATCH** — source dtype identical to target; the engine-type
    bypass should fire and the call returns the input unchanged.
  - **UNIT**  — same family, different unit (e.g. ``timestamp(us)``
    → ``timestamp(ms)``); the kernel actually casts and the per-engine
    fast path matters.
  - **STR**   — string source parsed into the temporal target; the
    parser cost dominates so this is the most-instructive shape for
    "did the parser regress / can we vectorise more of it".
  - **TZ**    — naive ↔ aware tz reinterpret on the same unit, the
    semantically tricky variant.

Usage::

    PYTHONPATH=src python benchmarks/data/bench_cast_temporal.py
    PYTHONPATH=src python benchmarks/data/bench_cast_temporal.py --repeat 5
    PYTHONPATH=src python benchmarks/data/bench_cast_temporal.py --rows 100000
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable

from yggdrasil.data.cast.datetime import (
    any_to_date,
    any_to_datetime,
    any_to_time,
    any_to_timedelta,
    any_to_tzinfo,
    float_to_datetime,
    int_to_datetime,
    normalize_datetime_string,
    str_to_date,
    str_to_datetime,
    str_to_time,
    str_to_timedelta,
    str_to_tzinfo,
)
from yggdrasil.data.cast.registry import convert


# ---------------------------------------------------------------------------
# Timing helpers (mirror bench_cast.py for a consistent header / scale)
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 200)):
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
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


_ISO_Z = "2024-03-15T12:34:56Z"
_ISO_OFFSET = "2024-03-15T12:34:56+02:00"
_ISO_FRAC = "2024-03-15T12:34:56.123456+00:00"
_ISO_PLAIN = "2024-03-15T12:34:56"
_SLASH = "2024/03/15"
_COMPACT_DATE = "20240315"
_COMPACT_DT = "20240315T123456"
_TIME_STR = "12:34:56.789012"
_TD_HMS = "12:34:56.500000"
_TD_UNIT = "30m"
_TD_ISO = "PT1H30M"
_TZ_NAMED = "Europe/Paris"
_TZ_OFFSET = "+02:30"

_DT_AWARE = dt.datetime(2024, 3, 15, 12, 34, 56, tzinfo=dt.timezone.utc)
_DT_NAIVE = dt.datetime(2024, 3, 15, 12, 34, 56)
_DATE = dt.date(2024, 3, 15)
_TIME = dt.time(12, 34, 56)
_TD = dt.timedelta(hours=1, minutes=30)
_EPOCH_S = 1_710_503_696
_EPOCH_MS = 1_710_503_696_000
_EPOCH_US = 1_710_503_696_000_000
_EPOCH_FLOAT = 1_710_503_696.123


def _str_scenarios(repeat: int) -> list[dict]:
    return [
        _time_one("str_to_datetime: ISO Z",
                  lambda: str_to_datetime(_ISO_Z),
                  repeat=repeat, inner=20_000),
        _time_one("str_to_datetime: ISO offset",
                  lambda: str_to_datetime(_ISO_OFFSET),
                  repeat=repeat, inner=20_000),
        _time_one("str_to_datetime: ISO fractional",
                  lambda: str_to_datetime(_ISO_FRAC),
                  repeat=repeat, inner=20_000),
        _time_one("str_to_datetime: ISO plain (no tz)",
                  lambda: str_to_datetime(_ISO_PLAIN),
                  repeat=repeat, inner=20_000),
        _time_one("str_to_datetime: compact date",
                  lambda: str_to_datetime(_COMPACT_DATE),
                  repeat=repeat, inner=20_000),
        _time_one("str_to_datetime: compact datetime",
                  lambda: str_to_datetime(_COMPACT_DT),
                  repeat=repeat, inner=20_000),
        _time_one("str_to_datetime: slash-form date",
                  lambda: str_to_datetime(_SLASH),
                  repeat=repeat, inner=20_000),
        _time_one("str_to_date: ISO Z",
                  lambda: str_to_date(_ISO_Z),
                  repeat=repeat, inner=20_000),
        _time_one("str_to_time: ISO time",
                  lambda: str_to_time(_TIME_STR),
                  repeat=repeat, inner=50_000),
        _time_one("normalize_datetime_string: already canonical",
                  lambda: normalize_datetime_string(_ISO_PLAIN + "+00:00"),
                  repeat=repeat, inner=50_000),
        _time_one("normalize_datetime_string: needs Z + frac fix",
                  lambda: normalize_datetime_string(_ISO_Z),
                  repeat=repeat, inner=50_000),
    ]


def _td_scenarios(repeat: int) -> list[dict]:
    return [
        _time_one("str_to_timedelta: HH:MM:SS.frac",
                  lambda: str_to_timedelta(_TD_HMS),
                  repeat=repeat, inner=20_000),
        _time_one("str_to_timedelta: '30m'",
                  lambda: str_to_timedelta(_TD_UNIT),
                  repeat=repeat, inner=20_000),
        _time_one("str_to_timedelta: ISO PT1H30M",
                  lambda: str_to_timedelta(_TD_ISO),
                  repeat=repeat, inner=20_000),
    ]


def _tz_scenarios(repeat: int) -> list[dict]:
    return [
        _time_one("str_to_tzinfo: 'UTC'",
                  lambda: str_to_tzinfo("UTC"),
                  repeat=repeat, inner=200_000),
        _time_one("str_to_tzinfo: '+02:30'",
                  lambda: str_to_tzinfo(_TZ_OFFSET),
                  repeat=repeat, inner=50_000),
        _time_one("str_to_tzinfo: 'Europe/Paris'",
                  lambda: str_to_tzinfo(_TZ_NAMED),
                  repeat=repeat, inner=20_000),
    ]


def _numeric_scenarios(repeat: int) -> list[dict]:
    return [
        _time_one("int_to_datetime: epoch seconds",
                  lambda: int_to_datetime(_EPOCH_S),
                  repeat=repeat, inner=50_000),
        _time_one("int_to_datetime: epoch millis",
                  lambda: int_to_datetime(_EPOCH_MS),
                  repeat=repeat, inner=50_000),
        _time_one("int_to_datetime: epoch micros",
                  lambda: int_to_datetime(_EPOCH_US),
                  repeat=repeat, inner=50_000),
        _time_one("float_to_datetime: epoch seconds",
                  lambda: float_to_datetime(_EPOCH_FLOAT),
                  repeat=repeat, inner=50_000),
    ]


def _any_scenarios(repeat: int) -> list[dict]:
    return [
        _time_one("any_to_datetime: from datetime (aware)",
                  lambda: any_to_datetime(_DT_AWARE),
                  repeat=repeat, inner=200_000),
        _time_one("any_to_datetime: from str (ISO Z)",
                  lambda: any_to_datetime(_ISO_Z),
                  repeat=repeat, inner=20_000),
        _time_one("any_to_datetime: from int (epoch s)",
                  lambda: any_to_datetime(_EPOCH_S),
                  repeat=repeat, inner=50_000),
        _time_one("any_to_datetime: from date",
                  lambda: any_to_datetime(_DATE),
                  repeat=repeat, inner=100_000),
        _time_one("any_to_date: from datetime",
                  lambda: any_to_date(_DT_AWARE),
                  repeat=repeat, inner=200_000),
        _time_one("any_to_time: from datetime",
                  lambda: any_to_time(_DT_AWARE),
                  repeat=repeat, inner=200_000),
        _time_one("any_to_timedelta: from int",
                  lambda: any_to_timedelta(60),
                  repeat=repeat, inner=200_000),
        _time_one("any_to_tzinfo: from str ('UTC')",
                  lambda: any_to_tzinfo("UTC"),
                  repeat=repeat, inner=100_000),
    ]


def _registry_scenarios(repeat: int) -> list[dict]:
    return [
        _time_one("registry: convert(ISO Z, dt.datetime)",
                  lambda: convert(_ISO_Z, dt.datetime),
                  repeat=repeat, inner=20_000),
        _time_one("registry: convert(epoch_s int, dt.datetime)",
                  lambda: convert(_EPOCH_S, dt.datetime),
                  repeat=repeat, inner=50_000),
        _time_one("registry: convert(dt_aware, dt.datetime) identity",
                  lambda: convert(_DT_AWARE, dt.datetime),
                  repeat=repeat, inner=200_000),
        _time_one("registry: convert(date, dt.datetime)",
                  lambda: convert(_DATE, dt.datetime),
                  repeat=repeat, inner=100_000),
        _time_one("registry: convert('UTC', dt.tzinfo)",
                  lambda: convert("UTC", dt.tzinfo),
                  repeat=repeat, inner=100_000),
        _time_one("registry: convert('30m', dt.timedelta)",
                  lambda: convert(_TD_UNIT, dt.timedelta),
                  repeat=repeat, inner=20_000),
    ]


def _array_scenarios(repeat: int) -> list[dict]:
    """Per-row Python loop over an array of values — the shape JSON / CSV
    / Excel ingest hits when the vectorised pyarrow path can't decode a
    column and we fall back to per-row coercion."""
    iso_arr = [_ISO_Z] * 10_000
    epoch_arr = [_EPOCH_S + i for i in range(10_000)]

    return [
        _time_one("loop x10k: any_to_datetime(str)",
                  lambda: [any_to_datetime(v) for v in iso_arr],
                  repeat=repeat, inner=5),
        _time_one("loop x10k: any_to_datetime(int)",
                  lambda: [any_to_datetime(v) for v in epoch_arr],
                  repeat=repeat, inner=5),
        _time_one("loop x10k: str_to_datetime",
                  lambda: [str_to_datetime(v) for v in iso_arr],
                  repeat=repeat, inner=5),
    ]


# ---------------------------------------------------------------------------
# Engine kernel scenarios
# ---------------------------------------------------------------------------


def _arrow_engine_scenarios(rows: int, repeat: int) -> list[dict]:
    import pyarrow as pa

    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.types.primitive import (
        DateType,
        DurationType,
        TimestampType,
        TimeType,
    )

    out: list[dict] = []

    ts_us_naive = pa.array([1_710_503_696_000_000 + i for i in range(rows)],
                           type=pa.timestamp("us"))
    ts_us_utc = ts_us_naive.cast(pa.timestamp("us", "UTC"))
    ts_us_paris = ts_us_naive.cast(pa.timestamp("us", "Europe/Paris"))
    date_arr = pa.array([19_800 + i for i in range(rows)], type=pa.date32())
    time_us = pa.array([(i % 86_400) * 1_000_000 for i in range(rows)],
                       type=pa.time64("us"))
    dur_us = pa.array([1_000_000 + i for i in range(rows)],
                      type=pa.duration("us"))
    iso_strs = pa.array(["2024-03-15T12:34:56Z"] * rows, type=pa.string())
    iso_dates = pa.array(["2024-03-15"] * rows, type=pa.string())

    ts_us = TimestampType(unit="us")
    ts_ms = TimestampType(unit="ms")
    ts_us_tz = TimestampType(unit="us", tz="UTC")
    date_t = DateType()
    time_us_t = TimeType(unit="us")
    dur_us_t = DurationType(unit="us")
    dur_ms_t = DurationType(unit="ms")

    def _bench(label: str, dtype, arr, *, inner: int) -> dict:
        opts = CastOptions(target=dtype)
        return _time_one(label, lambda: dtype._cast_arrow_array(arr, opts),
                         repeat=repeat, inner=inner)

    out.append(_bench(f"arrow: TimestampType MATCH us rows={rows}",
                      ts_us, ts_us_naive, inner=200))
    out.append(_bench(f"arrow: TimestampType UNIT us->ms rows={rows}",
                      ts_ms, ts_us_naive, inner=200))
    out.append(_bench(f"arrow: TimestampType TZ naive->UTC rows={rows}",
                      ts_us_tz, ts_us_naive, inner=200))
    out.append(_bench(f"arrow: TimestampType STR iso->us rows={rows}",
                      ts_us, iso_strs, inner=50))
    out.append(_bench(f"arrow: DateType MATCH rows={rows}",
                      date_t, date_arr, inner=200))
    out.append(_bench(f"arrow: DateType STR iso->date rows={rows}",
                      date_t, iso_dates, inner=50))
    out.append(_bench(f"arrow: TimeType MATCH us rows={rows}",
                      time_us_t, time_us, inner=200))
    out.append(_bench(f"arrow: DurationType MATCH us rows={rows}",
                      dur_us_t, dur_us, inner=200))
    out.append(_bench(f"arrow: DurationType UNIT us->ms rows={rows}",
                      dur_ms_t, dur_us, inner=200))
    out.append(_bench(f"arrow: TimestampType ts(UTC)->ts(naive) us->ms rows={rows}",
                      ts_ms, ts_us_utc, inner=200))
    # True same-tz fast-path: ``ts(us, UTC) → ts(ms, UTC)`` is a pure unit
    # conversion the polars detour can't add anything to.
    out.append(_bench(f"arrow: TimestampType ts(UTC)->ts(UTC) us->ms rows={rows}",
                      ts_us_tz.__class__(unit="ms", tz="UTC"), ts_us_utc, inner=200))
    # Cross-tz wall-clock convert — must stay on the polars route.
    out.append(_bench(f"arrow: TimestampType ts(UTC)->ts(Paris) us->us rows={rows}",
                      TimestampType(unit="us", tz="Europe/Paris"), ts_us_utc, inner=200))
    return out


def _polars_engine_scenarios(rows: int, repeat: int) -> list[dict]:
    try:
        import polars as pl
    except ImportError:
        return [{"label": "polars: not installed — skipped",
                 "best": 0.0, "median": 0.0, "mean": 0.0}]

    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.types.primitive import (
        DateType,
        DurationType,
        TimestampType,
        TimeType,
    )

    out: list[dict] = []

    base = dt.datetime(2024, 3, 15, 12, 34, 56)
    ts_us = pl.Series("ts", [base + dt.timedelta(microseconds=i)
                              for i in range(rows)],
                      dtype=pl.Datetime("us"))
    ts_us_utc = ts_us.dt.replace_time_zone("UTC")
    date_s = pl.Series("d", [_DATE + dt.timedelta(days=i % 1000)
                              for i in range(rows)],
                       dtype=pl.Date)
    iso_s = pl.Series("s", ["2024-03-15T12:34:56Z"] * rows, dtype=pl.String)
    iso_dates = pl.Series("s", ["2024-03-15"] * rows, dtype=pl.String)
    dur_us = pl.Series("d", [dt.timedelta(microseconds=1_000_000 + i)
                              for i in range(rows)],
                       dtype=pl.Duration("us"))

    ts_us_t = TimestampType(unit="us")
    ts_ms_t = TimestampType(unit="ms")
    ts_us_tz_t = TimestampType(unit="us", tz="UTC")
    date_t = DateType()
    dur_us_t = DurationType(unit="us")
    dur_ms_t = DurationType(unit="ms")

    def _bench(label, dtype, series, *, inner):
        opts = CastOptions(target=dtype)
        return _time_one(label, lambda: dtype._cast_polars_series(series, opts),
                         repeat=repeat, inner=inner)

    out.append(_bench(f"polars: TimestampType MATCH us rows={rows}",
                      ts_us_t, ts_us, inner=200))
    out.append(_bench(f"polars: TimestampType UNIT us->ms rows={rows}",
                      ts_ms_t, ts_us, inner=200))
    out.append(_bench(f"polars: TimestampType TZ naive->UTC rows={rows}",
                      ts_us_tz_t, ts_us, inner=200))
    out.append(_bench(f"polars: TimestampType STR iso->us rows={rows}",
                      ts_us_t, iso_s, inner=50))
    out.append(_bench(f"polars: DateType MATCH rows={rows}",
                      date_t, date_s, inner=200))
    out.append(_bench(f"polars: DateType STR iso->date rows={rows}",
                      date_t, iso_dates, inner=50))
    out.append(_bench(f"polars: DurationType MATCH us rows={rows}",
                      dur_us_t, dur_us, inner=200))
    out.append(_bench(f"polars: DurationType UNIT us->ms rows={rows}",
                      dur_ms_t, dur_us, inner=200))
    out.append(_bench(f"polars: TimestampType ts(UTC)->ts(UTC) us->ms rows={rows}",
                      ts_ms_t, ts_us_utc, inner=200))
    return out


def _pandas_engine_scenarios(rows: int, repeat: int) -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        return [{"label": "pandas: not installed — skipped",
                 "best": 0.0, "median": 0.0, "mean": 0.0}]

    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.types.primitive import (
        DateType,
        TimestampType,
    )

    out: list[dict] = []

    ts_us_naive = pd.Series(
        pd.date_range("2024-03-15", periods=rows, freq="us"),
    )
    iso_s = pd.Series(["2024-03-15T12:34:56Z"] * rows)
    iso_dates = pd.Series(["2024-03-15"] * rows)

    ts_us_t = TimestampType(unit="us")
    ts_ms_t = TimestampType(unit="ms")
    ts_us_tz_t = TimestampType(unit="us", tz="UTC")
    date_t = DateType()

    def _bench(label, dtype, series, *, inner):
        opts = CastOptions(target=dtype)
        return _time_one(label, lambda: dtype._cast_pandas_series(series, opts),
                         repeat=repeat, inner=inner)

    out.append(_bench(f"pandas: TimestampType MATCH us rows={rows}",
                      ts_us_t, ts_us_naive, inner=50))
    out.append(_bench(f"pandas: TimestampType UNIT us->ms rows={rows}",
                      ts_ms_t, ts_us_naive, inner=50))
    out.append(_bench(f"pandas: TimestampType TZ naive->UTC rows={rows}",
                      ts_us_tz_t, ts_us_naive, inner=50))
    out.append(_bench(f"pandas: TimestampType STR iso->us rows={rows}",
                      ts_us_t, iso_s, inner=20))
    out.append(_bench(f"pandas: DateType STR iso->date rows={rows}",
                      date_t, iso_dates, inner=20))
    return out


def _engine_scenarios(rows: int, repeat: int) -> list[dict]:
    return [
        *_arrow_engine_scenarios(rows, repeat),
        *_polars_engine_scenarios(rows, repeat),
        *_pandas_engine_scenarios(rows, repeat),
    ]


def scenarios(rows: int, repeat: int) -> list[dict]:
    return [
        *_str_scenarios(repeat),
        *_td_scenarios(repeat),
        *_tz_scenarios(repeat),
        *_numeric_scenarios(repeat),
        *_any_scenarios(repeat),
        *_registry_scenarios(repeat),
        *_array_scenarios(repeat),
        *_engine_scenarios(rows, repeat),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    ap.add_argument("--rows", type=int, default=10_000,
                    help="Per-array row count for engine kernel scenarios.")
    args = ap.parse_args()

    print(f"# rows={args.rows}  repeat={args.repeat}")
    print(f"# {'label':<60s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    for row in scenarios(args.rows, args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
