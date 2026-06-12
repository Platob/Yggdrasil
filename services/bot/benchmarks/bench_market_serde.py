"""Benchmark market data serialization: JSON vs Arrow IPC.

Compares:
- orjson serialize/deserialize of OHLCV list
- Arrow IPC write/read of OHLCV table
- Polars conversion

Usage::

    PYTHONPATH=../../.. python bench_market_serde.py
"""
from __future__ import annotations

import argparse
import io
import random
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import orjson
import pyarrow as pa
import pyarrow.ipc as ipc

from services.bot.api.core.market import ohlcv_to_arrow, ohlcv_to_polars
from services.bot.api.models.market import OHLCV


def _make_bars(n: int) -> list[OHLCV]:
    random.seed(7)
    price, base = 100.0, datetime(2024, 1, 1)
    bars = []
    for i in range(n):
        price = max(1.0, price + random.gauss(0, 1.5))
        bars.append(OHLCV(
            symbol="BENCH",
            timestamp=base + timedelta(days=i),
            open=price, high=price + 1, low=price - 1, close=price,
            volume=random.randint(1_000_000, 20_000_000),
        ))
    return bars


def _bench(name: str, fn, repeat: int) -> None:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    best = min(times) * 1e6
    med  = statistics.median(times) * 1e6
    size = len(result) if isinstance(result, (bytes, bytearray)) else 0
    size_str = f"  {size/1024:.1f} KB" if size else ""
    print(f"{name:<55} best={best:>8.1f} us  median={med:>8.1f} us{size_str}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    for n in [60, 252, 2520]:
        bars = _make_bars(n)
        table = ohlcv_to_arrow(bars)
        dicts = [b.model_dump(mode="json") for b in bars]
        json_bytes = orjson.dumps(dicts)

        print(f"\n=== n={n} bars ===")
        print(f"{'Benchmark':<55} {'Best':>16}  {'Median':>16}")
        print("-" * 90)

        _bench(f"orjson.dumps — {n} bars",
               lambda d=dicts: orjson.dumps(d), args.repeat)
        _bench(f"orjson.loads — {n} bars",
               lambda b=json_bytes: orjson.loads(b), args.repeat)
        _bench(f"Arrow IPC write — {n} bars",
               lambda t=table: _arrow_write(t), args.repeat)
        _bench(f"Arrow IPC read — {n} bars",
               lambda t=table: _arrow_read_write(t), args.repeat)
        _bench(f"ohlcv_to_arrow — {n} bars",
               lambda b=bars: ohlcv_to_arrow(b), args.repeat)
        _bench(f"ohlcv_to_polars — {n} bars",
               lambda b=bars: ohlcv_to_polars(b), args.repeat)


def _arrow_write(table: pa.Table) -> bytes:
    buf = io.BytesIO()
    with ipc.new_stream(buf, table.schema) as w:
        w.write_table(table)
    return buf.getvalue()


def _arrow_read_write(table: pa.Table) -> pa.Table:
    data = _arrow_write(table)
    return ipc.open_stream(io.BytesIO(data)).read_all()


if __name__ == "__main__":
    main()
