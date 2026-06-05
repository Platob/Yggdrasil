"""Benchmark yggdrasil.node transport layer.

Covers the serialization / deserialization round-trip for the two
transport formats (Arrow IPC stream and pickle) across common payload
shapes, and measures the call endpoint overhead via TestClient.

Metrics reported per scenario
------------------------------
- best / median round-trip time across ``--repeat`` outer loops
- wire size (bytes)
- throughput (MB/s for bulk tabular payloads)

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_bot_transport.py
    PYTHONPATH=src python benchmarks/bot/bench_bot_transport.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa

from yggdrasil.pickle.ser.serde import dumps as ygg_dumps, loads as ygg_loads

from yggdrasil.node.transport import (
    CONTENT_TYPE_ARROW_STREAM,
    CONTENT_TYPE_PICKLE,
    deserialize_pickle,
    deserialize_result,
    is_tabular,
    read_arrow_stream,
    serialize_pickle,
    serialize_result,
    to_arrow_table,
    write_arrow_stream,
    write_arrow_stream_chunked,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

INNER = 50

_SCALAR_INT = 42
_SCALAR_DICT = {"key": "value", "numbers": list(range(100)), "nested": {"a": 1}}
_SCALAR_LIST = [{"id": i, "name": f"item_{i}", "score": float(i) * 0.1} for i in range(100)]

_BYTES_64 = b"\xff" * 64
_BYTES_32K = b"\xaa" * 32_768


def _arrow_table(n: int) -> pa.Table:
    return pa.table({
        "id": pa.array(range(n), type=pa.int64()),
        "value": pa.array([float(i) * 1.1 for i in range(n)], type=pa.float64()),
        "label": pa.array([f"row_{i}" for i in range(n)], type=pa.string()),
        "flag": pa.array([i % 2 == 0 for i in range(n)], type=pa.bool_()),
    })


_TABLE_100 = _arrow_table(100)
_TABLE_10K = _arrow_table(10_000)
_TABLE_100K = _arrow_table(100_000)


try:
    import polars as pl
    _POLARS_DF = pl.DataFrame({"x": range(10_000), "y": [f"s_{i}" for i in range(10_000)]})
    _HAS_POLARS = True
except ImportError:
    _POLARS_DF = None
    _HAS_POLARS = False


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def _time_fn(fn: Callable[[], Any], *, repeat: int, inner: int) -> list[float]:
    for _ in range(min(inner, 10)):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return samples


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

_HDR = f"{'scenario':<42}  {'best µs':>10}  {'median µs':>10}  {'wire B':>10}"
_SEP = "-" * len(_HDR)


def _fmt(label: str, samples: list[float], size: int) -> str:
    best = min(samples) * 1e6
    med = statistics.median(samples) * 1e6
    return f"{label:<42}  {best:>10.1f}  {med:>10.1f}  {size:>10,}"


def _fmt_tp(label: str, samples: list[float], size: int, data_mb: float) -> str:
    best = min(samples) * 1e6
    med = statistics.median(samples) * 1e6
    tp = data_mb / min(samples) if min(samples) > 0 else 0
    return f"{label:<42}  {best:>10.1f}  {med:>10.1f}  {size:>10,}  {tp:>8.1f} MB/s"


# ---------------------------------------------------------------------------
# benchmark groups
# ---------------------------------------------------------------------------

def _bench_pickle_roundtrip(repeat: int) -> None:
    print("\n--- pickle transport round-trip ---")
    print(_HDR)
    print(_SEP)

    for label, obj in [
        ("int", _SCALAR_INT),
        ("dict (100 values)", _SCALAR_DICT),
        ("list[dict] (100 items)", _SCALAR_LIST),
        ("bytes (64 B)", _BYTES_64),
        ("bytes (32 KiB)", _BYTES_32K),
    ]:
        data = serialize_pickle(obj)
        size = len(data)
        samples = _time_fn(
            lambda o=obj: deserialize_pickle(serialize_pickle(o)),
            repeat=repeat,
            inner=INNER,
        )
        print(_fmt(label, samples, size))


def _bench_arrow_stream(repeat: int) -> None:
    print(f"\n--- Arrow IPC stream round-trip ---")
    tp_hdr = _HDR + "  throughput"
    print(tp_hdr)
    print("-" * len(tp_hdr))

    for label, table in [
        ("Arrow Table (100 rows × 4 cols)", _TABLE_100),
        ("Arrow Table (10K rows × 4 cols)", _TABLE_10K),
        ("Arrow Table (100K rows × 4 cols)", _TABLE_100K),
    ]:
        stream_bytes = b"".join(write_arrow_stream(table))
        size = len(stream_bytes)
        data_mb = size / 1e6

        samples = _time_fn(
            lambda t=table: read_arrow_stream(b"".join(write_arrow_stream(t))),
            repeat=repeat,
            inner=max(INNER // 5, 2),
        )
        print(_fmt_tp(label, samples, size, data_mb))


def _bench_arrow_chunked_vs_single(repeat: int) -> None:
    print("\n--- chunked vs single-shot Arrow stream (10K rows) ---")
    print(f"{'mode':<42}  {'best µs':>10}  {'median µs':>10}")
    print("-" * 66)

    table = _TABLE_10K

    single = _time_fn(
        lambda: b"".join(write_arrow_stream(table)),
        repeat=repeat,
        inner=INNER,
    )
    chunked = _time_fn(
        lambda: b"".join(write_arrow_stream_chunked(table, max_chunksize=8192)),
        repeat=repeat,
        inner=INNER,
    )

    print(f"{'single-shot':<42}  {min(single)*1e6:>10.1f}  {statistics.median(single)*1e6:>10.1f}")
    print(f"{'chunked (8192 rows)':<42}  {min(chunked)*1e6:>10.1f}  {statistics.median(chunked)*1e6:>10.1f}")


def _bench_serialize_result_dispatch(repeat: int) -> None:
    print("\n--- serialize_result dispatch (format selection) ---")
    print(_HDR)
    print(_SEP)

    for label, obj in [
        ("scalar dict", _SCALAR_DICT),
        ("Arrow Table (100 rows)", _TABLE_100),
        ("Arrow Table (10K rows)", _TABLE_10K),
    ]:
        data, ct = serialize_result(obj)
        size = len(data)
        samples = _time_fn(
            lambda o=obj: serialize_result(o),
            repeat=repeat,
            inner=max(INNER // 2, 5),
        )
        print(_fmt(f"{label} → {ct.split('/')[-1][:20]}", samples, size))


def _bench_polars(repeat: int) -> None:
    if not _HAS_POLARS:
        print("\n  [polars not installed — skipping]")
        return

    print("\n--- Polars DataFrame transport ---")
    tp_hdr = _HDR + "  throughput"
    print(tp_hdr)
    print("-" * len(tp_hdr))

    data, ct = serialize_result(_POLARS_DF)
    size = len(data)
    data_mb = size / 1e6

    samples = _time_fn(
        lambda: serialize_result(_POLARS_DF),
        repeat=repeat,
        inner=max(INNER // 5, 2),
    )
    print(_fmt_tp("Polars DF (10K × 2) → Arrow stream", samples, size, data_mb))

    arrow_bytes = data
    samples_read = _time_fn(
        lambda: read_arrow_stream(arrow_bytes),
        repeat=repeat,
        inner=max(INNER // 5, 2),
    )
    print(_fmt_tp("Arrow stream → pa.Table (10K × 2)", samples_read, size, data_mb))


def _bench_call_endpoint(repeat: int) -> None:
    from yggdrasil.node.app import create_app
    from yggdrasil.node.config import Settings
    from yggdrasil.node.remote import remote

    @remote(name="bench:add")
    def _bench_add(x: int, y: int) -> int:
        return x + y

    @remote(name="bench:make_table")
    def _bench_make_table(n: int) -> pa.Table:
        return _arrow_table(n)

    settings = Settings(allow_remote=True)
    app = create_app(settings)

    from fastapi.testclient import TestClient
    client = TestClient(app)

    print("\n--- /api/call endpoint overhead ---")
    print(f"{'scenario':<42}  {'best ms':>10}  {'median ms':>10}")
    print("-" * 66)

    for label, payload in [
        ("scalar return (add)", {"func": "bench:add", "args": (10, 20), "kwargs": {}}),
        ("tabular return (100 rows)", {"func": "bench:make_table", "args": (100,), "kwargs": {}}),
        ("tabular return (10K rows)", {"func": "bench:make_table", "args": (10_000,), "kwargs": {}}),
    ]:
        body = serialize_pickle(payload)
        headers = {"Content-Type": CONTENT_TYPE_PICKLE}

        samples = _time_fn(
            lambda b=body, h=headers: client.post("/api/call", content=b, headers=h),
            repeat=repeat,
            inner=max(INNER // 10, 2),
        )
        best_ms = min(samples) * 1e3
        med_ms = statistics.median(samples) * 1e3
        print(f"{label:<42}  {best_ms:>10.2f}  {med_ms:>10.2f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(repeat: int) -> None:
    print()
    print("=" * 90)
    print(f"  yggdrasil.node transport benchmark  (repeat={repeat}, inner={INNER})")
    print("=" * 90)

    _bench_pickle_roundtrip(repeat)
    _bench_arrow_stream(repeat)
    _bench_arrow_chunked_vs_single(repeat)
    _bench_serialize_result_dispatch(repeat)
    _bench_polars(repeat)
    _bench_call_endpoint(repeat)

    print()
    print("=" * 90)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5, help="Outer timing loops")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    run(repeat=args.repeat)
