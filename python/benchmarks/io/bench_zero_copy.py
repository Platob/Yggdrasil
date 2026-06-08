"""Memory / zero-copy profile of the HTTP path and IO byte operations.

Measures peak Python allocations (``tracemalloc``) for the common ways
bytes move through :mod:`yggdrasil.io` and :mod:`yggdrasil.http_`, to
surface where a full-payload copy happens vs where the path is
zero-copy. The goal isn't wall time — it's "how many extra copies of the
payload does this operation make".

Reading the numbers
-------------------

A value near the payload size = one full copy; near 0 = zero-copy (a
memoryview / pyarrow Buffer over the existing bytes). Note tracemalloc
traces Python allocations only — pyarrow C++ tables and ``io.BytesIO`` /
``requests`` C buffers are invisible, so an Arrow read that lands its
result in a C++ table reads as ~0 here (the point: no *intermediate*
Python copy).

Usage::

    python benchmarks/io/bench_zero_copy.py
    python benchmarks/io/bench_zero_copy.py --size-mib 64
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import tracemalloc

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse
from yggdrasil.path.memory import Memory


def _peak(fn) -> float:
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 1048576


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size-mib", type=int, default=16)
    args = ap.parse_args()

    n = args.size_mib * 1024 * 1024
    blob = b"x" * n

    table = pa.table({f"c{i}": pa.array(range(n // 128), type=pa.int64()) for i in range(8)})
    psink = io.BytesIO()
    pq.write_table(table, psink)
    parquet_bytes = psink.getvalue()

    rows = []

    # ---- IO / Memory ----
    m = Memory(binary=blob)
    rows.append(("Memory(binary=bytes)", _peak(lambda: Memory(binary=blob)), "1 buffer copy"))
    rows.append(("Memory.to_bytes()", _peak(lambda: m.to_bytes()), "1 copy (returns bytes)"))
    rows.append(("Memory.read_mv(-1,0)", _peak(lambda: m.read_mv(-1, 0)), "ZERO-COPY view"))
    rows.append(("Memory.read_bytes()", _peak(lambda: m.read_bytes()), "1 copy (returns bytes)"))

    # ---- Arrow read off an in-memory holder (snapshot path) ----
    mp = Memory(binary=parquet_bytes)
    rows.append((
        "parquet read via arrow_input_stream",
        _peak(lambda: pq.read_table(Memory(binary=parquet_bytes).arrow_input_stream().__enter__())),
        "ZERO-COPY wrap (no intermediate bytes)",
    ))

    # ---- HTTP request body (prepare + zero-copy send view) ----
    rows.append((
        "HTTPRequest.prepare(body=bytes)",
        _peak(lambda: HTTPRequest.prepare("PUT", "https://h/x", body=blob)),
        "1 buffer copy (owns body)",
    ))
    req = HTTPRequest.prepare("PUT", "https://h/x", body=blob)
    rows.append((
        "request.buffer.read_mv (send view)",
        _peak(lambda: req.buffer.read_mv(-1, 0)),
        "ZERO-COPY view (what _send_once sends)",
    ))

    # ---- HTTP response body access ----
    resp = HTTPResponse(
        request=HTTPRequest.prepare("GET", "https://h/x"),
        status_code=200, headers={}, tags={},
        buffer=Memory(binary=blob), received_at=dt.datetime.now(dt.timezone.utc),
    )
    rows.append(("HTTPResponse.content", _peak(lambda: resp.content), "1 copy (returns bytes)"))
    rows.append(("HTTPResponse.body.read_mv", _peak(lambda: resp.body.read_mv(-1, 0)), "ZERO-COPY view"))

    print(f"\nMemory / zero-copy profile — payload={args.size_mib} MiB\n")
    hdr = f"{'operation':<40} {'peak MiB':>9}  note"
    print(hdr)
    print("-" * (len(hdr) + 16))
    for name, peak, note in rows:
        print(f"{name:<40} {peak:>9.2f}  {note}")
    print(
        "\nFull-payload reads expose a zero-copy memoryview (read_mv); the "
        "bytes-returning accessors (to_bytes/content/read_bytes) copy once "
        "by contract. The hot wire + Arrow paths are zero-copy: _send_once "
        "sends a read_mv view, and arrow_input_stream wraps the buffer in a "
        "pyarrow Buffer (no intermediate bytes) for Parquet/Arrow reads."
    )


if __name__ == "__main__":
    main()
