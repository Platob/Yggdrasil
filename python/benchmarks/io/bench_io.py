"""Benchmark the :mod:`yggdrasil.io` core primitives.

What this covers
----------------

The pieces every HTTP / file / Databricks call chain reaches:

* :class:`URL` — parse, format, join, derived properties (``host`` /
  ``parts`` / ``extensions``), hash, equality. Built per request, per
  response, per filesystem walk.
* :class:`Headers` — construct / get / set / hash / version-bumped
  digest. Hit once per request and once per response.
* :class:`BytesIO` — write / read / seek round-trip + the
  ``pa.BufferReader`` interop the parquet / arrow IO leaves use.
* ``anonymize_parameters`` — runs on every observability emit.

Skips out-of-process work (network, disk) so the numbers measure
yggdrasil's own overhead rather than the OS or remote service.

Usage::

    PYTHONPATH=src python benchmarks/bench_io.py
    PYTHONPATH=src python benchmarks/bench_io.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import PurePosixPath
from typing import Callable

from yggdrasil.io import URL, BytesIO
from yggdrasil.io.headers import Headers
from yggdrasil.io.parameters import anonymize_parameters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Representative URL shapes — short HTTPS, file://, S3, mem://, query-heavy.
HTTPS_STR = "https://api.example.com:8443/v1/accounts/12345/transactions?from=2024-01-01&to=2024-12-31&page=3#row=12"
FILE_STR = "file:///home/data/2024/q1/orders.parquet"
S3_STR = "s3://bucket-name/datasets/raw/year=2024/month=01/part-00000.parquet"

URL_HTTPS = URL.from_str(HTTPS_STR)
URL_FILE = URL.from_str(FILE_STR)
URL_S3 = URL.from_str(S3_STR)
URL_DICT = URL_HTTPS.to_struct_dict()

HEADERS_SMALL = Headers({
    "Content-Type": "application/json",
    "User-Agent": "yggdrasil/0.x",
    "Accept-Encoding": "gzip,deflate",
})
HEADERS_LARGE = Headers({
    f"X-Header-{i:02d}": f"value-{i:02d}" for i in range(20)
})

PARAMS_FLAT = {
    "user": "alice",
    "password": "s3cret",
    "token": "abcdef0123456789",
    "page": 3,
    "limit": 100,
}
PARAMS_NESTED = {
    "auth": {"user": "alice", "password": "s3cret"},
    "page": 3,
    "filters": {"from": "2024-01-01", "to": "2024-12-31"},
}

PAYLOAD_SMALL = b"hello world\n" * 8                         # 96 B
PAYLOAD_MEDIUM = b"x" * 64_000                                # 64 KB
PAYLOAD_LARGE = b"x" * 1_000_000                              # ~1 MB


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
        f"{r['label']:<58s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# URL scenarios
# ---------------------------------------------------------------------------


def _url_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # Parsing — the workload "URL.from_(str)" pays per request /
    # response / volume read.
    out.append(_time_one(
        "URL.from_str(https://…?…#…)",
        lambda: URL.from_str(HTTPS_STR),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "URL.from_str(file:///…)",
        lambda: URL.from_str(FILE_STR),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "URL.from_str(s3://…)",
        lambda: URL.from_str(S3_STR),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "URL.from_(existing URL) identity",
        lambda: URL.from_(URL_HTTPS),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "URL.from_dict(url_dict)",
        lambda: URL.from_dict(URL_DICT),
        repeat=repeat, inner=20_000,
    ))

    # Formatting — warm cache + cold first call.
    out.append(_time_one(
        "URL.to_string() warm cache",
        lambda: URL_HTTPS.to_string(),
        repeat=repeat, inner=500_000,
    ))

    def fresh_to_string():
        URL.from_str(HTTPS_STR).to_string()
    out.append(_time_one(
        "URL.to_string() cold (per-call URL)",
        fresh_to_string,
        repeat=repeat, inner=10_000,
    ))

    # str() / repr() — used in every traceback, log line, debug print.
    out.append(_time_one(
        "str(URL)",
        lambda: str(URL_HTTPS),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "repr(URL)  (redacted)",
        lambda: repr(URL_HTTPS),
        repeat=repeat, inner=20_000,
    ))

    # Hash / equality — needed for dict / set membership in caches.
    out.append(_time_one(
        "hash(URL)",
        lambda: hash(URL_HTTPS),
        repeat=repeat, inner=500_000,
    ))
    url_other = URL.from_str(HTTPS_STR)
    out.append(_time_one(
        "URL == URL (equal, distinct)",
        lambda: URL_HTTPS == url_other,
        repeat=repeat, inner=500_000,
    ))

    # Derived properties — read on every routing / extension / codec
    # decision in the IO layer.
    out.append(_time_one(
        "URL.parts",
        lambda: URL_HTTPS.parts,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "URL.extensions",
        lambda: URL_FILE.extensions,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "URL.name",
        lambda: URL_FILE.name,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "URL.is_http",
        lambda: URL_HTTPS.is_http,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "URL.parent",
        lambda: URL_FILE.parent,
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "URL.is_urlish('https://...')",
        lambda: URL.is_urlish(HTTPS_STR),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "URL.joinpath('subdir', 'file.csv')",
        lambda: URL_S3.joinpath("subdir", "file.csv"),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "URL.to_struct_dict()",
        lambda: URL_HTTPS.to_struct_dict(),
        repeat=repeat, inner=100_000,
    ))

    return out


# ---------------------------------------------------------------------------
# Headers scenarios
# ---------------------------------------------------------------------------


def _headers_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "Headers() empty construct",
        lambda: Headers(),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "Headers(small dict)",
        lambda: Headers({
            "Content-Type": "application/json",
            "User-Agent": "yggdrasil/0.x",
            "Accept-Encoding": "gzip,deflate",
        }),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "Headers.from_(existing) identity",
        lambda: Headers.from_(HEADERS_SMALL),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "headers['Content-Type'] get",
        lambda: HEADERS_SMALL["Content-Type"],
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "'Content-Type' in headers",
        lambda: "Content-Type" in HEADERS_SMALL,
        repeat=repeat, inner=500_000,
    ))
    h = Headers(HEADERS_SMALL)
    out.append(_time_one(
        "headers['X-New'] = 'value' set",
        lambda: h.__setitem__("X-New", "value"),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "headers.copy() small",
        lambda: HEADERS_SMALL.copy(),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "headers.to_dict() small",
        lambda: HEADERS_SMALL.to_dict(),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "headers.byte_length cached",
        lambda: HEADERS_SMALL.byte_length,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "headers.xxh3_64 cached",
        lambda: HEADERS_SMALL.xxh3_64,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "headers.canonical_bytes cached",
        lambda: HEADERS_SMALL.canonical_bytes,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "headers.canonical_bytes 20-key cached",
        lambda: HEADERS_LARGE.canonical_bytes,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "Headers(HEADERS_SMALL) clone construct",
        lambda: Headers(HEADERS_SMALL),
        repeat=repeat, inner=200_000,
    ))

    return out


# ---------------------------------------------------------------------------
# BytesIO scenarios
# ---------------------------------------------------------------------------


def _bytesio_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "BytesIO(b'') empty construct",
        lambda: BytesIO(b""),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "BytesIO(small bytes) construct",
        lambda: BytesIO(PAYLOAD_SMALL),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "BytesIO(medium bytes) construct",
        lambda: BytesIO(PAYLOAD_MEDIUM),
        repeat=repeat, inner=10_000,
    ))

    def write_read_small():
        b = BytesIO(b"")
        b.write(PAYLOAD_SMALL)
        b.seek(0)
        return b.read()
    out.append(_time_one(
        "BytesIO write+seek+read 96B",
        write_read_small,
        repeat=repeat, inner=10_000,
    ))

    def write_read_medium():
        b = BytesIO(b"")
        b.write(PAYLOAD_MEDIUM)
        b.seek(0)
        return b.read()
    out.append(_time_one(
        "BytesIO write+seek+read 64KB",
        write_read_medium,
        repeat=repeat, inner=2_000,
    ))

    def write_read_large():
        b = BytesIO(b"")
        b.write(PAYLOAD_LARGE)
        b.seek(0)
        return b.read()
    out.append(_time_one(
        "BytesIO write+seek+read 1MB",
        write_read_large,
        repeat=repeat, inner=200,
    ))

    return out


# ---------------------------------------------------------------------------
# Parameters scenarios
# ---------------------------------------------------------------------------


def _parameters_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "anonymize_parameters(flat dict) remove",
        lambda: anonymize_parameters(PARAMS_FLAT, mode="remove"),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "anonymize_parameters(flat dict) redact",
        lambda: anonymize_parameters(PARAMS_FLAT, mode="redact"),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "anonymize_parameters(nested) redact",
        lambda: anonymize_parameters(PARAMS_NESTED, mode="redact"),
        repeat=repeat, inner=20_000,
    ))

    return out


# ---------------------------------------------------------------------------
# Aggregator + CLI
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    return [
        *_url_scenarios(repeat),
        *_headers_scenarios(repeat),
        *_bytesio_scenarios(repeat),
        *_parameters_scenarios(repeat),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<58s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
