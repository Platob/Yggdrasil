"""Benchmark MIME / MediaType inference + content sniffing.

What this covers
----------------

This drills into the surface every Holder, URL and IO open hits when
they need to figure out "what format is this byte stream?":

* :meth:`MimeType.from_str` — the path / extension / mime-value fast
  path. Run per URL on every routing decision.
* :meth:`MimeType.from_magic` — outer-magic byte sniff over bytes,
  ``BytesIO``, and yggdrasil's own ``BytesIO`` (the latter is what
  :meth:`MediaType.from_io` and ``Holder._media_type`` end up driving).
* :meth:`MimeType.parse_many` — Accept-header / extension-chain
  resolution (e.g. ``"trades.parquet.zst"``, ``"application/json,
  text/csv;q=0.8"``).
* :meth:`MediaType.from_url` — extension chain → MediaType resolution
  (the fast path on every ``Holder.media_type`` lazy resolve).
* :meth:`MediaType.from_magic` — two-stage bytes sniff (outer magic +
  inner codec decompression head).
* :meth:`MediaType.from_io` — same as ``from_magic`` but cursor-managed
  on a file-like. This is the Holder hot path: gzip-wrapped CSV from
  S3 / Azure / Databricks Volumes lands here.

Skips disk + network round-trips on purpose — we want to measure the
sniffing overhead itself, not block storage latency. ``BytesIO``
fixtures stand in for the file-like that real Holders pass through.

Usage::

    PYTHONPATH=src python benchmarks/data/enums/bench_mime_media.py
    PYTHONPATH=src python benchmarks/data/enums/bench_mime_media.py --repeat 7
"""
from __future__ import annotations

import argparse
import io as _stdio
import statistics
import time
from typing import Callable

from yggdrasil.enums.codec import GZIP, ZSTD
from yggdrasil.enums.media_type import MediaType
from yggdrasil.enums.mime_type import MimeType, MimeTypes
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.url import URL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Representative bytes payloads — match the first 64+ bytes of a real
# file of each type so :meth:`MimeType.from_magic` resolves through
# the same code path it does on disk-loaded data.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 56
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 56
PDF_BYTES = b"%PDF-1.7\n" + b"\x00" * 56
PARQUET_BYTES = b"PAR1" + b"\x00" * 60
ARROW_BYTES = b"ARROW1\x00\x00" + b"\x00" * 56
ORC_BYTES = b"ORC\x12" + b"\x00" * 56
NUMPY_BYTES = b"\x93NUMPY\x01\x00v\x00" + b"\x00" * 56
SQLITE_BYTES = b"SQLite format 3\x00" + b"\x00" * 56
ZIP_BYTES = b"PK\x03\x04" + b"\x00" * 60
JSON_BYTES = b'{"hello": "world", "count": 1, "tags": ["a","b","c"]}'
NDJSON_BYTES = b'{"a":1}\n{"a":2}\n{"a":3}\n{"a":4}\n{"a":5}\n{"a":6}\n'
XML_BYTES = b'<?xml version="1.0"?><root><a/></root>           '
UNKNOWN_BYTES = b"completely-unrecognised-payload-bytes-no-magic-prefix-matches"

# Codec-wrapped payloads — the gzip/zstd byte streams MediaType.from_io
# unwraps to discover the inner format.
_PAYLOAD = (
    b"name,timestamp,value\n"
    b"AAPL,2024-01-01T00:00:00Z,189.42\n"
    b"AAPL,2024-01-01T00:01:00Z,189.51\n"
    b"AAPL,2024-01-01T00:02:00Z,189.38\n"
    b"AAPL,2024-01-01T00:03:00Z,189.27\n"
) * 32
GZIP_CSV = GZIP.compress_bytes(_PAYLOAD)
ZSTD_CSV = ZSTD.compress_bytes(_PAYLOAD)
GZIP_PARQUET = GZIP.compress_bytes(b"PAR1" + b"\x00" * 1024 + _PAYLOAD)
ZSTD_PARQUET = ZSTD.compress_bytes(b"PAR1" + b"\x00" * 1024 + _PAYLOAD)


URL_PARQUET = URL.from_str("s3://bucket/datasets/year=2024/part-00000.parquet")
URL_CSV_GZ = URL.from_str("s3://bucket/trades/2024-01/trades.csv.gz")
URL_PARQUET_ZST = URL.from_str("s3://bucket/trades/2024-01/trades.parquet.zst")
URL_NDJSON = URL.from_str("https://api.example.com/events.ndjson")
URL_DOTLESS = URL.from_str("https://api.example.com/v1/accounts/12345/transactions")

ACCEPT_HEADER = "application/json, text/csv;q=0.8, application/vnd.apache.parquet;q=0.5"
COMPOSITE_STR = "application/csv+gzip"
EXTENSION_CHAIN = "trades.parquet.zst"


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
    scale, unit = 1e6, "us"
    if r["best"] < 1e-6:
        scale, unit = 1e9, "ns"
    elif r["best"] >= 1e-3:
        scale, unit = 1e3, "ms"
    return (
        f"{r['label']:<62s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# MimeType.from_str — extension / mime-value / path-shaped strings
# ---------------------------------------------------------------------------


def _mime_from_str_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    out.append(_time_one(
        "MimeType.from_str('csv') bare ext",
        lambda: MimeType.from_str("csv"),
        repeat=repeat, inner=300_000,
    ))
    out.append(_time_one(
        "MimeType.from_str('.parquet') dot ext",
        lambda: MimeType.from_str(".parquet"),
        repeat=repeat, inner=300_000,
    ))
    out.append(_time_one(
        "MimeType.from_str('text/csv') mime value",
        lambda: MimeType.from_str("text/csv"),
        repeat=repeat, inner=300_000,
    ))
    out.append(_time_one(
        "MimeType.from_str('application/vnd.apache.parquet')",
        lambda: MimeType.from_str("application/vnd.apache.parquet"),
        repeat=repeat, inner=300_000,
    ))
    out.append(_time_one(
        "MimeType.from_str('/data/trades.csv') path",
        lambda: MimeType.from_str("/data/trades.csv"),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "MimeType.from_str('unknown', default=None) miss",
        lambda: MimeType.from_str("not-a-real-mime", default=None),
        repeat=repeat, inner=200_000,
    ))
    return out


# ---------------------------------------------------------------------------
# MimeType.from_magic — bytes / BytesIO / yggdrasil BytesIO
# ---------------------------------------------------------------------------


def _mime_from_magic_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # Bytes head — the cheapest shape, what Holders pass when they
    # have already buffered the payload.
    for label, head in [
        ("png", PNG_BYTES),
        ("jpeg", JPEG_BYTES),
        ("pdf", PDF_BYTES),
        ("parquet", PARQUET_BYTES),
        ("arrow_ipc", ARROW_BYTES),
        ("orc", ORC_BYTES),
        ("numpy", NUMPY_BYTES),
        ("sqlite", SQLITE_BYTES),
        ("zip", ZIP_BYTES),
        ("json (structural)", JSON_BYTES),
        ("ndjson (structural)", NDJSON_BYTES),
        ("xml (structural)", XML_BYTES),
        ("unknown miss", UNKNOWN_BYTES),
    ]:
        out.append(_time_one(
            f"MimeType.from_magic(bytes, {label})",
            lambda h=head: MimeType.from_magic(h, default=None),
            repeat=repeat, inner=100_000,
        ))

    # stdlib BytesIO — fh.tell / seek / read(64) / seek-back loop.
    stdio_buf = _stdio.BytesIO(PARQUET_BYTES)
    out.append(_time_one(
        "MimeType.from_magic(io.BytesIO, parquet)",
        lambda: MimeType.from_magic(stdio_buf, default=None),
        repeat=repeat, inner=50_000,
    ))

    # yggdrasil's own BytesIO — :meth:`pread(64, 0)`, no cursor save.
    # IO is a Disposable; open() it so the bench probe doesn't pay
    # acquire/release on every call.
    ygg_buf = BytesIO(PARQUET_BYTES).open()
    try:
        out.append(_time_one(
            "MimeType.from_magic(ygg BytesIO, parquet)",
            lambda: MimeType.from_magic(ygg_buf, default=None),
            repeat=repeat, inner=50_000,
        ))
    finally:
        ygg_buf.close()

    return out


# ---------------------------------------------------------------------------
# MimeType.parse_many — Accept-header / composite / extension chain
# ---------------------------------------------------------------------------


def _mime_parse_many_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    out.append(_time_one(
        "MimeType.parse_many(Accept-header) 3 parts",
        lambda: MimeType.parse_many(ACCEPT_HEADER),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "MimeType.parse_many('application/csv+gzip')",
        lambda: MimeType.parse_many(COMPOSITE_STR),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "MimeType.parse_many('trades.parquet.zst')",
        lambda: MimeType.parse_many(EXTENSION_CHAIN),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "MimeType.parse_many(['csv', 'gzip']) list",
        lambda: MimeType.parse_many(["csv", "gzip"]),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "MimeType.parse_many('text/*') wildcard",
        lambda: MimeType.parse_many("text/*"),
        repeat=repeat, inner=20_000,
    ))
    return out


# ---------------------------------------------------------------------------
# MediaType.from_url / from_many — Holder.media_type fast path
# ---------------------------------------------------------------------------


def _media_from_url_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    out.append(_time_one(
        "MediaType.from_url(s3://…/part.parquet)",
        lambda: MediaType.from_url(URL_PARQUET, default=None),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "MediaType.from_url(s3://…/trades.csv.gz)",
        lambda: MediaType.from_url(URL_CSV_GZ, default=None),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "MediaType.from_url(s3://…/trades.parquet.zst)",
        lambda: MediaType.from_url(URL_PARQUET_ZST, default=None),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "MediaType.from_url(https://…/events.ndjson)",
        lambda: MediaType.from_url(URL_NDJSON, default=None),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "MediaType.from_url(extensionless) miss",
        lambda: MediaType.from_url(URL_DOTLESS, default=None),
        repeat=repeat, inner=100_000,
    ))

    # Direct from_many — the inner call inside from_url.
    csv_zst = URL_PARQUET_ZST.extensions
    out.append(_time_one(
        "MediaType.from_many(['parquet','zst'])",
        lambda: MediaType.from_many(csv_zst, default=None),
        repeat=repeat, inner=100_000,
    ))
    return out


# ---------------------------------------------------------------------------
# MediaType.from_magic — two-stage bytes sniff (outer + codec head)
# ---------------------------------------------------------------------------


def _media_from_magic_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    out.append(_time_one(
        "MediaType.from_magic(parquet bytes)",
        lambda: MediaType.from_magic(PARQUET_BYTES, default=None),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "MediaType.from_magic(arrow bytes)",
        lambda: MediaType.from_magic(ARROW_BYTES, default=None),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "MediaType.from_magic(json bytes)",
        lambda: MediaType.from_magic(JSON_BYTES, default=None),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "MediaType.from_magic(gzip-csv bytes) 2-stage",
        lambda: MediaType.from_magic(GZIP_CSV, default=None),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "MediaType.from_magic(zstd-csv bytes) 2-stage",
        lambda: MediaType.from_magic(ZSTD_CSV, default=None),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "MediaType.from_magic(gzip-parquet bytes) 2-stage",
        lambda: MediaType.from_magic(GZIP_PARQUET, default=None),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "MediaType.from_magic(zstd-parquet bytes) 2-stage",
        lambda: MediaType.from_magic(ZSTD_PARQUET, default=None),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "MediaType.from_magic(unknown) miss",
        lambda: MediaType.from_magic(UNKNOWN_BYTES, default=None),
        repeat=repeat, inner=50_000,
    ))
    return out


# ---------------------------------------------------------------------------
# MediaType.from_io — cursor-managed sniff over a file-like (Holder path)
# ---------------------------------------------------------------------------


def _media_from_io_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # stdlib BytesIO — what Holder.read_buffer / cursor surfaces feed us.
    for label, payload in [
        ("parquet", b"PAR1" + b"\x00" * 1024),
        ("arrow_ipc", b"ARROW1\x00\x00" + b"\x00" * 1024),
        ("png", PNG_BYTES + b"\x00" * 1024),
        ("json", JSON_BYTES + b"\n" + JSON_BYTES),
        ("gzip-csv (2-stage)", GZIP_CSV),
        ("zstd-csv (2-stage)", ZSTD_CSV),
        ("gzip-parquet (2-stage)", GZIP_PARQUET),
        ("zstd-parquet (2-stage)", ZSTD_PARQUET),
        ("unknown", UNKNOWN_BYTES + b"\x00" * 64),
    ]:
        buf = _stdio.BytesIO(payload)
        out.append(_time_one(
            f"MediaType.from_io(io.BytesIO, {label})",
            lambda b=buf: (b.seek(0), MediaType.from_io(b, default=None))[1],
            repeat=repeat, inner=20_000,
        ))

    return out


# ---------------------------------------------------------------------------
# Aggregator + CLI
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    out += _mime_from_str_scenarios(repeat)
    out += _mime_from_magic_scenarios(repeat)
    out += _mime_parse_many_scenarios(repeat)
    out += _media_from_url_scenarios(repeat)
    out += _media_from_magic_scenarios(repeat)
    out += _media_from_io_scenarios(repeat)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<62s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
