"""Benchmark :class:`RemotePath` construction + ``Path`` surface ops.

Runs against :class:`S3Path` with a stubbed boto client — no real
network. Covers the costs every remote backend pays before it
issues a single HTTP request: singleton-cache lookup,
``__init__`` URL parsing, parent / joinpath traversal, ``_from_url``
sibling construction (the inner loop of ``_ls`` listing), and
mocked ``_stat`` / ``_read_mv`` round-trips so the cache-warm path
is measurable too.

The companion :mod:`bench_io_remote` script measures real
network behavior; this one measures the in-process overhead and
runs in CI.

Usage::

    PYTHONPATH=src python benchmarks/io/path/bench_remote_path.py --repeat 3
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable
from unittest.mock import Mock

from yggdrasil.aws.fs.path import S3Path
from yggdrasil.aws.fs.service import S3Service
from yggdrasil.io.path.remote_path import RemotePath
from yggdrasil.io.url import URL


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
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
        scale, unit = 1e9, "ns"
    elif r["best"] >= 1e-3:
        scale, unit = 1e3, "ms"
    extra = ""
    if "calls" in r:
        extra = f"  sdk_calls={r['calls']:>3d}"
    return (
        f"{r['label']:<62s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}{extra}"
    )


# ---------------------------------------------------------------------------
# Mock client builders — return canned boto-shaped responses
# ---------------------------------------------------------------------------


def _stub_service(body_bytes: bytes = b"") -> Mock:
    """Return a mock :class:`S3Service` whose ``boto_client`` is a
    canned boto-shaped mock.

    :class:`S3Path` reaches the boto client through
    ``self.service.boto_client`` — wiring a service-shaped mock keeps
    the construction path independent of real AWS credentials.
    """
    svc = Mock(spec=S3Service)
    svc.boto_client = _stub_client(body_bytes)
    return svc


def _stub_client(body_bytes: bytes = b"") -> Mock:
    client = Mock()
    client.head_object.return_value = {
        "ContentLength": len(body_bytes),
        "ContentType": "application/octet-stream",
        "LastModified": _LastModified(),
    }
    client.get_object.return_value = {
        "Body": _ReadOnce(body_bytes),
        "ContentRange": f"bytes 0-{max(0, len(body_bytes) - 1)}/{len(body_bytes)}",
        "ContentType": "application/octet-stream",
        "LastModified": _LastModified(),
    }
    client.put_object.return_value = {}
    client.delete_object.return_value = {}
    client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": f"prefix/file-{i:04d}.parquet", "Size": 1024}
            for i in range(32)
        ],
        "KeyCount": 32,
    }
    paginator = Mock()
    paginator.paginate.return_value = iter([client.list_objects_v2.return_value])
    client.get_paginator.return_value = paginator
    return client


class _LastModified:
    def timestamp(self) -> float:  # mirror boto's datetime surface
        return 1.7e9


class _ReadOnce:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._read = False

    def read(self) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._payload

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    service = _stub_service(b"x" * 4096)

    # Singleton cache — same URL every call.
    RemotePath._INSTANCES.clear()
    S3Path("s3://bucket/key.parquet", service=service)  # prime
    out.append(_time_one(
        "S3Path(str) singleton hit",
        lambda: S3Path("s3://bucket/key.parquet", service=service),
        repeat=repeat, inner=20_000,
    ))

    url_obj = URL.from_("s3://bucket/key.parquet")
    out.append(_time_one(
        "S3Path(url=URL) singleton hit",
        lambda: S3Path(url=url_obj, service=service),
        repeat=repeat, inner=50_000,
    ))

    # Singleton miss — fresh URL each call. Pre-build a pool so the
    # construction cost is the *new instance* cost, not the URL parse.
    fresh_urls = [
        URL(scheme="s3", host="bucket", path=f"/k{i:06d}.parquet")
        for i in range(50_000)
    ]
    fresh_idx = [0]

    def construct_fresh() -> None:
        RemotePath._INSTANCES.clear()
        u = fresh_urls[fresh_idx[0] % len(fresh_urls)]
        fresh_idx[0] += 1
        S3Path(url=u, service=service)

    out.append(_time_one(
        "S3Path(url=fresh) singleton miss",
        construct_fresh,
        repeat=repeat, inner=10_000,
    ))

    # Traversal — parent walk + joinpath. Hot in folder readers,
    # Delta replay, partition discovery.
    RemotePath._INSTANCES.clear()
    deep = S3Path("s3://bucket/a/b/c/d/e/file.parquet", service=service)
    out.append(_time_one(
        "S3Path.parent (5 levels)",
        lambda: _walk_parents(deep, 5),
        repeat=repeat, inner=5_000,
    ))

    base = S3Path("s3://bucket/base/", service=service)
    out.append(_time_one(
        "S3Path.joinpath('sub','file.csv')",
        lambda: base.joinpath("sub", "file.csv"),
        repeat=repeat, inner=20_000,
    ))

    # ``_from_url`` is the inner loop of every ``_ls`` listing. Using
    # the URL._replace_path fast path keeps the cost in the construction
    # branch, not URL parsing.
    listing_url = base.url
    out.append(_time_one(
        "S3Path._from_url(replace_path)",
        lambda: base._from_url(listing_url._replace_path("/base/sub/file.parquet")),
        repeat=repeat, inner=20_000,
    ))

    # Stat — warm cache hit (post-priming round-trip).
    primed = S3Path("s3://bucket/key.parquet", service=service)
    primed.exists()  # prime cache
    out.append(_time_one(
        "S3Path.exists() (cache warm)",
        lambda: primed.exists(),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "S3Path.size (cache warm)",
        lambda: primed.size,
        repeat=repeat, inner=50_000,
    ))

    # Stat — cold (single HeadObject round-trip via mock).
    def stat_cold() -> None:
        primed.invalidate_singleton(remove_global=False)
        primed.exists()

    out.append(_time_one(
        "S3Path.exists() (cache cold→warm)",
        stat_cold,
        repeat=repeat, inner=10_000,
    ))

    # _read_mv — bench a small ranged GET through the mock so the
    # ContentRange seed path in :meth:`_read_mv` runs.
    primed.invalidate_singleton(remove_global=False)
    out.append(_time_one(
        "S3Path._read_mv(4096) via mock GetObject",
        lambda: primed._read_mv(4096, 0),
        repeat=repeat, inner=5_000,
    ))

    # Listing — every page hit yields N children; each child pays
    # ``_make_child`` + ``_from_url``. The mock returns 32 children
    # per page.
    listing_root = S3Path("s3://bucket/prefix/", service=_stub_service())
    out.append(_time_one(
        "S3Path._ls (32 children per page)",
        lambda: _exhaust(listing_root._ls(recursive=False)),
        repeat=repeat, inner=2_000,
    ))

    # -------------------------------------------------------------------
    # Write SDK-call profiles
    # -------------------------------------------------------------------

    write_payload = b"x" * 8192

    # write_all — truncate(0) + write_bytes
    service = _stub_service(b"old-content")
    RemotePath._INSTANCES.clear()
    p = S3Path("s3://bucket/out.bin", service=service)
    client = service.boto_client
    client.put_object.reset_mock()
    client.get_object.reset_mock()
    client.head_object.reset_mock()
    client.delete_object.reset_mock()
    p.write_all(write_payload)
    calls = sum(
        m.call_count for m in (
            client.put_object, client.get_object,
            client.head_object, client.delete_object,
        )
    )
    out.append({
        "label": "S3Path.write_all(8 KiB) — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": calls,
    })

    # write_bytes — single put
    service = _stub_service(b"old-content")
    RemotePath._INSTANCES.clear()
    p = S3Path("s3://bucket/out.bin", service=service)
    client = service.boto_client
    client.put_object.reset_mock()
    client.get_object.reset_mock()
    client.head_object.reset_mock()
    p.write_bytes(write_payload)
    calls = sum(
        m.call_count for m in (
            client.put_object, client.get_object,
            client.head_object,
        )
    )
    out.append({
        "label": "S3Path.write_bytes(8 KiB) — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": calls,
    })

    # open("wb") + write + close — cursor path
    service = _stub_service(b"old-content")
    RemotePath._INSTANCES.clear()
    p = S3Path("s3://bucket/out.bin", service=service)
    client = service.boto_client
    client.put_object.reset_mock()
    client.get_object.reset_mock()
    client.head_object.reset_mock()
    client.delete_object.reset_mock()
    with p.open("wb") as f:
        f.write(write_payload)
    calls = sum(
        m.call_count for m in (
            client.put_object, client.get_object,
            client.head_object, client.delete_object,
        )
    )
    out.append({
        "label": "S3Path open('wb') write close — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": calls,
    })

    # Write timings
    out.append(_time_one(
        "S3Path.write_all(8 KiB)",
        lambda: _write_s3(write_payload),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "S3Path open('wb') write close",
        lambda: _write_s3_cursor(write_payload),
        repeat=repeat, inner=2_000,
    ))

    RemotePath._INSTANCES.clear()
    return out


def _write_s3(payload: bytes) -> None:
    service = _stub_service()
    RemotePath._INSTANCES.clear()
    S3Path("s3://bucket/out.bin", service=service).write_all(payload)


def _write_s3_cursor(payload: bytes) -> None:
    service = _stub_service()
    RemotePath._INSTANCES.clear()
    p = S3Path("s3://bucket/out.bin", service=service)
    with p.open("wb") as f:
        f.write(payload)


def _walk_parents(p, n: int):
    for _ in range(n):
        p = p.parent


def _exhaust(it) -> None:
    for _ in it:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<62s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
