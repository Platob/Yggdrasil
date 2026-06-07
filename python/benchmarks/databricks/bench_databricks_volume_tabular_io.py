"""Tabular (Arrow/Parquet) IO + concurrent load test on :class:`VolumePath`,
direct-S3 vs the Databricks Files REST API, live.

Stands up a real EXTERNAL Unity Catalog volume rooted at a writable S3 prefix
and exercises the *tabular* surface — :meth:`VolumePath.write_table` /
:meth:`VolumePath.read_arrow_table` (Parquet) — both ways:

* **direct** — external-usable (``EXTERNAL USE SCHEMA``); ``write_table`` /
  ``read_arrow_table`` stream straight to/from the backing S3 bucket via
  ``VolumePath.storage_path``.
* **files** — the same ops forced through the Files REST API
  (``volume.mark_external_denied`` disables the fast path).

Two workloads:

1. **throughput** — write then read N standalone Parquet tables; reports
   rows/s, MB/s, wall time, and Databricks Files-API (governance) round trips.
2. **load** — *bombard* the volume with ``--load-ops`` write+read+verify
   round trips across ``--load-workers`` threads; reports ops/s, latency
   percentiles, error count, and governance round trips. This is the
   stress / soak side — crank ``--load-workers`` / ``--load-ops`` up.

Requires the ``aws`` extra (botocore) and a live workspace whose identity can
``CREATE EXTERNAL VOLUME`` + be granted ``EXTERNAL USE SCHEMA``.

Usage::

    DATABRICKS_HOST=... DATABRICKS_TOKEN=... \\
    uv run --extra dev --extra aws python \\
      benchmarks/databricks/bench_databricks_volume_tabular_io.py \\
      --base s3://your-bucket/3mv/ygg --catalog trading_tgp_dev --schema ygg_integration \\
      --rows 50000 --tables 6 --repeat 3 --load-workers 24 --load-ops 240
"""
from __future__ import annotations

import argparse
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.enums import Mode

# ── Files-API (governance) round-trip counter (thread-safe) ──────────────────
_FS_LOCK = threading.Lock()
_FS_CALLS = {"n": 0}
_ORIG_FS_REQUEST = VolumePath._fs_request


def _counting_fs_request(self, *a, **kw):
    with _FS_LOCK:
        _FS_CALLS["n"] += 1
    return _ORIG_FS_REQUEST(self, *a, **kw)


VolumePath._fs_request = _counting_fs_request


def _reset_calls() -> None:
    with _FS_LOCK:
        _FS_CALLS["n"] = 0


def _calls() -> int:
    with _FS_LOCK:
        return _FS_CALLS["n"]


def _set_mode(volume, *, direct: bool) -> None:
    """Force the volume's per-mode external access for both read and write."""
    if direct:
        volume._external_readable = None      # undetermined → resolve to direct
        volume._external_writable = None
        # Pre-resolve so the concurrent load doesn't thundering-herd the
        # eligibility check on first access.
        volume.external_storage_root(write=True)
    else:
        volume.mark_external_denied(write=False)
        volume.mark_external_denied(write=True)


def _make_table(rows: int) -> pa.Table:
    """A mixed-type table — int64 key, string, float64, timestamp."""
    base = pa.array(range(rows), pa.int64())
    return pa.table({
        "id": base,
        "label": pa.array([f"row-{i:08d}" for i in range(rows)]),
        "value": pa.array([i * 1.5 for i in range(rows)], pa.float64()),
        "ts": pa.array([1_700_000_000_000 + i for i in range(rows)], pa.int64())
        .cast(pa.timestamp("ms", tz="UTC")),
    })


def _pct(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    k = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.environ.get("YGG_TEST_EXTERNAL_LOCATION"),
                    help="writable s3:// base prefix (or set YGG_TEST_EXTERNAL_LOCATION)")
    ap.add_argument("--catalog", default=os.environ.get(
        "DATABRICKS_INTEGRATION_CATALOG", "trading_tgp_dev"))
    ap.add_argument("--schema", default="ygg_integration")
    ap.add_argument("--rows", type=int, default=50_000, help="rows per throughput table")
    ap.add_argument("--tables", type=int, default=6, help="standalone tables (throughput)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--load-workers", type=int, default=24, help="concurrent threads (load)")
    ap.add_argument("--load-ops", type=int, default=240, help="write+read round trips (load)")
    ap.add_argument("--load-rows", type=int, default=500, help="rows per load table")
    args = ap.parse_args()
    if not args.base:
        ap.error("--base or YGG_TEST_EXTERNAL_LOCATION required")

    tag = secrets.token_hex(4)
    client = DatabricksClient()
    me = client.workspace_client().current_user.me().user_name

    schema = client.schemas.schema(catalog_name=args.catalog, schema_name=args.schema)
    schema.get_or_create()
    try:
        client.sql.execute(
            f"GRANT EXTERNAL USE SCHEMA ON SCHEMA {args.catalog}.{args.schema} TO `{me}`")
    except Exception as exc:  # noqa: BLE001
        print(f"(grant skipped: {exc})")

    volume = client.volumes(catalog_name=args.catalog, schema_name=args.schema).volume(
        f"ygg_tab_{tag}")
    volume.get_or_create(volume_type="EXTERNAL",
                         storage_location=f"{args.base.rstrip('/')}/tab/{tag}")
    print(f"external volume: {volume.full_name()} -> {volume.storage_location}")

    if volume.path("_probe.parquet").storage_path(write=True) is None:
        print("DIRECT storage path unavailable (no grant / non-s3 / no botocore) — abort.")
        _cleanup(volume)
        return

    try:
        _throughput(volume, args, tag)
        _load(volume, args, tag)
    finally:
        _cleanup(volume)


# --------------------------------------------------------------------------- #
# 1. Tabular throughput — write then read N standalone Parquet tables
# --------------------------------------------------------------------------- #
def _throughput(volume, args, tag: str) -> None:
    table = _make_table(args.rows)
    mb = (table.nbytes * args.tables) / (1024 * 1024)
    total_rows = args.rows * args.tables

    print(
        f"\n== Tabular throughput — {args.tables} tables x {args.rows} rows "
        f"(~{mb:.1f} MiB Arrow), best of {args.repeat} ==")
    rows = {}
    for direct in (False, True):
        label = "direct" if direct else "files"
        _set_mode(volume, direct=direct)
        paths = [volume.path(f"tput_{label}_{tag}/t{i}.parquet") for i in range(args.tables)]

        best_w = float("inf")
        best_r = float("inf")
        w_calls = 0
        r_calls = 0
        for rep in range(args.repeat):
            _reset_calls()
            t0 = time.perf_counter()
            for p in paths:
                p.write_table(table, mode=Mode.OVERWRITE)
            dw = time.perf_counter() - t0
            if rep == 0:
                w_calls = _calls()
            best_w = min(best_w, dw)

            _reset_calls()
            t0 = time.perf_counter()
            for p in paths:
                got = p.read_arrow_table().num_rows
                assert got == args.rows, f"{got} != {args.rows}"
            dr = time.perf_counter() - t0
            if rep == 0:
                r_calls = _calls()
            best_r = min(best_r, dr)
        rows[label] = {
            "write_ms": best_w * 1000, "read_ms": best_r * 1000,
            "write_calls": w_calls, "read_calls": r_calls,
            "write_mbs": mb / best_w, "read_mbs": mb / best_r,
            "write_rps": total_rows / best_w, "read_rps": total_rows / best_r,
        }

    hdr = f"{'mode':<8}{'write ms':>10}{'read ms':>10}{'write MB/s':>12}{'read MB/s':>12}{'wr rows/s':>12}{'rd rows/s':>12}{'API':>6}"
    print(hdr)
    print("-" * len(hdr))
    for label in ("files", "direct"):
        r = rows[label]
        print(f"{label:<8}{r['write_ms']:>10.1f}{r['read_ms']:>10.1f}"
              f"{r['write_mbs']:>12.1f}{r['read_mbs']:>12.1f}"
              f"{r['write_rps']:>12.0f}{r['read_rps']:>12.0f}"
              f"{r['write_calls'] + r['read_calls']:>6}")
    f, d = rows["files"], rows["direct"]
    print(f"\nspeedup  write {f['write_ms'] / d['write_ms']:.2f}x   "
          f"read {f['read_ms'] / d['read_ms']:.2f}x   |   governance calls "
          f"files={f['write_calls'] + f['read_calls']}  direct={d['write_calls'] + d['read_calls']}")


# --------------------------------------------------------------------------- #
# 2. Load test — bombard the volume with concurrent write+read+verify
# --------------------------------------------------------------------------- #
def _load(volume, args, tag: str) -> None:
    table = _make_table(args.load_rows)
    print(
        f"\n== Load test — bombarding {args.load_ops} write+read round trips "
        f"across {args.load_workers} threads ({args.load_rows} rows each) ==")

    for direct in (False, True):
        label = "direct" if direct else "files"
        _set_mode(volume, direct=direct)

        latencies: list[float] = []
        errors: list[str] = []
        lat_lock = threading.Lock()

        def _op(i: int) -> None:
            p = volume.path(f"load_{label}_{tag}/w{i:05d}.parquet")
            t0 = time.perf_counter()
            try:
                p.write_table(table, mode=Mode.OVERWRITE)
                n = p.read_arrow_table().num_rows
                if n != args.load_rows:
                    raise AssertionError(f"row count {n} != {args.load_rows}")
            except Exception as exc:  # noqa: BLE001
                with lat_lock:
                    errors.append(f"{type(exc).__name__}: {exc}")
                return
            with lat_lock:
                latencies.append((time.perf_counter() - t0) * 1000)

        _reset_calls()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.load_workers,
                                thread_name_prefix=f"load-{label}") as pool:
            list(as_completed(pool.submit(_op, i) for i in range(args.load_ops)))
        wall = time.perf_counter() - t0
        calls = _calls()
        ok = len(latencies)

        print(f"\n[{label}]  {ok}/{args.load_ops} ok, {len(errors)} errors  "
              f"in {wall:.2f}s")
        print(f"  throughput : {ok / wall:.1f} ops/s  "
              f"({2 * ok / wall:.1f} req/s incl. read+write)")
        if latencies:
            print(f"  latency ms : p50 {_pct(latencies, .5):.1f}  "
                  f"p95 {_pct(latencies, .95):.1f}  p99 {_pct(latencies, .99):.1f}  "
                  f"max {max(latencies):.1f}")
        print(f"  governance : {calls} Files-API round trips "
              f"({calls / max(ok, 1):.2f}/op)")
        if errors:
            print(f"  first errors: {errors[:3]}")


def _cleanup(volume) -> None:
    try:
        sp = volume.storage_path(mode=Mode.AUTO)
        if sp is not None:
            sp.remove(recursive=True, missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"(storage cleanup: {exc})")
    try:
        volume.delete()
    except Exception as exc:  # noqa: BLE001
        print(f"(volume delete: {exc})")


if __name__ == "__main__":
    main()
