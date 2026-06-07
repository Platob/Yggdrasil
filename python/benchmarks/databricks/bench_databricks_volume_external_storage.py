"""Benchmark VolumePath direct-storage (S3) vs the Files API, live.

Stands up a real EXTERNAL Unity Catalog volume rooted at a writable S3
prefix and times the VolumePath operations both ways:

* **direct** — the volume is external-usable (``EXTERNAL USE SCHEMA``), so the
  Holder primitives + stat/delete short-circuit straight to the backing S3
  bucket (``VolumePath.storage_path``).
* **files** — the same ops forced through the Databricks Files REST API
  (``volume.mark_external_denied`` disables the fast path).

Same bytes, same paths — the delta is the transport. Reports per-op wall time,
the direct/files speedup, **and the number of Databricks Files-API (governance)
round trips each mode makes** — the direct path should drive that toward zero.

Requires the ``aws`` extra (botocore, for the credential refresher) and a
live workspace whose identity can ``CREATE EXTERNAL VOLUME`` + be granted
``EXTERNAL USE SCHEMA`` on the target schema.

Usage::

    DATABRICKS_HOST=... DATABRICKS_TOKEN=... \\
    uv run --extra dev --extra aws python \\
      benchmarks/databricks/bench_databricks_volume_external_storage.py \\
      --base s3://your-bucket/3mv/ygg \\
      --catalog trading_tgp_dev --schema ygg_bench \\
      --size-kib 256 --files 8 --repeat 3
"""
from __future__ import annotations

import argparse
import os
import secrets
import statistics
import time

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.enums import Mode

# ── Files-API (governance) round-trip counter ───────────────────────────────
# Wrap ``VolumePath._fs_request`` so every Databricks Files REST call (HEAD /
# GET / PUT / DELETE / list) bumps a counter. The direct-S3 path should make
# *zero*; the Files path pays one or more per op.
_FS_CALLS = {"n": 0}
_ORIG_FS_REQUEST = VolumePath._fs_request


def _counting_fs_request(self, *a, **kw):
    _FS_CALLS["n"] += 1
    return _ORIG_FS_REQUEST(self, *a, **kw)


VolumePath._fs_request = _counting_fs_request


def _timed(fn, repeat: int) -> float:
    """Best (min) wall time over *repeat* runs, in milliseconds."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def _timed_counted(fn, repeat: int) -> "tuple[float, int]":
    """``(best wall ms, Files-API calls on the first run)`` for *fn*."""
    _FS_CALLS["n"] = 0
    t0 = time.perf_counter()
    fn()
    first_ms = (time.perf_counter() - t0) * 1000.0
    calls = _FS_CALLS["n"]
    best = first_ms
    for _ in range(repeat - 1):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best, calls


def _set_mode(volume, *, direct: bool) -> None:
    """Force the volume's per-mode external access for both read and write."""
    if direct:
        volume._external_readable = None      # undetermined → resolve to direct
        volume._external_writable = None
    else:
        volume.mark_external_denied(write=False)
        volume.mark_external_denied(write=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.environ.get("YGG_TEST_EXTERNAL_LOCATION"),
                    help="writable s3:// base prefix (or set YGG_TEST_EXTERNAL_LOCATION)")
    ap.add_argument("--catalog", default=os.environ.get(
        "DATABRICKS_INTEGRATION_CATALOG", "trading_tgp_dev"))
    ap.add_argument("--schema", default="ygg_bench")
    ap.add_argument("--size-kib", type=int, default=256)
    ap.add_argument("--files", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    if not args.base:
        ap.error("--base or YGG_TEST_EXTERNAL_LOCATION required")

    payload = os.urandom(args.size_kib * 1024)
    tag = secrets.token_hex(4)
    client = DatabricksClient()
    me = client.workspace_client().current_user.me().user_name

    # Ensure the schema grants EXTERNAL USE SCHEMA (the direct-path prerequisite).
    schema = client.schemas.schema(catalog_name=args.catalog, schema_name=args.schema)
    schema.get_or_create()
    try:
        client.sql.execute(
            f"GRANT EXTERNAL USE SCHEMA ON SCHEMA {args.catalog}.{args.schema} TO `{me}`")
    except Exception as exc:  # noqa: BLE001
        print(f"(grant skipped: {exc})")

    volume = client.volumes(catalog_name=args.catalog, schema_name=args.schema).volume(
        f"ygg_bench_{tag}")
    volume.get_or_create(volume_type="EXTERNAL",
                         storage_location=f"{args.base.rstrip('/')}/bench/{tag}")
    print(f"external volume: {volume.full_name()} -> {volume.storage_location}")

    # Confirm the direct path is actually reachable before benchmarking it.
    probe = volume.path("_probe.bin")
    if probe.storage_path(write=True) is None:
        print("DIRECT storage path unavailable (no grant / non-s3 / no botocore) — abort.")
        _cleanup(volume)
        return

    # Suffix correctness: the per-path accessors must address *this file* under
    # the volume root, not the bare volume root.
    sub_probe = volume.path("sub/dir/leaf.bin")
    print(f"\nstorage_location: {sub_probe.storage_location}")
    print(f"storage_path:     {sub_probe.storage_path(write=True).full_path()}")
    assert sub_probe.storage_location.endswith("/sub/dir/leaf.bin"), "suffix not appended!"

    results: "dict[str, dict[str, float]]" = {}
    calls: "dict[str, dict[str, int]]" = {}
    try:
        for direct in (False, True):
            label = "direct" if direct else "files"
            sub = f"{label}_{tag}"
            paths = [volume.path(f"{sub}/f{i}.bin") for i in range(args.files)]

            _set_mode(volume, direct=direct)
            row: "dict[str, float]" = {}
            crow: "dict[str, int]" = {}
            row["write"], crow["write"] = _timed_counted(
                lambda: [p.write_bytes(payload) for p in paths], args.repeat)
            row["read"], crow["read"] = _timed_counted(
                lambda: [p.read_bytes() for p in paths], args.repeat)
            # stat is the ``exists()``-before-write hot path: count the
            # governance round trips it costs each transport.
            row["stat"], crow["stat"] = _timed_counted(
                lambda: [p._stat_uncached().size for p in paths], args.repeat)
            row["ls"], crow["ls"] = _timed_counted(
                lambda: list(volume.path(sub).iterdir()), args.repeat)
            row["delete"], crow["delete"] = _timed_counted(
                lambda: [p.remove(missing_ok=True) for p in paths], 1)
            results[label] = row
            calls[label] = crow
    finally:
        _cleanup(volume)

    # ---- report ----
    ops = ["write", "read", "stat", "ls", "delete"]
    print(
        f"\nVolumePath direct-S3 vs Files-API — {args.files} files x "
        f"{args.size_kib} KiB, best of {args.repeat}\n"
        f"{'op':<8}{'files (ms)':>12}{'direct (ms)':>12}{'speedup':>9}"
        f"{'files API':>11}{'direct API':>11}")
    print("-" * 66)
    for op in ops:
        f = results["files"][op]
        d = results["direct"][op]
        speed = f / d if d else float("inf")
        print(f"{op:<8}{f:>12.1f}{d:>12.1f}{speed:>8.2f}x"
              f"{calls['files'][op]:>11}{calls['direct'][op]:>11}")
    tot_f = statistics.fsum(results["files"].values())
    tot_d = statistics.fsum(results["direct"].values())
    cf = sum(calls["files"].values())
    cd = sum(calls["direct"].values())
    print("-" * 66)
    print(f"{'total':<8}{tot_f:>12.1f}{tot_d:>12.1f}{(tot_f / tot_d if tot_d else 0):>8.2f}x"
          f"{cf:>11}{cd:>11}")
    print(f"\nGovernance (Files-API) round trips: files={cf}  direct={cd}  "
          f"(direct eliminates {cf - cd} of {cf}).")


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
