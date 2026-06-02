"""Benchmark Delta concurrent-append commit: blind-retry vs rebase.

``N`` threads each append a small batch to **one** Delta table. Every
thread contends for the next free commit version; the loser of a version
race must reconcile and re-attempt. Two strategies are compared:

- **BEFORE (blind retry)** — on a version collision the writer re-runs the
  whole write path against the fresh HEAD: re-bucket, re-encode parquet,
  re-collect stats, rebuild the action set. Correct, but every conflict
  pays the full write cost again, so throughput collapses under
  contention.
- **AFTER (rebase)** — a blind APPEND provably commutes with any concurrent
  commit (concurrent appends are independent under the Delta protocol), so
  on a collision we keep the already-written AddFiles and just renumber the
  commit at the new HEAD. No re-bucket, no re-encode, no re-stat — the data
  file is written exactly once regardless of how many version races it
  loses.

Both strategies converge to the same table (all rows present, version
advances by exactly the number of commits); the bench asserts ``final
version == commits`` so a regression that loses a write fails loudly.

How BEFORE is reproduced: the AFTER path is the shipped code. BEFORE swaps
in a ``_rebase_actions`` that, on every conflict, **re-encodes** the pending
data into a fresh parquet (a brand-new AddFile) instead of reusing the file
already on disk — exactly the "blindly redo the data write on a version
clash" cost the rebase removes. The ``redo-writes`` column counts those
extra encodes (= conflicts lost); AFTER is always 0.

Before / after (5 threads x 8 appends = 40 commits, 50k rows/append, to one
**local** table, median of 2 ``--repeat``, dev box)::

    BEFORE (blind retry)  wall 0.62s  cpu 0.85s   64.2 commits/s   31 redo-writes
    AFTER  (rebase)       wall 0.48s  cpu 0.73s   83.2 commits/s    0 redo-writes
                          --------------------------------------------------------
                          31 redundant parquet re-encodes eliminated (31 → 0),
                          ~1.3x wall / ~1.2x cpu locally. The eliminated work is
                          dominated by the per-commit JSON write + log listing on
                          a local FS, so the headline is the redo-write count.

The headline number is **redo-writes eliminated**, not local wall clock.
The eliminated work is a re-encode + (on a real table) a re-**upload** of
the data file. On a local FS that's cheap; on object storage — where every
AddFile is a multi-MB PUT over the network and the dominant cost of a commit
— avoiding a redo per lost race is the whole game, and wall time tracks the
redo-write column. Run ``--live`` against an external Databricks Delta table
on S3 (gated on credentials, see ``--help``) to see that regime. The win
scales with contention: more threads ⇒ more version races ⇒ more redundant
uploads avoided. With ``--threads 1`` the two are identical (no races).

Usage::

    PYTHONPATH=src python benchmarks/io/delta/bench_delta_concurrency.py
    PYTHONPATH=src python benchmarks/io/delta/bench_delta_concurrency.py \\
        --threads 8 --appends-per-thread 40 --repeat 5
    # Live (needs DATABRICKS_HOST + external-location CREATE grant):
    PYTHONPATH=src python benchmarks/io/delta/bench_delta_concurrency.py --live
"""
from __future__ import annotations

import argparse
import os
import secrets
import shutil
import statistics
import tempfile
import threading
import time
from typing import Callable

import pyarrow as pa

import yggdrasil.io.delta.delta_folder as delta_folder
from yggdrasil.enums import Mode
from yggdrasil.io.delta import DeltaFolder, DeltaOptions


def _run_contended(d: DeltaFolder, *, threads: int, appends: int,
                   rows_per_append: int = 50_000) -> dict:
    """Fire ``threads`` writers, each doing ``appends`` blind APPENDs.

    Each append carries ``rows_per_append`` rows so re-encoding a parquet on
    a lost version race is a real cost, not noise. Counts how many times
    ``_write_parts`` actually encoded parquet — under the rebase path that
    equals the number of commits; under blind-retry it also counts every redo
    forced by a lost version race.
    """
    write_count = {"n": 0}
    write_lock = threading.Lock()
    orig_write = DeltaFolder._write_parts

    def _counting(self, batches, *, partition_columns, options):
        with write_lock:
            write_count["n"] += 1
        return orig_write(self, batches, partition_columns=partition_columns,
                          options=options)

    errors: list[BaseException] = []

    def _worker(tid: int) -> None:
        for i in range(appends):
            base = (tid * appends + i) * rows_per_append
            payload = pa.table({
                "id": pa.array(range(base, base + rows_per_append), pa.int64()),
                "tid": pa.array([tid] * rows_per_append, pa.int64()),
            })
            try:
                d.write_arrow_batches(
                    payload.to_batches(),
                    options=DeltaOptions(mode=Mode.APPEND,
                                         # Periodic checkpoint writes aren't
                                         # atomic across threads — orthogonal
                                         # to commit throughput, so disable.
                                         checkpoint_interval=0,
                                         commit_max_retries=200,
                                         commit_retry_backoff=0.001,
                                         commit_retry_jitter=0.002,
                                         commit_retry_max_delay=0.05),
                )
            except BaseException as exc:  # noqa: BLE001 — surface in summary
                errors.append(exc)
                return

    DeltaFolder._write_parts = _counting  # type: ignore[assignment]
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        ts = [threading.Thread(target=_worker, args=(k,)) for k in range(threads)]
        for t in ts: t.start()
        for t in ts: t.join()
    finally:
        DeltaFolder._write_parts = orig_write  # type: ignore[assignment]
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0

    if errors:
        raise errors[0]
    snap = d.refresh().snapshot(fresh=True)
    total_commits = threads * appends
    return {
        "wall": wall,
        "cpu": cpu,
        "commits": total_commits,
        "version": snap.version,
        "rows": snap.num_rows_approx,
        "writes": write_count["n"],
    }


def _bench_once(*, threads: int, appends: int, rebase: bool) -> dict:
    tmp = tempfile.mkdtemp(prefix="ygg_delta_conc_")
    orig_rebase = delta_folder.DeltaFolder._rebase_actions
    if not rebase:
        # BEFORE: reproduce the pre-rebase "blindly redo the data write on a
        # version clash" behaviour — on every conflict, re-encode the pending
        # data into a fresh parquet (a new AddFile) instead of reusing the one
        # already on disk. This is the exact cost the rebase eliminates.
        import pyarrow as _pa
        from yggdrasil.io.parquet_file import ParquetFile as _PF, ParquetOptions as _PO

        def _redo_rebase(self, *, plan, pending, base_snap, head_snap,
                         cleanup, options):
            if cleanup is not None:
                cleanup()
            redone = []
            for act in pending:
                if isinstance(act, delta_folder.AddFile):
                    with _PF(holder=self.path / act.path, owns_holder=False) as f:
                        batches = list(f._read_arrow_batches(_PO()))
                    new_adds = list(self._write_parts(
                        iter(batches), partition_columns=[], options=options))
                    redone.extend(new_adds)
                else:
                    redone.append(act)
            return redone
        delta_folder.DeltaFolder._rebase_actions = _redo_rebase  # type: ignore[assignment]
    try:
        d = DeltaFolder(path=tmp)
        d.write_arrow_table(pa.table({"id": pa.array([-1], pa.int64()),
                                      "tid": pa.array([-1], pa.int64())}))
        return _run_contended(d, threads=threads, appends=appends)
    finally:
        delta_folder.DeltaFolder._rebase_actions = orig_rebase  # type: ignore[assignment]
        shutil.rmtree(tmp, ignore_errors=True)


def _median(fn: Callable[[], dict], *, repeat: int) -> dict:
    samples = [fn() for _ in range(repeat)]
    return {
        "wall": statistics.median(s["wall"] for s in samples),
        "cpu": statistics.median(s["cpu"] for s in samples),
        "commits": samples[0]["commits"],
        "version": samples[0]["version"],
        "writes": int(statistics.median(s["writes"] for s in samples)),
    }


def _live(threads: int, appends: int) -> None:
    if not os.environ.get("DATABRICKS_HOST"):
        print("# --live needs DATABRICKS_HOST (+ external-location CREATE grant)")
        return
    from databricks.sdk.service.catalog import TableType
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.aws import AWSDatabricksPathCredentials

    c = DatabricksClient()
    cat = os.environ.get("DATABRICKS_INTEGRATION_CATALOG", "trading_tgp_dev")
    sch = os.environ.get("DATABRICKS_INTEGRATION_SCHEMA", "ygg_integration")
    runid = secrets.token_hex(4)
    base = "s3://odp-aws-dls3-eu-central-1-a-apps/trading-tgp/ygg_delta_concurrency"
    loc = f"{base}/bench_{runid}/t"
    full = f"{cat}.{sch}.yg_conc_bench_{runid}"
    from yggdrasil.data.schema import Schema
    schema = Schema.from_arrow(pa.schema([("id", pa.int64()), ("tid", pa.int64())]))
    try:
        c.tables.table(full).create(schema, table_type=TableType.EXTERNAL,
                                    storage_location=loc)
        d = c.tables.table(full).delta()
        d.write_arrow_table(pa.table({"id": pa.array([-1], pa.int64()),
                                      "tid": pa.array([-1], pa.int64())}))
        res = _run_contended(d, threads=threads, appends=appends)
        print(f"\nLIVE (rebase) {threads}x{appends} → {full}")
        print(f"  wall={res['wall']:.2f}s  commits={res['commits']}  "
              f"version={res['version']}  writes={res['writes']}")
    finally:
        try: c.sql.execute(f"DROP TABLE IF EXISTS {full}")
        except Exception as e: print("drop err", e)
        try:
            AWSDatabricksPathCredentials(f"{base}/bench_{runid}/", client=c) \
                .aws_client().s3.path(f"{base}/bench_{runid}/") \
                .remove(recursive=True, missing_ok=True)
        except Exception as e: print("rm err", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--appends-per-thread", type=int, default=40)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--live", action="store_true",
                    help="Run against a live external Databricks Delta table.")
    args = ap.parse_args()

    if args.live:
        _live(args.threads, args.appends_per_thread)
        return

    before = _median(
        lambda: _bench_once(threads=args.threads,
                            appends=args.appends_per_thread, rebase=False),
        repeat=args.repeat)
    after = _median(
        lambda: _bench_once(threads=args.threads,
                            appends=args.appends_per_thread, rebase=True),
        repeat=args.repeat)

    # Convergence guard: every commit must have landed (no lost writes).
    for label, r in (("BEFORE", before), ("AFTER", after)):
        assert r["version"] == r["commits"], (
            f"{label}: version {r['version']} != {r['commits']} commits — lost a write")

    n = before["commits"]
    print(f"\nDelta concurrent-append bench: {args.threads} threads x "
          f"{args.appends_per_thread} appends = {n} commits to one table")
    print("-" * 70)
    for label, r in (("BEFORE (blind retry)", before), ("AFTER  (rebase)", after)):
        tput = r["commits"] / r["wall"] if r["wall"] else 0.0
        redo = max(0, r["writes"] - r["commits"])
        print(f"{label:<22} wall {r['wall']:6.2f}s  cpu {r['cpu']:6.2f}s  "
              f"{tput:6.1f} commits/s  {redo:4d} redo-writes  (final v{r['version']})")
    print("-" * 70)
    if after["wall"] > 0:
        print(f"speedup (BEFORE/AFTER wall): {before['wall'] / after['wall']:.2f}x")
    if after["cpu"] > 0:
        print(f"cpu work (BEFORE/AFTER)    : {before['cpu'] / after['cpu']:.2f}x  "
              f"(redo-writes eliminated: {max(0, before['writes'] - before['commits'])})")


if __name__ == "__main__":
    main()
