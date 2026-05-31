"""Benchmark :class:`DeltaFolder` — streamed reads/writes over the full feature set.

The Delta read/write path in :class:`yggdrasil.io.delta.DeltaFolder`
runs end-to-end on Arrow record batches: never materialise the full
table into RAM unless the caller explicitly asks for it. This bench
pins the wall-clock + peak-memory cost of every step in the pipeline
so a regression that silently flips a streaming path into a "collect
then return" path shows up.

What this measures
------------------

Three orthogonal axes, all on the same partitioned Delta fixture:

1. **Read shape** — batched (``read_arrow_batches``, streaming) vs
   table (``read_arrow_table``, collect-all). The batched path is the
   one production pipelines hand to downstream sinks; the table path
   is what one-shot scripts call. Streaming should win at peak
   memory; the table path should win marginally at wall-clock for a
   small fixture because there's no per-batch Python overhead, but
   that flips as fixtures grow.
2. **Predicate driver** — no filter / explicit ``prune_values`` /
   single-EQ predicate / OR-of-EQ predicate / partition +
   row-level predicate / row-level-only predicate. The whole
   predicate-driven path runs the AST through
   :func:`extract_partition_filters` once at read time and folds
   the residual into a per-batch
   ``pyarrow.compute`` filter — file prune *and* row prune from one
   :class:`Predicate`.
3. **Write shape** — table write vs batch-iterator write. The
   iterator path keeps memory bounded on the source side too — the
   source can be a generator producing record batches lazily.

For each scenario we report wall time + peak memory (tracemalloc on
the Python heap) + the row count that survived so the caller knows
how much work the predicate actually did.

Usage::

    PYTHONPATH=src python benchmarks/io/nested/bench_delta.py
    PYTHONPATH=src python benchmarks/io/nested/bench_delta.py \\
        --rows 200000 --partitions 32 --batch-rows 5000 --repeat 5
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.enums import Mode
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import Int64Type, StringType
from yggdrasil.io.delta import DeltaFolder, DeltaOptions
from yggdrasil.execution.expr import col as expr_col


# ---------------------------------------------------------------------------
# Fixture — partitioned delta table built once per run.
# ---------------------------------------------------------------------------


PARTITION_KEYS = [f"p{i:02d}" for i in range(64)]


def _partition_schema() -> Schema:
    """Schema partitioned by ``region`` — drives the file split."""
    s = Schema()
    s.with_field(Field(name="id", dtype=Int64Type()))
    s.with_field(
        Field(name="region", dtype=StringType()).with_partition_by(True)
    )
    s.with_field(Field(name="val", dtype=StringType()))
    return s


def _arrow_table(rows: int, *, partitions: int) -> pa.Table:
    """Build the sample table: rows evenly distributed across N partitions."""
    keys = PARTITION_KEYS[:partitions]
    return pa.table({
        "id": pa.array(range(rows), type=pa.int64()),
        "region": pa.array([keys[i % partitions] for i in range(rows)]),
        "val": pa.array([f"row-{i}" for i in range(rows)]),
    })


def _slice_into_batches(
    table: pa.Table, batch_rows: int,
) -> list[pa.RecordBatch]:
    """Slice *table* into fixed-row record batches (write fixture)."""
    out: list[pa.RecordBatch] = []
    total = table.num_rows
    for offset in range(0, total, batch_rows):
        chunk = table.slice(offset, batch_rows)
        out.extend(chunk.to_batches())
    return out


def _batch_generator(
    batches: list[pa.RecordBatch],
) -> Iterator[pa.RecordBatch]:
    """One-shot generator over *batches* — mimics a real streaming source.

    Wrapping a list in ``iter()`` works, but a real generator is what
    upstream pipelines (HTTP paginated readers, Kafka consumers,
    Spark structured-streaming sinks) hand off — make the bench
    measure that shape too.
    """
    for b in batches:
        yield b


# ---------------------------------------------------------------------------
# Timing — wall time + peak heap (tracemalloc), aggregated across repeats.
# ---------------------------------------------------------------------------


def _time_one(
    label: str,
    fn: Callable[[], object],
    *,
    repeat: int,
    inner: int = 1,
    track_memory: bool = True,
) -> dict:
    """Run *fn* ``repeat`` times, capturing wall time and peak heap.

    ``inner`` repeats the call inside one timing sample — useful for
    sub-millisecond scenarios where one run is below clock noise.
    Heap measurement starts fresh per sample via
    :func:`tracemalloc.start` so cross-sample state doesn't bleed.
    """
    # Warmup — drop import / cold-cache cost out of the reported numbers.
    fn()
    samples: list[float] = []
    peaks: list[int] = []
    for _ in range(repeat):
        if track_memory:
            tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        elapsed = (time.perf_counter() - t0) / inner
        if track_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak)
        samples.append(elapsed)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "peak_mem_bytes": (min(peaks) if peaks else 0),
    }


def _fmt_time(secs: float) -> str:
    if secs < 1e-6:
        return f"{secs * 1e9:7.1f} ns"
    if secs < 1e-3:
        return f"{secs * 1e6:7.1f} us"
    if secs < 1:
        return f"{secs * 1e3:7.1f} ms"
    return f"{secs:7.3f}  s"


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n:6d}  B"
    if n < 1024 * 1024:
        return f"{n / 1024:6.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):6.1f} MB"
    return f"{n / (1024 * 1024 * 1024):6.2f} GB"


def _fmt(r: dict) -> str:
    mem = _fmt_bytes(r["peak_mem_bytes"]) if r["peak_mem_bytes"] else "      -"
    return (
        f"{r['label']:<72s}  "
        f"best={_fmt_time(r['best'])}  "
        f"median={_fmt_time(r['median'])}  "
        f"peak_mem={mem}"
    )


# ---------------------------------------------------------------------------
# Helpers — drain a stream into a row count so we measure the iterate-
# every-batch cost rather than just compiling the predicate.
# ---------------------------------------------------------------------------


def _count_streamed_rows(batches: Iterable[pa.RecordBatch]) -> int:
    """Walk the stream, return the total row count.

    The bench measures the *streaming* path, so we deliberately don't
    accumulate batches into a list — that would collapse to the
    collect-all shape and hide the bounded-memory property.
    """
    n = 0
    for b in batches:
        n += b.num_rows
    return n


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _extractor_scenarios(repeat: int) -> list[dict]:
    """Standalone cost of the AST passes a streaming read runs once.

    These are paid one time per :meth:`DeltaFolder._read_arrow_batches`
    invocation regardless of how many batches the read produces — so
    the per-stream amortised cost shrinks linearly with stream length.
    """
    from yggdrasil.execution.expr import extract_partition_filters

    out: list[dict] = []
    pcols = ("region", "date")

    out.append(_time_one(
        "extract: eq on partition col",
        lambda: extract_partition_filters(expr_col("region") == "us", pcols),
        repeat=repeat, inner=5_000,
    ))

    big_in = expr_col("region").is_in(PARTITION_KEYS[:32])
    out.append(_time_one(
        "extract: InList 32 values on partition col",
        lambda: extract_partition_filters(big_in, pcols),
        repeat=repeat, inner=5_000,
    ))

    or_chain = expr_col("region") == PARTITION_KEYS[0]
    for k in PARTITION_KEYS[1:16]:
        or_chain = or_chain | (expr_col("region") == k)
    out.append(_time_one(
        "extract: OR-of-EQ 16 values (per-operand union → extract)",
        lambda: extract_partition_filters(or_chain, pcols),
        repeat=repeat, inner=2_000,
    ))

    mixed = expr_col("region").is_in(["us", "eu"]) & (expr_col("id") > 100)
    out.append(_time_one(
        "extract: mixed AND (partition + non-partition)",
        lambda: extract_partition_filters(mixed, pcols),
        repeat=repeat, inner=5_000,
    ))

    range_only = expr_col("id") > 100
    out.append(_time_one(
        "extract: non-extractable (range on non-partition col)",
        lambda: extract_partition_filters(range_only, pcols),
        repeat=repeat, inner=5_000,
    ))
    return out


def _write_scenarios(
    table: pa.Table,
    rows: int,
    partitions: int,
    batch_rows: int,
    repeat: int,
) -> list[dict]:
    """Write throughput — table-shaped vs streamed-batches.

    ``write_arrow_table`` takes the full payload in one shot;
    ``write_arrow_batches`` accepts any ``Iterable[RecordBatch]`` so
    the upstream source can be a lazy generator. Peak memory should
    diverge sharply at large fixture sizes: the table form holds the
    whole payload + the staged parquet buffers, the batch form only
    holds one batch at a time plus the parquet writer's column-chunk
    buffer.
    """
    out: list[dict] = []
    pre_sliced = _slice_into_batches(table, batch_rows)
    schema_opt = DeltaOptions(target=_partition_schema())

    def _write_table():
        tmp = tempfile.mkdtemp(prefix="ygg-delta-w-")
        try:
            d = DeltaFolder(path=tmp + "/t")
            d.write_arrow_table(table, options=schema_opt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _write_batches_list():
        tmp = tempfile.mkdtemp(prefix="ygg-delta-w-")
        try:
            d = DeltaFolder(path=tmp + "/t")
            d.write_arrow_batches(list(pre_sliced), options=schema_opt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _write_batches_gen():
        # Wrap in a fresh generator each call — a generator is single-
        # shot, exactly what an upstream HTTP / Kafka / Spark source
        # hands us. Demonstrates that the writer doesn't need a re-
        # iterable; one pass is sufficient.
        tmp = tempfile.mkdtemp(prefix="ygg-delta-w-")
        try:
            d = DeltaFolder(path=tmp + "/t")
            d.write_arrow_batches(
                _batch_generator(pre_sliced), options=schema_opt,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    out.append(_time_one(
        f"write: table rows={rows} parts={partitions}",
        _write_table, repeat=repeat,
    ))
    out.append(_time_one(
        f"write: batches list rows={rows} batch_rows={batch_rows}",
        _write_batches_list, repeat=repeat,
    ))
    out.append(_time_one(
        f"write: batches generator rows={rows} batch_rows={batch_rows}",
        _write_batches_gen, repeat=repeat,
    ))
    return out


def _read_scenarios(
    table_path: Path,
    rows: int,
    partitions: int,
    repeat: int,
) -> list[dict]:
    """Read throughput — streaming vs collect-all, across predicate shapes.

    Each scenario reports peak memory: the streamed path should stay
    flat at "one batch + scratch" regardless of fixture size, the
    table path grows linearly with the result set.
    """
    out: list[dict] = []
    keys = PARTITION_KEYS[:partitions]
    target_key = keys[0]
    or_keys = keys[: min(4, partitions)]

    # Warm the snapshot once so each scenario starts with the log
    # already resolved — what production code sees after the first
    # read in a session.
    d = DeltaFolder(path=str(table_path))
    d.snapshot(fresh=True)

    # -----------------------------------------------------------------
    # Baselines: full table scan, streamed vs collect-all.
    # -----------------------------------------------------------------

    out.append(_time_one(
        f"read: streamed full-scan rows={rows} parts={partitions}",
        lambda: _count_streamed_rows(d.read_arrow_batches()),
        repeat=repeat,
    ))
    out.append(_time_one(
        f"read: table full-scan rows={rows} parts={partitions}",
        lambda: d.read_arrow_table(),
        repeat=repeat,
    ))

    # -----------------------------------------------------------------
    # Legacy explicit prune_values — the path that used to be the
    # fastest because no AST work fires. Still useful as a baseline
    # for the predicate-driven shape.
    # -----------------------------------------------------------------

    out.append(_time_one(
        f"read: streamed prune_values=[{target_key}] (1/{partitions} files)",
        lambda: _count_streamed_rows(d.read_arrow_batches(
            options=DeltaOptions(prune_values={"region": (target_key,)}),
        )),
        repeat=repeat,
    ))

    # -----------------------------------------------------------------
    # Predicate-driven — one Predicate drives file prune + row filter.
    # ``extract_partition_filters`` walks the AST, unions per-operand
    # accepted values into a partition-prune set, and the residual
    # flows into a per-batch pyarrow.compute filter.
    # -----------------------------------------------------------------

    eq_pred = expr_col("region") == target_key
    out.append(_time_one(
        f"read: streamed predicate region == {target_key}",
        lambda: _count_streamed_rows(d.read_arrow_batches(
            options=DeltaOptions(predicate=eq_pred),
        )),
        repeat=repeat,
    ))

    or_pred = expr_col("region") == or_keys[0]
    for k in or_keys[1:]:
        or_pred = or_pred | (expr_col("region") == k)
    out.append(_time_one(
        f"read: streamed predicate region == ... ({len(or_keys)} OR'd → InList)",
        lambda: _count_streamed_rows(d.read_arrow_batches(
            options=DeltaOptions(predicate=or_pred),
        )),
        repeat=repeat,
    ))

    mixed_pred = (expr_col("region") == target_key) & (expr_col("id") > rows // 2)
    out.append(_time_one(
        f"read: streamed predicate region == X AND id > N/2",
        lambda: _count_streamed_rows(d.read_arrow_batches(
            options=DeltaOptions(predicate=mixed_pred),
        )),
        repeat=repeat,
    ))

    non_part_pred = expr_col("id") > rows - 100
    out.append(_time_one(
        f"read: streamed predicate id > N-100 (row-only, all files)",
        lambda: _count_streamed_rows(d.read_arrow_batches(
            options=DeltaOptions(predicate=non_part_pred),
        )),
        repeat=repeat,
    ))

    # -----------------------------------------------------------------
    # Stream-then-stop — early termination. With a generator-shaped
    # read, the caller can break out after the first matching batch
    # and never open the rest. Pin the wall-clock saving so the
    # bounded-stream property stays load-bearing.
    # -----------------------------------------------------------------

    def _first_batch_only():
        for batch in d.read_arrow_batches():
            return batch.num_rows
        return 0

    out.append(_time_one(
        f"read: streamed early-exit after first batch (full scan)",
        _first_batch_only, repeat=repeat,
    ))

    def _first_batch_pred():
        opts = DeltaOptions(predicate=(expr_col("region") == target_key))
        for batch in d.read_arrow_batches(options=opts):
            return batch.num_rows
        return 0

    out.append(_time_one(
        f"read: streamed early-exit after first batch (predicate)",
        _first_batch_pred, repeat=repeat,
    ))

    # -----------------------------------------------------------------
    # Time-travel — Delta versioning is a Delta-only feature; pinning
    # the read at version 0 exercises the log-replay path that newer
    # versions skip.
    # -----------------------------------------------------------------

    out.append(_time_one(
        f"read: streamed full-scan at version=0",
        lambda: _count_streamed_rows(d.read_arrow_batches(
            options=DeltaOptions(version=0),
        )),
        repeat=repeat,
    ))
    return out


# ---------------------------------------------------------------------------
# Path-call counter — quantifies remote-store round trips
#
# Remote backends (S3 ``LIST/HEAD/GET``, Databricks REST ``list/read``,
# ABFS) pay one network RTT per ``Path`` method call. Local FS pays
# essentially nothing, so the wall-clock numbers above don't surface
# the cost. We instrument ``Path`` directly (patch the abstract base
# so every subclass — local, S3, DBFS — is counted in one shot),
# count the calls per high-level read/write, and report.
#
# This is the single most important number for the remote-path
# optimisation goal: every saved call here is a saved RTT in prod.
# ---------------------------------------------------------------------------


def _path_call_counts() -> dict:
    """Counter dict shared across spies for a single bench block."""
    return {
        "exists": 0,
        "iterdir": 0,
        "stat": 0,
        "open": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        "size": 0,
        "size_known": 0,
        "is_dir": 0,
        "is_file": 0,
        "unlink": 0,
        "mkdir": 0,
    }


class _PathCallSpy:
    """Context manager that counts Path-method calls across all subclasses.

    Monkey-patches the abstract :class:`yggdrasil.io.io.Path` so every
    concrete subclass (LocalPath, S3Path, VolumePath) inherits the
    counted version. Captures the original methods on enter and
    restores on exit — safe to nest only one at a time.
    """

    def __init__(self) -> None:
        self.counts: dict = _path_call_counts()
        self._originals: dict = {}

    def __enter__(self) -> dict:
        from yggdrasil.io.path.path import Path as _Path

        # Methods that hit the storage backend (or could). The size
        # property is also a stat in disguise on remote backends.
        for name in (
            "exists", "iterdir", "stat", "open", "read_bytes",
            "write_bytes", "is_dir", "is_file", "unlink", "mkdir",
        ):
            attr = getattr(_Path, name, None)
            if attr is None or not callable(attr):
                continue
            counts = self.counts
            self._originals[name] = attr

            def _wrap(orig=attr, counter_name=name):
                def _spy(*args, **kwargs):
                    counts[counter_name] += 1
                    return orig(*args, **kwargs)
                return _spy

            setattr(_Path, name, _wrap())
        # ``size`` is a property — patch via descriptor.
        size_prop = getattr(_Path, "size", None)
        if isinstance(size_prop, property):
            self._originals["size"] = size_prop
            counts = self.counts

            def _size_fget(s, _orig=size_prop.fget):
                counts["size"] += 1
                return _orig(s)
            setattr(_Path, "size", property(_size_fget))
        return self.counts

    def __exit__(self, *_exc) -> None:
        from yggdrasil.io.path.path import Path as _Path
        for name, orig in self._originals.items():
            setattr(_Path, name, orig)


def _path_call_scenarios(
    table_path: Path,
    rows: int,
    partitions: int,
    repeat: int,
) -> list[dict]:
    """Path-method call counters for the canonical read/write shapes.

    The wall-clock numbers in :func:`_read_scenarios` /
    :func:`_write_scenarios` reflect mostly the parquet decode + Arrow
    work. The interesting number for remote-store performance is the
    *count* of ``Path`` method calls each scenario makes — every
    ``exists`` / ``iterdir`` / ``open`` / ``stat`` is one network
    round trip on S3, DBFS, or ABFS.

    Reports one line per scenario with the call breakdown. ``best`` /
    ``median`` are still useful (the spy is cheap) so the table
    formatter doesn't need a separate row shape.
    """
    out: list[dict] = []
    keys = PARTITION_KEYS[:partitions]
    target_key = keys[0]

    def _run_with_counts(
        label: str,
        fn: Callable[[], object],
    ) -> dict:
        # Warmup once outside the spy so import-time path probes
        # don't pollute the counters.
        fn()
        timings: list[float] = []
        with _PathCallSpy() as counts:
            for _ in range(repeat):
                t0 = time.perf_counter()
                fn()
                timings.append(time.perf_counter() - t0)
        # Per-run averages — divide total counts by repeat so the
        # reported number is "calls per read/write" not "total".
        per_run = {k: v // max(1, repeat) for k, v in counts.items()}
        nonzero = {k: v for k, v in per_run.items() if v > 0}
        # Pack the counts into the label so the standard
        # _fmt-style row still works.
        suffix = " ".join(f"{k}={v}" for k, v in sorted(nonzero.items()))
        full_label = f"{label}  [{suffix}]"
        return {
            "label": full_label,
            "best": min(timings),
            "median": statistics.median(timings),
            "mean": statistics.fmean(timings),
            "peak_mem_bytes": 0,
        }

    out.append(_run_with_counts(
        "path-calls: read streamed full-scan",
        lambda: _count_streamed_rows(
            DeltaFolder(path=str(table_path)).read_arrow_batches()
        ),
    ))
    out.append(_run_with_counts(
        f"path-calls: read predicate region == {target_key}",
        lambda: _count_streamed_rows(
            DeltaFolder(path=str(table_path)).read_arrow_batches(
                options=DeltaOptions(predicate=expr_col("region") == target_key),
            )
        ),
    ))
    out.append(_run_with_counts(
        "path-calls: snapshot-only (resolve + log replay, no parquet open)",
        lambda: DeltaFolder(path=str(table_path)).snapshot(fresh=True),
    ))
    return out


# ---------------------------------------------------------------------------
# Concurrency — N writers racing on the same Delta table
#
# Each writer goes through the same retry loop. Exercises the path the
# real workloads hit when N ingestion tasks fan out on the same
# partition table. Reports wall-clock to land all N commits + the
# total retry count across writers — a regression that flips the
# retry loop into a hot busy-spin shows up as a retry explosion.
# ---------------------------------------------------------------------------


def _concurrency_scenarios(repeat: int) -> list[dict]:
    """N concurrent writers committing to one Delta table."""
    import threading

    out: list[dict] = []
    seed = pa.table({"id": [0]})

    def _run_concurrent(num_writers: int) -> tuple[float, int]:
        tmp = tempfile.mkdtemp(prefix="ygg-delta-cc-")
        try:
            # Initialise the table once so all writers start with the
            # same view of HEAD.
            DeltaFolder(path=tmp + "/t").write_arrow_table(seed)

            retries = {"n": 0}
            barrier = threading.Barrier(num_writers)

            def _writer(i: int) -> None:
                d = DeltaFolder(path=tmp + "/t")
                # Track retries by spying on _commit_atomic.
                orig = d._commit_atomic
                local_retries = [-1]  # first call is the initial attempt

                def _spy(version, actions):
                    local_retries[0] += 1
                    return orig(version, actions)

                d._commit_atomic = _spy  # type: ignore[assignment]
                barrier.wait()  # synchronise the race start
                d.write_arrow_batches(
                    pa.table({"id": [i + 1]}).to_batches(),
                    options=DeltaOptions(
                        mode=Mode.APPEND,
                        commit_max_retries=64,
                        commit_retry_backoff=0.005,
                        commit_retry_jitter=0.005,
                        commit_retry_max_delay=0.05,
                    ),
                )
                retries["n"] += local_retries[0]

            threads = [
                threading.Thread(target=_writer, args=(i,))
                for i in range(num_writers)
            ]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.perf_counter() - t0

            # Sanity — every writer's commit landed (versions
            # 1..num_writers on top of the seed at v0).
            head = DeltaFolder(path=tmp + "/t").snapshot(fresh=True)
            assert head.version == num_writers, (
                f"expected version={num_writers}, got {head.version}"
            )
            return elapsed, retries["n"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    for n in (2, 4, 8):
        elapsed_samples: list[float] = []
        retry_samples: list[int] = []
        for _ in range(repeat):
            e, r = _run_concurrent(n)
            elapsed_samples.append(e)
            retry_samples.append(r)
        # Pack retry count into the label (one number per scenario).
        median_retries = statistics.median(retry_samples)
        out.append({
            "label": f"concurrency: {n} writers racing  [retries={int(median_retries)}]",
            "best": min(elapsed_samples),
            "median": statistics.median(elapsed_samples),
            "mean": statistics.fmean(elapsed_samples),
            "peak_mem_bytes": 0,
        })
    return out


# ---------------------------------------------------------------------------
# Deletion vectors — DV decode + per-batch masking
# ---------------------------------------------------------------------------


def _deletion_vector_scenarios(repeat: int) -> list[dict]:
    """DV decode + mask throughput.

    Sparse vs dense DV against a 10k-row batch. The mask hot path was
    rewritten to vectorise through numpy + pyarrow.compute — this
    pins the per-batch cost so the row-loop regression surfaces.
    """
    from yggdrasil.io.delta.deletion_vector import (
        DeletionVector, DeletionVectorDescriptor, mask_batch_with_dv,
    )

    out: list[dict] = []
    n = 10_000
    batch = pa.RecordBatch.from_pydict({
        "id": pa.array(range(n), type=pa.int64()),
        "val": pa.array([f"row-{i}" for i in range(n)], type=pa.string()),
    })

    descriptor_inline = DeletionVectorDescriptor(
        storage_type="i",
        path_or_inline_dv="",  # not consulted when deleted_rows is set
        offset=0,
        size_in_bytes=0,
        cardinality=0,
    )

    sparse_dv = DeletionVector(
        descriptor=descriptor_inline,
        deleted_rows=frozenset(range(0, n, 1000)),  # 10 rows
    )
    medium_dv = DeletionVector(
        descriptor=descriptor_inline,
        deleted_rows=frozenset(range(0, n, 10)),  # 1000 rows
    )
    dense_dv = DeletionVector(
        descriptor=descriptor_inline,
        deleted_rows=frozenset(range(0, n // 2)),  # 5000 rows
    )

    out.append(_time_one(
        f"dv-mask: sparse ({len(sparse_dv.deleted_rows)} / {n} rows deleted)",
        lambda: mask_batch_with_dv(batch, sparse_dv),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"dv-mask: medium ({len(medium_dv.deleted_rows)} / {n} rows deleted)",
        lambda: mask_batch_with_dv(batch, medium_dv),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"dv-mask: dense ({len(dense_dv.deleted_rows)} / {n} rows deleted)",
        lambda: mask_batch_with_dv(batch, dense_dv),
        repeat=repeat, inner=200,
    ))
    return out


def scenarios(
    rows: int,
    partitions: int,
    batch_rows: int,
    repeat: int,
) -> list[dict]:
    tmp_root = Path(tempfile.mkdtemp(prefix="ygg-delta-bench-"))
    try:
        table = _arrow_table(rows, partitions=partitions)
        read_path = tmp_root / "read_t"
        d = DeltaFolder(path=str(read_path))
        d.write_arrow_table(
            table, options=DeltaOptions(target=_partition_schema()),
        )
        out: list[dict] = []
        out.extend(_extractor_scenarios(repeat))
        out.extend(_deletion_vector_scenarios(repeat))
        out.extend(_write_scenarios(
            table, rows, partitions, batch_rows, repeat,
        ))
        out.extend(_read_scenarios(read_path, rows, partitions, repeat))
        out.extend(_path_call_scenarios(read_path, rows, partitions, repeat))
        out.extend(_concurrency_scenarios(repeat))
        return out
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000,
                    help="Row count for the fixture table.")
    ap.add_argument("--partitions", type=int, default=16,
                    help="Distinct partition values "
                         f"(≤ {len(PARTITION_KEYS)}).")
    ap.add_argument("--batch-rows", type=int, default=5_000,
                    help="Row count per batch on the write-batches "
                         "scenarios.")
    ap.add_argument("--repeat", type=int, default=3,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    if args.partitions > len(PARTITION_KEYS):
        ap.error(f"--partitions must be ≤ {len(PARTITION_KEYS)}")

    print(
        f"# rows={args.rows}  partitions={args.partitions}  "
        f"batch_rows={args.batch_rows}  repeat={args.repeat}"
    )
    print(
        f"# {'label':<72s}  {'best':>13s}  {'median':>18s}  "
        f"{'peak_mem':>11s}"
    )
    for row in scenarios(
        args.rows, args.partitions, args.batch_rows, args.repeat,
    ):
        print(_fmt(row))


if __name__ == "__main__":
    main()
