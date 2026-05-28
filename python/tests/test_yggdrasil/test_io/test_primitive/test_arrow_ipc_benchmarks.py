"""Arrow IPC write/read benchmarks — wall-clock visibility for each
codec / batch shape / override knob.

Mirrors ``test_parquet_benchmarks.py``:

* **Regression mode** (default) — small inputs (~4k rows) so the
  suite stays fast (<1 s); only asserts that each combination
  produces a valid IPC file. Catches an upstream option rename
  before the benchmark sweep silently skips it.
* **Benchmark mode** — opt-in via ``YGG_BENCHMARK=1``. Pumps row
  counts up (~250k by default; ``YGG_BENCHMARK_ROWS=...`` to override)
  and prints per-axis tables.

Why this file is here: the benchmarks exercise the same code path
the IO tests cover — keeping them adjacent means a write-side
regression that breaks the format is caught by the IO tests in the
same run, before the benchmark numbers go misleading.
"""
from __future__ import annotations

import os
import random
import string
import time
import unittest
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile, ArrowIPCOptions


# ---------------------------------------------------------------------------
# Mode + sizing
# ---------------------------------------------------------------------------


def _benchmark_mode() -> bool:
    return os.environ.get("YGG_BENCHMARK", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _bench_rows() -> int:
    raw = os.environ.get("YGG_BENCHMARK_ROWS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 250_000


def _regression_rows() -> int:
    return 4_000


# ---------------------------------------------------------------------------
# Test data factories — deterministic so byte counts compare
# ---------------------------------------------------------------------------


def _make_numeric_table(n: int, seed: int = 0) -> pa.Table:
    rnd = random.Random(seed)
    return pa.table({
        "id": pa.array(range(n), type=pa.int64()),
        "x": pa.array([rnd.uniform(-1.0, 1.0) for _ in range(n)], type=pa.float64()),
        "y": pa.array([rnd.randint(-1_000_000, 1_000_000) for _ in range(n)], type=pa.int32()),
    })


def _make_mixed_table(n: int, seed: int = 0) -> pa.Table:
    rnd = random.Random(seed)
    return pa.table({
        "id": pa.array(range(n), type=pa.int64()),
        "v": pa.array([rnd.uniform(0, 100) for _ in range(n)], type=pa.float64()),
        "tag": pa.array([rnd.choice(["A", "B", "C", "D"]) for _ in range(n)], type=pa.string()),
        "flag": pa.array([rnd.random() < 0.5 for _ in range(n)], type=pa.bool_()),
    })


# ---------------------------------------------------------------------------
# Bench primitives
# ---------------------------------------------------------------------------


def _time(fn) -> tuple[float, int]:
    t0 = time.perf_counter()
    payload = fn()
    elapsed = (time.perf_counter() - t0) * 1000.0
    return elapsed, len(payload)


def _write_table_via_override(table: pa.Table) -> bytes:
    """``ArrowIPCFile.write_arrow_table`` — the
    ``writer.write_table`` override I just added."""
    io_obj = ArrowIPCFile()
    io_obj.write_arrow_table(table)
    return bytes(io_obj.getvalue())


def _write_table_via_batch_hook(table: pa.Table) -> bytes:
    """Pre-override shape: stream batches through
    ``_write_arrow_batches`` so each batch hits
    ``writer.write_batch`` individually."""
    io_obj = ArrowIPCFile()
    io_obj._write_arrow_batches(iter(table.to_batches()), ArrowIPCOptions())
    return bytes(io_obj.getvalue())


def _read_table_via_override(payload: bytes) -> pa.Table:
    """``read_arrow_table`` through the
    ``RecordBatchFileReader.read_all`` override."""
    io_obj = ArrowIPCFile()
    io_obj.write_bytes(payload, 0)
    return io_obj.read_arrow_table()


def _read_table_via_batch_hook(payload: bytes) -> pa.Table:
    """Stream ``_read_arrow_batches`` then stitch via
    ``pa.Table.from_batches`` — the base class shape."""
    io_obj = ArrowIPCFile()
    io_obj.write_bytes(payload, 0)
    batches = list(io_obj._read_arrow_batches(ArrowIPCOptions()))
    return pa.Table.from_batches(batches)


class _BenchReport:
    """Tiny perf-row accumulator (lifted from the parquet benchmark
    file). Prints a flat table on flush — stable, grep-friendly,
    diff-friendly."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.rows: list[tuple[str, float, int]] = []

    def add(self, label: str, elapsed_ms: float, nbytes: int) -> None:
        self.rows.append((label, elapsed_ms, nbytes))

    def flush(self) -> None:
        if not self.rows:
            return
        print(f"\n=== {self.name} ===")
        label_w = max(len(r[0]) for r in self.rows)
        for label, ms, nbytes in self.rows:
            print(f"{label.ljust(label_w)}  ms={ms:8.2f}  bytes={nbytes}")


# ---------------------------------------------------------------------------
# Regression-mode tests — always run
# ---------------------------------------------------------------------------


class TestArrowIPCRegressionRoundTrip(ArrowTestCase):
    """Each write/read shape produces a round-trippable IPC file."""

    def test_override_write_round_trips(self) -> None:
        table = _make_numeric_table(_regression_rows())
        payload = _write_table_via_override(table)
        out = ipc.RecordBatchFileReader(pa.BufferReader(payload)).read_all()
        self.assertEqual(out.num_rows, table.num_rows)

    def test_batch_hook_write_round_trips(self) -> None:
        table = _make_numeric_table(_regression_rows())
        payload = _write_table_via_batch_hook(table)
        out = ipc.RecordBatchFileReader(pa.BufferReader(payload)).read_all()
        self.assertEqual(out.num_rows, table.num_rows)

    def test_override_read_matches_batch_hook_read(self) -> None:
        table = _make_mixed_table(_regression_rows())
        payload = _write_table_via_override(table)
        fast = _read_table_via_override(payload)
        slow = _read_table_via_batch_hook(payload)
        self.assertTrue(fast.equals(slow))


# ---------------------------------------------------------------------------
# Benchmark-mode sweeps — gated by YGG_BENCHMARK=1
# ---------------------------------------------------------------------------


@unittest.skipUnless(_benchmark_mode(), "set YGG_BENCHMARK=1 to run benchmarks")
class TestArrowIPCTableOverrideBenchmark(ArrowTestCase):
    """Wall-clock measurement of the ``_write_arrow_table`` /
    ``_read_arrow_table`` overrides vs forcing the batch hook.

    The override should be measurably faster on a sufficiently large
    table because pyarrow lays out IPC blocks in one C-level call
    rather than N ``writer.write_batch`` hops through Python."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.n_rows = _bench_rows()
        cls.numeric = _make_numeric_table(cls.n_rows)
        cls.mixed = _make_mixed_table(cls.n_rows)

    def test_write_override_vs_batch_hook_numeric(self) -> None:
        report = _BenchReport(
            f"write override vs batch hook (numeric, n={self.n_rows})"
        )
        ms_fast, nb_fast = _time(
            lambda: _write_table_via_override(self.numeric),
        )
        ms_slow, nb_slow = _time(
            lambda: _write_table_via_batch_hook(self.numeric),
        )
        report.add("write_arrow_table  (override)", ms_fast, nb_fast)
        report.add("write_arrow_batches(batch hk)", ms_slow, nb_slow)
        self.assertGreater(nb_fast, 0)
        self.assertGreater(nb_slow, 0)
        report.flush()

    def test_write_override_vs_batch_hook_mixed(self) -> None:
        report = _BenchReport(
            f"write override vs batch hook (mixed, n={self.n_rows})"
        )
        ms_fast, nb_fast = _time(
            lambda: _write_table_via_override(self.mixed),
        )
        ms_slow, nb_slow = _time(
            lambda: _write_table_via_batch_hook(self.mixed),
        )
        report.add("write_arrow_table  (override)", ms_fast, nb_fast)
        report.add("write_arrow_batches(batch hk)", ms_slow, nb_slow)
        report.flush()

    def test_read_override_vs_batch_hook(self) -> None:
        payload = _write_table_via_override(self.mixed)
        report = _BenchReport(
            f"read override vs batch hook (mixed, n={self.n_rows})"
        )
        t0 = time.perf_counter()
        fast = _read_table_via_override(payload)
        ms_fast = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        slow = _read_table_via_batch_hook(payload)
        ms_slow = (time.perf_counter() - t0) * 1000.0
        report.add("read_arrow_table  (override)", ms_fast, fast.num_rows)
        report.add("read_arrow_batches(batch hk)", ms_slow, slow.num_rows)
        self.assertEqual(fast.num_rows, slow.num_rows)
        report.flush()

    def test_batch_chunking_effect(self) -> None:
        """How much does batch chunking cost vs a single batch?

        The batch hook iterates batches; ``writer.write_batch`` runs
        once per batch. Tiny batches amortize writer setup poorly.
        """
        report = _BenchReport(
            f"batch chunking (mixed, n={self.n_rows})"
        )
        single = self.mixed.combine_chunks().to_batches()

        def write_with(batches):
            io_obj = ArrowIPCFile()
            io_obj._write_arrow_batches(iter(batches), ArrowIPCOptions())
            return bytes(io_obj.getvalue())

        ms1, nb1 = _time(lambda: write_with(single))
        report.add("batches=1 (combined)            ", ms1, nb1)

        for bs in (4_096, 16_384, 65_536):
            batches = self.mixed.to_batches(max_chunksize=bs)
            n = len(batches)
            ms, nbytes = _time(lambda b=batches: write_with(b))
            report.add(f"batches={n:<4d} (max_chunksize={bs})", ms, nbytes)
            self.assertGreater(nbytes, 0)
        report.flush()
