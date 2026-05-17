"""Parquet write benchmarks — surface where the encoder time goes.

What this file is
-----------------
A wall-clock harness over :class:`ParquetFile._write_arrow_batches`
(``python/src/yggdrasil/io/primitive/parquet_io.py``) and the raw
``pq.write_table`` baseline. The goal is to make the cost of each
write-side knob visible per release, not to lock in a hard SLA.

Two modes:

* **Regression mode (default)** — runs with a small input (~4 k rows)
  so the suite stays fast (<1 s). Asserts that every combination
  actually produces a valid Parquet file. This catches breakage from
  an option being renamed / dropped without forcing every contributor
  to wait for a real benchmark.
* **Benchmark mode** — opt-in via ``YGG_BENCHMARK=1``. Pumps row counts
  up (~250 k rows by default, override with ``YGG_BENCHMARK_ROWS``)
  and prints a per-axis table:

  ::

      compression=snappy  level=None   rows=250000  ms=  43.21  bytes=2381204
      compression=gzip    level=None   rows=250000  ms= 187.65  bytes=1402113
      compression=zstd    level=3      rows=250000  ms=  61.40  bytes=1389877
      ...

  Read the table top-to-bottom: rows where ``ms`` jumps and ``bytes``
  doesn't change much are the obvious knobs to tune; rows where both
  ``ms`` and ``bytes`` move together are correctness-of-codec, not
  perf bugs.

Why this lives next to ``test_parquet_io.py``
---------------------------------------------
The benchmarks exercise the same code path the IO tests cover —
keeping them adjacent means a write-side regression that breaks the
codec is caught by the IO tests in the same run, before the benchmark
numbers go misleading.
"""
from __future__ import annotations

import os
import random
import string
import time
import unittest
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.io.primitive.parquet_file import ParquetFile, ParquetOptions


# ---------------------------------------------------------------------------
# Mode + sizing
# ---------------------------------------------------------------------------


def _benchmark_mode() -> bool:
    """``YGG_BENCHMARK=1`` flips on the big-input benchmark numbers.

    The default mode keeps row counts low so the harness runs as a
    regression test (correctness only). Set the env var to surface
    actual wall-clock and byte counts.
    """
    return os.environ.get("YGG_BENCHMARK", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _bench_rows() -> int:
    """Row count for benchmark mode, overridable via ``YGG_BENCHMARK_ROWS``."""
    raw = os.environ.get("YGG_BENCHMARK_ROWS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 250_000


def _regression_rows() -> int:
    """Row count for the fast regression path — keeps the suite snappy."""
    return 4_000


# ---------------------------------------------------------------------------
# Test data factories — keep them deterministic so byte counts compare
# ---------------------------------------------------------------------------


def _make_numeric_table(n: int, seed: int = 0) -> pa.Table:
    """Numeric-heavy frame — dictionary encoding has nothing to bite on."""
    rnd = random.Random(seed)
    return pa.table({
        "id": pa.array(range(n), type=pa.int64()),
        "x": pa.array([rnd.uniform(-1.0, 1.0) for _ in range(n)], type=pa.float64()),
        "y": pa.array([rnd.randint(-1_000_000, 1_000_000) for _ in range(n)], type=pa.int32()),
        "z": pa.array([rnd.random() for _ in range(n)], type=pa.float32()),
    })


def _make_string_table(n: int, *, cardinality: int = 32, seed: int = 0) -> pa.Table:
    """String-heavy frame with low cardinality — dictionary encoding wins big."""
    rnd = random.Random(seed)
    pool = ["".join(rnd.choices(string.ascii_lowercase, k=8)) for _ in range(cardinality)]
    return pa.table({
        "id": pa.array(range(n), type=pa.int64()),
        "label": pa.array([rnd.choice(pool) for _ in range(n)], type=pa.string()),
        "category": pa.array([rnd.choice(pool) for _ in range(n)], type=pa.string()),
    })


def _make_mixed_table(n: int, seed: int = 0) -> pa.Table:
    rnd = random.Random(seed)
    return pa.table({
        "id": pa.array(range(n), type=pa.int64()),
        "v": pa.array([rnd.uniform(0, 100) for _ in range(n)], type=pa.float64()),
        "tag": pa.array(
            [rnd.choice(["A", "B", "C", "D"]) for _ in range(n)], type=pa.string()
        ),
        "flag": pa.array([rnd.random() < 0.5 for _ in range(n)], type=pa.bool_()),
    })


# ---------------------------------------------------------------------------
# Bench primitives
# ---------------------------------------------------------------------------


def _time_write(write_fn) -> tuple[float, int]:
    """Run ``write_fn`` once, return ``(elapsed_ms, written_bytes)``.

    ``write_fn`` should write to an in-memory buffer and return its bytes.
    """
    t0 = time.perf_counter()
    payload = write_fn()
    elapsed = (time.perf_counter() - t0) * 1000.0
    return elapsed, len(payload)


def _write_via_parquet_io(
    table: pa.Table,
    *,
    compression: "str | None" = "snappy",
    compression_level: "int | None" = None,
    use_dictionary: bool = True,
    write_statistics: bool = True,
    row_group_size: "int | None" = None,
    use_threads: bool = True,
    batches: "Iterable[pa.RecordBatch] | None" = None,
) -> bytes:
    """Write ``table`` through the production :class:`ParquetFile` writer.

    Routes through ``_write_arrow_batches`` so the benchmarks track the
    same code path real callers hit — not just ``pq.write_table``.
    """
    io_obj = ParquetFile()
    opts = ParquetOptions(
        compression=compression,
        compression_level=compression_level,
        use_dictionary=use_dictionary,
        write_statistics=write_statistics,
        row_group_size=row_group_size,
        use_threads=use_threads,
    )
    src = batches if batches is not None else table.to_batches()
    io_obj._write_arrow_batches(src, opts)
    return bytes(io_obj.getvalue())


def _write_via_pq_write_table(
    table: pa.Table,
    *,
    compression: "str | None" = "snappy",
    compression_level: "int | None" = None,
    use_dictionary: bool = True,
    write_statistics: bool = True,
    row_group_size: "int | None" = None,
) -> bytes:
    """Raw ``pq.write_table`` baseline — no ygg wrapper overhead.

    Difference vs :func:`_write_via_parquet_io` reflects what the
    :class:`ParquetFile` layer is adding on top of pyarrow.
    """
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=use_dictionary,
        write_statistics=write_statistics,
        row_group_size=row_group_size,
    )
    return sink.getvalue().to_pybytes()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class _BenchReport:
    """Tiny perf-row accumulator. Prints a flat table on flush.

    The format is deliberately stable line-oriented text — easy to
    grep, easy to diff between runs, easy to paste into a PR body.
    """

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


class TestParquetWriteCompressionCodecs(ArrowTestCase):
    """Every supported compression codec produces a round-trippable file.

    Doubles as the "this option still exists in pyarrow" canary: if a
    codec gets renamed upstream (lz4_raw vs lz4, snappy vs SNAPPY)
    the benchmark sweep below would silently skip it — this test
    fails first.
    """

    def _round_trip(self, table: pa.Table, **kwargs: Any) -> pa.Table:
        payload = _write_via_parquet_io(table, **kwargs)
        return pq.read_table(pa.BufferReader(payload))

    def test_no_compression(self) -> None:
        table = _make_numeric_table(_regression_rows())
        out = self._round_trip(table, compression=None)
        self.assertEqual(out.num_rows, table.num_rows)

    def test_snappy(self) -> None:
        table = _make_numeric_table(_regression_rows())
        out = self._round_trip(table, compression="snappy")
        self.assertEqual(out.num_rows, table.num_rows)

    def test_gzip(self) -> None:
        table = _make_numeric_table(_regression_rows())
        out = self._round_trip(table, compression="gzip")
        self.assertEqual(out.num_rows, table.num_rows)

    def test_zstd_default_level(self) -> None:
        table = _make_numeric_table(_regression_rows())
        out = self._round_trip(table, compression="zstd")
        self.assertEqual(out.num_rows, table.num_rows)

    def test_zstd_level_explicit(self) -> None:
        table = _make_numeric_table(_regression_rows())
        out = self._round_trip(table, compression="zstd", compression_level=9)
        self.assertEqual(out.num_rows, table.num_rows)


class TestParquetWriteKnobs(ArrowTestCase):
    """Each write-side knob (use_dictionary, write_statistics,
    row_group_size, use_threads) still flows through to pyarrow.
    A misnamed option here would fail loudly — pyarrow ignores unknown
    kwargs silently in some versions.
    """

    def _table(self) -> pa.Table:
        return _make_string_table(_regression_rows())

    def test_use_dictionary_false_is_accepted(self) -> None:
        payload = _write_via_parquet_io(self._table(), use_dictionary=False)
        self.assertGreater(len(payload), 0)

    def test_write_statistics_false_is_accepted(self) -> None:
        payload = _write_via_parquet_io(self._table(), write_statistics=False)
        self.assertGreater(len(payload), 0)

    def test_row_group_size_small(self) -> None:
        # Small row group size → many row groups → larger footer.
        table = self._table()
        payload = _write_via_parquet_io(table, row_group_size=256)
        pf = pq.ParquetFile(pa.BufferReader(payload))
        self.assertGreater(pf.num_row_groups, 1)

    def test_row_group_size_none(self) -> None:
        # No explicit size → pyarrow default → single row group on
        # input of this size.
        table = self._table()
        payload = _write_via_parquet_io(table, row_group_size=None)
        pf = pq.ParquetFile(pa.BufferReader(payload))
        self.assertEqual(pf.num_row_groups, 1)


# ---------------------------------------------------------------------------
# Benchmark-mode sweeps — gated by YGG_BENCHMARK=1
# ---------------------------------------------------------------------------


@unittest.skipUnless(_benchmark_mode(), "set YGG_BENCHMARK=1 to run benchmarks")
class TestParquetWriteBenchmarks(ArrowTestCase):
    """Parameter sweeps — pure measurement, no assertions on perf.

    These are run on demand. The output is a printed report; the test
    body asserts only that each branch produced bytes (so a broken
    sweep still fails loudly), and that the byte count is non-zero.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.n_rows = _bench_rows()
        cls.numeric = _make_numeric_table(cls.n_rows)
        cls.strings = _make_string_table(cls.n_rows)
        cls.mixed = _make_mixed_table(cls.n_rows)

    # -- codec sweep ---------------------------------------------------------

    def test_compression_codecs_numeric(self) -> None:
        report = _BenchReport(
            f"compression sweep (numeric, n={self.n_rows}) — "
            f"parquet_io.py:_write_arrow_batches"
        )
        for codec in (None, "snappy", "gzip", "zstd", "lz4"):
            label = f"codec={str(codec):8s}"
            ms, nbytes = _time_write(
                lambda c=codec: _write_via_parquet_io(self.numeric, compression=c),
            )
            report.add(label, ms, nbytes)
            self.assertGreater(nbytes, 0)
        report.flush()

    def test_compression_codecs_strings(self) -> None:
        report = _BenchReport(
            f"compression sweep (low-card strings, n={self.n_rows}) — "
            f"parquet_io.py:_write_arrow_batches"
        )
        for codec in (None, "snappy", "gzip", "zstd", "lz4"):
            label = f"codec={str(codec):8s}"
            ms, nbytes = _time_write(
                lambda c=codec: _write_via_parquet_io(self.strings, compression=c),
            )
            report.add(label, ms, nbytes)
            self.assertGreater(nbytes, 0)
        report.flush()

    # -- zstd level sweep ----------------------------------------------------

    def test_zstd_level_sweep(self) -> None:
        report = _BenchReport(
            f"zstd compression_level sweep (mixed, n={self.n_rows})"
        )
        for level in (1, 3, 5, 9, 15):
            ms, nbytes = _time_write(
                lambda lv=level: _write_via_parquet_io(
                    self.mixed, compression="zstd", compression_level=lv,
                ),
            )
            report.add(f"zstd level={level:2d}", ms, nbytes)
            self.assertGreater(nbytes, 0)
        report.flush()

    # -- row_group_size sweep -----------------------------------------------

    def test_row_group_size_sweep(self) -> None:
        report = _BenchReport(
            f"row_group_size sweep (mixed, n={self.n_rows})"
        )
        for rgs in (1024, 8192, 65_536, 262_144, None):
            ms, nbytes = _time_write(
                lambda r=rgs: _write_via_parquet_io(self.mixed, row_group_size=r),
            )
            report.add(f"row_group_size={str(rgs):>8s}", ms, nbytes)
            self.assertGreater(nbytes, 0)
        report.flush()

    # -- dictionary / statistics --------------------------------------------

    def test_use_dictionary_and_statistics(self) -> None:
        report = _BenchReport(
            f"use_dictionary x write_statistics (low-card strings, n={self.n_rows})"
        )
        for use_dict in (True, False):
            for write_stats in (True, False):
                ms, nbytes = _time_write(
                    lambda ud=use_dict, ws=write_stats: _write_via_parquet_io(
                        self.strings,
                        use_dictionary=ud,
                        write_statistics=ws,
                    ),
                )
                report.add(
                    f"use_dict={str(use_dict):<5s} stats={str(write_stats):<5s}",
                    ms, nbytes,
                )
                self.assertGreater(nbytes, 0)
        report.flush()

    # -- threading -----------------------------------------------------------

    def test_use_threads(self) -> None:
        report = _BenchReport(
            f"use_threads (mixed, n={self.n_rows})"
        )
        for ut in (True, False):
            ms, nbytes = _time_write(
                lambda u=ut: _write_via_parquet_io(self.mixed, use_threads=u),
            )
            report.add(f"use_threads={ut!s:<5s}", ms, nbytes)
            self.assertGreater(nbytes, 0)
        report.flush()

    # -- ParquetFile wrapper vs raw pq.write_table ----------------------------

    def test_parquet_io_vs_pq_write_table(self) -> None:
        """How much overhead does the IO wrapper add over raw pyarrow?

        Numbers here drive whether ``_write_arrow_batches`` is worth
        any extra optimization vs. just delegating to ``pq.write_table``
        on the OVERWRITE path.
        """
        report = _BenchReport(
            f"ParquetFile vs raw pq.write_table (mixed, n={self.n_rows})"
        )
        ms1, nb1 = _time_write(lambda: _write_via_parquet_io(self.mixed))
        report.add("ParquetFile._write_arrow_batches", ms1, nb1)
        ms2, nb2 = _time_write(lambda: _write_via_pq_write_table(self.mixed))
        report.add("pq.write_table              ", ms2, nb2)
        self.assertGreater(nb1, 0)
        self.assertGreater(nb2, 0)
        report.flush()

    # -- single batch vs many batches ---------------------------------------

    def test_batch_chunking_effect(self) -> None:
        """How much does batch chunking cost vs a single batch?

        The production writer iterates batches; ``ParquetWriter.write_batch``
        runs once per batch. Tiny batches amortize encoder setup poorly.
        """
        report = _BenchReport(
            f"batch chunking (mixed, n={self.n_rows})"
        )
        single = self.mixed.combine_chunks().to_batches()
        ms1, nb1 = _time_write(
            lambda: _write_via_parquet_io(self.mixed, batches=single),
        )
        report.add("batches=1 (combined)            ", ms1, nb1)

        for bs in (4_096, 16_384, 65_536):
            batches = self.mixed.to_batches(max_chunksize=bs)
            n = len(batches)
            ms, nbytes = _time_write(
                lambda b=batches: _write_via_parquet_io(self.mixed, batches=b),
            )
            report.add(f"batches={n:<4d} (max_chunksize={bs})", ms, nbytes)
            self.assertGreater(nbytes, 0)
        report.flush()
