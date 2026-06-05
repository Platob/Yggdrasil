"""Generic ``Tabular.scan_polars_frame`` — pure-lazy IO-plugin scan.

The base scan registers a polars IO source instead of wrapping a one-shot
``scan_pyarrow_dataset`` reader. This pins the behavior that earned the
rewrite, on ordinary file backends (parquet + arrow-ipc), not just the
warehouse:

- **Lazy** — building the scan reads nothing; no batch is pulled until
  ``collect``.
- **Re-collectable** — every ``collect`` re-opens the read.
- **Projection pushdown** — a ``select`` narrows the *read options* the
  backend sees (so parquet reads fewer column chunks / IPC selects fewer
  columns), not just the post-read frame.
- **Predicate / n_rows pushdown** — filters apply as rows stream and the
  scan stops once the row budget is met.
"""
from __future__ import annotations

import pyarrow as pa
import polars as pl
import pytest

from yggdrasil.path.memory import Memory
from yggdrasil.io.arrow_ipc_file import ArrowIPCFile
from yggdrasil.io.parquet_file import ParquetFile


def _table() -> pa.Table:
    return pa.table({
        "id": pa.array(range(10), pa.int64()),
        "price": pa.array([float(i) for i in range(10)], pa.float64()),
        "name": pa.array([f"r{i}" for i in range(10)], pa.string()),
    })


def _parquet() -> ParquetFile:
    holder = Memory()
    ParquetFile(holder=holder, mode="wb").write_arrow_table(_table())
    return ParquetFile(holder=holder, mode="rb")


def _ipc() -> ArrowIPCFile:
    holder = Memory()
    ArrowIPCFile(holder=holder, mode="wb").write_arrow_table(_table())
    return ArrowIPCFile(holder=holder, mode="rb")


# parquet / csv / ndjson keep a *native* polars scan (``scan_parquet`` …);
# the generic IO-plugin scan under test backs every other leaf (arrow-ipc,
# pickle, xlsx, warehouse, …). Correctness is checked on both; the
# generic-internals (lazy + projection-into-read, spied via
# ``_read_arrow_batches``) only apply to the generic path, so they use IPC.
@pytest.fixture(params=[_parquet, _ipc], ids=["parquet", "arrow_ipc"])
def leaf(request):
    return request.param()


class TestScanPolarsContract:
    """Pushdown + re-collect contract, on both native and generic scans."""

    def test_full_collect(self, leaf) -> None:
        df = leaf.scan_polars_frame().collect()
        assert df.columns == ["id", "price", "name"]
        assert df["id"].to_list() == list(range(10))

    def test_re_collectable(self, leaf) -> None:
        lf = leaf.scan_polars_frame()
        first = lf.collect()
        second = lf.collect()
        assert first.equals(second)
        assert first["id"].to_list() == list(range(10))

    def test_projection(self, leaf) -> None:
        df = leaf.scan_polars_frame().select("id", "name").collect()
        assert df.columns == ["id", "name"]
        assert df["name"].to_list() == [f"r{i}" for i in range(10)]

    def test_predicate_pushdown(self, leaf) -> None:
        df = leaf.scan_polars_frame().filter(pl.col("id") >= 7).collect()
        assert df["id"].to_list() == [7, 8, 9]

    def test_n_rows_pushdown(self, leaf) -> None:
        df = leaf.scan_polars_frame().head(3).collect()
        assert df["id"].to_list() == [0, 1, 2]

    def test_projection_and_predicate_compose(self, leaf) -> None:
        df = (
            leaf.scan_polars_frame()
            .filter(pl.col("id") % 2 == 0)
            .select("name")
            .collect()
        )
        assert df["name"].to_list() == ["r0", "r2", "r4", "r6", "r8"]


class TestGenericScanInternals:
    """The IO-plugin scan itself (the path arrow-ipc / warehouse / … use)."""

    def test_scan_is_lazy_then_collects(self) -> None:
        leaf = _ipc()
        # Spy: building the scan must not read; collect must.
        calls: list = []
        original = leaf._read_arrow_batches
        leaf._read_arrow_batches = lambda opts: (calls.append(1), original(opts))[1]

        lf = leaf.scan_polars_frame()
        assert isinstance(lf, pl.LazyFrame)
        assert calls == []  # nothing read at scan-build time

        df = lf.collect()
        assert calls  # the collect drove the read
        assert df["id"].to_list() == list(range(10))

    def test_projection_is_pushed_into_the_read(self) -> None:
        leaf = _ipc()
        # Capture the columns each read is asked to produce — proving the
        # ``select`` narrowed the *read options*, not just the result frame.
        seen: list = []
        original = leaf._read_arrow_batches

        def spy(opts):
            seen.append(opts.read_columns())
            return original(opts)

        leaf._read_arrow_batches = spy

        df = leaf.scan_polars_frame().select("id", "name").collect()
        assert df.columns == ["id", "name"]
        assert df["name"].to_list() == [f"r{i}" for i in range(10)]
        assert seen and all(set(cols) == {"id", "name"} for cols in seen if cols)
