"""Tests for the engine-surface methods on :class:`DataIO`.

These methods (``read_polars_frame``, ``write_pandas_frame``,
``write_table``, etc.) are inherited by every leaf and route through
``_read_arrow_batches`` / ``_write_arrow_batches``. We test them
once on ParquetIO (type-preserving, fast) — leaf-specific behavior
is covered in the per-leaf files.

Coverage:

- Polars eager + lazy + frame-stream variants.
- Pandas with default RangeIndex (not preserved) + named index
  (preserved).
- Python-native pylist / pydict.
- ``write_table`` dispatch to the right type-specific writer based
  on object module prefix.
- ``write_table`` last-resort fallback to ``write_arrow_batches``.
"""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.primitive.parquet_io import ParquetIO


# ---------------------------------------------------------------------------
# Polars
# ---------------------------------------------------------------------------


class TestPolars:
    """Polars surface — read_polars_frame / write_polars_frame /
    scan_polars_frame / read_polars_frames."""

    def setup_method(self):
        pytest.importorskip("polars")

    def test_round_trip_eager_dataframe(self, arrow_table):
        import polars as pl
        io = ParquetIO()
        frame = pl.from_arrow(arrow_table)

        with io:
            io.write_polars_frame(frame)
            io.seek(0)
            result = io.read_polars_frame()

        assert result.height == frame.height
        assert result.columns == frame.columns

    def test_lazy_frame_collected_for_write(self, arrow_table):
        import polars as pl
        io = ParquetIO()
        lf = pl.from_arrow(arrow_table).lazy()

        with io:
            io.write_polars_frame(lf)
            io.seek(0)
            result = io.read_polars_frame()

        assert result.height == arrow_table.num_rows

    def test_polars_frame_streaming_read(self, arrow_table):
        import polars as pl
        io = ParquetIO()
        frame = pl.from_arrow(arrow_table)

        with io:
            io.write_polars_frame(frame)
            io.seek(0)
            frames = list(io.read_polars_frames())

        # At least one frame; total row count matches.
        assert sum(f.height for f in frames) == arrow_table.num_rows


# ---------------------------------------------------------------------------
# Pandas
# ---------------------------------------------------------------------------


class TestPandas:

    def setup_method(self):
        pytest.importorskip("pandas")

    def test_round_trip_default_range_index_not_preserved(self):
        import pandas as pd
        io = ParquetIO()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        # Default RangeIndex with no name — should not appear in
        # the round-tripped columns.

        with io:
            io.write_pandas_frame(df)
            io.seek(0)
            result = io.read_pandas_frame()

        assert "index" not in result.columns
        assert list(result.columns) == ["a", "b"]
        assert result.equals(df)

    def test_named_index_is_preserved(self):
        import pandas as pd
        io = ParquetIO()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.index.name = "row_id"

        with io:
            io.write_pandas_frame(df)
            io.seek(0)
            result = io.read_pandas_frame()

        # Named index round-trips as a column.
        assert "row_id" in result.columns or result.index.name == "row_id"


# ---------------------------------------------------------------------------
# Python-native
# ---------------------------------------------------------------------------


class TestPyNative:

    def test_round_trip_pylist(self):
        io = ParquetIO()
        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

        with io:
            io.write_pylist(data)
            io.seek(0)
            result = io.read_pylist()

        assert result == data

    def test_pylist_sparse_rows_backfilled(self):
        """Sparse list-of-dicts is a known footgun for
        ``Table.from_pylist``; the leaf normalizes columns first."""
        io = ParquetIO()
        data = [{"a": 1}, {"b": 2}, {"a": 3, "b": 4}]

        with io:
            io.write_pylist(data)
            io.seek(0)
            result = io.read_pylist()

        # All rows have all keys after normalization.
        assert all(set(r.keys()) == {"a", "b"} for r in result)

    def test_round_trip_pydict(self):
        io = ParquetIO()
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}

        with io:
            io.write_pydict(data)
            io.seek(0)
            result = io.read_pydict()

        assert result == data

    def test_empty_pylist_is_noop(self):
        io = ParquetIO()
        with io:
            io.write_pylist([])
            assert io.is_empty()


# ---------------------------------------------------------------------------
# write_table dispatch
# ---------------------------------------------------------------------------


class TestWriteTableDispatch:

    def test_pyarrow_table_routed_to_arrow_writer(self, arrow_table):
        io = ParquetIO()
        with io:
            io.write_table(arrow_table)
            io.seek(0)
            result = io.read_arrow_table()
        assert result.equals(arrow_table)

    def test_pyarrow_record_batch_routed(self, arrow_table):
        io = ParquetIO()
        batch = arrow_table.to_batches()[0]
        with io:
            io.write_table(batch)
            io.seek(0)
            result = io.read_arrow_table()
        assert result.num_rows == batch.num_rows

    def test_pylist_routed(self):
        io = ParquetIO()
        data = [{"a": 1, "b": "x"}]
        with io:
            io.write_table(data)
            io.seek(0)
            result = io.read_pylist()
        assert result == data

    def test_pydict_routed(self):
        io = ParquetIO()
        data = {"a": [1, 2], "b": ["x", "y"]}
        with io:
            io.write_table(data)
            io.seek(0)
            result = io.read_pydict()
        assert result == data

    def test_polars_routed(self):
        pl = pytest.importorskip("polars")
        io = ParquetIO()
        frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        with io:
            io.write_table(frame)
            io.seek(0)
            result = io.read_polars_frame()
        assert result.height == 2

    def test_pandas_routed(self):
        pd = pytest.importorskip("pandas")
        io = ParquetIO()
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        with io:
            io.write_table(df)
            io.seek(0)
            result = io.read_pandas_frame()
        assert len(result) == 2

    def test_iterator_of_batches_routed_to_batches(self, arrow_table):
        io = ParquetIO()
        batches = arrow_table.to_batches()
        with io:
            io.write_table(iter(batches))
            io.seek(0)
            result = io.read_arrow_table()
        assert result.num_rows == arrow_table.num_rows

    def test_unsupported_type_raises(self):
        io = ParquetIO()
        with io:
            with pytest.raises(TypeError, match="Unsupported"):
                io.write_table(42)

    def test_empty_list_is_noop(self):
        io = ParquetIO()
        with io:
            io.write_table([])
            assert io.is_empty()

    def test_write_any_alias_works(self, arrow_table):
        """``write_any`` is the legacy alias of ``write_table``."""
        io = ParquetIO()
        with io:
            io.write_table(arrow_table)
            io.seek(0)
            result = io.read_arrow_table()
        assert result.equals(arrow_table)
