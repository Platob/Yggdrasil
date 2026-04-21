"""Unit tests for :class:`yggdrasil.io.buffer.media_io.MediaIO`.

Covers the abstract-base contract via a minimal concrete subclass:

* options resolution (``check_options`` with ``None`` and kwargs)
* open/close/context-manager semantics
* compression transparency (codec-aware open/close, mark_dirty flush-back)
* save-mode guard (IGNORE, ERROR_IF_EXISTS, default)
* ``_normalize_records`` backfill
* ``iter_arrow_batches`` dispatch
* ``read_arrow_table`` with/without ``batch_size``
* ``write_table`` dispatch across all accepted input shapes
* ``write_pylist`` on sparse/heterogeneous rows
* ``MediaIO.make`` factory routing
* ``collect_schema`` default fallback + ``_collect_arrow_schema`` override hook
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import pyarrow as pa
import pytest

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.media_options import MediaOptions
from yggdrasil.io.enums import MediaType, SaveMode


# =====================================================================
# Minimal concrete subclass
# =====================================================================
#
# Stores Arrow record batches as IPC stream bytes in the underlying
# BytesIO. This is the simplest roundtrip-capable format I can build
# without reaching into a real MediaIO subclass — it lets us exercise
# the base-class plumbing without depending on Parquet/JSON/etc.

import pyarrow.ipc as _ipc
import io as _stdio


@dataclass
class _MockOptions(MediaOptions):
    """Trivial options subclass — just inherits everything."""

    extra_flag: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.extra_flag, bool):
            raise TypeError("extra_flag must be bool")


@dataclass(slots=True)
class _MockMediaIO(MediaIO[_MockOptions]):
    """Concrete MediaIO that stores Arrow IPC stream bytes.

    Kept deliberately minimal — just enough to exercise the base-class
    plumbing. Each read/write opens its own context via ``with self:``,
    matching the convention used by real MediaIO subclasses
    (ParquetIO, IPCIO, etc.).
    """

    @classmethod
    def check_options(
        cls,
        options: Optional[_MockOptions],
        *args,
        **kwargs,
    ) -> _MockOptions:
        return _MockOptions.check_parameters(options=options, **kwargs)

    def _read_arrow_batches(
        self,
        options: _MockOptions,
    ) -> Iterator["pa.RecordBatch"]:
        with self:
            data = self.buffer.to_bytes()
            if not data:
                return
            reader = _ipc.open_stream(pa.BufferReader(data))
            for batch in reader:
                if options.columns is not None:
                    batch = batch.select(
                        [c for c in options.columns if c in batch.schema.names]
                    )
                if options.ignore_empty and batch.num_rows == 0:
                    continue
                yield batch

    def _write_arrow_batches(
        self,
        batches: Iterator["pa.RecordBatch"],
        options: _MockOptions,
    ) -> None:
        with self:
            if self.skip_write(options.mode):
                return

            first = next(batches, None)
            if first is None:
                return

            # Build the full IPC stream bytes into a sink, then replace
            # the buffer in one shot. No append semantics here — this
            # mock treats every write as an overwrite, which is fine
            # for exercising the base-class contract.
            sink = _stdio.BytesIO()
            with _ipc.new_stream(sink, first.schema) as writer:
                writer.write_batch(first)
                for batch in batches:
                    if batch.num_rows == 0 and options.ignore_empty:
                        continue
                    writer.write_batch(batch)

            self.buffer.truncate(0)
            self.buffer.write_bytes(sink.getvalue())
            self.mark_dirty()


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture()
def sample_table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


@pytest.fixture()
def mock_io(tmp_path: Path) -> _MockMediaIO:
    """Fresh mock IO over an in-memory BytesIO."""
    buf = BytesIO()
    media = MediaType.parse("data.arrow", default=MediaType(MimeTypes.ARROW_IPC))
    return _MockMediaIO(media_type=media, holder=buf)


# =====================================================================
# Options resolution
# =====================================================================

class TestOptionsResolution:
    def test_check_options_none_returns_default(self):
        opt = _MockMediaIO.check_options(None)
        assert isinstance(opt, _MockOptions)
        assert opt.extra_flag is False

    def test_check_options_merges_kwargs(self):
        opt = _MockMediaIO.check_options(None, extra_flag=True)
        assert opt.extra_flag is True

    def test_check_options_preserves_instance(self):
        """Passing an existing instance with overrides should merge."""
        base = _MockOptions(extra_flag=True)
        opt = _MockMediaIO.check_options(base, columns=["x"])
        assert opt.extra_flag is True
        assert list(opt.columns) == ["x"]


# =====================================================================
# Open / close / context manager
# =====================================================================

class TestOpenClose:
    def test_initial_state_is_closed(self, mock_io: _MockMediaIO):
        assert mock_io.closed is True
        assert mock_io.opened is False
        assert mock_io.buffer is None

    def test_open_attaches_buffer(self, mock_io: _MockMediaIO):
        mock_io.open()
        try:
            assert mock_io.opened is True
            assert mock_io.buffer is mock_io.holder
        finally:
            mock_io.close()

    def test_close_detaches_buffer(self, mock_io: _MockMediaIO):
        mock_io.open()
        mock_io.close()
        assert mock_io.closed is True
        assert mock_io.buffer is None

    def test_close_is_idempotent(self, mock_io: _MockMediaIO):
        mock_io.open()
        mock_io.close()
        mock_io.close()  # second close should not raise

    def test_double_open_raises(self, mock_io: _MockMediaIO):
        mock_io.open()
        try:
            with pytest.raises(RuntimeError, match="already open"):
                mock_io.open()
        finally:
            mock_io.close()

    def test_context_manager_opens_and_closes(self, mock_io: _MockMediaIO):
        with mock_io as m:
            assert m.opened
        assert mock_io.closed

    def test_context_manager_reenter_is_noop_when_open(self, mock_io: _MockMediaIO):
        mock_io.open()
        try:
            with mock_io as m:  # already open — must not raise
                assert m.opened
        finally:
            if mock_io.opened:
                mock_io.close()


# =====================================================================
# Compression transparency
# =====================================================================

class TestCompressionTransparency:
    def test_non_compressed_shares_holder(self, mock_io: _MockMediaIO):
        """Without a codec, buffer IS the holder (shared storage)."""
        mock_io.open()
        try:
            assert mock_io.buffer is mock_io.holder
        finally:
            mock_io.close()

    def test_compressed_buffer_is_detached(self):
        """With a codec, open() returns a decompressed COPY."""
        from yggdrasil.io import ZSTD

        # Create a buffer and compress it.
        buf = BytesIO()
        buf.write_bytes(b"some payload")
        buf = buf.compress(codec=ZSTD, copy=True)
        media = MediaType.parse(
            "application/vnd.apache.arrow.stream+zstd",
            default=None,
        )
        if media is None or media.codec is None:
            pytest.skip("media+zstd mime not recognized in this env")

        io_ = _MockMediaIO(media_type=media, holder=buf)
        io_.open()
        try:
            # Buffer is a fresh decompressed object — NOT the holder.
            assert io_.buffer is not io_.holder
        finally:
            io_.close()


# =====================================================================
# Save-mode guard
# =====================================================================

class TestSkipWrite:
    def test_empty_buffer_never_skips(self, mock_io: _MockMediaIO):
        mock_io.open()
        try:
            # All modes on empty buffer → False (write proceeds).
            assert mock_io.skip_write(SaveMode.OVERWRITE) is False
            assert mock_io.skip_write(SaveMode.IGNORE) is False
            assert mock_io.skip_write(SaveMode.ERROR_IF_EXISTS) is False
            assert mock_io.skip_write(SaveMode.APPEND) is False
        finally:
            mock_io.close()

    def test_ignore_on_non_empty_buffer_skips(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_arrow_table(sample_table)
        mock_io.open()
        try:
            assert mock_io.skip_write(SaveMode.IGNORE) is True
        finally:
            mock_io.close()

    def test_error_if_exists_on_non_empty_raises(
        self, mock_io: _MockMediaIO, sample_table
    ):
        mock_io.write_arrow_table(sample_table)
        mock_io.open()
        try:
            with pytest.raises(IOError):
                mock_io.skip_write(SaveMode.ERROR_IF_EXISTS)
        finally:
            mock_io.close()

    def test_overwrite_on_non_empty_does_not_skip(
        self, mock_io: _MockMediaIO, sample_table
    ):
        mock_io.write_arrow_table(sample_table)
        mock_io.open()
        try:
            assert mock_io.skip_write(SaveMode.OVERWRITE) is False
        finally:
            mock_io.close()


# =====================================================================
# _normalize_records
# =====================================================================

class TestNormalizeRecords:
    def test_empty_list(self):
        assert MediaIO._normalize_records([]) == []

    def test_homogeneous_rows_passthrough(self):
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        # No backfill needed — same-key rows short-circuit.
        result = MediaIO._normalize_records(rows)
        assert result == rows

    def test_sparse_rows_backfilled_with_none(self):
        rows = [{"a": 1}, {"b": 2}, {"a": 3, "b": 4}]
        result = MediaIO._normalize_records(rows)
        # Union of keys: {"a", "b"}; each row has both.
        assert result == [
            {"a": 1, "b": None},
            {"a": None, "b": 2},
            {"a": 3, "b": 4},
        ]

    def test_first_seen_key_order_preserved(self):
        rows = [{"b": 1, "a": 2}, {"c": 3, "a": 4}]
        result = MediaIO._normalize_records(rows)
        # First row introduces "b", "a"; second adds "c". Order: b, a, c.
        assert list(result[0].keys()) == ["b", "a", "c"]

    def test_none_row_backfilled_entirely(self):
        rows = [{"a": 1, "b": 2}, None, {"a": 3, "b": 4}]
        result = MediaIO._normalize_records(rows)
        assert result[1] == {"a": None, "b": None}

    def test_iterable_input_accepted(self):
        def gen():
            yield {"a": 1}
            yield {"b": 2}
        result = MediaIO._normalize_records(gen())
        assert result == [{"a": 1, "b": None}, {"a": None, "b": 2}]


# =====================================================================
# iter_arrow_batches dispatch
# =====================================================================

class TestIterArrowBatches:
    def test_table(self, sample_table):
        batches = list(MediaIO.iter_arrow_batches(sample_table))
        assert len(batches) >= 1
        assert sum(b.num_rows for b in batches) == sample_table.num_rows

    def test_record_batch(self, sample_table):
        batch = sample_table.to_batches()[0]
        batches = list(MediaIO.iter_arrow_batches(batch))
        assert batches == [batch]

    def test_column_oriented_dict(self):
        d = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        batches = list(MediaIO.iter_arrow_batches(d))
        assert sum(b.num_rows for b in batches) == 3

    def test_iterator_of_tables(self, sample_table):
        def gen():
            yield sample_table
            yield sample_table
        batches = list(MediaIO.iter_arrow_batches(gen()))
        assert sum(b.num_rows for b in batches) == 6

    @pytest.mark.skipif(not HAS_POLARS, reason="polars required")
    def test_polars_dataframe(self):
        df = pl.DataFrame({"id": [1, 2, 3]})
        batches = list(MediaIO.iter_arrow_batches(df))
        assert sum(b.num_rows for b in batches) == 3

    @pytest.mark.skipif(not HAS_POLARS, reason="polars required")
    def test_polars_lazyframe_collected(self):
        df = pl.DataFrame({"id": [1, 2, 3]}).lazy()
        batches = list(MediaIO.iter_arrow_batches(df))
        assert sum(b.num_rows for b in batches) == 3

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
    def test_pandas_dataframe(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        batches = list(MediaIO.iter_arrow_batches(df))
        assert sum(b.num_rows for b in batches) == 3

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            list(MediaIO.iter_arrow_batches(42))  # type: ignore[arg-type]


# =====================================================================
# read_arrow_table / _read_arrow_table
# =====================================================================

class TestReadArrowTable:
    def test_returns_pa_table_by_default(
        self, mock_io: _MockMediaIO, sample_table
    ):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.read_arrow_table()
        assert isinstance(out, pa.Table)
        assert out.num_rows == sample_table.num_rows

    def test_empty_buffer_returns_empty_table(self, mock_io: _MockMediaIO):
        out = mock_io.read_arrow_table()
        assert isinstance(out, pa.Table)
        assert out.num_rows == 0

    def test_batch_size_returns_iterator(
        self, mock_io: _MockMediaIO, sample_table
    ):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.read_arrow_table(batch_size=2)
        # Must be an iterator, not a Table.
        assert not isinstance(out, pa.Table)
        tables = list(out)
        assert all(isinstance(t, pa.Table) for t in tables)
        assert sum(t.num_rows for t in tables) == sample_table.num_rows

    def test_columns_projects(
        self, mock_io: _MockMediaIO, sample_table
    ):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.read_arrow_table(columns=["id"])
        assert out.column_names == ["id"]


# =====================================================================
# write_table dispatch
# =====================================================================

class TestWriteTableDispatch:
    def test_arrow_table(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_table(sample_table)
        out = mock_io.read_arrow_table()
        assert out.num_rows == sample_table.num_rows

    def test_record_batch(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_table(sample_table.to_batches()[0])
        out = mock_io.read_arrow_table()
        assert out.num_rows == sample_table.num_rows

    @pytest.mark.skipif(not HAS_POLARS, reason="polars required")
    def test_polars_dataframe(self, mock_io: _MockMediaIO):
        df = pl.DataFrame({"id": [1, 2, 3]})
        mock_io.write_table(df)
        out = mock_io.read_arrow_table()
        assert out.num_rows == 3

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
    def test_pandas_dataframe(self, mock_io: _MockMediaIO):
        df = pd.DataFrame({"id": [1, 2, 3]})
        mock_io.write_table(df)
        out = mock_io.read_arrow_table()
        assert out.num_rows == 3

    def test_list_of_dicts(self, mock_io: _MockMediaIO):
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        mock_io.write_table(data)
        out = mock_io.read_arrow_table()
        assert out.num_rows == 3

    def test_column_oriented_dict(self, mock_io: _MockMediaIO):
        mock_io.write_table({"a": [1, 2, 3]})
        out = mock_io.read_arrow_table()
        assert out.num_rows == 3

    def test_list_of_dicts_rejects_mixed_types(self, mock_io: _MockMediaIO):
        with pytest.raises(TypeError, match="list\\[dict\\]"):
            mock_io.write_table([{"a": 1}, "not a dict"])

    def test_iterator_of_arrow_tables(self, mock_io: _MockMediaIO, sample_table):
        def gen():
            yield sample_table
            yield sample_table
        mock_io.write_table(gen())
        out = mock_io.read_arrow_table()
        assert out.num_rows == 2 * sample_table.num_rows

    def test_empty_iterator_is_noop(self, mock_io: _MockMediaIO):
        def gen():
            return
            yield  # unreachable
        mock_io.write_table(gen())
        # Buffer should remain empty (no header written either).
        assert mock_io.holder.size == 0

    def test_unsupported_type_raises(self, mock_io: _MockMediaIO):
        with pytest.raises(TypeError, match="Unsupported"):
            mock_io.write_table(42)  # type: ignore[arg-type]


# =====================================================================
# write_pylist with sparse rows
# =====================================================================

class TestWritePylist:
    def test_homogeneous_rows(self, mock_io: _MockMediaIO):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        mock_io.write_pylist(rows)
        out = mock_io.read_arrow_table()
        assert out.to_pylist() == rows

    def test_sparse_rows_backfilled(self, mock_io: _MockMediaIO):
        """The classic bug: pa.Table.from_pylist drops columns that
        appear only in later rows. _normalize_records must prevent this.
        """
        rows = [{"a": 1}, {"a": 2, "b": "only_in_second"}]
        mock_io.write_pylist(rows)
        out = mock_io.read_arrow_table()
        assert "b" in out.column_names
        result = out.to_pylist()
        assert result[0]["b"] is None
        assert result[1]["b"] == "only_in_second"

    def test_empty_list(self, mock_io: _MockMediaIO):
        mock_io.write_pylist([])
        # Nothing to write — buffer should be untouched.
        assert mock_io.holder.size == 0


# =====================================================================
# read_pylist, read_pydict
# =====================================================================

class TestReadGeneric:
    def test_read_pylist(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.read_pylist()
        assert isinstance(out, list)
        assert len(out) == sample_table.num_rows
        assert out[0] == {"id": 1, "name": "a"}

    def test_read_pylist_with_batch_size_returns_iterator(
        self, mock_io: _MockMediaIO, sample_table
    ):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.read_pylist(batch_size=2)
        assert not isinstance(out, list)
        chunks = list(out)
        assert all(isinstance(c, list) for c in chunks)

    def test_read_pydict(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.read_pydict()
        assert isinstance(out, dict)
        assert out["id"] == [1, 2, 3]


# =====================================================================
# Polars / pandas read roundtrips
# =====================================================================

@pytest.mark.skipif(not HAS_POLARS, reason="polars required")
class TestPolarsRoundtrip:
    def test_read_polars_frame(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.read_polars_frame()
        assert isinstance(out, pl.DataFrame)
        assert out.height == sample_table.num_rows

    def test_read_polars_frame_lazy(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.read_polars_frame(lazy=True)
        assert isinstance(out, pl.LazyFrame)

    def test_read_polars_frames_yields_one_per_batch(
        self, mock_io: _MockMediaIO, sample_table
    ):
        mock_io.write_arrow_table(sample_table)
        frames = list(mock_io.read_polars_frames())
        assert len(frames) >= 1
        assert all(isinstance(f, pl.DataFrame) for f in frames)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
class TestPandasRoundtrip:
    def test_read_pandas_frame(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.read_pandas_frame()
        assert isinstance(out, pd.DataFrame)
        assert len(out) == sample_table.num_rows

    def test_write_pandas_preserves_named_index(self, mock_io: _MockMediaIO):
        df = pd.DataFrame({"v": [1, 2, 3]})
        df.index.name = "my_idx"
        mock_io.write_pandas_frame(df)
        out = mock_io.read_arrow_table()
        assert "my_idx" in out.column_names

    def test_write_pandas_drops_unnamed_rangeindex(self, mock_io: _MockMediaIO):
        df = pd.DataFrame({"v": [1, 2, 3]})
        # default RangeIndex with name=None
        mock_io.write_pandas_frame(df)
        out = mock_io.read_arrow_table()
        assert out.column_names == ["v"]


# =====================================================================
# Schema collection
# =====================================================================

class TestCollectSchema:
    def test_empty_buffer_empty_schema(self, mock_io: _MockMediaIO):
        schema = mock_io._collect_arrow_schema()
        assert schema == pa.schema([])

    def test_populated_buffer_schema(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_arrow_table(sample_table)
        schema = mock_io._collect_arrow_schema()
        assert set(schema.names) == {"id", "name"}

    def test_collect_schema_wraps_in_yggdrasil_schema(
        self, mock_io: _MockMediaIO, sample_table
    ):
        try:
            from yggdrasil.data.schema import Schema
        except ImportError:
            pytest.skip("yggdrasil.data.schema not importable")

        mock_io.write_arrow_table(sample_table)
        schema = mock_io.collect_schema()
        assert isinstance(schema, Schema)


# =====================================================================
# Factory
# =====================================================================

class TestFactory:
    def test_make_routes_parquet(self):
        from yggdrasil.io.buffer.parquet_io import ParquetIO
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.PARQUET)
        assert isinstance(io_, ParquetIO)

    def test_make_routes_csv(self):
        from yggdrasil.io.buffer.csv_io import CsvIO
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.CSV)
        assert isinstance(io_, CsvIO)

    def test_make_routes_json(self):
        from yggdrasil.io.buffer.json_io import JsonIO
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.JSON)
        assert isinstance(io_, JsonIO)

    def test_make_routes_zip(self):
        from yggdrasil.io.buffer.zip_io import ZipIO
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        assert isinstance(io_, ZipIO)

    def test_make_routes_arrow_ipc(self):
        from yggdrasil.io.buffer.arrow_ipc_io import IPCIO
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        assert isinstance(io_, IPCIO)

    def test_make_raises_for_unsupported(self):
        buf = BytesIO()
        buf.set_media_type(MediaType(MimeTypes.OCTET_STREAM), safe=False)
        with pytest.raises(NotImplementedError):
            MediaIO.make(buf, MimeTypes.OCTET_STREAM)

    def test_make_requires_media_when_buffer_has_none(self):
        buf = BytesIO()
        # buf.media_type is None
        with pytest.raises(NotImplementedError, match="Cannot create media IO"):
            MediaIO.make(buf)


# =====================================================================
# mark_dirty flush-back
# =====================================================================

class TestMarkDirtyFlushBack:
    def test_dirty_flag_set_after_write(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_arrow_table(sample_table)
        # After close, buffer is None but holder has data.
        assert mock_io.holder.size > 0

    def test_close_without_dirty_does_not_compress(self, mock_io: _MockMediaIO):
        """Read-only open/close should not trigger compression."""
        mock_io.open()
        assert mock_io._dirty is False
        mock_io.close()
        assert mock_io._dirty is False


# =====================================================================
# SQL execute
# =====================================================================

class TestExecute:
    def test_select_all_returns_statement_result(
        self, mock_io: _MockMediaIO, sample_table
    ):
        from yggdrasil.data.statement import LocalStatementResult

        mock_io.write_arrow_table(sample_table)
        out = mock_io.execute("SELECT * FROM self ORDER BY id")
        assert isinstance(out, LocalStatementResult)
        assert out.done is True
        assert out.failed is False
        assert out.is_polars is True
        assert out.persisted is True

        df = out.to_polars(stream=False)
        assert isinstance(df, pl.DataFrame)
        assert df.height == sample_table.num_rows
        assert df.columns == sample_table.column_names

    def test_projection_and_filter(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.execute(
            "SELECT name FROM self WHERE id > 1 ORDER BY id"
        )
        df = out.to_polars(stream=False)
        assert df.columns == ["name"]
        assert df["name"].to_list() == ["b", "c"]

    def test_to_arrow_table_conversion(
        self, mock_io: _MockMediaIO, sample_table
    ):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.execute("SELECT id FROM self")
        table = out.to_arrow_table()
        assert isinstance(table, pa.Table)
        assert table.column_names == ["id"]

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
    def test_to_pandas_conversion(
        self, mock_io: _MockMediaIO, sample_table
    ):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.execute("SELECT id FROM self")
        frame = out.to_pandas()
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == ["id"]

    def test_custom_table_name(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_arrow_table(sample_table)
        out = mock_io.execute("SELECT COUNT(*) AS n FROM t", name="t")
        df = out.to_polars(stream=False)
        assert df["n"].to_list() == [sample_table.num_rows]

    def test_prepared_statement_input(
        self, mock_io: _MockMediaIO, sample_table
    ):
        from yggdrasil.data.statement import PreparedStatement

        mock_io.write_arrow_table(sample_table)
        prepared = PreparedStatement(text="SELECT COUNT(*) AS n FROM self")
        out = mock_io.execute(prepared)
        assert out.statement is prepared or out.statement == prepared
        assert out.to_polars(stream=False)["n"].to_list() == [
            sample_table.num_rows
        ]

    def test_external_tables_registers_alias(
        self, mock_io: _MockMediaIO, sample_table
    ):
        mock_io.write_arrow_table(sample_table)
        extra = pl.DataFrame({"id": [1], "bonus": [100]})
        out = mock_io.execute(
            "SELECT self.id, extra.bonus FROM self "
            "INNER JOIN extra ON self.id = extra.id",
            external_tables={"extra": extra},
        )
        df = out.to_polars(stream=False)
        assert df.columns == ["id", "bonus"]
        assert df["id"].to_list() == [1]
        assert df["bonus"].to_list() == [100]

    def test_empty_statement_raises(self, mock_io: _MockMediaIO, sample_table):
        mock_io.write_arrow_table(sample_table)
        with pytest.raises(ValueError, match="non-empty SQL statement"):
            mock_io.execute("")
        with pytest.raises(ValueError, match="non-empty SQL statement"):
            mock_io.execute("   ")


# =====================================================================
# Static helpers
# =====================================================================

class TestStaticHelpers:
    def test_is_path_input_accepts_str(self):
        assert MediaIO._is_path_input("/tmp/foo.parquet") is True

    def test_is_path_input_accepts_path(self):
        assert MediaIO._is_path_input(Path("/tmp/foo.parquet")) is True

    def test_is_path_input_rejects_bytesio(self):
        assert MediaIO._is_path_input(BytesIO()) is False

    def test_is_path_input_rejects_table(self, sample_table):
        assert MediaIO._is_path_input(sample_table) is False

    def test_is_path_input_rejects_int(self):
        assert MediaIO._is_path_input(42) is False