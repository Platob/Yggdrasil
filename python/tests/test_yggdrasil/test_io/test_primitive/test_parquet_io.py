"""Behavior tests for :class:`yggdrasil.io.primitive.parquet_io.ParquetIO`."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.parquet_io import ParquetIO, ParquetOptions
from yggdrasil.io.tabular import Tabular


@pytest.fixture
def table() -> pa.Table:
    return pa.table(
        {"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "v": [0.5, 1.5, 2.5, 3.5]}
    )


class TestRegistration:

    def test_mime_type_is_parquet(self) -> None:
        assert ParquetIO.mime_type is MimeTypes.PARQUET

    def test_registry_resolves(self) -> None:
        assert Tabular.class_for_media_type(MimeTypes.PARQUET) is ParquetIO

    def test_options_class(self) -> None:
        assert ParquetIO.options_class() is ParquetOptions


class TestRoundTrip:

    def test_arrow_table_round_trip(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        assert io.read_arrow_table().equals(table)

    def test_collect_schema(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        schema = io.collect_schema()
        assert set(schema.field_names()) == {"id", "name", "v"}

    def test_pandas_round_trip(self, table) -> None:
        pd = pytest.importorskip("pandas")
        io = ParquetIO()
        io.write_arrow_table(table)
        pd.testing.assert_frame_equal(io.read_pandas_frame(), table.to_pandas())

    def test_polars_round_trip(self, table) -> None:
        pl = pytest.importorskip("polars")
        io = ParquetIO()
        io.write_arrow_table(table)
        assert io.read_polars_frame().equals(pl.from_arrow(table))

    def test_pylist_round_trip(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        assert io.read_pylist() == table.to_pylist()


class TestEmpty:

    def test_read_empty_yields_no_batches(self) -> None:
        assert list(ParquetIO().read_arrow_batches()) == []

    def test_collect_schema_empty(self) -> None:
        from yggdrasil.data.schema import Schema
        assert ParquetIO().collect_schema() == Schema.empty()


class TestHolderBacked:

    def test_local_path_round_trip(self, tmp_path, table) -> None:
        target = LocalPath(str(tmp_path / "data.parquet"))
        io = ParquetIO(holder=target, owns_holder=False)
        io.write_arrow_table(table)
        assert target.size > 0

        # Verify that vanilla pyarrow can read the file we wrote.
        assert pq.read_table(target.os_path).equals(table)

    def test_memory_holder_round_trip(self, table) -> None:
        mem = Memory()
        io = ParquetIO(holder=mem, owns_holder=False)
        io.write_arrow_table(table)
        reader = ParquetIO(holder=mem, owns_holder=False)
        assert reader.read_arrow_table().equals(table)


class TestModes:

    def test_overwrite_replaces(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        smaller = pa.table({"id": [1], "name": ["x"], "v": [0.5]})
        io.write_arrow_table(smaller, options=ParquetOptions(mode=Mode.OVERWRITE))
        assert io.read_arrow_table().equals(smaller)

    def test_append_concatenates(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [5, 6], "name": ["e", "f"], "v": [4.5, 5.5]})
        io.write_arrow_batches(
            more.to_batches(), options=ParquetOptions(mode=Mode.APPEND),
        )
        assert io.read_arrow_table().num_rows == table.num_rows + more.num_rows

    def test_ignore_skips_when_non_empty(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        before = io.size
        io.write_arrow_batches(
            pa.table({"id": [99], "name": ["z"], "v": [9.5]}).to_batches(),
            options=ParquetOptions(mode=Mode.IGNORE),
        )
        assert io.size == before

    def test_error_if_exists_raises(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        with pytest.raises(FileExistsError):
            io.write_arrow_batches(
                table.to_batches(),
                options=ParquetOptions(mode=Mode.ERROR_IF_EXISTS),
            )


class TestCompression:

    @pytest.mark.parametrize("codec", ["snappy", "gzip", "zstd", "lz4"])
    def test_round_trip_under_codec(self, table, codec) -> None:
        io = ParquetIO()
        io.write_arrow_table(table, options=ParquetOptions(compression=codec))
        assert io.read_arrow_table().equals(table)

    def test_uncompressed_round_trip(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table, options=ParquetOptions(compression=None))
        assert io.read_arrow_table().equals(table)


class TestRowSize:

    def test_row_size_caps_batch_size(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        batches = list(io.read_arrow_batches(options=ParquetOptions(row_size=2)))
        # The reader respects the requested batch_size up to the row
        # group boundary; total rows preserved.
        assert sum(b.num_rows for b in batches) == table.num_rows


class TestExternalWriterPattern:
    """Pyarrow / polars / pandas writers via ``with path.open() as b: ...``."""

    def test_pyarrow_writer_into_path_open(self, tmp_path, table) -> None:
        target = LocalPath(str(tmp_path / "data.parquet"))
        with target.open("wb") as bio:
            with pq.ParquetWriter(bio, table.schema) as writer:
                writer.write_table(table)
        assert pq.read_table(target.os_path).equals(table)

    def test_polars_native_path_round_trip(self, tmp_path, table) -> None:
        pl = pytest.importorskip("polars")
        target = LocalPath(str(tmp_path / "data.parquet"))
        df = pl.from_arrow(table)
        with target.open("wb") as bio:
            df.write_parquet(bio)

        reader = ParquetIO(holder=target, owns_holder=False)
        assert reader.read_polars_frame().equals(df)

    def test_polars_lazy_scan(self, tmp_path, table) -> None:
        pl = pytest.importorskip("polars")
        target = LocalPath(str(tmp_path / "data.parquet"))
        ParquetIO(holder=target, owns_holder=False).write_arrow_table(table)

        reader = ParquetIO(holder=target, owns_holder=False)
        lf = reader.scan_polars_frame()
        assert isinstance(lf, pl.LazyFrame)
        assert lf.collect().equals(pl.from_arrow(table))
