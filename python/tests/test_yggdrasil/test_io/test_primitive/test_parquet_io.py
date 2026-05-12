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


class TestTargetSchemaCast:
    """``target_field`` flows through the read/write paths and casts
    every batch on its way through the encoder/decoder. With no target
    bound the path is a passthrough — covered by :class:`TestRoundTrip`."""

    def _target_field(self):
        from yggdrasil.data.data_field import Field
        return Field.from_(pa.schema([
            pa.field("id", pa.int64()),
            pa.field("v", pa.float64()),
        ]))

    def test_read_casts_to_target_schema(self) -> None:
        # Write as strings, read with a numeric target — the reader
        # should cast each batch on the way out.
        io = ParquetIO()
        io.write_arrow_table(pa.table({
            "id": ["1", "2", "3"], "v": ["1.5", "2.5", "3.5"],
        }))
        casted = io.read_arrow_table(target=self._target_field())
        assert casted.schema.field("id").type == pa.int64()
        assert casted.schema.field("v").type == pa.float64()
        assert casted.column("id").to_pylist() == [1, 2, 3]
        assert casted.column("v").to_pylist() == [1.5, 2.5, 3.5]

    def test_write_casts_to_target_schema(self) -> None:
        # Write strings with a numeric target — the file itself
        # should carry the target schema, not the source.
        io = ParquetIO()
        io.write_arrow_table(
            pa.table({"id": ["1", "2"], "v": ["1.5", "2.5"]}),
            target=self._target_field(),
        )
        # Read without a target so we see exactly what was persisted.
        raw = io.read_arrow_table()
        assert raw.schema.field("id").type == pa.int64()
        assert raw.schema.field("v").type == pa.float64()

    def test_no_target_is_passthrough(self) -> None:
        # Empty options round-trip preserves the source schema.
        original = pa.table({"id": [1, 2], "v": [1.5, 2.5]})
        io = ParquetIO()
        io.write_arrow_table(original)
        assert io.read_arrow_table().equals(original)

    def test_arrow_read_projects_to_target_subset(self, table) -> None:
        # ``target`` carrying a strict subset of the file's columns
        # should fan into the parquet reader as a ``columns=`` projection;
        # the returned table carries exactly the target columns in target
        # order.
        io = ParquetIO()
        io.write_arrow_table(table)
        from yggdrasil.data.data_field import Field
        target = Field.from_(pa.schema([
            pa.field("v", pa.float64()),
            pa.field("id", pa.int64()),
        ]))
        casted = io.read_arrow_table(target=target)
        assert casted.column_names == ["v", "id"]
        assert casted.column("id").to_pylist() == [1, 2, 3, 4]

    def test_polars_read_applies_target_cast(self, table) -> None:
        pl = pytest.importorskip("polars")
        io = ParquetIO()
        io.write_arrow_table(pa.table({
            "id": ["1", "2", "3"], "v": ["1.5", "2.5", "3.5"],
        }))
        df = io.read_polars_frame(target=self._target_field())
        assert df.columns == ["id", "v"]
        assert df.schema["id"] == pl.Int64
        assert df.schema["v"] == pl.Float64
        assert df["id"].to_list() == [1, 2, 3]

    def test_polars_scan_applies_target_cast(self, table) -> None:
        pl = pytest.importorskip("polars")
        io = ParquetIO()
        io.write_arrow_table(pa.table({
            "id": ["1", "2", "3"], "v": ["1.5", "2.5", "3.5"],
        }))
        lf = io.scan_polars_frame(target=self._target_field())
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert df.schema["id"] == pl.Int64
        assert df.schema["v"] == pl.Float64

    def test_pandas_read_applies_target_cast(self, table) -> None:
        pytest.importorskip("pandas")
        io = ParquetIO()
        io.write_arrow_table(pa.table({
            "id": ["1", "2", "3"], "v": ["1.5", "2.5", "3.5"],
        }))
        df = io.read_pandas_frame(target=self._target_field())
        assert list(df.columns) == ["id", "v"]
        assert df["id"].tolist() == [1, 2, 3]


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


class TestKeyedMerge:
    """``options.match_by`` drives key-aware APPEND / UPSERT."""

    def test_append_with_keys_drops_incoming_duplicates(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        more = pa.table(
            {"id": [2, 3, 5], "name": ["X", "Y", "e"], "v": [-1.0, -2.0, 4.5]}
        )
        io.write_arrow_batches(
            more.to_batches(),
            options=ParquetOptions(mode=Mode.APPEND, match_by=["id"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3, 4, 5]
        assert loaded.column("name").to_pylist() == ["a", "b", "c", "d", "e"]

    def test_upsert_with_keys_replaces_existing(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        more = pa.table(
            {"id": [2, 3, 5], "name": ["X", "Y", "e"], "v": [-1.0, -2.0, 4.5]}
        )
        io.write_arrow_batches(
            more.to_batches(),
            options=ParquetOptions(mode=Mode.UPSERT, match_by=["id"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 4, 2, 3, 5]
        assert loaded.column("name").to_pylist() == ["a", "d", "X", "Y", "e"]

    def test_auto_with_keys_acts_as_upsert(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [3], "name": ["Z"], "v": [9.0]})
        # Default Mode.AUTO + match_by → UPSERT semantics.
        io.write_arrow_batches(
            more.to_batches(),
            options=ParquetOptions(match_by=["id"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 4, 3]
        assert loaded.column("name").to_pylist() == ["a", "b", "d", "Z"]


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
