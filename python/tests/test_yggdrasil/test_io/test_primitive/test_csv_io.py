"""Behavior tests for :class:`yggdrasil.io.primitive.csv_io.CsvIO`."""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.csv_io import CsvIO, CsvOptions
from yggdrasil.io.tabular import Tabular


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


class TestRegistration:

    def test_mime_type_is_csv(self) -> None:
        assert CsvIO.mime_type is MimeTypes.CSV

    def test_registry(self) -> None:
        assert Tabular.class_for_media_type(MimeTypes.CSV) is CsvIO


class TestRoundTrip:

    def test_round_trip_arrow(self, table) -> None:
        io = CsvIO()
        io.write_arrow_table(table)
        loaded = io.read_arrow_table()
        assert loaded.equals(table)

    def test_csv_text_shape(self, table) -> None:
        io = CsvIO()
        io.write_arrow_table(table)
        # The pyarrow CSV writer quotes string columns by default.
        text = io.to_bytes().decode("utf-8")
        lines = text.strip().splitlines()
        assert lines[0] == '"id","name"'
        assert lines[1] == '1,"a"'

    def test_collect_schema(self, table) -> None:
        io = CsvIO()
        io.write_arrow_table(table)
        assert set(io.collect_schema().field_names()) == {"id", "name"}

    def test_pandas_round_trip(self, table) -> None:
        pd = pytest.importorskip("pandas")
        io = CsvIO()
        io.write_arrow_table(table)
        pd.testing.assert_frame_equal(io.read_pandas_frame(), table.to_pandas())

    def test_polars_round_trip(self, table) -> None:
        pl = pytest.importorskip("polars")
        io = CsvIO()
        io.write_arrow_table(table)
        assert io.read_polars_frame().equals(pl.from_arrow(table))


class TestEmpty:

    def test_read_empty(self) -> None:
        assert list(CsvIO().read_arrow_batches()) == []

    def test_collect_schema_empty(self) -> None:
        from yggdrasil.data.schema import Schema
        assert CsvIO().collect_schema() == Schema.empty()


class TestModes:

    def test_overwrite_replaces(self, table) -> None:
        io = CsvIO()
        io.write_arrow_table(table)
        smaller = pa.table({"id": [9], "name": ["z"]})
        io.write_arrow_table(smaller, options=CsvOptions(mode=Mode.OVERWRITE))
        assert io.read_arrow_table().equals(smaller)

    def test_append_concatenates_without_extra_header(self, table) -> None:
        io = CsvIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [4], "name": ["d"]})
        io.write_arrow_batches(more.to_batches(), options=CsvOptions(mode=Mode.APPEND))

        text = io.to_bytes().decode("utf-8")
        # Header should appear exactly once.
        assert text.count('"id","name"') == 1
        loaded = io.read_arrow_table()
        assert loaded.num_rows == table.num_rows + more.num_rows

    def test_append_on_empty_writes_header(self, table) -> None:
        io = CsvIO()
        io.write_arrow_batches(table.to_batches(), options=CsvOptions(mode=Mode.APPEND))
        text = io.to_bytes().decode("utf-8")
        assert text.startswith('"id","name"')

    def test_ignore_skips_when_non_empty(self, table) -> None:
        io = CsvIO()
        io.write_arrow_table(table)
        before = io.size
        io.write_arrow_batches(
            pa.table({"id": [9], "name": ["z"]}).to_batches(),
            options=CsvOptions(mode=Mode.IGNORE),
        )
        assert io.size == before

    def test_error_if_exists_raises(self, table) -> None:
        io = CsvIO()
        io.write_arrow_table(table)
        with pytest.raises(FileExistsError):
            io.write_arrow_batches(
                table.to_batches(), options=CsvOptions(mode=Mode.ERROR_IF_EXISTS),
            )


class TestKeyedMerge:
    """``options.match_by`` drives key-aware APPEND / UPSERT."""

    def test_append_with_keys_drops_incoming_duplicates(self, table) -> None:
        io = CsvIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [2, 4], "name": ["X", "d"]})
        io.write_arrow_batches(
            more.to_batches(),
            options=CsvOptions(mode=Mode.APPEND, match_by=["id"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3, 4]
        assert loaded.column("name").to_pylist() == ["a", "b", "c", "d"]

    def test_upsert_with_keys_replaces_existing(self, table) -> None:
        io = CsvIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [2, 4], "name": ["X", "d"]})
        io.write_arrow_batches(
            more.to_batches(),
            options=CsvOptions(mode=Mode.UPSERT, match_by=["id"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 3, 2, 4]
        assert loaded.column("name").to_pylist() == ["a", "c", "X", "d"]


class TestDelimiter:

    def test_tsv_round_trip(self, table) -> None:
        io = CsvIO()
        io.write_arrow_table(table, options=CsvOptions(delimiter="\t"))
        assert "\t" in io.to_bytes().decode("utf-8")
        loaded = io.read_arrow_table(options=CsvOptions(delimiter="\t"))
        assert loaded.equals(table)


class TestHolderBacked:

    def test_local_path_round_trip(self, tmp_path, table) -> None:
        target = LocalPath(str(tmp_path / "data.csv"))
        io = CsvIO(holder=target, owns_holder=False)
        io.write_arrow_table(table)
        # Vanilla read.
        text = target.read_text()
        assert "id" in text and "name" in text

        reader = CsvIO(holder=target, owns_holder=False)
        assert reader.read_arrow_table().equals(table)


class TestExternalWriterPattern:

    def test_pandas_to_csv_then_read_arrow(self, tmp_path, table) -> None:
        pd = pytest.importorskip("pandas")
        target = LocalPath(str(tmp_path / "data.csv"))
        with target.open("wb") as bio:
            table.to_pandas().to_csv(bio, index=False)

        # Read back through CsvIO. pandas writes unquoted strings by
        # default, so the reader does the inference.
        reader = CsvIO(holder=target, owns_holder=False)
        loaded = reader.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]
        assert loaded.column("name").to_pylist() == ["a", "b", "c"]

    def test_polars_native_path_round_trip(self, tmp_path, table) -> None:
        pl = pytest.importorskip("polars")
        target = LocalPath(str(tmp_path / "data.csv"))
        df = pl.from_arrow(table)
        with target.open("wb") as bio:
            df.write_csv(bio)

        reader = CsvIO(holder=target, owns_holder=False)
        out = reader.read_polars_frame()
        assert out.equals(df)
