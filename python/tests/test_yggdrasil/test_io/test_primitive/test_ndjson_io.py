"""Behavior tests for :class:`yggdrasil.io.primitive.ndjson_io.NDJsonIO`."""
from __future__ import annotations

import json

import pyarrow as pa
import pytest

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.ndjson_io import NDJsonIO, NDJsonOptions
from yggdrasil.io.tabular import Tabular


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


class TestRegistration:

    def test_mime_type_is_ndjson(self) -> None:
        assert NDJsonIO.mime_type is MimeTypes.NDJSON

    def test_registry(self) -> None:
        assert Tabular.class_for_media_type(MimeTypes.NDJSON) is NDJsonIO


class TestRoundTrip:

    def test_arrow_round_trip(self, table) -> None:
        io = NDJsonIO()
        io.write_arrow_table(table)
        loaded = io.read_arrow_table()
        assert loaded.equals(table)

    def test_one_object_per_line(self, table) -> None:
        io = NDJsonIO()
        io.write_arrow_table(table)
        lines = io.to_bytes().decode("utf-8").splitlines()
        assert len(lines) == table.num_rows
        for line, row in zip(lines, table.to_pylist()):
            assert json.loads(line) == row

    def test_collect_schema(self, table) -> None:
        io = NDJsonIO()
        io.write_arrow_table(table)
        assert set(io.collect_schema().field_names()) == {"id", "name"}


class TestEmpty:

    def test_read_empty(self) -> None:
        assert list(NDJsonIO().read_arrow_batches()) == []


class TestModes:

    def test_overwrite(self, table) -> None:
        io = NDJsonIO()
        io.write_arrow_table(table)
        smaller = pa.table({"id": [9], "name": ["z"]})
        io.write_arrow_table(smaller, options=NDJsonOptions(mode=Mode.OVERWRITE))
        assert io.read_arrow_table().equals(smaller)

    def test_append_concatenates_lines(self, table) -> None:
        io = NDJsonIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [4], "name": ["d"]})
        io.write_arrow_batches(more.to_batches(), options=NDJsonOptions(mode=Mode.APPEND))
        lines = io.to_bytes().decode("utf-8").splitlines()
        assert len(lines) == table.num_rows + more.num_rows
        assert json.loads(lines[-1]) == {"id": 4, "name": "d"}

    def test_append_on_buffer_without_trailing_newline(self) -> None:
        # NDJSON requires every line to end with \n; the leaf must
        # paper over a missing one before appending.
        io = NDJsonIO(b'{"id":1,"name":"a"}')
        io.write_arrow_batches(
            pa.table({"id": [2], "name": ["b"]}).to_batches(),
            options=NDJsonOptions(mode=Mode.APPEND),
        )
        lines = io.to_bytes().decode("utf-8").splitlines()
        assert lines == ['{"id":1,"name":"a"}', '{"id": 2, "name": "b"}']


class TestKeyedMerge:
    """``options.match_by_names`` drives key-aware APPEND / UPSERT."""

    def test_append_with_keys_drops_incoming_duplicates(self, table) -> None:
        io = NDJsonIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [2, 4], "name": ["X", "d"]})
        io.write_arrow_batches(
            more.to_batches(),
            options=NDJsonOptions(mode=Mode.APPEND, match_by_names=["id"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3, 4]
        assert loaded.column("name").to_pylist() == ["a", "b", "c", "d"]

    def test_upsert_with_keys_replaces_existing(self, table) -> None:
        io = NDJsonIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [2, 4], "name": ["X", "d"]})
        io.write_arrow_batches(
            more.to_batches(),
            options=NDJsonOptions(mode=Mode.UPSERT, match_by_names=["id"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 3, 2, 4]
        assert loaded.column("name").to_pylist() == ["a", "c", "X", "d"]


class TestExternalWriterPattern:

    def test_python_json_appender(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "stream.ndjson"))
        rows = [{"id": i, "name": f"row-{i}"} for i in range(5)]
        with target.open("wb") as bio:
            for row in rows:
                bio.write((json.dumps(row) + "\n").encode("utf-8"))

        reader = NDJsonIO(holder=target, owns_holder=False)
        assert reader.read_arrow_table().num_rows == 5

    def test_polars_native_path_round_trip(self, tmp_path, table) -> None:
        pl = pytest.importorskip("polars")
        target = LocalPath(str(tmp_path / "data.ndjson"))
        df = pl.from_arrow(table)
        with target.open("wb") as bio:
            df.write_ndjson(bio)
        reader = NDJsonIO(holder=target, owns_holder=False)
        out = reader.read_polars_frame()
        assert out.equals(df)
