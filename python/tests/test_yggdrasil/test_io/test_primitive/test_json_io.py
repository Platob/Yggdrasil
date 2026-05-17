"""Behavior tests for :class:`yggdrasil.io.primitive.json_io.JsonIO`."""
from __future__ import annotations

import json

import pyarrow as pa
import pytest

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.json_io import JsonIO, JsonOptions
from yggdrasil.io.holder import Holder
from yggdrasil.io.tabular import Tabular


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


class TestRegistration:

    def test_mime_type_is_json(self) -> None:
        assert JsonIO.mime_type is MimeTypes.JSON

    def test_registry(self) -> None:
        assert Holder.class_for_media_type(MimeTypes.JSON) is JsonIO


class TestRoundTrip:

    def test_arrow_round_trip(self, table) -> None:
        io = JsonIO()
        io.write_arrow_table(table)
        loaded = io.read_arrow_table()
        assert loaded.equals(table)

    def test_writes_a_json_array(self, table) -> None:
        io = JsonIO()
        io.write_arrow_table(table)
        text = io.to_bytes().decode("utf-8")
        assert text.startswith("[")
        assert text.rstrip().endswith("]")
        # Parse with stdlib to confirm shape.
        rows = json.loads(text)
        assert rows == table.to_pylist()

    def test_pretty_indent(self, table) -> None:
        io = JsonIO()
        io.write_arrow_table(table, options=JsonOptions(indent=2))
        text = io.to_bytes().decode("utf-8")
        assert "\n  " in text  # indented

    def test_collect_schema(self, table) -> None:
        io = JsonIO()
        io.write_arrow_table(table)
        assert set(io.collect_schema().field_names()) == {"id", "name"}


class TestTargetSchemaCast:
    """``target_field`` reshapes batches on read and write. With no
    target bound the path is a passthrough — covered by
    :class:`TestRoundTrip`."""

    def _target_field(self):
        from yggdrasil.data.data_field import Field
        return Field.from_(pa.schema([
            pa.field("id", pa.int64()),
            pa.field("v", pa.float64()),
        ]))

    def test_read_casts_array_to_target_schema(self) -> None:
        payload = json.dumps([{"id": "1", "v": "1.5"}, {"id": "2", "v": "2.5"}])
        io = JsonIO(payload.encode("utf-8"))
        casted = io.read_arrow_table(target=self._target_field())
        assert casted.schema.field("id").type == pa.int64()
        assert casted.schema.field("v").type == pa.float64()

    def test_read_casts_single_object_to_target_schema(self) -> None:
        # The single-object branch wraps into a one-row batch — verify
        # the cast still fires on that shape.
        payload = json.dumps({"id": "1", "v": "1.5"})
        io = JsonIO(payload.encode("utf-8"))
        casted = io.read_arrow_table(target=self._target_field())
        assert casted.schema.field("id").type == pa.int64()
        assert casted.column("v").to_pylist() == [1.5]

    def test_write_casts_to_target_schema(self) -> None:
        io = JsonIO()
        io.write_arrow_table(
            pa.table({"id": ["1", "2"], "v": ["1.5", "2.5"]}),
            target=self._target_field(),
        )
        raw = io.read_arrow_table()
        assert raw.schema.field("id").type == pa.int64()
        assert raw.schema.field("v").type == pa.float64()


class TestInputShapes:

    def test_reads_array_of_objects(self) -> None:
        payload = json.dumps([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])
        io = JsonIO(payload.encode("utf-8"))
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2]

    def test_reads_single_object(self) -> None:
        payload = json.dumps({"id": 1, "name": "a"})
        io = JsonIO(payload.encode("utf-8"))
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1]

    def test_reads_ndjson_input(self) -> None:
        # Newline-terminated NDJSON also works — pyarrow's reader
        # handles line-delimited objects directly.
        payload = b'{"id":1,"name":"a"}\n{"id":2,"name":"b"}\n'
        io = JsonIO(payload)
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2]

    def test_rejects_scalar_top_level(self) -> None:
        io = JsonIO(b"42")
        with pytest.raises(ValueError, match="expected a JSON"):
            io.read_arrow_table()


class TestModes:

    def test_overwrite(self, table) -> None:
        io = JsonIO()
        io.write_arrow_table(table)
        smaller = pa.table({"id": [9], "name": ["z"]})
        io.write_arrow_table(smaller, options=JsonOptions(mode=Mode.OVERWRITE))
        assert io.read_arrow_table().equals(smaller)

    def test_append_concatenates_rows(self, table) -> None:
        io = JsonIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [4], "name": ["d"]})
        io.write_arrow_batches(more.to_batches(), options=JsonOptions(mode=Mode.APPEND))
        loaded = io.read_arrow_table()
        assert loaded.num_rows == table.num_rows + more.num_rows

    def test_error_if_exists(self, table) -> None:
        io = JsonIO()
        io.write_arrow_table(table)
        with pytest.raises(FileExistsError):
            io.write_arrow_batches(
                table.to_batches(), options=JsonOptions(mode=Mode.ERROR_IF_EXISTS),
            )


class TestHolderBacked:

    def test_local_path_round_trip(self, tmp_path, table) -> None:
        target = LocalPath(str(tmp_path / "data.json"))
        io = JsonIO(holder=target, owns_holder=False)
        io.write_arrow_table(table)
        assert json.loads(target.read_text()) == table.to_pylist()
