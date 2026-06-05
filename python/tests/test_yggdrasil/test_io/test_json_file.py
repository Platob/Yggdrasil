"""Tests for :class:`yggdrasil.io.json_file.JSONFile`."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.path.memory import Memory
from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.json_file import JSONFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        assert Holder.class_for_media_type("application/json") is JSONFile
        assert Holder.class_for_media_type("json") is JSONFile

    def test_path_dispatches(self, tmp_path) -> None:
        from yggdrasil.io.base import IO

        b = IO(path=str(tmp_path / "x.json"))
        assert isinstance(b, JSONFile)


class TestMemoryRoundTrip:

    def test_write_then_read(self) -> None:
        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mem = Memory()
        JSONFile(holder=mem, owns_holder=False).write_arrow_table(table)

        got = JSONFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.column("id").to_pylist() == [1, 2, 3]
        assert got.column("name").to_pylist() == ["a", "b", "c"]


class TestLocalPathRoundTrip:

    def test_write_and_read(self, tmp_path) -> None:
        table = pa.table({"x": [10, 20, 30]})
        path = LocalPath(str(tmp_path / "out.json"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().column("x").to_pylist() == [10, 20, 30]

    def test_on_disk_is_valid_json(self, tmp_path) -> None:
        table = pa.table({"a": [1, 2], "b": ["x", "y"]})
        path = LocalPath(str(tmp_path / "out.json"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        data = json.loads((tmp_path / "out.json").read_text())
        # JSON output is row-oriented.
        assert isinstance(data, list)
        assert len(data) == 2


class TestStraddlingObjectFallback:
    """A single JSON object terminated with ``\\n`` is misclassified as
    NDJSON by the cheap sniff; if the object exceeds pyarrow's default
    1 MiB block size the streaming reader raises ``ArrowInvalid:
    straddling object …``. The reader must fall back to ``json.loads``
    instead of propagating the error.
    """

    def test_large_single_object_with_trailing_newline(self) -> None:
        big = "x" * (2 * 1024 * 1024)
        payload = json.dumps({"id": 1, "blob": big}).encode("utf-8") + b"\n"

        mem = Memory()
        mem.write(payload)

        got = JSONFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.column("id").to_pylist() == [1]
        assert got.column("blob").to_pylist() == [big]

    def test_pretty_printed_object_with_trailing_newline(self) -> None:
        payload = json.dumps({"id": 7, "name": "a"}, indent=2).encode("utf-8") + b"\n"

        mem = Memory()
        mem.write(payload)

        got = JSONFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.column("id").to_pylist() == [7]
        assert got.column("name").to_pylist() == ["a"]


class TestReadViaMemoryview:
    """JSONFile's full-buffer read now feeds orjson a zero-copy
    ``memoryview(read_buffer())`` and infers types from the raw dicts.
    Exercise the shapes that hit that path."""

    def test_top_level_array_of_objects(self) -> None:
        payload = json.dumps(
            [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        ).encode("utf-8")
        mem = Memory()
        mem.write(payload)
        got = JSONFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.column("id").to_pylist() == [1, 2]
        assert got.column("name").to_pylist() == ["a", "b"]

    def test_single_top_level_object(self) -> None:
        payload = json.dumps({"id": 9, "name": "solo"}).encode("utf-8")
        mem = Memory()
        mem.write(payload)
        got = JSONFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.column("id").to_pylist() == [9]
        assert got.column("name").to_pylist() == ["solo"]

    def test_read_with_target_projection(self) -> None:
        table = pa.table({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]})
        mem = Memory()
        JSONFile(holder=mem, owns_holder=False).write_arrow_table(table)
        got = JSONFile(holder=mem, owns_holder=False).read_arrow_table(
            target=pa.schema([("b", pa.string())]),
        )
        assert got.column_names == ["b"]
        assert got.column("b").to_pylist() == ["x", "y"]

    def test_empty_buffer_reads_empty(self) -> None:
        got = JSONFile(holder=Memory(), owns_holder=False).read_arrow_table()
        assert got.num_rows == 0

    def test_datetime_roundtrips_via_orjson(self) -> None:
        import datetime as dt

        table = pa.table({
            "id": pa.array([1], type=pa.int64()),
            "ts": pa.array([dt.datetime(2024, 1, 2, 3, 4, 5)],
                           type=pa.timestamp("us")),
        })
        mem = Memory()
        JSONFile(holder=mem, owns_holder=False).write_arrow_table(table)
        # orjson writes the timestamp as an ISO-8601 string; reading back
        # with the same timestamp target casts it home.
        got = JSONFile(holder=mem, owns_holder=False).read_arrow_table(
            target=table.schema,
        )
        assert got.column("ts").to_pylist() == [dt.datetime(2024, 1, 2, 3, 4, 5)]

    def test_codec_gzip_roundtrip(self, tmp_path) -> None:
        # A ``.json.gz`` holder carries a codec; arrow_input_stream
        # decompresses into a scratch and the memoryview read parses that.
        table = pa.table({"v": [1, 2, 3]})
        path = LocalPath(str(tmp_path / "out.json.gz"))
        with path.open("wb") as cur:
            cur.write_arrow_table(table)
        with path.open("rb") as cur:
            assert cur.read_arrow_table().column("v").to_pylist() == [1, 2, 3]


class TestEmptyOverwriteWritesValidFile:
    """An empty input under OVERWRITE through the generic path route must
    persist a valid empty JSON array (``[]``), not a 0-byte stub."""

    def test_empty_table_writes_empty_array(self, tmp_path) -> None:
        from yggdrasil.enums import Mode

        empty = pa.table(
            {"id": pa.array([], type=pa.int64()),
             "amount": pa.array([], type=pa.float64())}
        )
        path = LocalPath(str(tmp_path / "empty.json"))
        path.write_table(empty, mode=Mode.OVERWRITE)

        assert path.size > 0
        assert json.loads((tmp_path / "empty.json").read_text()) == []

    def test_empty_batches_overwrite_uses_bound_schema(self) -> None:
        from yggdrasil.enums import Mode
        from yggdrasil.data.options import CastOptions

        schema = pa.schema([("id", pa.int64()), ("amount", pa.float64())])
        mem = Memory()
        leaf = JSONFile(holder=mem, owns_holder=False)
        leaf._write_arrow_batches(
            iter([]), leaf.check_options(CastOptions(mode=Mode.OVERWRITE, target=schema)),
        )
        assert json.loads(mem.to_bytes().decode()) == []
