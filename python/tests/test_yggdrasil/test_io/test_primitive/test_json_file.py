"""Tests for :class:`yggdrasil.io.primitive.json_file.JSONFile`."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.primitive.json_file import JSONFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        assert Holder.class_for_media_type("application/json") is JSONFile
        assert Holder.class_for_media_type("json") is JSONFile

    def test_path_dispatches(self, tmp_path) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        b = BytesIO(path=str(tmp_path / "x.json"))
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
