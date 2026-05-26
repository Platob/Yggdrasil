"""Tests for :class:`yggdrasil.io.primitive.ndjson_file.NDJSONFile`."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.ndjson_file import NDJSONFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        from yggdrasil.enums import MimeTypes

        assert Holder.class_for_media_type(MimeTypes.NDJSON) is NDJSONFile

    def test_path_dispatches(self, tmp_path) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        b = BytesIO(path=str(tmp_path / "x.ndjson"))
        assert isinstance(b, NDJSONFile)


class TestMemoryRoundTrip:

    def test_write_then_read(self) -> None:
        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mem = Memory()
        NDJSONFile(holder=mem, owns_holder=False).write_arrow_table(table)
        got = NDJSONFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.column("id").to_pylist() == [1, 2, 3]
        assert got.column("name").to_pylist() == ["a", "b", "c"]


class TestLocalPathRoundTrip:

    def test_write_and_read(self, tmp_path) -> None:
        table = pa.table({"x": [10, 20, 30]})
        path = LocalPath(str(tmp_path / "out.ndjson"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().column("x").to_pylist() == [10, 20, 30]

    def test_one_object_per_line(self, tmp_path) -> None:
        table = pa.table({"a": [1, 2], "b": ["x", "y"]})
        path = LocalPath(str(tmp_path / "out.ndjson"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        text = (tmp_path / "out.ndjson").read_text().strip()
        lines = text.splitlines()
        assert len(lines) == 2
        for line in lines:
            assert isinstance(json.loads(line), dict)
