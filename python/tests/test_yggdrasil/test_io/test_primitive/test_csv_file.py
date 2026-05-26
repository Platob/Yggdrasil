"""Tests for :class:`yggdrasil.io.primitive.csv_file.CSVFile`."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.primitive.csv_file import CSVFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        assert Holder.class_for_media_type("text/csv") is CSVFile
        assert Holder.class_for_media_type("csv") is CSVFile

    def test_path_dispatches(self, tmp_path) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        b = BytesIO(path=str(tmp_path / "x.csv"))
        assert isinstance(b, CSVFile)


class TestMemoryRoundTrip:

    def test_write_then_read(self) -> None:
        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mem = Memory()
        CSVFile(holder=mem, owns_holder=False).write_arrow_table(table)
        assert mem.size > 0
        got = CSVFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.column("id").to_pylist() == [1, 2, 3]
        assert got.column("name").to_pylist() == ["a", "b", "c"]


class TestLocalPathRoundTrip:

    def test_write_and_read(self, tmp_path) -> None:
        table = pa.table({"x": [10, 20, 30]})
        path = LocalPath(str(tmp_path / "out.csv"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().column("x").to_pylist() == [10, 20, 30]

    def test_text_layout(self, tmp_path) -> None:
        # CSV is text; ensure the on-disk shape parses back correctly.
        path = LocalPath(str(tmp_path / "out.csv"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(pa.table({"a": [1, 2], "b": ["x", "y"]}))
        text = (tmp_path / "out.csv").read_text()
        lines = text.strip().splitlines()
        # Header is the column names, in some quoting style.
        assert {c.strip('"') for c in lines[0].split(",")} == {"a", "b"}
        # 2 data rows.
        assert len(lines) == 3
