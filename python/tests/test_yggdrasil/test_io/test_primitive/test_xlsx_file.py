"""Tests for :class:`yggdrasil.io.primitive.xlsx_file.XLSXFile`."""

from __future__ import annotations

import pyarrow as pa
import pytest

openpyxl = pytest.importorskip("openpyxl")
pytest.importorskip("fastexcel")

from yggdrasil.io.holder import Holder
from yggdrasil.path.memory import Memory
from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.primitive.xlsx_file import XLSXFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        from yggdrasil.enums import MimeTypes

        assert Holder.class_for_media_type(MimeTypes.XLSX) is XLSXFile

    def test_path_dispatches_xlsx_ext(self, tmp_path) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        b = BytesIO(path=str(tmp_path / "x.xlsx"))
        assert isinstance(b, XLSXFile)


class TestMemoryRoundTrip:

    def test_write_then_read(self) -> None:
        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mem = Memory()
        XLSXFile(holder=mem, owns_holder=False).write_arrow_table(table)
        assert mem.size > 0
        got = XLSXFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.column("id").to_pylist() == [1, 2, 3]
        assert got.column("name").to_pylist() == ["a", "b", "c"]


class TestLocalPathRoundTrip:

    def test_write_and_read(self, tmp_path) -> None:
        table = pa.table({"x": [10, 20, 30]})
        path = LocalPath(str(tmp_path / "out.xlsx"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().column("x").to_pylist() == [10, 20, 30]
