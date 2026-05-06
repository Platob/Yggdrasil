"""Behavior tests for :class:`yggdrasil.io.primitive.xlsx_io.XlsxIO`."""
from __future__ import annotations

import pyarrow as pa
import pytest

openpyxl = pytest.importorskip("openpyxl")  # noqa: E402  (after importorskip)

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.xlsx_io import XlsxIO, XlsxOptions
from yggdrasil.io.tabular import Tabular


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


class TestRegistration:

    def test_mime_type_is_xlsx(self) -> None:
        assert XlsxIO.mime_type is MimeTypes.XLSX

    def test_registry(self) -> None:
        assert Tabular.class_for_media_type(MimeTypes.XLSX) is XlsxIO


class TestRoundTrip:

    def test_arrow_round_trip(self, table) -> None:
        io = XlsxIO()
        io.write_arrow_table(table)
        loaded = io.read_arrow_table()
        # XLSX numbers come back as native ints / strings as native strings.
        assert loaded.column("id").to_pylist() == [1, 2, 3]
        assert loaded.column("name").to_pylist() == ["a", "b", "c"]

    def test_collect_schema(self, table) -> None:
        io = XlsxIO()
        io.write_arrow_table(table)
        assert set(io.collect_schema().field_names()) == {"id", "name"}

    def test_sheet_name_is_honored(self, table) -> None:
        io = XlsxIO()
        io.write_arrow_table(table, options=XlsxOptions(sheet_name="Trades"))
        wb = openpyxl.load_workbook(io)
        assert "Trades" in wb.sheetnames


class TestModes:

    def test_overwrite(self, table) -> None:
        io = XlsxIO()
        io.write_arrow_table(table)
        smaller = pa.table({"id": [9], "name": ["z"]})
        io.write_arrow_table(smaller, options=XlsxOptions(mode=Mode.OVERWRITE))
        assert io.read_arrow_table().column("id").to_pylist() == [9]

    def test_append_rejected(self, table) -> None:
        io = XlsxIO()
        io.write_arrow_table(table)
        with pytest.raises(NotImplementedError, match="OVERWRITE"):
            io.write_arrow_batches(
                table.to_batches(), options=XlsxOptions(mode=Mode.APPEND),
            )

    def test_error_if_exists(self, table) -> None:
        io = XlsxIO()
        io.write_arrow_table(table)
        with pytest.raises(FileExistsError):
            io.write_arrow_batches(
                table.to_batches(), options=XlsxOptions(mode=Mode.ERROR_IF_EXISTS),
            )


class TestHolderBacked:

    def test_local_path_round_trip(self, tmp_path, table) -> None:
        target = LocalPath(str(tmp_path / "data.xlsx"))
        io = XlsxIO(holder=target, owns_holder=False)
        io.write_arrow_table(table)
        # Vanilla openpyxl reads the produced file.
        wb = openpyxl.load_workbook(target.os_path, read_only=True)
        ws = wb[wb.sheetnames[0]]
        rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
        assert rows[0] == ("id", "name")
        assert rows[1:] == [(1, "a"), (2, "b"), (3, "c")]
