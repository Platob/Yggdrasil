"""Behavior tests for :class:`yggdrasil.io.primitive.xlsx_io.XlsxIO`."""
from __future__ import annotations

import pyarrow as pa
import pytest

openpyxl = pytest.importorskip("openpyxl")  # noqa: E402  (after importorskip)

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.xlsx_io import XlsxIO, XlsxOptions, XlsxSheetIO
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

    def test_append_replaces_target_sheet_keeps_others(self, table) -> None:
        io = XlsxIO()
        # Two sheets up front via the convenience writer.
        io.write_sheets({
            "S1": table,
            "S2": pa.table({"id": [4, 5], "name": ["d", "e"]}),
        })
        # APPEND a new payload onto the named sheet.
        io.write_arrow_batches(
            pa.table({"id": [9], "name": ["z"]}).to_batches(),
            options=XlsxOptions(sheet_name="S2", mode=Mode.APPEND),
        )
        sheets = set(io.list_sheets())
        assert sheets == {"S1", "S2"}
        # Only S2's payload changed; S1 is untouched.
        assert io.child("S1").read_arrow_table().column("id").to_pylist() == [1, 2, 3]
        assert io.child("S2").read_arrow_table().column("id").to_pylist() == [9]

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


# ---------------------------------------------------------------------------
# Entries API — per-sheet children
# ---------------------------------------------------------------------------


class TestEntries:
    """Multi-sheet workbook: each sheet is a lazy :class:`XlsxSheetIO`
    child, mirroring the :class:`ZipIO` / :class:`ZipEntryIO` pattern."""

    @pytest.fixture
    def workbook(self) -> XlsxIO:
        io = XlsxIO()
        io.write_sheets({
            "Sales": pa.table({"id": [1, 2], "price": [10, 20]}),
            "Inventory": pa.table({"sku": ["A", "B", "C"], "qty": [100, 200, 300]}),
        })
        return io

    def test_list_sheets(self, workbook: XlsxIO) -> None:
        assert workbook.list_sheets() == ["Sales", "Inventory"]

    def test_list_sheets_on_empty_workbook(self) -> None:
        assert XlsxIO().list_sheets() == []

    def test_iter_children_yields_sheet_io(self, workbook: XlsxIO) -> None:
        children = list(workbook.iter_children())
        assert [c.sheet_name for c in children] == ["Sales", "Inventory"]
        assert all(isinstance(c, XlsxSheetIO) for c in children)
        assert all(c.parent is workbook for c in children)

    def test_child_reads_only_its_sheet(self, workbook: XlsxIO) -> None:
        sales = workbook.child("Sales")
        rows = sales.read_arrow_table().to_pylist()
        assert rows == [{"id": 1, "price": 10}, {"id": 2, "price": 20}]

    def test_child_unknown_name_raises(self, workbook: XlsxIO) -> None:
        with pytest.raises(KeyError, match="Inventory"):
            workbook.child("Missing")

    def test_child_collect_schema(self, workbook: XlsxIO) -> None:
        inv = workbook.child("Inventory")
        assert set(inv.collect_schema().field_names()) == {"sku", "qty"}

    def test_child_byte_surface_materializes_csv(self, workbook: XlsxIO) -> None:
        sales = workbook.child("Sales")
        assert not sales._materialized
        # First byte-level access drives materialization.
        payload = sales.to_bytes()
        assert sales._materialized
        # CSV view of the sheet — header row plus two data rows.
        text = payload.decode("utf-8")
        assert text.splitlines()[0] == "id,price"
        assert "10" in text and "20" in text

    def test_iter_children_is_lazy(self, workbook: XlsxIO) -> None:
        # Walking the directory does NOT materialize per-sheet bytes.
        children = list(workbook.iter_children())
        assert all(not c._materialized for c in children)

    def test_child_write_replaces_only_target_sheet(
        self, workbook: XlsxIO,
    ) -> None:
        # Update Sales through its child handle; Inventory stays put.
        sales = workbook.child("Sales")
        sales.write_arrow_table(pa.table({"id": [9], "price": [99]}))
        assert set(workbook.list_sheets()) == {"Sales", "Inventory"}
        assert workbook.child("Sales").read_arrow_table().column("id").to_pylist() == [9]
        assert workbook.child("Inventory").read_arrow_table().column("qty").to_pylist() == [100, 200, 300]

    def test_write_sheets_replaces_workbook(self, workbook: XlsxIO) -> None:
        workbook.write_sheets({
            "Only": pa.table({"x": [1]}),
        })
        assert workbook.list_sheets() == ["Only"]
        assert workbook.child("Only").read_arrow_table().column("x").to_pylist() == [1]

    def test_write_sheets_accepts_record_batches(self) -> None:
        io = XlsxIO()
        batch = pa.record_batch({"id": pa.array([1, 2])})
        io.write_sheets({"S": batch})
        assert io.child("S").read_arrow_table().column("id").to_pylist() == [1, 2]
