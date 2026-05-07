"""Behavior tests for :class:`yggdrasil.io.primitive.xlsx_io.XlsxIO`."""
from __future__ import annotations

from datetime import date, datetime

import pyarrow as pa
import pytest

openpyxl = pytest.importorskip("openpyxl")  # noqa: E402  (after importorskip)

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.xlsx_io import XlsxIO, XlsxOptions, XlsxSheetIO
from yggdrasil.io.tabular import Tabular


def _xlsx_with_string_cells(
    rows: "list[list[object]]",
    *,
    sheet: str = "Sheet1",
) -> XlsxIO:
    """Build an XlsxIO whose cells carry the literal values in *rows*.

    openpyxl normally types cells based on Python value at write
    time, so writing strings keeps them as strings on read-back —
    which is exactly what we want for inference tests.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    for row in rows:
        ws.append(row)
    import io as _io
    buf = _io.BytesIO()
    wb.save(buf)
    return XlsxIO(buf.getvalue())


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


# ---------------------------------------------------------------------------
# Read-side type inference
# ---------------------------------------------------------------------------


class TestNullMarkers:
    """Configurable null tokens replace string-typed cells with ``None``."""

    def test_default_markers(self) -> None:
        io = _xlsx_with_string_cells([
            ["id", "name"],
            ["1", "Alice"],
            ["2", "NULL"],
            ["3", "N/A"],
            ["4", ""],
            ["5", "-"],
        ])
        loaded = io.read_arrow_table()
        assert loaded.column("name").to_pylist() == [
            "Alice", None, None, None, None,
        ]

    def test_custom_marker(self) -> None:
        io = _xlsx_with_string_cells([
            ["id", "name"],
            ["1", "MISSING"],
            ["2", "Bob"],
        ])
        loaded = io.read_arrow_table(
            options=XlsxOptions(null_values=("MISSING",)),
        )
        assert loaded.column("name").to_pylist() == [None, "Bob"]

    def test_disable_inference_keeps_literal(self) -> None:
        io = _xlsx_with_string_cells([
            ["id", "name"],
            ["1", "NULL"],
        ])
        loaded = io.read_arrow_table(options=XlsxOptions(infer_types=False))
        assert loaded.column("name").to_pylist() == ["NULL"]


class TestNumericInference:
    """Numeric strings — including thousands separators and signs."""

    def test_thousands_separated_floats(self) -> None:
        io = _xlsx_with_string_cells([
            ["price"],
            ["1,234.56"],
            ["123,892.0"],
            ["999.99"],
        ])
        loaded = io.read_arrow_table()
        assert loaded.column("price").to_pylist() == [1234.56, 123892.0, 999.99]
        # All-numeric → arrow promotes to float64.
        assert pa.types.is_floating(loaded.schema.field("price").type)

    def test_thousands_separated_ints(self) -> None:
        io = _xlsx_with_string_cells([
            ["qty"],
            ["1,000"],
            ["12,345"],
        ])
        loaded = io.read_arrow_table()
        assert loaded.column("qty").to_pylist() == [1000, 12345]
        assert pa.types.is_integer(loaded.schema.field("qty").type)

    def test_negative_numbers(self) -> None:
        io = _xlsx_with_string_cells([
            ["pnl"],
            ["-1,234.50"],
            ["+50.0"],
        ])
        loaded = io.read_arrow_table()
        assert loaded.column("pnl").to_pylist() == [-1234.50, 50.0]

    def test_european_decimal(self) -> None:
        # ``"1.234,56"`` → ``1234.56`` when caller flips the
        # separators around to en-EU style.
        io = _xlsx_with_string_cells([
            ["price"],
            ["1.234,56"],
            ["7,5"],
        ])
        loaded = io.read_arrow_table(
            options=XlsxOptions(
                thousands_separator=".",
                decimal_separator=",",
            ),
        )
        assert loaded.column("price").to_pylist() == [1234.56, 7.5]

    def test_scientific_notation(self) -> None:
        io = _xlsx_with_string_cells([
            ["x"],
            ["1e5"],
            ["2.5E-3"],
        ])
        loaded = io.read_arrow_table()
        assert loaded.column("x").to_pylist() == [1e5, 2.5e-3]

    def test_string_columns_bypass_numeric(self) -> None:
        # ID column has leading-zero strings; we don't want them
        # turned into ``"0042" → 42``.
        io = _xlsx_with_string_cells([
            ["id", "qty"],
            ["0042", "100"],
            ["0007", "200"],
        ])
        loaded = io.read_arrow_table(
            options=XlsxOptions(string_columns=("id",)),
        )
        assert loaded.column("id").to_pylist() == ["0042", "0007"]
        assert loaded.column("qty").to_pylist() == [100, 200]

    def test_mixed_numeric_and_null(self) -> None:
        io = _xlsx_with_string_cells([
            ["x"],
            ["1,000"],
            ["N/A"],
            ["2,000"],
        ])
        loaded = io.read_arrow_table()
        assert loaded.column("x").to_pylist() == [1000, None, 2000]


class TestDateInference:
    """ISO and a handful of common locale-flavored date / datetime strings."""

    def test_iso_date(self) -> None:
        io = _xlsx_with_string_cells([
            ["d"],
            ["2026-01-02"],
            ["2026-12-31"],
        ])
        loaded = io.read_arrow_table()
        assert loaded.column("d").to_pylist() == [
            date(2026, 1, 2), date(2026, 12, 31),
        ]

    def test_iso_datetime(self) -> None:
        io = _xlsx_with_string_cells([
            ["t"],
            ["2026-01-02T03:04:05"],
            ["2026-01-02 03:04:05"],
        ])
        loaded = io.read_arrow_table()
        assert loaded.column("t").to_pylist() == [
            datetime(2026, 1, 2, 3, 4, 5),
            datetime(2026, 1, 2, 3, 4, 5),
        ]

    def test_eu_format(self) -> None:
        io = _xlsx_with_string_cells([
            ["d"],
            ["02/01/2026"],
            ["31/12/2026"],
        ])
        loaded = io.read_arrow_table()
        assert loaded.column("d").to_pylist() == [
            date(2026, 1, 2), date(2026, 12, 31),
        ]

    def test_custom_format(self) -> None:
        io = _xlsx_with_string_cells([
            ["d"],
            ["Jan 02, 2026"],
        ])
        loaded = io.read_arrow_table(
            options=XlsxOptions(date_formats=("%b %d, %Y",)),
        )
        assert loaded.column("d").to_pylist() == [date(2026, 1, 2)]

    def test_unparseable_string_stays_string(self) -> None:
        io = _xlsx_with_string_cells([
            ["note"],
            ["just a label"],
            ["another"],
        ])
        loaded = io.read_arrow_table()
        assert loaded.column("note").to_pylist() == ["just a label", "another"]


class TestNativeTypesPassThrough:
    """openpyxl-native cell types (real numbers, real dates) flow
    through inference unchanged."""

    def test_native_int_and_float(self) -> None:
        io = XlsxIO()
        io.write_arrow_table(pa.table({
            "i": pa.array([1, 2, 3], type=pa.int64()),
            "f": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
        }))
        loaded = io.read_arrow_table()
        assert loaded.column("i").to_pylist() == [1, 2, 3]
        assert loaded.column("f").to_pylist() == [1.5, 2.5, 3.5]

    def test_native_datetime(self) -> None:
        io = XlsxIO()
        io.write_arrow_table(pa.table({
            "t": pa.array(
                [datetime(2026, 1, 2, 3, 4, 5), datetime(2026, 6, 7, 8, 9, 10)],
                type=pa.timestamp("us"),
            ),
        }))
        loaded = io.read_arrow_table()
        assert loaded.column("t").to_pylist() == [
            datetime(2026, 1, 2, 3, 4, 5),
            datetime(2026, 6, 7, 8, 9, 10),
        ]
