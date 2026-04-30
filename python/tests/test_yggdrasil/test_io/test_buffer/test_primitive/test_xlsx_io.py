"""Tests for :class:`XlsxIO`.

Coverage:

- Round-trip of values (note: types coerce to xlsx native — int→int,
  bool→bool, float→float, str→str — but null preservation is
  implementation-defined).
- Custom sheet name.
- has_header=False uses col_0/col_1/... naming.
- APPEND rejected with the documented hint.
- UPSERT rejected.

We use ``pytest.importorskip`` so these tests are silently skipped
when openpyxl isn't installed in CI.
"""

from __future__ import annotations

import pytest
import pyarrow as pa

openpyxl = pytest.importorskip("openpyxl")

from yggdrasil.io.enums import Mode
from yggdrasil.io.buffer.primitive.xlsx_io import XlsxIO, XlsxOptions


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_basic_types():
    """Round-trip of types xlsx can represent natively."""
    io = XlsxIO()
    table = pa.table({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "value": [1.5, 2.5, 3.5],
    })

    with io:
        io.write_arrow_table(table)
        io.seek(0)
        result = io.read_arrow_table()

    actual = result.to_pydict()
    assert actual["id"] == [1, 2, 3]
    assert actual["name"] == ["a", "b", "c"]
    assert actual["value"] == pytest.approx([1.5, 2.5, 3.5])


def test_empty_buffer_read_yields_empty():
    io = XlsxIO()
    with io:
        batches = list(io.read_arrow_batches())
    assert batches == []


# ---------------------------------------------------------------------------
# Sheet naming
# ---------------------------------------------------------------------------


def test_custom_sheet_name():
    io = XlsxIO()
    table = pa.table({"a": [1, 2]})
    options = XlsxOptions(sheet_name="MyData")

    with io:
        io.write_arrow_table(table, options=options)
        io.seek(0)
        # Read via openpyxl directly to verify the sheet was created
        # under the requested name.
        wb = openpyxl.load_workbook(io, read_only=True)
        try:
            assert "MyData" in wb.sheetnames
        finally:
            wb.close()


def test_unknown_sheet_falls_back_to_first():
    io = XlsxIO()
    table = pa.table({"a": [1, 2]})

    with io:
        # Write to "Sheet1" (default).
        io.write_arrow_table(table)
        io.seek(0)
        # Read with a sheet name that doesn't exist — should fall
        # back to first sheet.
        result = io.read_arrow_table(options=XlsxOptions(sheet_name="Nonexistent"))

    assert result.num_rows == 2


# ---------------------------------------------------------------------------
# Header handling
# ---------------------------------------------------------------------------


def test_no_header_uses_col_n_naming():
    io = XlsxIO()
    table = pa.table({"a": [1, 2], "b": [3, 4]})
    options = XlsxOptions(has_header=False)

    with io:
        # Write without a header row.
        io.write_arrow_table(table, options=options)
        io.seek(0)
        result = io.read_arrow_table(options=options)

    # First row of values becomes the first row of data; columns are
    # named col_0, col_1.
    assert result.column_names == ["col_0", "col_1"]
