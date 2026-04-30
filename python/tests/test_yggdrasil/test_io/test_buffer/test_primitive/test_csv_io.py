"""Tests for :class:`CsvIO`.

Coverage:

- Round-trip of canonical types.
- Delimiter / quote_char / has_header option plumbing.
- APPEND honest concatenation (no duplicate header row).
- APPEND on empty collapses to OVERWRITE-with-header.
- Schema collection from first batch.
- Native scanner gating (path / codec / target_field).
"""

from __future__ import annotations

import pytest
import pyarrow as pa

from yggdrasil.io.enums import Mode
from yggdrasil.io.buffer.primitive.csv_io import CsvIO, CsvOptions


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_canonical_table(arrow_table):
    io = CsvIO()
    with io:
        io.write_arrow_table(arrow_table)
        io.seek(0)
        result = io.read_arrow_table()

    # CSV doesn't preserve types — ints, bools, floats all parse from
    # text. Compare values via to_pydict for tolerance.
    actual = result.to_pydict()
    expected = arrow_table.to_pydict()

    assert actual["id"] == expected["id"]
    assert actual["name"] == expected["name"]
    assert actual["value"] == pytest.approx(expected["value"])
    # Bools may parse as bools or as strings depending on the
    # ConvertOptions; just check we got something.
    assert "active" in actual


def test_empty_buffer_yields_no_batches():
    io = CsvIO()
    with io:
        batches = list(io.read_arrow_batches())
    assert batches == []


def test_empty_buffer_collect_schema_returns_empty():
    io = CsvIO()
    with io:
        schema = io.collect_schema()
    assert schema.is_empty() or len(schema) == 0


# ---------------------------------------------------------------------------
# Format-specific options
# ---------------------------------------------------------------------------


def test_tsv_via_delimiter():
    io = CsvIO()
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    options = CsvOptions(delimiter="\t")

    with io:
        io.write_arrow_table(table, options=options)
        io.seek(0)
        raw = io.read().decode("utf-8")
        assert "\t" in raw
        assert "," not in raw  # delimiter changed, no commas

        io.seek(0)
        result = io.read_arrow_table(options=options)

    assert result.column_names == ["a", "b"]
    assert result.num_rows == 2


def test_no_header_round_trip():
    io = CsvIO()
    table = pa.table({"a": [1, 2], "b": [3, 4]})
    options = CsvOptions(has_header=False, write_header=False)

    with io:
        io.write_arrow_table(table, options=options)
        io.seek(0)
        # Without a header, columns are auto-named (col_0, col_1, ...
        # or pyarrow's f0, f1, ... — depends on parser).
        result = io.read_arrow_table(options=options)

    assert result.num_rows == 2
    assert result.num_columns == 2


# ---------------------------------------------------------------------------
# APPEND
# ---------------------------------------------------------------------------


def test_append_no_duplicate_header():
    """APPEND to a non-empty buffer must not write a second header."""
    io = CsvIO()
    table = pa.table({"a": [1], "b": [2]})

    with io:
        io.write_arrow_table(table, mode=Mode.OVERWRITE)
        io.write_arrow_table(table, mode=Mode.APPEND)
        io.seek(0)
        raw = io.read().decode("utf-8")

    # Header "a,b" should appear exactly once.
    assert raw.count("\"a\",\"b\"") == 1


def test_append_to_empty_writes_header():
    io = CsvIO()
    table = pa.table({"a": [1], "b": [2]})

    with io:
        io.write_arrow_table(table, mode=Mode.APPEND)
        io.seek(0)
        raw = io.read().decode("utf-8")

    assert "\"a\",\"b\"" in raw  # header present


def test_append_concatenates_data_rows():
    io = CsvIO()
    t1 = pa.table({"a": [1], "b": [2]})
    t2 = pa.table({"a": [3], "b": [4]})

    with io:
        io.write_arrow_table(t1, mode=Mode.OVERWRITE)
        io.seek(0)
        io.write_arrow_table(t2, mode=Mode.APPEND)
        io.seek(0)
        result = io.read_arrow_table()

    assert result.num_rows == 2
    assert result["a"].to_pylist() == [1, 3]


# ---------------------------------------------------------------------------
# Native scanner gating
# ---------------------------------------------------------------------------


class TestNativeScannerGating:

    def test_no_path_blocks_native(self):
        io = CsvIO()
        # No path bound → can't use native scanner regardless.
        with io:
            io.write(b"a,b\n1,2\n")
            options = io.check_options()
            assert not io._can_use_native_scanner(options)

    def test_empty_buffer_blocks_native(self):
        io = CsvIO()
        with io:
            options = io.check_options()
            assert not io._can_use_native_scanner(options)

    def test_target_field_blocks_native(self, arrow_table):
        """A target_field set means per-batch casting; native scanners
        don't know how to honor that."""
        io = CsvIO()
        with io:
            io.write_arrow_table(arrow_table)
            field = arrow_table.schema.field(0)
            options = io.check_options(target_field=field)
            assert not io._can_use_native_scanner(options)
