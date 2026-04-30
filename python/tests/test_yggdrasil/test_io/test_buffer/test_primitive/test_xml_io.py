"""Tests for :class:`XmlIO`.

Coverage:

- Round-trip of flat row-shape documents.
- Custom root_tag / row_tag.
- All values stringify on write (XML has no native types).
- APPEND / UPSERT rejected.
- Bounded-memory iterparse — read a large doc without OOM.

XmlIO depends on lxml; tests are skipped when lxml isn't available.
"""

from __future__ import annotations

import pytest
import pyarrow as pa

lxml = pytest.importorskip("lxml")

from yggdrasil.io.enums import Mode
from yggdrasil.io.buffer.primitive.xml_io import XmlIO, XmlOptions


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_basic():
    """Values round-trip as strings — XML has no type system."""
    io = XmlIO()
    table = pa.table({
        "id": ["1", "2", "3"],
        "name": ["alpha", "beta", "gamma"],
    })

    with io:
        io.write_arrow_table(table)
        io.seek(0)
        result = io.read_arrow_table()

    assert result.column_names == ["id", "name"]
    assert result["id"].to_pylist() == ["1", "2", "3"]
    assert result["name"].to_pylist() == ["alpha", "beta", "gamma"]


def test_numeric_values_serialize_as_strings():
    """Writing ints/floats produces string-shaped XML; reading back
    gets strings."""
    io = XmlIO()
    table = pa.table({"v": [1, 2, 3]})

    with io:
        io.write_arrow_table(table)
        io.seek(0)
        raw = io.read().decode("utf-8")
        assert "<v>1</v>" in raw
        assert "<v>2</v>" in raw

        io.seek(0)
        result = io.read_arrow_table()

    # On read-back, values come through as strings.
    assert result["v"].to_pylist() == ["1", "2", "3"]


def test_empty_buffer_read_yields_no_batches():
    io = XmlIO()
    with io:
        batches = list(io.read_arrow_batches())
    assert batches == []


# ---------------------------------------------------------------------------
# Tag customization
# ---------------------------------------------------------------------------


def test_custom_root_and_row_tags():
    io = XmlIO()
    table = pa.table({"x": ["1"]})
    options = XmlOptions(root_tag="trades", row_tag="trade")

    with io:
        io.write_arrow_table(table, options=options)
        io.seek(0)
        raw = io.read().decode("utf-8")
        assert "<trades>" in raw
        assert "<trade>" in raw
        assert "<rows>" not in raw
        assert "<row>" not in raw

        io.seek(0)
        result = io.read_arrow_table(options=options)

    assert result.num_rows == 1


# ---------------------------------------------------------------------------
# Save modes
# ---------------------------------------------------------------------------


def test_append_rejected():
    io = XmlIO()
    with pytest.raises(ValueError, match="XML append"):
        io._resolve_save_mode(Mode.APPEND)


def test_upsert_rejected():
    io = XmlIO()
    with pytest.raises(ValueError, match="UPSERT"):
        io._resolve_save_mode(Mode.UPSERT)


# ---------------------------------------------------------------------------
# Bounded-memory parse
# ---------------------------------------------------------------------------


def test_large_document_bounded_memory():
    """iterparse + element.clear() should handle a large doc without
    holding the whole tree in memory.

    We don't actually measure memory here (psutil dependency
    overkill); we just verify that 50k rows round-trip correctly.
    A leak in the iterparse loop would manifest as a hang or OOM.
    """
    io = XmlIO()
    n = 50_000
    table = pa.table({
        "id": [str(i) for i in range(n)],
        "v": [str(i * 2) for i in range(n)],
    })

    with io:
        io.write_arrow_table(table)
        io.seek(0)
        result = io.read_arrow_table()

    assert result.num_rows == n
    assert result["id"][0].as_py() == "0"
    assert result["id"][-1].as_py() == str(n - 1)
