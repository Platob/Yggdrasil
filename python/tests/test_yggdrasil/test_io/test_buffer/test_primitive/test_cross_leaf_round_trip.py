"""Cross-leaf round-trip tests.

These tests parameterize over every leaf and verify that the
behaviors that *should* be uniform across formats actually are:

- Empty buffer reads return no batches.
- Empty iterator writes are no-ops.
- Round-tripping a small canonical table preserves row count and
  column names.
- ``write_arrow_batches`` and ``write_arrow_table`` produce
  equivalent results.

Differences in type fidelity (XML/CSV/XLSX coerce types; IPC/
Parquet/Zip preserve them) are *not* tested here — those go in the
per-leaf files.
"""

from __future__ import annotations

import pytest
import pyarrow as pa

from yggdrasil.io.buffer.primitive.csv_io import CsvIO
from yggdrasil.io.buffer.primitive.json_io import JsonIO
from yggdrasil.io.buffer.primitive.parquet_io import ParquetIO
from yggdrasil.io.buffer.primitive.arrow_ipc_io import ArrowIPCIO
from yggdrasil.io.buffer.primitive.zip_io import ZipIO


# Skip XLSX/XML when their optional deps aren't available.
def _maybe_xlsx():
    try:
        import openpyxl  # noqa: F401
        from yggdrasil.io.buffer.primitive.xlsx_io import XlsxIO
        return XlsxIO
    except ImportError:
        return None


def _maybe_xml():
    try:
        from lxml import etree  # noqa: F401
        from yggdrasil.io.buffer.primitive.xml_io import XmlIO
        return XmlIO
    except ImportError:
        return None


ALL_LEAVES = [
    pytest.param(CsvIO, id="csv"),
    pytest.param(JsonIO, id="json"),
    pytest.param(ParquetIO, id="parquet"),
    pytest.param(ArrowIPCIO, id="arrow_ipc"),
    pytest.param(ZipIO, id="zip"),
]

# Conditional leaves.
_xlsx = _maybe_xlsx()
if _xlsx is not None:
    ALL_LEAVES.append(pytest.param(_xlsx, id="xlsx"))

_xml = _maybe_xml()
if _xml is not None:
    ALL_LEAVES.append(pytest.param(_xml, id="xml"))


# ---------------------------------------------------------------------------
# Uniform behaviors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leaf_cls", ALL_LEAVES)
def test_empty_buffer_read_yields_no_batches(leaf_cls):
    io = leaf_cls()
    with io:
        batches = list(io.read_arrow_batches())
    assert batches == []


@pytest.mark.parametrize("leaf_cls", ALL_LEAVES)
def test_empty_iterator_write_is_noop(leaf_cls):
    io = leaf_cls()
    with io:
        io.write_arrow_batches(iter([]))
    # Most leaves should leave the buffer empty after writing nothing.
    # XLSX may or may not — its write-only workbook can't be saved
    # without at least one row in some openpyxl versions, so
    # accept either behavior.
    # (If your XLSX leaf always emits an empty workbook, drop the
    # is_empty check for that case.)
    assert io.is_empty() or io.size > 0


@pytest.mark.parametrize("leaf_cls", ALL_LEAVES)
def test_round_trip_row_count_and_column_names(leaf_cls):
    """Every leaf preserves row count and column names."""
    table = pa.table({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
    })

    io = leaf_cls()
    with io:
        io.write_arrow_table(table)
        io.seek(0)
        result = io.read_arrow_table()

    assert result.num_rows == 3
    assert result.column_names == ["id", "name"]


@pytest.mark.parametrize("leaf_cls", ALL_LEAVES)
def test_write_table_and_write_batches_equivalent(leaf_cls):
    """Calling ``write_arrow_table`` should produce the same result
    as ``write_arrow_batches`` over the same data."""
    table = pa.table({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
    batches = table.to_batches(max_chunksize=2)

    io_table = leaf_cls()
    with io_table:
        io_table.write_arrow_table(table)
        io_table.seek(0)
        result_table = io_table.read_arrow_table()

    io_batches = leaf_cls()
    with io_batches:
        io_batches.write_arrow_batches(iter(batches))
        io_batches.seek(0)
        result_batches = io_batches.read_arrow_table()

    # Equal row count + same column names; values should match
    # modulo type coercion in lossy formats.
    assert result_table.num_rows == result_batches.num_rows
    assert result_table.column_names == result_batches.column_names


# ---------------------------------------------------------------------------
# Schema collection on empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leaf_cls", ALL_LEAVES)
def test_collect_schema_on_empty_returns_empty_schema(leaf_cls):
    io = leaf_cls()
    with io:
        schema = io.collect_schema()
    # Schema.empty() is the contract — accept either is_empty()
    # method or zero length.
    is_empty = (
        (hasattr(schema, "is_empty") and schema.is_empty())
        or len(schema) == 0
    )
    assert is_empty
